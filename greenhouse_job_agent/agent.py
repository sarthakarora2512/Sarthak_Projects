"""
Greenhouse Job Application Agent

An automated job discovery and application agent that:
1. Discovers jobs from Greenhouse boards
2. Tailors resumes using LangChain/LLM
3. Generates answers to application questions
4. Submits applications via Greenhouse API

Features:
- Retry logic with exponential backoff
- Rate limiting for API calls
- Persistent state storage
- Comprehensive logging
- Configuration via environment variables
"""

import hashlib
import io
import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar

import requests
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None
) -> logging.Logger:
    """
    Configure logging with console and optional file output.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        log_format: Optional custom format string

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("greenhouse_agent")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Clear existing handlers
    logger.handlers.clear()

    format_str = log_format or "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    formatter = logging.Formatter(format_str, datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


logger = setup_logging(
    level=os.getenv("LOG_LEVEL", "INFO"),
    log_file=os.getenv("LOG_FILE")
)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Config:
    """Application configuration loaded from environment variables."""

    # Greenhouse settings
    gh_base_url: str = "https://boards-api.greenhouse.io/v1/boards"
    board_token: str = ""

    # Request settings
    default_timeout: int = 20
    max_retries: int = 3
    retry_backoff_factor: float = 2.0
    retry_status_codes: tuple = (429, 500, 502, 503, 504)

    # Rate limiting
    rate_limit_calls: int = 10  # calls per window
    rate_limit_window: int = 60  # seconds

    # OpenAI settings
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_temperature: float = 0.2

    # Storage
    state_db_path: str = "greenhouse_agent_state.db"

    # Polling
    poll_interval_seconds: int = 900  # 15 minutes

    # Job filters
    job_query: str = ""
    job_locations: List[str] = field(default_factory=list)

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        locations_str = os.getenv("JOB_LOCATIONS", "")
        locations = [loc.strip() for loc in locations_str.split(",") if loc.strip()]

        return cls(
            gh_base_url=os.getenv("GH_BASE_URL", cls.gh_base_url),
            board_token=os.getenv("GH_BOARD_TOKEN", ""),
            default_timeout=int(os.getenv("REQUEST_TIMEOUT", "20")),
            max_retries=int(os.getenv("MAX_RETRIES", "3")),
            retry_backoff_factor=float(os.getenv("RETRY_BACKOFF", "2.0")),
            rate_limit_calls=int(os.getenv("RATE_LIMIT_CALLS", "10")),
            rate_limit_window=int(os.getenv("RATE_LIMIT_WINDOW", "60")),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            openai_temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.2")),
            state_db_path=os.getenv("STATE_DB_PATH", "greenhouse_agent_state.db"),
            poll_interval_seconds=int(os.getenv("POLL_INTERVAL", "900")),
            job_query=os.getenv("JOB_QUERY", ""),
            job_locations=locations,
        )


# =============================================================================
# RATE LIMITER
# =============================================================================

class RateLimiter:
    """
    Token bucket rate limiter to prevent API throttling.

    Tracks call timestamps and blocks if limit exceeded within the window.
    """

    def __init__(self, max_calls: int, window_seconds: int):
        """
        Args:
            max_calls: Maximum number of calls allowed in the window
            window_seconds: Time window in seconds
        """
        self.max_calls = max_calls
        self.window_seconds = window_seconds
        self.call_times: List[float] = []

    def _cleanup_old_calls(self) -> None:
        """Remove call timestamps outside the current window."""
        cutoff = time.time() - self.window_seconds
        self.call_times = [t for t in self.call_times if t > cutoff]

    def wait_if_needed(self) -> None:
        """Block until a call is allowed within rate limits."""
        self._cleanup_old_calls()

        while len(self.call_times) >= self.max_calls:
            oldest_call = self.call_times[0]
            sleep_time = oldest_call + self.window_seconds - time.time() + 0.1
            if sleep_time > 0:
                logger.debug(f"Rate limit reached, sleeping {sleep_time:.2f}s")
                time.sleep(sleep_time)
            self._cleanup_old_calls()

        self.call_times.append(time.time())

    def __call__(self, func: Callable) -> Callable:
        """Decorator to apply rate limiting to a function."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            self.wait_if_needed()
            return func(*args, **kwargs)
        return wrapper


# =============================================================================
# RETRY DECORATOR
# =============================================================================

T = TypeVar("T")


def retry_with_backoff(
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    exceptions: tuple = (requests.RequestException,),
    retry_on_status: tuple = (429, 500, 502, 503, 504),
) -> Callable:
    """
    Decorator for retrying functions with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        backoff_factor: Multiplier for exponential backoff (sleep = factor^attempt)
        exceptions: Tuple of exception types to catch and retry
        retry_on_status: HTTP status codes that trigger retry

    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    result = func(*args, **kwargs)

                    # Check for HTTP response with retryable status
                    if hasattr(result, "status_code") and result.status_code in retry_on_status:
                        if attempt < max_retries:
                            sleep_time = backoff_factor ** attempt
                            logger.warning(
                                f"Retryable status {result.status_code}, "
                                f"attempt {attempt + 1}/{max_retries + 1}, "
                                f"sleeping {sleep_time:.1f}s"
                            )
                            time.sleep(sleep_time)
                            continue

                    return result

                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        sleep_time = backoff_factor ** attempt
                        logger.warning(
                            f"Request failed: {e}, "
                            f"attempt {attempt + 1}/{max_retries + 1}, "
                            f"retrying in {sleep_time:.1f}s"
                        )
                        time.sleep(sleep_time)
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed: {e}")
                        raise

            if last_exception:
                raise last_exception

        return wrapper
    return decorator


# =============================================================================
# PERSISTENT STATE STORAGE
# =============================================================================

class StateStorage:
    """
    SQLite-based persistent storage for tracking applied jobs and agent state.

    Tables:
    - applied_jobs: Records of jobs we've applied to
    - agent_state: Key-value store for misc state
    """

    def __init__(self, db_path: str):
        """
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Create database tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS applied_jobs (
                    job_id INTEGER PRIMARY KEY,
                    board_token TEXT NOT NULL,
                    job_title TEXT,
                    company TEXT,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'submitted',
                    response_data TEXT
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS agent_state (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_applied_jobs_board
                ON applied_jobs(board_token)
            """)

            conn.commit()

        logger.debug(f"Initialized state database at {self.db_path}")

    def has_applied(self, job_id: int) -> bool:
        """Check if we've already applied to a job."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM applied_jobs WHERE job_id = ?", (job_id,))
            return cursor.fetchone() is not None

    def record_application(
        self,
        job_id: int,
        board_token: str,
        job_title: str,
        company: str,
        status: str = "submitted",
        response_data: Optional[dict] = None
    ) -> None:
        """Record a job application in the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO applied_jobs
                (job_id, board_token, job_title, company, status, response_data, applied_at)
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (
                    job_id,
                    board_token,
                    job_title,
                    company,
                    status,
                    json.dumps(response_data) if response_data else None
                )
            )
            conn.commit()

        logger.debug(f"Recorded application for job {job_id}")

    def get_applied_job_ids(self, board_token: Optional[str] = None) -> List[int]:
        """Get list of job IDs we've applied to."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            if board_token:
                cursor.execute(
                    "SELECT job_id FROM applied_jobs WHERE board_token = ?",
                    (board_token,)
                )
            else:
                cursor.execute("SELECT job_id FROM applied_jobs")
            return [row[0] for row in cursor.fetchall()]

    def get_state(self, key: str, default: Any = None) -> Any:
        """Get a value from the state store."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM agent_state WHERE key = ?", (key,))
            row = cursor.fetchone()
            if row:
                try:
                    return json.loads(row[0])
                except json.JSONDecodeError:
                    return row[0]
            return default

    def set_state(self, key: str, value: Any) -> None:
        """Set a value in the state store."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO agent_state (key, value, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                """,
                (key, json.dumps(value) if not isinstance(value, str) else value)
            )
            conn.commit()

    def get_application_stats(self) -> Dict[str, Any]:
        """Get statistics about applications."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM applied_jobs")
            total = cursor.fetchone()[0]

            cursor.execute(
                "SELECT COUNT(*) FROM applied_jobs WHERE date(applied_at) = date('now')"
            )
            today = cursor.fetchone()[0]

            cursor.execute(
                """
                SELECT board_token, COUNT(*) as cnt
                FROM applied_jobs
                GROUP BY board_token
                ORDER BY cnt DESC
                LIMIT 5
                """
            )
            by_board = dict(cursor.fetchall())

            return {
                "total_applications": total,
                "applications_today": today,
                "by_board": by_board
            }


# =============================================================================
# HTTP SESSION WITH RETRY
# =============================================================================

def create_session(config: Config) -> requests.Session:
    """
    Create a requests session with automatic retry on failure.

    Args:
        config: Application configuration

    Returns:
        Configured requests.Session
    """
    session = requests.Session()
    session.headers.update({
        "User-Agent": "greenhouse-job-agent/2.0",
        "Accept": "application/json",
    })

    # Configure retry strategy
    retry_strategy = Retry(
        total=config.max_retries,
        backoff_factor=config.retry_backoff_factor,
        status_forcelist=list(config.retry_status_codes),
        allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "TRACE"],
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    return session


# =============================================================================
# GREENHOUSE API CLIENT
# =============================================================================

class GreenhouseClient:
    """
    Client for interacting with the Greenhouse Job Board API.

    Handles job discovery and application submission with rate limiting
    and retry logic.
    """

    def __init__(self, config: Config, session: requests.Session, rate_limiter: RateLimiter):
        """
        Args:
            config: Application configuration
            session: Configured requests session
            rate_limiter: Rate limiter instance
        """
        self.config = config
        self.session = session
        self.rate_limiter = rate_limiter
        self.base_url = config.gh_base_url
        self.timeout = config.default_timeout

    def _get(self, url: str, params: Optional[dict] = None) -> dict:
        """
        Make a rate-limited GET request to the Greenhouse API.

        Args:
            url: Full URL to request
            params: Optional query parameters

        Returns:
            Parsed JSON response

        Raises:
            requests.HTTPError: On non-2xx response
        """
        self.rate_limiter.wait_if_needed()

        logger.debug(f"GET {url}")
        response = self.session.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()

        return response.json()

    def _post(
        self,
        url: str,
        data: Optional[dict] = None,
        files: Optional[dict] = None
    ) -> requests.Response:
        """
        Make a rate-limited POST request to the Greenhouse API.

        Args:
            url: Full URL to request
            data: Form data
            files: Files to upload

        Returns:
            Response object
        """
        self.rate_limiter.wait_if_needed()

        logger.debug(f"POST {url}")
        response = self.session.post(
            url,
            data=data,
            files=files,
            timeout=self.timeout
        )

        return response

    def discover_jobs(
        self,
        board_token: str,
        query: str = "",
        locations: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Discover and filter jobs from a Greenhouse board.

        Args:
            board_token: Greenhouse board identifier (e.g., "stripe")
            query: Search query to match against job titles
            locations: List of location keywords to filter by

        Returns:
            List of job dictionaries with full details including questions
        """
        logger.info(f"Discovering jobs on board '{board_token}' (query='{query}')")

        # 1) Fetch job list (minimal data)
        list_url = f"{self.base_url}/{board_token}/jobs"
        data = self._get(list_url)
        raw_jobs = data.get("jobs", [])

        logger.debug(f"Found {len(raw_jobs)} total jobs on board")

        # 2) Apply local filters
        q_lower = (query or "").lower()
        wanted_locs = {loc.lower() for loc in (locations or [])}

        filtered = []
        for job in raw_jobs:
            title = job.get("title", "")
            location = (job.get("location") or {}).get("name", "")

            # Filter by query in title
            if q_lower and q_lower not in title.lower():
                continue

            # Filter by location
            if wanted_locs and location:
                if all(w not in location.lower() for w in wanted_locs):
                    continue

            filtered.append(job)

        logger.info(f"Filtered to {len(filtered)} jobs matching criteria")

        # 3) Fetch full details for each filtered job
        jobs_out = []
        for job in filtered:
            job_id = job["id"]

            try:
                detail_url = f"{self.base_url}/{board_token}/jobs/{job_id}"
                detail = self._get(detail_url)

                questions = detail.get("questions") or []
                metadata = detail.get("metadata") or {}
                company = metadata.get("company") or job.get("company") or board_token
                content = detail.get("content", "")

                jobs_out.append({
                    "job_id": job_id,
                    "title": job.get("title"),
                    "company": company,
                    "location": (job.get("location") or {}).get("name", ""),
                    "apply_type": "api",
                    "questions": questions,
                    "job_description": content,
                    "absolute_url": job.get("absolute_url", ""),
                })

            except requests.RequestException as e:
                logger.warning(f"Failed to fetch details for job {job_id}: {e}")
                continue

        logger.info(f"Retrieved full details for {len(jobs_out)} jobs")
        return jobs_out

    def apply_to_job(
        self,
        board_token: str,
        job_id: int,
        user_info: Dict[str, str],
        resume_file: io.BytesIO,
        resume_filename: str,
        answers: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Submit a job application via Greenhouse API.

        Args:
            board_token: Greenhouse board identifier
            job_id: Job ID to apply for
            user_info: Dict with first_name, last_name, email, phone (optional)
            resume_file: File-like object containing resume PDF bytes
            resume_filename: Filename for the resume
            answers: List of answer dicts with 'id' and 'value' keys

        Returns:
            Dict with status, response data, and timestamp

        Raises:
            RuntimeError: On application failure
        """
        post_url = f"{self.base_url}/{board_token}/jobs/{job_id}/applications"

        # Build form data
        form_data = {
            "first_name": user_info.get("first_name", ""),
            "last_name": user_info.get("last_name", ""),
            "email": user_info.get("email", ""),
        }

        if user_info.get("phone"):
            form_data["phone"] = user_info["phone"]

        # Add answers
        for answer in answers or []:
            question_id = str(answer.get("id", ""))
            value = answer.get("value", "")
            if question_id:
                form_data[f"answers[{question_id}]"] = value

        # Prepare file upload
        files = {
            "resume": (resume_filename, resume_file, "application/pdf")
        }

        logger.info(f"Submitting application for job {job_id}")

        # Submit application
        response = self._post(post_url, data=form_data, files=files)

        # Handle response
        if response.status_code >= 400:
            try:
                error_data = response.json()
            except ValueError:
                error_data = {"error": response.text}

            logger.error(f"Application failed: {response.status_code} - {error_data}")
            raise RuntimeError(
                f"Greenhouse application failed: {response.status_code} {error_data}"
            )

        try:
            response_data = response.json()
        except ValueError:
            response_data = {"raw": response.text}

        logger.info(f"Application submitted successfully for job {job_id}")

        return {
            "status": "submitted",
            "response": response_data,
            "timestamp": int(time.time()),
            "job_id": job_id,
        }


# =============================================================================
# LANGCHAIN COMPONENTS
# =============================================================================

# ----- Resume Tailoring -----

class ResumeOut(BaseModel):
    """Structured output for tailored resume."""
    summary: str = Field(description="1-2 line role-aligned summary")
    bullets: List[str] = Field(description="3-5 concise, quantified bullets")
    filename: str = Field(description="Proposed filename for the tailored PDF")


RESUME_PROMPT = PromptTemplate.from_template(
    """You are a concise resume editor.

User base resume (Markdown):
---
{base_resume_md}
---

Target job title: {job_title}
Important keywords: {job_keywords}

Rewrite ONLY the top summary and the first 3-5 bullets to fit the target role.
Keep it factual; do not invent experience. Use action + impact + metrics.

Return JSON with:
- summary: 1-2 lines
- bullets: 3-5 bullets (<= 20 words each)
- filename: snake_case like "firstname_lastname_job_title.pdf"

Output valid JSON only, no markdown code blocks.
"""
)


# ----- Answer Generation -----

class QA(BaseModel):
    """Single question-answer pair."""
    id: str = Field(description="Question ID")
    value: str = Field(description="Answer value")


class AnswersOut(BaseModel):
    """Structured output for application answers."""
    answers: List[QA] = Field(description="List of question-answer pairs")


ANSWERS_PROMPT = PromptTemplate.from_template(
    """You draft short, truthful application answers.

User profile:
- name: {name}
- top skills: {skills}
- projects: {projects}

Job description (HTML allowed):
{job_description}

Questions (JSON):
{questions_json}

Rules:
- Short text answers <= 300 characters, specific and credible.
- For yes/no, answer "Yes" or "No" ONLY.
- For single/multi-select, answer with the option text EXACTLY as shown.
- Never fabricate eligibility/legal facts.
- If you don't know or can't answer, use a reasonable placeholder like "N/A" or leave empty.

Return JSON: {{"answers": [{{"id": "<id>", "value": "<answer>"}}]}}

Output valid JSON only, no markdown code blocks.
"""
)


class LLMService:
    """
    Service for LLM-based resume tailoring and answer generation.

    Uses LangChain with structured output parsing.
    """

    def __init__(self, config: Config):
        """
        Args:
            config: Application configuration with OpenAI settings
        """
        self.config = config

        if not config.openai_api_key:
            logger.warning("OpenAI API key not configured - LLM features disabled")
            self.llm = None
            return

        self.llm = ChatOpenAI(
            model=config.openai_model,
            temperature=config.openai_temperature,
            api_key=config.openai_api_key,
        )

        # Build chains
        self.resume_parser = JsonOutputParser(pydantic_object=ResumeOut)
        self.resume_chain = RESUME_PROMPT | self.llm | self.resume_parser

        self.answers_parser = JsonOutputParser(pydantic_object=AnswersOut)
        self.answers_chain = ANSWERS_PROMPT | self.llm | self.answers_parser

        logger.info(f"LLM service initialized with model {config.openai_model}")

    def update_resume(
        self,
        base_resume_md: str,
        job_title: str,
        job_keywords: List[str],
        tone: str = "concise"
    ) -> Dict[str, Any]:
        """
        Tailor a resume for a specific job.

        Args:
            base_resume_md: Base resume in Markdown format
            job_title: Target job title
            job_keywords: Important keywords to incorporate
            tone: Writing tone (currently unused, reserved for future)

        Returns:
            Dict with resume_pdf_url, checksum, summary, bullets
        """
        if not self.llm:
            raise RuntimeError("LLM not configured - cannot update resume")

        logger.info(f"Tailoring resume for: {job_title}")

        result = self.resume_chain.invoke({
            "base_resume_md": base_resume_md,
            "job_title": job_title,
            "job_keywords": ", ".join(job_keywords)
        })

        # Extract values (handle both dict and Pydantic model)
        if isinstance(result, dict):
            summary = result.get("summary", "")
            bullets = result.get("bullets", [])
            filename = result.get("filename", "resume.pdf")
        else:
            summary = result.summary
            bullets = result.bullets
            filename = result.filename

        # Generate PDF content (placeholder - replace with real PDF generation)
        pdf_content = f"{summary}\n\n" + "\n".join(f"- {b}" for b in bullets)
        pdf_bytes = pdf_content.encode("utf-8")
        checksum = "sha256:" + hashlib.sha256(pdf_bytes).hexdigest()

        # In production, upload to storage and get real URL
        resume_pdf_url = f"https://example.com/resumes/{filename}"

        logger.debug(f"Resume tailored: {filename}")

        return {
            "resume_pdf_url": resume_pdf_url,
            "checksum": checksum,
            "summary": summary,
            "bullets": bullets,
            "filename": filename,
            "pdf_bytes": pdf_bytes,  # Include raw bytes for direct use
        }

    def generate_answers(
        self,
        questions: List[Dict[str, Any]],
        user_profile: Dict[str, Any],
        job_description: str
    ) -> Dict[str, Any]:
        """
        Generate answers for application questions.

        Args:
            questions: List of question objects from Greenhouse
            user_profile: Dict with name, skills, projects
            job_description: HTML job description

        Returns:
            Dict with 'answers' list containing id/value pairs
        """
        if not self.llm:
            raise RuntimeError("LLM not configured - cannot generate answers")

        if not questions:
            logger.debug("No questions to answer")
            return {"answers": []}

        logger.info(f"Generating answers for {len(questions)} questions")

        result = self.answers_chain.invoke({
            "name": user_profile.get("name", ""),
            "skills": ", ".join(user_profile.get("skills", [])),
            "projects": ", ".join(user_profile.get("projects", [])),
            "job_description": job_description,
            "questions_json": json.dumps(questions, indent=2)
        })

        # Extract answers (handle both dict and Pydantic model)
        if isinstance(result, dict):
            answers = result.get("answers", [])
            # Ensure each answer is a dict
            answers = [
                a if isinstance(a, dict) else {"id": a.id, "value": a.value}
                for a in answers
            ]
        else:
            answers = [{"id": a.id, "value": a.value} for a in result.answers]

        logger.debug(f"Generated {len(answers)} answers")

        return {"answers": answers}


# =============================================================================
# MAIN AGENT
# =============================================================================

class GreenhouseJobAgent:
    """
    Main agent that orchestrates job discovery and application.

    Coordinates between Greenhouse API, LLM services, and state storage.
    """

    def __init__(
        self,
        config: Config,
        user_profile: Dict[str, Any],
        user_info: Dict[str, str],
        base_resume_md: str,
    ):
        """
        Args:
            config: Application configuration
            user_profile: Profile for answer generation (name, skills, projects)
            user_info: Contact info for applications (first_name, last_name, email, phone)
            base_resume_md: Base resume in Markdown format
        """
        self.config = config
        self.user_profile = user_profile
        self.user_info = user_info
        self.base_resume_md = base_resume_md

        # Initialize components
        self.session = create_session(config)
        self.rate_limiter = RateLimiter(
            max_calls=config.rate_limit_calls,
            window_seconds=config.rate_limit_window
        )
        self.storage = StateStorage(config.state_db_path)
        self.greenhouse = GreenhouseClient(config, self.session, self.rate_limiter)
        self.llm_service = LLMService(config)

        logger.info("Greenhouse Job Agent initialized")

    def _download_resume(self, url: str, filename: str) -> io.BytesIO:
        """Download resume from URL and return as file-like object."""
        self.rate_limiter.wait_if_needed()

        response = self.session.get(url, timeout=self.config.default_timeout)
        response.raise_for_status()

        file_obj = io.BytesIO(response.content)
        file_obj.name = filename

        return file_obj

    def process_job(self, job: Dict[str, Any], board_token: str) -> bool:
        """
        Process a single job: tailor resume, generate answers, and apply.

        Args:
            job: Job dictionary from discover_jobs
            board_token: Greenhouse board identifier

        Returns:
            True if application was successful, False otherwise
        """
        job_id = job["job_id"]
        job_title = job["title"]
        company = job["company"]

        logger.info(f"Processing job: {job_title} at {company} (ID: {job_id})")

        try:
            # 1) Tailor resume
            job_keywords = self.user_profile.get("skills", [])[:5]
            resume_data = self.llm_service.update_resume(
                base_resume_md=self.base_resume_md,
                job_title=job_title,
                job_keywords=job_keywords
            )

            # 2) Generate answers if questions exist
            answers_data = {"answers": []}
            if job.get("questions"):
                try:
                    answers_data = self.llm_service.generate_answers(
                        questions=job["questions"],
                        user_profile=self.user_profile,
                        job_description=job.get("job_description", "")
                    )
                except Exception as e:
                    logger.warning(f"Failed to generate answers for job {job_id}: {e}")
                    # Continue with empty answers - may fail if required questions exist

            # 3) Prepare resume file
            # In production, download from resume_pdf_url
            # For now, use the generated bytes directly
            resume_bytes = resume_data.get("pdf_bytes", b"")
            resume_file = io.BytesIO(resume_bytes)
            resume_filename = resume_data.get("filename", "resume.pdf")

            # 4) Submit application
            result = self.greenhouse.apply_to_job(
                board_token=board_token,
                job_id=job_id,
                user_info=self.user_info,
                resume_file=resume_file,
                resume_filename=resume_filename,
                answers=answers_data["answers"]
            )

            # 5) Record success
            self.storage.record_application(
                job_id=job_id,
                board_token=board_token,
                job_title=job_title,
                company=company,
                status="submitted",
                response_data=result
            )

            logger.info(f"Successfully applied to job {job_id}: {job_title}")
            return True

        except Exception as e:
            logger.error(f"Failed to process job {job_id}: {e}", exc_info=True)

            # Record failure
            self.storage.record_application(
                job_id=job_id,
                board_token=board_token,
                job_title=job_title,
                company=company,
                status="failed",
                response_data={"error": str(e)}
            )

            return False

    def run_once(self, board_token: Optional[str] = None) -> Dict[str, Any]:
        """
        Run a single iteration of job discovery and application.

        Args:
            board_token: Override board token from config

        Returns:
            Stats dict with jobs_found, jobs_applied, jobs_skipped, jobs_failed
        """
        board = board_token or self.config.board_token
        if not board:
            raise ValueError("No board token configured")

        stats = {
            "jobs_found": 0,
            "jobs_applied": 0,
            "jobs_skipped": 0,
            "jobs_failed": 0,
        }

        try:
            # Discover jobs
            jobs = self.greenhouse.discover_jobs(
                board_token=board,
                query=self.config.job_query,
                locations=self.config.job_locations
            )
            stats["jobs_found"] = len(jobs)

        except Exception as e:
            logger.error(f"Failed to discover jobs: {e}")
            return stats

        # Process each job
        for job in jobs:
            job_id = job["job_id"]

            # Skip if already applied
            if self.storage.has_applied(job_id):
                logger.debug(f"Skipping job {job_id} - already applied")
                stats["jobs_skipped"] += 1
                continue

            # Process the job
            success = self.process_job(job, board)

            if success:
                stats["jobs_applied"] += 1
            else:
                stats["jobs_failed"] += 1

        return stats

    def run_loop(self) -> None:
        """
        Run the agent in a continuous loop with polling interval.

        This is the main entry point for background operation.
        """
        logger.info(
            f"Starting agent loop (poll interval: {self.config.poll_interval_seconds}s)"
        )

        while True:
            try:
                stats = self.run_once()

                logger.info(
                    f"Iteration complete: "
                    f"found={stats['jobs_found']}, "
                    f"applied={stats['jobs_applied']}, "
                    f"skipped={stats['jobs_skipped']}, "
                    f"failed={stats['jobs_failed']}"
                )

                # Log overall stats periodically
                overall_stats = self.storage.get_application_stats()
                logger.info(f"Total applications: {overall_stats['total_applications']}")

            except Exception as e:
                logger.error(f"Error in agent loop: {e}", exc_info=True)

            # Sleep until next iteration
            logger.info(f"Sleeping for {self.config.poll_interval_seconds} seconds...")
            time.sleep(self.config.poll_interval_seconds)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_resume_from_file(file_path: str) -> Optional[str]:
    """
    Load resume content from a file.

    Args:
        file_path: Path to resume file (supports .md, .txt, .html)

    Returns:
        Resume content as string, or None if file doesn't exist
    """
    path = Path(file_path)
    if not path.exists():
        return None

    try:
        content = path.read_text(encoding="utf-8")
        logger.info(f"Loaded resume from {file_path} ({len(content)} chars)")
        return content
    except Exception as e:
        logger.warning(f"Failed to load resume from {file_path}: {e}")
        return None


def find_resume_file() -> Optional[str]:
    """
    Search for a resume file in common locations.

    Checks in order:
    1. RESUME_FILE environment variable
    2. ./resume.md
    3. ./resume.txt
    4. ./Resume.md
    5. ~/resume.md

    Returns:
        Resume content if found, None otherwise
    """
    # Check environment variable first
    env_path = os.getenv("RESUME_FILE")
    if env_path:
        content = load_resume_from_file(env_path)
        if content:
            return content

    # Check common file locations
    search_paths = [
        "resume.md",
        "resume.txt",
        "Resume.md",
        "RESUME.md",
        "resume.html",
        os.path.expanduser("~/resume.md"),
        os.path.expanduser("~/Documents/resume.md"),
    ]

    for path in search_paths:
        content = load_resume_from_file(path)
        if content:
            return content

    return None


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point for the agent."""

    # Load configuration
    config = Config.from_env()

    # Validate required config
    if not config.board_token:
        logger.error("GH_BOARD_TOKEN environment variable is required")
        print("Error: Please set GH_BOARD_TOKEN environment variable")
        print("Example: export GH_BOARD_TOKEN=stripe")
        return 1

    if not config.openai_api_key:
        logger.error("OPENAI_API_KEY environment variable is required")
        print("Error: Please set OPENAI_API_KEY environment variable")
        return 1

    # Define user profile and info
    user_profile = {
        "name": os.getenv("USER_NAME", "Alex Dev"),
        "skills": os.getenv("USER_SKILLS", "Python,LangChain,APIs,Kubernetes").split(","),
        "projects": os.getenv("USER_PROJECTS", "Embeddings search,Workflow automation").split(","),
    }

    user_info = {
        "first_name": os.getenv("USER_FIRST_NAME", "Alex"),
        "last_name": os.getenv("USER_LAST_NAME", "Dev"),
        "email": os.getenv("USER_EMAIL", "alex@example.com"),
        "phone": os.getenv("USER_PHONE", ""),
    }

    # Load resume: file > environment variable > default
    base_resume_md = find_resume_file()

    if not base_resume_md:
        base_resume_md = os.getenv("BASE_RESUME_MD")

    if not base_resume_md:
        logger.warning("No resume file found, using default sample resume")
        base_resume_md = """# Alex Dev
### Summary
Builder of reliable backend + AI workflows.

### Experience
- Automated LLM pipeline for FAQs; reduced handle time 35%.
- Built Kubernetes-backed API gateway handling 15k rps.
- Designed ETL with dbt + Snowflake; cut costs 22%.

### Skills
Python, Go, Postgres, Docker, K8s, LangChain, REST, CI/CD
"""
        print("\nWARNING: Using default sample resume.")
        print("To use your own resume, create one of these files:")
        print("  - ./resume.md")
        print("  - Set RESUME_FILE=/path/to/your/resume.md")
        print()

    # Create and run agent
    agent = GreenhouseJobAgent(
        config=config,
        user_profile=user_profile,
        user_info=user_info,
        base_resume_md=base_resume_md
    )

    # Check if running single iteration or continuous loop
    if os.getenv("RUN_ONCE", "").lower() in ("1", "true", "yes"):
        stats = agent.run_once()
        print(f"Completed: {json.dumps(stats, indent=2)}")
        return 0
    else:
        agent.run_loop()
        return 0


if __name__ == "__main__":
    exit(main())

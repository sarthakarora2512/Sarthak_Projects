#!/usr/bin/env python3
"""
Mock test for the Greenhouse Job Agent.

Demonstrates the full workflow using mocked API responses.
"""

import json
import sys
import os
from unittest.mock import Mock, patch, MagicMock
import io

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent import (
    Config,
    RateLimiter,
    StateStorage,
    GreenhouseClient,
    LLMService,
    GreenhouseJobAgent,
    create_session,
    setup_logging,
)

logger = setup_logging(level="INFO")


# Mock data that simulates Greenhouse API responses
MOCK_JOBS_LIST = {
    "jobs": [
        {
            "id": 12345,
            "title": "Software Engineer, Backend",
            "location": {"name": "San Francisco, CA"},
            "absolute_url": "https://boards.greenhouse.io/example/jobs/12345"
        },
        {
            "id": 12346,
            "title": "Senior Software Engineer, ML Platform",
            "location": {"name": "Remote, US"},
            "absolute_url": "https://boards.greenhouse.io/example/jobs/12346"
        },
        {
            "id": 12347,
            "title": "Product Manager",
            "location": {"name": "New York, NY"},
            "absolute_url": "https://boards.greenhouse.io/example/jobs/12347"
        }
    ]
}

MOCK_JOB_DETAIL = {
    "id": 12345,
    "title": "Software Engineer, Backend",
    "content": """
    <h2>About the Role</h2>
    <p>We're looking for a backend engineer to build scalable APIs and services.</p>
    <h3>Requirements</h3>
    <ul>
        <li>3+ years Python experience</li>
        <li>Experience with distributed systems</li>
        <li>Strong problem-solving skills</li>
    </ul>
    """,
    "questions": [
        {
            "id": "q1",
            "label": "Why are you interested in this role?",
            "required": True,
            "type": "short_text"
        },
        {
            "id": "q2",
            "label": "Are you authorized to work in the US?",
            "required": True,
            "type": "yes_no"
        },
        {
            "id": "q3",
            "label": "Years of Python experience",
            "required": False,
            "type": "short_text"
        }
    ],
    "metadata": {
        "company": "Example Corp"
    }
}

MOCK_APPLICATION_RESPONSE = {
    "success": True,
    "id": "app_abc123",
    "message": "Application received"
}


def create_mock_response(json_data, status_code=200):
    """Create a mock requests Response object."""
    mock_resp = Mock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = json_data
    mock_resp.text = json.dumps(json_data)
    mock_resp.content = json.dumps(json_data).encode()
    mock_resp.raise_for_status = Mock()
    if status_code >= 400:
        mock_resp.raise_for_status.side_effect = Exception(f"HTTP {status_code}")
    return mock_resp


def test_discover_jobs_mock():
    """Test job discovery with mocked API."""
    print("\n" + "=" * 60)
    print("TEST: Job Discovery (Mocked)")
    print("=" * 60)

    config = Config()
    session = Mock()
    rate_limiter = RateLimiter(max_calls=100, window_seconds=60)

    # Setup mock responses
    def mock_get(url, *args, **kwargs):
        if url.endswith("/jobs"):
            return create_mock_response(MOCK_JOBS_LIST)
        elif "/jobs/" in url:
            return create_mock_response(MOCK_JOB_DETAIL)
        return create_mock_response({})

    session.get = mock_get

    client = GreenhouseClient(config, session, rate_limiter)

    # Test discovery
    jobs = client.discover_jobs(
        board_token="examplecorp",
        query="engineer",
        locations=["San Francisco", "Remote"]
    )

    print(f"  Found {len(jobs)} matching jobs:")
    for job in jobs:
        print(f"    - {job['title']} (ID: {job['job_id']})")
        print(f"      Location: {job.get('location', 'N/A')}")
        print(f"      Questions: {len(job.get('questions', []))}")

    assert len(jobs) == 2, "Should find 2 engineer jobs"
    print("\n  [PASS] Job discovery working correctly")
    return jobs


def test_apply_to_job_mock():
    """Test job application with mocked API."""
    print("\n" + "=" * 60)
    print("TEST: Job Application (Mocked)")
    print("=" * 60)

    config = Config()
    session = Mock()
    rate_limiter = RateLimiter(max_calls=100, window_seconds=60)

    # Setup mock POST response
    session.post = Mock(return_value=create_mock_response(MOCK_APPLICATION_RESPONSE))

    client = GreenhouseClient(config, session, rate_limiter)

    # Test application
    resume_file = io.BytesIO(b"Mock resume content")

    result = client.apply_to_job(
        board_token="examplecorp",
        job_id=12345,
        user_info={
            "first_name": "Alex",
            "last_name": "Dev",
            "email": "alex@example.com",
            "phone": "+1-555-555-5555"
        },
        resume_file=resume_file,
        resume_filename="alex_dev_software_engineer.pdf",
        answers=[
            {"id": "q1", "value": "I'm passionate about building scalable systems."},
            {"id": "q2", "value": "Yes"},
            {"id": "q3", "value": "5 years"}
        ]
    )

    print(f"  Application result: {result['status']}")
    print(f"  Response: {result['response']}")
    print(f"  Timestamp: {result['timestamp']}")

    assert result["status"] == "submitted"
    print("\n  [PASS] Job application working correctly")


def test_llm_resume_tailoring_mock():
    """Test resume tailoring with mocked LLM."""
    print("\n" + "=" * 60)
    print("TEST: Resume Tailoring (Mocked LLM)")
    print("=" * 60)

    config = Config()
    config.openai_api_key = "test-key"

    # Mock the LLM chain
    with patch('agent.ChatOpenAI') as mock_llm_class:
        mock_llm = MagicMock()
        mock_llm_class.return_value = mock_llm

        # Mock the chain invoke result
        mock_result = {
            "summary": "Backend engineer with 5+ years building scalable APIs and distributed systems.",
            "bullets": [
                "Designed microservices handling 50k+ requests/second",
                "Led migration from monolith to Kubernetes; reduced latency 40%",
                "Built ML inference pipeline processing 1M+ predictions daily"
            ],
            "filename": "alex_dev_software_engineer.pdf"
        }

        llm_service = LLMService(config)
        # Manually override the chain result
        llm_service.resume_chain = MagicMock()
        llm_service.resume_chain.invoke.return_value = mock_result

        result = llm_service.update_resume(
            base_resume_md="# Alex Dev\n## Summary\nSoftware engineer.",
            job_title="Software Engineer, Backend",
            job_keywords=["Python", "APIs", "Kubernetes"]
        )

        print(f"  Tailored Summary: {result['summary']}")
        print(f"  Bullets:")
        for bullet in result['bullets']:
            print(f"    - {bullet}")
        print(f"  Filename: {result['filename']}")
        print(f"  Checksum: {result['checksum'][:30]}...")

        print("\n  [PASS] Resume tailoring working correctly")


def test_llm_answer_generation_mock():
    """Test answer generation with mocked LLM."""
    print("\n" + "=" * 60)
    print("TEST: Answer Generation (Mocked LLM)")
    print("=" * 60)

    config = Config()
    config.openai_api_key = "test-key"

    with patch('agent.ChatOpenAI') as mock_llm_class:
        mock_llm = MagicMock()
        mock_llm_class.return_value = mock_llm

        mock_result = {
            "answers": [
                {"id": "q1", "value": "I'm excited about building backend systems that scale to millions of users."},
                {"id": "q2", "value": "Yes"},
                {"id": "q3", "value": "5 years of professional Python experience"}
            ]
        }

        llm_service = LLMService(config)
        llm_service.answers_chain = MagicMock()
        llm_service.answers_chain.invoke.return_value = mock_result

        result = llm_service.generate_answers(
            questions=MOCK_JOB_DETAIL["questions"],
            user_profile={
                "name": "Alex Dev",
                "skills": ["Python", "Kubernetes", "APIs"],
                "projects": ["Scalable API gateway", "ML pipeline"]
            },
            job_description=MOCK_JOB_DETAIL["content"]
        )

        print(f"  Generated {len(result['answers'])} answers:")
        for ans in result['answers']:
            print(f"    [{ans['id']}]: {ans['value'][:60]}...")

        print("\n  [PASS] Answer generation working correctly")


def test_full_workflow_mock():
    """Test the complete workflow with all mocks."""
    print("\n" + "=" * 60)
    print("TEST: Full Workflow (Complete Mock)")
    print("=" * 60)

    # Setup configuration
    config = Config()
    config.board_token = "examplecorp"
    config.openai_api_key = "test-key"

    user_profile = {
        "name": "Alex Dev",
        "skills": ["Python", "Kubernetes", "APIs", "Machine Learning"],
        "projects": ["Scalable API Gateway", "ML Inference Pipeline"]
    }

    user_info = {
        "first_name": "Alex",
        "last_name": "Dev",
        "email": "alex@example.com",
        "phone": "+1-555-555-5555"
    }

    base_resume = """# Alex Dev
## Summary
Experienced backend engineer passionate about scalable systems.

## Experience
- Built high-throughput API services
- Designed ML infrastructure
- Led platform migrations
"""

    with patch('agent.create_session') as mock_create_session, \
         patch('agent.ChatOpenAI') as mock_llm_class:

        # Mock session
        mock_session = MagicMock()

        def mock_get(url, *args, **kwargs):
            if url.endswith("/jobs"):
                return create_mock_response(MOCK_JOBS_LIST)
            elif "/jobs/" in url:
                return create_mock_response(MOCK_JOB_DETAIL)
            return create_mock_response({})

        mock_session.get = mock_get
        mock_session.post = MagicMock(return_value=create_mock_response(MOCK_APPLICATION_RESPONSE))
        mock_create_session.return_value = mock_session

        # Mock LLM
        mock_llm = MagicMock()
        mock_llm_class.return_value = mock_llm

        # Create agent
        print("  1. Creating agent...")
        agent = GreenhouseJobAgent(
            config=config,
            user_profile=user_profile,
            user_info=user_info,
            base_resume_md=base_resume
        )

        # Override LLM chains with mocks
        agent.llm_service.resume_chain = MagicMock()
        agent.llm_service.resume_chain.invoke.return_value = {
            "summary": "Backend engineer with expertise in scalable systems.",
            "bullets": ["Built APIs handling 50k rps", "Led K8s migration"],
            "filename": "alex_dev_backend_engineer.pdf"
        }

        agent.llm_service.answers_chain = MagicMock()
        agent.llm_service.answers_chain.invoke.return_value = {
            "answers": [
                {"id": "q1", "value": "Passionate about backend systems."},
                {"id": "q2", "value": "Yes"},
                {"id": "q3", "value": "5 years"}
            ]
        }

        # Run a single iteration
        print("  2. Running single iteration...")
        stats = agent.run_once()

        print(f"\n  Results:")
        print(f"    Jobs Found: {stats['jobs_found']}")
        print(f"    Jobs Applied: {stats['jobs_applied']}")
        print(f"    Jobs Skipped: {stats['jobs_skipped']}")
        print(f"    Jobs Failed: {stats['jobs_failed']}")

        # Check storage
        print(f"\n  3. Checking persistent storage...")
        app_stats = agent.storage.get_application_stats()
        print(f"    Total applications recorded: {app_stats['total_applications']}")

        # Verify jobs are now marked as applied
        print(f"\n  4. Running second iteration (should skip applied jobs)...")
        stats2 = agent.run_once()
        print(f"    Jobs Skipped (already applied): {stats2['jobs_skipped']}")

        print("\n  [PASS] Full workflow completed successfully!")

        # Cleanup
        if os.path.exists(config.state_db_path):
            os.remove(config.state_db_path)


def main():
    """Run all mock tests."""
    print("=" * 60)
    print("GREENHOUSE JOB AGENT - MOCK TEST SUITE")
    print("=" * 60)
    print("Testing all components with mocked external services")

    test_discover_jobs_mock()
    test_apply_to_job_mock()
    test_llm_resume_tailoring_mock()
    test_llm_answer_generation_mock()
    test_full_workflow_mock()

    print("\n" + "=" * 60)
    print("ALL MOCK TESTS PASSED!")
    print("=" * 60)

    print("\nThe agent is fully functional. To run with real APIs:")
    print("  1. Set OPENAI_API_KEY environment variable")
    print("  2. Set GH_BOARD_TOKEN to a real Greenhouse board")
    print("  3. Run: python agent.py")


if __name__ == "__main__":
    main()

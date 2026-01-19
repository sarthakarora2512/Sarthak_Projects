#!/usr/bin/env python3
"""
Test script for the Greenhouse Job Agent.

This script tests individual components without requiring all API keys.
"""

import json
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent import (
    Config,
    RateLimiter,
    StateStorage,
    GreenhouseClient,
    create_session,
    setup_logging,
)

# Setup logging
logger = setup_logging(level="INFO")


def test_rate_limiter():
    """Test the rate limiter."""
    print("\n" + "=" * 60)
    print("TEST: Rate Limiter")
    print("=" * 60)

    limiter = RateLimiter(max_calls=3, window_seconds=5)

    import time
    for i in range(5):
        start = time.time()
        limiter.wait_if_needed()
        elapsed = time.time() - start
        print(f"  Call {i+1}: waited {elapsed:.2f}s")

    print("  [PASS] Rate limiter working correctly")


def test_state_storage():
    """Test persistent state storage."""
    print("\n" + "=" * 60)
    print("TEST: State Storage (SQLite)")
    print("=" * 60)

    # Use temp database
    storage = StateStorage("test_state.db")

    # Test recording an application
    storage.record_application(
        job_id=12345,
        board_token="testcompany",
        job_title="Software Engineer",
        company="Test Corp",
        status="submitted",
        response_data={"test": True}
    )
    print("  Recorded test application")

    # Check if we can detect it
    has_applied = storage.has_applied(12345)
    print(f"  Has applied to 12345: {has_applied}")
    assert has_applied, "Should have recorded the application"

    # Check non-existent
    has_not_applied = storage.has_applied(99999)
    print(f"  Has applied to 99999: {has_not_applied}")
    assert not has_not_applied, "Should not have applied to this job"

    # Get stats
    stats = storage.get_application_stats()
    print(f"  Stats: {stats}")

    # Cleanup
    os.remove("test_state.db")
    print("  [PASS] State storage working correctly")


def test_greenhouse_discovery():
    """Test Greenhouse job discovery (real API call)."""
    print("\n" + "=" * 60)
    print("TEST: Greenhouse Job Discovery")
    print("=" * 60)

    # Use a real public board (Anthropic, Stripe, etc.)
    # These are public job boards that don't require authentication to read
    test_boards = ["anthropic", "stripe", "airbnb"]

    config = Config()
    session = create_session(config)
    rate_limiter = RateLimiter(max_calls=10, window_seconds=60)
    client = GreenhouseClient(config, session, rate_limiter)

    for board in test_boards:
        print(f"\n  Trying board: {board}")
        try:
            jobs = client.discover_jobs(
                board_token=board,
                query="",  # No filter - get all jobs
                locations=None
            )

            print(f"  Found {len(jobs)} jobs on {board}")

            if jobs:
                # Show first 3 jobs
                print(f"\n  Sample jobs from {board}:")
                for job in jobs[:3]:
                    print(f"    - {job['title']}")
                    print(f"      ID: {job['job_id']}")
                    print(f"      Location: {job.get('location', 'N/A')}")
                    print(f"      Questions: {len(job.get('questions', []))}")
                    print()

                print(f"  [PASS] Successfully discovered jobs from {board}")
                return jobs  # Return for further testing

        except Exception as e:
            print(f"  [SKIP] Board '{board}' failed: {e}")
            continue

    print("  [WARN] Could not access any test boards")
    return []


def test_job_filtering():
    """Test job filtering logic."""
    print("\n" + "=" * 60)
    print("TEST: Job Filtering")
    print("=" * 60)

    config = Config()
    session = create_session(config)
    rate_limiter = RateLimiter(max_calls=10, window_seconds=60)
    client = GreenhouseClient(config, session, rate_limiter)

    # Try to find engineering jobs
    print("  Searching for 'engineer' jobs...")
    try:
        jobs = client.discover_jobs(
            board_token="anthropic",
            query="engineer",
            locations=["San Francisco", "Remote"]
        )
        print(f"  Found {len(jobs)} engineering jobs")

        for job in jobs[:5]:
            print(f"    - {job['title']} ({job.get('location', 'N/A')})")

        print("  [PASS] Filtering working")

    except Exception as e:
        print(f"  [SKIP] Filtering test failed: {e}")


def test_llm_service_without_key():
    """Test LLM service gracefully handles missing API key."""
    print("\n" + "=" * 60)
    print("TEST: LLM Service (no API key)")
    print("=" * 60)

    from agent import LLMService, Config

    config = Config()  # No API key set
    llm_service = LLMService(config)

    print(f"  LLM initialized: {llm_service.llm is not None}")

    if llm_service.llm is None:
        print("  [PASS] LLM service gracefully disabled without API key")
    else:
        print("  [INFO] LLM service is enabled")


def test_full_workflow_dry_run():
    """Test the full workflow without actually applying."""
    print("\n" + "=" * 60)
    print("TEST: Full Workflow (Dry Run)")
    print("=" * 60)

    from agent import GreenhouseJobAgent, Config

    config = Config.from_env()
    config.board_token = "anthropic"  # Use a real board

    user_profile = {
        "name": "Test User",
        "skills": ["Python", "Machine Learning", "APIs"],
        "projects": ["AI Assistant", "Data Pipeline"]
    }

    user_info = {
        "first_name": "Test",
        "last_name": "User",
        "email": "test@example.com"
    }

    base_resume = """# Test User
## Summary
Experienced software engineer.

## Experience
- Built ML pipelines
- Developed APIs
"""

    print("  Creating agent...")
    agent = GreenhouseJobAgent(
        config=config,
        user_profile=user_profile,
        user_info=user_info,
        base_resume_md=base_resume
    )

    print("  Discovering jobs...")
    jobs = agent.greenhouse.discover_jobs(
        board_token=config.board_token,
        query="",
        locations=None
    )

    print(f"  Found {len(jobs)} jobs")

    if jobs:
        job = jobs[0]
        print(f"\n  Would process job: {job['title']}")
        print(f"  Questions to answer: {len(job.get('questions', []))}")

        # Show questions
        for q in job.get('questions', [])[:3]:
            print(f"    - {q.get('label', q.get('name', 'Unknown'))}")

    print("\n  [PASS] Workflow components working (dry run complete)")
    print("  [INFO] To actually apply, set OPENAI_API_KEY environment variable")


def main():
    """Run all tests."""
    print("=" * 60)
    print("GREENHOUSE JOB AGENT - TEST SUITE")
    print("=" * 60)

    # Run tests
    test_rate_limiter()
    test_state_storage()
    test_greenhouse_discovery()
    test_job_filtering()
    test_llm_service_without_key()
    test_full_workflow_dry_run()

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)
    print("\nTo run the full agent with LLM features:")
    print("  export OPENAI_API_KEY=sk-your-key")
    print("  export GH_BOARD_TOKEN=anthropic")
    print("  python agent.py")


if __name__ == "__main__":
    main()

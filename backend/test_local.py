#!/usr/bin/env python3
"""
Local testing script for LLMGuard backend functionality.
Tests individual components without requiring Modal deployment.
"""

import os
import sys
from pathlib import Path

# Add the backend directory to Python path
sys.path.insert(0, str(Path(__file__).parent))


def test_environment_setup():
    """Test if environment is properly configured."""
    print("🧪 Testing Environment Setup")
    print("-" * 40)

    # Check for .env file
    env_file = Path(".env")
    if env_file.exists():
        print("✅ .env file found")

        # Load and check environment variables
        from dotenv import load_dotenv

        load_dotenv()

        github_token = os.getenv("GITHUB_TOKEN")
        hf_token = os.getenv("HF_TOKEN")

        if github_token and github_token != "your_github_token_here":
            print("✅ GITHUB_TOKEN configured")
        else:
            print("⚠️  GITHUB_TOKEN not configured or using template value")

        if hf_token and hf_token != "your_hf_token_here":
            print("✅ HF_TOKEN configured")
        else:
            print("⚠️  HF_TOKEN not configured or using template value")
    else:
        print("❌ .env file not found")
        return False

    # Check required packages
    try:
        import modal  # noqa: F401

        print("✅ Modal package available")
    except ImportError:
        print("❌ Modal package not installed")
        return False

    try:
        from github import Github  # noqa: F401

        print("✅ PyGithub package available")
    except ImportError:
        print("❌ PyGithub package not installed")
        return False

    return True


def test_github_api():
    """Test GitHub API connectivity."""
    print("\n🧪 Testing GitHub API")
    print("-" * 40)

    try:
        import re

        from dotenv import load_dotenv
        from github import Github

        load_dotenv()

        # Test with a small public repository
        test_repo_url = "https://github.com/octocat/Hello-World"

        # Extract owner/repo from URL
        pattern = r"https://github\.com/([^/]+)/([^/]+)"
        match = re.match(pattern, test_repo_url.rstrip("/"))
        if not match:
            print("❌ Invalid test repository URL")
            return False

        owner, repo_name = match.groups()

        # Initialize GitHub client
        github_token = os.getenv("GITHUB_TOKEN")
        g = Github(github_token) if github_token else Github()

        print(f"Testing with repository: {owner}/{repo_name}")

        # Try to access the repository
        repo = g.get_repo(f"{owner}/{repo_name}")
        print(f"✅ Successfully accessed repository: {repo.full_name}")
        print(f"   Description: {repo.description}")

        # Test getting contents
        contents = repo.get_contents("")
        print(f"✅ Retrieved repository contents ({len(contents)} items)")

        # Test rate limiting info
        rate_limit = g.get_rate_limit()
        remaining = rate_limit.core.remaining
        limit = rate_limit.core.limit
        print(f"✅ API Rate limit: {remaining}/{limit}")

        return True

    except Exception as e:
        print(f"❌ GitHub API test failed: {e}")
        return False


def test_file_filtering():
    """Test file filtering logic."""
    print("\n🧪 Testing File Filtering Logic")
    print("-" * 40)

    # Import the functions from modal_backend
    try:
        from modal_backend import detect_language, should_analyze_file

        # Test cases
        test_cases = [
            ("app.py", "py", True, "python"),
            ("script.js", "js", True, "javascript"),
            ("Component.tsx", "tsx", True, "typescript"),
            ("Main.java", "java", True, "java"),
            ("main.cpp", "cpp", True, "cpp"),
            ("config.json", "json", False, "unknown"),
            ("package.json", "json", False, "unknown"),
            ("README.md", "md", False, "unknown"),
            ("test.php", "php", True, "php"),
            ("app.rb", "rb", True, "ruby"),
        ]

        all_passed = True
        for filename, ext, should_analyze, expected_lang in test_cases:
            actual_should_analyze = should_analyze_file(filename, ext)
            actual_lang = detect_language(ext)

            analyze_ok = actual_should_analyze == should_analyze
            lang_ok = actual_lang == expected_lang
            analyze_result = "✅" if analyze_ok else "❌"
            lang_result = "✅" if lang_ok else "❌"

            status_msg = (
                f"   {filename}: {analyze_result} analyze="
                f"{actual_should_analyze}, {lang_result} "
                f"lang={actual_lang}"
            )
            print(status_msg)

            if not analyze_ok or not lang_ok:
                all_passed = False

        if all_passed:
            print("✅ All file filtering tests passed")
        else:
            print("❌ Some file filtering tests failed")

        return all_passed

    except Exception as e:
        print(f"❌ File filtering test failed: {e}")
        return False


def test_data_structures():
    """Test data structure definitions."""
    print("\n🧪 Testing Data Structures")
    print("-" * 40)

    try:
        from modal_backend import FileAnalysis, RepositoryAnalysis, SecurityIssue

        # Test SecurityIssue
        issue = SecurityIssue(
            file_path="test.py",
            line_number=42,
            severity="HIGH",
            vulnerability_type="SQL Injection",
            description="Test description",
            recommendation="Test recommendation",
            code_snippet="test code",
        )
        print("✅ SecurityIssue created successfully")

        # Test FileAnalysis
        analysis = FileAnalysis(
            file_path="test.py",
            language="python",
            issues=[issue],
            analysis_status="SUCCESS",
        )
        print("✅ FileAnalysis created successfully")

        # Test RepositoryAnalysis
        RepositoryAnalysis(
            repository_url="https://github.com/test/repo",
            total_files_scanned=1,
            files_with_issues=1,
            total_issues=1,
            file_analyses=[analysis],
            repository_structure={"test.py": {"type": "file"}},
        )
        print("✅ RepositoryAnalysis created successfully")

        return True

    except Exception as e:
        print(f"❌ Data structure test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🔧 LLMGuard Backend Local Testing")
    print("=" * 50)

    tests = [
        test_environment_setup,
        test_github_api,
        test_file_filtering,
        test_data_structures,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)

    print("\n" + "=" * 50)
    print("📊 Test Summary")
    print("-" * 20)

    passed = sum(results)
    total = len(results)

    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{i+1}. {test.__name__}: {status}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Ready for Modal deployment.")
        return True
    else:
        print("⚠️  Some tests failed. Please fix issues before deploying.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

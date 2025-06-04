#!/usr/bin/env python3
"""
LLMGuard Modal Testing Script

This script helps you test and deploy the LLMGuard security analyzer on Modal.

Usage:
    python test_modal.py --setup       # Setup model cache
    python test_modal.py --check       # Check cache status
    python test_modal.py --test-single # Test single file analysis
    python test_modal.py --test-repo   # Test repository analysis
    python test_modal.py --full-test   # Run complete test workflow
    python test_modal.py --deploy      # Deploy FastAPI app
    python test_modal.py --serve       # Serve FastAPI app for development
"""

import argparse
import subprocess
import sys


def run_modal_command(command: str, description: str):
    """Run a modal command and handle errors."""
    print(f"\n🚀 {description}")
    print(f"📋 Command: {command}")
    print("-" * 50)

    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=False)
        print(f"✅ {description} completed successfully!")
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code: {e.returncode}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test Modal backend functions")
    parser.add_argument("--setup", action="store_true", help="Setup model cache")
    parser.add_argument("--check", action="store_true", help="Check cache status")
    parser.add_argument(
        "--test-unsloth", action="store_true", help="Test Unsloth directly"
    )
    parser.add_argument(
        "--test-single", action="store_true", help="Test single file analysis"
    )
    parser.add_argument(
        "--test-repo", action="store_true", help="Test repository analysis"
    )
    parser.add_argument("--full-test", action="store_true", help="Run all tests")
    parser.add_argument("--deploy", action="store_true", help="Deploy FastAPI app")
    parser.add_argument(
        "--serve", action="store_true", help="Serve FastAPI app locally"
    )

    args = parser.parse_args()

    if args.setup or args.full_test:
        print("🚀 Setting up model cache...")
        import subprocess

        subprocess.run(["modal", "run", "modal_backend.py::setup_model_cache"])

    if args.check or args.full_test:
        print("🔍 Checking cache status...")
        import subprocess

        subprocess.run(["modal", "run", "modal_backend.py::check_cache"])

    if args.test_unsloth:
        print("🧪 Testing Unsloth directly...")
        import subprocess

        subprocess.run(["modal", "run", "modal_backend.py::test_unsloth"])

    if args.test_single or args.full_test:
        print("🧪 Testing single file analysis...")
        import subprocess

        subprocess.run(["modal", "run", "modal_backend.py::test_single_file"])

    if args.test_repo or args.full_test:
        print("🧪 Testing repository analysis...")
        import subprocess

        subprocess.run(["modal", "run", "modal_backend.py::test_repo"])

    if args.deploy:
        print("🚀 Deploying FastAPI app...")
        import subprocess

        subprocess.run(["modal", "deploy", "modal_backend.py::fastapi_app"])

    if args.serve:
        print("🌐 Serving FastAPI app locally...")
        import subprocess

        subprocess.run(["modal", "serve", "modal_backend.py::fastapi_app"])

    if not any(vars(args).values()):
        parser.print_help()
        return

    print("🔧 LLMGuard Modal Testing & Deployment")
    print("=" * 50)

    success = True

    if args.setup:
        success &= run_modal_command(
            "modal run modal_backend.py::setup_model_cache", "Setting up model cache"
        )

    if args.check:
        success &= run_modal_command(
            "modal run modal_backend.py::check_cache", "Checking cache status"
        )

    if args.test_single:
        success &= run_modal_command(
            "modal run modal_backend.py::test_single_file",
            "Testing single file analysis",
        )

    if args.test_repo:
        success &= run_modal_command(
            "modal run modal_backend.py::test_model", "Testing repository analysis"
        )

    if args.full_test:
        success &= run_modal_command(
            "modal run modal_backend.py::full_setup_and_test",
            "Running complete test workflow",
        )

    if args.deploy:
        success &= run_modal_command(
            "modal deploy modal_backend.py::fastapi_app",
            "Deploying FastAPI app to production",
        )

    if args.serve:
        print("\n🌐 Starting development server...")
        print("📋 Command: modal serve modal_backend.py::fastapi_app")
        print("💡 This will start a live development server")
        print("💡 Press Ctrl+C to stop the server")
        print("-" * 50)

        subprocess.run("modal serve modal_backend.py::fastapi_app", shell=True)

    if success:
        print(f"\n🎉 All operations completed successfully!")
        print("\n📝 Next steps:")
        if args.setup:
            print("  • Run 'python test_modal.py --check' to verify cache")
        if args.check:
            print("  • Run 'python test_modal.py --test-single' to test analysis")
        if args.test_single:
            print("  • Run 'python test_modal.py --test-repo' to test repositories")
        if args.deploy:
            print("  • Your FastAPI app is now live on Modal!")
            print("  • Check Modal dashboard for the URL")
    else:
        print(f"\n❌ Some operations failed. Check the logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()

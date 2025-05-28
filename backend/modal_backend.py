import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import modal
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Modal app configuration
app = modal.App("llmguard-security-analyzer")

# GPU configuration - Fixed: Use string instead of modal.gpu.T4()
GPU_CONFIG = "T4"

# Container image with all dependencies
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    [
        "vllm>=0.6.0",
        "torch>=2.0.0",
        "transformers>=4.40.0",
        "fastapi>=0.100.0",
        "pydantic>=2.0.0",
        "requests>=2.31.0",
        "PyGithub>=1.59.0",
        "tiktoken>=0.5.0",
        "bitsandbytes>=0.45.3",  # Required for quantized models
        "accelerate>=0.20.0",  # Required for model loading
    ]
)


@dataclass
class SecurityIssue:
    file_path: str
    line_number: Optional[int]
    severity: str  # "HIGH", "MEDIUM", "LOW"
    vulnerability_type: str
    description: str
    recommendation: str
    code_snippet: Optional[str] = None


@dataclass
class FileAnalysis:
    file_path: str
    language: str
    issues: List[SecurityIssue]
    analysis_status: str  # "SUCCESS", "SKIPPED", "ERROR", "TRUNCATED"
    error_message: Optional[str] = None


@dataclass
class RepositoryAnalysis:
    repository_url: str
    total_files_scanned: int
    files_with_issues: int
    total_issues: int
    file_analyses: List[FileAnalysis]
    repository_structure: Dict[str, Any]


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("github-token")],
    timeout=900,
)
def fetch_repository_contents_optimized(repo_url: str) -> Dict[str, Any]:
    """
    Optimized repository fetching with concurrent operations and Tree API.
    """
    import re

    from github import Github

    print(f"🚀 Starting optimized repository fetch for: {repo_url}")
    start_time = time.time()

    # Extract owner/repo from URL
    match = re.match(r"https://github\.com/([^/]+)/([^/]+)", repo_url.rstrip("/"))
    if not match:
        raise ValueError("Invalid GitHub repository URL")

    owner, repo_name = match.groups()

    # Initialize GitHub client
    github_token = os.getenv("GITHUB_TOKEN")
    g = Github(github_token) if github_token else Github()

    try:
        repo = g.get_repo(f"{owner}/{repo_name}")

        # Step 1: Get all files using Tree API (much faster)
        print("📋 Fetching repository tree...")
        tree = repo.get_git_tree(repo.default_branch, recursive=True)

        # Step 2: Filter files early (before downloading content)
        relevant_files = []
        repo_structure = {}

        for item in tree.tree:
            if item.type == "blob":  # It's a file
                file_extension = (
                    item.path.split(".")[-1].lower() if "." in item.path else ""
                )
                filename = os.path.basename(item.path)

                if should_analyze_file(filename, file_extension):
                    # Check file size (skip very large files)
                    if item.size and item.size < 1_000_000:  # 1MB limit
                        relevant_files.append(
                            {
                                "path": item.path,
                                "sha": item.sha,
                                "size": item.size,
                                "extension": file_extension,
                            }
                        )
                        repo_structure[item.path] = {
                            "type": "file",
                            "language": detect_language(file_extension),
                            "size": item.size,
                        }
                    else:
                        repo_structure[item.path] = {
                            "type": "file",
                            "skipped": True,
                            "reason": "too_large",
                        }
                else:
                    repo_structure[item.path] = {
                        "type": "file",
                        "skipped": True,
                        "reason": "unsupported_type",
                    }
            elif item.type == "tree":  # It's a directory
                repo_structure[item.path] = {"type": "directory"}

        print(f"📁 Found {len(relevant_files)} relevant files to analyze")

        # Step 3: Fetch file contents concurrently
        file_data = {}

        def fetch_file_content(file_info):
            """Fetch content for a single file."""
            try:
                blob = repo.get_git_blob(file_info["sha"])
                if blob.encoding == "base64":
                    import base64

                    content = base64.b64decode(blob.content).decode(
                        "utf-8", errors="ignore"
                    )
                else:
                    content = blob.content

                return file_info["path"], {
                    "content": content,
                    "language": detect_language(file_info["extension"]),
                    "size": file_info["size"],
                }
            except Exception as e:
                print(f"❌ Error fetching {file_info['path']}: {e}")
                return file_info["path"], None

        # Use ThreadPoolExecutor for concurrent fetching
        print("⚡ Fetching file contents concurrently...")
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_path = {
                executor.submit(fetch_file_content, file_info): file_info["path"]
                for file_info in relevant_files[:50]  # Limit to 50 files
            }

            for future in as_completed(future_to_path):
                path, content_data = future.result()
                if content_data:
                    file_data[path] = content_data

        elapsed = time.time() - start_time
        print(f"✅ Repository fetch completed in {elapsed:.2f}s")
        print(f"📊 Fetched {len(file_data)} files successfully")

        return {
            "repository_url": repo_url,
            "file_data": file_data,
            "repository_structure": repo_structure,
            "total_files": len(
                [
                    k
                    for k in repo_structure.keys()
                    if repo_structure[k]["type"] == "file"
                ]
            ),
            "fetch_time": elapsed,
        }

    except Exception as e:
        raise Exception(f"Failed to fetch repository: {str(e)}")


def should_analyze_file(filename: str, extension: str) -> bool:
    """Determine if a file should be analyzed based on its extension."""
    # Supported extensions based on training data
    supported_extensions = {
        "py",
        "js",
        "jsx",
        "ts",
        "tsx",
        "java",
        "cpp",
        "cc",
        "cxx",
        "hpp",
        "h",
        "cs",
        "php",
        "rb",
        "swift",
        "go",
        "kt",
        "kts",
        "f",
        "f90",
        "f95",
    }

    # Skip common non-code files
    skip_files = {
        "package.json",
        "package-lock.json",
        "yarn.lock",
        "requirements.txt",
        "setup.py",
        "pyproject.toml",
        "dockerfile",
        "docker-compose.yml",
        "readme.md",
        "license",
        ".gitignore",
    }

    return extension in supported_extensions and filename.lower() not in skip_files


def detect_language(extension: str) -> str:
    """Map file extension to programming language."""
    language_map = {
        "py": "python",
        "js": "javascript",
        "jsx": "javascript",
        "ts": "typescript",
        "tsx": "typescript",
        "java": "java",
        "cpp": "cpp",
        "cc": "cpp",
        "cxx": "cpp",
        "hpp": "cpp",
        "h": "cpp",
        "cs": "csharp",
        "php": "php",
        "rb": "ruby",
        "swift": "swift",
        "go": "go",
        "kt": "kotlin",
        "kts": "kotlin",
        "f": "fortran",
        "f90": "fortran",
        "f95": "fortran",
    }
    return language_map.get(extension, "unknown")


@app.cls(
    image=image.pip_install(["unsloth"]),
    gpu=GPU_CONFIG,
    scaledown_window=600,
    timeout=3600,
)
class SecurityAnalyzer:
    # Use Modal parameters instead of __init__
    model_name: str = modal.parameter(default="AdamDS/qwen3-security-dpo-4b")

    @modal.enter()
    def setup(self):
        """Initialize with Unsloth + PEFT (proven working approach)."""
        print("🚀 Initializing SecurityAnalyzer with Unsloth + PEFT...")
        print(f"📥 Loading LoRA adapters from: {self.model_name}")

        try:
            # Import required libraries
            from peft import PeftModel
            from unsloth import FastLanguageModel

            print("Loading base model with Unsloth...")
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name="unsloth/Qwen3-4B-unsloth-bnb-4bit",
                max_seq_length=2048,
                dtype=None,
                load_in_4bit=True,
            )

            print(f"Loading LoRA adapters from {self.model_name}...")
            self.model = PeftModel.from_pretrained(self.model, self.model_name)

            print("Enabling fast inference...")
            FastLanguageModel.for_inference(self.model)

            # Get model device
            self.device = next(self.model.parameters()).device
            print(f"🔧 Model loaded on device: {self.device}")

            print("✅ SecurityAnalyzer initialized successfully!")

        except Exception as e:
            print(f"❌ Initialization failed: {str(e)}")
            raise

    def analyze_file(
        self, file_path: str, file_content: str, language: str
    ) -> FileAnalysis:
        """Analyze a single file for security vulnerabilities."""
        try:
            # Check if file is too large for context window
            if self._estimate_tokens(file_content) > 1200:
                return self._analyze_large_file(file_path, file_content, language)

            # Create security analysis prompt
            prompt = self._create_security_prompt(file_path, file_content, language)

            # Generate analysis using Unsloth
            analysis_text = self._generate_with_unsloth(prompt)
            issues = self._parse_analysis(analysis_text, file_path)

            return FileAnalysis(
                file_path=file_path,
                language=language,
                issues=issues,
                analysis_status="SUCCESS",
            )

        except Exception as e:
            return FileAnalysis(
                file_path=file_path,
                language=language,
                issues=[],
                analysis_status="ERROR",
                error_message=str(e),
            )

    def _generate_with_unsloth(self, prompt: str) -> str:
        """Generate analysis using Unsloth model."""
        import torch

        # Tokenize input
        inputs = self.tokenizer([prompt], return_tensors="pt")

        # Move inputs to model device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
                repetition_penalty=1.1,
            )

        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract just the analysis part (remove the original prompt)
        if len(response) > len(prompt):
            analysis = response[len(prompt) :].strip()
        else:
            analysis = response.strip()

        return analysis

    def _create_security_prompt(
        self, file_path: str, file_content: str, language: str
    ) -> str:
        """Create a structured prompt for security analysis."""
        return f"""<|im_start|>system
You are a security expert analyzing code for vulnerabilities. Focus on:

1. **SQL Injection** - Unsafe database queries
2. **Cross-Site Scripting (XSS)** - Unescaped user input
3. **Authentication/Authorization** - Weak access controls
4. **Input Validation** - Missing input sanitization
5. **Cryptographic Issues** - Weak encryption, hardcoded secrets
6. **Path Traversal** - Unsafe file operations
7. **Command Injection** - Unsafe system commands
8. **Buffer Overflows** - Memory safety issues
9. **Race Conditions** - Concurrency issues
10. **Information Disclosure** - Sensitive data exposure

For each issue, provide:
```
ISSUE_START
File: {file_path}
Line: [line_number]
Severity: [HIGH/MEDIUM/LOW]
Type: [vulnerability_type]
Description: [description]
Recommendation: [fix_recommendation]
Code: [code_snippet]
ISSUE_END
```

If no issues found, respond: "NO_ISSUES_FOUND"
<|im_end|>

<|im_start|>user
Analyze this {language} file:

**File:** {file_path}

```{language}
{file_content}
```
<|im_end|>

<|im_start|>assistant
I'll analyze this {language} code for security vulnerabilities.

"""

    def _analyze_large_file(
        self, file_path: str, file_content: str, language: str
    ) -> FileAnalysis:
        """Handle files that exceed context window by chunking."""
        chunks = self._split_code_intelligently(file_content, language)
        all_issues = []

        for i, chunk in enumerate(chunks):
            chunk_prompt = self._create_security_prompt(
                f"{file_path} (chunk {i+1}/{len(chunks)})", chunk, language
            )

            try:
                analysis_text = self._generate_with_unsloth(chunk_prompt)
                chunk_issues = self._parse_analysis(analysis_text, file_path)
                all_issues.extend(chunk_issues)

            except Exception as e:
                print(f"Error analyzing chunk {i+1} of {file_path}: {e}")

        return FileAnalysis(
            file_path=file_path,
            language=language,
            issues=all_issues,
            analysis_status="TRUNCATED" if len(chunks) > 1 else "SUCCESS",
        )

    def _split_code_intelligently(self, content: str, language: str) -> List[str]:
        """Split code into logical chunks based on language syntax."""
        lines = content.split("\n")
        chunks = []
        current_chunk = []
        current_size = 0
        max_chunk_tokens = 1000

        # Language-specific function/class detection
        if language == "python":
            function_keywords = ["def ", "class ", "async def "]
        elif language in ["javascript", "typescript"]:
            keywords = ["function ", "class ", "const ", "let ", "var "]
            function_keywords = keywords
        elif language == "java":
            function_keywords = [
                "public ",
                "private ",
                "protected ",
                "class ",
                "interface ",
            ]
        else:
            function_keywords = ["{", "}", "function", "class", "def"]

        for line in lines:
            line_tokens = self._estimate_tokens(line)

            if (
                current_size + line_tokens > max_chunk_tokens
                and current_chunk
                and any(keyword in line for keyword in function_keywords)
            ):

                chunks.append("\n".join(current_chunk))
                current_chunk = [line]
                current_size = line_tokens
            else:
                current_chunk.append(line)
                current_size += line_tokens

        if current_chunk:
            chunks.append("\n".join(current_chunk))

        return chunks if chunks else [content]

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation."""
        return len(text) // 3

    def _parse_analysis(
        self, analysis_text: str, file_path: str
    ) -> List[SecurityIssue]:
        """Parse the model's analysis into structured SecurityIssue objects."""
        if "NO_ISSUES_FOUND" in analysis_text:
            return []

        issues = []
        import re

        issue_pattern = r"ISSUE_START\s*\n(.*?)\nISSUE_END"
        matches = re.findall(issue_pattern, analysis_text, re.DOTALL)

        for match in matches:
            try:
                issue_data = {}
                for line in match.split("\n"):
                    if ":" in line:
                        key, value = line.split(":", 1)
                        issue_data[key.strip().lower()] = value.strip()

                line_num = None
                if "line" in issue_data:
                    try:
                        line_num = int(issue_data["line"])
                    except (ValueError, TypeError):
                        line_num = None

                issue = SecurityIssue(
                    file_path=file_path,
                    line_number=line_num,
                    severity=issue_data.get("severity", "MEDIUM").upper(),
                    vulnerability_type=issue_data.get("type", "Unknown"),
                    description=issue_data.get("description", ""),
                    recommendation=issue_data.get("recommendation", ""),
                    code_snippet=issue_data.get("code", ""),
                )
                issues.append(issue)

            except Exception as e:
                print(f"Error parsing issue: {e}")
                continue

        return issues

    @modal.method()
    def analyze_repository(self, repo_data: Dict[str, Any]) -> RepositoryAnalysis:
        """Analyze an entire repository for security vulnerabilities."""
        file_analyses = []
        total_issues = 0
        files_with_issues = 0

        print(f"🔍 Analyzing {len(repo_data['file_data'])} files...")

        for file_path, file_info in repo_data["file_data"].items():
            print(f"Analyzing: {file_path}")

            # Call analyze_file as a regular instance method
            analysis = self.analyze_file(
                file_path=file_path,
                file_content=file_info["content"],
                language=file_info["language"],
            )

            file_analyses.append(analysis)

            if analysis.issues:
                files_with_issues += 1
                total_issues += len(analysis.issues)

        return RepositoryAnalysis(
            repository_url=repo_data["repository_url"],
            total_files_scanned=len(repo_data["file_data"]),
            files_with_issues=files_with_issues,
            total_issues=total_issues,
            file_analyses=file_analyses,
            repository_structure=repo_data["repository_structure"],
        )


def calculate_risk_score(severity_counts: Dict[str, int]) -> int:
    """Calculate overall risk score (0-100) based on issue severities."""
    high_weight = 10
    medium_weight = 5
    low_weight = 1

    total_score = (
        severity_counts.get("HIGH", 0) * high_weight
        + severity_counts.get("MEDIUM", 0) * medium_weight
        + severity_counts.get("LOW", 0) * low_weight
    )

    max_possible_score = 100
    return min(100, int((total_score / max_possible_score) * 100))


def generate_general_recommendations(vulnerability_types: Dict[str, int]) -> List[str]:
    """Generate general security recommendations."""
    recommendations = []

    if "SQL Injection" in vulnerability_types:
        recommendations.append("Implement parameterized queries and input validation")

    if "Cross-Site Scripting" in vulnerability_types:
        recommendations.append("Sanitize and escape all user inputs")

    if "Authentication" in vulnerability_types:
        recommendations.append("Strengthen authentication and session management")

    if "Cryptographic" in vulnerability_types:
        recommendations.append("Use strong encryption and secure key management")

    if not recommendations:
        recommendations.append("Continue following secure coding practices")

    return recommendations


@app.function(
    image=image.pip_install(["fastapi", "uvicorn"]),
    timeout=3600,
)
@modal.concurrent(max_inputs=10)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel

    app = FastAPI(title="LLMGuard Security Analyzer API")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    class AnalysisRequest(BaseModel):
        repository_url: str

    class AnalysisResponse(BaseModel):
        repository_url: str
        total_files_scanned: int
        files_with_issues: int
        total_issues: int
        file_analyses: List[Dict]
        repository_structure: Dict
        analysis_summary: Dict

    @app.post("/analyze", response_model=AnalysisResponse)
    async def analyze_repository(request: AnalysisRequest):
        """Main endpoint for repository security analysis."""
        try:
            # Step 1: Fetch repository contents (optimized)
            print(f"Fetching repository: {request.repository_url}")
            repo_data = fetch_repository_contents_optimized.remote(
                request.repository_url
            )

            # Step 2: Analyze with security model
            print("Starting security analysis...")
            analyzer = SecurityAnalyzer()
            analysis = analyzer.analyze_repository.remote(repo_data)

            # Step 3: Generate summary statistics
            severity_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
            vulnerability_types = {}

            for file_analysis in analysis.file_analyses:
                for issue in file_analysis.issues:
                    severity_counts[issue.severity] = (
                        severity_counts.get(issue.severity, 0) + 1
                    )
                    vuln_type = issue.vulnerability_type
                    vulnerability_types[vuln_type] = (
                        vulnerability_types.get(vuln_type, 0) + 1
                    )

            analysis_summary = {
                "severity_breakdown": severity_counts,
                "vulnerability_types": vulnerability_types,
                "risk_score": calculate_risk_score(severity_counts),
                "recommendations": generate_general_recommendations(
                    vulnerability_types
                ),
                "fetch_time": repo_data.get("fetch_time", 0),
            }

            return AnalysisResponse(
                repository_url=analysis.repository_url,
                total_files_scanned=analysis.total_files_scanned,
                files_with_issues=analysis.files_with_issues,
                total_issues=analysis.total_issues,
                file_analyses=[
                    file_analysis.__dict__ for file_analysis in analysis.file_analyses
                ],
                repository_structure=analysis.repository_structure,
                analysis_summary=analysis_summary,
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "model": "qwen3-security-dpo-4b"}

    return app

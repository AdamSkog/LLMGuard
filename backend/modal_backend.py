import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import modal

# Modal app configuration
app = modal.App("llmguard-security-analyzer")

# GPU configuration - Either T4 or A10G
GPU_CONFIG = "A10G"

# Model caching with Modal Volume
model_volume = modal.Volume.from_name(
    "qwen3-security-model-cache", create_if_missing=True
)
MODEL_CACHE_PATH = "/models"

# Container image with all dependencies - UPDATED FOR UNSLOTH
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        [
            "unsloth",
            "torch>=2.0.0",
            "transformers>=4.40.0",
            "fastapi>=0.100.0",
            "pydantic>=2.0.0",
            "requests>=2.31.0",
            "PyGithub>=1.59.0",
            "tiktoken>=0.5.0",
            "bitsandbytes>=0.45.3",
            "accelerate>=0.20.0",
            "huggingface_hub>=0.19.0",
            "hf_transfer>=0.1.0",
            "python-dotenv>=1.0.0",
            "peft>=0.8.0",  # For LoRA adapters
            "xformers",  # For memory efficiency
        ]
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",  # Faster downloads from HF
            "HF_HUB_CACHE": MODEL_CACHE_PATH,  # Point HF cache to our volume
        }
    )
)

# Model configuration - UPDATED FOR UNSLOTH + LORA
BASE_MODEL_NAME = "unsloth/Qwen3-4B-unsloth-bnb-4bit"
LORA_ADAPTERS_NAME = "AdamDS/qwen3-security-dpo-4b"

# Your merged model configuration
MERGED_MODEL_NAME = "AdamDS/qwen3-security-merged-4b"

# Create global SecurityAnalyzer instance at module level
# security_analyzer = SecurityAnalyzer()  -> MOVE THIS AFTER CLASS DEFINITION


@dataclass
class SecurityIssue:
    file_path: str
    line_number: Optional[int]
    severity: str  # "HIGH", "MEDIUM", "LOW"
    vulnerability_type: str
    description: str
    recommendation: str
    code_snippet: Optional[str] = None

    def to_dict(self):
        """Convert SecurityIssue to dictionary for JSON serialization."""
        return {
            "file_path": self.file_path,
            "line_number": self.line_number,
            "severity": self.severity,
            "vulnerability_type": self.vulnerability_type,
            "description": self.description,
            "recommendation": self.recommendation,
            "code_snippet": self.code_snippet,
        }


@dataclass
class FileAnalysis:
    file_path: str
    language: str
    issues: List[SecurityIssue]
    analysis_status: str  # "SUCCESS", "SKIPPED", "ERROR", "TRUNCATED"
    error_message: Optional[str] = None

    def to_dict(self):
        """Convert FileAnalysis to dictionary for JSON serialization."""
        return {
            "file_path": self.file_path,
            "language": self.language,
            "issues": [issue.to_dict() for issue in self.issues],
            "analysis_status": self.analysis_status,
            "error_message": self.error_message,
        }


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
    volumes={MODEL_CACHE_PATH: model_volume},
    secrets=[modal.Secret.from_name("github-token")],
    timeout=1800,  # 30 minutes for initial download
)
def download_base_model_to_volume():
    """
    Download the base Unsloth model to Modal Volume for caching.
    """
    from pathlib import Path

    from huggingface_hub import snapshot_download

    print(f"🚀 Downloading base model {BASE_MODEL_NAME} to volume cache...")
    print(f"📁 Cache directory: {MODEL_CACHE_PATH}")

    # Check if base model is already cached
    base_model_path = Path(MODEL_CACHE_PATH) / BASE_MODEL_NAME.replace("/", "--")

    if base_model_path.exists() and any(base_model_path.iterdir()):
        print("✅ Base model already cached in volume!")
        return {"status": "already_cached", "model_path": str(base_model_path)}

    try:
        # Download base model to volume
        local_model_path = snapshot_download(
            repo_id=BASE_MODEL_NAME,
            local_dir=str(base_model_path),
            ignore_patterns=["*.pt", "*.bin"],
            resume_download=True,
        )

        # Commit changes to volume
        model_volume.commit()

        print(f"✅ Base model downloaded successfully to {local_model_path}")
        print("📊 Volume committed - base model ready for fast loading!")

        return {
            "status": "downloaded",
            "model_path": local_model_path,
            "cached_size": sum(
                f.stat().st_size for f in base_model_path.rglob("*") if f.is_file()
            ),
        }

    except Exception as e:
        print(f"❌ Error downloading base model: {e}")
        raise


@app.function(
    image=image,
    volumes={MODEL_CACHE_PATH: model_volume},
    secrets=[modal.Secret.from_name("github-token")],
    timeout=1800,
)
def download_lora_adapters_to_volume():
    """
    Download the LoRA adapters to Modal Volume for caching.
    """
    from pathlib import Path

    from huggingface_hub import snapshot_download

    print(f"🚀 Downloading LoRA adapters {LORA_ADAPTERS_NAME} to volume cache...")
    print(f"📁 Cache directory: {MODEL_CACHE_PATH}")

    # Check if LoRA adapters are already cached
    lora_path = Path(MODEL_CACHE_PATH) / LORA_ADAPTERS_NAME.replace("/", "--")

    if lora_path.exists() and any(lora_path.iterdir()):
        print("✅ LoRA adapters already cached in volume!")
        return {"status": "already_cached", "model_path": str(lora_path)}

    try:
        # Download LoRA adapters to volume
        local_lora_path = snapshot_download(
            repo_id=LORA_ADAPTERS_NAME,
            local_dir=str(lora_path),
            resume_download=True,
        )

        # Commit changes to volume
        model_volume.commit()

        print(f"✅ LoRA adapters downloaded successfully to {local_lora_path}")
        print("📊 Volume committed - LoRA adapters ready for fast loading!")

        return {
            "status": "downloaded",
            "model_path": local_lora_path,
            "cached_size": sum(
                f.stat().st_size for f in lora_path.rglob("*") if f.is_file()
            ),
        }

    except Exception as e:
        print(f"❌ Error downloading LoRA adapters: {e}")
        raise


@app.function(
    image=image,
    volumes={MODEL_CACHE_PATH: model_volume},
    timeout=300,
)
def check_model_cache():
    """Check if both base model and LoRA adapters are cached."""
    from pathlib import Path

    base_model_path = Path(MODEL_CACHE_PATH) / BASE_MODEL_NAME.replace("/", "--")
    lora_path = Path(MODEL_CACHE_PATH) / LORA_ADAPTERS_NAME.replace("/", "--")

    base_cached = base_model_path.exists() and any(base_model_path.iterdir())
    lora_cached = lora_path.exists() and any(lora_path.iterdir())

    result = {
        "base_model": {
            "cached": base_cached,
            "path": str(base_model_path),
        },
        "lora_adapters": {
            "cached": lora_cached,
            "path": str(lora_path),
        },
        "both_cached": base_cached and lora_cached,
    }

    if base_cached:
        base_file_count = len(list(base_model_path.rglob("*")))
        base_size = sum(
            f.stat().st_size for f in base_model_path.rglob("*") if f.is_file()
        )
        result["base_model"].update(
            {
                "file_count": base_file_count,
                "total_size_mb": base_size / (1024 * 1024),
            }
        )

    if lora_cached:
        lora_file_count = len(list(lora_path.rglob("*")))
        lora_size = sum(f.stat().st_size for f in lora_path.rglob("*") if f.is_file())
        result["lora_adapters"].update(
            {
                "file_count": lora_file_count,
                "total_size_mb": lora_size / (1024 * 1024),
            }
        )

    return result


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
    image=image,
    volumes={MODEL_CACHE_PATH: model_volume},  # Mount the model cache volume
    gpu=GPU_CONFIG,
    scaledown_window=600,
    timeout=7200,  # Increased to 2 hours for large repository analysis
)
class SecurityAnalyzer:
    # Use Modal parameters instead of __init__
    base_model_name: str = modal.parameter(default=BASE_MODEL_NAME)
    lora_adapters_name: str = modal.parameter(default=LORA_ADAPTERS_NAME)

    @modal.enter()
    def setup(self):
        """Initialize with Unsloth + LoRA adapters using cached models."""
        from pathlib import Path

        print("🚀 Initializing SecurityAnalyzer with Unsloth + LoRA...")
        print(f"📥 Base model: {self.base_model_name}")
        print(f"🎯 LoRA adapters: {self.lora_adapters_name}")
        print(f"📁 Model cache path: {MODEL_CACHE_PATH}")

        # Check if both models are cached locally
        base_model_path = Path(MODEL_CACHE_PATH) / self.base_model_name.replace(
            "/", "--"
        )
        lora_path = Path(MODEL_CACHE_PATH) / self.lora_adapters_name.replace("/", "--")

        base_cached = base_model_path.exists() and any(base_model_path.iterdir())
        lora_cached = lora_path.exists() and any(lora_path.iterdir())

        if not base_cached:
            print(f"❌ Base model not found in cache: {base_model_path}")
            print("Please run download_base_model_to_volume() first.")
        if not lora_cached:
            print(f"❌ LoRA adapters not found in cache: {lora_path}")
            print("Please run download_lora_adapters_to_volume() first.")

        if not (base_cached and lora_cached):
            raise RuntimeError(
                "Required models not cached. Run download functions first."
            )

        print(f"✅ Found cached base model at: {base_model_path}")
        print(f"✅ Found cached LoRA adapters at: {lora_path}")

        try:
            from peft import PeftModel
            from unsloth import FastLanguageModel

            print("🔥 Loading base model with Unsloth...")

            # Load base model from cache or HuggingFace
            model_to_load = (
                str(base_model_path) if base_cached else self.base_model_name
            )

            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_to_load,
                max_seq_length=2048,
                dtype=None,  # Auto-detect
                load_in_4bit=True,
                trust_remote_code=True,
            )

            print("🎯 Loading LoRA adapters...")

            # Load LoRA adapters from cache or HuggingFace
            lora_to_load = str(lora_path) if lora_cached else self.lora_adapters_name

            self.model = PeftModel.from_pretrained(self.model, lora_to_load)

            print("⚡ Enabling fast inference...")
            FastLanguageModel.for_inference(self.model)

            # Get model device for later use
            self.device = next(self.model.parameters()).device
            print(f"🎮 Model loaded on device: {self.device}")

            print("✅ Unsloth SecurityAnalyzer initialization complete!")

        except Exception as e:
            print(f"❌ Error loading models: {str(e)}")
            # Fallback: Try loading from HuggingFace directly
            print("🔄 Falling back to HuggingFace direct loading...")
            try:
                from peft import PeftModel
                from unsloth import FastLanguageModel

                self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                    model_name=self.base_model_name,
                    max_seq_length=2048,
                    dtype=None,
                    load_in_4bit=True,
                    trust_remote_code=True,
                )

                self.model = PeftModel.from_pretrained(
                    self.model, self.lora_adapters_name
                )
                FastLanguageModel.for_inference(self.model)
                self.device = next(self.model.parameters()).device

                print("✅ Fallback model loading successful!")

            except Exception as e2:
                print(f"❌ All loading strategies failed: {str(e2)}")
                raise RuntimeError(f"Failed to load models: {str(e2)}")

    def analyze_file(
        self, file_path: str, file_content: str, language: str
    ) -> FileAnalysis:
        """Analyze a single file for security vulnerabilities - OPTIMIZED."""
        try:
            print(f"🔍 Starting analysis of {file_path} ({language})")

            # SPEED OPTIMIZATION: Skip tiny files and non-risky file types
            if len(file_content.strip()) < 50:
                print(f"⏭️  Skipping {file_path}: file too small")
                return FileAnalysis(
                    file_path=file_path,
                    language=language,
                    issues=[],
                    analysis_status="SKIPPED",
                    error_message="File too small",
                )

            # SPEED OPTIMIZATION: Skip obviously safe files
            safe_patterns = [
                "test",
                "spec",
                "config",
                "README",
                "package.json",
                "yarn.lock",
            ]
            if any(pattern.lower() in file_path.lower() for pattern in safe_patterns):
                print(f"⏭️  Skipping {file_path}: low-risk file type")
                return FileAnalysis(
                    file_path=file_path,
                    language=language,
                    issues=[],
                    analysis_status="SKIPPED",
                    error_message="Low-risk file type",
                )

            # Check if file is too large for context window
            if self._estimate_tokens(file_content) > 800:
                print(f"📄 {file_path}: Large file, using chunking")
                return self._analyze_large_file(file_path, file_content, language)

            # Create shortened security analysis prompt
            prompt = self._create_security_prompt_short(
                file_path, file_content, language
            )

            print(f"🤖 Generating analysis for {file_path}")

            # Generate analysis using Unsloth (removed signal-based timeout)
            analysis_text = self._generate_with_unsloth(prompt)
            print(f"📝 Raw analysis for {file_path}: {analysis_text[:200]}...")
            issues = self._parse_analysis(analysis_text, file_path)
            print(f"✅ Found {len(issues)} issues in {file_path}")

            return FileAnalysis(
                file_path=file_path,
                language=language,
                issues=issues,
                analysis_status="SUCCESS",
            )

        except Exception as e:
            print(f"❌ Error analyzing {file_path}: {e}")
            return FileAnalysis(
                file_path=file_path,
                language=language,
                issues=[],
                analysis_status="ERROR",
                error_message=str(e),
            )

    def _generate_with_unsloth(self, prompt: str) -> str:
        """Generate analysis using Unsloth - OPTIMIZED FOR SPEED."""
        try:
            import torch

            # Check if tokenizer exists (remove debug logs)
            if not hasattr(self, "tokenizer"):
                return "ERROR: SecurityAnalyzer not properly initialized"

            if not hasattr(self, "model"):
                return "ERROR: SecurityAnalyzer model not properly initialized"

            # Tokenize and ensure tensors are on the same device as model
            inputs = self.tokenizer([prompt], return_tensors="pt")

            # Move inputs to the same device as the model
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,  # Reduced from 2048 to 512 for faster generation
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                    repetition_penalty=1.05,
                )

            # Decode the response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract just the generated part (after the prompt)
            generated_text = response[len(prompt) :].strip()
            return generated_text

        except Exception as e:
            print(f"❌ Unsloth generation error: {e}")
            import traceback

            traceback.print_exc()
            return "ERROR: Generation failed"

    def _create_security_prompt_short(
        self, file_path: str, file_content: str, language: str
    ) -> str:
        """Create a SHORTENED prompt for faster analysis using chat template."""
        # Use proper chat template with thinking disabled for Qwen3
        messages = [
            {
                "role": "system",
                "content": "You are a security scanner. Output ONLY the specified format. No explanations.\n\nSCAN FOR: SQL injection, XSS, auth issues, input validation, crypto problems, path traversal, command injection.\n\nOUTPUT FORMAT (use exactly this):\nISSUE_START\nFile: "
                + file_path
                + '\nLine: [number]\nSeverity: HIGH/MEDIUM/LOW\nType: [vulnerability_type]\nDescription: [brief_description]\nRecommendation: [fix]\nISSUE_END\n\nIf no issues: "NO_ISSUES_FOUND"\n\nDO NOT explain your reasoning. Output the format immediately.',
            },
            {
                "role": "user",
                "content": f"Scan this {language} code:\n\n```{language}\n{file_content}\n```",
            },
        ]

        # Apply chat template with thinking disabled for Qwen3
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,  # Disable Qwen3 thinking mode for speed
        )

        return prompt

    def _analyze_large_file(
        self, file_path: str, file_content: str, language: str
    ) -> FileAnalysis:
        """Handle files that exceed context window by chunking."""
        chunks = self._split_code_intelligently(file_content, language)
        all_issues = []

        # Process chunks sequentially for now (can be optimized later)
        for i, chunk in enumerate(chunks):
            try:
                chunk_prompt = self._create_security_prompt_short(
                    f"{file_path} (chunk {i+1}/{len(chunks)})", chunk, language
                )
                analysis_text = self._generate_with_unsloth(chunk_prompt)
                chunk_issues = self._parse_analysis(analysis_text, file_path)
                all_issues.extend(chunk_issues)
            except Exception as e:
                print(f"Error analyzing chunk {i+1} of {file_path}: {e}")
                continue

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

    def analyze_repository_parallel(
        self, repo_data: Dict[str, Any]
    ) -> RepositoryAnalysis:
        """Analyze multiple files in a repository concurrently."""
        print("🚀 Starting repository analysis...")
        print(f"📁 Total files to analyze: {len(repo_data['file_data'])}")

        files_to_analyze = [
            (file_path, file_info)
            for file_path, file_info in repo_data["file_data"].items()
        ]

        files_with_issues = 0
        total_issues = 0
        file_analyses = []

        # Process files sequentially for now (Unsloth doesn't support batch like vLLM)
        for file_path, file_info in files_to_analyze:
            content = file_info["content"]
            language = file_info["language"]

            analysis = self.analyze_file(file_path, content, language)
            file_analyses.append(analysis)

            if analysis.issues:
                files_with_issues += 1
                total_issues += len(analysis.issues)

        print(
            f"✅ Analysis complete: {files_with_issues}/{len(files_to_analyze)} files with issues"
        )

        return RepositoryAnalysis(
            repository_url=repo_data["repository_url"],
            total_files_scanned=len(files_to_analyze),
            files_with_issues=files_with_issues,
            total_issues=total_issues,
            file_analyses=file_analyses,
            repository_structure=repo_data["repository_structure"],
        )

    def _generate_llm_recommendations(
        self, vulnerability_types: Dict[str, int]
    ) -> List[str]:
        """Generate LLM-based recommendations for found vulnerabilities."""
        if not vulnerability_types:
            return [
                "No security issues found. Continue following secure coding practices."
            ]

        # Create a prompt for generating recommendations
        vuln_summary = ", ".join(
            [
                f"{count} {vuln_type} issues"
                for vuln_type, count in vulnerability_types.items()
            ]
        )

        # Use much more directive prompt to avoid extra text
        messages = [
            {
                "role": "system",
                "content": "You are a security expert. Output ONLY numbered recommendations. No introduction. No explanations. Start immediately with '1.'",
            },
            {
                "role": "user",
                "content": f"Vulnerabilities found: {vuln_summary}. Output exactly 3 recommendations:\n\n1.\n2.\n3.",
            },
        ]

        try:
            # Apply chat template with thinking disabled
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,  # Disable thinking for speed
            )

            # Use higher token limit for recommendations to avoid truncation
            import torch

            inputs = self.tokenizer([prompt], return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=800,  # Reduced but still enough for 3 recommendations
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                )

            recommendations_text = self.tokenizer.decode(
                outputs[0], skip_special_tokens=True
            )
            generated_part = recommendations_text[len(prompt) :].strip()

            # Much more aggressive parsing to extract clean recommendations
            recommendations = []

            # Split by lines and process
            lines = generated_part.split("\n")
            current_recommendation = ""

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Check if this starts a new numbered recommendation
                if any(line.startswith(f"{i}.") for i in range(1, 6)):
                    # Save previous recommendation if exists
                    if current_recommendation:
                        recommendations.append(current_recommendation.strip())

                    # Start new recommendation, remove number
                    if line[1:2] == ".":
                        current_recommendation = line[2:].strip()
                    elif line[1:3] == ". ":
                        current_recommendation = line[3:].strip()
                    else:
                        current_recommendation = line
                else:
                    # Continue previous recommendation
                    if current_recommendation:
                        current_recommendation += " " + line

            # Don't forget the last recommendation
            if current_recommendation:
                recommendations.append(current_recommendation.strip())

            # Clean up and filter recommendations
            clean_recommendations = []
            for rec in recommendations:
                # Remove common prefixes and clean up
                rec = rec.strip("- •*").strip()
                if rec.startswith("**"):
                    rec = rec.strip("*").strip()

                # Only keep substantial recommendations
                if len(rec) > 20 and not rec.lower().startswith(
                    ("here", "based", "the following", "to address")
                ):
                    clean_recommendations.append(rec)

            # Ensure we have at least some recommendations
            if not clean_recommendations:
                print("⚠️ LLM recommendations parsing failed, using fallback")
                return generate_fallback_recommendations(vulnerability_types)

            # Limit to 3 recommendations max
            return clean_recommendations[:3]

        except Exception as e:
            print(f"❌ Error generating LLM recommendations: {e}")
            # Fallback to basic recommendations
            return generate_fallback_recommendations(vulnerability_types)

    @modal.fastapi_endpoint(method=["POST", "OPTIONS"], label="analyze-repository")
    def analyze_repository_endpoint(self, request: Dict):
        """Modal web endpoint for repository analysis with CORS support."""
        try:
            # Handle CORS preflight
            from fastapi import Request
            from fastapi.responses import JSONResponse

            # Check if this is an OPTIONS request
            if hasattr(request, "method") and request.method == "OPTIONS":
                response = JSONResponse(content={})
                response.headers["Access-Control-Allow-Origin"] = "*"
                response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
                response.headers["Access-Control-Allow-Headers"] = "*"
                response.headers["Access-Control-Max-Age"] = "86400"
                return response

            repository_url = request.get("repository_url")
            if not repository_url:
                response = {"error": "repository_url is required"}
            else:
                print(f"🔍 Analyzing repository: {repository_url}")

                # Check model initialization
                if not hasattr(self, "model") or not hasattr(self, "tokenizer"):
                    print("❌ Model not properly initialized!")
                    response = {"error": "Security analyzer model not initialized"}
                else:
                    print("✅ Model and tokenizer are initialized")
                    print(f"📊 Model device: {getattr(self, 'device', 'unknown')}")

                    # Fetch repository data
                    repo_data = fetch_repository_contents_optimized.remote(
                        repository_url
                    )
                    print(
                        f"📁 Fetched {len(repo_data['file_data'])} files for analysis"
                    )

                    # Call analysis method differently to avoid Function object issue
                    analysis = self._perform_repository_analysis(repo_data)

                    print(
                        f"🔍 Analysis results: {analysis.total_issues} total issues found"
                    )
                    print(f"📄 Files with issues: {analysis.files_with_issues}")

                    # Generate summary statistics
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
                        "recommendations": self._generate_llm_recommendations(
                            vulnerability_types
                        ),
                        "fetch_time": repo_data.get("fetch_time", 0),
                    }

                    response = {
                        "repository_url": analysis.repository_url,
                        "total_files_scanned": analysis.total_files_scanned,
                        "files_with_issues": analysis.files_with_issues,
                        "total_issues": analysis.total_issues,
                        "file_analyses": [
                            fa.to_dict() for fa in analysis.file_analyses
                        ],
                        "repository_structure": analysis.repository_structure,
                        "analysis_summary": analysis_summary,
                    }

            # Create response with CORS headers
            json_response = JSONResponse(content=response)
            json_response.headers["Access-Control-Allow-Origin"] = "*"
            json_response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            json_response.headers["Access-Control-Allow-Headers"] = "*"

            return json_response

        except Exception as e:
            print(f"❌ Analysis error: {e}")
            import traceback

            traceback.print_exc()

            error_response = JSONResponse(
                content={"error": f"Analysis failed: {str(e)}"}
            )
            error_response.headers["Access-Control-Allow-Origin"] = "*"
            error_response.headers["Access-Control-Allow-Methods"] = (
                "GET, POST, OPTIONS"
            )
            error_response.headers["Access-Control-Allow-Headers"] = "*"

            return error_response

    def _perform_repository_analysis(
        self, repo_data: Dict[str, Any]
    ) -> RepositoryAnalysis:
        """Internal method to perform repository analysis - avoids Modal Function object issues."""
        print("🚀 Starting repository analysis...")
        print(f"📁 Total files to analyze: {len(repo_data['file_data'])}")

        files_to_analyze = [
            (file_path, file_info)
            for file_path, file_info in repo_data["file_data"].items()
        ]

        files_with_issues = 0
        total_issues = 0
        file_analyses = []

        # Process files sequentially with progress tracking
        for i, (file_path, file_info) in enumerate(files_to_analyze, 1):
            print(f"📄 Analyzing file {i}/{len(files_to_analyze)}: {file_path}")
            content = file_info["content"]
            language = file_info["language"]

            analysis = self.analyze_file(file_path, content, language)
            file_analyses.append(analysis)

            if analysis.issues:
                files_with_issues += 1
                total_issues += len(analysis.issues)
                print(f"   ⚠️  Found {len(analysis.issues)} issues")
            else:
                print("   ✅ No issues found")

        print(
            f"✅ Analysis complete: {files_with_issues}/{len(files_to_analyze)} files with issues"
        )

        return RepositoryAnalysis(
            repository_url=repo_data["repository_url"],
            total_files_scanned=len(files_to_analyze),
            files_with_issues=files_with_issues,
            total_issues=total_issues,
            file_analyses=file_analyses,
            repository_structure=repo_data["repository_structure"],
        )

    @modal.fastapi_endpoint(method=["POST", "OPTIONS"], label="analyze-file")
    def analyze_file_endpoint(self, request: Dict):
        """Modal web endpoint for single file analysis with CORS support."""
        try:
            from fastapi.responses import JSONResponse

            # Handle CORS preflight
            if hasattr(request, "method") and request.method == "OPTIONS":
                response = JSONResponse(content={})
                response.headers["Access-Control-Allow-Origin"] = "*"
                response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
                response.headers["Access-Control-Allow-Headers"] = "*"
                response.headers["Access-Control-Max-Age"] = "86400"
                return response

            file_path = request.get("file_path", "unknown.py")
            file_content = request.get("file_content", "")
            language = request.get("language", "python")

            if not file_content:
                response = {"error": "file_content is required"}
            else:
                print(f"🔍 Analyzing file: {file_path}")

                # Analyze with this class instance
                analysis = self.analyze_file(file_path, file_content, language)

                response = {
                    "file_path": analysis.file_path,
                    "language": analysis.language,
                    "analysis_status": analysis.analysis_status,
                    "issues_found": len(analysis.issues),
                    "issues": [issue.to_dict() for issue in analysis.issues],
                    "error_message": analysis.error_message,
                }

            json_response = JSONResponse(content=response)
            json_response.headers["Access-Control-Allow-Origin"] = "*"
            json_response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            json_response.headers["Access-Control-Allow-Headers"] = "*"

            return json_response

        except Exception as e:
            print(f"❌ File analysis error: {e}")
            import traceback

            traceback.print_exc()

            error_response = JSONResponse(
                content={"error": f"File analysis failed: {str(e)}"}
            )
            error_response.headers["Access-Control-Allow-Origin"] = "*"
            error_response.headers["Access-Control-Allow-Methods"] = (
                "GET, POST, OPTIONS"
            )
            error_response.headers["Access-Control-Allow-Headers"] = "*"

            return error_response

    @modal.fastapi_endpoint(method=["POST", "OPTIONS"], label="debug-analysis")
    def debug_analysis_endpoint(self, request: Dict):
        """Debug endpoint to see raw model output with CORS support."""
        try:
            from fastapi.responses import JSONResponse

            # Handle CORS preflight
            if hasattr(request, "method") and request.method == "OPTIONS":
                response = JSONResponse(content={})
                response.headers["Access-Control-Allow-Origin"] = "*"
                response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
                response.headers["Access-Control-Allow-Headers"] = "*"
                response.headers["Access-Control-Max-Age"] = "86400"
                return response

            file_path = request.get("file_path", "unknown.py")
            file_content = request.get("file_content", "")
            language = request.get("language", "python")

            if not file_content:
                response = {"error": "file_content is required"}
            else:
                print(f"🔍 Debug analyzing file: {file_path}")

                # Create the prompt
                prompt = self._create_security_prompt_short(
                    file_path, file_content, language
                )

                # Generate raw analysis
                raw_analysis = self._generate_with_unsloth(prompt)

                # Try to parse it
                parsed_issues = self._parse_analysis(raw_analysis, file_path)

                response = {
                    "file_path": file_path,
                    "language": language,
                    "prompt": prompt,
                    "raw_model_output": raw_analysis,
                    "parsed_issues_count": len(parsed_issues),
                    "parsed_issues": [issue.to_dict() for issue in parsed_issues],
                    "contains_no_issues": "NO_ISSUES_FOUND" in raw_analysis,
                }

            json_response = JSONResponse(content=response)
            json_response.headers["Access-Control-Allow-Origin"] = "*"
            json_response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            json_response.headers["Access-Control-Allow-Headers"] = "*"

            return json_response

        except Exception as e:
            print(f"❌ Debug analysis error: {e}")
            import traceback

            traceback.print_exc()

            error_response = JSONResponse(
                content={"error": f"Debug analysis failed: {str(e)}"}
            )
            error_response.headers["Access-Control-Allow-Origin"] = "*"
            error_response.headers["Access-Control-Allow-Methods"] = (
                "GET, POST, OPTIONS"
            )
            error_response.headers["Access-Control-Allow-Headers"] = "*"

            return error_response

    @modal.fastapi_endpoint(method=["GET", "OPTIONS"], label="health")
    def health_endpoint(self):
        """Modal health check endpoint with CORS support."""
        from fastapi.responses import JSONResponse

        # Handle CORS preflight
        if hasattr(self, "method") and self.method == "OPTIONS":
            response = JSONResponse(content={})
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "*"
            response.headers["Access-Control-Max-Age"] = "86400"
            return response

        response = {
            "status": "healthy",
            "base_model": BASE_MODEL_NAME,
            "lora_adapters": LORA_ADAPTERS_NAME,
            "gpu_available": str(self.device) if hasattr(self, "device") else "unknown",
        }

        json_response = JSONResponse(content=response)
        json_response.headers["Access-Control-Allow-Origin"] = "*"
        json_response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        json_response.headers["Access-Control-Allow-Headers"] = "*"

        return json_response


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


def generate_fallback_recommendations(vulnerability_types: Dict[str, int]) -> List[str]:
    """Generate fallback security recommendations when LLM fails."""
    recommendations = []

    if "SQL Injection" in vulnerability_types or "SQL_INJECTION" in vulnerability_types:
        recommendations.append("Implement parameterized queries and input validation")

    if "Cross-Site Scripting" in vulnerability_types or "XSS" in vulnerability_types:
        recommendations.append("Sanitize and escape all user inputs")

    if "Authentication" in vulnerability_types or "AUTH" in vulnerability_types:
        recommendations.append("Strengthen authentication and session management")

    if "Cryptographic" in vulnerability_types or "CRYPTO" in vulnerability_types:
        recommendations.append("Use strong encryption and secure key management")

    if not recommendations:
        recommendations.append("Continue following secure coding practices")

    return recommendations


@app.function(
    image=image,
    volumes={MODEL_CACHE_PATH: model_volume},
    gpu=GPU_CONFIG,
    secrets=[modal.Secret.from_name("github-token")],
    timeout=1800,
)
def test_repository_analysis():
    """Test repository analysis using SecurityAnalyzer class methods directly."""
    print("🧪 Testing repository analysis on Modal servers...")

    # Test repository - use a smaller repo for faster testing
    test_repo = "https://github.com/AdamSkog/Hadoop-DocuSearch"

    print(f"🎯 Testing analysis on: {test_repo}")

    # Fetch repository
    repo_data = fetch_repository_contents_optimized.remote(test_repo)
    print(f"📊 Fetched {len(repo_data['file_data'])} files")

    return {
        "test_status": "SUCCESS",
        "files_found": len(repo_data["file_data"]),
        "fetch_time": repo_data.get("fetch_time", 0),
    }

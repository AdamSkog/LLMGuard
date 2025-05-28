# LLMGuard Backend - Modal + vLLM Implementation

This backend implements a serverless security analysis system using Modal Labs for deployment and vLLM for high-performance inference with a fine-tuned Qwen3-4B model.

## Architecture

- **Modal Labs**: Serverless GPU compute platform
- **vLLM**: High-performance LLM inference engine
- **Qwen3-4B**: Base model with security-focused LoRA adapters
- **GitHub API**: Repository content fetching without cloning
- **FastAPI**: REST API endpoints

## Prerequisites

### 1. Modal Setup

```bash
# Install Modal
pip install modal

# Authenticate with Modal
modal setup
```

### 2. Environment Variables

Create a `.env` file in the backend directory:

```bash
# GitHub API (optional for public repos)
GITHUB_TOKEN=your_github_token_here

# Hugging Face Token (for model access)
HF_TOKEN=your_hf_token_here
```

### 3. Modal Secrets

You need to set up Modal secrets for secure access to external APIs:

```bash
# Create GitHub token secret
modal secret create github-token GITHUB_TOKEN=your_github_token_here
```

## Quick Start

### 1. Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt
# or if using uv:
uv sync
```

### 2. Deploy to Modal

```bash
# Use the deployment script
python deploy.py

# Or deploy manually
modal deploy modal_backend.py
```

### 3. Test the Deployment

```bash
# Run the test function
modal run modal_backend.py::test_analysis

# Or use the deployment script
python deploy.py  # Choose option 3
```

### 4. Serve the API

```bash
# Serve locally for development
modal serve modal_backend.py::fastapi_app

# Or use the deployment script
python deploy.py  # Choose option 2
```

## API Endpoints

### POST /analyze

Analyze a GitHub repository for security vulnerabilities.

**Request:**
```json
{
  "repository_url": "https://github.com/user/repo"
}
```

**Response:**
```json
{
  "repository_url": "https://github.com/user/repo",
  "total_files_scanned": 25,
  "files_with_issues": 8,
  "total_issues": 15,
  "file_analyses": [
    {
      "file_path": "src/auth.py",
      "language": "python",
      "issues": [
        {
          "file_path": "src/auth.py",
          "line_number": 42,
          "severity": "HIGH",
          "vulnerability_type": "SQL Injection",
          "description": "Direct string concatenation in SQL query",
          "recommendation": "Use parameterized queries",
          "code_snippet": "query = f\"SELECT * FROM users WHERE id = {user_id}\""
        }
      ],
      "analysis_status": "SUCCESS"
    }
  ],
  "repository_structure": {
    "src/": {"type": "directory"},
    "src/auth.py": {"type": "file", "language": "python", "size": 1024}
  },
  "analysis_summary": {
    "severity_breakdown": {"HIGH": 3, "MEDIUM": 8, "LOW": 4},
    "vulnerability_types": {"SQL Injection": 2, "XSS": 3},
    "risk_score": 75,
    "recommendations": ["Implement parameterized queries"]
  }
}
```

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model": "qwen3-security-dpo-4b"
}
```

## Supported Programming Languages

The model supports security analysis for:

- **Python** (.py)
- **JavaScript/TypeScript** (.js, .jsx, .ts, .tsx)
- **Java** (.java)
- **C++** (.cpp, .cc, .cxx, .hpp, .h)
- **C#** (.cs)
- **PHP** (.php)
- **Ruby** (.rb)
- **Swift** (.swift)
- **Go** (.go)
- **Kotlin** (.kt, .kts)
- **Fortran** (.f, .f90, .f95)

## Security Vulnerability Types

The system detects:

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

## Configuration

### GPU Settings

The system is optimized for T4 GPUs with the following settings:

```python
GPU_CONFIG = modal.gpu.T4()

# vLLM optimizations
gpu_memory_utilization=0.85
max_model_len=2048
max_num_batched_tokens=2048
max_num_seqs=4
enforce_eager=True
swap_space=2
```

### Model Configuration

```python
# Base model (no Unsloth dependency required for inference)
model="unsloth/Qwen3-4B-unsloth-bnb-4bit"

# LoRA adapter
lora_path="AdamDS/qwen3-security-dpo-4b"
```

## Development

### Local Testing

```bash
# Test with a specific repository
modal run modal_backend.py::test_analysis

# Serve API locally
modal serve modal_backend.py::fastapi_app
```

### Debugging

```bash
# Check Modal logs
modal logs llmguard-security-analyzer

# View app status
modal app list
```

### Performance Monitoring

The system includes built-in metrics for:
- Analysis duration
- Files processed
- Issues found
- Model performance

## Deployment Options

### Development
```bash
modal serve modal_backend.py::fastapi_app
```

### Production
```bash
modal deploy modal_backend.py
```

### Custom Domain (Optional)
```bash
# Configure custom domain in Modal dashboard
# Update CORS settings in fastapi_app() function
```

## Troubleshooting

### Common Issues

1. **Modal Authentication Error**
   ```bash
   modal setup
   ```

2. **Missing Secrets**
   ```bash
   modal secret create github-token GITHUB_TOKEN=your_token
   ```

3. **GPU Memory Issues**
   - Reduce `max_model_len` or `gpu_memory_utilization`
   - Use smaller batch sizes

4. **Model Loading Errors**
   - Verify HF_TOKEN has access to the model
   - Check model path: `AdamDS/qwen3-security-dpo-4b`

### Performance Optimization

1. **For Large Repositories**
   - Files are automatically chunked if they exceed context window
   - Batch processing is used for multiple files

2. **For High Throughput**
   - Increase `max_num_seqs` for more concurrent requests
   - Use A10G or A100 GPUs for better performance

## Cost Optimization

- **T4 GPU**: ~$0.60/hour (sufficient for most workloads)
- **A10G GPU**: ~$1.10/hour (better for high throughput)
- **Container idle timeout**: 300 seconds (reduces costs)
- **Auto-scaling**: Modal automatically scales based on demand

## Security Considerations

- GitHub tokens are stored as Modal secrets
- API endpoints include CORS configuration
- No sensitive data is logged or stored
- All processing happens in isolated containers

## Integration with Frontend

The API is designed to work seamlessly with the Next.js frontend:

1. **Repository Structure**: Provides file tree for UI display
2. **File-specific Issues**: Enables highlighting problematic files
3. **Severity Levels**: Supports color-coded issue display
4. **Risk Scoring**: Provides overall repository security score

## Support

For issues or questions:
1. Check Modal logs: `modal logs llmguard-security-analyzer`
2. Verify environment setup with `python deploy.py`
3. Test with known vulnerable repositories (e.g., OWASP WebGoat)

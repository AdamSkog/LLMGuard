# 🚀 LLMGuard Backend - Terminal Setup

Simple manual setup for deploying the LLMGuard security analyzer backend.

## 📋 Prerequisites

1. **Hugging Face Token** (Required)
   - Go to: https://huggingface.co/settings/tokens
   - Create token with "Read" permission

2. **GitHub Token** (Optional but recommended)
   - Go to: https://github.com/settings/personal-access-tokens
   - Create token with "Contents" read permission

3. **Modal Account** 
   - Sign up at: https://modal.com

## 🔧 Setup Steps

### 1. Create Environment File
```bash
cat > .env << 'EOF'
GITHUB_TOKEN=your_github_token_here
HF_TOKEN=your_hf_token_here
MODEL_NAME=AdamDS/qwen3-security-dpo-4b
BASE_MODEL=unsloth/Qwen3-4B-unsloth-bnb-4bit
EOF
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Setup Modal
```bash
# Authenticate with Modal
modal setup

# Create secrets (replace with your actual token)
modal secret create github-token GITHUB_TOKEN=your_actual_github_token
```

### 4. Test Setup
```bash
python test_local.py
```

### 5. Deploy to Modal
```bash
modal deploy modal_backend.py
```

### 6. Serve the API
```bash
modal serve modal_backend.py
```

## 📡 API Usage

After deployment, your API will be available at:
`https://your-app--llmguard-security-analyzer-api.modal.run`

### Analyze a Repository
```bash
curl -X POST "https://your-url/analyze" \
  -H "Content-Type: application/json" \
  -d '{"repository_url": "https://github.com/owner/repo"}'
```

### Health Check
```bash
curl "https://your-url/health"
```

## 🎯 Supported Languages

- Python (.py)
- JavaScript (.js)
- TypeScript (.ts, .tsx)
- Java (.java)
- C++ (.cpp, .cc, .cxx)
- C# (.cs)
- PHP (.php)
- Ruby (.rb)
- Swift (.swift)
- Go (.go)
- Kotlin (.kt)

## 🔍 Vulnerability Types Detected

1. SQL Injection
2. Cross-Site Scripting (XSS)
3. Cross-Site Request Forgery (CSRF)
4. Insecure Direct Object References
5. Security Misconfiguration
6. Sensitive Data Exposure
7. Missing Function Level Access Control
8. Unvalidated Redirects and Forwards
9. Using Components with Known Vulnerabilities
10. Insufficient Logging and Monitoring

## 💡 Tips

- **Free Tier**: Modal provides free credits to get started
- **Cost Optimization**: Uses T4 GPU with optimized settings
- **Rate Limits**: GitHub API has rate limits (5000/hour authenticated, 60/hour anonymous)
- **Large Repos**: Files are chunked automatically if they exceed context window

## 🐛 Troubleshooting

### Modal Authentication Issues
```bash
modal setup --force
```

### Secrets Not Found
```bash
modal secret list
modal secret create github-token GITHUB_TOKEN=your_token
```

### Model Download Issues
- Ensure HF_TOKEN has correct permissions
- Check model exists: https://huggingface.co/AdamDS/qwen3-security-dpo-4b

### API Timeout
- Large repositories may take 30+ seconds
- Consider analyzing specific files/folders for faster results

## 📞 Quick Commands Reference

```bash
# Deploy and serve in one go
modal serve modal_backend.py

# Deploy without serving
modal deploy modal_backend.py

# Check logs
modal logs llmguard-security-analyzer

# Stop all apps
modal app stop llmguard-security-analyzer
``` 
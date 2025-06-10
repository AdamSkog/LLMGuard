[![LinkedIn][linkedin-shield]][linkedin-url]
[![MIT License][license-shield]][license-url]

<br />
<div align="center">
  <a href="https://github.com/AdamSkog/LLMGuard">
    <img src="frontend/public/logo.png" alt="Logo" width="80" height="80">
  </a>

<h2 align="center">LLMGuard</h2>

  <p align="center">
    AI-Powered Code Security Analysis Platform using Fine-tuned Large Language Models
    <br />
    <a href=""><strong>Explore the Research Paper » (TBD)</strong></a>
    <br />
    <br />
    <a href="https://llmguard-frontend.vercel.app">View Demo</a>
    ·
    <a href="https://github.com/AdamSkog/LLMGuard/issues">Report Bug</a>
    ·
    <a href="https://github.com/AdamSkog/LLMGuard/pulls">Request Feature</a>
  </p>
</div>

---

## Table of Contents
- [Table of Contents](#table-of-contents)
- [About the Project](#about-the-project)
  - [Built With](#built-with)
  - [Key Features](#key-features)
- [Architecture](#architecture)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Repository Analysis](#repository-analysis)
  - [API Endpoints](#api-endpoints)
- [Model Fine-tuning](#model-fine-tuning)
  - [Dataset](#dataset)
  - [DPO Training](#dpo-training)
- [Deployment](#deployment)
  - [Backend (Modal Labs)](#backend-modal-labs)
  - [Frontend (Vercel)](#frontend-vercel)
- [Supported Languages](#supported-languages)
- [Security Vulnerabilities Detected](#security-vulnerabilities-detected)
- [Research & Notebooks](#research--notebooks)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## About the Project

LLMGuard is a cutting-edge security analysis platform that leverages state-of-the-art Large Language Models (LLMs) to automatically detect security vulnerabilities in code repositories. The system combines advanced dataset curation, model fine-tuning using Direct Preference Optimization (DPO), and scalable cloud inference to provide comprehensive security analysis across multiple programming languages.

The platform features a **Qwen3-4B model fine-tuned specifically for code security analysis** using the `CyberNative/Code_Vulnerability_Security_DPO` dataset, achieving superior performance in identifying real-world security vulnerabilities compared to general-purpose models.

### Built With

[![Python][python-shield]][python-url]
[![Next.js][nextjs-shield]][nextjs-url]
[![FastAPI][fastapi-shield]][fastapi-url]
[![Modal][modal-shield]][modal-url]
[![PyTorch][pytorch-shield]][pytorch-url]
[![Transformers][transformers-shield]][transformers-url]
[![Vercel][vercel-shield]][vercel-url]
[![TypeScript][typescript-shield]][typescript-url]

### Key Features

- 🤖 **AI-Powered Analysis**: Fine-tuned Qwen3-4B model specialized for security vulnerability detection
- 🚀 **Serverless Architecture**: Modal Labs backend with automatic scaling and GPU optimization
- 🌐 **Modern Web Interface**: Next.js frontend with real-time analysis and interactive results
- 📊 **Comprehensive Reporting**: Detailed vulnerability reports with severity levels and recommendations
- 🔄 **Multi-Language Support**: Analysis across 12+ programming languages
- 📈 **Research-Grade**: Complete model training pipeline with EDA and DPO fine-tuning
- ⚡ **High Performance**: Optimized inference with memory-efficient techniques
- 🔍 **GitHub Integration**: Direct repository analysis without cloning

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│                 │    │                 │    │                 │
│   Next.js       │    │   Modal Labs    │    │   Qwen3-4B      │
│   Frontend      │──▶│   Backend       │───▶│   Fine-tuned    │
│                 │    │   (FastAPI)     │    │   Model         │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│                 │    │                 │    │                 │
│   Vercel        │    │   GitHub API    │    │   HuggingFace   │
│   Deployment    │    │   Integration   │    │   Model Hub     │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Getting Started

### Prerequisites

- **Python 3.11+**
- **Node.js 18+**
- **Modal Labs Account** (for backend deployment)
- **Vercel Account** (for frontend deployment, optional)
- **GitHub Token** (for repository access)
- **HuggingFace Token** (for model access)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AdamSkog/LLMGuard.git
   cd LLMGuard
   ```

2. **Backend Setup:**
   ```bash
   cd backend
   pip install -r requirements.txt
   
   # Setup Modal
   modal setup
   
   # Create environment file
   cp .env.example .env
   # Edit .env with your tokens
   
   # Create Modal secrets
   modal secret create github-token GITHUB_TOKEN=your_github_token
   ```

3. **Frontend Setup:**
   ```bash
   cd frontend
   npm install
   
   # Create environment file
   cp .env.example .env.local
   # Edit .env.local with your API URLs
   ```

4. **Notebooks Setup (for research/training):**
   ```bash
   cd notebooks
   pip install -r requirements.txt
   # or
   uv sync
   ```

## Usage

### Repository Analysis

1. **Deploy the backend:**
   ```bash
   cd backend
   modal deploy modal_backend.py
   ```

2. **Start the frontend:**
   ```bash
   cd frontend
   npm run dev
   ```

3. **Analyze a repository:**
   - Visit the web interface
   - Enter a GitHub repository URL
   - View comprehensive security analysis results

### API Endpoints

**Analyze Repository:**
```bash
POST /analyze-repository
{
  "repository_url": "https://github.com/user/repo"
}
```

**Analyze Single File:**
```bash
POST /analyze-file
{
  "file_path": "src/auth.py",
  "file_content": "...",
  "language": "python"
}
```

**Health Check:**
```bash
GET /health
```

## Model Fine-tuning

### Dataset

The model is trained on the `CyberNative/Code_Vulnerability_Security_DPO` dataset:
- **4,656 code examples** across multiple programming languages
- **Preference pairs** of secure vs. vulnerable code implementations
- **Comprehensive vulnerability types** including SQL injection, XSS, buffer overflows, etc.

### DPO Training

Our implementation uses **Direct Preference Optimization (DPO)** with:
- **Base Model**: `unsloth/Qwen3-4B-unsloth-bnb-4bit`
- **Training Framework**: Unsloth for 2-5x speedup
- **Memory Optimization**: 4-bit quantization, LoRA adapters
- **Hardware**: Optimized for T4/A10G GPUs
- **Monitoring**: Weights & Biases integration

Key training configurations:
- **LoRA Rank**: 32 (higher rank for security tasks)
- **Learning Rate**: 5e-6 (stable for DPO)
- **Batch Size**: 1 with gradient accumulation
- **Context Length**: 1024 tokens (optimized for code analysis)

## Deployment

### Backend (Modal Labs)

The backend leverages Modal Labs for serverless GPU compute:

```bash
# Deploy to production
modal deploy modal_backend.py

# Serve for development
modal serve modal_backend.py
```

**Features:**
- Automatic GPU scaling
- Model caching with Modal Volumes
- Concurrent repository processing
- CORS-enabled API endpoints

### Frontend (Vercel) (TO BE DEPLOYED AS OF NOW)

The frontend is deployed on Vercel for global edge distribution:

```bash
# Deploy to Vercel
vercel --prod
```

**Features:**
- Server-side rendering with Next.js
- Responsive design for all devices
- Real-time analysis progress
- Interactive vulnerability reports

## Supported Languages

The model provides security analysis for:

- **Python** (.py)
- **JavaScript/TypeScript** (.js, .jsx, .ts, .tsx)
- **Java** (.java)
- **C/C++** (.c, .cpp, .cc, .cxx, .h, .hpp)
- **C#** (.cs)
- **PHP** (.php)
- **Ruby** (.rb)
- **Swift** (.swift)
- **Go** (.go)
- **Kotlin** (.kt, .kts)
- **Fortran** (.f, .f90, .f95)

## Security Vulnerabilities Detected

The system identifies 10+ vulnerability types:

1. **SQL Injection** - Unsafe database query construction
2. **Cross-Site Scripting (XSS)** - Unescaped user input in web applications
3. **Authentication/Authorization Flaws** - Weak access controls and session management
4. **Input Validation Issues** - Missing or insufficient input sanitization
5. **Cryptographic Weaknesses** - Poor encryption practices and key management
6. **Path Traversal** - Unsafe file system operations
7. **Command Injection** - Unsafe execution of system commands
8. **Buffer Overflows** - Memory safety issues in low-level languages
9. **Race Conditions** - Concurrency-related security issues
10. **Information Disclosure** - Accidental exposure of sensitive data

## Research & Notebooks

The `notebooks/` directory contains comprehensive research materials:

- **`eda.ipynb`**: Exploratory Data Analysis of the DPO dataset
- **`finetune-qwen3-dpo.ipynb`**: Complete model fine-tuning pipeline
- **Dataset Analysis**: Statistical insights and vulnerability distribution
- **Model Training**: DPO implementation with Unsloth optimizations

Key findings from the research:
- Secure code implementations are ~38% longer than vulnerable versions
- Buffer overflows are most common in C/C++ codebases
- The fine-tuned model shows significant improvement in security-specific tasks

## Contributing

Contributions make the open-source community an amazing place to learn, inspire, and create. Any contributions are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Areas for Contribution

- 🔍 **New Vulnerability Types**: Expand detection capabilities
- 🌐 **Language Support**: Add support for more programming languages
- 🎨 **Frontend Features**: Enhance the user interface and experience
- 📊 **Analytics**: Improve reporting and visualization
- 🚀 **Performance**: Optimize inference speed and accuracy

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

**Adam Skoglund** - [@AdamSkog](https://linkedin.com/in/adam-skoglund) - adamskoglund2022@gmail.com

**Project Link**: [https://github.com/AdamSkog/LLMGuard](https://github.com/AdamSkog/LLMGuard)

**Research Paper**: TBD

---

<div align="center">
  <p>⭐ Star this repo if you find it helpful!</p>
  <p>Built with ❤️ by Adam Skoglund</p>
</div>

<!-- MARKDOWN LINKS & IMAGES -->
[license-shield]: https://img.shields.io/badge/MIT-red?style=for-the-badge&label=LICENSE
[license-url]: https://github.com/AdamSkog/LLMGuard/blob/main/LICENSE

[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/adam-skoglund

[python-shield]: https://img.shields.io/badge/Python-%233776AB?style=for-the-badge&logo=Python&labelColor=black
[python-url]: https://python.org

[nextjs-shield]: https://img.shields.io/badge/Next.js-%23000000?style=for-the-badge&logo=next.js&labelColor=black
[nextjs-url]: https://nextjs.org

[fastapi-shield]: https://img.shields.io/badge/FastAPI-%23009688?style=for-the-badge&logo=fastapi&labelColor=black
[fastapi-url]: https://fastapi.tiangolo.com

[modal-shield]: https://img.shields.io/badge/Modal-%23FF6B6B?style=for-the-badge&logo=modal&labelColor=black
[modal-url]: https://modal.com

[pytorch-shield]: https://img.shields.io/badge/PyTorch-%23EE4C2C?style=for-the-badge&logo=PyTorch&labelColor=black
[pytorch-url]: https://pytorch.org

[transformers-shield]: https://img.shields.io/badge/🤗%20Transformers-%23FF6F00?style=for-the-badge&labelColor=black
[transformers-url]: https://huggingface.co/transformers

[vercel-shield]: https://img.shields.io/badge/Vercel-%23000000?style=for-the-badge&logo=vercel&labelColor=black
[vercel-url]: https://vercel.com

[typescript-shield]: https://img.shields.io/badge/TypeScript-%233178C6?style=for-the-badge&logo=typescript&labelColor=black
[typescript-url]: https://www.typescriptlang.org 
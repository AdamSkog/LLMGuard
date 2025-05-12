# Notebooks Directory

This directory contains Jupyter notebooks and related code for experimentation and analysis. It is set up to use [uv](https://github.com/astral-sh/uv) for fast Python dependency management and reproducible environments.

## Getting Started

### 1. Install `uv`

If you don't have `uv` installed, you can install it with:

```bash
pip install uv
```

Or, for the latest version (recommended):

```bash
pip install --upgrade pip
pip install uv
```

You can also use [other installation methods](https://github.com/astral-sh/uv#installation) (e.g., Homebrew, prebuilt binaries).

### 2. Set Up the Virtual Environment

From this `notebooks/` directory, run:

```bash
uv venv
```

This will create a `.venv` folder with an isolated Python environment.

### 3. Install Dependencies

To install all dependencies specified in `pyproject.toml` (including dev dependencies for Jupyter):

```bash
uv sync --all
```

- Use `--all` to include all dependency groups (such as `[dev]` for Jupyter and ipykernel).

### 4. Activate the Virtual Environment

- **On Windows (Git Bash/WSL) OR macOS/Linux:**

  ```bash
  source .venv/Scripts/activate
  ```

### 5. Make the Virtual Environment Available in Jupyter

Register the environment as a Jupyter kernel so you can select it in your notebooks:

```bash
python -m ipykernel install --user --name=notebooks-venv --display-name="Python (notebooks/.venv)"
```

- This will make a kernel called "Python (notebooks/.venv)" available in Jupyter.
- When opening a notebook, select this kernel to ensure you are using the correct environment.

### 6. Launch Jupyter

You can now start Jupyter Lab or Notebook:

```bash
jupyter lab
# or
jupyter notebook
```

## Notes

- All dependencies are managed via `pyproject.toml` and `uv.lock`.
- If you add new packages, use `uv add <package>` (optionally with `--group dev` for development-only dependencies).
- For reproducibility, always use `uv sync` to install dependencies.
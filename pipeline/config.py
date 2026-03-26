"""Pipeline constants and configuration."""

import os
from dotenv import load_dotenv

load_dotenv()

# Supabase
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]

# GitHub
GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]
GITHUB_API_BASE = "https://api.github.com"
GITHUB_RATE_LIMIT_BUFFER = 50  # sleep when remaining requests drop below this

# Pipeline thresholds
MIN_REPO_SIZE_KB = 200
MIN_CODE_TOKENS = 64_000
MAX_FILE_SIZE_BYTES = 1_000_000  # 1MB
MAX_CONCURRENT_CLONES = 10
SUPABASE_PAGE_SIZE = 1000

# Clone settings
CLONE_BASE_DIR = "/tmp/lcr_clones"

# Source code extensions to count tokens for
SOURCE_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".cpp", ".cc", ".cxx", ".c", ".h", ".hpp",
    ".java", ".rs", ".go", ".rb", ".php", ".swift", ".kt", ".kts", ".scala",
    ".cs", ".m", ".mm", ".r", ".R", ".jl", ".lua", ".sh", ".bash", ".zsh",
    ".pl", ".pm", ".ex", ".exs", ".erl", ".hrl", ".hs", ".ml", ".mli",
    ".f", ".f90", ".f95", ".v", ".sv", ".vhd", ".vhdl",
    ".sql", ".proto", ".thrift", ".graphql",
    ".html", ".css", ".scss", ".sass", ".less",
    ".yaml", ".yml", ".toml", ".json", ".xml", ".ini", ".cfg",
    ".md", ".rst", ".txt",
    ".cmake", ".make", ".mk",
    ".dockerfile",
}

# Extensions to always skip (binary/data files)
SKIP_EXTENSIONS = {
    ".pt", ".pth", ".h5", ".hdf5", ".pkl", ".pickle", ".parquet", ".csv", ".tsv",
    ".npy", ".npz", ".safetensors", ".onnx", ".pb", ".tflite",
    ".bin", ".dat", ".db", ".sqlite", ".sqlite3",
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".svg", ".ico", ".webp",
    ".mp3", ".mp4", ".wav", ".avi", ".mov", ".flv",
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".rar", ".7z",
    ".whl", ".egg", ".so", ".dylib", ".dll", ".exe", ".o", ".a",
    ".pdf", ".doc", ".docx", ".pptx", ".xls", ".xlsx",
    ".lock",
}

# Directories to always skip
SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", "venv", ".venv", "env", ".env",
    ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "dist", "build", "egg-info", ".eggs",
    "vendor", "third_party", "3rdparty",
    ".idea", ".vscode", ".vs",
    "data", "datasets", "checkpoints", "weights", "models",
    "logs", "wandb", "mlruns",
}

# Dependency scoring weights
SCORE_WEIGHTS = {
    "density": 0.25,
    "normalized_avg_indeg": 0.25,
    "largest_component_fraction": 0.30,
    "edge_node_ratio": 0.20,
}

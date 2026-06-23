# my_toolkit
[![GitHub Repo stars](https://img.shields.io/github/stars/Enzohj/my_toolkit?style=social)](https://github.com/Enzohj/my_toolkit/stargazers)
[![GitHub last commit](https://img.shields.io/github/last-commit/Enzohj/my_toolkit)](https://github.com/Enzohj/my_toolkit/commits/main)
[![GitHub license](https://img.shields.io/github/license/Enzohj/my_toolkit)](https://github.com/Enzohj/my_toolkit/blob/main/LICENSE)

A simple and easy-to-use Python toolkit designed to streamline common operations in daily development.

[ English | [中文](README_zh.md) ]

---

## Table of Contents

- [✨ Features](#-features)
- [💾 Installation](#-installation)
- [🚀 Quickstart](#-quickstart)
  - [File Operations](#file-operations)
  - [Image Processing](#image-processing)
  - [Logging](#logging)
  - [Parallel Processing](#parallel-processing)
  - [Useful Decorators](#useful-decorators)
  - [Text Processing](#text-processing)
- [📜 Scripts Usage](#-scripts-usage)
- [🤔 FAQ](#-faq)
- [📄 License](#-license)

## ✨ Features

- **Unified File Interface**: Standardized read/write support for multiple formats, including `TXT`, `CSV`, `TSV`, `JSON`, `JSONL`, `Parquet`, and `Pickle`, without worrying about the underlying details.
- **Convenient Image Processing**: Effortlessly convert between `PIL.Image`, `Bytes`, and `Base64`, with support for loading images from local paths or URLs.
- **Practical Logging System**: Provides colored console logs, optional rotating file logs, global level switching, and safe logger reuse with the standard `logging` library.
- **Efficient Parallel Processing**: Simplifies multi-threading and multi-processing tasks through one ordered `apply_parallel` entry point, with an optional `tqdm` progress bar.
- **Practical Decorators**: Offers common decorators like `@timer`, `@timeout`, and `@retry` to enhance code robustness.
- **Lightweight Text Utilities**: Includes common text processing functions for cleaning text, extracting `#hashtags#`, and more.

## 💾 Installation

1.  **Clone the Repository**

    ```bash
    git clone https://github.com/Enzohj/my_toolkit.git
    cd my_toolkit
    ```

2.  **Install Dependencies**

    The basic dependencies are listed in `setup_env/requirements.txt`.

    ```bash
    pip install -r setup_env/requirements.txt
    ```

    Additionally, some features depend on the following third-party libraries. It is recommended to install them for the full experience:

    - `Pillow`: For image processing.
    - `requests`: For downloading images from URLs.
    - `tqdm`: For displaying progress bars in parallel computations.

    You can install all recommended dependencies with the following command:

    ```bash
    pip install pandas huggingface_hub pyarrow Pillow pillow-heif requests tqdm
    ```

## 🚀 Quickstart

### File Operations

`my_toolkit` provides two high-level functions, `read_file` and `write_file`, which automatically select the appropriate reader/writer based on the file extension.

```python
from my_toolkit.file import read_file, write_file

# Read a JSONL file
data_list = read_file('data.jsonl')

# Read a CSV file as a DataFrame
df = read_file('data.csv', format='dataframe')

# Write a dictionary to a JSON file
my_dict = {"name": "my_toolkit", "version": "1.0"}
write_file(my_dict, 'config.json', indent=4)

# Append lines to a TXT file
lines_to_append = ["hello", "world"]
write_file(lines_to_append, 'log.txt', append=True)

# Append is supported for TXT, CSV, TSV, and JSONL.
# Unsupported suffixes raise a clear ValueError.
```

### Image Processing

The `MyImage` class accepts local paths, URLs, raw bytes, Base64 strings, or existing `PIL.Image` objects. Module-level helpers are also available for common conversions.

```python
from my_toolkit.image import MyImage, img_to_base64, base64_to_img

# Load an image from a local path or URL
image = MyImage(path='path/to/your/image.jpg')
# image = MyImage(url='https://example.com/image.png')

# Get the PIL.Image object
pil_image = image.img

# Convert between image formats
img_base64 = img_to_base64(pil_image, fmt='png')

# Restore an image from a Base64 string
restored_pil_image = base64_to_img(img_base64)

# Convert format and save
image.convert('webp').save('converted.webp')

# Base64 data URLs are supported as input and output
data_url = image.base64_with_prefix
same_image = MyImage(data_url)
```

### Logging

Create reusable standard-library loggers with colored console output and optional rotating file output.

```python
from my_toolkit.logger import init_logger, set_level

log = init_logger("demo", level="INFO", save_to="logs/app.log")

log.debug("This is a debug message.")
log.info("Welcome to my_toolkit!")
log.warning("Please note, this operation may take a long time.")
log.error("File not found!")

# Switch all loggers created by init_logger
set_level("WARNING")

# LOG_LEVEL can also be set in the environment, for example LOG_LEVEL=DEBUG
```

### Parallel Processing

Easily execute ordered parallel tasks with `apply_parallel`. Use `method="thread"` for I/O-bound work and `method="process"` for CPU-bound work.

```python
from my_toolkit.mp import apply_parallel
import time

def task(item):
    time.sleep(0.1)
    return item * 2

data = range(20)

# Use multi-threading for I/O-bound tasks
results_thread = apply_parallel(data, task, method="thread", num_workers=4)

# Use multi-processing for CPU-bound tasks
results_process = apply_parallel(data, task, method="process", num_workers=4)

# error_policy controls task failures: "store" (default), "raise", or "ignore"
results = apply_parallel(data, task, error_policy="store")
```

### Useful Decorators

Simplify common functionalities with decorators.

```python
from my_toolkit.decorator import timer, retry, timeout

@retry(max_attempts=3, delay=1)
@timeout(seconds=5)
@timer
def risky_operation(should_fail):
    if should_fail:
        raise ValueError("Operation failed!")
    print("Operation successful!")
    return "OK"

# Example: The function will automatically retry and print the execution time
print("--- First call (will fail and retry) ---")
risky_operation(should_fail=True)

print("\n--- Second call (will succeed directly) ---")
risky_operation(should_fail=False)
```

`@retry` validates retry parameters up front. Set `raise_on_failure=True` to re-raise the last exception after all attempts are exhausted.

### Text Processing

Provides simple and fast text utility functions.

```python
from my_toolkit.text import normalize_text, extract_hashtag, remove_emoji_and_hashtag

text = "   Welcome to #my_toolkit  , this is a #Python library!   😊 "

# Normalize text (remove extra spaces)
normalized = normalize_text(text)
print(f"Normalized text: {normalized}")

# Extract hashtags
tags = extract_hashtag(text)
print(f"Extracted tags: {tags}")

# Remove emojis and hashtags
cleaned_text = remove_emoji_and_hashtag(text)
print(f"Cleaned text: {cleaned_text}")
```

## 📜 Scripts Usage

The `scripts` directory contains some useful scripts for daily development and management.

-   **`hang.sh`**: Runs a long-running command in the background and redirects its standard output and error to a log file.

    ```bash
    # Usage: ./scripts/hang.sh <your_command> [your_args...]
    # Example: Run a Python script in the background
    ./scripts/hang.sh python my_train_script.py --epochs 100
    ```
    Logs are saved by default to `./logs/hang_YYYYMMDD_HHMMSS.log`.

-   **`download_hf_ckpt.sh`**: Downloads a model or dataset from a Hugging Face mirror (`hf-mirror.com`).

    ```bash
    # Usage: ./scripts/download_hf_ckpt.sh <model_name> [save_directory]
    # Example: Download Llama-3-8B-Instruct to a specific directory
    ./scripts/download_hf_ckpt.sh meta-llama/Meta-Llama-3-8B-Instruct /path/to/models
    ```

-   **`kill.sh` & `cmd.sh`**: Used for process management.
    - `kill.sh`: Finds and kills processes based on a keyword, with an interactive confirmation prompt.
      ```bash
      # Usage: ./scripts/kill.sh <keyword>
      # Example: Find and kill all processes containing "python"
      ./scripts/kill.sh python
      ```
    - `cmd.sh`: Forcibly kills all processes using NVIDIA GPUs. Use with caution.
      ```bash
      # Usage: ./scripts/cmd.sh
      ```

## 🤔 FAQ

**Q: Why do I get a `ModuleNotFoundError` when importing `my_toolkit` from another directory?**

A: This is because the root directory of `my_toolkit` has not been added to Python's search path. You can solve this by adding the project's root directory to the `PYTHONPATH` environment variable.

Add the following command to your `~/.bashrc` or `~/.zshrc` file:

```bash
# Replace /path/to/your/my_toolkit with the actual path to your project
export PYTHONPATH=$PYTHONPATH:/path/to/your/my_toolkit
```

Then, run `source ~/.bashrc` or `source ~/.zshrc` to apply the changes.

## 📄 License

This repository is licensed under the [MIT License](LICENSE).

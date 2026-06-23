# my_toolkit

[![GitHub Repo stars](https://img.shields.io/github/stars/Enzohj/my_toolkit?style=social)](https://github.com/Enzohj/my_toolkit/stargazers)
[![GitHub last commit](https://img.shields.io/github/last-commit/Enzohj/my_toolkit)](https://github.com/Enzohj/my_toolkit/commits/main)
[![GitHub license](https://img.shields.io/github/license/Enzohj/my_toolkit)](https://github.com/Enzohj/my_toolkit/blob/main/LICENSE)

一个简单易用的 Python 工具包，旨在简化日常开发中的常用操作。

---

## 目录

- [✨ 特性亮点](#-特性亮点)
- [💾 安装指南](#-安装指南)
- [🚀 快速开始](#-快速开始)
  - [文件操作](#文件操作)
  - [图像处理](#图像处理)
  - [日志记录](#日志记录)
  - [并行计算](#并行计算)
  - [实用装饰器](#实用装饰器)
  - [文本处理](#文本处理)
- [📜 常用脚本说明](#-常用脚本说明)
- [🤔 常见问题](#-常见问题)
- [📄 许可](#-许可)

## ✨ 特性亮点

- **统一文件接口**: 支持 `TXT`, `CSV`, `TSV`, `JSON`, `JSONL`, `Parquet`, `Pickle` 等多种格式的标准化读写，无需关心底层细节。
- **便捷图像处理**: 轻松实现 `PIL.Image`, `Bytes`, `Base64` 之间的相互转换，支持从本地或 URL 加载图像。
- **实用日志系统**: 基于标准 `logging` 提供彩色控制台日志、可选滚动文件日志、全局等级切换和安全复用。
- **高效并行处理**: 通过统一的 `apply_parallel` 入口简化多线程和多进程任务，并保证结果顺序与输入一致。
- **实用装饰器**: 提供 `@timer` (计时), `@timeout` (超时), `@retry` (重试) 等常用装饰器，提升代码健壮性。
- **轻量文本工具**: 包含文本清洗、`#hashtags#` 提取等常用文本处理功能。

## 💾 安装指南

1.  **克隆仓库**

    ```bash
    git clone https://github.com/Enzohj/my_toolkit.git
    cd my_toolkit
    ```

2.  **安装依赖**

    基础依赖项已在 `requirements.txt` 中列出。

    ```bash
    pip install -r setup_env/requirements.txt
    ```

    此外，部分功能依赖于以下第三方库，建议一并安装以获得完整体验：

    - `Pillow`: 图像处理
    - `requests`: 从 URL 下载图像
    - `tqdm`: 在并行计算中显示进度条

    可以使用以下命令安装所有推荐依赖：

    ```bash
    pip install pandas huggingface_hub pyarrow Pillow pillow-heif requests tqdm
    ```

## 🚀 快速开始

### 文件操作

`my_toolkit` 提供了 `read_file` 和 `write_file` 两个高级函数，能够根据文件扩展名自动选择合适的读写方式。

```python
from my_toolkit.file import read_file, write_file

# 读取 JSONL 文件
data_list = read_file('data.jsonl')

# 读取 CSV 文件为 DataFrame
df = read_file('data.csv', format='dataframe')

# 写入 JSON 文件
my_dict = {"name": "my_toolkit", "version": "1.0"}
write_file(my_dict, 'config.json', indent=4)

# 以追加模式写入 TXT 文件
lines_to_append = ["hello", "world"]
write_file(lines_to_append, 'log.txt', append=True)

# 追加模式支持 TXT、CSV、TSV、JSONL。
# 其他后缀传入 append=True 会抛出明确的 ValueError。
```

### 图像处理

`MyImage` 类支持从本地路径、URL、原始 bytes、Base64 字符串或已有 `PIL.Image` 对象加载图像，也提供常用的模块级转换函数。

```python
from my_toolkit.image import MyImage, img_to_base64, base64_to_img

# 从本地路径或 URL 加载图像
image = MyImage(path='path/to/your/image.jpg')
# image = MyImage(url='https://example.com/image.png')

# 获取 PIL.Image 对象
pil_image = image.img

# 图像格式转换
img_base64 = img_to_base64(pil_image, fmt='png')

# 从 Base64 恢复图像
restored_pil_image = base64_to_img(img_base64)

# 转换格式并保存
image.convert('webp').save('converted.webp')

# 支持读取和输出 Base64 data URL
data_url = image.base64_with_prefix
same_image = MyImage(data_url)
```

### 日志记录

创建可复用的标准库 logger，支持彩色控制台输出和可选滚动文件输出。

```python
from my_toolkit.logger import init_logger, set_level

log = init_logger("demo", level="INFO", save_to="logs/app.log")

log.debug("这是一条调试信息。")
log.info("欢迎使用 my_toolkit！")
log.warning("请注意，这个操作可能耗时较长。")
log.error("文件未找到！")

# 切换所有通过 init_logger 创建的 logger
set_level("WARNING")

# 也可以通过环境变量设置日志等级，例如 LOG_LEVEL=DEBUG
```

### 并行计算

通过 `apply_parallel` 轻松执行有序并行任务。I/O 密集型任务使用 `method="thread"`，CPU 密集型任务使用 `method="process"`。

```python
from my_toolkit.mp import apply_parallel
import time

def task(item):
    time.sleep(0.1)
    return item * 2

data = range(20)

# 使用多线程处理 I/O 密集型任务
results_thread = apply_parallel(data, task, method="thread", num_workers=4)

# 使用多进程处理 CPU 密集型任务
results_process = apply_parallel(data, task, method="process", num_workers=4)

# error_policy 控制任务失败策略："store"（默认）、"raise" 或 "ignore"
results = apply_parallel(data, task, error_policy="store")
```

### 实用装饰器

用装饰器简化常用功能。

```python
from my_toolkit.decorator import timer, retry, timeout

@retry(max_attempts=3, delay=1)
@timeout(seconds=5)
@timer
def risky_operation(should_fail):
    if should_fail:
        raise ValueError("操作失败！")
    print("操作成功！")
    return "OK"

# 示例：函数将自动重试，并在计时结束后打印耗时
print("--- 第一次调用 (会失败并重试) ---")
risky_operation(should_fail=True)

print("\n--- 第二次调用 (直接成功) ---")
risky_operation(should_fail=False)
```

`@retry` 会在创建装饰器时校验重试参数。设置 `raise_on_failure=True` 可在所有尝试失败后重新抛出最后一次异常。

### 文本处理

提供简单快捷的文本工具函数。

```python
from my_toolkit.text import normalize_text, extract_hashtag, remove_emoji_and_hashtag

text = "   欢迎来到 #my_toolkit  , 这是一个 #Python 库!   😊 "

# 标准化文本 (去除多余空格)
normalized = normalize_text(text)
print(f"标准化文本: {normalized}")

# 提取 hashtags
tags = extract_hashtag(text)
print(f"提取的标签: {tags}")

# 移除 emoji 和 hashtags
cleaned_text = remove_emoji_and_hashtag(text)
print(f"清洗后文本: {cleaned_text}")
```

## 📜 常用脚本说明

`scripts` 目录下提供了一些实用脚本，方便日常开发和管理。

-   **`hang.sh`**: 在后台挂起一个长时间运行的命令，并将标准输出和错误重定向到日志文件。

    ```bash
    # 用法: ./scripts/hang.sh <你的命令> [你的参数...]
    # 示例: 在后台运行 Python 脚本
    ./scripts/hang.sh python my_train_script.py --epochs 100
    ```
    日志会默认保存在 `./logs/hang_YYYYMMDD_HHMMSS.log`。

-   **`download_hf_ckpt.sh`**: 从 Hugging Face 镜像（`hf-mirror.com`）下载模型或数据集。

    ```bash
    # 用法: ./scripts/download_hf_ckpt.sh <模型名称> [保存目录]
    # 示例: 下载 Llama-3-8B-Instruct 到指定目录
    ./scripts/download_hf_ckpt.sh meta-llama/Meta-Llama-3-8B-Instruct /path/to/models
    ```

-   **`kill.sh` & `cmd.sh`**: 用于进程管理。
    - `kill.sh`: 根据关键词查找并杀死相关进程，支持交互式确认。
      ```bash
      # 用法: ./scripts/kill.sh <关键词>
      # 示例: 查找并杀死所有包含 "python" 的进程
      ./scripts/kill.sh python
      ```
    - `cmd.sh`: 强制杀死所有占用 NVIDIA GPU 的进程，请谨慎使用。
      ```bash
      # 用法: ./scripts/cmd.sh
      ```

## 🤔 常见问题

**Q: 为什么在其他目录导入 `my_toolkit` 时会提示 `ModuleNotFoundError`？**

A: 这是因为 `my_toolkit` 的根目录没有被添加到 Python 的搜索路径中。你可以通过将项目根目录添加到 `PYTHONPATH` 环境变量来解决这个问题。

将以下命令添加到你的 `~/.bashrc` 或 `~/.zshrc` 文件中：

```bash
# 将 /path/to/your/my_toolkit 替换为你的实际项目路径
export PYTHONPATH=$PYTHONPATH:/path/to/your/my_toolkit
```

然后执行 `source ~/.bashrc` 或 `source ~/.zshrc` 使其生效。

## 📄 许可

本仓库遵循 [MIT License](LICENSE) 许可。

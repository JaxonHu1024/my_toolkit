"""
file.py — 统一的文件读写工具模块

支持格式: TXT / CSV / TSV / JSON / JSONL / Parquet / Pickle
提供 read_file / write_file 两个统一入口，根据后缀自动分发。
"""

from __future__ import annotations

import csv
import inspect
import json
import os
import pickle
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from .logger import init_logger

logger = init_logger(name=__name__)

__all__ = [
    # TXT
    "read_txt", "write_txt",
    # CSV
    "read_csv", "write_csv",
    # JSON
    "read_json", "write_json",
    # JSONL
    "read_jsonl", "write_jsonl",
    # Parquet
    "read_parquet", "write_parquet",
    # Pickle
    "read_pickle", "write_pickle",
    # Dispatcher
    "read_file", "write_file",
]

PathLike = Union[str, Path]


# ========================
# 内部工具
# ========================

def _ensure_parent(file_path: Path) -> None:
    """若父目录不存在则自动创建。"""
    if file_path.parent != Path("."):
        file_path.parent.mkdir(parents=True, exist_ok=True)


def _to_path(file_path: PathLike) -> Path:
    """统一转换为 Path 对象。"""
    return Path(file_path)


def _filter_supported_kwargs(
    func: Any,
    kwargs: Dict[str, Any],
    *,
    allowed: Optional[set[str]] = None,
) -> Dict[str, Any]:
    """过滤出目标函数支持的关键字参数。

    Python 3.11 中许多 C 扩展/内建函数（例如 ``csv.reader``）没有
    ``__code__`` 属性，不能再依赖 ``func.__code__.co_varnames`` 做参数过滤。
    这里优先使用 ``inspect.signature``，对无法 introspect 的内建函数允许传入
    显式白名单，保证在 Python 3.11 下稳定运行。
    """
    if not kwargs:
        return {}

    if allowed is not None:
        return {k: v for k, v in kwargs.items() if k in allowed}

    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        # 无法获取签名时保守透传，让被调用方给出更准确的 TypeError。
        return dict(kwargs)

    parameters = signature.parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return dict(kwargs)

    supported = {
        name
        for name, param in parameters.items()
        if param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }
    return {k: v for k, v in kwargs.items() if k in supported}


def _file_is_empty_or_missing(file_path: Path) -> bool:
    """文件不存在或大小为 0 时返回 True。"""
    return not file_path.exists() or file_path.stat().st_size == 0


def _needs_append_newline(file_path: Path) -> bool:
    """追加写入前判断目标文件末尾是否缺少换行。"""
    if not file_path.exists() or file_path.stat().st_size == 0:
        return False
    with open(file_path, "rb") as rf:
        try:
            rf.seek(-1, os.SEEK_END)
            return rf.read(1) not in (b"\n", b"\r")
        except OSError:
            return False


_CSV_FMTPARAMS = {
    "dialect",
    "doublequote",
    "escapechar",
    "lineterminator",
    "quotechar",
    "quoting",
    "skipinitialspace",
    "strict",
}


# ========================
# TXT 文件读写
# ========================

def read_txt(
    file_path: PathLike,
    encoding: str = "utf-8",
    as_lines: bool = True,
) -> Union[str, List[str]]:
    """
    读取 TXT 文件内容。

    参数:
        file_path: 文件路径。
        encoding: 文件编码，默认 utf-8。
        as_lines: 是否按行读取，默认 True。

    返回:
        若 as_lines=True 返回行列表，否则返回完整字符串。
    """
    file_path = _to_path(file_path)
    with open(file_path, "r", encoding=encoding) as f:
        if as_lines:
            content = [line.rstrip("\r\n") for line in f]
            logger.info(f"Read {len(content)} lines from '{file_path}'")
        else:
            content = f.read()
            logger.info(f"Read {len(content)} characters from '{file_path}'")
    return content


def write_txt(
    content: Union[str, List[str]],
    file_path: PathLike,
    encoding: str = "utf-8",
    append: bool = False,
) -> None:
    """
    写入 TXT 文件内容。

    参数:
        content: 要写入的内容（字符串或字符串列表）。
        file_path: 文件路径。
        encoding: 文件编码，默认 utf-8。
        append: 是否追加写入，默认 False。
    """
    file_path = _to_path(file_path)
    _ensure_parent(file_path)
    mode = "a" if append else "w"

    # 在打开目标文件前探测末尾换行，避免在同一文件句柄期间再次 open
    # 对 str 和 List[str] 两种分支都适用
    prefix_newline = ""
    if append and _needs_append_newline(file_path):
        prefix_newline = "\n"

    with open(file_path, mode, encoding=encoding) as f:
        if isinstance(content, list):
            if prefix_newline:
                f.write(prefix_newline)
            f.writelines(f"{i}\n" for i in content)
            logger.info(
                f"Write {len(content)} lines to '{file_path}' "
                f"in {'append' if append else 'write'} mode"
            )
        else:
            payload = prefix_newline + content
            f.write(payload)
            logger.info(
                f"Write {len(payload)} characters to '{file_path}' "
                f"in {'append' if append else 'write'} mode"
            )


# ========================
# CSV 文件读写
# ========================

def read_csv(
    file_path: PathLike,
    encoding: str = "utf-8",
    sep: str = ",",
    format: Literal["dataframe", "list"] = "dataframe",
    skip_header: bool = True,
    replace_na: bool = True,
    **kwargs: Any,
) -> Union[pd.DataFrame, List[List[str]]]:
    """
    读取 CSV 文件。

    参数:
        file_path: 文件路径。
        encoding: 文件编码,默认 utf-8。
        sep: 分隔符,默认 ','。
        format: 读取格式,'dataframe' 或 'list',默认 'dataframe'。
        skip_header: 仅在 format='list' 时生效,是否跳过首行表头,默认 True。
        replace_na: dataframe 模式: 将 NaN 替换为 None;
        **kwargs: dataframe 模式下透传给 pandas.read_csv;
                  list 模式下透传给 csv.reader。
                  注意两者参数集不同,请按模式传递对应参数。

    返回:
        pd.DataFrame 或嵌套列表(List[List[Optional[str]]])。

    异常:
        FileNotFoundError: 文件不存在。
        ValueError: format 参数非法。
    """
    path = _to_path(file_path)

    if format == "dataframe":
        # 校验 kwargs 是否包含 pandas.read_csv 不支持的参数
        supported_kwargs = _filter_supported_kwargs(pd.read_csv, kwargs)
        df = pd.read_csv(path, sep=sep, encoding=encoding, **supported_kwargs)
        if replace_na:
            # 转为 object dtype 后再替换,避免 pandas 2.x 的 FutureWarning
            df = df.astype(object).where(df.notna(), None)
        logger.info(
            "Read CSV '%s' as DataFrame. Shape: %s, header: %s",
            path, df.shape, df.columns.tolist(),
        )
        return df

    if format == "list":
        with path.open("r", encoding=encoding, newline="") as f:
            # csv.reader 是 C 内建函数，Python 3.11 下没有 __code__ 属性；
            # 使用显式 fmtparams 白名单过滤。
            supported_kwargs = _filter_supported_kwargs(
                csv.reader,
                kwargs,
                allowed=_CSV_FMTPARAMS,
            )
            reader = csv.reader(f, delimiter=sep, **supported_kwargs)
            if skip_header:
                header = next(reader, None)
                if header is not None:
                    logger.info("Skipped CSV header: %s", header)
            data = [row for row in reader]
        logger.info("Read CSV '%s' as list. Rows: %d", path, len(data))
        return data

    raise ValueError(f"Unsupported format: {format!r}. Choose 'dataframe' or 'list'.")


def write_csv(
    data: Union[pd.DataFrame, Dict[str, Any], List[List[Any]]],
    file_path: PathLike,
    encoding: str = "utf-8",
    append: bool = False,
    sep: str = ",",
    header: Optional[List[str]] = None,
    **kwargs: Any,
) -> None:
    """
    写入 CSV 文件。

    参数:
        data: 要写入的数据（DataFrame / dict / 二维列表）。
        file_path: 文件路径。
        encoding: 文件编码，默认 utf-8。
        append: 是否追加写入，默认 False。
        sep: 分隔符，默认 ','。
        header: format='list' 时的列名列表，默认 None。
        **kwargs: 传递给 pandas.to_csv 或 csv.writer 的额外参数。
    """
    file_path = _to_path(file_path)
    _ensure_parent(file_path)
    mode = "a" if append else "w"

    if isinstance(data, dict):
        data = pd.DataFrame(data)

    if isinstance(data, pd.DataFrame):
        # 追加模式下不重复写入 header
        write_header = not append or _file_is_empty_or_missing(file_path)
        data.to_csv(
            file_path, index=False, sep=sep, mode=mode,
            encoding=encoding, header=write_header, **kwargs,
        )
        logger.info(
            f"Write DataFrame to '{file_path}' "
            f"in {'append' if append else 'write'} mode. Shape: {data.shape}"
        )
    elif isinstance(data, list):
        # 拒绝一维 list（csv.writerows 会把字符串逐字符拆成列）
        if data and not isinstance(data[0], (list, tuple)):
            raise TypeError(
                "write_csv 期望二维可迭代（List[List] 或 List[Tuple]），"
                f"但收到一维序列，首元素类型为 {type(data[0]).__name__}。"
            )
        # 只要提供了 header，且为非追加模式或文件为空/不存在，就写入 header
        write_header = bool(header) and (
            not append or _file_is_empty_or_missing(file_path)
        )
        with open(file_path, mode, newline="", encoding=encoding) as f:
            supported_kwargs = _filter_supported_kwargs(
                csv.writer,
                kwargs,
                allowed=_CSV_FMTPARAMS,
            )
            writer = csv.writer(f, delimiter=sep, **supported_kwargs)
            if write_header:
                logger.info(f"CSV header: {header}")
                writer.writerow(header)
            writer.writerows(data)
            logger.info(
                f"Write {len(data)} rows to '{file_path}' "
                f"in {'append' if append else 'write'} mode"
            )
    else:
        raise TypeError(
            f"Unsupported data type: {type(data).__name__}. "
            "Expected pd.DataFrame, dict, or list."
        )


# ========================
# JSON 文件读写
# ========================

def read_json(file_path: PathLike, encoding: str = "utf-8") -> Any:
    """
    读取 JSON 文件。

    参数:
        file_path: 文件路径。
        encoding: 文件编码，默认 utf-8。

    返回:
        反序列化后的 Python 对象（通常为 dict 或 list）。
    """
    file_path = _to_path(file_path)
    with open(file_path, "r", encoding=encoding) as f:
        data = json.load(f)
    if isinstance(data, dict):
        desc = f"dict with {len(data)} keys"
    elif isinstance(data, (list, tuple, set)):
        desc = f"{type(data).__name__} with {len(data)} items"
    else:
        desc = f"scalar ({type(data).__name__})"
    logger.info(f"Read JSON '{file_path}' successfully. Top-level: {desc}")
    return data


def write_json(
    data: Any,
    file_path: PathLike,
    encoding: str = "utf-8",
    ensure_ascii: bool = False,
    indent: int = 4,
) -> None:
    """
    写入 JSON 文件。

    参数:
        data: 可序列化为 JSON 的 Python 对象。
        file_path: 文件路径。
        encoding: 文件编码，默认 utf-8。
        ensure_ascii: 是否确保 ASCII 编码，默认 False。
        indent: 缩进空格数，默认 4。
    """
    file_path = _to_path(file_path)
    _ensure_parent(file_path)
    with open(file_path, "w", encoding=encoding) as f:
        json.dump(data, f, ensure_ascii=ensure_ascii, indent=indent)
    logger.info(f"Write JSON data to '{file_path}'")


# ========================
# JSONL 文件读写
# ========================

def read_jsonl(file_path: PathLike, encoding: str = "utf-8") -> List[Any]:
    """
    读取 JSONL 文件（每行一个 JSON 对象）。

    参数:
        file_path: 文件路径。
        encoding: 文件编码，默认 utf-8。

    返回:
        每行解析后的对象列表。
    """
    file_path = _to_path(file_path)
    data: List[Any] = []
    with open(file_path, "r", encoding=encoding) as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"JSONL parse error at line {lineno} of '{file_path}': {e}"
                ) from e
    logger.info(f"Read {len(data)} JSON objects from '{file_path}'")
    return data


def write_jsonl(
    data: List[Any],
    file_path: PathLike,
    encoding: str = "utf-8",
    append: bool = False,
) -> None:
    """
    写入 JSONL 文件（每行一个 JSON 对象）。

    参数:
        data: 要写入的对象列表。
        file_path: 文件路径。
        encoding: 文件编码，默认 utf-8。
        append: 是否追加写入，默认 False。
    """
    file_path = _to_path(file_path)
    _ensure_parent(file_path)
    mode = "a" if append else "w"
    prefix_newline = "\n" if append and data and _needs_append_newline(file_path) else ""
    with open(file_path, mode, encoding=encoding) as f:
        if prefix_newline:
            f.write(prefix_newline)
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info(f"Write {len(data)} JSON objects to '{file_path}'")


# ========================
# Parquet 文件读写
# ========================

def read_parquet(
    file_root: PathLike,
    engine: str = "auto",
    ignore: Optional[List[str]] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    读取 Parquet 文件或目录。

    参数:
        file_root: 文件路径或包含多个 Parquet 分片的目录路径。
        engine: Parquet 引擎，默认 'auto'。
        ignore: 目录模式下需要忽略的文件名或后缀，默认 ['_SUCCESS']。
                匹配规则为精确文件名匹配或以 '.' 开头的后缀匹配，
                不再使用子串匹配，避免误杀正常文件。
        verbose: 是否额外打印 DataFrame.info()，默认 False（避免大表日志刷屏）。

    返回:
        合并后的 DataFrame；读取失败时返回空 DataFrame。
    """
    if ignore is None:
        ignore = ["_SUCCESS"]

    file_root = _to_path(file_root)

    # ---------- 单文件 ----------
    if file_root.is_file():
        # 单文件读取失败直接抛出异常，避免静默返回空 DataFrame 导致调用方误用；
        # 仅目录逐分片模式才做兜底。
        data = pd.read_parquet(file_root, engine=engine)
        logger.info(f"Read Parquet file '{file_root}'. Shape: {data.shape}")
        if verbose:
            _log_dataframe_info(data)
        return data

    # ---------- 目录 ----------
    if file_root.is_dir():
        # 优先尝试引擎原生目录读取
        try:
            data = pd.read_parquet(file_root, engine=engine)
            logger.info(f"Read Parquet dir '{file_root}'. Shape: {data.shape}")
            if verbose:
                _log_dataframe_info(data)
            return data
        except Exception as e:
            logger.error(
                f"Error: {e}, falling back to per-file reading."
            )

        # 精确匹配忽略规则：完整文件名 或 以 '.' 开头的后缀
        ignore_names = {kw for kw in ignore if not kw.startswith(".")}
        ignore_suffixes = {kw.lower() for kw in ignore if kw.startswith(".")}

        def _should_ignore(p: Path) -> bool:
            return p.name in ignore_names or p.suffix.lower() in ignore_suffixes

        # 递归收集所有文件，按路径排序保证顺序稳定
        part_paths = sorted(
            p for p in file_root.rglob("*") if p.is_file() and not _should_ignore(p)
        )

        # 逐文件读取并拼接
        chunks: List[pd.DataFrame] = []
        iterator = part_paths
        if tqdm is not None:
            iterator = tqdm(
                part_paths,
                desc="Reading Parquet files",
                leave=False,
            )

        for part_path in iterator:
            try:
                chunk = pd.read_parquet(part_path, engine=engine)
                logger.debug(
                    f"Read '{part_path.relative_to(file_root)}'. Shape: {chunk.shape}"
                )
                chunks.append(chunk)
            except Exception as e:
                logger.error(f"Error reading '{part_path}': {e}")

        if chunks:
            data = pd.concat(chunks, ignore_index=True)
            logger.info(
                f"Concatenated {len(chunks)} Parquet files. Shape: {data.shape}"
            )
            if verbose:
                _log_dataframe_info(data)
            return data

        logger.warning(f"No valid Parquet data found in '{file_root}'")
        return pd.DataFrame()

    raise FileNotFoundError(f"Path does not exist: {file_root}")


def write_parquet(df: pd.DataFrame, file_path: PathLike, **kwargs: Any) -> None:
    """
    写入 Parquet 文件。

    参数:
        df: 要写入的 DataFrame。
        file_path: 文件路径。
        **kwargs: 传递给 DataFrame.to_parquet 的额外参数。
    """
    file_path = _to_path(file_path)
    _ensure_parent(file_path)
    df.to_parquet(file_path, **kwargs)
    logger.info(f"Write Parquet '{file_path}'. Shape: {df.shape}")


def _log_dataframe_info(df: pd.DataFrame) -> None:
    """将 DataFrame.info() 输出写入日志。"""
    buf = StringIO()
    df.info(buf=buf)
    logger.info(f"\n{buf.getvalue()}")


# ========================
# Pickle 文件读写
# ========================

def read_pickle(file_path: PathLike, **kwargs: Any) -> Any:
    """
    读取 Pickle 文件。

    参数:
        file_path: 文件路径。
        **kwargs: 传递给 pickle.load 的额外参数。

    返回:
        反序列化后的 Python 对象。
    """
    file_path = _to_path(file_path)
    with open(file_path, "rb") as f:
        data = pickle.load(f, **kwargs)
    logger.info(f"Read Pickle '{file_path}'")
    return data


def write_pickle(obj: Any, file_path: PathLike, **kwargs: Any) -> None:
    """
    写入 Pickle 文件。

    参数:
        obj: 要序列化的 Python 对象。
        file_path: 文件路径。
        **kwargs: 传递给 pickle.dump 的额外参数。
    """
    file_path = _to_path(file_path)
    _ensure_parent(file_path)
    with open(file_path, "wb") as f:
        pickle.dump(obj, f, **kwargs)
    logger.info(f"Write Pickle '{file_path}'")


# ========================
# 统一入口 (Dispatcher)
# ========================

def _read_tsv(p: PathLike, **kw: Any) -> Any:
    user_sep = kw.pop("sep", None)
    if user_sep is not None and user_sep != "\t":
        raise ValueError(
            f"TSV 文件的 sep 必须为 '\\t'，收到: {user_sep!r}。"
            "如需自定义分隔符请直接调用 read_csv。"
        )
    return read_csv(p, sep="\t", **kw)


def _write_tsv(d: Any, p: PathLike, **kw: Any) -> None:
    user_sep = kw.pop("sep", None)
    if user_sep is not None and user_sep != "\t":
        raise ValueError(
            f"TSV 文件的 sep 必须为 '\\t'，收到: {user_sep!r}。"
            "如需自定义分隔符请直接调用 write_csv。"
        )
    write_csv(d, p, sep="\t", **kw)


_READ_DISPATCH = {
    ".json": read_json,
    ".jsonl": read_jsonl,
    ".parquet": read_parquet,
    ".csv": read_csv,
    ".tsv": _read_tsv,
    ".txt": read_txt,
    ".pickle": read_pickle,
    ".pkl": read_pickle,
}

_WRITE_DISPATCH = {
    ".json": write_json,
    ".jsonl": write_jsonl,
    ".parquet": write_parquet,
    ".csv": write_csv,
    ".tsv": _write_tsv,
    ".txt": write_txt,
    ".pickle": write_pickle,
    ".pkl": write_pickle,
}

# 支持 append 写入的文件类型；其他类型若传入 append=True 将被拦截报错
_APPEND_SUPPORTED_SUFFIXES = {".txt", ".csv", ".tsv", ".jsonl"}


def read_file(file_path: PathLike, **kwargs: Any) -> Any:
    """
    根据文件后缀自动选择读取函数。

    支持: .json / .jsonl / .parquet / .csv / .tsv / .txt / .pickle / .pkl
    """
    path = _to_path(file_path)
    suffix = path.suffix.lower()
    reader = _READ_DISPATCH.get(suffix)
    if reader is None:
        raise ValueError(
            f"Unsupported file format: {suffix!r}. "
            f"Supported: {sorted(_READ_DISPATCH.keys())}"
        )
    return reader(path, **kwargs)


def write_file(data: Any, file_path: PathLike, **kwargs: Any) -> None:
    """
    根据文件后缀自动选择写入函数。

    支持: .json / .jsonl / .parquet / .csv / .tsv / .txt / .pickle / .pkl
    """
    path = _to_path(file_path)
    suffix = path.suffix.lower()
    writer = _WRITE_DISPATCH.get(suffix)
    if writer is None:
        raise ValueError(
            f"Unsupported file format: {suffix!r}. "
            f"Supported: {sorted(_WRITE_DISPATCH.keys())}"
        )
    if kwargs.get("append") and suffix not in _APPEND_SUPPORTED_SUFFIXES:
        raise ValueError(
            f"File {str(path)!r} (format {suffix!r}) does not support append mode. "
            f"Append is only supported for: {sorted(_APPEND_SUPPORTED_SUFFIXES)}"
        )
    writer(data, path, **kwargs)

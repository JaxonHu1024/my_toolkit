"""
mp.py - 并行处理工具模块

multi_thread
    受 GIL（全局解释器锁）限制。
    在 Python 中，同一时间只有一个线程能执行 Python 字节码。
    因此，不适合 CPU 密集型任务（如大量计算）。
    适合 I/O 密集型任务（如网络请求、文件读写、数据库操作）。

multi_process
    每个进程有独立的 Python 解释器和内存空间，因此不受 GIL 影响。
    可以真正实现并行计算。
    适合 CPU 密集型任务（如图像处理、数学计算、数据压缩等）。

提供 apply_parallel 函数，支持多线程/多进程并行处理，并保证结果顺序与输入一致。
"""

from __future__ import annotations

import os
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
)
from concurrent.futures import (
    ThreadPoolExecutor,
    ProcessPoolExecutor,
    Future,
    as_completed,
)

from .logger import init_logger

logger = init_logger(name=__name__)

# 延迟导入 pandas，避免不需要时浪费内存
_pd = None


def _get_pd():
    global _pd
    if _pd is None:
        try:
            import pandas as pd
            _pd = pd
        except ImportError:
            _pd = False  # 标记为不可用，避免反复导入
    return _pd if _pd is not False else None


try:
    from tqdm.auto import tqdm  # auto 可自动适配 notebook / terminal
except ImportError:
    tqdm = None

# ---------------------------------------------------------------------------
# 常量与默认配置
# ---------------------------------------------------------------------------
_VALID_METHODS = ("thread", "process")

_EXECUTOR_MAP = {
    "thread": ThreadPoolExecutor,
    "process": ProcessPoolExecutor,
}

NUM_WORKERS: int = int(
    os.environ.get("NUM_WORKERS", min(os.cpu_count() or 1, 8))
)

# 分批提交的默认批次大小，防止一次性创建过多 Future 导致 OOM
_DEFAULT_BATCH_SIZE = 5000


# ---------------------------------------------------------------------------
# 内部辅助
# ---------------------------------------------------------------------------
def _call_func(func: Callable, element: Any) -> Any:
    """根据 element 的类型选择调用方式：

    - dict           → func(**element)
    - tuple / list   → func(*element)
    - other          → func(element)
    """
    if isinstance(element, dict):
        return func(**element)
    if isinstance(element, (tuple, list)):
        return func(*element)
    return func(element)


def _resolve_iterable(iterable: Any) -> tuple[Sequence[Any], int]:
    """将各种可迭代类型统一为可切片序列，并在遇到 DataFrame 时转为 records。

    Returns
    -------
    (list, length)
    """
    pd = _get_pd()
    if pd is not None and isinstance(iterable, pd.DataFrame):
        items = iterable.to_dict(orient="records")
        return items, len(items)

    if isinstance(iterable, (list, tuple, range)):
        return iterable, len(iterable)

    # 有 __len__ / __iter__ / __getitem__ 的类序列对象（如 ndarray 等）
    if (
        hasattr(iterable, "__len__")
        and hasattr(iterable, "__iter__")
        and hasattr(iterable, "__getitem__")
    ):
        return iterable, len(iterable)

    # 生成器 / 纯迭代器 / set 等非稳定可切片对象 — 必须物化
    logger.warning(
        f"iterable type is {type(iterable).__name__}, materializing to list..."
    )
    try:
        items = list(iterable)
    except Exception as e:
        logger.error(f"Failed to materialize iterable: {e}")
        raise
    return items, len(items)


def _chunked(seq: Sequence[Any], size: int) -> Iterator[tuple[Sequence[Any], int]]:
    """将序列按 size 分批 yield，避免一次性生成所有 Future。"""
    for start in range(0, len(seq), size):
        yield seq[start: start + size], start


# ---------------------------------------------------------------------------
# 核心函数
# ---------------------------------------------------------------------------
def apply_parallel(
    iterable: Iterable,
    func: Callable,
    method: Literal["thread", "process"] = "thread",
    num_workers: int = NUM_WORKERS,
    show_progress: bool = True,
    total_num: Optional[int] = None,
    error_policy: Literal["store", "raise", "ignore"] = "store",
    progress_desc: Optional[str] = None,
    batch_size: Optional[int] = None,
) -> List[Any]:
    """对 *iterable* 中的每个元素并行调用 *func*，返回与输入顺序严格一致的结果列表。

    Parameters
    ----------
    iterable : Iterable
        任意可迭代对象（列表、元组、生成器、``pandas.DataFrame`` 等）。
        当传入 DataFrame 时，自动按行转为 ``dict`` 列表。
    func : callable
        对每个元素执行的函数。根据元素类型自动选择解包方式。
    method : ``"thread"`` | ``"process"``, default ``"thread"``
        并行方式。传入其他值将抛出 ``ValueError``。
    num_workers : int, default ``NUM_WORKERS``
    show_progress : bool, default ``True``
    total_num : int | None, default ``None``
    error_policy : ``"store"`` | ``"raise"`` | ``"ignore"``, default ``"store"``
        任务异常处理策略：
        - ``"store"``  — 将异常对象存入结果列表对应位置（默认，向后兼容）。
        - ``"raise"``  — 遇到第一个异常立即取消剩余任务并抛出。
        - ``"ignore"`` — 记录日志，结果位置填 ``None``。
    progress_desc : str | None, default ``None``
        自定义进度条描述文字。为 ``None`` 时使用默认格式。
    batch_size : int | None, default ``None``
        分批提交的批次大小。为 ``None`` 时，当任务数 > 10000 自动启用
        (默认 5000)，否则一次性提交。设为 0 或负数表示禁用分批。

    Returns
    -------
    list
        结果列表，第 *i* 个元素对应 ``iterable`` 中第 *i* 个输入。

    Raises
    ------
    ValueError
        当 ``method`` 不是 ``"thread"`` 或 ``"process"`` 时。
    RuntimeError
        当 ``error_policy="raise"`` 且有任务抛出异常时（封装原始异常）。

    Examples
    --------
    >>> from mp import apply_parallel
    >>> results = apply_parallel(range(10), lambda x: x ** 2, method="thread")
    >>> assert results == [i ** 2 for i in range(10)]
    """

    # ---- 1. 参数校验 -----------------------------------------------------
    if method not in _VALID_METHODS:
        raise ValueError(
            f"method 参数仅支持 {_VALID_METHODS!r}，收到: {method!r}"
        )
    if error_policy not in ("store", "raise", "ignore"):
        raise ValueError(
            f"error_policy 参数仅支持 'store' / 'raise' / 'ignore'，收到: {error_policy!r}"
        )
    if not isinstance(num_workers, int):
        raise TypeError(f"num_workers 必须为 int，收到: {type(num_workers).__name__}")
    if num_workers < 1:
        raise ValueError(f"num_workers 必须 >= 1，收到: {num_workers!r}")
    if total_num is not None and total_num < 0:
        raise ValueError(f"total_num 必须 >= 0，收到: {total_num!r}")
    if batch_size is not None and not isinstance(batch_size, int):
        raise TypeError(f"batch_size 必须为 int 或 None，收到: {type(batch_size).__name__}")

    # ---- 2. 物化可迭代对象 -----------------------------------------------
    items, inferred_total = _resolve_iterable(iterable)
    # total_num 仅用于进度条显示，真实任务数以 items 长度为准
    actual_total = inferred_total
    display_total = total_num if total_num is not None else inferred_total

    # 边界: 空任务直接返回
    if actual_total == 0:
        return []

    # 裁剪 num_workers 到合理范围
    num_workers = max(1, min(num_workers, actual_total))

    # ---- 3. 决定是否分批提交 ---------------------------------------------
    if batch_size is None:
        # 超过 10000 条自动启用分批，防止 Future 过多占满内存
        effective_batch = _DEFAULT_BATCH_SIZE if actual_total > 10000 else actual_total
    elif batch_size <= 0:
        effective_batch = actual_total  # 禁用分批
    else:
        effective_batch = batch_size

    # ---- 4. 选择执行器 ---------------------------------------------------
    executor_cls = _EXECUTOR_MAP[method]

    # ---- 5. 进度条准备 ---------------------------------------------------
    use_tqdm = show_progress and tqdm is not None
    if show_progress and tqdm is None:
        logger.warning(
            "show_progress=True 但 tqdm 未安装，将跳过进度条显示。"
            "可通过 `pip install tqdm` 安装。"
        )

    logger.info(
        f"apply_parallel 启动 | method={method}, workers={num_workers}, "
        f"total={actual_total}, batch={effective_batch}, error_policy={error_policy}",
    )

    # ---- 6. 提交与收集 ---------------------------------------------------
    results: list = [None] * actual_total
    error_count = 0
    completed_count = 0
    should_abort = False  # raise 策略下的中止标志

    pbar = None
    if use_tqdm:
        pbar = tqdm(
            total=display_total,
            desc=progress_desc,
            dynamic_ncols=True,
        )

    try:
        executor = executor_cls(max_workers=num_workers)
        try:
            for chunk, chunk_start in _chunked(items, effective_batch):
                if should_abort:
                    break

                # 提交当前批次
                future_to_idx: Dict[Future, int] = {}
                for local_idx, elem in enumerate(chunk):
                    global_idx = chunk_start + local_idx
                    fut = executor.submit(_call_func, func, elem)
                    future_to_idx[fut] = global_idx

                # 收集当前批次结果
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        results[idx] = future.result()
                    except Exception as exc:
                        error_count += 1

                        if error_policy == "raise":
                            should_abort = True
                            # 取消当前批次尚未开始的任务
                            for f in future_to_idx:
                                f.cancel()
                            raise RuntimeError(
                                f"任务 #{idx} 执行失败: {exc}"
                            ) from exc
                        else:
                            logger.error(f"任务 #{idx} 执行失败: {exc}")
                            if error_policy == "store":
                                results[idx] = exc
                            # "ignore" → results[idx] 保持 None

                    completed_count += 1
                    if pbar is not None:
                        pbar.update(1)

                # 批次间释放 Future 引用
                del future_to_idx
        finally:
            # raise 策略下立即取消尚未开始的任务，避免等待长任务
            if should_abort:
                try:
                    executor.shutdown(wait=False, cancel_futures=True)
                except TypeError:
                    # Python < 3.9 不支持 cancel_futures
                    executor.shutdown(wait=False)
            else:
                executor.shutdown(wait=True)

    finally:
        if pbar is not None:
            pbar.close()

    # ---- 7. 日志汇总 -----------------------------------------------------
    if error_count:
        logger.warning(f"共有 {error_count} / {actual_total} 个任务执行失败")
    else:
        logger.info(f"全部 {actual_total} 个任务执行完成")

    return results

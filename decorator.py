import asyncio
import inspect
import random
import time
from functools import wraps
from typing import Any, Callable, Optional, Tuple, Type, TypeVar, Union
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

try:
    from typing import ParamSpec
except ImportError:
    from typing_extensions import ParamSpec

from .logger import init_logger
logger = init_logger(name=__name__)

P = ParamSpec("P")
R = TypeVar("R")

__all__ = ["timer", "timeout", "retry"]

_UNSET = object()  # 哨兵值，区分 "返回 None" 与 "未设置"


# ────────────────────────────── timer ──────────────────────────────

class timer:
    """
    记录函数执行耗时（秒），无论成功或异常均会输出。

    用法:
        # 作为装饰器
        @timer
        def foo(): ...

        @timer
        async def bar(): ...

        # 作为上下文管理器
        with timer("load_data"):
            heavy_io()
    """

    def __new__(cls, func_or_label: Union[Callable[P, R], str, None] = None):
        instance = super().__new__(cls)

        # @timer  —— 直接装饰（无括号）
        if callable(func_or_label):
            return instance._wrap(func_or_label)

        # timer("label") —— 上下文管理器
        instance._label = func_or_label or "block"
        return instance

    def __init__(self, func_or_label: Union[Callable, str, None] = None):
        # 当作为装饰器直接返回 wrapper 时，__init__ 不会被调用到 self 上
        pass

    # ── 装饰器路径 ──
    def _wrap(self, func: Callable[P, R]) -> Callable[P, R]:
        label = func.__name__

        if inspect.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                start = time.perf_counter()
                try:
                    return await func(*args, **kwargs)
                finally:
                    logger.info(f"Function '{label}' elapsed: {(time.perf_counter() - start):.4f} s")

            return async_wrapper  # type: ignore[return-value]

        @wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                logger.info(f"Function '{label}' elapsed: {(time.perf_counter() - start):.4f} s")

        return sync_wrapper  # type: ignore[return-value]

    # ── 上下文管理器路径 ──
    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc_info):
        logger.info(f"Block '{self._label}' elapsed: {(time.perf_counter() - self._start):.4f} s")
        return False

    @property
    def elapsed(self) -> float:
        """在上下文管理器内部调用，返回当前已过时间。"""
        return time.perf_counter() - self._start


# ────────────────────────────── timeout ──────────────────────────────

def timeout(seconds: float) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    限制函数执行时间。超时后抛出 TimeoutError。

    - 同步函数: 使用 daemon ThreadPoolExecutor，超时后线程不会阻止进程退出。
      注意线程本身不会被强制终止，仅在调用侧抛出异常。
    - 异步函数: 使用 asyncio.wait_for，超时后任务被取消。
    """
    if seconds <= 0:
        raise ValueError("timeout seconds must be positive, got %s" % seconds)

    def decorator(func: Callable[P, R]) -> Callable[P, R]:

        if inspect.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                try:
                    return await asyncio.wait_for(
                        func(*args, **kwargs), timeout=seconds
                    )
                except asyncio.TimeoutError:
                    raise TimeoutError(
                        "Function '%s' timed out after %ss" % (func.__name__, seconds)
                    ) from None

            return async_wrapper  # type: ignore[return-value]

        @wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            # 不使用 `with ThreadPoolExecutor(...)`，避免 __exit__ 中
            # shutdown(wait=True) 在超时后仍阻塞等待子线程结束。
            executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix=f"timeout-{func.__name__}",
            )
            try:
                future = executor.submit(func, *args, **kwargs)
                try:
                    return future.result(timeout=seconds)
                except FutureTimeoutError:
                    raise TimeoutError(
                        "Function '%s' timed out after %ss" % (func.__name__, seconds)
                    ) from None
            finally:
                # 不等待子线程：线程本身无法被强制终止，但主线程立即返回。
                executor.shutdown(wait=False)

        return sync_wrapper  # type: ignore[return-value]

    return decorator


# ────────────────────────────── retry ──────────────────────────────

def retry(
    max_attempts: int = 3,
    delay: float = 0.1,
    backoff: float = 1,
    jitter: float = 0,
    exceptions: Tuple[Type[BaseException], ...] = (Exception,),
    fail_return: Any = _UNSET,
    raise_on_failure: bool = False,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    重试装饰器：在指定异常发生时自动重试。支持同步和异步函数。

    参数:
        max_attempts:     最大尝试次数（>= 1）
        delay:            初始延迟时间（秒）
        backoff:          退避因子（1=固定间隔，>1=指数退避）
        jitter:           随机抖动上限（秒），实际睡眠 = sleep_time + random(0, jitter)，
                          用于防止多实例同步重试造成惊群效应
        exceptions:       需要捕获并重试的异常类型元组
        fail_return:      所有尝试失败后的默认返回值（仅 raise_on_failure=False 时生效）；
                          未设置且 raise_on_failure=False 时返回 None
        raise_on_failure:  True 时在最终失败后重新抛出最后一次异常
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1, got %s" % max_attempts)

    effective_fail_return = None if fail_return is _UNSET else fail_return

    def decorator(func: Callable[P, R]) -> Callable[P, R]:

        def _compute_sleep(attempt: int) -> float:
            base = delay * (backoff ** (attempt - 1))
            return (base + random.uniform(0, jitter)) if jitter > 0 else base

        def _log_retry(attempt: int, exc: BaseException, sleep_time: float):
            logger.debug(
                f"[retry] '{func.__name__}' attempt {attempt}/{max_attempts} failed: {exc}; "
                f"retrying in {sleep_time:.3f}s …",
            )

        def _log_exhausted(last_exc: BaseException):
            logger.error(
                f"[retry] '{func.__name__}' exhausted {max_attempts} attempts. Last error: {last_exc}",
            )
            # 使用 exc_info=True，自动记录当前异常栈，避免手写 traceback.format_exc
            logger.error("[retry] traceback:", exc_info=True)

        # ── 异步版本 ──
        if inspect.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                last_exception: Optional[BaseException] = None

                for attempt in range(1, max_attempts + 1):
                    try:
                        return await func(*args, **kwargs)
                    except exceptions as exc:
                        last_exception = exc

                        if attempt < max_attempts:
                            sleep_time = _compute_sleep(attempt)
                            _log_retry(attempt, exc, sleep_time)
                            await asyncio.sleep(sleep_time)
                        else:
                            _log_exhausted(last_exception)
                            logger.debug(f"[retry] call args={args}, kwargs={kwargs}")

                if raise_on_failure:
                    raise last_exception  # type: ignore[misc]
                return effective_fail_return  # type: ignore[return-value]

            return async_wrapper  # type: ignore[return-value]

        # ── 同步版本 ──
        @wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            last_exception: Optional[BaseException] = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exception = exc

                    if attempt < max_attempts:
                        sleep_time = _compute_sleep(attempt)
                        _log_retry(attempt, exc, sleep_time)
                        time.sleep(sleep_time)
                    else:
                        _log_exhausted(last_exception)
                        logger.debug(f"[retry] call args={args}, kwargs={kwargs}")

            if raise_on_failure:
                raise last_exception  # type: ignore[misc]
            return effective_fail_return  # type: ignore[return-value]

        return sync_wrapper  # type: ignore[return-value]

    return decorator
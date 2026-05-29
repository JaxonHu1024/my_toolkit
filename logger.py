"""
logger.py — 一个开箱即用、可复用的 Logger 模块
===============================================

Features
--------
1. 控制台输出按级别着色（仅时间与级别名着色，消息正常显示）
2. 日志同时写入文件（按大小切割，自动保留 N 份备份）
3. 全局 / 单实例都支持一行切换日志等级
4. 单例缓存：同名 logger 不会重复添加 handler（多次 import 安全）

Quick Start
-----------
>>> from logger import init_logger, set_level
>>> log = init_logger("demo")
>>> log.debug("debug msg")
>>> log.info("hello %s", "world")
>>> log.warning("be careful")
>>> log.error("something broken")
>>> log.critical("boom!")
>>> log = init_logger("demo", save_to=True)       # 同时写入 ./logs/yyyymmdd-hhmmss.log
>>> set_level("DEBUG")           # 全局切级
>>> log.setLevel("WARNING")      # 单实例切级
"""

from __future__ import annotations

import functools
import logging
import os
import re
import sys
import threading
import warnings
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Final, NamedTuple
from types import MappingProxyType

# ---------------------------------------------------------------------------
# ANSI 颜色
# ---------------------------------------------------------------------------

def _fg(code: int) -> str:
    """256 色前景色 ANSI 序列。"""
    return f"\033[38;5;{code}m"


def _bg(code: int) -> str:
    """256 色背景色 ANSI 序列。"""
    return f"\033[48;5;{code}m"


class _Ansi:
    """终端 ANSI 控制序列常量。"""

    RESET:  Final[str] = "\033[0m"
    BOLD:   Final[str] = "\033[1m"
    DIM:    Final[str] = "\033[2m"
    ITALIC: Final[str] = "\033[3m"

    # 前景色（亮色系,观感更舒服）
    GREY:    Final[str] = _fg(245)
    CYAN:    Final[str] = _fg(87)
    GREEN:   Final[str] = _fg(114)
    YELLOW:  Final[str] = _fg(221)
    RED:     Final[str] = _fg(203)
    MAGENTA: Final[str] = _fg(213)
    BLUE:    Final[str] = _fg(75)

    # 背景色
    BG_RED:  Final[str] = _bg(213)


class _LevelStyle(NamedTuple):
    color: str
    label: str


_LEVEL_STYLE: Final[MappingProxyType[int, _LevelStyle]] = MappingProxyType({
    logging.DEBUG:    _LevelStyle(_Ansi.CYAN,   "DEBUG"),
    logging.INFO:     _LevelStyle(_Ansi.GREEN,  "INFO"),
    logging.WARNING:  _LevelStyle(_Ansi.YELLOW, "WARNING"),
    logging.ERROR:    _LevelStyle(_Ansi.RED,    "ERROR"),
    logging.CRITICAL: _LevelStyle(_Ansi.BG_RED, "CRITICAL"),
})

def _supports_color(stream) -> bool:
    """判断当前输出流是否支持 ANSI 彩色。

    额外尊重 ``NO_COLOR`` / ``FORCE_COLOR`` 环境变量约定。
    """
    force = os.getenv("FORCE_COLOR")
    if force is not None and force.strip() not in ("", "0", "false", "False"):
        return True
    if os.getenv("NO_COLOR") is not None:
        return False
    return hasattr(stream, "isatty") and stream.isatty()


# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------

def _relative_path(pathname: str) -> str:
    """尝试将绝对路径转为相对于 cwd 的相对路径，失败则保持原样。

    使用 LRU 缓存避免高频日志场景下重复计算。注意：若运行期间 cwd 发生变化，
    缓存结果不会失效；通常日志场景下 cwd 不变，影响可忽略。
    """
    return _relative_path_cached(pathname)


@functools.lru_cache(maxsize=1024)
def _relative_path_cached(pathname: str) -> str:
    try:
        return os.path.relpath(pathname)
    except Exception:
        # Windows 跨盘符会抛 ValueError；cwd 不存在等情形下也兜底
        return pathname


class ColorFormatter(logging.Formatter):
    def __init__(self, *, use_color: bool = True) -> None:
        super().__init__()
        self.use_color = use_color

    # -- helpers ------------------------------------------------------------
    def _c(self, text: str, color: str) -> str:
        return f"{color}{text}{_Ansi.RESET}" if self.use_color else text

    # -- main ---------------------------------------------------------------
    def format(self, record: logging.LogRecord) -> str:
        color, level_name = _LEVEL_STYLE.get(record.levelno, (_Ansi.GREY, record.levelname))

        ts = self.formatTime(record, "%Y-%m-%d %H:%M:%S")
        location = f"{_relative_path(record.pathname)}:{record.lineno}"

        # 仅对时间和级别着色，消息正常显示
        ts_s  = self._c(ts, _Ansi.GREY)
        lvl_s = self._c(level_name.ljust(8), color + _Ansi.BOLD)
        sep   = self._c("│", _Ansi.GREY)
        record_name = self._c(record.name, _Ansi.GREY)
        msg_s = record.getMessage()

        line = f"{ts_s} {sep} {lvl_s} {sep} {record_name} {sep} {location} {sep} {msg_s}"

        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        if record.stack_info:
            line += "\n" + self.formatStack(record.stack_info)
        return line


class PlainFormatter(logging.Formatter):
    """文件输出用的无色 formatter。"""

    default_fmt = (
        "%(asctime)s │ %(levelname)-8s │ %(name)s │ "
        "%(_relpath)s:%(lineno)d │ %(message)s"
    )

    def __init__(self) -> None:
        super().__init__(self.default_fmt, datefmt="%Y-%m-%d %H:%M:%S")

    def format(self, record: logging.LogRecord) -> str:
        # 临时挂到 record 上以支持 %(_relpath)s，格式化后立刻清理，
        # 避免污染被多 handler 共享的 record。
        record._relpath = _relative_path(record.pathname)
        try:
            return super().format(record)
        finally:
            try:
                del record._relpath
            except AttributeError:
                pass


# ---------------------------------------------------------------------------
# 公共 API
# ---------------------------------------------------------------------------

_DEFAULT_LEVEL: int = logging.INFO
_MAX_BYTES: int = 10 * 1024 * 1024      # 10 MB
_BACKUP_COUNT: int = 5
_PROPAGATE: bool = False

# 内部 handler 标记，避免误清外部添加的 handler
_OWNED_HANDLER_ATTR = "_is_owned_by_init_logger"

# 已创建 logger 的缓存，保证同名 logger 复用；同时记录首次 save_to 配置
_LOGGER_CACHE: dict[str, logging.Logger] = {}
_LOGGER_SAVE_TO: dict[str, bool | str] = {}
_CACHE_LOCK = threading.RLock()

_INVALID_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_filename_component(name: str) -> str:
    """将 logger name 变为更安全的文件名片段（避免路径穿越/非法字符）。"""
    cleaned = _INVALID_FILENAME_CHARS.sub("_", name.strip())
    cleaned = cleaned.strip("._-")
    return (cleaned or "logger")[:80]


def _clear_and_close_handlers(logger: logging.Logger) -> None:
    """移除并关闭本模块添加的 handlers，避免文件句柄泄露。

    不会清理外部代码（例如 ``logging.basicConfig()``）添加的 handler。
    """
    for h in list(logger.handlers):
        if not getattr(h, _OWNED_HANDLER_ATTR, False):
            continue
        logger.removeHandler(h)
        try:
            h.close()
        except Exception:
            # close 失败不应影响 logger 的正常初始化
            pass


def _resolve_level(
    level: int | str | None,
    *,
    fallback: int | None = None,
) -> int | None:
    """解析等级：显式参数 > 环境变量 LOG_LEVEL > fallback。

    返回 ``None`` 表示不需要调整（例如三者都没提供）。
    """
    if level is not None:
        return _coerce_level(level)
    env_level = os.getenv("LOG_LEVEL")
    if env_level is not None and env_level.strip() != "":
        try:
            return _coerce_level(env_level)
        except ValueError:
            # 环境变量是非关键来源，非法时 warn 并回退，避免 init 整体失败
            warnings.warn(
                f"Invalid LOG_LEVEL env value: {env_level!r}, "
                f"falling back to default.",
                stacklevel=2,
            )
    if fallback is not None:
        return _coerce_level(fallback)
    return None


def init_logger(
    name: str = "DefaultLogger",
    *,
    level: int | str | None = None,
    save_to: bool | str | os.PathLike = False,
) -> logging.Logger:
    """
    获取一个配置好的 logger。

    Parameters
    ----------
    name : str
        logger 名字，同名会复用已有实例。
    level : int | str | None
        日志等级，可以是 ``logging.DEBUG`` 或 ``"DEBUG"``，大小写不敏感。
        解析优先级：显式参数 > 环境变量 ``LOG_LEVEL`` > 默认 INFO。
    save_to : bool | str | PathLike
        是否写入日志文件。

        - ``False``（默认）—— 不写文件。
        - ``True`` —— 自动写入 ``./logs/<name>_yyyymmdd_hhmmss_ffffff_<pid>.log``。
        - 非空路径（str/PathLike）—— 写入指定路径（不能是目录）。
    """
    # 规范化 save_to 的类型
    if isinstance(save_to, os.PathLike):
        save_to = os.fspath(save_to)

    with _CACHE_LOCK:
        if name in _LOGGER_CACHE:
            logger = _LOGGER_CACHE[name]

            # 若外部代码移除了 handlers（或被关闭），允许自动恢复配置
            if not logger.handlers:
                _LOGGER_CACHE.pop(name, None)
                _LOGGER_SAVE_TO.pop(name, None)
            else:
                prev = _LOGGER_SAVE_TO.get(name)
                if save_to != prev:
                    warnings.warn(
                        f"Logger {name!r} 已存在（首次 save_to={prev!r}），"
                        f"本次传入的 save_to={save_to!r} 被忽略。"
                        f"如需不同配置请使用不同的 name。",
                        stacklevel=2,
                    )
                # 仅在显式传入 level 时覆盖，避免默默改掉别处的设置
                if level is not None:
                    logger.setLevel(_coerce_level(level))
                return logger

        logger = logging.getLogger(name)
        _clear_and_close_handlers(logger)
        logger.propagate = _PROPAGATE

        eff_level = _resolve_level(level, fallback=_DEFAULT_LEVEL)
        logger.setLevel(eff_level)  # type: ignore[arg-type]

        # --- 控制台（始终启用） ------------------------------------------------
        sh = logging.StreamHandler(stream=sys.stdout)
        sh.setFormatter(ColorFormatter(use_color=_supports_color(sys.stdout)))
        setattr(sh, _OWNED_HANDLER_ATTR, True)
        logger.addHandler(sh)

        # --- 文件 ------------------------------------------------------------
        if save_to is not False:
            if save_to is True:
                # 自动写入 ./logs/<name>_yyyymmdd_hhmmss_ffffff_<pid>.log
                safe_name = _safe_filename_component(name)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                log_path = Path("logs") / f"{safe_name}_{ts}_{os.getpid()}.log"
            elif isinstance(save_to, str) and save_to.strip():
                # 以路径分隔符结尾的字符串显然意图指向目录，直接拒绝
                if save_to.endswith(("/", os.sep)):
                    raise IsADirectoryError(
                        f"save_to 需指向文件而非目录: {save_to!r}"
                    )
                log_path = Path(save_to)
            else:
                raise ValueError(
                    f"save_to 必须为 True 或非空文件路径，当前值: {save_to!r}"
                )
            logger.info("save log to: %s", log_path)
            if log_path.exists() and log_path.is_dir():
                raise IsADirectoryError(f"save_to 指向目录而非文件: {log_path}")

            log_path.parent.mkdir(parents=True, exist_ok=True)
            fh = RotatingFileHandler(
                log_path,
                maxBytes=_MAX_BYTES,
                backupCount=_BACKUP_COUNT,
                encoding="utf-8",
            )
            fh.setFormatter(PlainFormatter())
            setattr(fh, _OWNED_HANDLER_ATTR, True)
            logger.addHandler(fh)

        _LOGGER_CACHE[name] = logger
        _LOGGER_SAVE_TO[name] = save_to
        return logger


def set_level(level: int | str) -> None:
    """全局切换日志等级（作用于所有通过 init_logger 创建的实例）。"""
    lv = _coerce_level(level)
    with _CACHE_LOCK:
        for lg in _LOGGER_CACHE.values():
            lg.setLevel(lv)


def _coerce_level(level: int | str) -> int:
    if isinstance(level, bool):
        # bool 是 int 的子类，但语义上不应作为日志等级
        raise ValueError(f"Invalid log level: {level!r}")
    if isinstance(level, int):
        if level < 0 or level > logging.CRITICAL:
            raise ValueError(f"Invalid log level: {level!r}")
        return level
    if isinstance(level, str):
        s = level.strip()
        # 支持 "10" / "20" 这类数字字符串（常见于环境变量）
        if s.isdigit():
            n = int(s)
            if n < 0 or n > logging.CRITICAL:
                raise ValueError(f"Invalid log level: {level!r}")
            return n

        lv = logging.getLevelName(s.upper())
        if isinstance(lv, int):
            return lv
    raise ValueError(f"Invalid log level: {level!r}")

# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger = init_logger(level="INFO", save_to=False)
    logger.debug("这是一条 debug 日志，通常用于排查细节")
    logger.info("服务启动成功,监听端口 %d", 8080)
    logger.warning("磁盘使用率 %.1f%%,请注意", 87.3)
    logger.error("连接数据库失败: %s", "timeout")
    logger.critical("系统即将宕机!!!")

    try:
        1 / 0
    except ZeroDivisionError:
        logger.exception("捕获到异常")

    # 全局切到 WARNING,下面的 info 不再显示
    set_level("WARNING")
    logger.info("这条不会显示")
    logger.warning("这条会显示")

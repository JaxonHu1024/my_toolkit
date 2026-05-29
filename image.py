"""
image.py - 通用图像处理工具模块

提供多种图像格式之间的转换函数（URL/本地路径、bytes、base64、Pillow Image），
以及统一的 MyImage 类，可接收多种输入并自动识别格式。

主要功能:
    - 多格式互转: URL/本地路径 ↔ bytes ↔ base64 ↔ Pillow Image
    - 自动格式识别: 基于 magic bytes / 文件后缀 / data URL 前缀
    - MyImage 统一封装: 接收任意来源，提供一致的属性与方法接口

典型用法::

    # 从 URL 加载
    image = MyImage(url="https://example.com/photo.jpg")

    # 从本地文件加载
    image = MyImage(path="/tmp/photo.png")

    # 获取 base64
    b64_str = image.base64

    # 转换格式并保存
    image.convert("webp").save("/tmp/photo.webp")

    # 作为上下文管理器使用
    with MyImage(url="https://example.com/photo.jpg") as img:
        img.save("/tmp/photo.jpg")
"""

from __future__ import annotations

import base64
import io
import os
import re
from pathlib import Path
from typing import Optional, Union

import requests

from PIL import Image, ExifTags

from .logger import init_logger
logger = init_logger(name=__name__)

try:
    import pillow_heif
except ImportError:
    pillow_heif = None
    logger.warning(
        "pillow_heif 未安装，HEIF/HEIC 读写能力将不可用。"
        "如需支持请安装: pip install pillow_heif"
    )
else:
    pillow_heif.register_heif_opener()

# ---------------------------------------------------------------------------
# 常量与辅助
# ---------------------------------------------------------------------------

# 支持的图像格式集合（统一小写）
SUPPORTED_FORMATS: frozenset[str] = frozenset(
    {"png", "jpeg", "webp", "heif", "gif", "bmp", "tiff", "ico"}
)
_ALPHA_MODES = frozenset({"RGBA", "PA", "LA"})
_HILO_GREY_MODES = frozenset({"I", "F"})

# 格式别名映射（统一到标准名称）
_FORMAT_ALIASES: dict[str, str] = {
    "jpg": "jpeg",
    "jpe": "jpeg",
    "jfif": "jpeg",
    "tif": "tiff",
    "heic": "heif",
}

# Pillow Image.save() 所接受的 format 参数映射
_PILLOW_SAVE_FORMAT: dict[str, str] = {
    "jpeg": "JPEG",
    "jpg": "JPEG",
    "png": "PNG",
    "webp": "WEBP",
    "gif": "GIF",
    "bmp": "BMP",
    "tiff": "TIFF",
    "ico": "ICO",
    "heif": "HEIF",
}

_DEFAULT_FORMAT: str = "jpeg"

# 文件头 magic bytes 映射（按匹配优先级排列）
_MAGIC_BYTES: list[tuple[bytes, str]] = [
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"\xff\xd8\xff", "jpeg"),
    (b"GIF87a", "gif"),
    (b"GIF89a", "gif"),
    (b"BM", "bmp"),
    (b"II\x2a\x00", "tiff"),
    (b"MM\x00\x2a", "tiff"),
]

# HEIF/HEIC brand 标识（用于 ftyp box 检测）
_HEIF_BRANDS: frozenset[str] = frozenset(
    {"heic", "heix", "hevc", "hevx", "mif1", "msf1"}
)

# data-URL 正则：匹配 data:image/<fmt>;base64, 前缀
_DATA_URL_RE: re.Pattern[str] = re.compile(
    r"^\s*data:image/(?P<fmt>[a-zA-Z0-9.+-]+);base64,", re.IGNORECASE
)

# 下载相关默认配置
_DEFAULT_DOWNLOAD_TIMEOUT: int = 30
_MAX_DOWNLOAD_SIZE: int = 100 * 1024 * 1024  # 100 MB


# ---------------------------------------------------------------------------
# 异常定义
# ---------------------------------------------------------------------------


class ImageError(Exception):
    """图像处理基础异常。"""


class ImageFormatError(ImageError):
    """不支持或无法识别的图像格式。"""


class ImageDownloadError(ImageError):
    """图像下载失败。"""


# ---------------------------------------------------------------------------
# 内部辅助函数
# ---------------------------------------------------------------------------


def _normalize_format(fmt: Optional[str]) -> Optional[str]:
    """统一格式字符串：小写化，并将常见别名映射为标准名称。

    Args:
        fmt: 原始格式字符串，可为 None。

    Returns:
        标准化后的格式字符串，或 None（当输入为 None / 空字符串时）。

    Examples:
        >>> _normalize_format("JPG")
        'jpeg'
        >>> _normalize_format("TIFF")
        'tiff'
        >>> _normalize_format(None)
        None
    """
    if not fmt:
        return None
    fmt = fmt.strip().lower()
    if not fmt:
        return None
    return _FORMAT_ALIASES.get(fmt, fmt)


def _guess_format_from_suffix(path_or_url: str) -> Optional[str]:
    """尝试从文件路径或 URL 的后缀推断图像格式。

    Args:
        path_or_url: 文件路径或 URL 字符串。

    Returns:
        推断出的格式（已标准化），或 None。
    """
    # 去掉查询参数与锚点
    clean = path_or_url.split("?")[0].split("#")[0]
    suffix = Path(clean).suffix  # 包含 '.'
    if not suffix:
        return None

    fmt = _normalize_format(suffix.lstrip("."))
    if fmt and fmt in SUPPORTED_FORMATS:
        return fmt

    return None


def _guess_format_from_bytes(data: bytes) -> Optional[str]:
    """通过 magic bytes 或 Pillow 从原始字节推断格式。

    Args:
        data: 原始图像字节数据（至少需要前 32 字节用于判断）。

    Returns:
        推断出的格式（已标准化），或 None。
    """
    if not data:
        return None

    # 1. 标准 magic bytes 匹配
    for magic, fmt in _MAGIC_BYTES:
        if data[: len(magic)] == magic:
            return fmt

    # 2. RIFF 容器 → 检查是否为 WEBP
    if data[:4] == b"RIFF" and len(data) >= 12 and data[8:12] == b"WEBP":
        return "webp"

    # 3. HEIF/HEIC → ftyp box 检测
    if len(data) >= 12 and data[4:8] == b"ftyp":
        brand = data[8:12].decode("ascii", errors="ignore").strip("\x00").lower()
        if brand in _HEIF_BRANDS:
            return "heif"

    # 4. 回退到 Pillow 解析
    try:
        with Image.open(io.BytesIO(data)) as img:
            pil_fmt = _normalize_format(img.format)
            if pil_fmt:
                return pil_fmt
    except Exception:
        logger.warning("无法从 bytes 识别图像格式")

    return None


def _pillow_save_format(fmt: str) -> str:
    """返回 Pillow ``Image.save`` 所接受的 format 参数值。"""
    return _PILLOW_SAVE_FORMAT.get(fmt, fmt.upper())


def _ensure_rgb_for_jpeg(img: Image.Image) -> Image.Image:
    """将非 JPEG 兼容模式安全转换为 RGB。

    JPEG 仅原生支持 RGB / L / CMYK。此函数处理含 alpha、调色板、
    高位深灰度等不兼容模式，确保保存时不会抛出异常。
    """
    mode = img.mode

    # 快速路径：最常见的情况直接返回
    if mode == "RGB":
        return img

    if mode in _ALPHA_MODES:
        logger.debug("将模式 %s 转换为 RGB 以兼容 JPEG", mode)
        return img.convert("RGB")

    if mode == "P":
        # 调色板可能含透明色，须先展开为 RGBA 再丢弃 alpha
        logger.debug("将模式 P 经 RGBA 转换为 RGB 以正确处理透明色")
        return img.convert("RGBA").convert("RGB")

    if mode in _HILO_GREY_MODES:
        # I(32-bit int) / F(float) 先降为 8-bit 灰度再扩展
        logger.debug("将模式 %s 经 L 转换为 RGB 以兼容 JPEG", mode)
        return img.convert("L").convert("RGB")

    # L / CMYK 是 JPEG 原生支持的模式，直接返回
    if mode in ("L", "CMYK"):
        return img

    # 兜底：1、YCbCr 等其他罕见模式
    logger.debug("将模式 %s 转换为 RGB 以兼容 JPEG", mode)
    return img.convert("RGB")


def _strip_data_url_prefix(b64_string: str) -> tuple[Optional[str], str]:
    """从 base64 字符串中分离 data URL 前缀。

    Args:
        b64_string: 可能包含 data:image/...;base64, 前缀的字符串。

    Returns:
        (从前缀提取的格式 或 None, 纯 base64 字符串)
    """
    b64_string = b64_string.strip()
    match = _DATA_URL_RE.match(b64_string)
    if match:
        fmt = _normalize_format(match.group("fmt"))
        pure_b64 = b64_string[match.end() :].strip()
        logger.debug("检测到 data URL 前缀, 图像格式=%s", fmt)
        return fmt, pure_b64
    return None, b64_string

def readable_bytes_size(size_bytes: int) -> str:
    """将字节数转换为人类可读的格式（KB、MB）。"""
    # 可读大小
    if size_bytes < 1024:
        readable = f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        readable = f"{size_bytes / 1024:.2f} KB"
    else:
        readable = f"{size_bytes / (1024 * 1024):.2f} MB"
    return readable


# ---------------------------------------------------------------------------
# 公开转换函数
# ---------------------------------------------------------------------------


def download_bytes_from_url(
    url: str,
    *,
    timeout: int = _DEFAULT_DOWNLOAD_TIMEOUT,
    max_size: int = _MAX_DOWNLOAD_SIZE,
) -> bytes:
    """从给定 URL 下载图片数据，返回原始二进制 bytes。

    Args:
        url: 图片的 HTTP/HTTPS 地址。
        timeout: 请求超时时间（秒），默认 30s。
        max_size: 最大允许下载大小（字节），默认 100 MB。

    Returns:
        下载得到的原始字节数据。

    Raises:
        ImageDownloadError: 下载失败、超时或文件过大时。
        ValueError: URL 不是有效的 HTTP/HTTPS 地址时。
    """
    # URL 合法性校验
    if not url or not url.strip().startswith(("http://", "https://")):
        raise ValueError(f"URL 必须以 http:// 或 https:// 开头，收到: {url!r}")

    try:
        response = requests.get(url, timeout=timeout, stream=True)
        response.raise_for_status()

        # 检查 Content-Length（如果服务端提供）
        content_length = response.headers.get("Content-Length")
        if content_length and int(content_length) > max_size:
            raise ImageDownloadError(
                f"文件大小: {readable_bytes_size(int(content_length))}, 超过限制: {readable_bytes_size(max_size)}"
            )

        # 流式读取，防止大文件撑爆内存
        chunks: list[bytes] = []
        downloaded = 0
        for chunk in response.iter_content(chunk_size=8192):
            if not chunk:
                continue
            downloaded += len(chunk)
            if downloaded > max_size:
                raise ImageDownloadError(
                    f"下载数据量: {readable_bytes_size(downloaded)}, 超过限制: {readable_bytes_size(max_size)}"
                )
            chunks.append(chunk)

        data = b"".join(chunks)
        logger.debug("下载完成，数据大小: %s", readable_bytes_size(len(data)))
        return data

    except requests.Timeout as exc:
        raise ImageDownloadError(f"下载超时 (timeout={timeout}s): {url}") from exc
    except requests.HTTPError as exc:
        raise ImageDownloadError(
            f"HTTP 错误 {exc.response.status_code}: {url}"
        ) from exc
    except requests.RequestException as exc:
        raise ImageDownloadError(f"下载失败: {url} - {exc}") from exc


def bytes_to_img(data: bytes) -> Image.Image:
    """将二进制 bytes 转换为 Pillow Image。

    注意: 返回的 Image 已调用 load()，数据完全驻留内存，
    不依赖底层 BytesIO 缓冲区的生命周期。

    Args:
        data: 图像的原始字节数据。

    Returns:
        Pillow Image 对象（数据已完全加载至内存）。

    Raises:
        ImageFormatError: 无法解析的图像数据。
    """
    if not data:
        raise ImageFormatError("图像数据为空")

    logger.debug("输入数据大小: %s", readable_bytes_size(len(data)))
    try:
        buf = io.BytesIO(data)
        img = Image.open(buf)
        img.load()  # 确保像素数据完全读入内存，防止 buf 被 GC 后读取失败
        return img
    except Exception as exc:
        raise ImageFormatError(f"无法解析图像数据: {exc}") from exc


def img_to_bytes(img: Image.Image, fmt: Optional[str] = None) -> bytes:
    """将 Pillow Image 转换为 bytes。

    Args:
        img: Pillow Image 对象。
        fmt: 目标格式（小写），默认为 None 时使用 img.format 或 'jpeg'。

    Returns:
        编码后的图像字节数据。
    """
    if fmt is None:
        fmt = _normalize_format(img.format) or _DEFAULT_FORMAT
    else:
        fmt = _normalize_format(fmt) or _DEFAULT_FORMAT

    save_fmt = _pillow_save_format(fmt)
    buf = io.BytesIO()

    # 处理 JPEG 不支持的色彩模式
    save_img = _ensure_rgb_for_jpeg(img) if fmt == "jpeg" else img

    save_img.save(buf, format=save_fmt)
    data = buf.getvalue()
    logger.debug("将图像编码为 %s, 大小: %s", fmt, readable_bytes_size(len(data)))
    return data


def base64_to_bytes(data: str) -> bytes:
    """将 base64 字符串转换为 bytes。

    支持纯 base64 内容和带 data URL 前缀两种形式。

    Args:
        data: base64 编码的字符串（可带 data:image/...;base64, 前缀）。

    Returns:
        解码后的原始字节数据。

    Raises:
        ValueError: base64 解码失败。
    """
    if not data or not data.strip():
        raise ValueError("base64 字符串不能为空")

    _, pure_b64 = _strip_data_url_prefix(data)

    try:
        decoded = base64.b64decode(pure_b64, validate=True)
    except Exception as exc:
        raise ValueError(f"base64 解码失败: {exc}") from exc

    logger.debug("解码后大小: %d KB", len(decoded) / 1024)
    return decoded


def bytes_to_base64(data: bytes, *, with_data_prefix: bool = False) -> str:
    """将 bytes 转换为 base64 字符串。

    Args:
        data: 原始字节数据。
        with_data_prefix: 为 True 时添加 data:image/<fmt>;base64, 前缀。

    Returns:
        base64 编码的字符串。
    """
    encoded = base64.b64encode(data).decode("ascii")

    if with_data_prefix:
        fmt = _guess_format_from_bytes(data) or _DEFAULT_FORMAT
        return f"data:image/{fmt};base64,{encoded}"

    logger.debug("编码后长度=%d", len(encoded))
    return encoded


def base64_to_img(data: str) -> Image.Image:
    """将 base64 字符串转换为 Pillow Image。

    Args:
        data: base64 编码的图像字符串（支持 data URL 前缀）。

    Returns:
        Pillow Image 对象。
    """
    raw_bytes = base64_to_bytes(data)
    return bytes_to_img(raw_bytes)


def img_to_base64(
    img: Image.Image, *, with_data_prefix: bool = False, fmt: Optional[str] = None
) -> str:
    """将 Pillow Image 转换为 base64 字符串。

    Args:
        img: Pillow Image 对象。
        with_data_prefix: 为 True 时添加 data URL 前缀。
        fmt: 目标编码格式，默认为 None（使用 img.format）。

    Returns:
        base64 编码的字符串。
    """
    raw_bytes = img_to_bytes(img, fmt=fmt)
    return bytes_to_base64(raw_bytes, with_data_prefix=with_data_prefix)


# ---------------------------------------------------------------------------
# MyImage 类
# ---------------------------------------------------------------------------


class MyImage:
    """通用图像包装类，支持多种输入来源并统一为 Pillow Image。

    支持的输入来源（互斥，仅允许传入其中一种）：
        - path:   本地文件路径（str 或 Path）。
        - url:    HTTP/HTTPS 图片地址。
        - byte:   原始字节 bytes。
        - base64: base64 编码字符串（支持 data URL 前缀）。
        - img:    已有的 Pillow Image.Image 对象。

    也可通过位置参数 ``source`` 传入，类会自动识别类型。

    支持上下文管理器协议::

        with MyImage(url="https://example.com/photo.jpg") as img:
            img.save("/tmp/photo.jpg")

    Attributes:
        format (str): 图像格式，小写（如 'png', 'jpeg', 'webp'）。

    Examples:
        >>> img = MyImage(path="photo.jpg")
        >>> img.format
        'jpeg'
        >>> img.size
        (1920, 1080)

        >>> web_img = MyImage(url="https://example.com/image.png")
        >>> web_img.save("local_copy.png")

        >>> converted = web_img.convert("webp")
        >>> converted.format
        'webp'
    """

    __slots__ = ("_img", "_format", "_bytes", "_base64")

    def __init__(
        self,
        source: Optional[Union[str, Path, bytes, Image.Image]] = None,
        *,
        path: Optional[Union[str, Path]] = None,
        url: Optional[str] = None,
        byte: Optional[bytes] = None,
        # 注意: 参数名 base64 会在此作用域内遮蔽同名模块，
        # 内部通过调用模块级函数（base64_to_bytes 等）间接使用模块，无影响。
        base64: Optional[str] = None,
        img: Optional[Image.Image] = None,
    ) -> None:
        # ------ 解析位置参数 source ------
        if source is not None:
            if isinstance(source, Image.Image):
                img = source
            elif isinstance(source, bytes):
                byte = source
            elif isinstance(source, Path):
                path = source
            elif isinstance(source, str):
                if source.startswith("data:image/"):
                    base64 = source
                elif source.startswith(("http://", "https://")):
                    url = source
                elif os.path.isfile(source):
                    path = source
                else:
                    base64 = source
            else:
                raise TypeError(
                    f"source 参数类型不支持: {type(source).__name__}, "
                    f"期望 str / Path / bytes / Image.Image"
                )

        # ------ 互斥检查 ------
        sources = {
            "path": path,
            "url": url,
            "byte": byte,
            "base64": base64,
            "img": img,
        }
        provided = {k: v for k, v in sources.items() if v is not None}

        if len(provided) == 0:
            raise ValueError(
                "必须提供至少一个图像来源 (path/url/byte/base64/img)"
            )
        if len(provided) > 1:
            raise ValueError(
                f"仅允许传入一种图像来源，但同时传入了: {list(provided.keys())}"
            )

        self._img: Optional[Image.Image] = None
        self._bytes: Optional[bytes] = None
        self._base64: Optional[str] = None # 始终存储 *纯* base64 字符串（不含 data URL 前缀）
        self._format: Optional[str] = None

        # ------ 根据来源进行加载 ------
        if path is not None:
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"文件不存在: {path}")
            self._img = Image.open(path)
            self._img.load()  # 读入内存，释放文件句柄

        elif url is not None:
            raw_bytes = download_bytes_from_url(url)
            self._img = bytes_to_img(raw_bytes)

        elif byte is not None:
            self._img = bytes_to_img(byte)

        elif base64 is not None:
            raw_bytes = base64_to_bytes(base64)
            self._img = bytes_to_img(raw_bytes)

        elif img is not None:
            self._img = img

        # ------ 统一为 RGB 格式的 JPEG 作为标准格式 ------
        self._img = _ensure_rgb_for_jpeg(self._img)
        if self._img.mode != "RGB":
            self._img = self._img.convert("RGB")
        self._format = _DEFAULT_FORMAT

    # ---- 便捷属性 ----

    @property
    def img(self) -> Image.Image:
        """返回内部持有的 Pillow Image 对象。"""
        return self._img

    @property
    def width(self) -> int:
        """图像宽度（像素）。"""
        return self._img.width

    @property
    def height(self) -> int:
        """图像高度（像素）。"""
        return self._img.height

    @property
    def size(self) -> tuple[int, int]:
        """图像尺寸 ``(宽, 高)``。"""
        return self._img.size

    @property
    def mode(self) -> str:
        """图像色彩模式（如 ``'RGB'``, ``'RGBA'``, ``'L'``）。"""
        return self._img.mode

    @property
    def byte(self) -> bytes:
        """返回当前图像按 self.format 编码后的 bytes（惰性缓存）。"""
        if self._bytes is None:
            self._bytes = img_to_bytes(self._img, fmt=self._format)
        return self._bytes

    @property
    def base64(self) -> str:
        """返回纯 base64 字符串（**不含** data URL 前缀），惰性缓存。"""
        if self._base64 is None:
            self._base64 = img_to_base64(self._img, fmt=self._format)
        return self._base64
    
    @property
    def base64_with_prefix(self) -> str:
        """返回带 ``data:image/<fmt>;base64,`` 前缀的完整 data URL。"""
        # 统一通过 self.base64 获取纯 base64，再拼接前缀，避免重复前缀
        return f"data:image/{self._format};base64,{self.base64}"

    @property
    def format(self) -> str:
        """返回当前图像的格式（小写）。"""
        return self._format

    def get_info(self) -> dict:
        """获取当前图像的基本信息。

        Returns:
            包含 format, size, readable_size, mode, exif 等键的字典。
        """
        # 通过 self.byte 属性触发惰性初始化，避免 self._bytes 为 None
        raw = self.byte
        size_bytes = len(raw)
        readable = readable_bytes_size(size_bytes)

        # EXIF 信息
        exif_data: dict = {}
        try:
            raw_exif = self._img.getexif()
            if raw_exif:
                for tag_id, value in raw_exif.items():
                    tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                    if isinstance(value, bytes):
                        value = value.hex()
                    elif isinstance(value, (list, tuple)):
                        value = str(value)
                    exif_data[tag_name] = value
        except Exception as exc:
            logger.error(f"读取 EXIF 信息时出错: {exc}")

        info = {
            "format": self._format,
            "size": self._img.size,
            "mode": self._img.mode,
            "bytes_size": readable,
            "exif": exif_data,
        }

        return info

    # ---- I/O ----

    def save(self, path: Union[str, Path], fmt: Optional[str] = None) -> Path:
        """将图像保存到本地文件。

        Args:
            path: 目标文件路径。
            fmt: 目标格式（小写），为 None 时优先从路径后缀推断，
                 否则使用 self._format。

        Returns:
            保存后的文件路径（Path 对象）。
        """
        path = Path(path)

        # 优先从路径后缀推断格式
        if fmt is None:
            fmt = _guess_format_from_suffix(str(path)) or self._format
        else:
            fmt = _normalize_format(fmt) or self._format

        # 自动创建父目录
        path.parent.mkdir(parents=True, exist_ok=True)

        pil_fmt = _pillow_save_format(fmt)
        save_img = _ensure_rgb_for_jpeg(self._img) if fmt == "jpeg" else self._img

        save_img.save(str(path), format=pil_fmt)
        logger.debug(f"图像已保存至: {path} (格式={fmt})")
        return path

    # ---- 资源管理 ----

    def close(self) -> None:
        """关闭内部 Pillow Image 并释放资源。"""
        if self._img is not None:
            try:
                self._img.close()
            except Exception:
                pass

    def __enter__(self) -> MyImage:
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"<MyImage format={self._format!r} size={self._img.size} "
            f"mode={self._img.mode!r}>"
        )

"""Web 抓取工具：web_fetch。

从 agent.lc_tools 提取。
"""
from __future__ import annotations

import gzip
import re
import urllib.request
from html.parser import HTMLParser

from langchain_core.tools import tool


class _TextExtractor(HTMLParser):
    """HTML 文本提取器：去除脚本和样式标签"""
    def __init__(self):
        super().__init__()
        self._parts: list[str] = []
        self._skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip = True

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self._skip = False
        if tag in ("p", "br", "div", "li", "tr", "h1", "h2", "h3", "h4"):
            self._parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self._parts.append(data)

    def get_text(self) -> str:
        return re.sub(r"\n{3,}", "\n\n", "".join(self._parts)).strip()


def _fetch(url: str, extract_mode: str = "text", max_chars: int = 8000) -> str:
    """抓取网页内容并提取文本"""
    _UA = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    )
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": _UA,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw_bytes = resp.read()
    except Exception as e:
        return f"Error fetching {url}: {e}"

    # 解压 gzip/deflate
    content_encoding = resp.headers.get("Content-Encoding", "")
    if content_encoding == "gzip":
        try:
            raw_bytes = gzip.decompress(raw_bytes)
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning("gzip decompression failed for %s: %s", url, e)
    elif content_encoding == "deflate":
        try:
            raw_bytes = gzip.decompress(raw_bytes)
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning("deflate decompression failed for %s: %s", url, e)

    # 检测字符编码
    charset = "utf-8"
    content_type = resp.headers.get("Content-Type", "")
    m = re.search(r"charset=([\w-]+)", content_type, re.IGNORECASE)
    if m:
        charset = m.group(1).lower()
    if charset in ("gbk", "gb2312", "gb18030"):
        charset = "gb18030"

    try:
        text = raw_bytes.decode(charset, errors="replace")
    except (LookupError, ValueError):
        text = raw_bytes.decode("utf-8", errors="replace")

    if extract_mode == "text":
        parser = _TextExtractor()
        parser.feed(text)
        text = parser.get_text()
    return text[:max_chars]


@tool
def web_fetch(url: str, extract_mode: str = "text", max_chars: int = 8000) -> str:
    """获取指定 URL 的网页内容。extract_mode: text（纯文本，默认）或 raw（原始 HTML）
    Args:
        url: 要抓取的网页 URL
        extract_mode: 提取模式，"text" 提取纯文本（去除脚本和样式），"raw" 返回原始 HTML（默认 "text"）
        max_chars: 最大返回字符数（默认 8000）
    """
    return _fetch(url, extract_mode, max_chars)

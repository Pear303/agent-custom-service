from __future__ import annotations

from unittest.mock import patch, MagicMock


def _make_mock_resp(html: str) -> MagicMock:
    """创建一个模拟 urllib HTTPResponse 对象。"""
    mock_resp = MagicMock()
    mock_resp.read.return_value = html.encode("utf-8")
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)
    # headers 需要支持 .get() 返回字符串（模拟 email.message.Message）
    mock_resp.headers = {"Content-Type": "text/html; charset=utf-8"}
    return mock_resp


def test_web_fetch_returns_text():
    from agent.lc_tools import web_fetch
    mock_html = "<html><body><p>Hello world</p></body></html>"
    mock_resp = _make_mock_resp(mock_html)
    with patch("urllib.request.urlopen", return_value=mock_resp):
        result = web_fetch.invoke({"url": "https://example.com"})
        assert "Hello world" in result


def test_web_fetch_raw_mode():
    from agent.lc_tools import web_fetch
    mock_html = "<html><body>raw</body></html>"
    mock_resp = _make_mock_resp(mock_html)
    with patch("urllib.request.urlopen", return_value=mock_resp):
        result = web_fetch.invoke({
            "url": "https://example.com", "extract_mode": "raw"
        })
        assert "raw" in result


def test_web_fetch_error():
    from agent.lc_tools import web_fetch
    with patch("urllib.request.urlopen", side_effect=Exception("timeout")):
        result = web_fetch.invoke({"url": "https://bad.url"})
        assert result.startswith("Error")

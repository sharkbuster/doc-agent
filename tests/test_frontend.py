"""前端的静态检查。

跑不了浏览器，但有两类错误纯靠读文本就能查出来，而且这两类都真出过：

1. JS 里引用了 HTML 中不存在的 id。浏览器不会报错，只会静默失效——
   之前有个项目因此整个基础数据加载模块直接不工作了。
2. CSS 缺少 `[hidden]` 规则。只要哪个选择器写了 `display: flex`，
   `hidden` 属性就会失效，所有本该收起的面板会一起展开。
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

STATIC_DIR = Path(__file__).resolve().parent.parent / "app" / "static"
HTML = STATIC_DIR / "index.html"
CSS = STATIC_DIR / "style.css"
JS = STATIC_DIR / "app.js"

# 只抓完整的字符串字面量。像 getElementById('cite-' + i) 这种运行时拼出来的 id
# 本来就不在 HTML 里（元素是 JS 自己建的），不能当成错误。
GET_BY_ID = re.compile(r"""getElementById\(\s*['"]([^'"]+)['"]\s*\)""")
QUERY_ID = re.compile(r"""querySelector(?:All)?\(\s*['"]#([^'"\s.\[:>]+)['"]\s*\)""")
HTML_ID = re.compile(r"""\bid\s*=\s*["']([^"']+)["']""")

pytestmark = pytest.mark.skipif(
    not HTML.is_file(), reason="前端还没生成"
)


@pytest.fixture(scope="module")
def sources() -> dict[str, str]:
    return {
        "html": HTML.read_text(encoding="utf-8"),
        "css": CSS.read_text(encoding="utf-8"),
        "js": JS.read_text(encoding="utf-8"),
    }


def test_all_three_files_exist():
    for path in (HTML, CSS, JS):
        assert path.is_file(), f"缺少 {path.name}"
        assert path.stat().st_size > 500, f"{path.name} 太小了，不像写完了"


def test_js_only_references_ids_that_exist(sources):
    html_ids = set(HTML_ID.findall(sources["html"]))
    referenced = set(GET_BY_ID.findall(sources["js"])) | set(QUERY_ID.findall(sources["js"]))
    missing = sorted(referenced - html_ids)
    assert not missing, f"JS 引用了 HTML 里不存在的 id：{missing}"


def test_html_has_the_core_controls(sources):
    html_ids = set(HTML_ID.findall(sources["html"]))
    for required in ("messages", "input", "btn-send", "composer", "health-body", "tools-body"):
        assert required in html_ids, f"缺少关键元素 #{required}"


def test_hidden_attribute_is_enforced(sources):
    # 少了这条 !important，任何 display:flex 的选择器都会盖掉 hidden
    assert re.search(r"\[hidden\][^{]*\{[^}]*display\s*:\s*none\s*!important", sources["css"]), (
        "style.css 里必须有 [hidden] { display: none !important; }"
    )


def test_html_references_assets_under_the_static_mount(sources):
    # 必须是 /static/... 这种绝对路径。写相对路径 style.css 的话，
    # 页面在 / 上会被解析成 /style.css，而静态目录挂在 /static 下 —— 浏览器 404，
    # 页面变成一堆没样式的裸 HTML，JS 也完全不执行。这个坑真踩过。
    assert 'href="/static/style.css"' in sources["html"]
    assert 'src="/static/app.js"' in sources["html"]


def test_no_external_resources(sources):
    # 前端必须能完全离线打开：一个 CDN 链接就能让内网演示变成白屏
    forbidden = ("http://", "https://", "//cdn", "@import", "unpkg", "jsdelivr", "cdnjs")
    for name, text in sources.items():
        for token in forbidden:
            assert token not in text, f"{name} 里有外部资源引用：{token}"


def test_frontend_uses_the_documented_endpoints(sources):
    for endpoint in ("/api/health", "/api/tools", "/api/demo/orders", "/api/chat/stream"):
        assert endpoint in sources["js"], f"前端没有调用 {endpoint}"


def test_frontend_parses_sse_manually(sources):
    # EventSource 不支持 POST + JSON body，前端必须自己解析流
    assert "getReader" in sources["js"] or "ReadableStream" in sources["js"]
    assert "text/event-stream" in sources["js"] or "data: " in sources["js"]


def test_no_inline_event_handlers(sources):
    # 内联 onclick 会和 CSP 打架，也不好调试
    assert not re.search(r"""\son(click|change|submit)\s*=""", sources["html"])


def test_frontend_labels_the_score_by_the_actual_signal(sources):
    # 判据是覆盖度就不能写成「相似度」：纯词法模式下余弦恒为 0，
    # 标成相似度会显示成「相似度 0.00」，看着像检索挂了
    assert "refusal_signal" in sources["js"]
    assert "覆盖度" in sources["js"]


def test_stylesheet_targets_narrow_screens(sources):
    # 手机上打开也不能难看
    assert "@media" in sources["css"]

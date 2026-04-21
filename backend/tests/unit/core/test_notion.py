from __future__ import annotations

from app.adapters.ingestion.notion import (
    _extract_title,
    _rich_text_to_str,
    render_blocks,
)

# ---------------------------------------------------------------------------
# _rich_text_to_str
# ---------------------------------------------------------------------------


def test_rich_text_to_str_concatenates_spans():
    spans = [
        {"plain_text": "Hello "},
        {"plain_text": "world"},
    ]
    assert _rich_text_to_str(spans) == "Hello world"


def test_rich_text_to_str_empty():
    assert _rich_text_to_str([]) == ""


# ---------------------------------------------------------------------------
# _extract_title
# ---------------------------------------------------------------------------


def test_extract_title_from_properties():
    props = {
        "Name": {
            "type": "title",
            "title": [{"plain_text": "My Page Title"}],
        },
        "Tags": {"type": "multi_select", "multi_select": []},
    }
    assert _extract_title(props) == "My Page Title"


def test_extract_title_returns_empty_when_no_title_property():
    props = {"Status": {"type": "select", "select": None}}
    assert _extract_title(props) == ""


# ---------------------------------------------------------------------------
# render_blocks
# ---------------------------------------------------------------------------


def _block(block_type: str, text: str, **extra) -> dict:
    """Build a minimal Notion block dict for testing."""
    content: dict = {"rich_text": [{"plain_text": text}]}
    content.update(extra)
    return {"type": block_type, block_type: content}


def test_render_paragraph():
    blocks = [_block("paragraph", "Some paragraph text.")]
    assert render_blocks(blocks) == "Some paragraph text."


def test_render_headings():
    blocks = [
        _block("heading_1", "Title"),
        _block("heading_2", "Subtitle"),
        _block("heading_3", "Section"),
    ]
    result = render_blocks(blocks)
    assert "# Title" in result
    assert "## Subtitle" in result
    assert "### Section" in result


def test_render_bulleted_list():
    blocks = [
        _block("bulleted_list_item", "First"),
        _block("bulleted_list_item", "Second"),
    ]
    result = render_blocks(blocks)
    assert "- First" in result
    assert "- Second" in result


def test_render_numbered_list():
    blocks = [_block("numbered_list_item", "Step one")]
    assert "1. Step one" in render_blocks(blocks)


def test_render_to_do():
    blocks = [
        _block("to_do", "Done task", checked=True),
        _block("to_do", "Pending task", checked=False),
    ]
    result = render_blocks(blocks)
    assert "- [x] Done task" in result
    assert "- [ ] Pending task" in result


def test_render_code_block():
    blocks = [_block("code", "print('hello')", language="python")]
    result = render_blocks(blocks)
    assert "```python" in result
    assert "print('hello')" in result
    assert result.strip().endswith("```")


def test_render_divider():
    blocks = [{"type": "divider", "divider": {}}]
    assert render_blocks(blocks) == "---"


def test_render_skips_empty_blocks():
    blocks = [_block("paragraph", "")]
    assert render_blocks(blocks) == ""


def test_render_nested_children():
    parent = _block("toggle", "Parent toggle")
    child = _block("paragraph", "Child content")
    parent["_children"] = [child]

    result = render_blocks([parent])
    assert "Parent toggle" in result
    assert "Child content" in result


def test_render_mixed_blocks_preserves_order():
    blocks = [
        _block("heading_2", "Overview"),
        _block("paragraph", "LangGraph is a framework."),
        _block("bulleted_list_item", "Stateful"),
        _block("bulleted_list_item", "Graph-based"),
    ]
    result = render_blocks(blocks)
    lines = [ln for ln in result.splitlines() if ln.strip()]
    assert lines[0] == "## Overview"
    assert lines[1] == "LangGraph is a framework."
    assert lines[2] == "- Stateful"
    assert lines[3] == "- Graph-based"

from __future__ import annotations

from pathlib import Path

import pytest


# ---------- read_file ----------

def test_read_file_returns_numbered_lines(temp_dir):
    from agent.lc_tools import read_file
    f = temp_dir / "hello.txt"
    f.write_text("line1\nline2\nline3", encoding="utf-8")
    result = read_file.invoke({"path": str(f)})
    assert "1| line1" in result
    assert "2| line2" in result
    assert "3| line3" in result


def test_read_file_not_found(temp_dir):
    from agent.lc_tools import read_file
    result = read_file.invoke({"path": str(temp_dir / "nope.txt")})
    assert result.startswith("Error")


def test_read_file_offset(temp_dir):
    from agent.lc_tools import read_file
    f = temp_dir / "data.txt"
    f.write_text("\n".join(str(i) for i in range(1, 11)), encoding="utf-8")
    result = read_file.invoke({"path": str(f), "offset": 5})
    assert "5| 5" in result
    assert "1| 1" not in result


def test_read_file_limit(temp_dir):
    from agent.lc_tools import read_file
    f = temp_dir / "data.txt"
    f.write_text("\n".join(str(i) for i in range(1, 11)), encoding="utf-8")
    result = read_file.invoke({"path": str(f), "limit": 3})
    assert "3| 3" in result
    assert "4| 4" not in result


# ---------- write_file ----------

def test_write_file_creates_file(temp_dir):
    from agent.lc_tools import write_file
    target = temp_dir / "out.txt"
    result = write_file.invoke({"path": str(target), "content": "hello"})
    assert "Success" in result
    assert target.read_text(encoding="utf-8") == "hello"


def test_write_file_overwrites(temp_dir):
    from agent.lc_tools import write_file
    target = temp_dir / "out.txt"
    target.write_text("old")
    write_file.invoke({"path": str(target), "content": "new"})
    assert target.read_text(encoding="utf-8") == "new"


def test_write_file_creates_parent_dirs(temp_dir):
    from agent.lc_tools import write_file
    target = temp_dir / "a" / "b" / "c.txt"
    write_file.invoke({"path": str(target), "content": "deep"})
    assert target.read_text(encoding="utf-8") == "deep"


# ---------- edit_file ----------

def test_edit_file_exact_replace(temp_dir):
    from agent.lc_tools import edit_file
    f = temp_dir / "code.py"
    f.write_text("x = 1\ny = 2\n", encoding="utf-8")
    result = edit_file.invoke({
        "path": str(f), "old_text": "x = 1", "new_text": "x = 100"
    })
    assert "Success" in result
    assert f.read_text(encoding="utf-8") == "x = 100\ny = 2\n"


def test_edit_file_not_found(temp_dir):
    from agent.lc_tools import edit_file
    f = temp_dir / "code.py"
    f.write_text("hello", encoding="utf-8")
    result = edit_file.invoke({
        "path": str(f), "old_text": "nope", "new_text": "yes"
    })
    assert "Error" in result


def test_edit_file_replace_all(temp_dir):
    from agent.lc_tools import edit_file
    f = temp_dir / "code.py"
    f.write_text("x = 1\nx = 1\n", encoding="utf-8")
    result = edit_file.invoke({
        "path": str(f), "old_text": "x = 1", "new_text": "x = 0",
        "replace_all": True,
    })
    assert "Success" in result
    assert f.read_text(encoding="utf-8") == "x = 0\nx = 0\n"

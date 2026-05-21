from __future__ import annotations


def test_glob_finds_files(temp_dir):
    from agent.lc_tools import glob_tool
    (temp_dir / "a.py").write_text("")
    (temp_dir / "b.txt").write_text("")
    (temp_dir / "c.py").write_text("")
    result = glob_tool.invoke({"pattern": "*.py", "path": str(temp_dir)})
    assert "a.py" in result
    assert "c.py" in result
    assert "b.txt" not in result


def test_glob_no_match(temp_dir):
    from agent.lc_tools import glob_tool
    result = glob_tool.invoke({"pattern": "*.xyz", "path": str(temp_dir)})
    assert "No paths matched" in result


def test_glob_directories(temp_dir):
    from agent.lc_tools import glob_tool
    (temp_dir / "src").mkdir()
    (temp_dir / "main.py").write_text("")
    result = glob_tool.invoke({
        "pattern": "*", "path": str(temp_dir), "entry_type": "dirs"
    })
    assert "src" in result
    assert "main.py" not in result


def test_grep_finds_match(temp_dir):
    from agent.lc_tools import grep_tool
    (temp_dir / "config.py").write_text("DEBUG = True\nHOST = localhost\n")
    result = grep_tool.invoke({
        "pattern": "DEBUG", "path": str(temp_dir),
        "output_mode": "content",
    })
    assert "DEBUG" in result


def test_grep_files_with_matches_mode(temp_dir):
    from agent.lc_tools import grep_tool
    (temp_dir / "a.py").write_text("TODO: fix")
    (temp_dir / "b.py").write_text("all good")
    result = grep_tool.invoke({
        "pattern": "TODO", "path": str(temp_dir),
        "output_mode": "files_with_matches",
    })
    assert "a.py" in result
    assert "b.py" not in result


def test_grep_count_mode(temp_dir):
    from agent.lc_tools import grep_tool
    (temp_dir / "data.txt").write_text("apple\nbanana\napple\n")
    result = grep_tool.invoke({
        "pattern": "apple", "path": str(temp_dir),
        "output_mode": "count",
    })
    assert ": 2" in result


def test_grep_case_insensitive(temp_dir):
    from agent.lc_tools import grep_tool
    (temp_dir / "text.txt").write_text("Hello\n")
    result = grep_tool.invoke({
        "pattern": "hello", "path": str(temp_dir),
        "output_mode": "content", "case_insensitive": True,
    })
    assert "Hello" in result


def test_grep_fixed_strings(temp_dir):
    from agent.lc_tools import grep_tool
    (temp_dir / "text.txt").write_text("a.b\n")
    result = grep_tool.invoke({
        "pattern": "a.b", "path": str(temp_dir),
        "output_mode": "content", "fixed_strings": True,
    })
    assert "a.b" in result

"""TODO 待办管理器 - 测试文件"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))

from todo import TodoManager


def test_add_and_list():
    """测试添加和列出。"""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        filepath = f.name

    try:
        manager = TodoManager(filepath)
        manager.add("买牛奶")
        manager.add("写代码")
        tasks = manager.list()
        assert len(tasks) == 2
        # BUG 5: 断言用了 == 比较列表但顺序不确定（多线程/异步场景）
        assert [t["title"] for t in tasks] == ["买牛奶", "写代码"]
    finally:
        os.unlink(filepath)


def test_complete():
    """测试完成待办。"""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        filepath = f.name

    try:
        manager = TodoManager(filepath)
        task = manager.add("测试任务")
        manager.complete(task["id"])
        # BUG 3: complete 写 "done" 但 list("completed") 查 "completed"
        completed = manager.list("completed")
        assert len(completed) == 1, f"期望1个已完成任务，实际{len(completed)}"
    finally:
        os.unlink(filepath)


def test_delete():
    """测试删除待办。"""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        filepath = f.name

    try:
        manager = TodoManager(filepath)
        manager.add("任务A")
        manager.add("任务B")
        # BUG 2: delete 用 int(task_id) 做索引而非匹配 id 字段
        result = manager.delete("0")
        assert result is not False, "删除应该成功"
        remaining = manager.list()
        assert len(remaining) == 1
    finally:
        os.unlink(filepath)


if __name__ == "__main__":
    test_add_and_list()
    test_complete()
    test_delete()
    print("所有测试通过！")

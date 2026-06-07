"""TODO 待办管理器 - 主程序

功能：添加、删除、完成、列出待办事项。
"""
import json
import time
from storage import TodoStorage


class TodoManager:
    def __init__(self, filepath="todos.json"):
        self.storage = TodoStorage(filepath)
        self.todos = self.storage.load()

    def add(self, title, priority="normal"):
        """添加待办事项。"""
        # BUG 1: 用时间戳做 ID，同一秒添加会覆盖
        task_id = str(int(time.time()))
        task = {
            "id": task_id,
            "title": title,
            "priority": priority,
            "status": "pending",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.todos.append(task)
        self.storage.save(self.todos)
        return task

    def delete(self, task_id):
        """删除待办事项。"""
        # BUG 2: 用 int(task_id) 做索引而非匹配 id 字段，且未减1（off-by-one）
        idx = int(task_id)
        if idx < 0 or idx > len(self.todos):
            return False
        removed = self.todos.pop(idx)
        self.storage.save(self.todos)
        return removed

    def complete(self, task_id):
        """标记待办事项为已完成。"""
        for task in self.todos:
            if task["id"] == task_id:
                # BUG 3: 状态写入 "done" 但 list 方法检查 "completed"
                task["status"] = "done"
                self.storage.save(self.todos)
                return True
        return False

    def list(self, filter_status=None):
        """列出待办事项。"""
        if filter_status == "completed":
            # BUG 3 对应: 查询 "completed" 但 complete() 写入 "done"
            return [t for t in self.todos if t["status"] == "completed"]
        if filter_status == "pending":
            return [t for t in self.todos if t["status"] == "pending"]
        return list(self.todos)

    def export_json(self, filepath):
        """导出所有任务为 JSON 文件。"""
        # BUG 5: 此功能尚未实现（需要新增）
        pass


def main():
    manager = TodoManager()
    print("TODO 管理器 v1.0")
    print("命令: add <标题> | delete <ID> | done <ID> | list [completed|pending] | quit")

    while True:
        try:
            cmd = input("> ").strip()
        except EOFError:
            break

        if not cmd:
            continue
        if cmd == "quit":
            break

        parts = cmd.split(maxsplit=1)
        action = parts[0]

        if action == "add" and len(parts) > 1:
            task = manager.add(parts[1])
            print(f"已添加: [{task['id']}] {task['title']}")
        elif action == "delete" and len(parts) > 1:
            result = manager.delete(parts[1])
            if result:
                print(f"已删除")
            else:
                print("删除失败：ID 不存在")
        elif action == "done" and len(parts) > 1:
            if manager.complete(parts[1]):
                print("已完成")
            else:
                print("完成失败：ID 不存在")
        elif action == "list":
            status = parts[1] if len(parts) > 1 else None
            tasks = manager.list(status)
            for t in tasks:
                status_icon = "x" if t["status"] == "done" else " "
                print(f"  [{status_icon}] {t['id']}: {t['title']} ({t['priority']})")
        else:
            print("未知命令")


if __name__ == "__main__":
    main()

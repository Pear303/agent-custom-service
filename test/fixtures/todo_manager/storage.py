"""TODO 待办管理器 - 存储模块

负责 JSON 文件的读写。
"""
import json
import os


class TodoStorage:
    def __init__(self, filepath="todos.json"):
        self.filepath = filepath

    def load(self):
        """从文件加载待办列表。"""
        if not os.path.exists(self.filepath):
            return []
        # BUG 4: 打开文件时未指定 encoding，Windows 下中文会乱码
        with open(self.filepath, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []

    def save(self, todos):
        """保存待办列表到文件。"""
        # BUG 4: 写入时也未指定 encoding
        with open(self.filepath, "w") as f:
            json.dump(todos, f, indent=2, ensure_ascii=False)

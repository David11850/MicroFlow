from __future__ import annotations

import os
import re
import shlex
import subprocess
from dataclasses import dataclass
from typing import Optional


@dataclass
class TerminalResult:
    command: str
    stdout: str
    stderr: str
    exit_code: int


class TerminalSkill:
    """Execute shell commands from NL requests with basic safety checks."""

    ALLOWED_BINARIES = {
        "pwd",
        "ls",
        "cat",
        "head",
        "tail",
        "sed",
        "rg",
        "find",
        "python",
        "python3",
        "pytest",
        "git",
        "echo",
        "wc",
        "du",
    }

    BLOCKED_PATTERNS = [
        r"\brm\s+-rf\b",
        r"\bshutdown\b",
        r"\breboot\b",
        r"\bmkfs\b",
        r"\bdd\b",
        r":\(\)\s*\{",
        r"\bpoweroff\b",
        r"\binit\s+0\b",
    ]

    def __init__(self, workdir: Optional[str] = None, timeout_s: int = 30):
        self.workdir = workdir or os.getenv("AGENT_WORKDIR", os.getcwd())
        self.timeout_s = timeout_s

    @staticmethod
    def extract_command_from_text(text: str) -> Optional[str]:
        stripped = text.strip()

        # 显式命令模式：执行命令: xxx
        explicit = re.match(r"^(执行命令|运行命令|run command|cmd)\s*[:：]\s*(.+)$", stripped, re.IGNORECASE)
        if explicit:
            return explicit.group(2).strip()

        # shell 风格：!ls -la
        if stripped.startswith("!") and len(stripped) > 1:
            return stripped[1:].strip()

        # 常见自然语言指令映射
        if "列出" in stripped and "目录" in stripped:
            return "ls -la"

        if "当前目录" in stripped or "当前路径" in stripped:
            return "pwd"

        file_match = re.search(r"查看文件\s+([^\s]+)", stripped)
        if file_match:
            path = file_match.group(1)
            return f"sed -n '1,120p' {shlex.quote(path)}"

        search_match = re.search(r"搜索\s+([^\s]+)", stripped)
        if search_match:
            keyword = search_match.group(1)
            return f"rg {shlex.quote(keyword)}"

        if "运行测试" in stripped or "跑测试" in stripped:
            return "pytest -q"

        return None

    def _is_safe(self, command: str) -> tuple[bool, str]:
        lowered = command.lower()
        for pattern in self.BLOCKED_PATTERNS:
            if re.search(pattern, lowered):
                return False, f"命中危险命令规则: {pattern}"

        try:
            tokens = shlex.split(command)
        except ValueError:
            return False, "命令解析失败"

        if not tokens:
            return False, "空命令"

        binary = tokens[0]
        if binary not in self.ALLOWED_BINARIES:
            return False, f"不允许的命令: {binary}"

        return True, "ok"

    def execute(self, command: str) -> TerminalResult:
        ok, reason = self._is_safe(command)
        if not ok:
            return TerminalResult(command=command, stdout="", stderr=reason, exit_code=126)

        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=self.workdir,
                text=True,
                capture_output=True,
                timeout=self.timeout_s,
            )
            return TerminalResult(
                command=command,
                stdout=(proc.stdout or "")[:8000],
                stderr=(proc.stderr or "")[:4000],
                exit_code=proc.returncode,
            )
        except subprocess.TimeoutExpired:
            return TerminalResult(command=command, stdout="", stderr="命令执行超时", exit_code=124)


def format_terminal_reply(result: TerminalResult) -> str:
    return (
        "[Terminal Skill]\n"
        f"command: `{result.command}`\n"
        f"exit_code: {result.exit_code}\n\n"
        "stdout:\n"
        f"```\n{result.stdout or '(empty)'}\n```\n\n"
        "stderr:\n"
        f"```\n{result.stderr or '(empty)'}\n```"
    )

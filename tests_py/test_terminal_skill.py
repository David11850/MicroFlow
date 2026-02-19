from agent.terminal_skill import TerminalSkill


def test_extract_command_from_nl():
    assert TerminalSkill.extract_command_from_text("列出当前目录") == "ls -la"
    assert TerminalSkill.extract_command_from_text("执行命令: pwd") == "pwd"


def test_block_dangerous_command():
    skill = TerminalSkill()
    result = skill.execute("rm -rf /tmp/test")
    assert result.exit_code == 126
    assert "危险命令" in result.stderr


def test_execute_safe_command():
    skill = TerminalSkill()
    result = skill.execute("pwd")
    assert result.exit_code == 0
    assert result.stdout.strip()

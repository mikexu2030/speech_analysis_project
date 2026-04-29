#!/usr/bin/env python3
"""
Git自动上传模块
读取config/git_credentials.json配置，自动commit/push代码
"""

import os
import json
import subprocess
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime

CONFIG_PATH = Path(__file__).parent.parent / "config" / "git_credentials.json"


def load_config() -> dict:
    """加载Git配置"""
    if not CONFIG_PATH.exists():
        return {"git": {"enabled": False}}
    
    with open(CONFIG_PATH, 'r') as f:
        return json.load(f)


def check_git_installed() -> bool:
    """检查git是否安装"""
    try:
        subprocess.run(['git', '--version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def check_credentials(config: dict) -> Tuple[bool, str]:
    """检查凭证是否已配置"""
    creds = config.get("credentials", {})
    
    if creds.get("token"):
        return True, "token"
    elif creds.get("username") and creds.get("password"):
        return True, "password"
    else:
        return False, "none"


def setup_remote_auth(config: dict) -> bool:
    """设置远程仓库认证"""
    creds = config.get("credentials", {})
    token = creds.get("token", "")
    username = creds.get("username", "")
    password = creds.get("password", "")
    
    # 获取远程URL
    result = subprocess.run(
        ['git', 'remote', 'get-url', 'origin'],
        capture_output=True, text=True
    )
    
    if result.returncode != 0:
        print("[Git] No remote origin configured")
        return False
    
    remote_url = result.stdout.strip()
    
    # 如果URL已经是HTTPS且包含token，跳过
    if token and 'https://' in remote_url:
        # 提取域名
        if 'github.com' in remote_url:
            new_url = f"https://{token}@github.com/{'/'.join(remote_url.split('/')[-2:])}"
        elif 'gitlab' in remote_url:
            new_url = f"https://oauth2:{token}@{remote_url.split('://')[1]}"
        else:
            new_url = remote_url
        
        subprocess.run(['git', 'remote', 'set-url', 'origin', new_url], capture_output=True)
        return True
    
    return True


def git_status(project_dir: str) -> Tuple[bool, str]:
    """检查git状态，返回是否有变更"""
    result = subprocess.run(
        ['git', 'status', '--porcelain'],
        cwd=project_dir, capture_output=True, text=True
    )
    
    has_changes = bool(result.stdout.strip())
    return has_changes, result.stdout


def git_add_all(project_dir: str) -> bool:
    """添加所有变更"""
    result = subprocess.run(
        ['git', 'add', '-A'],
        cwd=project_dir, capture_output=True
    )
    return result.returncode == 0


def git_commit(project_dir: str, message: str) -> bool:
    """提交代码"""
    result = subprocess.run(
        ['git', 'commit', '-m', message],
        cwd=project_dir, capture_output=True
    )
    return result.returncode == 0


def git_push(project_dir: str, branch: str = "master") -> Tuple[bool, str]:
    """推送代码"""
    result = subprocess.run(
        ['git', 'push', 'origin', branch],
        cwd=project_dir, capture_output=True, text=True
    )
    
    success = result.returncode == 0
    output = result.stdout if success else result.stderr
    return success, output


def auto_git_push(project_dir: str, description: str = "update") -> dict:
    """
    自动Git上传主函数
    
    Args:
        project_dir: 项目目录
        description: 提交描述
    
    Returns:
        操作结果字典
    """
    result = {
        "success": False,
        "committed": False,
        "pushed": False,
        "message": "",
        "error": None
    }
    
    # 1. 检查配置
    config = load_config()
    if not config.get("git", {}).get("enabled", False):
        result["message"] = "Git auto-push disabled in config"
        return result
    
    # 2. 检查git安装
    if not check_git_installed():
        result["error"] = "Git not installed"
        return result
    
    # 3. 检查凭证
    cred_ok, cred_type = check_credentials(config)
    if not cred_ok:
        result["error"] = f"Git credentials not configured. Please fill in {CONFIG_PATH}"
        return result
    
    # 4. 检查是否有变更
    has_changes, status = git_status(project_dir)
    if not has_changes:
        result["message"] = "No changes to commit"
        return result
    
    # 5. 设置远程认证
    if not setup_remote_auth(config):
        result["error"] = "Failed to setup remote auth"
        return result
    
    # 6. 添加所有变更
    if not git_add_all(project_dir):
        result["error"] = "Failed to git add"
        return result
    
    # 7. 提交
    template = config.get("git", {}).get("commit_message_template", "auto: {description}")
    commit_msg = template.format(description=description, time=datetime.now().isoformat())
    
    if not git_commit(project_dir, commit_msg):
        result["error"] = "Failed to git commit"
        return result
    
    result["committed"] = True
    result["message"] = f"Committed: {commit_msg}"
    
    # 8. 推送（如果启用）
    if config.get("git", {}).get("auto_push", False):
        branch = config.get("git", {}).get("branch", "master")
        push_ok, push_output = git_push(project_dir, branch)
        if push_ok:
            result["pushed"] = True
            result["message"] += f" | Pushed to {branch}"
        else:
            result["error"] = f"Push failed: {push_output}"
    
    result["success"] = True
    return result


def test_git_connection(project_dir: str) -> dict:
    """测试Git连接"""
    result = {
        "git_installed": check_git_installed(),
        "config_exists": CONFIG_PATH.exists(),
        "credentials_ok": False,
        "has_changes": False,
        "remote_url": None,
        "error": None
    }
    
    if not result["git_installed"]:
        result["error"] = "Git not installed"
        return result
    
    config = load_config()
    cred_ok, _ = check_credentials(config)
    result["credentials_ok"] = cred_ok
    
    # 检查远程URL
    remote_result = subprocess.run(
        ['git', 'remote', 'get-url', 'origin'],
        cwd=project_dir, capture_output=True, text=True
    )
    if remote_result.returncode == 0:
        result["remote_url"] = remote_result.stdout.strip()
    
    # 检查变更
    has_changes, _ = git_status(project_dir)
    result["has_changes"] = has_changes
    
    return result


if __name__ == '__main__':
    # 测试模式
    import sys
    project_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    
    print("=== Git Auto-Push Test ===")
    test = test_git_connection(project_dir)
    for k, v in test.items():
        print(f"  {k}: {v}")
    
    if test["credentials_ok"] and test["has_changes"]:
        print("\n=== Auto Push ===")
        result = auto_git_push(project_dir, "test upload")
        for k, v in result.items():
            print(f"  {k}: {v}")

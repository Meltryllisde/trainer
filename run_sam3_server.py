#!/usr/bin/env python
"""
启动SAM3 HTTP服务器的便捷脚本
在训练前，先执行此脚本启动SAM3服务器
"""
import sys
import subprocess
import time
import requests
from pathlib import Path

def wait_for_server(url="http://127.0.0.1:5000/health", timeout=600000000000000):
    """等待服务器启动"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=60000)
            if response.status_code == 200:
                print("✓ SAM3服务器已启动！")
                return True
        except:
            pass
        time.sleep(2)
    print("✗ 服务器启动超时")
    return False


if __name__ == '__main__':
    print("=" * 60)
    print("启动 SAM3 HTTP 服务器")
    print("=" * 60)
    
    # 获取脚本所在目录
    script_dir = Path(__file__).parent
    server_script = script_dir / "sam3_http_server.py"
    
    if not server_script.exists():
        print(f"✗ 找不到 {server_script}")
        sys.exit(1)
    
    print(f"✓ 找到服务器脚本: {server_script}")
    print()
    
    # 启动服务器
    print("启动服务器进程...")
    process = subprocess.Popen(
        [sys.executable, str(server_script)],
        cwd=str(script_dir)
    )
    
    print("等待服务器启动...")
    if wait_for_server():
        print()
        print("=" * 60)
        print("SAM3服务器已成功启动！")
        print("=" * 60)
        print()
        print("服务器地址: http://127.0.0.1:5000/process")
        print("API端点:")
        print("  - POST /process    - 计算两张图像的IoU")
        print("  - POST /segment    - 分割单张图像")
        print("  - GET  /health     - 健康检查")
        print()
        print("现在可以运行训练脚本了！")
        print()
        
        # 保持进程运行
        try:
            process.wait()
        except KeyboardInterrupt:
            print("\n正在关闭服务器...")
            process.terminate()
            process.wait()
            print("服务器已关闭")
    else:
        print("✗ 服务器启动失败")
        process.terminate()
        sys.exit(1)

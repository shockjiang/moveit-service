#!/usr/bin/env python3
"""
HTTP服务测试脚本
用于测试MoveIt HTTP服务的各个接口
"""

import requests
import json
import sys
import time

BASE_URL = "http://localhost:8000"

def test_health():
    """测试健康检查接口"""
    print("\n=== 测试健康检查接口 ===")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"状态码: {response.status_code}")
        print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
        return response.status_code == 200
    except Exception as e:
        print(f"错误: {e}")
        return False

def test_grasp():
    """测试抓取接口"""
    print("\n=== 测试抓取接口 ===")

    data = {
        "robot_name": "xarm7",
        "target_object_index": 0,
        "depth_path": "test_data/grasp-wrist-dpt_opt.png",
        "seg_json_path": "test_data/rgb_detection_wrist.json",
        "affordance_path": "test_data/affordance.json"
    }

    print(f"请求数据: {json.dumps(data, indent=2, ensure_ascii=False)}")
    print("发送请求...")

    try:
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/grasp",
            json=data,
            timeout=300  # 5分钟超时
        )
        elapsed = time.time() - start_time

        print(f"\n状态码: {response.status_code}")
        print(f"耗时: {elapsed:.2f}秒")
        print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")

        result = response.json()
        return result.get("success", False)

    except requests.exceptions.Timeout:
        print("错误: 请求超时（超过5分钟）")
        return False
    except Exception as e:
        print(f"错误: {e}")
        return False

def main():
    """主函数"""
    print("=" * 50)
    print("MoveIt HTTP服务测试")
    print("=" * 50)

    # 测试健康检查
    health_ok = test_health()
    if not health_ok:
        print("\n❌ 健康检查失败，请确保服务已启动")
        print("启动命令: ./start_service.sh")
        return 1

    print("\n✓ 健康检查通过")

    # 如果指定了full参数，则运行完整测试
    if len(sys.argv) > 1 and sys.argv[1] == "full":
        print("\n运行完整测试（包括抓取任务）...")
        grasp_ok = test_grasp()

        if grasp_ok:
            print("\n✓✓✓ 所有测试通过 ✓✓✓")
            return 0
        else:
            print("\n❌ 抓取测试失败")
            return 1
    else:
        print("\n提示: 运行 'python3 test_service.py full' 执行完整测试")
        return 0

if __name__ == "__main__":
    sys.exit(main())

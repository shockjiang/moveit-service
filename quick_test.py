#!/usr/bin/env python3
"""快速抓取规划测试"""
import requests
import json
import time
import os

# 禁用代理，避免干扰本地连接
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''

BASE_URL = "http://localhost:8000"

def quick_grasp_test():
    """快速测试单个抓取候选点"""
    print("=" * 50)
    print("快速抓取规划测试")
    print("=" * 50)

    # 1. 健康检查
    print("\n=== 1. 健康检查 ===")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"✓ 服务健康: {response.json()}")
    except Exception as e:
        print(f"✗ 服务未响应: {e}")
        return

    # 2. 发送抓取请求
    print("\n=== 2. 开始抓取规划 ===")
    data = {
        "depth_path": "test_data/grasp-wrist-dpt_opt.png",
        "seg_json_path": "test_data/rgb_detection_wrist.json",
        "affordance_path": "test_data/affordance.json",
        "target_object_index": 3,  # 只测试第一个物体
        "plan_only": True  # 只规划不执行
    }

    print(f"目标物体索引: {data['target_object_index']}")
    print(f"规划模式: 只规划不执行")
    print(f"\n⏳ 正在规划... (最多可能需要5-25分钟)")
    print("   提示: 你可以在服务端终端查看实时进度")

    start_time = time.time()

    try:
        response = requests.post(
            f"{BASE_URL}/grasp",
            json=data,
            timeout=1800  # 30分钟超时
        )

        elapsed = time.time() - start_time

        print(f"\n=== 3. 结果 ===")
        print(f"耗时: {elapsed:.2f}秒 ({elapsed/60:.1f}分钟)")
        print(f"状态码: {response.status_code}")

        result = response.json()
        print(f"\n响应内容:")
        print(json.dumps(result, indent=2, ensure_ascii=False))

        if result.get("success"):
            print("\n✓✓✓ 规划成功！✓✓✓")
            if "results" in result and result["results"]:
                for r in result["results"]:
                    print(f"\n物体: {r.get('instance_id')}")
                    print(f"  规划时间: {r.get('planning_time', 0):.2f}秒")
                    print(f"  成功: {r.get('success')}")
        else:
            print(f"\n✗ 规划失败: {result.get('message')}")

    except requests.exceptions.Timeout:
        print("\n✗ 请求超时（超过30分钟）")
    except Exception as e:
        print(f"\n✗ 错误: {e}")

if __name__ == "__main__":
    quick_grasp_test()

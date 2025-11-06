"""
测试Web应用连接
用于检查Flask应用是否正常运行
"""
import requests
import sys

def test_connection(base_url='http://127.0.0.1:5000'):
    """测试Web应用连接"""
    print("=" * 60)
    print("测试 Web 应用连接")
    print("=" * 60)
    
    endpoints = [
        ('/api/health', '健康检查'),
        ('/api/analyzer/info', '分析器信息'),
        ('/api/analyze_stock', '分析股票页面（GET）'),
    ]
    
    success_count = 0
    fail_count = 0
    
    for endpoint, name in endpoints:
        try:
            url = f"{base_url}{endpoint}"
            print(f"\n测试: {name}")
            print(f"  URL: {url}")
            
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                print(f"  ✓ 成功 (状态码: {response.status_code})")
                if endpoint == '/api/health':
                    data = response.json()
                    print(f"    状态: {data.get('status', 'unknown')}")
                elif endpoint == '/api/analyzer/info':
                    data = response.json()
                    if data.get('success'):
                        print(f"    组件数量: {len(data.get('components', {}))}")
                success_count += 1
            else:
                print(f"  ✗ 失败 (状态码: {response.status_code})")
                fail_count += 1
                
        except requests.exceptions.ConnectionError:
            print(f"  ✗ 连接失败 - 服务器可能未运行")
            print(f"    请确保 Flask 应用正在运行:")
            print(f"    python apps/run_web.py")
            fail_count += 1
        except requests.exceptions.Timeout:
            print(f"  ✗ 请求超时")
            fail_count += 1
        except Exception as e:
            print(f"  ✗ 错误: {str(e)}")
            fail_count += 1
    
    print("\n" + "=" * 60)
    print(f"测试结果: 成功 {success_count}, 失败 {fail_count}")
    print("=" * 60)
    
    if fail_count == 0:
        print("\n✓ 所有测试通过！Web 应用运行正常。")
        return True
    else:
        print("\n✗ 部分测试失败。请检查 Flask 应用是否正在运行。")
        print("\n启动方法:")
        print("  cd C:\\Users\\111\\Desktop\\Reflexivity")
        print("  python apps/run_web.py")
        return False

if __name__ == '__main__':
    base_url = sys.argv[1] if len(sys.argv) > 1 else 'http://127.0.0.1:5000'
    test_connection(base_url)



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的GFW API Key获取脚本
目的：使用已有的access_token获取长期有效的API Key
作者：锐多宝 (ruiduobao)
"""

import requests
import json
from datetime import datetime

# --- 用户配置参数 ---
# 从第一步获取的access_token（从JSON响应中提取）
ACCESS_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjY4Y2QwNDk5YzBiZWJiNTY1NWRlNmI5YiIsInJvbGUiOiJVU0VSIiwicHJvdmlkZXIiOiJsb2NhbCIsImVtYWlsIjoiMTU4ODQ0MTE3MjRAMTYzLmNvbSIsImV4dHJhVXNlckRhdGEiOnsiYXBwcyI6WyJnZnciXX0sImNyZWF0ZWRBdCI6MTc1ODI2OTMzMjcyMCwiaWF0IjoxNzU4MjY5MzMyfQ.ThD0oTAd1k0O3nCUYCHBE2XWwB6alcGsO1gYVx26_GM"

# API Key配置信息
EMAIL = "15884411724@163.com"  # 您的邮箱
ORGANIZATION = "Research"       # 您的组织名称
API_KEY_ALIAS = f"my_api_key_{datetime.now().strftime('%Y%m%d_%H%M%S')}"  # API Key别名

def get_api_key():
    """
    使用access_token获取API Key
    """
    print("=" * 50)
    print("开始获取GFW API Key...")
    print("=" * 50)
    
    # API端点
    url = "https://data-api.globalforestwatch.org/auth/apikey"
    
    # 请求头 - 使用Bearer认证
    headers = {
        'Authorization': f'Bearer {ACCESS_TOKEN}',
        'Content-Type': 'application/json'
    }
    
    # 请求体数据
    data = {
        "alias": API_KEY_ALIAS,
        "email": EMAIL,
        "organization": ORGANIZATION,
        "domains": []  # 空数组表示不限制域名（但会有最低速率限制）
    }
    
    print(f"请求URL: {url}")
    print(f"API Key别名: {API_KEY_ALIAS}")
    print(f"邮箱: {EMAIL}")
    print(f"组织: {ORGANIZATION}")
    print()
    
    try:
        # 发送POST请求
        print("正在发送请求...")
        response = requests.post(url, headers=headers, json=data, timeout=30)
        
        print(f"响应状态码: {response.status_code}")
        
        if response.status_code in [200, 201]:
            # 解析响应
            response_data = response.json()
            print("✅ 请求成功!")
            print(f"完整响应: {json.dumps(response_data, indent=2, ensure_ascii=False)}")
            
            # 提取API Key
            if 'data' in response_data:
                # 检查data是字典还是列表
                data_obj = response_data['data']
                if isinstance(data_obj, dict):
                    # data是字典格式
                    api_key = data_obj.get('api_key')
                    expires_on = data_obj.get('expires_on')
                elif isinstance(data_obj, list) and len(data_obj) > 0:
                    # data是列表格式
                    api_key = data_obj[0].get('api_key')
                    expires_on = data_obj[0].get('expires_on')
                else:
                    api_key = None
                    expires_on = None
                
                if api_key:
                    print("\n" + "=" * 60)
                    print("🎉 API Key获取成功!")
                    print("=" * 60)
                    print(f"API Key: {api_key}")
                    print(f"过期时间: {expires_on}")
                    print("=" * 60)
                    
                    # 保存到文件
                    filename = f"gfw_api_key_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(f"GFW API Key\n")
                        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"API Key: {api_key}\n")
                        f.write(f"别名: {API_KEY_ALIAS}\n")
                        f.write(f"邮箱: {EMAIL}\n")
                        f.write(f"组织: {ORGANIZATION}\n")
                        f.write(f"过期时间: {expires_on}\n")
                    
                    print(f"\n✅ API Key已保存到文件: {filename}")
                    return api_key
                else:
                    print("❌ 响应中未找到api_key字段")
                    return None
            else:
                print("❌ 响应格式异常，未找到data字段")
                return None
        else:
            print(f"❌ 请求失败，状态码: {response.status_code}")
            print(f"错误响应: {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"❌ 网络请求异常: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ JSON解析异常: {e}")
        return None
    except Exception as e:
        print(f"❌ 未知异常: {e}")
        return None

def main():
    """
    主函数
    """
    print("GFW API Key获取工具")
    print(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 获取API Key
    api_key = get_api_key()
    
    if api_key:
        print("\n🎯 使用建议:")
        print("1. 请妥善保管您的API Key，不要泄露给他人")
        print("2. API Key有效期为1年，到期前请重新获取")
        print("3. 可以将API Key复制到您的数据下载脚本中使用")
        print("4. 如需更高的速率限制，请联系GFW技术支持")
    else:
        print("\n❌ API Key获取失败")
        print("💡 可能的解决方案:")
        print("1. 检查access_token是否有效（可能已过期）")
        print("2. 检查网络连接")
        print("3. 重新获取access_token后再试")

if __name__ == "__main__":
    main()
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试临时文件清理功能
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# 添加当前目录到路径，以便导入主模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入主模块中的函数
try:
    from importlib import import_module
    main_module = import_module('1.GEE分类结果合并批量.py')
    force_remove_directory = main_module.force_remove_directory
    print("✅ 成功导入 force_remove_directory 函数")
except Exception as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

def test_force_remove_directory():
    """测试强制删除目录功能"""
    print("\n🧪 测试强制删除目录功能...")
    
    # 创建测试目录结构
    test_base = tempfile.mkdtemp(prefix="test_cleanup_")
    print(f"创建测试目录: {test_base}")
    
    try:
        # 创建嵌套目录和文件
        nested_dir = os.path.join(test_base, "temp_batch", "subdir")
        os.makedirs(nested_dir, exist_ok=True)
        
        # 创建一些测试文件
        test_files = [
            os.path.join(test_base, "temp_batch", "file1.txt"),
            os.path.join(test_base, "temp_batch", "file2.txt"),
            os.path.join(nested_dir, "nested_file.txt")
        ]
        
        for file_path in test_files:
            with open(file_path, 'w') as f:
                f.write("test content")
        
        print(f"创建了 {len(test_files)} 个测试文件")
        
        # 测试删除功能
        temp_batch_dir = os.path.join(test_base, "temp_batch")
        print(f"尝试删除目录: {temp_batch_dir}")
        
        result = force_remove_directory(temp_batch_dir)
        
        if result:
            print("✅ 目录删除成功")
            if not os.path.exists(temp_batch_dir):
                print("✅ 验证：目录确实已被删除")
            else:
                print("❌ 验证失败：目录仍然存在")
        else:
            print("❌ 目录删除失败")
            
    except Exception as e:
        print(f"❌ 测试过程中出错: {e}")
    
    finally:
        # 清理测试目录
        try:
            if os.path.exists(test_base):
                shutil.rmtree(test_base)
                print(f"清理测试目录: {test_base}")
        except Exception as e:
            print(f"清理测试目录失败: {e}")

def test_empty_directory():
    """测试删除不存在的目录"""
    print("\n🧪 测试删除不存在的目录...")
    
    non_existent_dir = "/path/that/does/not/exist"
    result = force_remove_directory(non_existent_dir)
    
    if result:
        print("✅ 正确处理了不存在的目录")
    else:
        print("❌ 处理不存在目录时出错")

if __name__ == "__main__":
    print("🚀 开始测试临时文件清理功能")
    
    test_force_remove_directory()
    test_empty_directory()
    
    print("\n✅ 测试完成")
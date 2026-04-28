"""
目的：
读取包含“国家名称-年份-各类别像素数”的CSV，对目标的11个东南亚国家在每一个年份进行求和，
得到“每年（如2017、2024）的人工林/自然林/其他/无效像素”的总值，并输出新的年度汇总CSV。
"""  # 顶部说明：脚本用途

import os  # 引入os模块用于路径处理
import csv  # 引入csv模块用于读取/写入CSV文件
from collections import defaultdict  # 引入defaultdict便于按年份累计
from datetime import datetime  # 引入datetime用于记录处理时间

# 输入CSV路径（包含各国家逐年像素统计）
CSV_PATH = r"F:\地理所\论文\东南亚10m人工林提取\数据\正式分类_10.29\6.GEE导出结果_结果合并_马尔可夫模型_逐年合并_裁剪东南亚_转为等面积投影并裁剪每个国家\面积统计结果\southeast_asia_forest_pixels_2017_2024.csv"  # 原始CSV文件路径

# 输出CSV路径（年度总和结果）
OUT_PATH = os.path.join(os.path.dirname(CSV_PATH), "southeast_asia_forest_pixels_year_totals_2017_2024.csv")  # 输出文件路径

# 目标的11个国家（英文名称与CSV一致）
TARGET_COUNTRIES = {  # 用集合便于快速判断是否在目标国家内
    "Brunei", "Cambodia", "Indonesia", "Laos", "Malaysia",
    "Myanmar", "Philippines", "Singapore", "Thailand", "Timor-Leste", "Vietnam"
}

def to_int(val):  # 安全转换为整数的辅助函数
    """将可能为None/空串的值安全转换为int，失败则返回0。"""
    try:  # 尝试转换
        return int(val) if val not in (None, "") else 0  # 正常返回整数或0
    except Exception:  # 转换失败
        return 0  # 返回0，不中断统计

def read_and_sum_by_year(csv_path: str, target_countries=None):  # 读取并按年份汇总（兼容旧版Python，省略联合类型注解）
    sums = defaultdict(lambda: {"plantation": 0, "natural": 0, "others": 0, "invalid": 0, "versions": set()})  # 初始化按年份的累计容器
    with open(csv_path, "r", encoding="utf-8") as f:  # 打开输入CSV（UTF-8）
        reader = csv.DictReader(f)  # 用字典读取器按表头读行
        for row in reader:  # 遍历每一行记录
            country = (row.get("国家名称") or row.get("country") or "").strip()  # 读取国家名称并去除空白
            if target_countries and country not in target_countries:  # 如果设置了目标国家，则过滤非目标国家
                continue  # 跳过非目标国家记录

            y = row.get("年份") or row.get("year")  # 获取年份字段（兼容中英文表头）
            if not y:  # 若缺少年份
                continue  # 跳过该行
            try:  # 尝试将年份转为整数
                yr = int(y)  # 转换成功的年份整数
            except Exception:  # 转换失败
                continue  # 跳过该行

            # 累加各类别像素数（安全转换，空值按0处理）
            sums[yr]["plantation"] += to_int(row.get("人工林像素数"))  # 人工林像素累加
            sums[yr]["natural"] += to_int(row.get("自然林像素数"))  # 自然林像素累加
            sums[yr]["others"] += to_int(row.get("其他像素数"))  # 其他像素累加
            sums[yr]["invalid"] += to_int(row.get("无效像素数"))  # 无效像素累加

            # 收集该年份涉及到的“数据版本”（可能有多个国家版本不同）
            ver = (row.get("数据版本") or row.get("version") or "").strip()  # 读取数据版本
            if ver:  # 若有值
                sums[yr]["versions"].add(ver)  # 加入集合以去重
    return sums  # 返回年度累计结果

def write_year_totals(sums: dict, out_path: str):  # 写出年度汇总CSV
    cols = ["年份", "人工林像素数", "自然林像素数", "其他像素数", "无效像素数", "处理时间", "数据版本"]  # 输出表头（处理时间与输入保持一致用词）
    now = datetime.now().isoformat(timespec="seconds")  # 当前时间（到秒）
    years_sorted = sorted(sums.keys())  # 对年份排序，保证输出有序
    with open(out_path, "w", newline="", encoding="utf-8") as f:  # 打开输出CSV文件
        w = csv.writer(f)  # CSV写入器
        w.writerow(cols)  # 写入表头
        for yr in years_sorted:  # 遍历每个年份
            info = sums[yr]  # 取该年份的统计信息
            version = ",".join(sorted(info["versions"])) if info["versions"] else ""  # 合并版本集合为逗号串
            w.writerow([  # 写入一行年度总和
                yr,  # 年份
                info["plantation"],  # 人工林像素总数
                info["natural"],  # 自然林像素总数
                info["others"],  # 其他像素总数
                info["invalid"],  # 无效像素总数
                now,  # 处理时间（生成时间）
                version  # 数据版本（多个用逗号分隔）
            ])

def main():  # 主函数入口
    sums = read_and_sum_by_year(CSV_PATH, TARGET_COUNTRIES)  # 读取并按年份对11国进行汇总
    write_year_totals(sums, OUT_PATH)  # 写出年度汇总CSV
    print(f"汇总完成：{OUT_PATH}")  # 打印完成信息（便于在终端查看）

if __name__ == "__main__":  # 脚本入口判断
    main()  # 执行主函数
# 脚本目的：汇总年度精度指标并制图输出，包括总体精度折线图、各类用户精度与生产者精度折线图；同时生成年度汇总表（每年三类的用户精度、生产者精度、总体精度），基于前一脚本的结果数据进行可视化与统计输出。

# 说明：本脚本自动在已知目录中查找上一阶段生成的年度指标CSV（year_{year}_metrics.csv），
# 读取2017-2024各年的数据并汇总为宽表；生成PNG图保存到指定figures目录，同时日志写入文件与终端。

import os  # 导入os用于路径与文件操作
import sys  # 导入sys用于日志输出到控制台
import logging  # 导入logging用于统一日志记录
from datetime import datetime  # 导入datetime用于日志文件名的时间戳
import pandas as pd  # 导入pandas用于表格数据处理

try:
    import matplotlib.pyplot as plt  # 尝试导入matplotlib用于绘图
    HAS_MPL = True  # 标记matplotlib可用
except Exception:
    HAS_MPL = False  # 若导入失败，标记不可用

try:
    import seaborn as sns  # 尝试导入seaborn用于更美观的绘图
    HAS_SNS = True  # 标记seaborn可用
except Exception:
    HAS_SNS = False  # 若导入失败，标记不可用


# ============================ 路径配置 ============================
# 候选输入根目录（上一脚本可能输出到不同盘符），按顺序检测第一个存在的目录作为输入来源
INPUT_ROOT_CANDIDATES = [
  r"D:\地理所\论文\东南亚10m人工林提取\数据\测试集\precision_results_HMM_backward",
]  # 定义可能的输入precision_results根目录列表

# 固定输出图件目录（用户指定为K盘的figures目录）
FIGURE_DIR = r"D:\地理所\论文\东南亚10m人工林提取\数据\测试集\precision_results_HMM_backward\figures"  # 图件输出目录

# 年度指标所在的子目录名（在输入precision_results下的summary）
SUMMARY_SUBDIR = "summary"  # 定义年度指标子目录名

# 年份范围（2017-2024）
YEARS = list(range(2017, 2024 + 1))  # 定义需要处理的年份范围


# ============================ 日志初始化 ============================
def setup_logging(log_base_dir: str) -> str:
    """初始化日志记录到文件与控制台，并返回日志文件路径。"""  # 函数注释
    os.makedirs(log_base_dir, exist_ok=True)  # 创建日志基目录（若不存在）
    log_filename = f"2_统计精度并出图_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"  # 生成日志文件名
    log_path = os.path.join(log_base_dir, log_filename)  # 拼接完整日志路径
    logging.basicConfig(  # 配置日志输出
        level=logging.INFO,  # 设置日志级别为INFO
        format="%(asctime)s - %(levelname)s - %(message)s",  # 设置日志格式
        handlers=[  # 定义输出到文件与控制台的处理器
            logging.FileHandler(log_path, encoding="utf-8"),  # 文件处理器
            logging.StreamHandler(sys.stdout),  # 控制台处理器
        ],
    )  # 结束basicConfig配置
    return log_path  # 返回日志文件路径


# ============================ 输入根目录解析 ============================
def resolve_input_root() -> str:
    """按候选列表选择存在的precision_results根目录。"""  # 函数注释
    for root in INPUT_ROOT_CANDIDATES:  # 遍历候选根目录
        if os.path.exists(root):  # 若该目录存在
            summary_dir = os.path.join(root, SUMMARY_SUBDIR)  # 拼接summary子目录
            if os.path.exists(summary_dir):  # 若summary存在
                return root  # 返回该根目录
    return INPUT_ROOT_CANDIDATES[0]  # 若都不存在，回退到首选（后续会提示无数据）


# ============================ 读取年度指标 ============================
def load_year_metrics(summary_dir: str) -> dict:
    """读取各年的年度指标CSV，返回字典year->DataFrame。"""  # 函数注释
    data = {}  # 初始化字典
    for y in YEARS:  # 遍历年份
        fp = os.path.join(summary_dir, f"year_{y}_metrics.csv")  # 构造年度指标CSV路径
        if os.path.exists(fp):  # 若文件存在
            try:
                df = pd.read_csv(fp)  # 读取CSV为DataFrame
                data[y] = df  # 存入字典
            except Exception as e:  # 捕获读取异常
                logging.warning(f"读取失败：{fp}，原因：{e}")  # 写入警告日志
        else:  # 文件不存在
            logging.warning(f"缺少年度指标文件：{fp}")  # 写入警告日志
    return data  # 返回读取到的年度数据字典


# ============================ 汇总为宽表 ============================
def build_wide_summary(year_df_map: dict) -> pd.DataFrame:
    """将每年的指标表转换为宽表，含三类的用户/生产者精度与总体精度。"""  # 函数注释
    rows = []  # 初始化行记录列表
    for y, df in year_df_map.items():  # 遍历年份与对应DataFrame
        rec = {"year": y}  # 初始化该年的记录字典
        if "overall_accuracy" in df.columns and len(df) > 0:  # 检查列存在
            rec["overall_accuracy"] = float(df["overall_accuracy"].iloc[0])  # 总体精度（各类相同，取第一行）
        else:
            rec["overall_accuracy"] = None  # 若缺失则置空
        for cls in [1, 2, 3]:  # 遍历类别1/2/3
            sub = df[df["class"] == cls]  # 过滤该类别的行
            if not sub.empty:  # 若存在
                rec[f"user_accuracy_{cls}"] = float(sub["user_accuracy"].iloc[0])  # 记录用户精度
                rec[f"producer_accuracy_{cls}"] = float(sub["producer_accuracy"].iloc[0])  # 记录生产者精度
            else:  # 若不存在
                rec[f"user_accuracy_{cls}"] = None  # 置空
                rec[f"producer_accuracy_{cls}"] = None  # 置空
        rows.append(rec)  # 追加该年记录
    wide_df = pd.DataFrame(rows).sort_values("year")  # 构建DataFrame并按年排序
    return wide_df  # 返回宽表


# ============================ 制图函数 ============================
def plot_metrics(wide_df: pd.DataFrame, out_dir: str) -> None:
    """基于宽表绘制总体精度与各类用户/生产者精度的年度折线图。"""  # 函数注释
    os.makedirs(out_dir, exist_ok=True)  # 创建输出图件目录
    if not HAS_MPL:  # 若matplotlib不可用
        logging.error("matplotlib未安装，无法制图。请先安装matplotlib")  # 记录错误
        return  # 直接返回
    if HAS_SNS:  # 若seaborn可用
        sns.set(style="whitegrid")  # 设置白底网格风格

    # 总体精度折线图
    plt.figure(figsize=(10, 5))  # 创建画布并设置尺寸
    plt.plot(wide_df["year"], wide_df["overall_accuracy"], marker="o", label="总体精度")  # 绘制总体精度折线
    plt.ylim(0, 1)  # 设定y轴范围0-1
    plt.xlabel("年份")  # 设置x轴标签
    plt.ylabel("精度")  # 设置y轴标签
    plt.title("总体精度（2017-2024）")  # 设置图标题
    plt.legend()  # 显示图例
    out_fp = os.path.join(out_dir, "overall_accuracy_by_year.png")  # 输出文件路径
    plt.savefig(out_fp, dpi=300, bbox_inches="tight")  # 保存图像为PNG
    plt.close()  # 关闭画布释放内存
    logging.info(f"已输出图件：{out_fp}")  # 记录输出日志

    # 各类用户精度折线图
    plt.figure(figsize=(10, 5))  # 创建画布
    for cls, color in zip([1, 2, 3], ["tab:blue", "tab:orange", "tab:green"]):  # 遍历类别与颜色
        plt.plot(wide_df["year"], wide_df[f"user_accuracy_{cls}"], marker="o", label=f"用户精度-类{cls}", color=color)  # 绘制折线
    plt.ylim(0, 1)  # 设置y轴范围
    plt.xlabel("年份")  # 设置x轴标签
    plt.ylabel("精度")  # 设置y轴标签
    plt.title("用户精度（2017-2024，三类）")  # 设置图标题
    plt.legend()  # 显示图例
    out_fp = os.path.join(out_dir, "user_accuracy_by_year.png")  # 输出文件路径
    plt.savefig(out_fp, dpi=300, bbox_inches="tight")  # 保存图像
    plt.close()  # 关闭画布
    logging.info(f"已输出图件：{out_fp}")  # 记录输出日志

    # 各类生产者精度折线图
    plt.figure(figsize=(10, 5))  # 创建画布
    for cls, color in zip([1, 2, 3], ["tab:blue", "tab:orange", "tab:green"]):  # 遍历类别与颜色
        plt.plot(wide_df["year"], wide_df[f"producer_accuracy_{cls}"], marker="o", label=f"生产者精度-类{cls}", color=color)  # 绘制折线
    plt.ylim(0, 1)  # 设置y轴范围
    plt.xlabel("年份")  # 设置x轴标签
    plt.ylabel("精度")  # 设置y轴标签
    plt.title("生产者精度（2017-2024，三类）")  # 设置图标题
    plt.legend()  # 显示图例
    out_fp = os.path.join(out_dir, "producer_accuracy_by_year.png")  # 输出文件路径
    plt.savefig(out_fp, dpi=300, bbox_inches="tight")  # 保存图像
    plt.close()  # 关闭画布
    logging.info(f"已输出图件：{out_fp}")  # 记录输出日志


# ============================ 主流程 ============================
def main():
    """主入口：解析输入目录、读取年度指标、生成宽表并制图与输出CSV。"""  # 函数注释
    os.makedirs(FIGURE_DIR, exist_ok=True)  # 创建图件输出目录
    log_dir = os.path.join(FIGURE_DIR, "logs")  # 定义日志目录路径
    log_path = setup_logging(log_dir)  # 初始化日志并获取日志文件路径
    logging.info(f"日志文件：{log_path}")  # 输出日志文件路径

    input_root = resolve_input_root()  # 解析输入precision_results根目录
    summary_dir = os.path.join(input_root, SUMMARY_SUBDIR)  # 拼接年度指标目录
    logging.info(f"输入数据目录：{summary_dir}")  # 记录输入目录

    year_map = load_year_metrics(summary_dir)  # 读取年度指标文件
    if len(year_map) == 0:  # 若没有任何年度数据
        logging.error("未找到年度指标CSV，请先运行前一脚本生成数据")  # 记录错误日志
        return  # 退出主流程

    wide_df = build_wide_summary(year_map)  # 构建年度宽表
    # 将宽表保存到summary目录下
    wide_out = os.path.join(summary_dir, "year_metrics_summary_wide.csv")  # 定义宽表输出路径
    wide_df.to_csv(wide_out, index=False, encoding="utf-8-sig")  # 保存宽表CSV
    logging.info(f"已输出年度汇总表：{wide_out}")  # 记录输出日志

    # 制图输出到FIGURE_DIR
    plot_metrics(wide_df, FIGURE_DIR)  # 调用制图函数

    logging.info("全部完成")  # 记录完成日志


if __name__ == "__main__":  # 判断脚本是否作为主程序运行
    main()  # 调用主函数


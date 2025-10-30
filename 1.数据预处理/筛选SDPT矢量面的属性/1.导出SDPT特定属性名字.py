"""
脚本目的：读取SDPT矢量GeoPackage，统计'common_nam'和'species_si'两个属性的唯一值及其计数，同时在这些唯一值中查找包含'槟榔'的项；最终输出一个CSV到当前脚本所在文件夹，并将详细日志写入同一文件夹。
"""
import os  # 文件与路径处理
import sys  # 标准输出到控制台，用于日志同时打印
import logging  # 记录运行过程日志
from datetime import datetime  # 生成带时间戳的输出文件名
from collections import Counter  # 对属性值进行计数统计
import fiona  # 流式读取GeoPackage，避免一次性加载大数据
import pandas as pd  # 将统计结果构建为表格并导出为CSV

sdpt_path = r"D:\地理所\论文\东南亚10m人工林提取\数据\SDPT_2.0_东南亚\按国家筛选\sdpt_southeast_asia_20251006_181147.gpkg"  # SDPT数据路径（保持原有设置）


def setup_logging(output_dir: str) -> str:
    """配置日志记录到文件与控制台，并返回日志文件路径"""
    log_filename = f"sdpt_unique_values_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"  # 日志文件名（带时间戳）
    log_path = os.path.join(output_dir, log_filename)  # 日志文件完整路径
    logging.basicConfig(
        level=logging.INFO,  # 设置日志级别为INFO，便于查看关键过程
        format='%(asctime)s - %(levelname)s - %(message)s',  # 日志格式包含时间与等级
        handlers=[
            logging.FileHandler(log_path, encoding='utf-8'),  # 写入日志文件，编码utf-8避免中文乱码
            logging.StreamHandler(sys.stdout)  # 同步输出到控制台，方便实时查看
        ]
    )
    return log_path  # 返回日志文件路径，便于在结束时打印提示


def get_layer_name(gpkg_path: str) -> str:
    """获取GeoPackage的第一个图层名称，便于fiona按图层读取"""
    try:
        layers = fiona.listlayers(gpkg_path)  # 列出GeoPackage中的所有图层
        layer_name = layers[0] if layers else None  # 选择第一个图层作为默认读取对象
        if layer_name is None:
            logging.error("GeoPackage中未找到任何图层")  # 无图层时报错
        else:
            logging.info(f"检测到图层: {layer_name}")  # 记录检测到的图层名称
        return layer_name  # 返回图层名称
    except Exception as e:
        logging.exception(f"获取图层名称失败: {e}")  # 捕获异常并记录堆栈
        return None  # 返回None表示失败


def list_schema_properties(gpkg_path: str, layer_name: str):
    """列出GeoPackage图层的属性字段列表，便于核对字段名"""
    props = []  # 存储属性字段名
    try:
        with fiona.open(gpkg_path, layer=layer_name) as src:  # 打开指定图层
            schema_props = src.schema.get('properties', {})  # 获取schema中的properties字典
            props = list(schema_props.keys())  # 字段名列表
            logging.info(f"图层属性字段数量: {len(props)}")  # 记录字段数量
            for p in props:  # 逐个打印字段名
                logging.info(f"  - {p}")  # 打印字段
    except Exception as e:
        logging.exception(f"读取图层属性字段失败: {e}")  # 捕获异常并记录
    return props  # 返回字段列表


def collect_unique_values(gpkg_path: str, layer_name: str, attr_names: list):
    """流式遍历要素，收集指定属性的唯一值与计数"""
    counters = {attr: Counter() for attr in attr_names}  # 为每个属性准备一个计数器
    total_features = 0  # 统计总要素数量
    try:
        with fiona.open(gpkg_path, layer=layer_name) as src:  # 打开GeoPackage指定图层
            logging.info(f"开始遍历要素，共计 {len(src)} 条（可能为近似值）")  # 记录要素数量（fiona可能返回近似值）
            for feat in src:  # 逐条遍历要素（流式，节省内存）
                total_features += 1  # 累加要素计数
                props = feat.get('properties', {})  # 读取属性字典
                for attr in attr_names:  # 遍历目标属性集合
                    val = props.get(attr, None)  # 获取该属性的值
                    if val is None:  # 若为空则跳过
                        continue  # 跳过空值
                    v = val.strip() if isinstance(val, str) else str(val).strip()  # 统一为去空格的字符串
                    if v == '':  # 若为空字符串则跳过
                        continue  # 跳过空字符串
                    counters[attr][v] += 1  # 在对应属性的计数器中累加出现次数
        logging.info(f"遍历完成，总要素数: {total_features}")  # 完成遍历后记录总要素数
        return counters, total_features  # 返回计数器与总要素数
    except Exception as e:
        logging.exception(f"流式读取失败: {e}")  # 捕获异常并记录堆栈
        return None, total_features  # 返回None表示失败


def build_summary_dataframe(counters: dict, keyword: str = '槟榔') -> pd.DataFrame:
    """将计数器汇总为DataFrame，并标注是否包含指定关键词（槟榔）"""
    rows = []  # 用于存放每一行统计结果
    for attr, counter in counters.items():  # 遍历每个属性的计数器
        for v, cnt in counter.items():  # 遍历每个唯一值及其计数
            contains_keyword = keyword in str(v)  # 判断唯一值中是否包含关键词“槟榔”
            rows.append({  # 追加一行统计结果
                'attribute': attr,  # 属性名
                'value': v,  # 唯一值内容
                'count': cnt,  # 出现次数
                'contains_槟榔': '是' if contains_keyword else '否'  # 是否包含“槟榔”
            })
    df = pd.DataFrame(rows)  # 构建DataFrame表格
    if not df.empty:  # 若表非空则排序
        df.sort_values(by=['attribute', 'count'], ascending=[True, False], inplace=True)  # 按属性与计数排序
    return df  # 返回汇总表


def search_keyword_across_all_attributes(gpkg_path: str, layer_name: str, keyword: str = '槟榔'):
    """在所有属性中搜索包含指定关键词的值，返回按属性聚合的计数器"""
    attr_counters = {}  # 属性->Counter映射
    total_features = 0  # 统计要素数量（用于日志）
    try:
        with fiona.open(gpkg_path, layer=layer_name) as src:  # 打开图层
            logging.info(f"在所有属性中搜索关键词: {keyword}")  # 记录搜索关键词
            for feat in src:  # 遍历每个要素
                total_features += 1  # 累加要素计数
                props = feat.get('properties', {})  # 读取属性字典
                for attr, val in props.items():  # 遍历每个属性
                    if val is None:  # 为空跳过
                        continue  # 跳过空值
                    v = val.strip() if isinstance(val, str) else str(val).strip()  # 统一为去空格的字符串
                    if v == '':  # 空字符串跳过
                        continue  # 跳过空字符串
                    if keyword in v:  # 如果包含关键词
                        if attr not in attr_counters:  # 若该属性还未初始化计数器
                            attr_counters[attr] = Counter()  # 初始化计数器
                        attr_counters[attr][v] += 1  # 计数累加
        logging.info(f"关键词搜索完成，总要素数: {total_features}")  # 打印总要素数
    except Exception as e:
        logging.exception(f"关键词搜索失败: {e}")  # 捕获异常
        attr_counters = {}  # 返回空结果
    return attr_counters  # 返回属性计数器


def main():
    """主函数：执行读取、统计与结果输出"""
    output_dir = os.path.dirname(os.path.abspath(__file__))  # 将输出目录设置为当前脚本所在文件夹
    log_path = setup_logging(output_dir)  # 初始化日志系统并获取日志文件路径
    logging.info("==== 开始统计SDPT属性唯一值 ==== ")  # 记录流程开始
    logging.info(f"输入数据: {sdpt_path}")  # 打印输入数据路径
    logging.info(f"输出目录: {output_dir}")  # 打印输出目录路径

    if not os.path.exists(sdpt_path):  # 检查SDPT文件是否存在
        logging.error(f"SDPT文件不存在: {sdpt_path}")  # 若不存在则报错
        logging.info(f"日志文件: {log_path}")  # 打印日志文件位置
        return  # 结束程序

    attr_names = ['common_name', 'species_sis','species_simp']  # 目标属性列表，仅统计这两个字段的唯一值
    logging.info(f"目标属性: {attr_names}")  # 记录目标属性

    layer_name = get_layer_name(sdpt_path)  # 获取GeoPackage图层名称
    if layer_name is None:  # 若获取失败则退出
        logging.error("无法获取图层名称，程序终止")  # 打印错误
        logging.info(f"日志文件: {log_path}")  # 打印日志文件位置
        return  # 结束程序

    # 新增：打印schema字段列表，并提示缺失的目标字段
    schema_props = list_schema_properties(sdpt_path, layer_name)  # 获取schema字段列表
    missing_attrs = [a for a in attr_names if a not in schema_props]  # 统计缺失字段
    if missing_attrs:  # 若有缺失
        logging.warning(f"以下目标字段在schema中未发现: {missing_attrs}")  # 打印警告

    counters, total_features = collect_unique_values(sdpt_path, layer_name, attr_names)  # 流式统计唯一值
    if counters is None:  # 若统计失败则退出
        logging.error("统计唯一值失败，程序终止")  # 打印错误
        logging.info(f"日志文件: {log_path}")  # 打印日志文件位置
        return  # 结束程序

    df = build_summary_dataframe(counters, keyword='槟榔')  # 将计数结果构建为表格并标注是否包含“槟榔”
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')  # 生成时间戳，便于区分多次运行的输出文件
    csv_name = f"sdpt_unique_values_{ts}.csv"  # 输出CSV文件名
    csv_path = os.path.join(output_dir, csv_name)  # 拼接输出CSV完整路径

    df.to_csv(csv_path, index=False, encoding='utf-8-sig')  # 导出为CSV，utf-8-sig方便Excel识别中文

    # 在日志中打印简要统计信息，详细内容查看CSV
    logging.info(f"总要素数量: {total_features}")  # 打印总要素数量
    for attr in attr_names:  # 遍历两字段
        unique_count = len(counters[attr])  # 统计唯一值个数
        keyword_count = sum(cnt for v, cnt in counters[attr].items() if '槟榔' in str(v))  # 统计包含“槟榔”的记录数
        logging.info(f"字段 {attr} 的唯一值数量: {unique_count}")  # 打印唯一值数量
        logging.info(f"字段 {attr} 中包含“槟榔”的记录数: {keyword_count}")  # 打印包含槟榔的记录数

    # 新增：在所有属性中搜索包含“槟榔”的值，并输出另一份CSV
    keyword_counters = search_keyword_across_all_attributes(sdpt_path, layer_name, keyword='槟榔')  # 搜索关键词
    keyword_df = build_summary_dataframe(keyword_counters, keyword='槟榔')  # 构建结果表格
    keyword_csv_name = f"sdpt_keyword_matches_{ts}.csv"  # 关键词匹配结果CSV文件名
    keyword_csv_path = os.path.join(output_dir, keyword_csv_name)  # 拼接输出路径
    keyword_df.to_csv(keyword_csv_path, index=False, encoding='utf-8-sig')  # 导出关键词匹配结果

    # 打印每个字段的关键词匹配数量汇总
    for attr, counter in keyword_counters.items():  # 遍历属性计数器
        match_count = sum(counter.values())  # 计算该属性中匹配值出现总次数
        logging.info(f"字段 {attr} 中包含“槟榔”的记录总数: {match_count}")  # 打印统计

    logging.info("==== 统计完成 ==== ")  # 记录流程结束
    logging.info(f"CSV已输出: {csv_path}")  # 打印CSV路径
    logging.info(f"关键词匹配CSV已输出: {keyword_csv_path}")  # 打印关键词匹配CSV路径
    logging.info(f"日志文件: {log_path}")  # 打印日志路径


if __name__ == "__main__":  # 作为脚本执行时
    main()  # 调用主函数执行流程
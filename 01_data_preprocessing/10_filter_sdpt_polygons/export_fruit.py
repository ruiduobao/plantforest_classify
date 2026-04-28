#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_sdpt_fruit_features.py

目的：从 SDPT GeoPackage 中筛选出属性中包含 "fruit"（按单词边界匹配）的要素，
并导出为新的 GeoPackage（保留原 schema/CRS）以及一个统计 CSV。

使用：
    python export_sdpt_fruit_features.py

注意：
- 请确认 fiona, pandas 已安装。
- 修改 sdpt_path 为你的实际文件路径（脚本当前默认目录为输出目录）。
"""

import os
import sys
import logging
from datetime import datetime
import re
from collections import Counter

import fiona
import pandas as pd

# ---------------------- 配置区 ----------------------
sdpt_path = r"D:\地理所\论文\东南亚10m人工林提取\数据\SDPT_2.0_东南亚\按国家筛选\sdpt_southeast_asia_20251006_181147.gpkg"
# 如果你希望在更多字段中搜索，请在这里扩展
search_attrs = ['common_name', 'species', 'species_simp', 'species_sis']  # 加入可能存在的字段名（冗余也没关系）

# 匹配关键词（可以改成其他词或加入多个关键词）
keyword = 'fruit'
# 分隔符（遇到逗号/分号/竖线等时会尝试拆分再精确匹配）
split_seps = r'[,\;/\|]'
# ---------------------------------------------------

def setup_logging(output_dir: str):
    fname = f"export_fruit_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    log_path = os.path.join(output_dir, fname)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return log_path

def get_layer_name(gpkg_path: str):
    try:
        layers = fiona.listlayers(gpkg_path)
        if not layers:
            logging.error("GeoPackage 中没有检测到图层。")
            return None
        logging.info(f"检测到图层列表（取第一个为默认）：{layers[:5]}{'...' if len(layers)>5 else ''}")
        return layers[0]
    except Exception as e:
        logging.exception("获取图层名称失败")
        return None

def matches_keyword(value: str, kw_pattern, split_seps_pattern) -> bool:
    """两步判断：
       1) 先用正则单词边界快速判断（\bfruit\b）
       2) 若第一步为真，再按分隔符拆分逐项精确匹配，或仅以第一步结果为准也行（这里做双重保险）
    """
    if not value:
        return False
    # 统一处理为字符串
    s = str(value).strip()
    if s == '':
        return False
    # 快速全文匹配单词边界
    if kw_pattern.search(s):
        # 再按分隔符拆分，看看是否有词项等于 keyword（避免 fruit 在其他词内部的误匹配，但一般上面已够）
        parts = [p.strip().lower() for p in re.split(split_seps_pattern, s) if p.strip()!='']
        if any(part == keyword for part in parts):
            return True
        # 若拆分项没有严格等于 keyword，仍可接受如 "fruit mix" 这样的项（包含 keyword）
        if any(keyword in part for part in parts):
            return True
        # 退而求其次：如果全文匹配但没有分隔后的项可判为 True（例如单词在中间）
        return True
    return False

def export_fruit_features(gpkg_path: str, output_dir: str):
    layer_name = get_layer_name(gpkg_path)
    if layer_name is None:
        return None

    kw_re = re.compile(rf'\b{re.escape(keyword)}\b', flags=re.IGNORECASE)
    split_re = split_seps

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_gpkg = os.path.join(output_dir, f"sdpt_fruit_features_{ts}.gpkg")
    out_csv = os.path.join(output_dir, f"sdpt_fruit_features_stats_{ts}.csv")

    total = 0
    selected = 0
    # 用于统计不同属性值的计数（便于审查哪些值被认为是 fruit）
    value_counters = {attr: Counter() for attr in search_attrs}
    selected_records = []  # 保存少量属性用于 CSV（例如 id, common_name, species, selected_match_field）

    try:
        with fiona.open(gpkg_path, layer=layer_name) as src:
            meta = src.meta.copy()
            # 强制 driver 为 GPKG（fiona 写入 gpkg 需要 driver 明确）
            meta['driver'] = 'GPKG'
            # 写入新 gpkg 时指定一个新的 layer 名称，避免覆盖（fiona 会用 layer 名称作为 gpkg 中的一个 table）
            out_layer = f"sdpt_fruit_{ts}"
            # fiona.open 写入时可以通过 layer 参数指定 (some versions accept 'layer' in kwargs)
            # 为了兼容性，我们在打开写入时传入 layer=out_layer
            with fiona.open(out_gpkg, 'w', layer=out_layer, **meta) as dst:
                logging.info(f"开始遍历图层 {layer_name}，总要素（fiona 返回值，近似）: {len(src)}")
                for feat in src:
                    total += 1
                    props = feat.get('properties', {}) or {}
                    matched_any = False
                    matched_fields = []
                    # 检查每个目标字段
                    for attr in search_attrs:
                        val = props.get(attr, None)
                        if val is None:
                            continue
                        if matches_keyword(val, kw_re, split_re):
                            matched_any = True
                            matched_fields.append(attr)
                            value_counters[attr][str(val).strip()] += 1
                    if matched_any:
                        dst.write(feat)
                        selected += 1
                        # 记录少量字段到 CSV 便于审查
                        record = {
                            'fid_index': total,  # 文件顺序索引（若要用真实的 id，请替换为 feat.get('id') 或属性中的 id 字段）
                            'matched_fields': ';'.join(matched_fields)
                        }
                        # 把感兴趣的字段值也写出来（若不存在则写空）
                        for a in ['common_name', 'species', 'species_simp']:
                            record[a] = props.get(a, '')
                        selected_records.append(record)

        logging.info(f"遍历完成。共遍历 {total} 条要素，筛选出 {selected} 条匹配 '{keyword}' 的要素。")
        # 导出统计 CSV
        if selected_records:
            df = pd.DataFrame(selected_records)
            # 保存 selected features list
            df.to_csv(out_csv, index=False, encoding='utf-8-sig')
            logging.info(f"已导出所选要素属性列表到: {out_csv}")
        else:
            logging.info("未筛选出任何要素（selected_records 为空），未生成 CSV。")

        # 将 value_counters 汇总为 DataFrame 方便审查哪些具体值被匹配
        rows = []
        for attr, counter in value_counters.items():
            for v, cnt in counter.items():
                rows.append({'attribute': attr, 'value': v, 'count': cnt})
        if rows:
            df_vals = pd.DataFrame(rows)
            values_csv = os.path.join(output_dir, f"sdpt_fruit_value_counts_{ts}.csv")
            df_vals.sort_values(['attribute', 'count'], ascending=[True, False], inplace=True)
            df_vals.to_csv(values_csv, index=False, encoding='utf-8-sig')
            logging.info(f"已导出匹配到的具体属性值计数到: {values_csv}")
        else:
            logging.info("没有匹配到任何属性值，未生成 value counts CSV.")

        logging.info(f"导出 GeoPackage: {out_gpkg} (layer={out_layer})")
        return out_gpkg

    except Exception as e:
        logging.exception("导出过程中出现错误")
        return None

def main():
    output_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = setup_logging(output_dir)
    logging.info("==== 开始：导出包含 'fruit' 的要素 ====")
    logging.info(f"输入文件: {sdpt_path}")
    logging.info(f"搜索字段: {search_attrs}")
    logging.info(f"匹配关键字（单词边界）: '{keyword}'")
    if not os.path.exists(sdpt_path):
        logging.error(f"输入文件不存在: {sdpt_path}")
        return
    out = export_fruit_features(sdpt_path, output_dir)
    if out:
        logging.info("==== 完成：导出成功 ====")
    else:
        logging.info("==== 完成：导出失败或无匹配 ====")
    logging.info(f"日志文件: {log_path}")

if __name__ == "__main__":
    main()

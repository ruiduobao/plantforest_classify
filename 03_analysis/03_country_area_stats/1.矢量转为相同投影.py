# 目的：将输入矢量southeast_asia_ADM_0.shp重投影为与栅格一致的自定义等面积投影，并输出新的Shapefile，带进度与日志
import os
import sys
import math
from datetime import datetime

# 解释：路径与目标投影定义（与栅格一致）
INPUT_SHP = r"F:\地理所\论文\东南亚10m人工林提取\数据\正式分类_10.29\6.GEE导出结果_结果合并_马尔可夫模型_逐年合并_裁剪东南亚_转为等面积投影并裁剪每个国家\矢量\southeast_asia_ADM_0.shp"
OUTPUT_DIR = r"F:\地理所\论文\东南亚10m人工林提取\数据\正式分类_10.29\6.GEE导出结果_结果合并_马尔可夫模型_逐年合并\矢量_equal_area"
OUTPUT_SHP = os.path.join(OUTPUT_DIR, "southeast_asia_ADM_0_aea.shp")
LOG_PATH = os.path.join(OUTPUT_DIR, "vector_reproject.log")
PROJ4 = "+proj=aea +lat_0=0 +lon_0=115 +lat_1=-5 +lat_2=15 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"

# 解释：库导入
try:
    import fiona
    from fiona import Env
    from rasterio.crs import CRS
    from rasterio.warp import transform_geom
except Exception as e:
    raise RuntimeError(f"缺少必要库，请安装：fiona、rasterio；错误：{e}")

def _ensure_dirs():
    # 解释：确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def _log(msg: str):
    # 解释：统一日志到控制台与文件
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass

def reproject_shapefile(in_path: str, out_path: str, target_proj4: str):
    # 解释：将输入Shapefile重投影到目标等面积投影并输出
    _ensure_dirs()
    target_crs = CRS.from_proj4(target_proj4)
    target_wkt = target_crs.to_wkt()
    _log("开始重投影矢量到等面积坐标系")
    with fiona.open(in_path, 'r') as src:
        src_crs_wkt = src.crs_wkt or src.crs
        schema = src.schema.copy()
        # 解释：创建输出Shapefile
        with fiona.open(out_path, 'w', driver=src.driver, crs_wkt=target_wkt, schema=schema, encoding=src.encoding or 'utf-8') as dst:
            total = len(src)
            if total == 0:
                _log("输入矢量为空，无要素")
                return
            step = max(1, total // 10)
            _log(f"要素总数：{total}")
            for idx, feat in enumerate(src, start=1):
                geom = feat.get('geometry')
                props = feat.get('properties')
                if not geom:
                    continue
                try:
                    # 解释：执行几何重投影
                    geom_out = transform_geom(src_crs_wkt, target_crs.to_string(), geom, precision=6)
                except Exception as e:
                    _log(f"要素{idx}重投影异常：{e}，跳过")
                    continue
                # 解释：写入要素
                dst.write({
                    'type': 'Feature',
                    'geometry': geom_out,
                    'properties': props
                })
                # 解释：进度输出
                if (idx % step == 0) or (idx == total):
                    pct = idx / total * 100.0
                    _log(f"进度：{idx}/{total} ({pct:.1f}%)")
    _log(f"重投影完成，输出：{out_path}")

def main():
    # 解释：支持命令行自定义输入/输出，默认用预设路径
    in_path = INPUT_SHP
    out_path = OUTPUT_SHP
    if len(sys.argv) >= 2:
        in_path = sys.argv[1]
    if len(sys.argv) >= 3:
        out_path = sys.argv[2]
    _log(f"输入：{in_path}")
    _log(f"输出：{out_path}")
    reproject_shapefile(in_path, out_path, PROJ4)

if __name__ == "__main__":
    main()

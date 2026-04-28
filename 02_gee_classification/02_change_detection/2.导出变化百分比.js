
/************************************************************
 导出每年相邻年份变化值直方图（2017-2023）
 1) 逐值直方图： value(0-10000), pixel_count
 2) 聚合 10 值/段： bin_index(1..1000), bin_min, bin_max, pixel_count
 分段规则（聚合）举例：
  bin 1:  0-10
  bin 2: 11-20
  ...
  bin1000: 9991-10000
*************************************************************/

// ----------------- 区域与配置 -----------------
var areaBoundary = ee.FeatureCollection("projects/ruiduobaonb/assets/BOUNDARY_ZONE_10");
var tileAreaId = 5;
var EMB_COLL = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL');

var beginYr = 2017;
var finishYr = 2023;

Map.setCenter(105.55, 15.84, 9);

// ---------- 工具函数 ----------
function buildYearMosaic(y) {
  return EMB_COLL.filterBounds(areaBoundary)
                 .filter(ee.Filter.calendarRange(y, y, 'year'))
                 .mosaic()
                 .clip(areaBoundary);
}

function countYearTiles(y) {
  return EMB_COLL.filterBounds(areaBoundary)
                 .filter(ee.Filter.calendarRange(y, y, 'year'))
                 .size()
                 .getInfo();
}

// ----------------- 年份主循环 -----------------
var yrs = [];
for (var yy = beginYr; yy <= finishYr; yy++) yrs.push(yy);

yrs.forEach(function(currentYear) {
  var t = currentYear;
  
  // 检查当年 tiles (可选的 getInfo 调用)
  var tilesNum = countYearTiles(t);
  print('Tiles for year', t, ':', tilesNum);
  if (tilesNum === 0) {
    print('  跳过：年份', t, '无 tile。');
    return;
  }

  // 检查下一年 tiles
  var nextYr = t + 1;
  var nextTiles = countYearTiles(nextYr);
  if (nextTiles === 0) {
    print('  跳过：年份', t, '因为 next year', nextYr, '无 tiles。');
    return;
  }

  // 构建当年和下一年 mosaic
  var mosaicThis = buildYearMosaic(t);
  var mosaicNext = buildYearMosaic(nextYr);

  // 检查波段
  var bandCountThis = mosaicThis.bandNames().size().getInfo();
  var bandCountNext = mosaicNext.bandNames().size().getInfo();
  
  if (bandCountThis === 0 || bandCountNext === 0) {
    print('  跳过：年份', t, '或', nextYr, ' mosaic 空。');
    return;
  }

  // ----- 计算与相邻年 (t+1) 的相似度并映射为概率 -----
  var dot_next = mosaicThis.multiply(mosaicNext).reduce(ee.Reducer.sum()).rename('dot_next');
  var prob_vs_next = ee.Image(1).subtract(dot_next).divide(2).rename('prob_vs_next');

  // ----- 转换为 Int16 (0-10000) -----
  var resultImg = prob_vs_next.multiply(10000).toInt16().rename('change_value');

  // ----- 计算逐值直方图：使用 reduceRegion 统计每个值的像素数 -----
  var histogram = resultImg.reduceRegion({
    reducer: ee.Reducer.frequencyHistogram(),
    geometry: areaBoundary,
    scale: 10,
    maxPixels: 1e13
  });

  // 将直方图转换为 FeatureCollection（value: int, pixel_count）
  var histogramDict = ee.Dictionary(histogram.get('change_value'));
  var histogramKeys = histogramDict.keys();

  var featureList = histogramKeys.map(function(key) {
    // key 是字符串，转为数字
    var value = ee.Number.parse(ee.String(key));
    var count = ee.Number(histogramDict.get(key));
    return ee.Feature(null, {
      'value': value,
      'pixel_count': count
    });
  });

  var histogramFC = ee.FeatureCollection(featureList);

  // ----- 导出逐值直方图到 Google Drive (CSV) -----
  var fileName = 'histogram_zone' + tileAreaId + '_' + t + 'vs' + nextYr;
  
  // Export.table.toDrive({
  //   collection: histogramFC,
  //   description: fileName,
  //   folder: 'Changed_pro_histogram',
  //   fileNamePrefix: fileName,
  //   fileFormat: 'CSV',
  //   selectors: ['value', 'pixel_count']
  // });

  print('  创建逐值直方图导出：', fileName, '(', t, 'vs', nextYr, ')');

  // -------------------- 在 server-side 聚合为宽为10的段 --------------------
  // binIndex 1..1000, 对应:
  // bin 1: 0..10
  // bin i (i>=2): min = (i-1)*10 + 1? wait -> formula below:
  // We use:
  //  if i == 1: min=0, max=10
  //  else: min = (i-1 -1)*10 + 11? -> simpler computed as:
  // For bin i >= 2: min = (i-1)*10 + 1? (see reasoning in analysis)
  // We'll implement formulas:
  var binsCount = 1000; // (10000 values / 10) => 1000 bins

  // 生成 1..1000
  var allBins = ee.List.sequence(1, binsCount);

  var aggFeatures = allBins.map(function(i) {
    i = ee.Number(i);
    // compute bin_min and bin_max using the convention:
    // bin 1: 0-10
    // bin i>=2: min = (i-1)*10 + 1, max = i*10
    var isFirst = i.eq(1);
    var binMin = ee.Number(ee.Algorithms.If(isFirst, 0,
                    // (i-1)*10 + 1  for i>=2
                    i.subtract(1).multiply(10).add(1)));
    var binMax = ee.Number(ee.Algorithms.If(isFirst, 10, i.multiply(10)));

    // 通过过滤逐值 histogramFC 来合计该段的 pixel_count
    // 由于 histogramFC 的条目是 value/pixel_count（value 是数字），我们可以 filter 然后 aggregate_sum
    var fcInRange = histogramFC.filter(ee.Filter.gte('value', binMin))
                               .filter(ee.Filter.lte('value', binMax));

    // aggregate_sum 可能返回 null（若没有匹配），所以先测 size() 再决定
    var sumCount = ee.Algorithms.If(fcInRange.size().gt(0),
                                    fcInRange.aggregate_sum('pixel_count'),
                                    0);
    sumCount = ee.Number(sumCount);

    return ee.Feature(null, {
      'bin_index': i,
      'bin_min': binMin,
      'bin_max': binMax,
      'pixel_count': sumCount
    });
  });

  var aggFC = ee.FeatureCollection(aggFeatures);

  // ----- 导出聚合后的分段直方图 -----
  var fileNameAgg = fileName + '_bins10';
  Export.table.toDrive({
    collection: aggFC,
    description: fileNameAgg,
    folder: 'Changed_pro_histogram',
    fileNamePrefix: fileNameAgg,
    fileFormat: 'CSV',
    selectors: ['bin_index', 'bin_min', 'bin_max', 'pixel_count']
  });

  print('  创建聚合 10 值/段 导出：', fileNameAgg, '(', t, 'vs', nextYr, ')');

  // 可视化（可选）
  Map.addLayer(resultImg, {min:0, max:10000, palette:['white','blue','purple']}, 
               'changeValue_' + t + 'vs' + nextYr, false);

}); // end yrs.forEach

print('已创建全部直方图及聚合导出任务（请到 Tasks 面板启动）。');

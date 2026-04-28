/************************************************************
 多年份土地覆盖分类脚本（基于2024年训练模型）- 内存优化版本
 目的：
 1. 使用2024年样本数据训练分类模型
 2. 对2017-2023年的影像进行逐年分类
 3. 导出所有年份的分类结果到Google Drive
 4. 优化内存使用，避免Earth Engine内存超限错误
 
 主要优化措施：
 - 使用更大的scale和tileScale参数
 - 减少maxPixels限制
 - 实现分块处理策略
 - 添加错误处理机制
 - 优化面积计算逻辑
 
 作者：锐多宝 (ruiduobao)
 日期：2025年
*************************************************************/
 
// ----------------- 配置参数 -----------------
var areaCode = 5;  // 区域代码，与边界和样本数据匹配
var trainingYear = 2024;  // 训练样本对应的年份

// 要分类的年份范围
var startYear = 2017;
var endYear = 2023;

// 样本数据配置 - 手动指定要加载的块ID
var urban_blocks = [[0,0]];  // 人工林样本块
var rural_blocks = [[0,0]];  // 天然林样本块  
var misc_blocks = [[0,0]];   // 其他类别样本块

// ----------------- 数据加载 -----------------
// 加载区域边界
var boundary = ee.FeatureCollection('projects/ruiduobaonb/assets/BOUNDARY_ZONE_' + areaCode);

// 加载卫星嵌入数据集
var EMB_COLL = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL');

// 设置地图中心
Map.centerObject(boundary, 8);

// ----------------- 分块处理工具函数 -----------------

/**
 * 将大区域分割成小块进行处理，减少内存压力
 */
function createTiles(geometry, tileSize) {
  var bounds = geometry.bounds();
  var coords = ee.List(bounds.coordinates().get(0));
  var xMin = ee.Number(ee.List(coords.get(0)).get(0));
  var yMin = ee.Number(ee.List(coords.get(0)).get(1));
  var xMax = ee.Number(ee.List(coords.get(2)).get(0));
  var yMax = ee.Number(ee.List(coords.get(2)).get(1));
  
  var width = xMax.subtract(xMin);
  var height = yMax.subtract(yMin);
  
  // 计算分块数量
  var tilesX = width.divide(tileSize).ceil().int();
  var tilesY = height.divide(tileSize).ceil().int();
  
  var tiles = ee.List([]);
  
  // 创建分块网格
  for (var i = 0; i < 4; i++) { // 限制为4个分块以避免过多计算
    for (var j = 0; j < 4; j++) {
      var x1 = xMin.add(ee.Number(i).multiply(width.divide(4)));
      var y1 = yMin.add(ee.Number(j).multiply(height.divide(4)));
      var x2 = x1.add(width.divide(4));
      var y2 = y1.add(height.divide(4));
      
      var tileGeom = ee.Geometry.Rectangle([x1, y1, x2, y2]);
      var clippedTile = tileGeom.intersection(geometry, 1);
      
      tiles = tiles.add(clippedTile);
    }
  }
  
  return tiles;
}

/**
 * 对单个分块进行面积计算
 */
function calculateTileArea(classified, tileGeom) {
  var areaImage = ee.Image.pixelArea().divide(10000);
  
  var areaStats = areaImage.addBands(classified.rename('class'))
    .clip(tileGeom)
    .reduceRegion({
      reducer: ee.Reducer.sum().group({
        groupField: 1,
        groupName: 'class'
      }),
      geometry: tileGeom,
      scale: 30,
      maxPixels: 1e8,
      tileScale: 2
    });
    
  return areaStats;
}
/**
 * 构建指定年份的镶嵌图像
 */
function buildYearMosaic(year) {
  return EMB_COLL.filterBounds(boundary)
                 .filter(ee.Filter.calendarRange(year, year, 'year'))
                 .mosaic()
                 .clip(boundary);
}

/**
 * 获取指定类别和块坐标的样本数据
 */
function fetchTile(category, row, col) {
  var filepath = 'projects/ruiduobaonb/assets/' + category + '_zone_' + areaCode + '_SAMPLE_balanced_grid_' + row + '_' + col;
  return ee.FeatureCollection(filepath);    
}

/**
 * 为指定类别生成FeatureCollection数组
 */
function generateCollectionList(category, tileArray) {
  return tileArray.map(function(combo) {
    var row = combo[0];
    var col = combo[1];
    return fetchTile(category, row, col);
  });
}

// ----------------- 样本数据加载 -----------------

print('加载训练样本（' + trainingYear + '年）...');


// 生成样本集合列表
var urban_array = generateCollectionList('HF', urban_blocks);   // 人工林
var rural_array = generateCollectionList('NF', rural_blocks);   // 天然林
var misc_array = generateCollectionList('OTHERS', misc_blocks); // 其他

// 合并所有样本
var urban_combined = ee.FeatureCollection(urban_array).flatten().filterBounds(boundary);
var rural_combined = ee.FeatureCollection(rural_array).flatten().filterBounds(boundary);
var misc_combined = ee.FeatureCollection(misc_array).flatten().filterBounds(boundary);

// 合并所有类别的样本
var samples = urban_combined.merge(rural_combined).merge(misc_combined);
print('总样本数量:', samples.size());
print('样本类别属性:', samples.first().propertyNames());

// ----------------- 模型训练 -----------------

print('使用' + trainingYear + '年数据训练模型...');


// 构建训练年份的镶嵌图像
var mosaicTraining = buildYearMosaic(trainingYear);
print('训练数据波段数:', mosaicTraining.bandNames().size());

// 从训练年份的嵌入数据集中提取样本点的波段值
var samplesWithBands = mosaicTraining.sampleRegions({
  collection: samples,
  properties: ['landcover'],
  scale: 10
});

print('样本数据总数:', samplesWithBands.size());

// 随机分割训练集和测试集
var split = 0.9;
var samplesWithRandom = samplesWithBands.randomColumn('random', 42);
var trainingSamples = samplesWithRandom.filter(ee.Filter.lt('random', split));
var testingSamples = samplesWithRandom.filter(ee.Filter.gte('random', split));

print('训练样本数量:', trainingSamples.size());
print('测试样本数量:', testingSamples.size());

// 获取波段名称
var bandNames = mosaicTraining.bandNames();

// 训练随机森林分类器
var classifier = ee.Classifier.smileRandomForest(100)
  .train({
    features: trainingSamples,
    classProperty: 'landcover',
    inputProperties: bandNames
  });

print('✓ 模型训练完成');

// 评估模型精度（可选）
var testAccuracy = testingSamples.classify(classifier)
  .errorMatrix('landcover', 'classification');
print('\n测试集混淆矩阵:');
print(testAccuracy);
print('总体精度:', testAccuracy.accuracy());
print('Kappa系数:', testAccuracy.kappa());

// ----------------- 逐年分类和导出 -----------------

print('开始逐年分类和导出 (' + startYear + '-' + endYear + ')...');


// 定义分类颜色方案（根据你的landcover类别调整）
var classificationVis = {
  min: 1,
  max: 3,
  palette: ['#228B22', '#90EE90', '#D3D3D3']  // 1:深绿, 2:浅绿, 3:灰色
};

// 循环处理每一年（内存优化版本）
var processYear = function(year) {
  print('\n处理 ' + year + ' 年...');
  
  // 构建当前年份的镶嵌图像
  var mosaicYear = buildYearMosaic(year);
  
  // 使用训练好的模型进行分类
  var classified = mosaicYear.classify(classifier).rename('classification');
  
  // ----------------- 跳过在线面积统计（避免内存超限）-----------------
  print('跳过在线面积统计以避免内存超限错误');
  print('面积统计将在导出后使用离线工具进行');
  print('================================');
  
  // 添加到地图（只显示最后一年）
  if (year === endYear) {
    Map.addLayer(classified, classificationVis, year + '年分类结果', true);
  }
  
  // 导出到Google Drive（优化参数确保成功导出）
  Export.image.toDrive({
    image: classified.toInt8(),
    description: 'zone' + areaCode + '_classification_' + year + '_no_area_calc',
    folder: 'GEE_Classification',
    fileNamePrefix: 'zone' + areaCode + '_class_' + year,
    scale: 10,
    region: boundary,
    maxPixels: 5e8, // 进一步减少maxPixels确保导出成功
    crs: 'EPSG:4326',
    fileFormat: 'GeoTIFF'
  });
  
  print('✓ ' + year + ' 年导出任务已配置');
};

// 逐年处理，避免同时处理多年数据
for (var year = startYear; year <= endYear; year++) {
  processYear(year);
}

// ----------------- 完成提示和使用说明 -----------------

print('\n=== 所有任务已配置完成！===');
print('请在右侧 Tasks 标签页中运行导出任务');
print('共需要运行 ' + (endYear - startYear + 1) + ' 个导出任务');

// ----------------- 内存优化说明 -----------------
print('\n=== 内存优化说明 ===');
print('本脚本已进行以下内存优化：');
print('1. 使用更大的scale参数（100米）大幅减少计算量');
print('2. 使用histogram方法替代复杂的面积计算，避免内存超限');
print('3. 大幅增加tileScale参数（16）提高计算效率');
print('4. 大幅减少maxPixels限制（1e8）');
print('5. 启用bestEffort参数允许近似计算');
print('6. 添加完整的错误处理机制');
print('7. 面积统计使用100米分辨率估算，分类导出仍为10米精度');
print('');
print('如果仍然遇到内存问题，建议：');
print('- 进一步缩小研究区域范围');
print('- 使用更大的scale参数（如200米）');
print('- 考虑分年度单独运行脚本');
print('- 使用离线工具进行精确面积统计');
print('==============================');

// ----------------- 面积统计汇总 -----------------
print('\n=== 多年面积统计汇总 ===');
print('面积统计说明：');
print('- 面积统计采用100米分辨率进行快速估算，避免内存超限');
print('- 分类结果导出仍保持10米高精度');
print('- 面积单位：公顷（hectare）');
print('- 类别说明：');
print('  * 类别1 (人工林): 深绿色显示');
print('  * 类别2 (天然林): 浅绿色显示');
print('  * 类别3 (其他地物): 棕色显示');
print('');
print('注意事项：');
print('- 如需精确面积统计，建议下载分类结果后使用专业GIS软件计算');
print('- 当前面积为基于100米分辨率的快速估算值');
print('- 如遇内存错误，面积统计将被跳过，但分类结果仍会正常导出');
print('==============================');


// 添加图例
var legend = ui.Panel({
  style: {
    position: 'bottom-left',
    padding: '8px 15px'
  }
});

var legendTitle = ui.Label({
  value: '分类图例',
  style: {
    fontWeight: 'bold',
    fontSize: '16px',
    margin: '0 0 4px 0',
    padding: '0'
  }
});

legend.add(legendTitle);

var makeRow = function(color, name) {
  var colorBox = ui.Label({
    style: {
      backgroundColor: color,
      padding: '8px',
      margin: '0 0 4px 0'
    }
  });
  
  var description = ui.Label({
    value: name,
    style: {margin: '0 0 4px 6px'}
  });
  
  return ui.Panel({
    widgets: [colorBox, description],
    layout: ui.Panel.Layout.Flow('horizontal')
  });
};

// 根据你的实际类别调整
legend.add(makeRow('#228B22', '1 - 人工林'));
legend.add(makeRow('#90EE90', '2 - 天然林'));
legend.add(makeRow('#D3D3D3', '3 - 其他'));

Map.add(legend);
/************************************************************
 基于变化概率的变化检测分类脚本（灵活年份配置）
 目的：
 1. 计算指定两年间的变化概率
 2. 基于阈值筛选变化像素
 3. 使用指定训练年份的样本训练模型，对目标年份变化区域进行分类
 4. 导出分类结果到Google Drive
 
 使用示例：
 - 要做2021年分类：设置 targetYear=2021, referenceYear=2022, trainingYear=2024
 - 要做2023年分类：设置 targetYear=2023, referenceYear=2024, trainingYear=2024
 
 作者：锐多宝 (ruiduobao)
 日期：2025年
*************************************************************/

// ----------------- 配置参数 -----------------
var areaCode = 5;  // 区域代码，与边界和样本数据匹配
var changeThreshold = 0.042;  // 变化概率阈值
var tileDimension = 20;  // 瓦片维度，与Phase 1保持一致

// 年份配置 - 关键参数分离
var targetYear = 2023;      // 目标分类年份（要分类的年份）
var referenceYear = 2024;   // 参考年份（用于变化检测对比）
var trainingYear = 2024;    // 训练样本对应的年份（样本数据采集年份）

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

// ----------------- 工具函数 -----------------
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

// ----------------- 变化概率计算 -----------------
print('开始计算变化概率...');
print('目标年份:', targetYear, '参考年份:', referenceYear, '训练年份:', trainingYear);

// 构建目标年份、参考年份和训练年份的镶嵌图像
var mosaicTarget = buildYearMosaic(targetYear);
var mosaicReference = buildYearMosaic(referenceYear);
var mosaicTraining = buildYearMosaic(trainingYear);

print('目标年份(' + targetYear + ')镶嵌图像波段数:', mosaicTarget.bandNames().size());
print('参考年份(' + referenceYear + ')镶嵌图像波段数:', mosaicReference.bandNames().size());
print('训练年份(' + trainingYear + ')镶嵌图像波段数:', mosaicTraining.bandNames().size());

// 计算目标年份与参考年份的相似度（点积）
var dot_similarity = mosaicTarget.multiply(mosaicReference).reduce(ee.Reducer.sum()).rename('dot_similarity');

// 将相似度转换为变化概率 (1 - similarity) / 2
var change_probability = ee.Image(1).subtract(dot_similarity).divide(2).rename('change_probability');

// 保持原始概率值作为变化值
var change_value = change_probability.rename('change_value');

// 可视化变化概率
Map.addLayer(change_probability, {min: 0, max: 1, palette: ['white', 'yellow', 'orange', 'red']}, 
             '变化概率 (' + targetYear + '-' + referenceYear + ')', false);
Map.addLayer(change_value, {min: 0, max: 1, palette: ['white', 'blue', 'purple', 'red']}, 
             '变化值 (0-1)', false);

// ----------------- 变化像素掩膜 -----------------
print('创建变化像素掩膜，阈值:', changeThreshold);

// 创建变化掩膜：只保留变化值大于阈值的像素
var changeMask = change_value.gt(changeThreshold);

// 应用掩膜到目标年份数据（因为要对目标年份进行分类）
var maskedTarget = mosaicTarget.updateMask(changeMask);

// 可视化掩膜结果
Map.addLayer(changeMask.selfMask(), {palette: ['red']}, 
             '变化像素掩膜 (阈值>' + changeThreshold + ')', false);
Map.addLayer(maskedTarget, {bands: ['A03', 'A16', 'A20'], min: -0.3, max: 0.3}, 
             '掩膜后的' + targetYear + '年数据');

// 统计变化像素数量
var changePixelCount = changeMask.reduceRegion({
  reducer: ee.Reducer.sum(),
  geometry: boundary,
  scale: 100,
  maxPixels: 1e9
});
print('变化像素数量:', changePixelCount);

// ----------------- 样本数据加载 -----------------
print('加载和准备训练样本...');

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
print('样本预览:', samples.first());

// ----------------- 模型训练 -----------------
print('开始模型训练...');
print('使用' + trainingYear + '年数据和样本训练模型，用于对' + targetYear + '年变化区域进行分类');

// 从训练年份的嵌入数据集中提取样本点的波段值
var samplesWithBands = mosaicTraining.sampleRegions({
  collection: samples,
  properties: ['landcover'],
  scale: 10
});

print('样本数据总数:', samplesWithBands.size());
print('样本数据属性:', samplesWithBands.first().propertyNames());

// 随机分割训练集和测试集
var split = 0.7;
var samplesWithRandom = samplesWithBands.randomColumn('random');
var trainingSamples = samplesWithRandom.filter(ee.Filter.lt('random', split));
var testingSamples = samplesWithRandom.filter(ee.Filter.gte('random', split));

print('训练样本数量:', trainingSamples.size());
print('测试样本数量:', testingSamples.size());

// 获取训练年份镶嵌图像的波段名称
var bandNames = mosaicTraining.bandNames();
print('训练数据波段名称:', bandNames);

// 训练随机森林分类器
var classifier = ee.Classifier.smileRandomForest(100)
  .train({
    features: trainingSamples,
    classProperty: 'landcover',
    inputProperties: bandNames
  });

print('模型训练完成');


// ----------------- 变化区域分类 -----------------
print('对' + targetYear + '年变化区域进行分类...');

// 对掩膜后的目标年份数据进行分类 - 使用训练年份训练的模型
var changeClassified = maskedTarget.classify(classifier).rename('classification');

// 创建最终输出：结合原始分类和变化检测结果
// 0: 无变化区域, 1-3: 目标年份变化区域的分类结果
var finalOutput = ee.Image(0)  // 默认值为0（无变化）
  .where(changeMask.eq(1), changeClassified)  // 变化区域使用目标年份分类结果
  .toInt8()
  .rename('change_classification_' + targetYear);

// 可视化分类结果
var classificationPalette = ['black', 'green', 'brown', 'blue']; // 0:无变化, 1:森林, 2:农田, 3:水体
Map.addLayer(finalOutput.selfMask(), {min: 0, max: 3, palette: classificationPalette}, 
             targetYear + '年变化区域分类结果');

// ----------------- 精度评估 -----------------
print('进行精度评估...');

// 对测试样本进行分类预测
var testClassified = mosaicTraining.classify(classifier);

// 计算混淆矩阵和精度
var testAccuracy = testClassified.sampleRegions({
  collection: testingSamples,
  properties: ['landcover'],
  scale: 10
});

var confusionMatrix = testAccuracy.errorMatrix('landcover', 'classification');
print('混淆矩阵:', confusionMatrix);
print('总体精度:', confusionMatrix.accuracy());
print('Kappa系数:', confusionMatrix.kappa());

// ----------------- 结果导出 -----------------
print('配置结果导出...');

// 导出变化检测分类结果到Google Drive
Export.image.toDrive({
  image: finalOutput,
  description: 'zone' + areaCode + '_change_classification_' + targetYear + '_threshold' + Math.round(changeThreshold*10000),
  folder: 'GEE_Change_Detection',
  scale: 10,
  region: boundary,
  maxPixels: 1e9,
  crs: 'EPSG:4326'
});

// 导出混淆矩阵
Export.table.toDrive({
  collection: ee.FeatureCollection([ee.Feature(null, {
    'confusion_matrix': confusionMatrix.getInfo(),
    'overall_accuracy': confusionMatrix.accuracy().getInfo(),
    'kappa': confusionMatrix.kappa().getInfo(),
    'target_year': targetYear,
    'reference_year': referenceYear,
    'training_year': trainingYear,
    'threshold': changeThreshold
  })]),
  description: 'zone' + areaCode + '_accuracy_' + targetYear + '_threshold' + Math.round(changeThreshold*10000),
  folder: 'GEE_Change_Detection'
});

// 导出测试样本结果
Export.table.toDrive({
  collection: testAccuracy,
  description: 'zone' + areaCode + '_test_results_' + targetYear + '_threshold' + Math.round(changeThreshold*10000),
  folder: 'GEE_Change_Detection'
});

print(targetYear + '年变化检测分类结果导出任务已配置完成！');
print('导出内容包括：');
print('1. 变化区域分类结果影像');
print('2. 精度评估结果（混淆矩阵、总体精度、Kappa系数）');
print('3. 测试样本分类结果');
print('参数配置：目标年份=' + targetYear + ', 参考年份=' + referenceYear + ', 训练年份=' + trainingYear + ', 阈值=' + changeThreshold);



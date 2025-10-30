/************************************************************
  Phase 2: Load designated samples by block IDs, combine and perform classification
*************************************************************/

// —— Configuration Input Section ——
// areaCode: The zone you're processing (matches the zone number in directory boundaries and block samples)
var areaCode =5;  

// tileDimension: Keep consistent with Phase 1
var tileDimension = 20; 

// For each category, manually specify the row i and column j arrays to load
// For instance, urban_blocks = [[0,0], [0,1], [1,0], [1,1]] loads urban_zone2_tile_0_0, urban_zone2_tile_0_1, etc.
// var urban_blocks = [[0,0], [0,1], [1,0], [1,1], [2,0], [2,1], [3,0]];
// var rural_blocks = [[0,0], [1,0], [1,0], [1,1], [2,0], [2,1], [3,0]];
// var misc_blocks =[[0,0], [1,0], [1,0], [1,1], [2,0], [2,1], [3,0]];

var urban_blocks = [[0,0]];
var rural_blocks = [[0,0]];
var misc_blocks =  [[0,0]];

// —— Boundary and satellite embedding layer —— BOUNDARY_ZONE_9
var boundary = ee.FeatureCollection('projects/ruiduobaonb/assets/' +  'BOUNDARY_ZONE_'+areaCode );
var SAT_EMB = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL');

var beginDate = '2024-01-01';
var finishDate   = '2025-01-01';

var sat_emb = SAT_EMB
  .filterBounds(boundary)
  .filterDate(beginDate, finishDate)
  .mosaic()
  .clip(boundary);

Map.centerObject(boundary, 8);
Map.addLayer(sat_emb, {bands:['A03','A16','A20'], min:-0.3, max:0.3}, 'Satellite Embedding');

// —— Helper function: Fetch a block's FeatureCollection, return empty if path doesn't exist —— 
function fetchTile(category, row, col) {
  var filepath = 'projects/ruiduobaonb/assets/' + category + '_zone_' + areaCode + '_SAMPLE_balanced_grid_' + row + '_' + col;
  // Use a try-catch approach to check if loading succeeds
  var collection = ee.FeatureCollection(filepath);
  // We can't catch "path not found" server-side directly,
  // but we can check size: if size errors or null, return empty
  // Wrap with filterBounds on boundary to handle potential failures.
  return collection;
}

// —— For each category, convert input block lists to FeatureCollection arrays —— 
function generateCollectionList(category, tileArray) {
  return tileArray.map(function(combo) {
    var row = combo[0];
    var col = combo[1];
    return fetchTile(category, row, col);
  });
}

// —— Construct category arrays + merge them —— 
var urban_array = generateCollectionList('HF', urban_blocks);
var rural_array = generateCollectionList('NF', rural_blocks);
var misc_array = generateCollectionList('OTHERS', misc_blocks);

// urban_combined = Merge all FeatureCollections in urban_array
var urban_combined = ee.FeatureCollection(urban_array).flatten();
var rural_combined = ee.FeatureCollection(rural_array).flatten();
var misc_combined = ee.FeatureCollection(misc_array).flatten();

// Filter within boundary
urban_combined = urban_combined.filterBounds(boundary);
rural_combined = rural_combined.filterBounds(boundary);
misc_combined = misc_combined.filterBounds(boundary);

// Combine the three sample categories
var combinedSamples = urban_combined.merge(rural_combined).merge(misc_combined);
print('Total merged samples:', combinedSamples.size());
print('Sample preview:', combinedSamples.first());

// —— Randomly split into training / testing sets —— 
combinedSamples = combinedSamples.randomColumn('randomVal',42);
var trainingSet = combinedSamples.filter(ee.Filter.lt('randomVal', 0.8));
var testingSet = combinedSamples.filter(ee.Filter.gte('randomVal', 0.8));
print('Training size:', trainingSet.size(), 'Testing size:', testingSet.size());

// —— Train the model —— 
var model = ee.Classifier.smileRandomForest(300).train({
  features: trainingSet,
  classProperty: 'landcover',
  inputProperties: sat_emb.bandNames()
});

// —— Apply classification to image —— 
var outputClassified = sat_emb.classify(model);
Map.addLayer(outputClassified, {min:1, max:3, palette:['FF0000','00FF00','0000FF']}, 'Model output');

// —— Evaluate accuracy on test set —— 
var tested = testingSet.classify(model);
var errorMatrix = tested.errorMatrix('landcover', 'classification');
print('Error matrix:', errorMatrix);
print('Total accuracy:', errorMatrix.accuracy());
print('Kappa coefficient:', errorMatrix.kappa());

// —— Export error matrix as CSV —— 
// Assuming classes: 1:urban, 2:rural, 3:misc (based on min:1, max:3)
var categoryNames = ee.List(['HF', 'NF', 'OTH']);  // Adjust to match your categories

var matrixArray = errorMatrix.array();  // Get the array
var matrixRows = matrixArray.toList();  // Convert to list of rows

// Create feature collection: each row as a feature with category properties and counts
var matrixFeatures = matrixRows.map(function(rowData, rowIdx) {
  var rowValues = ee.List(rowData);
  // Improved: single feature with multiple columns
  var properties = ee.Dictionary.fromLists(categoryNames, rowValues);
  properties = properties.set('true_category', categoryNames.get(rowIdx));  // Add true category row label
  return ee.Feature(null, properties);
});

var matrixCollection = ee.FeatureCollection(matrixFeatures);

// Export to Drive
Export.table.toDrive({
  collection: matrixCollection,
  description: 'area' + areaCode + '_error_matrix',
  folder: 'GEE_Exports',
  fileFormat: 'CSV'
});

// —— Export test set points with true and predicted values as CSV —— 
Export.table.toDrive({
  collection: tested,
  description: 'area' + areaCode + '_test_points',
  folder: 'GEE_Exports',
  fileFormat: 'CSV',
  selectors: ['landcover', 'classification', 'geometry']  // Export true class, predicted class, and geometry (optional, key properties only)
});

// —— Export classified image as int8 GeoTIFF —— 
Export.image.toDrive({
  image: outputClassified.toInt8(),  // Convert to int8
  description: 'area' + areaCode + '_model_classified',
  folder: 'GEE_Exports',
  scale: 10,
  region: boundary,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF'
});
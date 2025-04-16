
import numpy as np
import pandas as pd
from mgwr.mgwr.gwr import MGWR, GWR
from mgwr.mgwr.sel_bw import Sel_BW
from shapely.geometry import Point
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import pickle

# import os
# from pathlib import Path

# src_folder_path = os.path.abspath('..')
# project_folder_path = Path(src_folder_path).resolve().parents[0]
# train_set_path = project_folder_path.joinpath(r'data\train_set_with_feat_cleanCorr.csv'))
train_set_whole = pd.read_csv('data/train_set_with_feat_cleanCorr.csv')
test_data = pd.read_csv('data/test_set_with_feat_cleanCorr.csv')



k = 20 
kf = KFold(n_splits=k, shuffle=True, random_state=42)


data = train_set_whole


posibble_preds = list(data)
items_to_remove = ['id', 'mean_gs', 'sd', 'skewness', 'kurtosis', 'current_max', 'current_min','sample_type','y_im', 'x_im', 'x_m', 'y_m', 'current_mean', 'current_range', 'gebco', 'x', 'y', 'island_dist', 'comb_dist']
#remove "base" as well
updated_list = [item for item in posibble_preds if item not in items_to_remove]

#data['current_range'] = data['current_max'] - data['current_min']



predictors = ['current_mean', 'current_range', 'gebco']
response = 'mean_gs'



gdf = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data['x'], data['y']))
gdf = gdf.set_crs(epsg=4326)  # Assuming WGS84
gdf = gdf.to_crs(epsg=32633)  # Convert to UTM zone 33N

data['x'] = gdf.geometry.x
data['y'] = gdf.geometry.y


test_gdf = gpd.GeoDataFrame(test_data, geometry=gpd.points_from_xy(test_data['x'], test_data['y']))
test_gdf = test_gdf.set_crs(epsg=4326)
test_gdf = test_gdf.to_crs(epsg=32633)

test_data['x'] = test_gdf.geometry.x
test_data['y'] = test_gdf.geometry.y



import itertools


combinations = []
for r in range(0, len(updated_list) + 1):
    combinations.extend(itertools.combinations(updated_list, r))

# Add each combination to A
#pred_result_t = [predictors + list(comb) for comb in combinations]
updated_list.append('island_dist')

comb2 = []
for r in range(1,len(updated_list)+1):
    comb2.extend(itertools.combinations(updated_list, r))

combtot = combinations + comb2

#all_combinations = set(combinations_without_new + combinations_with_new)

pred_result_t = [predictors + list(comb) for comb in combtot]

# Remove duplicates from pred_result_t
pred_result = []
for comb in pred_result_t:
    if comb not in pred_result:
        pred_result.append(comb)


coords = pd.DataFrame(data[['x', 'y']])
#data = data.drop(columns=['x', 'y'])
coords_test = pd.DataFrame(test_data[['x', 'y']])


pred = 5
predictorsN = pred_result[pred]
print(predictorsN)
X = pd.DataFrame(data[predictorsN])
y = pd.DataFrame(data[response])
X_test_t = pd.DataFrame(test_data[predictorsN])
count = 0
mean_gs_all = np.zeros((k, len(X_test_t),))
for train_index, test_index in kf.split(X):
    print(count)
    X_train_t = X.loc[train_index].values
    y_train = y.loc[train_index].values.reshape(-1, 1)
    scaler = StandardScaler().fit(X_train_t)
    X_train = scaler.transform(X_train_t)
    X_test = scaler.transform(X_test_t)
    #y_train = y_train
    coords_train = coords.loc[train_index].values
    selector = Sel_BW(coords_train, y_train, X_train, multi=False) # This creates the bandwidths for different input features
    bws = selector.search(verbose=True, search_method='golden_section', max_iter=100) # This searches for the optimal bandwidth (fields of influence)
    mgwr_model = GWR(coords_train, y_train, X_train, bws)
    results = mgwr_model.fit() # This fits the model to the data
    scale = results.scale
    residuals = results.resid_response
    mean_gs_all[count,:] = np.squeeze(mgwr_model.predict(coords_test.values, X_test, scale, residuals).predictions)
    count +=1


print('finished')

mean_gs = np.median(mean_gs_all, axis=0)
test_data['mean_gs'] = mean_gs
submission = test_data[['id', 'mean_gs']]
submission.to_csv(f'data/Submissions/k_fold_feature_{pred}_k_20_median.csv', index=False)

mean_gs = np.percentile(mean_gs_all, 70, axis=0)
test_data['mean_gs'] = mean_gs
submission = test_data[['id', 'mean_gs']]
submission.to_csv(f'data/Submissions/k_fold_feature_{pred}_k_20_percentile_70.csv', index=False)
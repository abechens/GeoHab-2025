
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

import pickle

from sklearn.preprocessing import StandardScaler

# Assuming X is your feature DataFrame
#scaler = StandardScaler()



# import os
# from pathlib import Path

# src_folder_path = os.path.abspath('..')
# project_folder_path = Path(src_folder_path).resolve().parents[0]
# train_set_path = project_folder_path.joinpath(r'data\train_set_with_feat_cleanCorr.csv'))
train_set_whole = pd.read_csv('data/train_set_with_feat_cleanCorr.csv')



k = 10 
kf = KFold(n_splits=k, shuffle=True, random_state=42)


data = train_set_whole


print(list(data))
posibble_preds = list(data)
items_to_remove = ['id', 'mean_gs', 'sd', 'skewness', 'kurtosis', 'current_max', 'current_min','sample_type','y_im', 'x_im', 'x_m', 'y_m', 'current_mean', 'current_range', 'gebco', 'x', 'y']
#remove "base" as well
updated_list = [item for item in posibble_preds if item not in items_to_remove]

#data['current_range'] = data['current_max'] - data['current_min']
print(updated_list)



predictors = ['current_mean', 'current_range', 'gebco']
response = 'mean_gs'



gdf = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data['x'], data['y']))
gdf = gdf.set_crs(epsg=4326)  # Assuming WGS84
gdf = gdf.to_crs(epsg=32633)  # Convert to UTM zone 33N

data['x'] = gdf.geometry.x
data['y'] = gdf.geometry.y
print(data['x'])



import itertools

combinations = []
for r in range(0, len(updated_list) + 1):
    combinations.extend(itertools.combinations(updated_list, r))

# Add each combination to A
pred_result = [predictors + list(comb) for comb in combinations]

coords = data[['x', 'y']]
#data = data.drop(columns=['x', 'y'])



res_dict = dict()
for pred in range(len(pred_result)):
    predictorsN = pred_result[pred]
    print(predictorsN)
    X = data[predictorsN]
    
    y = data[response]
    count = 0
    r2_scores = []
    for train_index, test_index in kf.split(X):
        print(count)
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        #scaler = StandardScaler().fit(X)
        #X_train = scaler.transform(X_train_t)
        #X_test = scaler.transform(X_test_t)
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        y_train = y_train.values.reshape(-1, 1)
        coords_train, coords_test = coords.iloc[train_index], coords.iloc[test_index]
        selector = Sel_BW(coords_train.values, y_train, X_train, multi=False) # This creates the bandwidths for different input features
        bws = selector.search(verbose=True, search_method='golden_section', max_iter=10) # This searches for the optimal bandwidth (fields of influence)
        mgwr_model = GWR(coords_train.values, y_train, X_train, bws)
        results = mgwr_model.fit() # This fits the model to the data
        scale = results.scale
        residuals = results.resid_response
        mean_gs_pred = mgwr_model.predict(coords_test.values, X_test, scale, residuals).predictions
        r2_scores.append(r2_score(y_test.values.reshape(-1, 1), mean_gs_pred))
        count +=1

    res_dict[pred] = r2_scores



with open('r2_scores_dict.pkl', 'wb') as f:
    pickle.dump(res_dict, f)



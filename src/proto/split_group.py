
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



with open('k_fold_investigate_errors.pkl', 'rb') as f:
    loaded_r2_scores_dict = pickle.load(f)
    loaded_grain_pred = pickle.load(f)

train_set_whole = pd.read_csv('data/train_set_with_feat_cleanCorr.csv')
train_set = train_set_whole


med_grain_pred = {keys: np.nanmedian(loaded_grain_pred[keys],axis=0) for keys in loaded_grain_pred.keys()}


diff_grain_size = {keys: train_set['mean_gs']-med_grain_pred[keys] for keys in  loaded_grain_pred.keys()}

key_to_watch = 20
error_thres = 5
idx_error = np.abs(diff_grain_size[key_to_watch])>=error_thres

idx_Corr = idx_error == False
error_dataframe = train_set.loc[idx_error]
correct_dataframe = train_set.loc[idx_Corr]



#train_set, test_data=train_test_split(train_set_whole, test_size=0.0, random_state=2)


k = 10 
kf = KFold(n_splits=k, shuffle=True, random_state=42)


data = train_set


posibble_preds = list(data)
items_to_remove = ['id', 'mean_gs', 'sd', 'skewness', 'kurtosis', 'current_max', 'current_min','sample_type','y_im', 'x_im', 'x_m', 'y_m', 'current_mean', 'current_range', 'gebco', 'x', 'y', 'island_dist', 'comb_dist']
#remove "base" as well
updated_list = [item for item in posibble_preds if item not in items_to_remove]

#data['current_range'] = data['current_max'] - data['current_min']



predictors = ['current_mean', 'current_range', 'gebco']
response = 'mean_gs'







## Train data
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


coords = data[['x', 'y']]



## Dele opp:
train_set_1 = error_dataframe
train_set_2 = correct_dataframe




pred_res_comb = [ 20] #Best combinations from inv
res_dict = dict()
gs_res = dict()
pred = 9

data = train_set_1
predictorsN =  pred_result[pred]
print(predictorsN)
X = data[predictorsN]
#X_test_t = test_data[predictorsN]

y = data[response]
#y_test = test_data[response]
count = 0
r2_scores = []
mean_gs_all = np.zeros((k, len(train_set)))

for train_index, test_index in kf.split(X):
    print(count)
    X_train_t= X.iloc[train_index]
    scaler = StandardScaler().fit(X_train_t)
    X_train = scaler.transform(X_train_t)
    #X_test = scaler.transform(X_test_t)
    y_train= y.iloc[train_index]
    y_train = y_train.values.reshape(-1, 1)
    coords_train = coords.iloc[train_index]
    selector = Sel_BW(coords_train.values, y_train, X_train, multi=False) # This creates the bandwidths for different input features
    bws = selector.search(verbose=True, search_method='golden_section', max_iter=100) # This searches for the optimal bandwidth (fields of influence)
    mgwr_model = GWR(coords_train.values, y_train, X_train, bws)
    results = mgwr_model.fit() # This fits the model to the data
    scale = results.scale
    residuals = results.resid_response
    mean_gs_pred = mgwr_model.predict(coords_train.values, X_train, scale, residuals).predictions
    r2_scores.append(r2_score(y_train, mean_gs_pred))
    t = np.zeros(len(train_set)) + np.nan
    org_idx = X.index[train_index].tolist()
    t[org_idx] = np.squeeze(mean_gs_pred)

    mean_gs_all[count,:] = t
    count += 1

gs_res[1] = mean_gs_all
res_dict[1] = r2_scores

data = train_set_2
predictorsN =  pred_result[pred]
print(predictorsN)
X = data[predictorsN]
#X_test_t = test_data[predictorsN]

y = data[response]
#y_test = test_data[response]
count = 0
r2_scores = []
mean_gs_all = np.zeros((k, len(train_set)))

for train_index, test_index in kf.split(X):
    print(count)
    X_train_t= X.iloc[train_index]
    scaler = StandardScaler().fit(X_train_t)
    X_train = scaler.transform(X_train_t)
    #X_test = scaler.transform(X_test_t)
    y_train= y.iloc[train_index]
    y_train = y_train.values.reshape(-1, 1)
    coords_train = coords.iloc[train_index]
    selector = Sel_BW(coords_train.values, y_train, X_train, multi=False) # This creates the bandwidths for different input features
    bws = selector.search(verbose=True, search_method='golden_section', max_iter=100) # This searches for the optimal bandwidth (fields of influence)
    mgwr_model = GWR(coords_train.values, y_train, X_train, bws)
    results = mgwr_model.fit() # This fits the model to the data
    scale = results.scale
    residuals = results.resid_response
    mean_gs_pred = mgwr_model.predict(coords_train.values, X_train, scale, residuals).predictions
    r2_scores.append(r2_score(y_train, mean_gs_pred))
    t = np.zeros(len(train_set)) + np.nan
    org_idx = X.index[train_index].tolist()
    t[org_idx] = np.squeeze(mean_gs_pred)

    mean_gs_all[count,:] = t
    count += 1

gs_res[2] = mean_gs_all
res_dict[2] = r2_scores





data = train_set
predictorsN =  pred_result[pred]
print(predictorsN)
X = data[predictorsN]
#X_test_t = test_data[predictorsN]

y = data[response]
#y_test = test_data[response]
count = 0
r2_scores = []
mean_gs_all = np.zeros((k, len(train_set)))

for train_index, test_index in kf.split(X):
    print(count)
    X_train_t= X.iloc[train_index]
    scaler = StandardScaler().fit(X_train_t)
    X_train = scaler.transform(X_train_t)
    #X_test = scaler.transform(X_test_t)
    y_train= y.iloc[train_index]
    y_train = y_train.values.reshape(-1, 1)
    coords_train = coords.iloc[train_index]
    selector = Sel_BW(coords_train.values, y_train, X_train, multi=False) # This creates the bandwidths for different input features
    bws = selector.search(verbose=True, search_method='golden_section', max_iter=100) # This searches for the optimal bandwidth (fields of influence)
    mgwr_model = GWR(coords_train.values, y_train, X_train, bws)
    results = mgwr_model.fit() # This fits the model to the data
    scale = results.scale
    residuals = results.resid_response
    mean_gs_pred = mgwr_model.predict(coords_train.values, X_train, scale, residuals).predictions
    r2_scores.append(r2_score(y_train, mean_gs_pred))
    t = np.zeros(len(train_set)) + np.nan
    t[train_index] = np.squeeze(mean_gs_pred)

    mean_gs_all[count,:] = t
    count += 1

gs_res[3] = mean_gs_all
res_dict[3] = r2_scores





with open('k_fold_split_Group_pred.pkl', 'wb') as f:
    pickle.dump(res_dict, f)
    pickle.dump(gs_res,f)



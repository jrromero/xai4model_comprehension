import pandas as pd
import re
import pickle
import numpy as np
import seaborn as sns
import dalex as dx
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import functions_uml as fe

import warnings
warnings.filterwarnings('ignore')

columns_multi = pd.MultiIndex.from_tuples([
    ('Breakdown', 'Ranking'), ('Breakdown', 'Sign'),
    ('Shap', 'Ranking'), ('Shap', 'Sign'),
    ('Lime', 'Ranking'), ('Lime', 'Sign')
])

X_test = pd.read_csv('/home/f52almef/datasets/uml_dummy_x_test.csv', index_col=0)
X_train = pd.read_csv('/home/f52almef/datasets/uml_dummy_x_train.csv', index_col=0)
y_test_csv = pd.read_csv('/home/f52almef/datasets/uml_dummy_y_test.csv', index_col=0)
y_test = pd.Series(y_test_csv['label'], index=y_test_csv.index)
y_train_csv = pd.read_csv('/home/f52almef/datasets/uml_dummy_y_train.csv', index_col=0)
y_train = pd.Series(y_train_csv['label'], index=y_train_csv.index)

with open('/home/f52almef/indices/uml_dummy_indexes.pickle', 'rb') as f:
    loaded_indexes = pickle.load(f)

ind_tp = loaded_indexes['df_tp']
ind_tn = loaded_indexes['df_tn']
ind_fp = loaded_indexes['df_fp']
ind_fn = loaded_indexes['df_fn']

print("No. TP instances: ", len(ind_tp))
print("No. TN instances: ", len(ind_tn))
print("No. FP instances: ", len(ind_fp))
print("No. FN instances: ", len(ind_fn))

svc = SVC(C=300.0, gamma=0.1, probability=True, random_state=42)
model_svc = svc.fit(X_train, y_train)
rf = RandomForestClassifier(min_samples_split=5, n_estimators=300, random_state=42)
model_rf = rf.fit(X_train,y_train)
knn = KNeighborsClassifier(leaf_size=5, n_neighbors=3, p=1, weights='distance')
model_knn = knn.fit(X_train,y_train)

num_features = len (X_test.columns)
top_num_features = 10

models_dict = {
    "SVC": model_svc,
    "RF": model_rf,
    "KNN": model_knn
}

index_dict = {
    "TP": ind_tp,
    "TN": ind_tn,
    "FP": ind_fp,
    "FN": ind_fn
}

metrics_dict, results_dict, top_metric_df, rank_metric_df, sign_metric_df, rank_sign_metric_df = fe.calculate_metrics_for_indices(models_dict, index_dict, X_test, X_train, y_train, num_features, top_num_features )


with open ('uml_dummy_exp_results.pickle', 'wb') as f:
    pickle.dump(results_dict, f)

with open ('uml_dummy_metrics_results.pickle', 'wb') as f:
    pickle.dump(metrics_dict, f)

with open ('uml_dummy_top_metric_results.pickle', 'wb') as f:
    pickle.dump(top_metric_df, f)

with open ('uml_dummy_rank_metric_results.pickle', 'wb') as f:
    pickle.dump(rank_metric_df, f)

with open ('uml_dummy_sign_metric_results.pickle', 'wb') as f:
    pickle.dump(sign_metric_df, f)

with open ('uml_dummy_ranksign_metric_results.pickle', 'wb') as f:
    pickle.dump(rank_sign_metric_df, f)

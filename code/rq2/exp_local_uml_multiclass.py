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

X_test = pd.read_csv('/home/f52almef/datasets/uml_multiclass_x_test.csv', index_col=0)
X_train = pd.read_csv('/home/f52almef/datasets/uml_multiclass_x_train.csv', index_col=0)
y_test_csv = pd.read_csv('/home/f52almef/datasets/uml_multiclass_y_test.csv', index_col=0)
y_test = pd.Series(y_test_csv['label'], index=y_test_csv.index)
y_train_csv = pd.read_csv('/home/f52almef/datasets/uml_multiclass_y_train.csv', index_col=0)
y_train = pd.Series(y_train_csv['label'], index=y_train_csv.index)

with open('/home/f52almef/indices/uml_multiclass_indexes.pickle', 'rb') as f:
    loaded_indexes = pickle.load(f)

ind_comp = loaded_indexes['df_class_comp']
ind_rest = loaded_indexes['df_class_rest']
ind_shop = loaded_indexes['df_class_shop']
ind_emp = loaded_indexes['df_class_emp']
ind_smart = loaded_indexes['df_class_smart']

print("No. Computerarchitecture instances: ", len(ind_comp))
print("No. Restaurant instances: ", len(ind_rest))
print("No. Shopping instances: ", len(ind_shop))
print("No. Employee instances: ", len(ind_emp))
print("No. Smarthouse instances: ", len(ind_smart))

svc = SVC(C=500.0, gamma=0.1, probability=True, random_state=42)
model_svc = svc.fit(X_train, y_train)
rf = RandomForestClassifier(n_estimators=200, random_state=42)
model_rf = rf.fit(X_train,y_train)
knn = KNeighborsClassifier(leaf_size=50, n_neighbors=3, p=1, weights='distance')
model_knn = knn.fit(X_train,y_train)

num_features = len (X_test.columns)
top_num_features = 10

models_dict = {
    "SVC": model_svc,
    "RF": model_rf,
    "KNN": model_knn
}

index_dict = {
    "Computerarchitecture": ind_comp,
    "Restaurant": ind_rest,
    "Shopping": ind_shop,
    "Employee": ind_emp,
    "Smarthouse": ind_smart
}

metrics_dict, results_dict, top_metric_df, rank_metric_df, sign_metric_df, rank_sign_metric_df = fe.calculate_metrics_for_indices(models_dict, index_dict, X_test, X_train, y_train, num_features, top_num_features )


with open ('uml_multiclass_exp_results.pickle', 'wb') as f:
    pickle.dump(results_dict, f)

with open ('uml_multiclass_metrics_results.pickle', 'wb') as f:
    pickle.dump(metrics_dict, f)

with open ('uml_multiclass_top_metric_results.pickle', 'wb') as f:
    pickle.dump(top_metric_df, f)

with open ('uml_multiclass_rank_metric_results.pickle', 'wb') as f:
    pickle.dump(rank_metric_df, f)

with open ('uml_multiclass_sign_metric_results.pickle', 'wb') as f:
    pickle.dump(sign_metric_df, f)

with open ('uml_multiclass_ranksign_metric_results.pickle', 'wb') as f:
    pickle.dump(rank_sign_metric_df, f)

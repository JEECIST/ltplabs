#region imports
import pandas as pd
import h2o
import os
import warnings
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from sklearn.model_selection import train_test_split
from h2o.grid.grid_search import H2OGridSearch
from sklearn.metrics import f1_score, accuracy_score
warnings.filterwarnings("ignore")
#endregion

#region import data
df = pd.read_excel('20240222Dataset.xlsx',engine='openpyxl')
#endregion

#region data preparation
df['data_entrada'] = pd.to_datetime(df['data_entrada'])
df['day_of_week'] = df['data_entrada'].dt.day_name()
y = df['sold']
X = df[['idstore', 'brand_','oldpvp', 'newpvp', 'discount','validade_restante','perc_validade_sku','day_of_week']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
train_df = X_train.copy()
train_df['sold'] = y_train
test_df = X_test.copy()
test_df['sold'] = y_test

# Convert pandas DataFrame to H2OFrame
h2o.init()
train_hf = h2o.H2OFrame(train_df)
test_hf = h2o.H2OFrame(test_df)

# Set categorical variables to as factor
train_hf['day_of_week'] = train_hf['day_of_week'].asfactor()
test_hf['day_of_week'] = test_hf['day_of_week'].asfactor()
train_hf['sold'] = train_hf['sold'].asfactor()
test_hf['sold'] = test_hf['sold'].asfactor()

y = 'sold'
x = train_hf.columns
x.remove(y)
#endregion

#region Model Tunning

gbm = H2OGradientBoostingEstimator(seed=1)

# Define hyperparameters for grid search
hyper_params = {
    'ntrees': [50, 100, 150],  # Number of trees
    'learn_rate': [0.01, 0.05, 0.1],  # Learning rate
    'max_depth': [3, 5, 7]  # Maximum depth of trees
}


gbm_grid = H2OGridSearch(model=gbm, hyper_params=hyper_params)

gbm_grid.train(x=x, y=y, training_frame=train_hf,max_runtime_secs=60)
#endregion

#region Analysis
best_gbm = gbm_grid.get_grid()[0]

predictions = best_gbm.predict(test_hf)

actual_labels = test_hf.as_data_frame()['sold'].values
predicted_labels = predictions.as_data_frame()['predict'].values

f1 = f1_score(actual_labels, predicted_labels)
accuracy = accuracy_score(actual_labels, predicted_labels)
print("F1 score:", f1)
print("Accuracy:", accuracy)
#endregion



#region Explanaibility
#best_gbm.explain(test_hf)
obj = best_gbm.explain(frame=test_hf, render=False, columns=x)
for key in obj.keys():
    if not obj.get(key).get("plots"):
        continue
    plots = obj.get(key).get("plots").keys()
    os.makedirs(f".\\model_explain\\{key}", exist_ok=True)
    for plot in plots:
        fig = obj.get(key).get("plots").get(plot).figure()
        fig.savefig(f".\\model_explain\\{key}\\{plot}.png")

#endregion
# scripts/fit.py

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from category_encoders import CatBoostEncoder
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
import yaml
import os
import joblib


# обучение модели
def fit_model():
    def fill_missing_values(data: pd.DataFrame):
        cols_with_nans = data.isnull().sum()
        cols_with_nans = cols_with_nans[cols_with_nans > 0].index
        for col in cols_with_nans:
            if data[col].dtype in [float, int]:
                fill_value = data[col].mean()
            elif data[col].dtype == 'object':
                fill_value = data[col].mode().iloc[0]
            data[col] = data[col].fillna(fill_value)
        return data
    with open('params.yaml', 'r') as fd:
        params = yaml.safe_load(fd)

	# загрузите результат предыдущего шага: inital_data.csv
  
    data = pd.read_csv('data/initial_data.csv')
    data.drop(columns = params['drop_cols'], inplace = True)
    data = fill_missing_values(data)
    data = data.dropna()
    
	# реализуйте основную логику шага с использованием гиперпараметров
    # обучение модели
    cat_features = data.select_dtypes(include='object')
    potential_binary_features = cat_features.nunique() == 2

    binary_cat_features = cat_features[potential_binary_features[potential_binary_features].index]
    
    other_cat_features = cat_features[potential_binary_features[~potential_binary_features].index]
    
    num_features = data.select_dtypes(['float'])
    

    preprocessor = ColumnTransformer(
        [
        ('binary', OneHotEncoder(drop=params['one_hot_drop']), binary_cat_features.columns.tolist()),
        ('cat', OneHotEncoder(drop=params['one_hot_drop']), other_cat_features.columns.tolist()),
        ('num', StandardScaler(), num_features.columns.tolist())
        ],
        remainder='drop',
        verbose_feature_names_out=False
    )

    #model = CatBoostClassifier(auto_class_weights=params['auto_class_weights'])
    model = LogisticRegression(C=params['l_C'], penalty=params['l_penalty'])


    pipeline = Pipeline(
        [
            ('preprocessor', preprocessor),
            ('model', model)
        ]
    )
    model = pipeline.fit(data, data[params['target_col']])  

	# сохраните обученную модель в models/fitted_model.pkl
    os.makedirs('models', exist_ok=True) # создание директории, если её ещё нет
    with open('models/fitted_model.pkl', 'wb') as fd:
        joblib.dump(model, fd) 
        
# 4 — защищённый вызов главной функции
if __name__ == '__main__':
    fit_model()
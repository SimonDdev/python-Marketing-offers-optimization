import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import MODEL_CONFIG


class ModelTrainer:
    def __init__(self, clients, offers, history):
        self.clients = clients
        self.offers = offers
        self.history = history
        self.scaler = None
        self.model = None

    def add_features(self, data):
        # Ако има transaction_date, изчисляваме дни от покупката
        if 'transaction_date' in data.columns:
            data['transaction_date'] = pd.to_datetime(data['transaction_date'], format="%Y-%m-%d", errors='coerce')
            reference_date = data['transaction_date'].max()
            data['days_since_purchase'] = (reference_date - data['transaction_date']).dt.days
        else:
            data['days_since_purchase'] = 0

        # Обработка на количествените признаци, ако съществуват
        if 'quantity' in data.columns:
            data['quantity'] = pd.to_numeric(data['quantity'], errors='coerce').fillna(0)
        else:
            data['quantity'] = 0

        if 'cross_sell_count' in data.columns:
            data['cross_sell_count'] = pd.to_numeric(data['cross_sell_count'], errors='coerce').fillna(0)
        else:
            data['cross_sell_count'] = 0

        # Добавяне на нови фийчъри
        data['age_group'] = data['age'].apply(lambda x: 0 if x < 30 else (1 if x < 50 else 2))
        data['income_bracket'] = data['income'].apply(lambda x: 0 if x < 40000 else (1 if x < 80000 else 2))
        data['loyal_client'] = data['previous_purchases'].apply(lambda x: 1 if x >= 10 else 0)
        return data

    def preprocess_data(self):
        # Обединяване на данните: history + clients + offers
        data = self.history.merge(self.clients, on='client_id', how='left')
        data = data.merge(self.offers, on='offer_id', how='left')
        data['response_binary'] = data['response'].map({'accepted': 1, 'rejected': 0})
        data = self.add_features(data)
        # Избираме признаците, които се използват при обучението
        feature_cols = ['age', 'income', 'previous_purchases', 'price',
                        'days_since_purchase', 'age_group', 'income_bracket',
                        'loyal_client', 'quantity', 'cross_sell_count']
        features = data[feature_cols]
        labels = data['response_binary']
        return features, labels

    def train_model(self):
        features, labels = self.preprocess_data()

        # Ще използваме Pipeline с StandardScaler, PCA и LogisticRegression
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA()),  # ще търсим оптимален брой компоненти чрез grid search
            ('lr', LogisticRegression(max_iter=MODEL_CONFIG['max_iter']))
        ])

        # Оптимизация чрез GridSearchCV, като претърсваме параметрите на PCA и Logistic Regression
        param_grid = {
            'pca__n_components': [5, 7, 10],  # Пробваме да намалим параметрите до 5, 7 или 10 компоненти
            'lr__C': [0.01, 0.1, 1, 10, 100],
            'lr__solver': ['lbfgs', 'saga', 'newton-cg']
        }
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc')
        grid_search.fit(features, labels)
        self.model = grid_search.best_estimator_
        self.scaler = self.model.named_steps['scaler']

        cv_scores = cross_val_score(self.model, features, labels, cv=5, scoring='roc_auc')
        print("Cross-validation ROC AUC scores with PCA:", cv_scores)
        print("Average ROC AUC:", np.mean(cv_scores))
        print("Best parameters:", grid_search.best_params_)
        return self.model, self.scaler







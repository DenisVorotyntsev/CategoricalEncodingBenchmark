import pandas as pd
import numpy as np
from tqdm import tqdm
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

from utils import MultipleEncoder, DoubleValidationEncoderNumerical

from utils import get_single_encoder, cat_cols_info


class Model:
    def __init__(self, cat_validation="None", encoders_names=None, cat_cols=None,
                 model_validation=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                 model_params=None):
        self.cat_validation = cat_validation
        self.encoders_names = encoders_names
        self.cat_cols = cat_cols
        self.model_validation = model_validation

        if model_params is None:
            self.model_params = {"metrics": "AUC", "n_estimators": 5000, "learning_rate": 0.02, "random_state": 42}
        else:
            self.model_params = model_params

        self.encoders_list = []
        self.models_list = []
        self.scores_list_train = []
        self.scores_list_val = []

    def fit(self, X: pd.DataFrame, y: np.array) -> tuple:
        # process cat cols
        if self.cat_validation == "None":
            encoder = MultipleEncoder(cols=self.cat_cols, encoders_names_tuple=self.encoders_names)
            X = encoder.fit_transform(X, y)

        for n_fold, (train_idx, val_idx) in enumerate(self.model_validation.split(X, y)):
            X_train, X_val = X.loc[train_idx].reset_index(drop=True), X.loc[val_idx].reset_index(drop=True)
            y_train, y_val = y[train_idx], y[val_idx]
            print(f"shapes before encoder : ", X_train.shape, X_val.shape)

            if self.cat_validation == "Single":
                encoder = MultipleEncoder(cols=self.cat_cols, encoders_names_tuple=self.encoders_names)
                X_train = encoder.fit_transform(X_train, y_train)
                X_val = encoder.transform(X_val)
            if self.cat_validation == "Double":
                encoder = DoubleValidationEncoderNumerical(cols=self.cat_cols, encoders_names_tuple=self.encoders_names)
                X_train = encoder.fit_transform(X_train, y_train)
                X_val = encoder.transform(X_val)
                pass
            self.encoders_list.append(encoder)

            # check for OrdinalEncoder encoding
            for col in [col for col in X_train.columns if "OrdinalEncoder" in col]:
                X_train[col] = X_train[col].astype("category")
                X_val[col] = X_val[col].astype("category")

            # fit model
            print(f"shapes before model : ", X_train.shape, X_val.shape)
            model = LGBMClassifier(**self.model_params)
            model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)],
                      verbose=100, early_stopping_rounds=100)
            self.models_list.append(model)

            y_hat = model.predict_proba(X_train)[:, 1]
            score_train = roc_auc_score(y_train, y_hat)
            self.scores_list_train.append(score_train)
            y_hat = model.predict_proba(X_val)[:, 1]
            score_val = roc_auc_score(y_val, y_hat)
            self.scores_list_val.append(score_val)

            print(f"AUC on {n_fold} fold train : {np.round(score_train, 4)}\n\n ")
            print(f"AUC on {n_fold} fold val : {np.round(score_val, 4)}\n\n ")

        mean_score_train = np.mean(self.scores_list_train)
        mean_score_val = np.mean(self.scores_list_val)
        print(f"\n\n Mean score train : {np.round(mean_score_train, 4)}\n\n ")
        print(f"\n\n Mean score val : {np.round(mean_score_val, 4)}\n\n ")
        return mean_score_train, mean_score_val

    def predict(self, X: pd.DataFrame) -> np.array:
        y_hat = np.zeros(X.shape[0])
        for encoder, model in zip(self.encoders_list, self.models_list):
            X_test = X.copy()
            X_test = encoder.transform(X_test)

            # check for OrdinalEncoder encoding
            for col in [col for col in X_test.columns if "OrdinalEncoder" in col]:
                X_test[col] = X_test[col].astype("category")

            unranked_preds = model.predict_proba(X_test)[:, 1]
            y_hat += rankdata(unranked_preds)
        return y_hat, X_test.shape[1]

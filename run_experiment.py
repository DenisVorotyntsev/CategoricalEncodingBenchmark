import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from model import Model
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import itertools

from utils import MultipleEncoder, cat_cols_info, save_dict_to_file


def execute_experiment(dataset_name, encoders_list, validation_type, file_name_apex, experiment_description):
    dataset_pth = f"./data/{dataset_name}/{dataset_name}.gz"
    results = {}

    # training params
    N_SPLITS = 5
    model_validation = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    encoder_validation = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=2019)

    # load processed dataset
    data = pd.read_csv(dataset_pth)

    # make train-test split
    cat_cols = [col for col in data.columns if col.startswith("cat")]
    X_train, X_test, y_train, y_test = train_test_split(data.drop("target", axis=1), data["target"],
                                                        test_size=0.4, shuffle=False)
    X_train, X_test = X_train.reset_index(drop=False), X_test.reset_index(drop=False)
    y_train, y_test = np.array(y_train), np.array(y_test)

    results[dataset_name] = {}
    results[dataset_name]["info"] = {
        "experiment_description": experiment_description,
        "train_shape": X_train.shape, "test_shape": X_test.shape,
        "mean_target_train": np.mean(y_train), "mean_target_test": np.mean(y_test),
        "num_cat_cols": len(cat_cols), "cat_cols_info": cat_cols_info(X_train, X_test, cat_cols),
                                 }

    for encoders_tuple in encoders_list:
        print(f"\n\nCurrent itteration : {encoders_tuple}, {dataset_name}\n\n")

        time_start = time.time()

        # train models
        lgb_model = Model(cat_validation=validation_type, encoders_names=encoders_tuple, cat_cols=cat_cols)
        train_score, val_score, avg_num_trees= lgb_model.fit(X_train, y_train)
        y_hat, test_features = lgb_model.predict(X_test)
        #file_pth = f"./preds/{file_name_apex}_{dataset_name}_{str(encoders_tuple)}.csv"
        #pd.DataFrame({"predictions": y_hat}).to_csv(file_pth, index=False)

        # check score
        test_score = roc_auc_score(y_test, y_hat)
        time_end = time.time()

        # write and save results
        results[dataset_name][str(encoders_tuple)] = {"train_score": train_score,
                                                      "val_score": val_score,
                                                      "test_score": test_score,
                                                      "time": time_end-time_start,
                                                      "features_before_encoding": X_train.shape[1],
                                                      "features_after_encoding": test_features,
                                                      "avg_tress_number": avg_num_trees
                                                      }

    for k, v in results[dataset_name].items():
        print(k, v, "\n\n")

    save_dict_to_file(dic=results[dataset_name], path=f"./results/{file_name_apex}{dataset_name}.txt", save_raw=False)
    save_dict_to_file(dic=results[dataset_name], path=f"./results/{file_name_apex}{dataset_name}_r.txt", save_raw=True)


if __name__ == "__main__":
    encoders_list = [
        ("HelmertEncoder",),  # non double
        ("SumEncoder",),  # non double
        ("LeaveOneOutEncoder",),
        ("FrequencyEncoder",),
        ("MEstimateEncoder",),
        ("TargetEncoder",),
        ("WOEEncoder",),
        ("BackwardDifferenceEncoder",),  # non double
        ("JamesSteinEncoder",),
        ("OrdinalEncoder",),
        ("CatBoostEncoder",),
    ]

    encoders_list_double = [
        ("LeaveOneOutEncoder",),
        ("FrequencyEncoder",),
        ("MEstimateEncoder",),
        ("TargetEncoder",),
        ("WOEEncoder",),
        ("JamesSteinEncoder",),
        ("CatBoostEncoder",),
    ]

    dataset_list = [
        "telecom", "adult", "employee", "credit", "mortgages",
        "poverty_A","poverty_B", "poverty_C",
        "promotion", "kdd_upselling", "taxi", "kick"
    ]

    for dataset_name in dataset_list:
        validation_type = "None"
        file_name_apex = "exp_1_"
        experiment_description = f"Check single encoder, {validation_type} validation"
        execute_experiment(dataset_name, encoders_list, validation_type, file_name_apex, experiment_description)

        validation_type = "Single"
        file_name_apex = "exp_2_"
        experiment_description = f"Check single encoder, {validation_type} validation"
        execute_experiment(dataset_name, encoders_list, validation_type, file_name_apex, experiment_description)

        validation_type = "Double"
        file_name_apex = "exp_3_"
        experiment_description = f"Check single encoder, {validation_type} validation"
        execute_experiment(dataset_name, encoders_list_double, validation_type, file_name_apex, experiment_description)


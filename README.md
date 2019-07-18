# CategoricalEncodingBenchmark
Benchmarking different approaches for categorical encoding  

# Reproducibility of results

### Requirements 

```
numpy==1.15.1
pandas==0.23.4
sklearn==0.20.3
category_encoders==2.0.0
lightgbm==2.2.3
```

### Benchmark the dataset 

To benchmark encoders for your dataset: 

1. Install libraries in requirements

2. Process the dataset as in `prepare_datasets.ipynb`

3. Add name of the dataset in `dataset_list` in `run_experiment.py`

4. `python run_experiment.py`

5. Run `show_results.ipynb`


# Used datasets and raw scores 

All datasets except poverty_A(B,C) came from different domains; they have a different number of observations, number of categorical and numerical features. 
The objective for all datasets - binary classification. 
Preprocessing of datasets were simple: I removed all time-based columns from datasets. 
Remaining columns were either categorical or numerical. 
Details of the experiments could be found in my blog post: [Benchmarking Categorical Encoders](https://towardsdatascience.com/benchmarking-categorical-encoders-9c322bd77ee8). 

**Table 1.1** Used datasets 

| Name | Total points | Train points | Test points | Number of features | Number of categorical features | Short description | 
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| [Telecom](https://www.kaggle.com/blastchar/telco-customer-churn)   | 7.0k | 4.2k |  2.8k |  20   |  16  | Churn prediction for telecom data |
| [Adult](https://www.kaggle.com/wenruliu/adult-income-dataset)   | 48.8k | 29.3k | 19.5k  |  15  | 8 | Predict if persons' income is bigger 50k |
| [Employee](https://www.kaggle.com/c/amazon-employee-access-challenge/data)   | 32.7k | 19.6k | 13.1k  | 10  | 9 | Predict an employee's access needs, given his/her job role|
| [Credit](https://www.kaggle.com/c/home-credit-default-risk/data)   | 307.5k | 184.5k | 123k  |  121  | 18 | Loan repayment |
| [Mortgages](https://www.crowdanalytix.com/contests/propensity-to-fund-mortgages)   |  45.6k | 27.4k | 18.2k | 20 | 9 | Predict if house mortgage is founded |
| [Promotion](https://datahack.analyticsvidhya.com/contest/wns-analytics-hackathon-2018/)   |   54.8   | 32.8k | 21.9k  |  13  | 5  | Predict if an employee will get a promotion |
| [Kick](https://www.kaggle.com/c/DontGetKicked/data)   | 72.9k | 43.7k | 29.1k | 32 | 19 | Predict if a car purchased at auction is good/bad buy |
| [Kdd_upselling](https://www.kdd.org/kdd-cup/view/kdd-cup-2009/Data)   |   50k   | 30k | 20k |  230  |  40 | Predict up-selling for a customer |
| [Taxi](https://www.crowdanalytix.com/contests/mckinsey-big-data-hackathon) | 892.5k | 535.5k | 357k | 8 | 5 | Predict the probability of an offer being accepted by a certain driver |
| [Poverty_A](https://www.drivendata.org/competitions/50/worldbank-poverty-prediction/page/99/)   | 37.6k | 22.5k | 15.0k | 41 | 38 | Predict whether or not a given household for a given country is poor or not |
| [Poverty_B](https://www.drivendata.org/competitions/50/worldbank-poverty-prediction/page/99/)   | 20.2k | 12.1k | 8.1k | 224 | 191 | Predict whether or not a given household for a given country is poor or not |
| [Poverty_C](https://www.drivendata.org/competitions/50/worldbank-poverty-prediction/page/99/)   | 29.9k | 17.9k | 11.9k  | 41  | 35 | Predict whether or not a given household for a given country is poor or not |

The ROC AUC scores for each dataset are presented in tables below. 
**Note**: some experiments required too much memory to run, so some values are missing. 

**Table 1.2** ROC AUC scores for None Validation 

|                           |   telecom |   adult |   employee |   credit |   mortgages |   promotion | kick   | kdd_upselling   | taxi   |   poverty_A |   poverty_B |   poverty_C |
|:--------------------------|:----------:|:--------:|:-----------:|:---------:|:------------:|:------------:|:-------:|:----------------:|:-------:|:------------:|:------------:|:------------:|
| BackwardDifferenceEncoder |    0.6454 |  0.8555 |     0.5006 |   0.7442 |      0.5997 |      0.6482 |        |                 |        |      0.5149 |      0.5484 |      0.4945 |
| CatBoostEncoder           |    0.7666 |  0.868  |     0.5004 |   0.7478 |      0.6279 |      0.7811 | 0.6583 | 0.8549          | 0.5477 |      0.5179 |      0.5638 |      0.5427 |
| FrequencyEncoder          |    **0.8405** |  0.9291 |     0.807  |   0.7593 |      0.6949 |      0.9052 | **0.7907** | **0.8643**          | **0.5656** |      0.7276 |      0.6164 |      0.7177 |
| HelmertEncoder            |    0.8404 |  **0.9297** |     **0.83**   |   **0.7601** |      **0.7001** |      **0.9079** |        |                 |        |      0.7325 |      **0.6343** |      0.7168 |
| JamesSteinEncoder         |    0.7195 |  0.8688 |     0.5003 |   0.7485 |      0.6049 |      0.7984 | 0.6592 | 0.8516          | 0.5432 |      0.4918 |      0.5304 |      0.4836 |
| LeaveOneOutEncoder        |    0.5    |  0.5214 |     0.6233 |   0.4957 |      0.5    |      0.5457 | 0.5027 | 0.5             | 0.5    |      0.5006 |      0.5002 |      0.4527 |
| MEstimateEncoder          |    0.6944 |  0.8617 |     0.4998 |   0.7368 |      0.6086 |      0.8156 | 0.653  | 0.8448          | 0.5091 |      0.5254 |      0.434  |      0.4528 |
| OrdinalEncoder            |    0.7409 |  0.8616 |     0.501  |   0.7445 |      0.6008 |      0.7124 | 0.6531 | 0.8448          | 0.5498 |      0.473  |      0.4683 |      0.5611 |
| SumEncoder                |    0.8404 |  0.929  |     0.8053 |   0.7593 |      0.6944 |      0.9073 |        |                 |        |      **0.7355** |      0.6206 |      **0.7372** |
| TargetEncoder             |    0.7195 |  0.8696 |     0.5003 |   0.7483 |      0.6064 |      0.7971 | 0.6594 | 0.8483          | 0.5428 |      0.4955 |      0.5401 |      0.4751 |
| WOEEncoder                |    0.7056 |  0.8645 |     0.5012 |   0.7439 |      0.615  |      0.7345 | 0.6398 | 0.844           | 0.5485 |      0.478  |      0.5356 |      0.4671 |

**Table 1.3** ROC AUC scores for Single Validation

|                           |   telecom |   adult |   employee |   credit |   mortgages |   promotion | kick   | kdd_upselling   | taxi   |   poverty_A |   poverty_B |   poverty_C |
|:--------------------------|:----------:|:--------:|:-----------:|:---------:|:------------:|:------------:|:-------:|:----------------:|:-------:|:------------:|:------------:|:------------:|
| BackwardDifferenceEncoder |    0.8382 |  0.9293 |     0.7569 |   0.7595 |      0.6894 |      0.9064 |        |                 |        |      0.7323 |      0.6151 |      0.7108 |
| CatBoostEncoder           |    0.8392 |  0.9292 |     **0.8498** |   0.7594 |      0.6951 |      0.8918 | 0.7901 | **0.8654**          | 0.5844 |      **0.7429** |      **0.6902** |      0.7333 |
| FrequencyEncoder          |    0.8392 |  0.9293 |     0.8138 |   0.7592 |      0.6937 |      0.9055 | **0.7902** | 0.8634          | 0.582  |      0.7302 |      0.6128 |      0.7195 |
| HelmertEncoder            |    **0.8404** |  0.9297 |     0.8344 |   0.7597 |      **0.7027** |      **0.9083** |        |                 |        |      0.7297 |      0.6374 |      0.7196 |
| JamesSteinEncoder         |    0.8388 |  0.9292 |     0.7817 |   0.7597 |      0.667  |      0.9053 | 0.5835 | 0.726           | 0.5898 |      0.7303 |      0.6764 |      0.7217 |
| LeaveOneOutEncoder        |    0.5    |  0.5182 |     0.6121 |   0.4997 |      0.5    |      0.5403 | 0.4682 | 0.5             | 0.5    |      0.5103 |      0.5    |      0.4959 |
| MEstimateEncoder          |    0.8394 |  0.929  |     0.7353 |   0.7593 |      0.6957 |      0.9054 | 0.5877 | 0.5953          | 0.5946 |      0.7302 |      0.6493 |      0.7076 |
| OrdinalEncoder            |    **0.8404** |  **0.9299** |     0.8274 |   0.7585 |      0.6917 |      0.9078 | 0.7809 | 0.8465          | **0.6034** |      0.7337 |      0.6635 |      **0.742**  |
| SumEncoder                |    **0.8404** |  0.929  |     0.8053 |   0.7593 |      0.6944 |      0.9073 |        |                 |        |      0.7355 |      0.6206 |      0.7372 |
| TargetEncoder             |    0.8388 |  0.9293 |     0.815  |   **0.7599** |      0.6702 |      0.9057 | 0.7042 | 0.713           | 0.5894 |      0.7292 |      0.6742 |      0.7207 |
| WOEEncoder                |    0.8393 |  0.9294 |     0.8325 |   **0.7599** |      0.6801 |      0.9056 | 0.7172 | 0.8391          | 0.5903 |      0.7279 |      0.6737 |      0.7224 |

**Table 1.4** ROC AUC scores for Double Validation

|                    |   telecom |   adult |   employee |   credit |   mortgages |   promotion |   kick |   kdd_upselling |   taxi |   poverty_A |   poverty_B |   poverty_C |
|:-------------------|:----------:|:--------:|:-----------:|:---------:|:------------:|:------------:|:-------:|:----------------:|:-------:|:------------:|:------------:|:------------:|
| CatBoostEncoder    |    0.8394 |  0.9293 |     0.8529 |   0.7592 |      0.6967 |      0.9056 | 0.7899 |          0.8633 | 0.6031 |      **0.7418** |      0.6902 |      0.7343 |
| FrequencyEncoder   |    0.8371 |  0.9221 |     0.5563 |   0.755  |      0.6582 |      0.8749 | 0.7655 |          0.8551 | 0.5657 |      0.6873 |      0.6037 |      0.6961 |
| JamesSteinEncoder  |    0.8398 |  **0.9296** |     0.8489 |   0.7598 |      **0.6981** |      0.905  | 0.7901 |          0.8628 | **0.6033** |      0.7412 |      0.6895 |      **0.7366** |
| LeaveOneOutEncoder |    0.8393 |  0.9295 |     0.8496 |   0.7595 |      0.6963 |      0.9055 | 0.7902 |          0.8635 | 0.602  |      0.7416 |      **0.6931** |      0.7345 |
| MEstimateEncoder   |    **0.8405** |  0.9292 |     0.8125 |   0.7597 |      0.6939 |      **0.9063** | 0.7881 |          0.863  | 0.5984 |      0.7375 |      0.6801 |      0.7204 |
| TargetEncoder      |    0.8393 |  0.9294 |     **0.8537** |   0.7596 |      0.6954 |      0.9057 | **0.7909** |          **0.8643** | 0.6025 |      0.7415 |      0.6903 |      0.7352 |
| WOEEncoder         |    0.8401 |  0.9294 |     0.824  |   **0.7599** |      0.6977 |      0.9041 | 0.7905 |          0.8631 | 0.6011 |      0.7407 |      0.6911 |      0.7345 |

# Results

To determine the best encoder, I scaled the ROC AUC scores of each dataset (min-max scale) and then averaged results among the encoder. 
The obtained result represents the average performance score for each encoder (higher is better). 
The encoders performance scores for each type of validation are shown in tables 2.1–2.3. 


To determine the best validation strategy, I compared the top score of each dataset for each type of validation. 
The scores improvement (top score for a dataset and an average score for encoder) are shown in table 2.4 and 2.5 below.

**Table 2.1** Encoders performance scores - None Validation

|                           |      None Validation |
|:--------------------------|:-------:|
| HelmertEncoder            | 0.9517 |
| SumEncoder                | 0.9434 |
| FrequencyEncoder          | 0.9176 |
| CatBoostEncoder           | 0.5728 |
| TargetEncoder             | 0.5174 |
| JamesSteinEncoder         | 0.5162 |
| OrdinalEncoder            | 0.4964 |
| WOEEncoder                | 0.4905 |
| MEstimateEncoder          | 0.4501 |
| BackwardDifferenceEncoder | 0.4128 |
| LeaveOneOutEncoder        | 0.0697 |

**Table 2.2** Encoders performance scores - Single Validation

|                           |      Single Validation |
|:--------------------------|:-------:|
| CatBoostEncoder           | 0.9726 |
| OrdinalEncoder            | 0.9694 |
| HelmertEncoder            | 0.9558 |
| SumEncoder                | 0.9434 |
| WOEEncoder                | 0.9326 |
| FrequencyEncoder          | 0.9315 |
| BackwardDifferenceEncoder | 0.9108 |
| TargetEncoder             | 0.8915 |
| JamesSteinEncoder         | 0.8555 |
| MEstimateEncoder          | 0.8189 |
| LeaveOneOutEncoder        | 0.0729 |

**Table 2.3** Encoders performance scores - Double Validation

|                    |      Double Validation |
|:-------------------|:-------:|
| JamesSteinEncoder  | 0.9918 |
| CatBoostEncoder    | 0.9917 |
| TargetEncoder      | 0.9916 |
| LeaveOneOutEncoder | 0.9909 |
| WOEEncoder         | 0.9838 |
| MEstimateEncoder   | 0.9686 |
| FrequencyEncoder   | 0.8018 |

**Table 2.4** Top score improvement (percent)

|               |   None -> Single |   Single -> Double |
|:--------------|:-----------------:|:-------------------:|
| telecom       |            0.00    |               0.01 |
| adult         |             0.02 |              -0.03 |
| employee      |             1.98 |               0.39 |
| credit        |            -0.01 |              -0.00    |
| mortgages     |             0.26 |              -0.47 |
| promotion     |             0.04 |              -0.20  |
| kick          |            -0.05 |               0.06 |
| kdd_upselling |             0.10  |              -0.11 |
| taxi          |             3.78 |              -0.01 |
| poverty_A     |             0.74 |              -0.11 |
| poverty_B     |             5.59 |               0.29 |
| poverty_C     |             0.48 |              -0.54 |


**Table 2.5** Encoders performance scores improvement (percent)

|                           |   None -> Single | Single -> Double   |
|:--------------------------|:-----------------:|:-------------------:|
| BackwardDifferenceEncoder |             27.20 |                    |
| CatBoostEncoder           |             20.10 | 0.40                |
| FrequencyEncoder          |              0.30 | -4.90               |
| HelmertEncoder            |              0.20 |                    |
| JamesSteinEncoder         |             17.70 | 6.30                |
| LeaveOneOutEncoder        |              0.20 | 53.20               |
| MEstimateEncoder          |             18.90 | 8.10                |
| OrdinalEncoder            |             24.10 |                    |
| SumEncoder                |              0.00   |                    |
| TargetEncoder             |             19.60 | 4.20                |
| WOEEncoder                |             23.40 | 1.90                |



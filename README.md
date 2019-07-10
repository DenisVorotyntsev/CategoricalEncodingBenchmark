# CategoricalEncodingBenchmark
Benchmarking different approaches for categorical encoding  

# Used datasets 

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

# Raw scores 

Table 1.1 None validation 

|                           |   telecom |   adult |   employee |   credit |   mortgages |   promotion | kick   | kdd_upselling   | taxi   |   poverty_A |   poverty_B |   poverty_C |
|:--------------------------|:----------:|:--------:|:-----------:|:---------:|:------------:|:------------:|:-------:|:----------------:|:-------:|:------------:|:------------:|:------------:|
| BackwardDifferenceEncoder |     0.645 |   0.855 |      0.501 |    0.744 |       0.6   |       0.648 |        |                 |        |       0.515 |       0.548 |       0.495 |
| CatBoostEncoder           |     0.767 |   0.868 |      0.5   |    0.748 |       0.628 |       0.781 | 0.658  | 0.855           | 0.548  |       0.518 |       0.564 |       0.543 |
| FrequencyEncoder          |     0.84  |   0.929 |      0.807 |    0.759 |       0.695 |       0.905 | 0.791  | 0.864           | 0.566  |       0.728 |       0.616 |       0.718 |
| HelmertEncoder            |     0.84  |   0.93  |      0.83  |    0.76  |       0.7   |       0.908 |        |                 |        |       0.732 |       0.634 |       0.717 |
| JamesSteinEncoder         |     0.72  |   0.869 |      0.5   |    0.749 |       0.605 |       0.798 | 0.659  | 0.852           | 0.543  |       0.492 |       0.53  |       0.484 |
| LeaveOneOutEncoder        |     0.5   |   0.521 |      0.623 |    0.496 |       0.5   |       0.546 | 0.503  | 0.5             | 0.5    |       0.501 |       0.5   |       0.453 |
| MEstimateEncoder          |     0.694 |   0.862 |      0.5   |    0.737 |       0.609 |       0.816 | 0.653  | 0.845           | 0.509  |       0.525 |       0.434 |       0.453 |
| OrdinalEncoder            |     0.741 |   0.862 |      0.501 |    0.744 |       0.601 |       0.712 | 0.653  | 0.845           | 0.55   |       0.473 |       0.468 |       0.561 |
| SumEncoder                |     0.84  |   0.929 |      0.805 |    0.759 |       0.694 |       0.907 |        |                 |        |       0.735 |       0.621 |       0.737 |
| TargetEncoder             |     0.72  |   0.87  |      0.5   |    0.748 |       0.606 |       0.797 | 0.659  | 0.848           | 0.543  |       0.495 |       0.54  |       0.475 |
| WOEEncoder                |     0.706 |   0.864 |      0.501 |    0.744 |       0.615 |       0.734 | 0.64   | 0.844           | 0.548  |       0.478 |       0.536 |       0.467 |

Table 1.2 Single Validation

|                           |   telecom |   adult |   employee |   credit |   mortgages |   promotion | kick   | kdd_upselling   | taxi   |   poverty_A |   poverty_B |   poverty_C |
|:--------------------------|:----------:|:--------:|:-----------:|:---------:|:------------:|:------------:|:-------:|:----------------:|:-------:|:------------:|:------------:|:------------:|
| BackwardDifferenceEncoder |     0.838 |   0.929 |      0.757 |    0.76  |       0.689 |       0.906 |        |                 |        |       0.732 |       0.615 |       0.711 |
| CatBoostEncoder           |     0.839 |   0.929 |      0.85  |    0.759 |       0.695 |       0.892 | 0.79   | 0.865           | 0.584  |       0.743 |       0.69  |       0.733 |
| FrequencyEncoder          |     0.839 |   0.929 |      0.814 |    0.759 |       0.694 |       0.906 | 0.79   | 0.863           | 0.582  |       0.73  |       0.613 |       0.72  |
| HelmertEncoder            |     0.84  |   0.93  |      0.834 |    0.76  |       0.703 |       0.908 |        |                 |        |       0.73  |       0.637 |       0.72  |
| JamesSteinEncoder         |     0.839 |   0.929 |      0.782 |    0.76  |       0.667 |       0.905 | 0.583  | 0.726           | 0.59   |       0.73  |       0.676 |       0.722 |
| LeaveOneOutEncoder        |     0.5   |   0.518 |      0.612 |    0.5   |       0.5   |       0.54  | 0.468  | 0.5             | 0.5    |       0.51  |       0.5   |       0.496 |
| MEstimateEncoder          |     0.839 |   0.929 |      0.735 |    0.759 |       0.696 |       0.905 | 0.588  | 0.595           | 0.595  |       0.73  |       0.649 |       0.708 |
| OrdinalEncoder            |     0.84  |   0.93  |      0.827 |    0.759 |       0.692 |       0.908 | 0.781  | 0.847           | 0.603  |       0.734 |       0.663 |       0.742 |
| SumEncoder                |     0.84  |   0.929 |      0.805 |    0.759 |       0.694 |       0.907 |        |                 |        |       0.735 |       0.621 |       0.737 |
| TargetEncoder             |     0.839 |   0.929 |      0.815 |    0.76  |       0.67  |       0.906 | 0.704  | 0.713           | 0.589  |       0.729 |       0.674 |       0.721 |
| WOEEncoder                |     0.839 |   0.929 |      0.832 |    0.76  |       0.68  |       0.906 | 0.717  | 0.839           | 0.59   |       0.728 |       0.674 |       0.722 |

Table 1.3 Double Validation

|                    |   telecom |   adult |   employee |   credit |   mortgages |   promotion |   kick |   kdd_upselling |   taxi |   poverty_A |   poverty_B |   poverty_C |
|:-------------------|:----------:|:--------:|:-----------:|:---------:|:------------:|:------------:|:-------:|:----------------:|:-------:|:------------:|:------------:|:------------:|
| CatBoostEncoder    |     0.839 |   0.929 |      0.853 |    0.759 |       0.697 |       0.906 |  0.79  |           0.863 |  0.603 |       0.742 |       0.69  |       0.734 |
| FrequencyEncoder   |     0.837 |   0.922 |      0.556 |    0.755 |       0.658 |       0.875 |  0.766 |           0.855 |  0.566 |       0.687 |       0.604 |       0.696 |
| JamesSteinEncoder  |     0.84  |   0.93  |      0.849 |    0.76  |       0.698 |       0.905 |  0.79  |           0.863 |  0.603 |       0.741 |       0.69  |       0.737 |
| LeaveOneOutEncoder |     0.839 |   0.929 |      0.85  |    0.759 |       0.696 |       0.906 |  0.79  |           0.863 |  0.602 |       0.742 |       0.693 |       0.735 |
| MEstimateEncoder   |     0.841 |   0.929 |      0.812 |    0.76  |       0.694 |       0.906 |  0.788 |           0.863 |  0.598 |       0.737 |       0.68  |       0.72  |
| TargetEncoder      |     0.839 |   0.929 |      0.854 |    0.76  |       0.695 |       0.906 |  0.791 |           0.864 |  0.603 |       0.741 |       0.69  |       0.735 |
| WOEEncoder         |     0.84  |   0.929 |      0.824 |    0.76  |       0.698 |       0.904 |  0.79  |           0.863 |  0.601 |       0.741 |       0.691 |       0.734 |


# Results


Table 2.1 Encoders performance scores - None Validation

|                           | None Validation |
|:--------------------------|:------:|
| HelmertEncoder            | 0.952 |
| SumEncoder                | 0.943 |
| FrequencyEncoder          | 0.918 |
| CatBoostEncoder           | 0.573 |
| TargetEncoder             | 0.517 |
| JamesSteinEncoder         | 0.516 |
| OrdinalEncoder            | 0.496 |
| WOEEncoder                | 0.490  |
| MEstimateEncoder          | 0.450  |
| BackwardDifferenceEncoder | 0.413 |
| LeaveOneOutEncoder        | 0.070  |

Table 2.2 Encoders performance scores - Single Validation

|                           | Single Validation |
|:--------------------------|:------:|
| CatBoostEncoder           | 0.973 |
| OrdinalEncoder            | 0.969 |
| HelmertEncoder            | 0.956 |
| SumEncoder                | 0.943 |
| WOEEncoder                | 0.933 |
| FrequencyEncoder          | 0.931 |
| BackwardDifferenceEncoder | 0.911 |
| TargetEncoder             | 0.891 |
| JamesSteinEncoder         | 0.856 |
| MEstimateEncoder          | 0.819 |
| LeaveOneOutEncoder        | 0.073 |

Table 2.3 Encoders performance scores - Double Validation

|                    | Double Validation |
|:-------------------|:------:|
| JamesSteinEncoder  | 0.992 |
| CatBoostEncoder    | 0.992 |
| TargetEncoder      | 0.992 |
| LeaveOneOutEncoder | 0.991 |
| WOEEncoder         | 0.984 |
| MEstimateEncoder   | 0.969 |
| FrequencyEncoder   | 0.802 |

Table 2.4 Top score improvement (percent)

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


Table 2.5 Encoders performance scores improvement (percent)

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



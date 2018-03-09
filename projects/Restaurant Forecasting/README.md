# Recruit Restaurant Visitor Forecasting

Dimas Adinugroho
March 8th, 2018

This project provides a simple solution to [Recruit Restaurant Visitor Forecasting](https://www.kaggle.com/c/recruit-restaurant-visitor-forecasting) on Kaggle. It is a public competition and availabel for everyone. Rather than focusing to become winner in the competition which requires good hardware and amount of time, this project aim to give walkthrough the process of using different approaches (Statistical and Machine Learning) to solve a time-series problems. This will get beginners to understand the basic knowledge of analysis time-series data and can help them to apply in the practice.

## Problem Statement

Kaggle is a platform for data science competition, where many people trying to solve company's problem by building predictive modeling and produce the best model. This project is one of the kaggle competition that involve many Japanese restaurants and help them forecasting how many customers to expect each day in the future. The forecasting is necessary for restaurant's owner because this forecasting is an important aid the effectively and efficiently planning the schedule staff members and purchase ingredients. The forecasting won't be easy to make because many unexpected variables affect the visitor's judgment, for example, weather condition, preferences, date, popularity, etc. Some things are easier to forecast than others. How much data is available and how further are we are going to forecast will affect the forecasting model we build. It will be more difficult when the restaurant only has little data.

## Software Requirements
This project uses the following software:

-   **Python stack**: python 3.6, numpy, sklearn, scipy, pandas, matplotlib, seaborn, and statsmodel.
-   **XGBoost package**: multi-threaded xgboost should be compiled, xgboost python package is also required.

## Data Requirements
The [dataset](https://www.kaggle.com/c/recruit-restaurant-visitor-forecasting/data) is already publicly accessible on Kaggle sites. The data comes from many .csv and need to be download separately (total: 71.3 MB ). 

## Evaluation Metrics

Every kaggle competition have different quality metrics that used by participants to evaluate their model performance. Restaurant Forecasting competition features Root Mean Squared Logarithmic Error [(RMSLE)](https://www.quora.com/What-is-the-difference-between-an-RMSE-and-RMSLE-logarithmic-error-and-does-a-high-RMSE-imply-low-RMSLE). RMSLE used to find out the difference between predicted values and the actual one. RMSLE usually used when we don't want to penalize huge differences in the predicted and actual values when both of them are huge numbers.

The RMSLE is calculated as:

$$\sqrt{ \frac{1}{N} \sum_{i=1}^{n} (log(P_i + 1) - log(\alpha_i + 1))^2 }$$

where:
$n$  is the total number of observations.
$P_i$ is your prediction of visitors.
$\alpha_i$ is the actual number of visitors.
$log(x)$ is the natural logarithm of x

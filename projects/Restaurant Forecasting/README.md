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


## Project Design
This project is divided into 3 sections:

### Data Exploration:
Before making some predictions, we need to properly understand and get the insight from Restaurants Forecasting's dataset.  By doing data exploration, we poke around the data and getting a sense of what happen in the data. Due to data comes from different sources, we might have to use query and merge data  to get more insight. Also, to make better model, it will be useful to get some basic statistics, plot the data, and understand the features that correlated to the customer visits.

###  Statistical Approach
To forecast something, we need to understand the several factors that affect it. A good forecasting captures the patterns and relationship between variables and historical data that impact future events, not only random fluctuation or noise. To understand the pattern and behaviors in time-series data, we need to split the data into several components and then analyze each component.

Most of the statistical approach only use historical data to predict the future. By identifying the historical pattern, we can reconstruct data and use it to forecast the future.  ARIMA model able to explain the trend and seasonality pattern from the historical data and reconstructed it. It is a popular and traditional method for forecasting time series data. Usually, when people come into time-series problem, this approach is preferable and already widely used.

We start with understanding the pattern from ACF and PACF plot. ACF and PACF is a tool that used to identify the historical pattern from time series, how much related is the data from its past. By understanding this plot, we can try to do a simple forecasting model. After that, we try to tune the parameters and adding some features to make better predictions.

### Machine Learning Approach

Machine Learning can be used for forecasting a time series model. What makes time series different from the normal regression is that they are **time-dependent**. The basic assumption of regression is that the observation is independent with each other. Time series data also have some non-linear pattern in it, for example, seasonal and trends. To make it into regression problem, we need to capture the patterns and include them in the feature.

XGBoost algorithm can be used to solve the problem with regression method. It is a popular algorithm and usually used for kaggle competition because of its flexibility and predictive power. The only problem to use XGBoost is that there are many hyper-parameters that can be tuned to improve the model. After we create a simple model, we will try to tune the hyper-parameters by using grid search to improve the model.

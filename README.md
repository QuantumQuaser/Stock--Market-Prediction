#  LSTM-Based Stock Prediction Model

## Introduction

This README documents the most refined version of  LSTM-based stock prediction model to date. While the model has shown promise, it remains a work in progress, continually evolving to adapt to the dynamic nature of stock markets. The complete code for this model is accessible via this [link](https://github.com/QuantumQuaser/Stock--Market-Prediction/tree/main/improvised%20_keras).
[ provides an in-depth explanation of a machine learning model designed to predict stock performance. The model uses a Long Short-Term Memory (LSTM) neural network to recommend the best out of five stocks for trading in the upcoming week. ]

## Table of Contents
1.  [Introduction](#Introduction)
2. [Environment setup](#Environment-Setup)
3. **Data Acquisition** 
4. **Feature Engineering**
5. **LSTM Model Explanation**
6. **Model Training and Evaluation**
7. **Results and Interpretation**
8. **Conclusions and Recommendations**


## Environment Setup
```python
# Required libraries
import yfinance as yf
import pandas as pd
import numpy as np
import feedparser
from textblob import TextBlob
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
```
These libraries are essential for data handling (`pandas`, `numpy`), accessing financial data (`yfinance`), parsing news feeds (`feedparser`), sentiment analysis (`TextBlob`), and building the LSTM model (`keras`).

## Data Acquisition
Stock data is downloaded using `yfinance`, providing historical price information essential for the analysis. Additionally, news articles related to each stock are fetched and analyzed for sentiment, offering insights into market perceptions that could influence stock prices.

## Feature Engineering
### Technical Indicators
- **Relative Strength Index (RSI)**: Measures the speed and change of price movements.
- **Simple Moving Average (SMA) and Exponential Moving Average (EMA)**: Indicate trends over time.

### Sentiment Analysis
Sentiment scores are derived from news headlines, providing a qualitative measure of market sentiment towards each stock.

## LSTM Model Explanation
LSTM networks are chosen for their ability to capture time-dependent patterns in data, which is crucial for accurate stock price prediction.

### Model Architecture
The model includes LSTM layers followed by dense layers. The LSTM layers extract features from the time-series data, learning from the temporal dependencies. The dense layers then interpret these features to make predictions.

### Activation Functions
- `relu` (Rectified Linear Unit): Helps the network learn non-linear patterns.
- `sigmoid`: Converts outputs to a probability format, ideal for binary classification tasks.

## Model Training and Evaluation
The model is trained using a cross-validation approach to ensure robustness. The performance is evaluated using accuracy, F1 score, ROC-AUC, and a confusion matrix, providing a comprehensive view of the model's predictive capabilities.


## 7. Results and Interpretation

### Model Performance Overview
| Fold | Accuracy | F1 Score | ROC-AUC | Confusion Matrix |
|------|----------|----------|---------|------------------|
| 1    | 77.43%   | 0.73%    | 57.44%  | TP: 1851, FP: 0, FN: 540, TN: 2 |
| 2    | 77.14%   | 0.72%    | 53.53%  | TP: 1844, FP: 7, FN: 540, TN: 2 |
| 3    | 77.42%   | 0.73%    | 61.85%  | TP: 1850, FP: 0, FN: 540, TN: 2 |
| 4    | 77.50%   | 2.18%    | 55.76%  | TP: 1848, FP: 3, FN: 535, TN: 6 |
| 5    | 77.34%   | 0.36%    | 56.04%  | TP: 1849, FP: 2, FN: 540, TN: 1 |

*TP: True Positive, FP: False Positive, FN: False Negative, TN: True Negative*

### Interpretation
The model shows moderate accuracy in predicting stock performance. However, the low F1 scores indicate challenges in balancing precision and recall. The ROC-AUC scores suggest a need for improvement in distinguishing between high and low-performing stocks.

## 8. Conclusions and Recommendations
The model demonstrates potential in stock prediction using LSTM. Future improvements could include:
- Enhancing data sources for better sentiment analysis.
- Optimizing LSTM architecture.
- Exploring additional financial indicators.


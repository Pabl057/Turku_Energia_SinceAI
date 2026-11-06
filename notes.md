# 1. The core Problem:
* electricty generation and consumption must remain in absolute balance every second
* more renewable energy -> the energy market changes very fast
* one miscalculation from a market participant -> inbalaance in the power grid -> **financial penalties: imbalance costs**

**Problem:** predicting these grid imbalances and their associated prices is very difficult. They are driven by many overlapping, fast moving factors

# 2. The Mission:
Traditiional forecast just spit out one number. System needs to go much deeper.
-> build an intelligent forecasting system for imbalance prices

1. Probabilistic Forecasting: Model Output: 80% chance it stays between 40€ and 70€, but if the wind drops suddenly heree is a risk of a spike to 150€
2. Explainibility: Energy traders need explanation why prices will drop
3. Building Human Trust: goal: fully autamated trading without oversight, but creating a transparent decision- support tool that helps human tradees

# 3. Recommended Tech Stack & Approaches
## Time-Series Modelling:
Time Series model is a machine learing mdoel that can analyze sequential time series data and predict future values. Time series **datasets** consist of data values ordered over time, with time as the **independent variable**. It allows forecasting of future data values based on previous values in the sequence.
### Python Libraries:
Darts (u8darts): unifies traditional mdoels (ARIMA, Exponential Smoothing= and modern amchine learning/deep learning models (like N-BEATS) under one api

## Probabilistic Machine Learning
## Explainable AI (XAI)
## Interactive Data Visualization






































## Notes from the class
we dont know the real imbalance until it has been measured


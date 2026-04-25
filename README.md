# IPC Time Series Analysis - Romanian Inflation Forecasting

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Statsmodels](https://img.shields.io/badge/Statsmodels-0.14+-green.svg)](https://www.statsmodels.org/)

A comprehensive time series analysis and forecasting project for the Romanian IPC (Consumer Price Index) from January 2000 to March 2026. This project implements ARIMA/ARMA modeling to forecast inflation rates with rigorous diagnostic testing.

## 📊 Overview

This project performs a complete econometric analysis of Romanian inflation data including:
- Stationarity testing (ADF & KPSS)
- Model selection (AR, MA, ARMA)
- Residual diagnostics (autocorrelation, normality, heteroskedasticity)
- 6-month inflation forecasting
- Visualization of economic events (GFC, COVID-19, Ukraine war)

## ✨ Features

- **Data Processing**: Automatic CSV loading and date parsing
- **Stationarity Analysis**: ADF and KPSS unit root tests
- **Model Estimation**: AR(1) and ARMA(1,1) models
- **Theoretical vs Empirical Validation**: Compare model-implied vs actual ACF/PACF
- **Comprehensive Residual Diagnostics**:
  - Ljung-Box (Q) and Box-Pierce (Q*) tests
  - Normality tests (Jarque-Bera, Shapiro-Wilk, Kolmogorov-Smirnov)
  - Heteroskedasticity tests (White, Breusch-Pagan, ARCH)
  - Outlier detection with economic event mapping
- **Model Selection**: AIC/BIC criteria evaluation
- **Forecasting**: 6-month inflation rate and IPC level forecasts
- **Publication-Ready Plots**: All visualizations saved as high-resolution PNG files



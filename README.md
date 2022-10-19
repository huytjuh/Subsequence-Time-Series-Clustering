# **Subsequence Time Series Clustering**
![](https://i.ytimg.com/vi/wqQKFu41FIw/maxresdefault.jpg)
![](https://img.shields.io/github/license/huytjuh/Recommender-System-Basket-Analysis) ![](https://img.shields.io/maintenance/no/2021)

Subsequence Time Series (STS) Clustering model for discovering hidden patterns and complex seasonality within univariate time-series datasets by clustering similar groups of time windows based on their structural characteristics using advanced statistics.

Python implementation from scratch inspired by paper [Wang et al. (2006)](https://link.springer.com/content/pdf/10.1007/s10618-005-0039-x.pdf).

***Version: 2.1 (2021)*** 

---

## Introduction
Time-series analysis allows us to predict future values based on historical observed values, but they can only do so to the point where the model is able to differentiate between seasonal fluctuations within the univariate time-series dataset. So far, many papers consider relatively simple seasonal patterns such as weekly and monthly effects. However, higher frequency time-series often exhibit more complicated seasonal patterns. An alternative to using dummy variables, especially for multiple complex seasonal patterns, is to use Fourier terms. Using linear combinations of sine and cosine functions, successive Fourier terms represents the harmonics of the multiple seasonality components, and thus can be added as explanatory regressors to the forecasting models.

While traditional Fourier term analysis is able to capture the pattern of an univariate time-series relatively well, it tends to overestimate for day-of-the-week, monthly, and other known events that are self-evident without requiring extensive analysis. Hence, we suggest residual modified Fourier terms obtained from the residuals of ARIMA, allowing us to redirect our focus on capturing the more complex hidden patterns. On top of that, we propose a Subsequence Time series Clustering framework to enforce the forecasting models to adjust their parameters according to the clustered seasonal time windows, bringing the complex seasonality model to another level. That is, by incorporating advanced statistical operations and defining more complex characteristics of the univariate time-series such as non-linearity, self-similarity, and chaos on top of the decomposed time-series (trend, seasonality, noise), allows us to further refine and improve the forecasting accuracy using a fully data-driven framework.

## Colab Notebook

STS Clustering based on Hierarchical Clustering:<br/>
[Google Colab]() | [Code]()

STS Clustering based on Self-Organizing Maps (SOM):<br/>
[Google Colab]() | [Code]()

## Prerequisites
* Linux or macOS
* python 3.8
* nolds 0.5.2
* pmarima 1.8.4
* bayesian-optimization 1.2.0
* CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started

### Installation
* Clone this repository.
```
git clone https://github.com/huytjuh/Subsequence-Time-Series-Clustering
cd Subsequence-Time-Series-Clustering
```
* Install Python dependencies using `requirements.txt`.
```
pip install -r requirements.txt
```

### Run Subsequence Time Series Clustering
* Download an univariate time-series dataset:
```
datasets/station_case.csv
```
* Train STS Clustering (default: Hierarchical Clustering)
```
#!./scripts/run_train.sh
python3 train.py --cluster_method hierarchical
```
* Test STS Clustering ARIMA forecasting model (default: Hierarchical Clustering)
```
#!./scripts/run_main.sh
pyton3 main.py --cluster_method hierarchical --forecast ARIMA
```

## Algorithms
The table below lists the global measures describing the univariate time-series obtained using advanced statistical operations that best capture the underlying characteristics of the given time horizon. The following global characteristics are measured and scaled normally: Trend, Seasonality, Periodicity, Serial Correlation, Skewness, Kurtosis, Non-Linearity, Self-Similarity, and Chaos. References and formulas are linked in the Reference column, explaining in detail the math and implementation of the statistics.

| Statistics | Description | Reference |
|---|---|---|
| Trend | A trend appears when there is a long-term change in the mean level estimated by applying a convolution filter to the univariate time-series and can be detrended accordingly. | [Reference]() |
| Seasonality | Seasonality exists when the time-series is influenced by seasonal factors, such as day-of-the week and can be defined as a pattern that repeats itself over the time horizon that can be de-seasonalized accordingly. | [Reference]() |
| Periodicity (Fourier Term) | Periodicity examines the cyclic pattern of the time-series by including Fourier analysis on top of the seasonality to estimate the periodic pattern and hidden complex seasonality using a harmonic function of sine and cosine functions. | [Reference]() |
| Serial Correlation | Degree of serial correlation is measured by exhibition of white noise, that is no signs of periodic cycles, where we use Ljung-Box statistical test to identify completely independent observations within the univariate time-series | [Reference]() |
| Skewness | Skewness is the degree of asymmetry of a distribution and measures the deviation of the distribution of the univariate time-series from a symmetric distribution. | [Reference]() |
| Kurtosis | Kurtosis is a statistical measure that defines how heavily the tails of a distribution deviate from the tails of a normal distribution and whether the tails of the distribution contain extreme values. | [Reference]() |
| Self-Similarity (Hurst-Exponent) | Self-similarity is measured by the Hurst Exponent and infers that the statistical properties of the univariate time-series are the same for all its sub-sections, i.e. each day are similar to one another meaning that there is no strong sign of day-of-the-week effect. | [Reference]() |
| Non-Linearity <br />(BDS test) | Extracting the degree of non-linearity is measured by BDS statistical test and important for linear models that are generally not sufficiently capable of forecasting univariate time-series that exhibit more complex patterns compared to non-linear models. | [Reference]() |
| Chaos <br /> (Lyapunov-Exponent)| Presence of chaos is refered as the degree of disorder calculated by the Lyapunov Exponent (LE) and describes the  growth rate of small differences in the initial values becoming very large over time. | [Reference]() |

To further improve the forecasting performances, STS Clustering is used on the global measures and statistical operations to discover hidden seasons and similar patterns exhibiting within an univariate time-series. That is, the objective is to find groups of similar time windows based on their structural characteristics described previously. We consider two types of clustering methods: Agglomerative Hierarchical Clustering and Self-Organizing Maps (SOM).

| Algorithm | Type | Description | Code |
|---|---|---|---|
| Hierarchical Clustering | Agglomerative Clustering | Hierarchical Clustering is a method of cluster analysis which seeks to build a hierarchy of clusters visualized with a dendrogram where we use a bottom-up approach on structural similarities for each time window clusters; that is, it is more versatile than partitional algorithms (i.e. Kmeans) and with Ward's minimum variance criterion it does not measure the distance directly making it less sesensitive to initial seed selection. | [Code]() |
| Self-Organizing Maps (SOM) | Deep Neural Network | Self-Organizing Maps (SOM) is a specific class of Neural Network used extensively as a clustering and visualization tool in Exploratory Data Analysis (EDA); that is, it both a projection method which maps high-dimensional data space into simpler low-dimensional space mapping similar data samples to nearby neurons. | [Code]() |

***Note.*** The univariate time-series has to be partitioned deterministically in order to apply STS Clustering, i.e. split into weeks (52 partitions).

## Test Results & Performances
A comparison between seasonal self-evident explanatory variables that fall under the naive methods and STS clustering methods that fall under the more complex methods. We run the evaluation on five different forecasting models, namely ARIMA, RF, LSTM, Hybrid ARIMA-RF, and Hybrid ARIMA-LSTM. Additionally, we provide a [Notebook]() to illustrate how the different algorithms could be evaluated and compared.
| N-Step Ahead | Clustering | ARIMA | RF | LSTM | ARIMA-RF | ARIMA-LSTM |
|---|---|---:|---:|---:|---:|---:|
| 1-month | Hierarchical | -12.06% | 26.34% | 3.14% | 7.20% | -3.82% |
| 1-month | SOM | -3.28% | 27.10% | 27.33% | 9.26% | -1.80% |
| 5-month | Hierarchical | -4.03% | -8.94% | 3.32% | 6.65% | 11.90% |
| 5-month | SOM | -0.67% | -8.74% | -2.60% | 6.37% | 10.52% |
| 12-month | Hierarchical | -9.53% | -33.80% | 0.92% | 5.07% | 5.89% |
| 12-month | SOM | -2.56% | -18.45% | 1.91% | 4.25% | 4.71% |

***Note.*** The value represents the relative improvement in RMSE when the respective STS Clustering is incorporated compared to the naive method of adding seasonal component as explanatory variables. The respective values are a mean RMSE estimate of over 100 univariate time-series.

![](https://github.com/huytjuh/Subsequence-Time-Series-Clustering/blob/main/example/example_graph.png)

***Example.*** Each color represents a cluster of time horizon that can be split deterministically each featuring unique characteristics and similar patterns.

## Reference Papers

* Wang, X., Smith, K., & Hyndman, R. (2006). Characteristic-based clustering for time series data. Data mining and knowledge Discovery, 13(3), 335-364. Available online: [Link](https://link.springer.com/content/pdf/10.1007/s10618-005-0039-x.pdf)

.

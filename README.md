# **Subsequence Time Series Clustering**
![](https://i.ytimg.com/vi/wqQKFu41FIw/maxresdefault.jpg)
![](https://img.shields.io/github/license/huytjuh/Recommender-System-Basket-Analysis) ![](https://img.shields.io/maintenance/no/2019)

Basket-Sensitive Random Walk & Factorization Machine Recommendations for Grocery Shopping. 
Item-based Collaborative Filtering (CF) using hybrid memory- and model-based methods with Factorization Machines and Alternative Least Squares.

R implementation from scratch inspired by paper [Li et al (2009)](https://www.researchgate.net/profile/Paulo-Lisboa/publication/221653590_Grocery_shopping_recommendations_based_on_basket-sensitive_random_walk/links/09e4150cb9fb091a30000000/Grocery-shopping-recommendations-based-on-basket-sensitive-random-walk.pdf).

***Version: 1.0 (2019)*** 

---

## Introduction
While recommendation systems have been a hot topic for a long time now due to its success in business applications, it is still facing substantial challenges. As grocery shopping is most often considered as a real drudgery, many online stores provide a shopping recommendation system for their customers to facilitate this purchase process. However, there is still a large majority of people who still hesitate from doing their groceries online even though this form of shopping provides consumers with distinct advantages. Hence, the chasm between online retail and its brick-and-mortar counterpart keeps expanding in numbers, and people’s shopping preferences are evolving in turn, leaving retailers with little choice but to adapt.  

This has led to online grocery shopping becoming more and more prominent, and therefore resulted in radical adjustments within the marketing decision framework of many retailers. Thus, we investigate whether traditional collaborative filtering techniques are applicable in the domain of grocery shopping, and further improve its recommendations using more advanced models and machine learning techniques. Hence, various CF-based models have been constructed including your traditional similarity-based collaborative filtering models, a basket-sensitive random walk model, and a basket-sensitive factorization machine. Here, we found that our basket-sensitive factorization machine comes out on top when it comes to recommending less popular items. However, due to its computational time, it remains to be a question whether this model is applicable in practical use.

## Colab Notebook

Basket-Sensitive Random Walk & Factorization Machine Recommendation for Grocery Shopping in R:<br/>
[Google Colab]() | [Code]()

## Prerequisites
* Linux or macOS
* Rstudio 
* CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started

### Installation
* Clone this repository.
```
git clone https://github.com/huytjuh/Recommender-System-Basket-Analysis
cd Recommender-System-Basket-Analysis
```
* Install R dependencies using `requirements.txt`.
```
#!./scripts/install_pkgs.sh
while IFS=" " read -r package version; 
do 
  Rscript -e "devtools::install_version('"$package"', version='"$version"')"; 
done < "requirements.txt"
```

### Run Recommender System
* Download a Basket Grocery dataset:
```
datasets/ta_feng_all_months_merged.csv
```
* Train Recommender System & Calculate Similariy Scores
```
#!./scripts/run_train.sh
Rscript train.R
```
* Test Recommender System
```
#!./scripts/run_train.sh
Rscript main.R
```

## Algorithms
The table below lists the recommender algorithms currently available in the repository. Python scripts are linked under the Code column, explaining in detail the math and implementation of the algorithm including comments and documentations.

| Algorithms                                | Type                                   | Description                                                                                                                                                                                                                       | Code |
|-------------------------------------------|----------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------|
| Popularity<br /> (pop)                    | Naive                                  | Naive recommendations based on most popular items bought by users and not in the basket.                                                                                                                                          | [Code]() |
| Cosine<br /> (CF(cos))                    | Collaborative Filtering (Memory-based) | Cosine-based similarities calculated from the cosine of the angle between two items thought of as two vectors in the m dimensional user-space.                                                                                    | [Code]() |
| Conditional Probability<br /> (CF(cp))    | Collaborative Filtering (Memory-based) | Conditional probability based similarities taking rating scale between users into account and normalized including a control variable alpha to penalize popular items.                                                            | [Code]() |
| Bipartite Network<br /> (CF(bn))          | Collaborative Filtering (Memory-based) | Bipartite network based similarities calculated from a bipartite graph describing the shopping basket data containing two nodes: consumers and products; thus, can be defined as the transition probability between each product. | [Code]() |
| Alternate Least Square<br /> (ALS)        | Collaborative Filtering (Model-based)  | Matrix factorization algorithm for explicit or implicit feedback in large datasets by decomposing the user-matrix into smaller dimension user and item features.                                                                  | [Code]() |
| Factorization Machine<br /> (FM)          | Collaborative Filtering (Model-based)  | Extended matrix factorization model allowing for feature-rich datasets by including higher-order interactions between variables of larger domain and combining both regression and factorization methods.                         | [Code]() |
| Basket-Sensitive Random Walk<br /> (BSRW) | Hybrid                                 | A stochastic process dictating the likelihood of jumping from one item to another as extension to further explore transitive associations by incorporating the current shopping context into the Collaborative Filtering models.  | [Code]() |

## Test Results & Performances
A comparison between different Recommender System algorithms which can be categorized into three types of models: similarity-based CF methods, BSRW-based methods, and model-based methods. We run the comparison on three different evaluation metrics: Binary Hit Rates on least three popular items *bHR(pop)* and three randomly selected items *bHR(rnd)*, and Weighted Hit Rate based on leave-one-out cross-validation *wHR(loo)*. Additionally, we provide a [Notebook]() to illustrate how the different algorithms could be evaluated and compared.

| <br /> Algorithm | L-3-O<br /> bHR(pop) | L-3-O<br /> bHR(rnd) | L-1-O<br /> wHR(loo) |
|---|:---:|:---:|:---:|
| pop | 0.43 | 16.80 | 2.83 |
| CF(cos) | 16.72 | 31.62 | 5.65 |
| CF(cp) | 16.46 | 30.84 | 5.67 |
| CF(bn) | 16.75 | 31.88 | 5.79 |
| CF(cos) + BSRW | 16.63 | 31.70 | 5.67 |
| CF(cp) + BSRW | 16.46 | 30.80 | 5.71 |
| CF(bn) + BSRW | 16.75 | 31.84 | 5.78 |
| ALS | 15.28 | 26.28 | 4.34 |
| BFSM* | 20.17 | 19.21 | 2.25 |
| Hybrid | 15.28 | 26.36 | 4.32 |

*A subset of 10% of the testing had to be taken instead due to its computational heavy nature of Factorization Machine

## Reference Papers

* Li, M., Dias, B. M., Jarman, I., El-Deredy, W., & Lisboa, P. J. (2009, June). Grocery shopping recommendations based on basket-sensitive random walk. In Proceedings of the 15th ACM SIGKDD international conference on Knowledge discovery and data mining (pp. 1215-1224).\
Available online: [Link](https://www.researchgate.net/profile/Paulo-Lisboa/publication/221653590_Grocery_shopping_recommendations_based_on_basket-sensitive_random_walk/links/09e4150cb9fb091a30000000/Grocery-shopping-recommendations-based-on-basket-sensitive-random-walk.pdf)
* Le, D. T., Lauw, H. W., & Fang, Y. (2017). Basket-sensitive personalized item recommendation. IJCAI.\
Available online: [Link](http://ink.library.smu.edu.sg/cgi/viewcontent.cgi?article=4767&context=sis_research)



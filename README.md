# Customer Segmentation Using PCA, K-Means, and RFM Analysis

The core machine-learning workflow is implemented from scratch without using pre-built scikit-learn algorithms, including PCA, K-Means clustering, and the elbow method. They are implemented manually to demonstrate their underlying logic
This repository contains the code appendix for a customer segmentation project based on grocery customer and marketing campaign data. The analysis combines data cleaning, feature engineering, dimensionality reduction, clustering, customer profiling, visualisation, and computational complexity testing.

## Dataset

The project uses the **Customer Personality Analysis** dataset from Kaggle. It describes customer demographics, household characteristics, purchasing behaviour, product preferences, sales-channel activity, and responses to marketing campaigns. The data represents an unnamed grocery business and is not identified as Tesco data.

### Dataset dimensions and complexity

| Stage | Size |
| --- | ---: |
| Original dataset | 2,240 customers x 29 columns |
| Variables excluding the customer ID | 28 |
| Original data types | 26 numerical and 3 categorical columns |
| Missing values | 24, all in `Income` |
| Records after outlier removal | 2,236 |
| Features supplied to PCA | 19 |
| Principal components supplied to K-Means | 3 |
| Final customer clusters | 4 |

Although the dataset is structured, it requires several preprocessing steps because it contains missing income values, unusual categorical labels, variables measured on different scales, redundant columns, and abnormal age and income records.

## Repository contents

- [`code appendix.ipynb`](./code%20appendix.ipynb) - complete analysis with explanations, code, outputs, and visualisations.
- [`marketing_campaign.csv`](./marketing_campaign.csv) - input dataset required by the notebook.
- `README.md` - project overview and instructions.

The notebook and CSV file must remain in the same directory unless the data-loading path in the notebook is changed.

## Analysis workflow

1. **Load and inspect the data**
   - Examine the dataset dimensions, data types, categorical values, and missing values.
   - Replace the 24 missing `Income` values with the median income.

2. **Engineer customer features**
   - `Customer_For`: length of the customer relationship.
   - `Age`: customer age derived from year of birth.
   - `Living_With`: simplified household relationship category.
   - `Children`, `Family_Size`, and `Is_Parent`: household characteristics.
   - `Monetary`: total spending across product categories.
   - `Frequency`: total number of purchases across channels.

3. **Clean and prepare the data**
   - Remove redundant, constant, and replaced columns.
   - Exclude records with `Age >= 90` or `Income >= 400,000`.
   - Encode categorical variables numerically.
   - Apply manual min-max scaling so numerical features fall between 0 and 1.

4. **Reduce dimensionality with PCA**
   - Standardise the data.
   - Calculate the covariance matrix.
   - Estimate eigenvalues and eigenvectors using power iteration and deflation.
   - Project 19 modelling features onto three principal components.

5. **Cluster customers with K-Means**
   - Initialise centroids using a fixed random seed.
   - Assign customers to the nearest centroid using Euclidean distance.
   - Recalculate centroids until convergence or the iteration limit is reached.
   - Use the elbow method to select `k = 4` clusters.

6. **Profile and visualise the clusters**
   - Compare cluster sizes and RFM distributions.
   - Visualise the clusters in three-dimensional PCA space.
   - Assign interpretable RFM segments such as champions, loyal customers, potential loyalists, and at-risk customers.
   - Compare the distribution of RFM segments across clusters.

7. **Evaluate computational complexity**
   - Measure runtime and loop counts using resampled datasets of different sizes.
   - Compare the observed growth pattern with the theoretical complexity.

## Computational complexity

For K-Means, the main time complexity is:

```text
O(T x n x k x p)
```

where:

- `T` is the number of iterations;
- `n` is the number of customer records;
- `k` is the number of clusters; and
- `p` is the number of dimensions after PCA.

This project uses `k = 4`, `p = 3`, and a maximum of 100 iterations. When these values are treated as fixed, runtime grows approximately linearly with the number of customers, or `O(n)`. The dataset is small enough to run comfortably on a typical personal computer, although the manual Python implementation will be slower than optimised NumPy or scikit-learn implementations on much larger datasets.

## Requirements

- Python 3
- Jupyter Notebook or JupyterLab
- pandas
- NumPy
- Matplotlib
- seaborn
- Plotly

Install the required packages with:

```bash
pip install pandas numpy matplotlib seaborn plotly jupyter
```

## Running the notebook

1. Download or clone this repository.
2. Confirm that `code appendix.ipynb` and `marketing_campaign.csv` are in the same directory.
3. Start Jupyter Notebook or JupyterLab.
4. Open `code appendix.ipynb`.
5. Select **Restart Kernel and Run All Cells**.

From a terminal, the notebook can be opened with:

```bash
jupyter notebook "code appendix.ipynb"
```

The analysis was verified by executing the notebook from its first cell through its final cell without errors.

## Data source and attribution

- Dataset: [Customer Personality Analysis on Kaggle](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis/data)
- Reference notebook: [Customer Segmentation by Karnika Kapoor](https://www.kaggle.com/code/karnikakapoor/customer-segmentation-clustering/notebook)

The reference notebook informed parts of the original data preparation and visualisation workflow. This project extends and modifies that work through additional RFM features and segmentation, custom preprocessing functions, manual PCA and K-Means implementations, cluster interpretation, and complexity analysis.

## Purpose

This repository is an academic code appendix intended to demonstrate the design, implementation, and evaluation of a reusable customer segmentation workflow.

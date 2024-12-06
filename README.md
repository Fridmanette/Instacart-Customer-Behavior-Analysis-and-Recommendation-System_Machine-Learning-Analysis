Data sourced from: https://www.kaggle.com/competitions/instacart-market-basket-analysis/data

This repository contains a comprehensive data analysis and machine learning pipeline developed to explore customer purchasing behaviors using the Instacart dataset. 


About the Dataset:

The Instacart Market Basket Analysis dataset contains transactional data from Instacart's online grocery store. It provides anonymized data on: 

- User order patterns

- Product details

- Reorder behaviors


The dataset includes 3 million grocery orders from over 200,000 users, with details such as

- Order timing

- Product categories

- Reorder flags


The code combines advanced data processing, exploratory analysis, and cutting-edge machine learning techniques to deliver actionable insights into customer habits and product trends.


Tools and Technologies:

- Data Processing: PySpark, Pandas, NumPy

- Machine Learning: MLlib, Scikit-learn, XGBoost, TensorFlow, SMOTE

- Data Visualization: Matplotlib, Seaborn, NetworkX, Pyvis

- Natural Language Processing: SpaCy

- Big Data: Apache Spark


Key Features:

Machine Learning Models

a. Clustering

- K-Means Clustering: Grouped customers based on purchasing habits to identify meaningful customer segments.

- Visualization: Applied PCA and t-SNE for better interpretability of clusters.

b. Recommendation System

- ALS (Alternating Least Squares): Predicted user-product interactions to generate personalized product recommendations.

c. Classification Models

- Reorder Prediction: Built models (e.g., XGBoost) to predict the likelihood of product reorders.

- SMOTE: Addressed class imbalances in the dataset to improve prediction accuracy.

- Hyperparameter Tuning: Optimized model performance using Bayesian Optimization and RandomizedSearchCV.

3. Association Rule Mining

- Frequent Pattern Mining: Used FP-Growth to identify frequently purchased product combinations.

- Visualization: Represented associations through interactive network graphs for better understanding of product relationships.

4. Deep Learning

- Neural Collaborative Filtering (NCF): Developed a TensorFlow-based model for personalized recommendations leveraging embeddings for users and products.


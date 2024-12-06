This repository contains a comprehensive data analysis and machine learning pipeline developed to explore customer purchasing behaviors using the Instacart dataset. The project includes detailed preprocessing, exploratory data analysis (EDA), feature engineering, and the application of advanced machine learning and recommendation algorithms. The analysis delivers actionable insights and highlights methods for enhancing inventory strategies and customer satisfaction.
Key Features

Data Preprocessing and Cleaning:

Temporal analysis of reorder rates across days of the week and time of day.

Departmental and product-level insights into reorder behavior.

Machine Learning Models:
Clustering:

- k-means clustering to group users based on purchasing habits.

- Visualization using PCA and t-SNE for interpretability.

Recommendation System:

- ALS (Alternating Least Squares) algorithm to predict user-product interactions.

Classification Models:

- Predictive models (e.g., XGBoost, SMOTE-enhanced classifiers) to predict reorder likelihood.

- Hyperparameter tuning with Bayesian optimization and RandomizedSearchCV.

- Association Rule Mining:

Applied FP-Growth to identify frequently purchased product combinations.

- Associations using network graphs.
 
- Natural Language Processing using SpaCy.
  
- Deep Learning: Developed a neural collaborative filtering model using TensorFlow for personalized recommendations.


Tools and Technologies

Data Processing: PySpark, Pandas, NumPy

Machine Learning: MLlib, Scikit-learn, XGBoost, TensorFlow, SMOTE

Data Visualization: Matplotlib, Seaborn, NetworkX, Pyvis

Natural Language Processing: SpaCy

Big Data: Apache Spark

Development: Google Colab, Jupyter Notebooks, GitHub

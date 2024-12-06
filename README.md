This repository contains a comprehensive data analysis and machine learning pipeline developed to explore customer purchasing behaviors using the Instacart dataset. The project includes detailed preprocessing, exploratory data analysis (EDA), feature engineering, and the application of advanced machine learning and recommendation algorithms. The analysis delivers actionable insights and highlights methods for enhancing inventory strategies and customer satisfaction.
Key Features

Data Preprocessing and Cleaning:
Temporal analysis of reorder rates across days of the week and time of day.
Departmental and product-level insights into reorder behavior.

Machine Learning Models:
Clustering:
Applied k-means clustering to group users based on purchasing habits.
Visualized clusters using PCA and t-SNE for interpretability.
Recommendation System:
Leveraged the ALS (Alternating Least Squares) algorithm to predict user-product interactions.
Evaluated recommendations with metrics like Precision@K.
Classification Models:
Built predictive models (e.g., XGBoost, SMOTE-enhanced classifiers) to predict reorder likelihood.
Performed hyperparameter tuning with Bayesian optimization and RandomizedSearchCV.
Association Rule Mining:
Applied FP-Growth to identify frequently purchased product combinations.
Visualized associations using network graphs to highlight relationships between items.
Natural Language Processing:
Processed textual data using SpaCy for potential future use cases like product sentiment analysis.
Deep Learning:
Developed a neural collaborative filtering model using TensorFlow for personalized recommendations.

Tools and Technologies
Data Processing: PySpark, Pandas, NumPy
Machine Learning: MLlib, Scikit-learn, XGBoost, TensorFlow, SMOTE
Data Visualization: Matplotlib, Seaborn, NetworkX, Pyvis
Natural Language Processing: SpaCy
Big Data: Apache Spark
Development: Google Colab, Jupyter Notebooks, GitHub

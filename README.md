# Prediction of Thyroid Cancer Recurrence
**Documentation is also available in Serbian in the file Dokumentacija.pdf.**

## Introduction
This project focuses on the application of machine learning in medicine, specifically on predicting the recurrence of thyroid cancer. A dataset comprising demographic data of patients and blood test results, along with diagnostic information about thyroid disease, was used. The data was collected over a 15-year period, with each patient being monitored for at least 10 years. The goal of the project is to develop a model capable of accurately predicting the recurrence of thyroid cancer.

## Data Preprocessing - Data Wrangling
Data is located in [Thyroid_Diff.csv](https://github.com/vranjesevica/Thyroid-Cancer-Analysis-and-Prediction-ANN/blob/master/Thyroid_Diff.csv "Thyroid_Diff.csv") file and the first task I did was data preprocessing.
#### Checking for Missing Values
Based on the results shown in the left image below, it was confirmed that the dataset contains no missing values.
<div align = "middle">
		<img width = "198px" src="https://github.com/user-attachments/assets/91fc2f65-d280-413e-91bf-19ff6f730663"/>
		<img width = "650px" src="https://github.com/user-attachments/assets/29594c24-38f5-41cf-b7f9-40d40aea4b74"/>
</div>

#### Anomaly Detection

Anomaly detection was conducted using the Isolation Forest algorithm, with the contamination parameter set to 5%. Detected anomalies were marked in a new column labeled **"Anomaly"**, assigned a value of -1, as shown in the right image. The data identified as anomalies was displayed and subsequently removed from the dataset. The cleaned dataset is now ready for further processing.

#### Data Encoding

Before converting strings to numbers, all unique values in the dataset columns were listed to obtain an overview of the values present in each column.
<div align = "middle"> 
	<img src="https://github.com/user-attachments/assets/548dd759-a591-4cb1-ac8f-fc2931e161f6"/>
</div>

-   The **"Age"** column already contained numeric values.
-   Columns with values like **"Yes"** and **"No"** (e.g., **"Smoking"**, **"Hx Smoking"**, and **"Hx Radiotherapy"**), the **"Gender"** column (with values **"F"** and **"M"**), as well as the **"Risk"**, **"Stage"**, and **"Response"** columns (which have a natural order and hierarchy), were mapped to numeric values using the `map()` function to preserve their order.
-   Other columns (nominal columns) were encoded using **One-Hot Encoding**, creating binary columns for each category.
-   Finally, the target column **"Recurred"** was encoded into binary values.

## Exploratory Data Analysis
#### Correlation Analysis
To understand the relationships between various features in the dataset, I calculated a correlation matrix, visualized as a heatmap, which highlights strong positive and negative correlations.  
Based on this analysis, the **"Adenopathy_No"** column was removed to avoid redundancy and improve efficiency.  
<div align = "middle"> 
	<img src="https://github.com/user-attachments/assets/82ed8d21-5820-4e46-9f9d-36a418bb4956"/>
</div>
Throughout the exploratory analysis, I used various data visualization techniques to examine and identify potential trends and relationships between attributes, as shown below.
<div align = "middle">
		<img height = "370" src="https://github.com/user-attachments/assets/f61c4d2c-35a1-4542-bf91-2c0f693f8c18"/>
		<img height = "370" src="https://github.com/user-attachments/assets/eaaf6d47-8ce5-4806-a446-e467d69c7026"/>
</div>
<div align = "middle">
		<img height = "300" src="https://github.com/user-attachments/assets/fba9c962-c756-4ced-b819-dfed040241a3"/>
		<img height = "300" src="https://github.com/user-attachments/assets/6879c01f-a31b-4d9f-b3b1-278f3afc7e01"/>
</div>
<div align = "middle">
		<img height = "310" src="https://github.com/user-attachments/assets/fae095d0-3bb7-4083-ad65-cec42f8ecbac"/>
		<img height = "310" src="https://github.com/user-attachments/assets/52cffd53-d1d8-4b27-8a72-d07f8186134e"/>
</div>


#### Data Balancing and Normalization
Balancing was performed using **SMOTE** (Synthetic Minority Over-sampling Technique). After applying SMOTE, the distribution of the target attribute **"Recurred"** was adjusted to achieve balance between the classes, ensuring better model training. A comparison of the number of samples before and after balancing was provided.
<div align = "middle">
		<img height = "310" src="https://github.com/user-attachments/assets/5005e9d1-b0ef-41ee-92cb-b80a32af1ed6"/>
		<img height = "310" src="https://github.com/user-attachments/assets/ed208a3a-f35f-48a3-9514-01a331c76068"/>
</div>

Normalization was conducted using **Min-Max Scaling**, ensuring that the data falls within the range [0, 1]. This facilitates interpretation and comparison of the features' influence on the final result. The normalized data was then used for training and testing the model.

## Model Creation

In this part of the project, I created various models to achieve the best possible performance.  
The models I used are:
-   **Logistic Regression** with cross-validation. It is a statistical method used for binary classification. It predicts the probability of a data point belonging to a specific class by applying the logistic function to a linear combination of input features. The model is simple, interpretable, and performs well on linearly separable data.
-   **SVM (Support Vector Machine)** - supervised learning model that finds the optimal hyperplane to separate data points from different classes. It uses support vectors (critical data points) to define the margin and maximize separation. SVM works well with high-dimensional data and can handle non-linear relationships using kernel functions.
-   **K-Nearest Neighbors (KNN)** – For this, I call the method `find_optimal_k(X_train, y_train, X_test, y_test, max_k=10)`, which selects the optimal number of neighbors (K). It is a lazy learning algorithm that classifies data points based on the majority class of their K nearest neighbors. It calculates distances (e.g., Euclidean) between points and assigns the label of the most common neighbor. It is called lazy because it doesn't learn anything, only classifies objects depending on their distance from others. We can't judge a book by its cover in some aspects, but KNN is simple to implement and effective for smaller datasets.
-   **Decision Tree Classifier** - it splits data into subsets based on feature values, forming a tree-like structure. Each node represents a decision rule, and the leaves represent the output class. The model is intuitive and interpretable, but it can overfit without proper pruning
-   **GaussianNB (Naive Bayes Classifier)** – Based on a Gaussian distribution. It assumes all features are independent, which is not the case in real-world data, leading to poor results. Therefore, this model is commented out.
-   **Random Forest Classifier** – Utilizes multiple decision trees trained on different subsets of data and features. This technique reduces variance and overfitting, providing stable and accurate predictions.
-   **Gradient Boosting Classifier** – A model that uses gradient boosting to combine multiple weak learners into a strong learner. Each subsequent model corrects the errors of the previous ones, improving prediction accuracy.
-   **AdaBoost Classifier** – Employs adaptive boosting to train a series of weak learners, where each subsequent model focuses more on the points that previous models classified incorrectly, thus increasing accuracy.
-   **LightGBM Classifier** – A highly efficient, fast, and accurate model.
-   **Bagging Model** – Bagging (Bootstrap Aggregating) reduces model variance by combining predictions from multiple models trained on different subsets of data. I use `DecisionTreeClassifier` as the base model because bagging is effective for models prone to overfitting, such as decision trees.
-   **Stacking Model** – Stacking (Stacked Generalization) combines predictions from multiple base models to create a meta-model for final predictions. I selected **Logistic Regression** as the meta-model, which learns how to best combine the outputs of the base models for improved results. The base models are: **Random Forest Classifier**, **Gradient Boosting Classifier**, **SVM**, and **K-Nearest Neighbors (KNN)**.

## Training and Evaluation
**Hyperparameter Tuning**  
The hyperparameters of each model were adjusted using the **Grid Search** method. Information about the hyperparameters and their possible values for different models was gathered from the Scikit-learn library documentation. While many parameters set to their default values delivered good performance, tuning the hyperparameters slightly improved the overall results, as demonstrated by the metrics provided in subsequent records.
<div align = "middle">
		<img height = "310" src="https://github.com/user-attachments/assets/496f6081-ce85-4cc3-b8fd-e7dc4fc32a63"/>
		<img height = "310" src="https://github.com/user-attachments/assets/d36a789d-d16d-4daf-baf5-e7b0f0cd3884"/>
</div>

**Cross-Validation**  
Cross-validation was used both to evaluate model performance and for hyperparameter tuning. A 5-fold cross-validation method was employed, calculating the mean accuracy across different data folds for each model to obtain a more reliable performance estimate. Here we can see how 5-fold cross-validation is working, white blocks are for training, and blue blocks of data are for testing:
<div align = "middle">
<img src = "https://github.com/user-attachments/assets/11da2a71-f5ef-48db-9578-5362ac788015"/>
</div>

**Performance Metrics Analysis**  
After training each model, the following performance metrics were analyzed:
-   **Accuracy**: Reflects the proportion of correctly classified instances out of the total number of instances.
-   **Precision**: Indicates the proportion of true positive instances among all instances classified as positive by the model.
-   **Recall**: Measures the proportion of true positive instances among all actual positive instances.
-   **F1 Score**: Balances precision and recall.
-   **Confusion Matrix**: Displays the number of correctly and incorrectly classified instances for each class.
-   **Jaccard Index**: Measures similarity between two datasets, in this case, actual and predicted values.
-   **Area Under the ROC Curve (AUC)**: Quantifies the model’s ability to distinguish between positive and negative instances.
-   **Log Loss**: Measures the model's performance using predicted probabilities for each instance, accounting for prediction uncertainty.
confusion matrix 
<div align="middle">
		<img height = "80" src="https://github.com/user-attachments/assets/7d75bc05-95f1-4807-b5d9-f7cbb86bbf4b"/>
		<img height = "80" src="https://github.com/user-attachments/assets/e6cd9eee-bce7-494b-9be7-e512632a6a75"/>
		<img height = "80" src="https://github.com/user-attachments/assets/c4fa1c37-a014-4181-9c99-683a059ea4ad"/>
</div>

<div align="middle">
	<img height = "70" src="https://github.com/user-attachments/assets/7e78b1b8-61a7-493a-8b90-c1554c12d253"/>
		<img height = "150" src="https://github.com/user-attachments/assets/75be0a19-d7f4-4ee5-aab0-7c8e790e0fd7"/>
		<img height = "150" src="https://github.com/user-attachments/assets/d2e3c4f8-b479-4f45-8fff-c0e6ba71ecde"/>
</div>


These metrics provided insights into how well models predicted outcomes and highlighted areas for improvement. Metrics were calculated before and after hyperparameter tuning to compare results.

**Top Performing Models**:

-   **RandomForestClassifier**: Achieved approximately 95% accuracy, 96% precision, 92% recall, and a high Jaccard index of 91%, with a low log loss of around 13%.
-   **AdaBoostClassifier**: Demonstrated high accuracy, precision, and recall, with a stable Jaccard index and low log loss, indicating efficient learning and generalization.
-   **LGBMClassifier**: Stood out with high accuracy, precision, and recall, making it one of the most reliable models, supported by a high Jaccard index and low log loss.

**ROC Curves**  
For models supporting the `predict_proba` function, ROC curves were calculated and presented to evaluate model performance. These curves illustrate the relationship between the false positive rate and true positive rate at various classification thresholds.
<div align="middle">
	<img height = "300" src="https://github.com/user-attachments/assets/32f7bfa4-28b9-4d34-a2c3-b98923c7090c"/>
</div>

**Feature Importance Selection**  
Identifying and analyzing key features in the dataset was crucial for understanding the prediction process of the models. If a model supported feature importance calculation, the most important features were identified and aggregated across all models for visualization. Here we can see the example of the most important features for LGBMClassifier:
<div align="middle">
<img height = "130" src="https://github.com/user-attachments/assets/c655bc59-17a7-4d4e-b939-b8658ab58b0c"/>
</div>

The **Recursive Feature Elimination (RFE)** technique was applied to identify the top 10 features for each model. Models were retrained using only the selected features, and relevant metrics and cross-validation scores were recalculated. Results showed that model performance was similar to training on all features, indicating that models could be effectively trained on a smaller feature set. This optimization improves resource efficiency and speeds up data processing, simplifying further analysis and interpretation. In the image, we can see the most important attributes for all models.
<div align="middle">
<img height = "600" src="https://github.com/user-attachments/assets/bb9a6765-2e4f-4c95-b57b-8202e0d699d0"/>
</div>

## Conclusion
Based on all the obtained results and a deeper analysis, I conclude that models based on algorithms such as --**RandomForestClassifier**, **GradientBoostingClassifier**, and **LGBMClassifier** demonstrated the best performance in predicting the recurrence of thyroid cancer. These models achieved high accuracy, precision, and recall, indicating that they are effective in distinguishing between patients at risk of thyroid cancer recurrence and those with no risk of the disease returning.

While the results are promising, it is important to acknowledge that this project has certain limitations. The dataset may not be sufficiently representative of all patient subgroups and might be limited in size, as it does not include a large number of samples. Additionally, some important features may be missing or inadequately represented in the data.

The results of this project provide valuable insights into the potential application of machine learning in thyroid cancer diagnostics and encourage further research in this field.

# Ucitavanje potrebnih biblioteka
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import IsolationForest, BaggingClassifier, StackingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score, log_loss, roc_curve, auc, jaccard_score
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import RFE

import warnings
warnings.filterwarnings("ignore")

# PREPROCESIRANJE PODATAKA
dataset = pd.read_csv("Thyroid_Diff.csv")
#print(dataset.head(10))

#provera da li ima nedostajucih vrednosti
missing_values = dataset.isnull().sum()
print(missing_values)

X = dataset.drop('Recurred', axis = 1)
y = dataset['Recurred']

# detekcija anomalija:
iso_forest = IsolationForest(contamination=0.05)
outliers = iso_forest.fit_predict(dataset.select_dtypes(include=[np.number]))

X['Anomaly'] = outliers
print(X[X['Anomaly'] == -1])
anomaly_indices = X[X['Anomaly'] == -1].index
dataset = dataset.drop(anomaly_indices)
#print(dataset.head())

# prikaz svih jedinstvenih vrednosti u kolonama
for column in dataset.columns:
    unique_values = dataset[column].unique()
    print("Unique values in column '{}': {}".format(column, unique_values))

#pretvaranje stringova u brojeve
hotData = pd.DataFrame()
nominal_cols = ['Thyroid Function', 'Physical Examination', 'Adenopathy', 'Pathology', 'Focality', 'T', 'N', 'M']
ordinal_cols = ['Risk', 'Stage', 'Response']

dataset2 = pd.DataFrame()

hotEncoder = OneHotEncoder()
dataset2['Age'] = pd.to_numeric(X['Age'])
dataset2['Gender'] = X['Gender'].map({'M': 0, 'F': 1})
dataset2['Smoking'] = X['Smoking'].map({'No': 0, 'Yes': 1})
dataset2['Hx Smoking'] = X['Hx Smoking'].map({'No': 0, 'Yes': 1})
dataset2['Hx Radiothreapy'] = X['Hx Radiothreapy'].map({'No': 0, 'Yes': 1})
dataset2['Risk'] = X['Risk'].map({'Low': 0,
                                  'Intermediate': 1,
                                  'High': 2})
dataset2['Stage'] = X['Stage'].map({'I': 0,
                                    'II': 1,
                                    'IVB': 2,
                                    'III': 3,
                                    'IVA': 4})

dataset2['Response'] = X['Response'].map({'Indeterminate': 0,
                                          'Excellent': 1,
                                          'Structural Incomplete': 2,
                                          'Biochemical Incomplete': 3})
dataset2['Thyroid Function'] = dataset['Thyroid Function']
dataset2['Physical Examination'] = dataset['Physical Examination']
dataset2['Adenopathy'] = dataset['Adenopathy']
dataset2['Pathology'] = dataset['Pathology']
dataset2['Focality'] = dataset['Focality']
dataset2['T'] = dataset['T']
dataset2['N'] = dataset['N']
dataset2['M'] = dataset['M']

# One-Hot Encoding za nominalne kolone
dataset2 = pd.get_dummies(dataset2, columns=nominal_cols, drop_first=True)
dataset2 = dataset2.astype(int)
pd.set_option('display.max_columns', None)
#print(dataset2.head(10))

y = y.map({'No': 0,
           'Yes': 1})
#print("y: \n", y.head())


# EKSPLORATIVNA ANALIZA
#provera korelacija:
correlation_matrix = dataset2.corr()
plt.figure(figsize = (18, 18))
sns.heatmap(correlation_matrix, annot = True, cmap = 'coolwarm', fmt = '.2f', linewidth = 0.5)
plt.title('Correlation Matrix')
plt.show()
corr_series = correlation_matrix.unstack()
corr_series = corr_series[corr_series != 1]
max_corr = corr_series.max()
min_corr = corr_series.min()
max_pair = corr_series[corr_series == max_corr].index[0]
min_pair = corr_series[corr_series == min_corr].index[0]

print(f"Najveća korelacija: {max_corr} između parova kolona {max_pair}")
print(f"Najmanja korelacija: {min_corr} između parova kolona {min_pair}")

#s obzirom na jaku korelaciju izmedju Adenopathy_No i N_N1b nakon analize izbacujem Adenopathy_No
dataset2 = dataset2.drop("Adenopathy_No", axis = 1)

# histogram za Age - za razumevanje raspodele uzrasta pacijenata
plt.figure(figsize=(8, 6))
sns.histplot(dataset['Age'], bins=20, kde=True)
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# box plot Age po Gender-u - razlike u uzrastu muskaraca i zena
plt.figure(figsize=(8, 6))
sns.boxplot(x='Gender', y='Age', data=dataset)
plt.title('Age Distribution by Gender')
plt.xlabel('Gender')
plt.ylabel('Age')
plt.show()

# pita za Risk kategorije
plt.figure(figsize=(8, 6))
dataset['Risk'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Risk Distribution')
plt.ylabel('')
plt.show()

# prikaz prosečne starosti pacijenata po kategoriji rizika
plt.figure(figsize=(10, 6))
sns.lineplot(data=dataset, x='Risk', y='Age', estimator='mean', ci=None, marker='o')
plt.title('Average Age by Risk Category')
plt.xlabel('Risk Category')
plt.ylabel('Average Age')
plt.show()

# zavisnosti Recurred u odnosu na Gender
plt.figure(figsize=(10, 6))
sns.countplot(x='Gender', hue='Recurred', data=dataset)
plt.title('Recurred Cases by Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

# recurred u odnosu na Age
plt.figure(figsize=(8, 6))
sns.boxplot(x='Recurred', y='Age', data=dataset)
plt.title('Age Distribution by Recurred')
plt.xlabel('Recurred')
plt.ylabel('Age')
plt.show()

# box plot recurred u odnosu na smoking
plt.figure(figsize=(8, 6))
sns.countplot(x='Smoking', hue='Recurred', data=dataset)
plt.title('Recurred Cases by Smoking Status')
plt.xlabel('Smoking')
plt.ylabel('Count')
plt.show()

# box plot za Risk kategorije u odnosu na Recurred
plt.figure(figsize=(10, 6))
sns.boxplot(x='Risk', y='Recurred', data=dataset)
plt.title('Recurred Cases by Risk Category')
plt.xlabel('Risk Category')
plt.ylabel('Recurred')
plt.show()

# bar plot za Stage u odnosu na Recurred
plt.figure(figsize=(10, 6))
sns.countplot(x='Stage', hue='Recurred', data=dataset)
plt.title('Recurred Cases by Cancer Stage')
plt.xlabel('Stage')
plt.ylabel('Count')
plt.show()

# prikazivanje raspodele ciljnog atributa 'Recurred'
plt.figure(figsize=(8, 6))
sns.countplot(x='Recurred', data=dataset)
plt.title('Distribution of Recurred')
plt.xlabel('Recurred')
plt.ylabel('Count')
plt.show()

# balansiranje preko smote:
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(dataset2, y)
#print("SMOTE ", pd.Series(y_smote).value_counts())
#print(X_smote)
# prikazivanje raspodele ciljnog atributa 'Recurred' nakon SMOTE
plt.figure(figsize=(8, 6))
sns.countplot(x=y_smote)
plt.title('Distribution of Recurred (After SMOTE)')
plt.xlabel('Recurred')
plt.ylabel('Count')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.3, stratify=y_smote)

#Normalizacija
scaler = MinMaxScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)
y_train_norm = y_train
y_test_norm = y_test
#print("X_train_norm: \n", X_train_norm.head())
#print("y_train_norm: \n", y_test_norm.head())
#print("X_test_norm: \n", X_test_norm.head())
#print("y_test_norm: \n", y_test_norm)

# KREIRANJE MODELA
# Stacking:
estimators = [
    ('rf', RandomForestClassifier(n_estimators = 100, random_state = 42)),
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ('svm', SVC(probability=True, random_state=42)),
    ('knn', KNeighborsClassifier())
]
meta_model = LogisticRegressionCV()
stacking_model = StackingClassifier(estimators = estimators, final_estimator = meta_model)

# Bagging
base_bagging_model = DecisionTreeClassifier(random_state=42)
bagging_model = BaggingClassifier(base_bagging_model, n_estimators = 100, random_state = 42)

# KNeighborsClassifier - odabir optimalnog k
def find_optimal_k(X_train, y_train, X_test, y_test, max_k=10):
    k_values = []
    accuracy_values = []
    for k in range(3, max_k+1, 2):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        knn_pred = knn.predict(X_test)
        accuracy = 1 - accuracy_score(y_test, knn_pred)
        k_values.append(k)
        accuracy_values.append(accuracy)

    optimal_k = k_values[np.argmin(accuracy_values)]
    return optimal_k, k_values, accuracy_values

optimal_k, k_values, accuracy_values = find_optimal_k(X_train_norm, y_train_norm, X_test_norm, y_test_norm, max_k=10)
print("Uzeta vrednost za k kod KNeighborsClassifier: ", optimal_k)

plt.figure(figsize=(10, 6))
plt.scatter(k_values, accuracy_values, color='blue', label='Tačnost za svaki k')
plt.plot(k_values, accuracy_values, color='red', linestyle='dashed', marker='o', markerfacecolor='red', markersize = 10)
plt.title('Metoda lakta za određivanje optimalnog broja k')
plt.xlabel('Broj k najbližih suseda (k)')
plt.ylabel('Greska')
plt.xticks(k_values)
plt.grid(True)
plt.legend()
plt.show()

# HIPERPARAMETRI
hiperparams = {
    'LogisticRegressionCV': {
        'Cs': [0.001, 0.01, 0.1, 1, 10, 100],
        'cv': [2, 3, 5, 10, 15]
    },
    'SVC': {
        'C': [0.01, 0.1, 1, 10, 100]
    },
    'KNeighborsClassifier': {
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    },
    'DecisionTreeClassifier': {
        'criterion': ['gini', 'entropy', 'log_loss']
    },
    'RandomForestClassifier': {
        'n_estimators': [50, 75, 100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    },
    'GradientBoostingClassifier': {
        'n_estimators': [75, 100, 200],   #default = 100
        'learning_rate': [0.1, 0.01],   #default = 0.1
        'max_depth': [3, 5, 7]   #default = 3
    },
    'AdaBoostClassifier': {
        'n_estimators': [50, 100, 150],  #default = 50
        'learning_rate': [0.1, 0.01, 1.0]    #default = 1.0
    },
    'LGBMClassifier': {
        'n_estimators': [100, 200],
        'learning_rate': [0.1, 0.01],
        'num_leaves': [20, 31, 50, 100]
    },
    'BaggingClassifier': {
        'n_estimators': [10, 50, 100, 150, 200],
    },
    'StackingClassifier': {
        'cv': [3, 5]
    }
}

models = [
    LogisticRegressionCV(),
    SVC(),
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    #GaussianNB(),                                  #losi rezultati
    RandomForestClassifier(),                       #bagging
    GradientBoostingClassifier(),                   #boosting
    AdaBoostClassifier(),                           #boosting
    LGBMClassifier(verbose = -1),                   #boosting
    bagging_model,                                  #bagging
    stacking_model                                  #stacking
]

feature_importances = []



for model in models:
    name = model.__class__.__name__
    model.fit(X_train_norm, y_train_norm)
    y_pred = model.predict(X_test_norm)

    if name == 'KNeighborsClassifier':
        find_optimal_k(X_train_norm, y_train_norm, X_test_norm, y_test_norm)

    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test_norm)

        # CRTANJE ROC KRIVE
        fpr, tpr, _ = roc_curve(y_test_norm, y_proba[:, 1])
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic Curve - ' + name)
        plt.legend(loc="lower right")
        #plt.show()

    print("------------------------------ ", name, " ------------------------------")
    #Metrike
    print("Tacnost: ", accuracy_score(y_test_norm, y_pred))
    print("Preciznost: ", precision_score(y_test_norm, y_pred))
    print("Odziv: ", recall_score(y_test_norm, y_pred))
    print("F1: ", f1_score(y_test_norm, y_pred))
    print("Matrica konfuzije: \n", confusion_matrix(y_test_norm, y_pred))
    # 2 nove metrike:
    print("Jaccard similarity coefficient:", jaccard_score(y_test_norm, y_pred))
    print("Area under the ROC Curve: ", roc_auc_score(y_test_norm, y_proba[:, 1]))
    print("Log loss: ", log_loss(y_test_norm, y_proba))

    #Unakrsna validacija
    cv_score = cross_val_score(model, X_train_norm, y_train_norm, cv = 5)
    print("Unakrsna validacija: ", cv_score)
    print("Srednja tacnost unakrsne validacije: ", cv_score.mean())

    model_hiperparam = hiperparams.get(name, {})
    if model_hiperparam:
        grid = GridSearchCV(model, model_hiperparam, cv = 5)
        grid.fit(X_train_norm, y_train_norm)
        print("---------------------------- NAKON PODESAVANJA HIPERPARAMETARA ----------------------------")
        print("Najbolji parametri:", grid.best_params_)

        best_model = grid.best_estimator_
        best_model.fit(X_train_norm, y_train_norm)
        y_pred = best_model.predict(X_test_norm)
        # Metrike
        print("Tacnost: ", accuracy_score(y_test_norm, y_pred))
        print("Preciznost: ", precision_score(y_test_norm, y_pred))
        print("Odziv: ", recall_score(y_test_norm, y_pred))
        print("F1: ", f1_score(y_test_norm, y_pred))
        print("Matrica konfuzije: \n", confusion_matrix(y_test_norm, y_pred))
        # 2 nove metrike:
        print("Jaccard similarity coefficient:", jaccard_score(y_test_norm, y_pred))
        if hasattr(best_model, 'predict_proba'):
            y_proba = best_model.predict_proba(X_test_norm)
            print("Area under the ROC Curve: ", roc_auc_score(y_test_norm, y_proba[:, 1]))
            print("Log loss: ", log_loss(y_test_norm, y_proba))

        # Unakrsna validacija
        cv_score = cross_val_score(best_model, X_train_norm, y_train_norm, cv=5)
        print("Unakrsna validacija: ", cv_score)
        print("Srednja tacnost unakrsne validacije: ", cv_score.mean())

    else:
        print("Model", name, "nema mrežu parametara za optimizaciju.")

    # Ispisivanje važnosti atributa
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_.flatten())
    else:
        result = permutation_importance(model, X_test_norm, y_test_norm, n_repeats=10, random_state=42, n_jobs=2)
        importances = result.importances_mean

    feature_importances.append(importances)

# agregiranje važnosti atributa
feature_importances = np.array(feature_importances)
mean_importances = np.mean(feature_importances, axis=0)

# ispis najvažnijih atributa
feature_names = dataset2.columns
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': mean_importances
})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

#print("Najvažniji atributi za sve modele:")
#print(feature_importance_df)

# vizualizacija najbitnijih atributa za sve modele
plt.figure(figsize=(15, 10))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('Importance')
plt.title('Feature Importances Aggregated Over All Models')
plt.gca().invert_yaxis()
plt.show()

# provera najbitnijih atributa za svaki model pojedinacno, obucavanje modela na odabranim atributima i ispis rezultata
for model in models:
    name = model.__class__.__name__
    print("------------------------------ ", name, " (najbitniji atributi) ------------------------------")

    if hasattr(model, 'feature_importances_'):
        rfe = RFE(model, n_features_to_select=10)
        rfe.fit(X_train_norm, y_train_norm)
        X_train_selected = rfe.transform(X_train_norm)
        X_test_selected = rfe.transform(X_test_norm)
        selected_features = dataset2.columns[rfe.support_]
        print("Najbitniji atributi za model ", name, " su: \n", selected_features)

    elif hasattr(model, 'coef_'):
        rfe = RFE(model, n_features_to_select=10)
        rfe.fit(X_train_norm, y_train_norm)
        X_train_selected = rfe.transform(X_train_norm)
        X_test_selected = rfe.transform(X_test_norm)
        selected_features = dataset2.columns[rfe.support_]
        print("Najbitniji atributi za model ", name, " su: ", selected_features)

    else:
        print("Model ", name, "nema opciju odabira najbitnijih atributa.")
        continue

    model.fit(X_train_selected, y_train_norm)
    y_pred = model.predict(X_test_selected)
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test_selected)

    # Metrike
    print("Tacnost: ", accuracy_score(y_test_norm, y_pred))
    print("Preciznost: ", precision_score(y_test_norm, y_pred))
    print("Odziv: ", recall_score(y_test_norm, y_pred))
    print("F1: ", f1_score(y_test_norm, y_pred))
    print("Matrica konfuzije: \n", confusion_matrix(y_test_norm, y_pred))
    print("Jaccard similarity coefficient:", jaccard_score(y_test_norm, y_pred))
    if hasattr(model, 'predict_proba'):
        print("Area under the ROC Curve: ", roc_auc_score(y_test_norm, y_proba[:, 1]))
        print("Log loss: ", log_loss(y_test_norm, y_proba))

    # Unakrsna validacija
    cv_score = cross_val_score(model, X_train_selected, y_train_norm, cv=5)
    print("Unakrsna validacija: ", cv_score)
    print("Srednja tacnost unakrsne validacije: ", cv_score.mean())
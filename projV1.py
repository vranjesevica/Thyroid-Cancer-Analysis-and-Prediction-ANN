### Ucitavanje potrebni biblioteka\
import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest, BaggingClassifier, StackingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV

import warnings

# Isključivanje upozorenja
warnings.filterwarnings("ignore", category=Warning)
# Potiskivanje specifičnih upozorenja
warnings.filterwarnings('ignore', category=UserWarning, module='lightgbm')
warnings.filterwarnings('ignore', category=ConvergenceWarning)

### 1. Preprocesiranje podataka
dataset = pd.read_csv("Thyroid_Diff.csv")
#print(dataset.head(10))


#provera da li ima nedostajucih vrednosti
missing_values = dataset.isnull().sum()
#print(missing_values)
#s obzirom da sam dobila ispis gde su svuda 0 -> nema nedostajucih podataka

X = dataset.drop('Recurred', axis = 1)
y = dataset['Recurred']
# detekcija anomalija:
# Izolaciona šuma za detekciju anomalija
iso_forest = IsolationForest(contamination=0.1)
outliers = iso_forest.fit_predict(X.select_dtypes(include=[np.number]))

X['Anomaly'] = outliers
print(X[X['Anomaly'] == -1])
#postoje neke anomalije -> videti kako resiti
#proveriti da li su anomalije dobro odradjenem, mzd ih ne raditi za age i jos nes

# Prikaži sve jedinstvene vrednosti u kolonama
for column in dataset.columns:
    # Prikazi sve jedinstvene vrednosti u trenutnoj koloni
    unique_values = dataset[column].unique()
    #print("Unique values in column '{}': {}".format(column, unique_values))

#pretvaranje stringova u brojeve
'''
label_cols = ['Gender', 'Smoking', 'Hx Smoking', 'Hx Radiothreapy', 'Thyroid Function', 'Physical Examination',
              'Adenopathy', 'Pathology', 'Focality', 'Risk', 'T', 'N', 'M', 'Stage', 'Response', 'Recurred']

le = LabelEncoder()
for col in label_cols:
    dataset[col] = le.fit_transform(dataset[col])

print(dataset.head(20))

'''

dataset['Age'] = pd.to_numeric(dataset['Age'])
dataset['Gender'] = dataset['Gender'].map({'M': 0, 'F': 1})
dataset['Smoking'] = dataset['Smoking'].map({'No': 0, 'Yes': 1})
dataset['Hx Smoking'] = dataset['Hx Smoking'].map({'No': 0, 'Yes': 1})
dataset['Hx Radiothreapy'] = dataset['Hx Radiothreapy'].map({'No': 0, 'Yes': 1})
dataset['Thyroid Function'] = dataset['Thyroid Function'].map({'Euthyroid': 0,
                                                               'Clinical Hyperthyroidism': 1,
                                                               'Clinical Hypothyroidism': 2,
                                                               'Subclinical Hyperthyroidism': 3,
                                                               'Subclinical Hypothyroidism': 4})
dataset['Physical Examination'] = dataset['Physical Examination'].map({'Single nodular goiter-left': 0,
                                                                       'Multinodular goiter': 1,
                                                                       'Single nodular goiter-right': 2,
                                                                       'Normal': 3,
                                                                       'Diffuse goiter': 4})
dataset['Adenopathy'] = dataset['Adenopathy'].map({'No': 0,
                                                   'Right': 1,
                                                   'Extensive': 2,
                                                   'Left': 3,
                                                   'Bilateral': 4,
                                                   'Posterior': 5})

dataset['Pathology'] = dataset['Pathology'].map({'Micropapillary': 0,
                                                 'Papillary': 1,
                                                 'Follicular': 2,
                                                 'Hurthel cell': 3})

dataset['Focality'] = dataset['Focality'].map({'Uni-Focal': 0,
                                               'Multi-Focal': 1})

dataset['Risk'] = dataset['Risk'].map({'Low': 0,
                                       'Intermediate': 1,
                                       'High': 2})

dataset['T'] = dataset['T'].map({'T1a': 0,
                                 'T1b': 1,
                                 'T2': 2,
                                 'T3a': 3,
                                 'T3b': 4,
                                 'T4a': 5,
                                 'T4b': 6})

dataset['N'] = dataset['N'].map({'N0': 0,
                                 'N1b': 1,
                                 'N1a': 2})

dataset['M'] = dataset['M'].map({'M0': 0,
                                 'M1': 1})

dataset['Stage'] = dataset['Stage'].map({'I': 0,
                                         'II': 1,
                                         'IVB': 2,
                                         'III': 3,
                                         'IVA': 4})

dataset['Response'] = dataset['Response'].map({'Indeterminate': 0,
                                               'Excellent': 1,
                                               'Structural Incomplete': 2,
                                               'Biochemical Incomplete': 3})

dataset['Recurred'] = dataset['Recurred'].map({'No': 0,
                                               'Yes': 1})
print(dataset)

X = dataset.drop('Recurred', axis = 1)
y = dataset['Recurred']
#izbacivanje atributa za koje smatram da ne uticu na formiranje izlaza
#videti kasnije

#da li ima jos nesto sa vezbi?
#razumevanje postojecih podataka

#normalizacija podataka
#scaler = StandardScaler()
#normalized_dataset = scaler.fit_transform(dataset.drop('Recurred', axis = 1))
#normalized_df = pd.DataFrame(normalized_dataset, columns = dataset.columns[:-1])
#normalized_df['Recurred'] = dataset['Recurred']
#print(normalized_df.head(20))
#mozda ne bih trebala za sve podatke da radim normalizaciju, vec samo za godine
# Izračunavanje srednje vrednosti i standardne devijacije za kolonu 'Age'
mean_age = dataset['Age'].mean()
std_age = dataset['Age'].std()

# Normalizacija kolone 'Age'
dataset['Age'] = (dataset['Age'] - mean_age) / std_age

# Prikaz prvih nekoliko redova DataFrame-a sa normalizovanom kolonom 'Age'
print(dataset[['Age']].head())

#drugi tip normalizacije je max min scaler

#encoding podataka
#to bi zapravo bilo prebacivanje stringova u num vrednosti,
#ja sam to uradila na neki nacin pa treba samo proveriti da li
#postoji nesto malo bolje od mog, preko tog encodinga
#ovo sam proverila ali sada ne znam da li znam koje su mi vrednosti koji broj


### 2. Eksplorativna analiza skupa

#provera da li postoje neke jake korelacije
#videti da li raditi nad X ili nad dataset
correlation_matrix = X.corr()
#prikaz korelacione matrice
plt.figure(figsize = (14, 10))
sns.heatmap(correlation_matrix, annot = True, cmap = 'coolwarm', fmt = '.2f', linewidth = 0.5)
plt.title('Correlation Matrix')
#plt.show()


#vizualizacija podataka
#prikaz na grafiku raspodele ciljnog atributa
# Histogram za raspodelu ciljnog atributa 'Recurred'
plt.figure(figsize=(8, 6))
sns.countplot(x='Recurred', data=dataset)
plt.xlabel('Recurred')
plt.ylabel('Count')
plt.title('Distribution of Recurred')
#plt.show()
#u odnosu na neke nezavisne promenljive
# Boxplot za prikaz distribucije recidiva po kategorijama rizika
plt.figure(figsize=(10, 6))
sns.boxplot(x='Risk', y='Recurred', data=dataset)
plt.xlabel('Risk')
plt.ylabel('Recurred')
plt.title('Box Plot of Recurred by Risk')
#plt.show()

# Histogram za raspodelu starosti
plt.figure(figsize=(8, 6))
sns.histplot(dataset['Age'], bins=20, kde=True)
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age Distribution')
#plt.show()

# Pie chart za raspodelu pola
gender_counts = dataset['Gender'].value_counts()
plt.figure(figsize=(8, 6))
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=140, colors=['lightblue', 'lightgreen'])
plt.title('Gender Distribution')
#plt.show()

# Linijski grafikon za prikaz starosti u odnosu na recidiv
plt.figure(figsize=(10, 6))
sns.lineplot(x='Age', y='Recurred', data=dataset)
plt.xlabel('Age')
plt.ylabel('Recurred')
plt.title('Age vs Recurred')
#plt.show()

# Primer box plotova za različite atribute
plt.figure(figsize=(10, 6))
sns.boxplot(data=dataset)
plt.title('Boxplot of Features')
plt.xticks(rotation=45)
#plt.show()
#print(dataset.dtypes)
#jos nesto sa vezbi?
#prikaz raspodele - varijansa, medijana, srednja vrednost
# Definišemo kolone koje želimo analizirati
columns = ['Age', 'Gender', 'Smoking', 'Hx Smoking', 'Hx Radiothreapy', 'Thyroid Function',
           'Physical Examination', 'Adenopathy', 'Pathology', 'Focality', 'Risk', 'T', 'N', 'M',
           'Stage', 'Response']

# Kreiramo DataFrame za skladištenje rezultata
results = pd.DataFrame(columns=['Column', 'Mean', 'Variance', 'Median'])
for column in columns:
    if dataset[column].dtype in [np.float64, np.int64]:  # Numerički podaci
        mean_value = dataset[column].mean()
        variance_value = dataset[column].var()
        median_value = dataset[column].median()
    else:
        mean_value = variance_value = median_value = np.nan  # Za nenumeričke podatke ne računamo ove statistike

    results = results._append({
        'Column': column,
        'Mean': mean_value,
        'Variance': variance_value,
        'Median': median_value
    }, ignore_index=True)

# Prikazujemo rezultate
print(results)

#imam bas veliku varijansu kod age pa bih mozda trebalo da ih normalizujem ->jesam
#varijansa, mean i medianu ima smisla raditi samo za age, pre i posle normalizacije
#ostalo ne moram raditi jer su podaci 0, 1 i sl
#redukcija dimenzionalnosti -> vidim na kraju
#to je zapravo smanjivanje broja atiributa

#iskoristiti razlicite tipove grafika: histograme, pite, linijske grafike...


print(X.dtypes)
print(y.dtypes)

### 3. Balansiranje skupa podataka
# Balansiranje podataka oversamplingom
oversampler = RandomOverSampler(random_state=42)
X_resampled, y_resampled = oversampler.fit_resample(X, y)

# Prikazivanje novog broja instanci po klasi
print(pd.Series(y_resampled).value_counts())
print(X_resampled)

#videti mozda jos neku tehniku za balansiranje pa probati kasnije


X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, stratify=y_resampled)

# Min Max normalizacija nad training i nas test skupom
scaler = MinMaxScaler()

# Skalirajte trening podatke
X_train_scaled = scaler.fit_transform(X_train)

# Skalirajte test podatke
X_test_scaled = scaler.transform(X_test)



###4. Kreiranje modela koji ce vrsiti klasifikaciju
# Definisanje modela

# Definisanje estimatora za stacking classifier modela
estimators = [
    ('lr', LogisticRegressionCV()),
    ('dt', DecisionTreeClassifier()),
    ('svc', SVC())
]

# Definisanje raspona hiperparametara za svaki model
param_grids = {
    'LogisticRegressionCV': {
        'Cs': [1, 10],
        'cv': [3, 5]
    },
    'SVC': {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf']
    },
    'RandomForestClassifier': {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    },
    'GradientBoostingClassifier': {
        'n_estimators': [100, 200],
        'learning_rate': [0.1, 0.01],
        'max_depth': [3, 5, 7]
    },
    'AdaBoostClassifier': {
        'n_estimators': [50, 100],
        'learning_rate': [1.0, 0.1, 0.01]
    },
    'XGBClassifier': {
        'n_estimators': [100, 200],
        'learning_rate': [0.1, 0.01],
        'max_depth': [3, 6, 9]
    },
    'LGBMClassifier': {
        'n_estimators': [100, 200],
        'learning_rate': [0.1, 0.01],
        'num_leaves': [31, 50, 100]
    },
    'BaggingClassifier': {
        'n_estimators': [10, 20],
        'max_samples': [0.5, 1.0],
        'max_features': [0.5, 1.0]
    },
    'StackingClassifier': {
        'cv': [3, 5]
    }
}

models = [
    LogisticRegressionCV(),
    SVC(),
    RandomForestClassifier(),
    GradientBoostingClassifier(),  # 1. algoritam koji nije koriscen na vezbama
    AdaBoostClassifier(),   # 2. algoritam koji nije koriscen na vezbama
    XGBClassifier(),   # 3. algoritam koji nije koriscen na vezbama
    LGBMClassifier(),   # 4. algoritam koji nije koriscen na vezbama
    BaggingClassifier(estimator=RandomForestClassifier(), n_estimators=10),  # Bagging
    AdaBoostClassifier(n_estimators=50),  # Boosting
    StackingClassifier(estimators=estimators, final_estimator=LogisticRegressionCV())  # Stacking
]

# Iteracija kroz modele i podešavanje hiperparametara
for model in models:
    model_name = model.__class__.__name__

    # Treniranje i evaluacija modela sa podrazumevanim hiperparametrima
    print(f"Training and evaluating {model_name} with default parameters:")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    print(f"{model_name} classification report (default parameters):")
    print(classification_report(y_test, y_pred))
    print()

    # Podesavanje hiperparametara i evaluacija modela
    param_grid = param_grids.get(model_name, {})  # Koristi parametre iz param_grids ako su definisani
    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=100, cv=5, n_jobs=-1,
                                       verbose=2, random_state=42)
    random_search.fit(X_train_scaled, y_train)
    best_model = random_search.best_estimator_
    y_pred = best_model.predict(X_test_scaled)
    print(f"Best parameters for {model_name}: {random_search.best_params_}")
    print(f"{model_name} classification report (optimized parameters):")
    print(classification_report(y_test, y_pred))
    print()
    '''
    #sa grid searchom
    # Podesavanje hiperparametara i evaluacija modela
    if model_name in param_grids:
        param_grid = param_grids[model_name]
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
        grid_search.fit(X_train_scaled, y_train)
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test_scaled)
        print(f"Best parameters for {model_name}: {grid_search.best_params_}")
        print(f"{model_name} classification report (optimized parameters):")
        print(classification_report(y_test, y_pred))
        print()
    else:
        print(f"No hyperparameter grid defined for {model_name}, skipping optimization.\n")
'''
'''
# Iteracija kroz modele
for model in models:
    imeKlase = model.__class__.__name__

    # Obuka modela
    model.fit(X_train_scaled, y_train)

    # Predviđanje rezultata
    y_pred = model.predict(X_test_scaled)

    # Evaluacija modela
    print(f"{imeKlase}:")
    print(classification_report(y_test, y_pred))
    print()



#stacking, bagging, boosting sta su kako funkcionisu i od svake grupe iskoristiti po jedan barem
base_model = RandomForestClassifier()
bagging_model = BaggingClassifier(base_model, n_estimators=10)
bagging_model.fit(X_train, y_train)

ada_model = AdaBoostClassifier(n_estimators=50)
ada_model.fit(X_train, y_train)

estimators = [
    ('lr', LogisticRegressionCV()),
    ('dt', DecisionTreeClassifier()),
    ('svc', SVC())
]

stacking_model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegressionCV())
stacking_model.fit(X_train, y_train)'''
### 5. Podesavanje hiperparametara kreiranih modela
#uradjeno gore nemam pojma sta se tu desava


### 6. Unakrsna validacija kreiranih modela
# Iteracija kroz modele
for idx, model in enumerate(models):
    model_name = f"Model_{idx+1}"

    # Unakrsna validacija
    print(f"Cross-validation scores for {model_name}:")
    scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    print(scores)
    print(f"Mean cross-validation score for {model_name}: {scores.mean()}")
    print()

print(X_resampled.head())

X_resampled_df = pd.DataFrame(X_resampled)
X_resampled_df['target'] = y_resampled
X_resampled_df.to_csv('X_resampled_with_target.csv', index=False)
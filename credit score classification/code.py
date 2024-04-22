##---------Type the code below this line------------------##
!pip install opendatasets
import pandas as pd
import os
import opendatasets as od
import warnings

# warnings.filterwarnings('ignore', category=pd.core.common.SettingWithCopyWarning)

"""
To download from kaggle, we need kaggle.json file in the present directory. it will download into a directory credit-score-classification
"""

dataset_url = 'https://www.kaggle.com/datasets/parisrohan/credit-score-classification'
od.download(dataset_url)
os.listdir('credit-score-classification')
##---------Type the code below this line------------------##

"""
Using pandas library, we can read the downloaded csv to convert it as dataFrame
"""
df = pd.read_csv('credit-score-classification/train.csv',low_memory=False)
print("First 5 records : ")
df.head(5)
print("Last 5 records : ")
df.tail(5)
"""
To display available columns

"""
df.columns
"""
Statistical Information
"""
df.describe()
df.describe(include='object').T
"""
Statistical Information about uniqueness
"""
df.nunique()
df.size
df.shape
##---------Type the code below this line------------------##
"""
To check for duplicate Data
"""
df.duplicated().value_counts()
"""
Source code to display missing records percentage in the data set
"""
null_percentage = (df.isnull().sum() / len(df)) * 100
pd.set_option('display.float_format', lambda x: '%.7f' % x)
print(null_percentage)
"""
To check for inconsistency
"""
# Validating for inconsistancy based on type
print("Percentage of Inconsistent column")
unique_column = []
inconsistency_column = []
for column in df.columns:
    if df[column].dtype == 'object':
        df[column].unique()
        unique_column.append(column)
    else:
        inconsistency_column.append(column)
percentage_based_on_type = len(inconsistency_column) / (len(inconsistency_column) + len(unique_column)) * 100
print(percentage_based_on_type)
##---------Type the code below this line------------------##
"""
To remove the duplicates if present
"""

#We don't have duplicates in our data set. If it exists, we can execute this command
df_duplicates = df.drop_duplicates()
df_duplicates.describe()
#To Update Missing data.
"""
To update Missing records based on column type/characteristics
"""

LIST_OF_COLUMNS_CAN_BE_UPDATED_BY_MOD = ["Num_Credit_Card","Num_Bank_Accounts","Num_of_Loan","Num_Credit_Inquiries","Credit_Mix"]
for col in LIST_OF_COLUMNS_CAN_BE_UPDATED_BY_MOD:
    most_frequent_item = df.groupby('Customer_ID')[col].agg(lambda x: x.mode().iloc[0])
    df[col] = df['Customer_ID'].map(most_frequent_item)
"""
To update Missing records based on column type/characteristics
"""
#name
string='nan'
df['Name'] = df.groupby('Customer_ID')['Name'].transform(lambda x: x.replace(string, x.mode().iloc[0]))
#replace with NaNfillna
df['Name'] = df.groupby('Customer_ID')['Name'].transform(lambda x: x.fillna(x.mode().iloc[0]))
#occupation
df['Occupation'] = df['Occupation'].replace('','NA')
df['Occupation'] = df.groupby('Customer_ID')['Occupation'].transform(lambda x: x.replace('NA', x.mode().iloc[0]))
#SSN
pattern = r'^\d{3}-\d{2}-\d{4}$'
df['SSN'] = df['SSN'].where(df['SSN'].str.match(pattern, na=False), other='NA')
df['SSN'] = df.groupby('Customer_ID')['SSN'].transform(lambda x: x.replace('NA', x.mode().iloc[0]))
#Age
pat = '.*_.*'
condition1 = (df['Age'] > '100') | (df['Age'].str.contains(pat, regex=True)) | (df['Age'] < '0')
df['Age'] = df.groupby('Customer_ID')['Age'].transform(lambda x: x.mask(condition1, x.mode().iloc[0]))
# df['Age'] = df['Age'].astype(int)

#All other columns

df = df.apply(lambda x: x.str.replace('_', '') if x.dtype == 'object' else x)
df.fillna(df.mode().iloc[0], inplace=True)
null_percentage = (df.isnull().sum() / len(df)) * 100
pd.set_option('display.float_format', lambda x: '%.7f' % x)
print(f"Null percentage post clean up : \n{null_percentage}")
"To handle All Other"
df.replace('', pd.NA, inplace=True)
df.fillna(df.mode().iloc[0], inplace=True)
required_cloumns = df[["Monthly_Inhand_Salary","Num_Bank_Accounts","Num_Credit_Card","Interest_Rate","Delay_from_due_date","Num_Credit_Inquiries","Credit_Utilization_Ratio","Total_EMI_per_month"]]
# BULK convert to Integer and float for graphs and other computations
df[["Annual_Income", "Num_of_Loan", "Num_of_Delayed_Payment", "Changed_Credit_Limit", "Outstanding_Debt"]] = df[["Annual_Income", "Num_of_Loan", "Num_of_Delayed_Payment", "Changed_Credit_Limit", "Outstanding_Debt"]].astype(float)
df[["Age","Amount_invested_monthly", "Monthly_Balance"]] = df[["Age","Amount_invested_monthly", "Monthly_Balance"]].astype(float)
"""
Removing the attributes which will not be used for computation
"""
df = df.drop(columns= ["ID", "Customer_ID", "Month", "Name", "SSN"])
df.dtypes
df['Occupation_Copy'] = df['Occupation']
"""
Performing OneHot Encoding using sklearn library
"""
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

df['Occupation'] = label_encoder.fit_transform(df['Occupation'])
df['Type_of_Loan'] = label_encoder.fit_transform(df['Type_of_Loan'])
df['Credit_Mix'] = label_encoder.fit_transform(df['Credit_Mix'])
df['Credit_History_Age'] = label_encoder.fit_transform(df['Credit_History_Age'])
df['Payment_of_Min_Amount'] = label_encoder.fit_transform(df['Payment_of_Min_Amount'])
df['Payment_Behaviour'] = label_encoder.fit_transform(df['Payment_Behaviour'])
#TO Validate whether the Occupation and other feature updated or not
df.dtypes
##---------Type the code below this line------------------##
"""
To handle outliers on numeric data
"""
columns_excluded_numbertype = df.select_dtypes(exclude=['number']).columns
numeric_df = df.drop(columns=columns_excluded_numbertype)
Q1 = numeric_df.quantile(0.25)
Q3 = numeric_df.quantile(0.75)
IQR = Q3 - Q1
outliers_set = ((numeric_df < (Q1 - 1.5 * IQR)) | (numeric_df > (Q3 + 1.5 * IQR))).any(axis=1)
outliers_set.describe()
new_df_for_processing = df[~outliers_set]
new_df_for_processing.describe()
##---------Type the code below this line------------------##
numerical= new_df_for_processing.select_dtypes('number').columns
categorical = new_df_for_processing.select_dtypes('object').columns
"""
Cleaned data set statistical information
"""
new_df_for_processing.describe(include = 'all')
##---------Type the code below this line------------------##
"""
Separate the data from the target such that the dataset is in the form of (X,y) or (Features, Label)
"""
df_target_excluded = df.drop('Credit_Score',axis=1)
df_target_excluded.columns
df_target_included = df['Credit_Score']

"""
Discretize / Encode the target variable or perform one-hot encoding on the target or any other as and if required.
"""

print("ANSWER: We have used encoding method from sklearn library. Source code has been updated in Section 3.3")

"""
Report the observations

"""

print("We have choosen `Credit_Score` as the target. Using this, we will be able to classify customer's risk category")
import seaborn as sns
import matplotlib.pyplot as plt

# Select numeric columns from the DataFrame
numeric_columns = new_df_for_processing.select_dtypes('number').columns
num_rows = (len(numeric_columns) + 1) // 2  # You can adjust the number based on your preference
fig, axes = plt.subplots(num_rows, 2, figsize=(8, num_rows * 5))
axes = axes.flatten()
for i, col in enumerate(numeric_columns):
    sns.scatterplot(y=new_df_for_processing["Credit_Score"], x=new_df_for_processing[col], ax=axes[i])
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Credit Score")
    axes[i].set_title(f"Credit Score vs {col}")

for j in range(i + 1, len(axes)):
    axes[j].axis('off')
plt.tight_layout()
"""
Correlation plots (to identify the optimal set of attributes that can be used for classification.
"""
# columns=new_df_for_processing[["Monthly_Inhand_Salary","Num_Bank_Accounts","Num_Credit_Card","Interest_Rate","Delay_from_due_date","Num_Credit_Inquiries","Credit_Utilization_Ratio","Total_EMI_per_month"]]
columns = new_df_for_processing
correlation_matrix = columns.corr(numeric_only=True)
plt.figure(figsize=(15, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.show()

"""
Why we choose this ?
"""

print('''
Identifying Relationships : Using corre
lation plots, we will be able to understand the relation between the attributess

Feature Selection: In predictive modeling tasks, it's essential to select features (independent variables) that are most relevant to the target variable. Correlation plots provide insights into which features
are highly correlated with the target variable and can be useful for feature selection.

Model Interpretation: Understanding the correlations between variables can aid in interpreting the results of predictive models. It helps in understanding which features are driving the predictions and how they relate to each other
''')
g = sns.pairplot(columns)
print('EDA FOR AGE GROUP')
new_df_for_processing['Age'].describe()
print('In the above output, max Age is 56, so we can have the bin max value around 60 number')
bin_size = [10, 20, 30, 40, 50, 60]
labels = ['10-19','20-29', '30-39', '40-49', '50-59']
new_df_for_processing['Age_Bucket'] = pd.cut(new_df_for_processing['Age'], bins=bin_size, labels=labels)
new_df_for_processing['Age_Bucket'].unique()

plt.figure(figsize=(15, 12))
sns.countplot(x='Age_Bucket', data=new_df_for_processing, palette='rainbow', order=new_df_for_processing['Age_Bucket'].value_counts().index)
plt.xticks(rotation=45)
plt.xlabel('Age_Bucket')
plt.ylabel('Count')
plt.title('Age Counts')
new_df_for_processing.select_dtypes('number')
##---------Type the code below this line------------------##

"""
Mutual Information (Information Gain):
"""

from sklearn.feature_selection import mutual_info_classif

X = new_df_for_processing.select_dtypes('number')# Replace "target_variable" with the actual column name
y = new_df_for_processing["Credit_Score"]

mutual_info = mutual_info_classif(X, y)

# Create a DataFrame to display feature importance
mi_df = pd.DataFrame({"Feature": X.columns, "Mutual_Information": mutual_info})
mi_df = mi_df.sort_values(by="Mutual_Information", ascending=False)
print(f"Top 5 : \n\n {mi_df.head(5)}")
## GINI INDEX

from sklearn.tree import DecisionTreeClassifier

X = new_df_for_processing.select_dtypes('number')
y = new_df_for_processing["Credit_Score"]

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X, y)

# Create a DataFrame to display feature importance
gini_df = pd.DataFrame({"Feature": X.columns, "Gini_Index": clf.feature_importances_})
gini_df = gini_df.sort_values(by="Gini_Index", ascending=True)
print(gini_df.tail(5))
##---------Type the code below this line------------------##
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score,confusion_matrix,roc_curve, auc

"""
Using scikit-learn (sklearn), we are performing splits on a dataset into two subsets: 
1) one for training a machine learning model 
2) Second one for testing its performance.
"""

y = new_df_for_processing['Credit_Score']
X = new_df_for_processing.drop(['Credit_Score','Occupation_Copy','Age_Bucket','Credit_History_Age','Num_of_Loan','Age','Monthly_Balance','Payment_Behaviour','Occupation','Amount_invested_monthly','Credit_Utilization_Ratio'], axis=1)
# X = new_df_for_processing.drop(['Credit_Score','Occupation_Copy','Age_Bucket'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
"""
Random Forest Classifier Approach
"""

ml_tech_name = "Random Forest Classifier"
rf_params = {
    'max_depth': None,
    'min_samples_split': 2,
    'n_estimators': 200
}

model_rf = RandomForestClassifier(**rf_params)

model_rf.fit(X_train, y_train)

y_pred_rf = model_rf.predict(X_test)



accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf, average='weighted')
recall_score_rf = recall_score(y_test, y_pred_rf, average='weighted')
f1_score_rf = f1_score(y_test, y_pred_rf, average='weighted')

print(f"Results of {ml_tech_name} \n Accuracy Score :{accuracy_rf} \n Precision Score : {precision_rf} \n Recall_Score: {recall_score_rf}\n F1_score: {f1_score_rf}")

cm_rf = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Blues", xticklabels=model_rf.classes_, yticklabels=model_rf.classes_)
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.title(f"{ml_tech_name} Confusion Matrix \nAccuracy: {accuracy:.2f}")
plt.show()
##---------Type the code below this line------------------##

from sklearn.ensemble import GradientBoostingClassifier

ml_tech_name = "Gradient Boosting"
# Define the model parameters
gb_params = {
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 3
}

# Initialize the GradientBoostingClassifier
model_gb = GradientBoostingClassifier(**gb_params)

# Fit the model to the training data
model_gb.fit(X_train, y_train)

# Make predictions on the testing data
y_pred_gb = model_gb.predict(X_test)

accuracy_gb = accuracy_score(y_test, y_pred_gb)
precision_gb = precision_score(y_test, y_pred_gb, average='weighted')
recall_score_gb = recall_score(y_test, y_pred_gb, average='weighted')
f1_score_gb = f1_score(y_test, y_pred_gb, average='weighted')

# accuracy_gb = accuracy_score(y_test, y_pred_gb)
# print(f"Gradient Boosting Model Accuracy: {accuracy_gb}")
print(f"Results of Gradient Boosting Model Accuracy \n Accuracy Score :{accuracy_gb} \n Precision Score: {precision_gb} \n Recall_Score: {recall_score_gb} \n F1_score: {f1_score_gb}")

cm_gb = confusion_matrix(y_test, y_pred_gb)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_gb, annot=True, fmt="d", cmap="Greens", xticklabels=model_gb.classes_, yticklabels=model_gb.classes_)
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.title(f"{ml_tech_name} Confusion Matrix \nAccuracy: {accuracy_gb:.2f}")
plt.show()
##---------Type the code below this line------------------##

import matplotlib.pyplot as plt

# Define performance metrics for Gradient Boosting
gb_metrics = {
    'Accuracy': accuracy_gb,
    'Precision': precision_gb,
    'Recall': recall_score_gb,
    'F1 Score': f1_score_gb
}

# Define performance metrics for Random Forest
rf_metrics = {
    'Accuracy': accuracy_rf,
    'Precision': precision_rf,
    'Recall': recall_score_rf,
    'F1 Score': f1_score_rf
}

# Metrics and corresponding labels
metrics = list(gb_metrics.keys())
gb_values = list(gb_metrics.values())
rf_values = list(rf_metrics.values())

# Plotting
x = range(len(metrics))

plt.figure(figsize=(10, 6))

plt.bar(x, gb_values, width=0.4, label='Gradient Boosting', color='skyblue', align='center')
plt.bar(x, rf_values, width=0.4, label='Random Forest', color='lightcoral', align='edge')

plt.xlabel('Performance Metrics')
plt.ylabel('Scores')
plt.title('Comparison of Performance Metrics: Gradient Boosting vs. Random Forest')
plt.xticks(x, metrics)
plt.legend()
plt.show()

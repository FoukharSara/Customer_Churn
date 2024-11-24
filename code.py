import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import xgboost as xgb
from imblearn.over_sampling import SMOTE




# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report

dataset="data/data.csv"

df = pd.read_csv(dataset)

#print(df.shape)
#print(df.info'())
#print(df.isnull().sum())
# 11 in Total Charges no missing values
#print(df.duplicated().sum())
#no duplicated

#print(df.describe(include="object").T)

df = df.drop(columns=['customerID'], errors='ignore') 

df['TotalCharges']=pd.to_numeric(df.TotalCharges,errors='coerce')
df.drop(labels=df[df['tenure'] == 0].index, axis=0, inplace=True)

#print(df.isnull().sum())


#heatmap
# plt.figure(figsize=(20, 15))
# encoded_df = df.apply(lambda x: pd.factorize(x)[0])
# s = encoded_df.corr()
# mask = np.triu(np.ones_like(s, dtype=bool))
# sns.heatmap(s,mask=mask,annot=True)
# plt.show()

df['Churn'] = pd.factorize(df['Churn'])[0]


df=pd.get_dummies(data=df,columns=['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
     'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
     'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
     'PaperlessBilling', 'PaymentMethod'])

#print(df.columns)
#print(df.corr()['Churn_Yes'].sort_values())

X = df.drop(columns=['Churn'])
Y = df['Churn'].values
low_importance_features = ['SeniorCitizen', 'gender_Male','gender_Female']
X = X.drop(columns=low_importance_features)

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=123)

numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
#print(df[numerical_cols].describe())

scaler = MinMaxScaler()

X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

xgb_model = xgb.XGBClassifier(n_estimators=500, random_state=50, max_depth=6, learning_rate=0.1, use_label_encoder=False)


# Fit the model
xgb_model.fit(X_train, Y_train)

# Make predictions
prediction_test = xgb_model.predict(X_test)

# Print the accuracy
print("Accuracy:", accuracy_score(Y_test, prediction_test))
print("Precision:", precision_score(Y_test, prediction_test))
print("Recall:", recall_score(Y_test, prediction_test))
print("F1-Score:", f1_score(Y_test, prediction_test))
print("AUC-ROC:", roc_auc_score(Y_test, prediction_test))

# Classification report
print("\nClassification Report:\n", classification_report(Y_test, prediction_test))

# Confusion Matrix
cm = confusion_matrix(Y_test, prediction_test)
print("\nConfusion Matrix:\n", cm)














# df=pd.get_dummies(data=df,columns=['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
#     'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
#     'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
#     'PaperlessBilling', 'PaymentMethod', 'Churn'])


# s = df.corr()
# print(s)
# plt.figure(figsize=(15,15))
# sns.heatmap(s,annot=True)
# plt.title("Correlation Heatmap with Categorical Data Included")

# churn_corr = df.corr()['Churn_Yes'].sort_values(ascending=False)
# print(churn_corr)
# plt.figure(figsize=(10, 8))
# sns.heatmap(df[['Churn_Yes', 'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']].corr(), annot=True, cmap='coolwarm', fmt=".2f")
# plt.title("Correlation between Key Features and Churn")
# plt.show()

# # Bar plots for categorical features
# categorical_cols = ['gender_Female', 'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes', 
#                     'InternetService_Fiber optic', 'Contract_Month-to-month', 'PaymentMethod_Electronic check']

# for col in categorical_cols:
#     plt.figure(figsize=(6, 4))
#     sns.countplot(data=df, x=col, hue='Churn_Yes')
#     plt.title(f'Relation between {col} and Churn')
#     plt.show()




# X = df.drop(columns=['Churn_Yes', 'Churn_No'])  # Drop churn columns, as they're the target
# y = df['Churn_Yes']  # Target: Churn

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# rf = RandomForestClassifier(n_estimators=100, random_state=42)
# rf.fit(X_train, y_train)

# importances = rf.feature_importances_
# features = X.columns

# feature_importance = pd.DataFrame({'Feature': features, 'Importance': importances})
# feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

# print(feature_importance)

# y_pred = rf.predict(X_test)
# print(classification_report(y_test, y_pred))

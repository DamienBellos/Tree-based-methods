import numpy as np
import pandas as pd
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

loan_data = pd.read_csv('C:/Users/PC1/Desktop/GitHub/Decision trees and random forests/LoanStats3a.csv',
                        header=1, skip_blank_lines=False)

loan_data = loan_data.dropna(thresh=500, axis=1)

df_loan = loan_data.filter(['purpose', 'funded_amnt_inv', 'int_rate',
                            'installment', 'annual_inc', 'dti', 'earliest_cr_line',
                            'revol_bal', 'revol_util', 'inq_last_6mths', 'loan_status'], axis=1, )

# separate data by policy code -> 1 = permitted loan, 0 = no loan
df1_loan = np.split(df_loan, df_loan[df_loan.isnull().all(axis=1)].index)
df1 = df1_loan[0]
df2 = df1_loan[3]
df1['policy_code'] = 1
df2['policy_code'] = 0
df_loan = df1.append(df2[1:])

# clean column data
df_loan['fully_paid'] = np.where(df_loan['loan_status'] == 'Fully Paid', 1, 0)
df_loan['log.annual.inc'] = np.log(df_loan['annual_inc'])
df_loan['int.rate'] = df_loan['int_rate'].str.rstrip('%').astype('float') / 100.0
df_loan['revol.util'] = df_loan['int_rate'].str.rstrip('%').astype('float') / 100.0
df_loan.drop(['int_rate', 'revol_util', 'annual_inc', 'loan_status'], axis=1, inplace=True)

# Determine the number of days individuals have had a credit line
end_date = datetime(2011, 12, 31)
df_loan['earliest_cr_line'] = pd.to_datetime(df_loan['earliest_cr_line'])
df_loan['earliest_cr_line'] = (end_date - df_loan['earliest_cr_line']).dt.days

# df_loan = df_loan.fillna({'desc':'none'})
df_loan = df_loan.fillna(0)

## Only 182 rows contain the FICO score so the description column will not be used
# df_loan['fico']= np.where(df_loan['desc'].str.contains('fico', case=False), 1,0)
# print(len(df_loan[df_loan['fico']==1]))
# sns.pairplot(df_loan, hue='policy_code')
# plt.show()

# Debt to income for the two credit.policy outcomes
plt.figure(figsize=(10, 6))
df_loan[df_loan['policy_code'] == 1]['dti'].hist(bins=30, label='Loan approved: Yes', alpha=0.5)
df_loan[df_loan['policy_code'] == 0]['dti'].hist(color='red', bins=30, label='Loan approved: No', alpha=0.5)
plt.legend()
plt.xlabel('Debt to income ratio')
plt.ylabel('Individuals')
plt.title('Lending club - loan approvals by debt to income ratio')
plt.show()

# Countplot showing  counts of loans by purpose and hue fully paid. **
plt.figure(figsize=(20, 6))
c_plot = sns.countplot(data=df_loan, x='purpose', hue='policy_code', palette='RdBu')
plt.xlabel('Purpose', fontweight='bold')
plt.ylabel('Count', fontweight='bold')
plt.title('Loan status by purpose', fontweight='bold')
L = plt.legend()
L.get_texts()[0].set_text('Not repaid')
L.get_texts()[1].set_text('Repaid')
plt.tight_layout()

# Compare log of debt to income ratio to the interest rate
sns.jointplot(data=df_loan, x='log.annual.inc', y='int.rate', color='indigo')
plt.show()

# Compare the trend between fully_paid and policy_code.
sns.set_style('darkgrid')
sns.lmplot(data=df_loan, x='log.annual.inc', y='int.rate', hue='policy_code',
           legend_out=True, col='fully_paid', palette='Set1')
plt.show()

'''
A decision tree or random forest model would likely be able to predict the policy_code
since there is a little separation between groups (ex. sns.pairplot(df_loan))
'''
# Generate separate columns of dummy variables from the purpose column
cat_feats = ['purpose']
df_loan = pd.get_dummies(df_loan, columns=cat_feats, drop_first=True)

# Train Test Split
X = df_loan.drop('policy_code', axis=1)
y = df_loan['policy_code']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Train a Decision Tree model
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)

# Predict and evaluate the Decision Tree
predictions = dtree.predict(X_test)
print('Decision Tree')
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))

# Compare to a Random Forest model
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)

# Predict and evaluate
rfc_pred = rfc.predict(X_test)
print('Random Forest')
print(classification_report(y_test, rfc_pred))
print(confusion_matrix(y_test, rfc_pred))
# As you might expect the random forest did slighlty better than the decision tree

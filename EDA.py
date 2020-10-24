import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train_transaction_ds = pd.read_csv('/content/drive/My Drive/Group #2: Detecting Fraudulent Transactions/Project Code/dataset/train_transaction.csv')
train_identity_ds = pd.read_csv('/content/drive/My Drive/Group #2: Detecting Fraudulent Transactions/Project Code/dataset/train_identity.csv')
test_identity_ds = pd.read_csv('/content/drive/My Drive/Group #2: Detecting Fraudulent Transactions/Project Code/dataset/test_identity.csv')
test_transaction_ds = pd.read_csv('/content/drive/My Drive/Group #2: Detecting Fraudulent Transactions/Project Code/dataset/test_transaction.csv')

print(train_transaction_ds['isFraud'].dtype)
num_fraud = train_transaction_ds.groupby(['isFraud']).count()
print(num_fraud)

plt.xlabel('Transaction Classification')
plt.ylabel('Count')
plt.title('Percentage of Fraud vs Non-Fraud Transactions')

plt.xlabel('Transaction Classification')
plt.ylabel('Count')
plt.title('Percentage of Fraud vs Non-Fraud Transactions')

total = len(train_transaction_ds)
plt.figure(figsize=(20,6))

plt.subplot(121)
g = sns.countplot(x='isFraud', data=train_transaction_ds)
g.set_title("Transaction Classification Distribution", fontsize=22)
g.set_xlabel("Transaction Classification", fontsize=18)
g.set_ylabel('Count', fontsize=18)
for p in g.patches:
    height = p.get_height()
    g.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}%'.format(height/total*100),
            ha="center", fontsize=15) 
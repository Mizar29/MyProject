# 一、数据预处理
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 设置文件路径
data_folder = '/mnt/data'

# 读取数据
train_data = pd.read_csv('D:/myprojects/Spaceship Titanic/train.csv')
test_data = pd.read_csv('D:/myprojects/Spaceship Titanic/test.csv')

# 填补缺失值
train_data['HomePlanet'].fillna(train_data['HomePlanet'].mode()[0], inplace=True)
train_data['CryoSleep'].fillna(False, inplace=True)
train_data['Cabin'].fillna('Unknown', inplace=True)
train_data['Destination'].fillna(train_data['Destination'].mode()[0], inplace=True)
train_data['Age'].fillna(train_data['Age'].median(), inplace=True)
train_data['VIP'].fillna(False, inplace=True)
train_data.fillna(0, inplace=True)

test_data['HomePlanet'].fillna(test_data['HomePlanet'].mode()[0], inplace=True)
test_data['CryoSleep'].fillna(False, inplace=True)
test_data['Cabin'].fillna('Unknown', inplace=True)
test_data['Destination'].fillna(test_data['Destination'].mode()[0], inplace=True)
test_data['Age'].fillna(test_data['Age'].median(), inplace=True)
test_data['VIP'].fillna(False, inplace=True)
test_data.fillna(0, inplace=True)

# 特征工程
def process_cabin(df):
    df['Deck'] = df['Cabin'].apply(lambda x: x.split('/')[0] if x != 'Unknown' else 'Unknown')
    df['Side'] = df['Cabin'].apply(lambda x: x.split('/')[-1] if x != 'Unknown' else 'Unknown')
    return df

train_data = process_cabin(train_data)
test_data = process_cabin(test_data)

# 转换分类变量为数值
categorical_features = ['HomePlanet', 'Destination', 'Deck', 'Side']

for feature in categorical_features:
    le = LabelEncoder()
    train_data[feature] = le.fit_transform(train_data[feature])
    test_data[feature] = le.transform(test_data[feature])

# 删除无关特征
train_data.drop(['Name', 'Cabin', 'PassengerId'], axis=1, inplace=True)
test_data.drop(['Name', 'Cabin'], axis=1, inplace=True)

# 准备训练和测试数据
X = train_data.drop('Transported', axis=1).values
y = train_data['Transported'].astype(int).values
X_test = test_data.drop('PassengerId', axis=1).values

# 将数据分割成训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 二、构建和训练梯度提升树模型
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score,f1_score

# 定义模型
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

# 训练模型
model.fit(X_train, y_train)

# 在验证集上评估模型
y_val_pred = model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
f1 = f1_score(y_val, y_val_pred)
print(f"Accuracy: {val_accuracy}, F1 Score: {f1}")

# 三、验证测试集
# 验证测试集
predictions = model.predict(X_test)
predictions_bool = predictions.astype(bool)

# 获取特征重要性
feature_importances = model.feature_importances_
print(f'feature_importances = {feature_importances}')
feature_names = train_data.drop('Transported', axis=1).columns

# 可视化特征重要性
sorted_idx = feature_importances.argsort()[::-1]
print(f'sorted_idx = {sorted_idx}')
plt.barh(feature_names[sorted_idx], feature_importances[sorted_idx])
plt.xlabel("Feature Importance")
plt.show()

# 四、休眠与传送的关系
# 提取 "CryoSleep" 和 "Transported" 列
cryo_sleep_transported = train_data[['CryoSleep', 'Transported']]

# 按 "CryoSleep" 状态分组，并计算被传送的比例
group_analysis = cryo_sleep_transported.groupby('CryoSleep')['Transported'].mean()
print(group_analysis)

# 可视化
import seaborn as sns

# 绘制条形图
sns.barplot(x='CryoSleep', y='Transported', data=cryo_sleep_transported)
plt.xlabel('CryoSleep')
plt.ylabel('Proportion of Transported')
plt.title('Impact of CryoSleep on Being Transported')
plt.show()

# 创建提交文件
submission = pd.DataFrame({
    'PassengerId': test_data['PassengerId'],
    'Transported': predictions_bool
})

submission.to_csv('D:/myprojects/Spaceship Titanic/submission.csv', index=False)
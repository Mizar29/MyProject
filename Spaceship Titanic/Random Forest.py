# 一、数据预处理
import pandas as pd

# 读取数据
train_data = pd.read_csv('D:/myprojects/Spaceship Titanic/train.csv')
test_data = pd.read_csv('D:/myprojects/Spaceship Titanic/test.csv')

# 查看数据结构
print(train_data.head())
print(test_data.head())

# 二、数据清洗和特征工程
# 查看缺失值情况
print(train_data.isnull().sum())
print(test_data.isnull().sum())

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
from sklearn.preprocessing import LabelEncoder

categorical_features = ['HomePlanet', 'Destination', 'Deck', 'Side']

for feature in categorical_features:
    le = LabelEncoder()
    train_data[feature] = le.fit_transform(train_data[feature])
    test_data[feature] = le.transform(test_data[feature])

# 删除无关特征
train_data.drop(['Name', 'Cabin', 'PassengerId'], axis=1, inplace=True)
test_data.drop(['Name', 'Cabin'], axis=1, inplace=True)

# 三、建立模型并确定重要特征
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 分割数据集
X = train_data.drop('Transported', axis=1)
y = train_data['Transported']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 建立模型
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 验证模型
y_pred = rf_model.predict(X_val)
print(f"Accuracy: {accuracy_score(y_val, y_pred)}")
print(classification_report(y_val, y_pred))

# 确定重要特征
importances = rf_model.feature_importances_
features = X.columns
feature_importances = pd.DataFrame({'Feature': features, 'Importance': importances})
feature_importances.sort_values(by='Importance', ascending=False, inplace=True)
print(feature_importances.head(3))

# 四、预测测试集
# 预测测试集
test_data['Transported'] = rf_model.predict(test_data.drop('PassengerId', axis=1))

# 创建提交文件
submission = test_data[['PassengerId', 'Transported']]
submission.to_csv('D:/myprojects/Spaceship Titanic/submission.csv', index=False)

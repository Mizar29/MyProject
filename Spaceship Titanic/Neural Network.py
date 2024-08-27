# 一、数据预处理
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

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

print(train_data.dtypes)
print(test_data.dtypes)

# 删除无关特征
train_data.drop(['Name', 'Cabin', 'PassengerId'], axis=1, inplace=True)
test_data.drop(['Name', 'Cabin'], axis=1, inplace=True)

print('*' * 50)
print(train_data.dtypes)
print('*' * 50)
print(test_data.dtypes)

train_data = train_data.astype(float)  # numpy强制类型转换
test_data = test_data.astype(float)

print('*' * 50)
print(train_data.dtypes)
print('*' * 50)
print(test_data.dtypes)

# 准备训练和测试数据
X = train_data.drop('Transported', axis=1).values
y = train_data['Transported'].astype(int).values
X_test = test_data.drop('PassengerId', axis=1).values

# 将数据转换为PyTorch张量
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# 二、构建神经网络模型
import torch.nn as nn
import torch.optim as optim

# 定义神经网络模型
class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.relu(self.layer3(x))
        x = torch.sigmoid(self.output(x))
        return x

# 初始化模型、损失函数和优化器
model = NeuralNetwork(X_train_tensor.shape[1])
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch.view(-1, 1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * X_batch.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# 三、评估模型
model.eval()
val_loss = 0.0
with torch.no_grad():
    for X_batch, y_batch in val_loader:
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch.view(-1, 1))
        val_loss += loss.item() * X_batch.size(0)

val_loss /= len(val_loader.dataset)
print(f"Validation Loss: {val_loss:.4f}")

# 四、预测测试集
# 预测测试集
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor)
    predictions = (predictions > 0.5).int().numpy().reshape(-1)

# 创建提交文件
submission = pd.DataFrame({
    'PassengerId': test_data['PassengerId'],
    'Transported': predictions
})

submission.to_csv('D:/myprojects/Spaceship Titanic/submission.csv', index=False)

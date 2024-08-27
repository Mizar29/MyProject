import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from scipy.stats import chi2_contingency,ttest_ind

# 读取数据
data = pd.read_csv('D:/myprojects/Spaceship Titanic/train.csv')

new_data = data[['CryoSleep', 'Transported']]

new_data['CryoSleep'].fillna(False, inplace=True)

# 构建混淆矩阵
cm = confusion_matrix(new_data['CryoSleep'], new_data['Transported'])

# 使用seaborn绘制混淆矩阵
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[False, True], yticklabels=[False, True])
plt.xlabel('Predicted Transported')
plt.ylabel('Actual CryoSleep')
plt.title('Confusion Matrix: CryoSleep vs Transported')
plt.show()

# 起始星球检验
crosstab = pd.crosstab(data['HomePlanet'], data['Transported'])
chi2, p, dof, expected = chi2_contingency(crosstab)
print(f"Chi-Square Test: chi2 = {chi2}, p-value = {p}")
print(crosstab)

# 数值型数据t检验
transported = data[data['Transported'] == True]['Age'].dropna()
not_transported = data[data['Transported'] == False]['Age'].dropna()
t_stat, p_value = ttest_ind(transported, not_transported)
print(f"T-Test: t-statistic = {t_stat}, p-value = {p_value}")

# 箱线图
# 设置图形风格
sns.set(style="whitegrid")

# 创建一个5个子图的画布
fig, axes = plt.subplots(1, 5, figsize=(20, 6))

# 消费特征列表
consumption_features = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

# 绘制每个消费特征的箱线图
for i, feature in enumerate(consumption_features):
    sns.boxplot(x='Transported', y=feature, data=data, ax=axes[i])
    axes[i].set_title(f'{feature} by Transported')

plt.tight_layout()
plt.show()

# 密度图
# 创建一个5个子图的画布
fig, axes = plt.subplots(1, 5, figsize=(20, 6))

# 绘制每个消费特征的密度图
for i, feature in enumerate(consumption_features):
    sns.kdeplot(data[data['Transported'] == True][feature], shade=True, label='Transported', ax=axes[i])
    sns.kdeplot(data[data['Transported'] == False][feature], shade=True, label='Not Transported', ax=axes[i])
    axes[i].set_title(f'{feature} Density by Transported')
    axes[i].legend()

plt.tight_layout()
plt.show()

# 分布图
# 创建一个5个子图的画布
fig, axes = plt.subplots(1, 5, figsize=(20, 6))

# 绘制每个消费特征的分布图
for i, feature in enumerate(consumption_features):
    sns.histplot(data=data, x=feature, hue='Transported', element='step', ax=axes[i], kde=True)
    axes[i].set_title(f'{feature} Distribution by Transported')

plt.tight_layout()
plt.show()

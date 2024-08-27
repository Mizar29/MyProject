import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from scipy.stats import chi2_contingency,ttest_ind

# 读取数据
data = pd.read_csv('D:/myprojects/Spaceship Titanic/train.csv')

# 消费行为特征
for col in ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']:
    transported = data[data['Transported'] == True][col].dropna()
    not_transported = data[data['Transported'] == False][col].dropna()
    t_stat, p_value = ttest_ind(transported, not_transported)
    print(f"T-Test for {col}: t-statistic = {t_stat}, p-value = {p_value}")

# 定义消费特征列名
consumption_features = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

# 分组计算统计量
grouped_stats = data.groupby('Transported')[consumption_features].agg(['mean', 'median', 'std'])

# 显示统计结果
print(grouped_stats)

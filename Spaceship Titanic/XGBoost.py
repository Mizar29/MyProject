from graphviz import Digraph

# 创建一个有颜色和样式的流程图
dot = Digraph(comment='Gradient Boosting Model Pipeline')

# 设置节点的形状、颜色和样式
dot.attr('node', shape='box', style='filled', fontname="Arial")

# 添加节点，指定颜色
dot.node('A', 'Data Preparation', fillcolor='lightblue')
dot.node('B', 'Feature Engineering', fillcolor='lightgreen')
dot.node('C', 'Model Initialization', fillcolor='yellow')
dot.node('D', 'Hyperparameter Tuning', fillcolor='orange')
dot.node('E', 'Model Training', fillcolor='lightcoral')
dot.node('F', 'Model Evaluation', fillcolor='lightgoldenrod')
dot.node('G', 'Model Deployment', fillcolor='lightpink')

# 设置边的颜色和样式
dot.attr('edge', color='black')

# 添加边连接节点
dot.edges(['AB', 'BC', 'CD', 'DE', 'EF', 'FG'])

# 输出流程图
dot.render('gbt_pipeline_colored', format='png', view=True)


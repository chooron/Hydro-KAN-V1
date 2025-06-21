import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# --- 1. 全局样式设置：字体和字号 ---
# 注意: 要使用 'Times New Roman', 您的操作系统中必须安装了该字体。
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 18 # 设置一个更大的基础字号

m50_criteria = pd.read_csv("src/stats/m50_base-criteria.csv")
k50_criteria = pd.read_csv("src/stats/k50_base-criteria.csv")
exphydro_criteria = pd.read_csv("src/stats/exphydro(disc,withst)-criteria.csv")

m50_test_nse = m50_criteria["nse-test"].values
k50_test_nse = k50_criteria["nse-test"].values
exphydro_test_nse = exphydro_criteria["nse-test"].values

m50_test_nse[m50_test_nse < -1] = -1
k50_test_nse[k50_test_nse < -1] = -1
exphydro_test_nse[exphydro_test_nse < -1] = -1
n_points = len(exphydro_test_nse)

# --- 3. 将数据整理成“长格式” DataFrame (与之前相同) ---
data = {
    'nse': np.concatenate([m50_test_nse, k50_test_nse]),
    'model': ['M50'] * n_points + ['K50'] * n_points,
    'exphydro_nse': np.concatenate([exphydro_test_nse, exphydro_test_nse])
}
df = pd.DataFrame(data)

# --- 4. 开始绘图 ---

# 设置图形尺寸
fig, ax = plt.subplots(figsize=(9, 7), dpi=300)

# Part 1: "云" - 绘制半透明的提琴图作为背景
sns.violinplot(
    data=df,
    x='model',
    y='nse',
    ax=ax,
    inner=None,
    color=".8",
    cut=0
)

# Part 2: "蜂群" - 绘制核心的蜂群散点图
sns.swarmplot(
    data=df,
    x='model',
    y='nse',
    ax=ax,
    hue='exphydro_nse',
    palette='viridis',
    s=6, # 可以适当增大点的大小
    legend=False
)

# --- 5. 创建并添加颜色条 (Colorbar) ---
norm = plt.Normalize(df['exphydro_nse'].min(), df['exphydro_nse'].max())
sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
sm.set_array([])

# 添加颜色条并设置其标签
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label('ExpHydro NSE', rotation=270, labelpad=20) # 增加了 labelpad 使标签离颜色条远一些

# --- 6. 美化图表 (已更新) ---
# ax.set_title(...) # 这一行被移除，不再生成标题
ax.set_ylabel('NSE (Test)')
ax.set_xlabel('Model')
ax.set_ylim(-1, 1)

# 调整刻度标签的字号 (可选, 如果全局设置不够,可以单独调整)
# ax.tick_params(axis='both', which='major', labelsize=14)

# 调整布局以避免标签重叠
plt.tight_layout()

# 显示图表
plt.show()
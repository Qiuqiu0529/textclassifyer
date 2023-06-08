import matplotlib.pyplot as plt

# 错误数据
data = {
    "小组人员合作不协调": 7,
    "任务量与时间安排不妥当": 5,
    "计划划分不明确": 5,
    "需求不明确": 4,
    "设计延迟": 3,
    "花费时间预估错误": 3,
    "没做好文档版本管理": 2
}

plt.rcParams['font.sans-serif'] = ['SimHei']


# 对错误数据按照发生次数进行排序
sorted_data = sorted(data.items(), key=lambda x: x[1], reverse=True)

# 计算累积百分比
total_errors = sum(data.values())
cumulative_percentage = 0
percentages = []
errors = []

for error, count in sorted_data:
    cumulative_percentage += count / total_errors
    percentages.append(cumulative_percentage * 100)
    errors.append(error)

# 绘制帕累托分析图
fig, ax1 = plt.subplots()

bar_color = 'steelblue'  # 条形图颜色
line_color = 'orange'  # 累积百分比曲线颜色

ax1.bar(range(len(errors)), [data[error] for error in errors], color=bar_color)
ax1.set_xlabel("错误")
ax1.set_ylabel("错误发生次数", color=bar_color)
ax1.tick_params('y', colors=bar_color)
ax1.set_xticks(range(len(errors)))
ax1.set_xticklabels(errors, rotation=45, ha='right')

ax2 = ax1.twinx()
ax2.plot(range(len(errors)), percentages, 'r-', marker='o', color=line_color)
ax2.set_ylabel("累积百分比", color=line_color)
ax2.tick_params('y', colors=line_color)

fig.tight_layout()
plt.title("帕累托分析图")
plt.show()
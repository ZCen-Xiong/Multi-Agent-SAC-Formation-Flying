import pandas as pd
import matplotlib.pyplot as plt

smooth_factor=50 #对列表中多少位数求平均，该值越大输出图像越平滑
# 读取第一个CSV文件
csv_file_path1 = 'policy loss.csv'  # 替换为第一个CSV文件的路径
df1 = pd.read_csv(csv_file_path1, header=None, skiprows=1)

# 提取第一个CSV文件的x和y轴数据
x_data1 = df1.iloc[1:, 1].astype(float)
y_data1 = df1.iloc[1:, 2].astype(float)
y_data2 = []
for i in range(len(y_data1)-1):
    if i==0:
        y_data2.append(y_data1.iloc[0])
    if i <=smooth_factor and i>0:
        print(sum(y_data1.iloc[0:i])/i)
        y_data2.append(sum(y_data1.iloc[0:i])/i)
    else:
        y_data2.append(sum(y_data1.iloc[i-smooth_factor:i])/smooth_factor)
# 读取第二个CSV文件
# csv_file_path2 = 'train_loss.csv'  # 替换为第二个CSV文件的路径
# df2 = pd.read_csv(csv_file_path2, header=None, skiprows=1)

# 提取第二个CSV文件的x和y轴数据
# x_data2 = df2.iloc[1:, 1].astype(float)
# y_data2 = df2.iloc[1:, 2].astype(float)

# 设置绘图风格，使用科学论文常见的线条样式和颜色
plt.style.use('seaborn-whitegrid')

# 设置字体和字号
font = {'family': 'serif',
         'serif': 'Times New Roman',
         'weight': 'normal',
         'size': 15,
         }
plt.rc('font', **font)

# 绘制第一幅图像
plt.figure(1)
plt.plot(x_data1, y_data1,color='blue', linewidth=0.5, label="loss")
plt.plot(x_data1, y_data2,'r',linewidth=1, label="smoothed loss")
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('policy loss-epochs')
plt.legend()
plt.tight_layout()
# 调整布局使得图像不溢出
plt.savefig('test_loss.svg', format='svg', bbox_inches='tight')

# 绘制第二幅图像
# plt.figure(2)
# plt.plot(x_data2, y_data2, color='red', linewidth=2)
# plt.xlabel('epochs')
# plt.ylabel('train_loss')
# plt.title('train_loss')

plt.tight_layout()
 # 调整布局使得图像不溢出
plt.savefig('test_reward.svg', format='svg', bbox_inches='tight')

 # 显示图形
plt.show()
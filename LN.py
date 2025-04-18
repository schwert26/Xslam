import numpy as np

# 读取文件内容，假设每行有8个数，以空格或其他分隔符分开
data = np.loadtxt('CameraTrajectory.txt')

# 打印原始第一行数据以便检查
print("修改前：", data[0])

# 将第一列（时间戳）除以 100000000
data[:, 0] = data[:, 0] / 1000000000

# 打印修改后第一行数据以验证更改
print("修改后：", data[0])

# 将修改后的数据保存到一个新的文件中
np.savetxt('CameraTrajectory_modified.txt', data, fmt='%.8f')

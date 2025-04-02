import json
import os

# 定义需要处理的文件路径
file_paths = [
    "./user_bundle_train.txt"
]

# 初始化一个空字典来存储每个捆绑包的交互次数
bundle_interaction_counts = {}

# 遍历每个文件路径
for file_path in file_paths:
    # 检查文件是否存在
    if os.path.exists(file_path):
        # 打开文件进行读取
        with open(file_path, 'r') as file:
            # 逐行读取文件内容
            for line in file:
                try:
                    # 假设每行格式为 "user_id\tbundle_id"，将其拆分为用户 ID 和捆绑包 ID
                    _, bundle_id = line.strip().split('\t')
                    # 将捆绑包 ID 转换为整数类型
                    bundle_id = int(bundle_id)
                    # 如果捆绑包 ID 已经在字典中，将其交互次数加 1
                    if bundle_id in bundle_interaction_counts:
                        bundle_interaction_counts[bundle_id] += 1
                    # 否则，将该捆绑包 ID 加入字典，并将其交互次数初始化为 1
                    else:
                        bundle_interaction_counts[bundle_id] = 1
                except ValueError:
                    # 如果解析过程中出现错误，打印错误信息
                    print(f"解析错误: {line} 在文件 {file_path} 中")

# 定义保存结果的文件路径
output_file = "bundle_interaction_counts.json"

# 将捆绑包交互次数字典保存为 JSON 文件
with open(output_file, 'w') as outfile:
    json.dump(bundle_interaction_counts, outfile)

# 打印保存结果的信息
print(f"捆绑包交互次数已保存到 {output_file}")

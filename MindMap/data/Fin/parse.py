# 读取json每一行，并写进.txt文件
import json
import os

# 读取JSON文件并转换为字典
def json_to_dict(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# 示例用法
file_path = '/root/autodl-tmp/MindMap/data/Fin/relationships.json'  # 替换为你的JSON文件路径
data_dict = json_to_dict(file_path)

lis = data_dict["relationships"]

for ele in lis:
    le = ele[0].replace(" ", "")
    r = ele[1].replace(" ", "")
    re = ele[2].replace(" ", "")

    content = f"{le} {r} {re}\n"

    with open('/root/autodl-tmp/MindMap/data/Fin/relationships.txt', 'a') as f:
        f.write(content)
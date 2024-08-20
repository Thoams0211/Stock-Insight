import ast
import re


with open("/root/autodl-tmp/Dataset/PiVeRes/yiwei.txt", 'r') as file:
    datas = file.readlines()

    tripLis = []

    for data in datas:
        data_str = data

        try:
            # 使用正则表达式查找所有子字符串
            pattern = re.compile(r'\[.*?\]')
            matches = pattern.findall(data_str)

            # 解析每个子字符串为 Python 列表
            parsed_data = [ast.literal_eval(item) for item in matches]

            # 打印解析后的结果
            for item in parsed_data:
                tripLis.append(item)
        except:
            continue

with open("/root/autodl-tmp/MindMap/data/Fin/relYiwei.txt", 'w') as file:
    for trip in tripLis:
        try:
            file.write(f"{trip[0]}##{trip[1]}##{trip[2]}" + "\n")
        except:
            print(trip)
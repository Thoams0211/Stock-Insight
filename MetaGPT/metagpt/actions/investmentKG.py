import asyncio
import requests
import json
import datetime
from metagpt.actions import Action
from metagpt.rag.engines import SimpleEngine
from metagpt.const import EXAMPLE_DATA_PATH
from metagpt.rag.schema import ChromaRetrieverConfig, BM25RetrieverConfig, FAISSRetrieverConfig
import pandas as pd
import matplotlib.dates as mdates
import socket


# DOC_PATH = EXAMPLE_DATA_PATH / "rag/travel.txt"
DOC_PATH = "/root/autodl-tmp/Dataset/ning.txt"
# SEARCH_METHOD = [ChromaRetrieverConfig(), BM25RetrieverConfig(), FAISSRetrieverConfig(dimensions=1536)]
SEARCH_METHOD = [BM25RetrieverConfig()]
EMBED_PATH = "local:/root/autodl-tmp/nomic-embed-text"
# SEARCH_METHOD = [BM25RetrieverConfig()]

# TODO: 修改检索方式，可以对比几种检索方式的效果


class KG_DrawPlot(Action):
    stockName: str = None

    def draw_plot(self, dataPath):
        import pandas as pd
        import matplotlib.pyplot as plt

        df = pd.read_csv(dataPath)
        df = df[0:7]
        # 设定日期列为索引，假设日期列的列名是 'Date'
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

        # 假设收盘价的列名是 'Close'
        plt.figure(figsize=(10, 6))  # 设置图形大小
        plt.plot(df.index, df['Close'], marker='o', linestyle='-', color='b', label='Close Price')  # 添加标记

        # 设置日期格式
        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())  # 自动根据图形尺寸选择日期间隔
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # 日期格式
        plt.xticks(rotation=60)  # 旋转90度以适应所有日期

        # 美化图表
        plt.title('Time Series of Close Price', fontsize=16, fontweight='bold')  # 设置图表标题
        plt.xlabel('Date', fontsize=14)  # 设置x轴标签
        plt.ylabel('Close Price', fontsize=14)  # 设置y轴标签
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # 显示网格
        plt.legend()  # 显示图例
        plt.tight_layout()  # 紧凑布局

        # 以时间戳为文件名保存图片
        timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        plotPath = f"/root/autodl-tmp/InvestReport/imgs/{timestamp}.png"
        plt.savefig(plotPath)

        return f"![img](imgs/{timestamp}.png)"

    async def run(self, stockName, dataPath):
        

        # 读取csv每一行的日期、收盘价，并保存在字典中
        df = pd.read_csv(dataPath)
        data_dict = {}

        for index, row in df.iterrows():
            data_dict[row["Date"]] = row["Close"]
        
        prompt = f"近7日，股票{stockName}收盘价如下："

        for key, value in data_dict.items():
            prompt += f"{key}, {value}元"

        prompt += """
请用一到两句话定性描述一下近七天该股票的价格走势。你的回答需满足以下要求:
1. 直接回复答案，不允许添加任何备注，也不需要引言。
2. 尽可能不要包含任何数字
"""

        # prompt += "\nPlease briefly describe the closing trend of this stock over the past 7 days in 1 to 2 sentences, try NOT to include any numbers. Reply in CHINESE. Directly reply with the answer, DO NOT reply with other prompt words."

        anl =  await self._aask(prompt)
        print("=" * 80)
        # anl = await _translate(anl, constraint=False)
        anl = self.draw_plot(dataPath) + "\n" + anl

        return anl


class KG_GoodNews(Action):
    stockName: str = None


    async def run(self, stockName):
        self.stockName = stockName


        question = f"{stockName}有哪些利好消息？请你回答的同时注明消息来源"

        # 创建 socket 对象
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # 获取本地主机名
        host = 'localhost'
        port = 9000

        # 连接到服务器
        client_socket.connect((host, port))

        # 发送数据到服务器
        message = question
        client_socket.send(message.encode('utf-8'))

        # 接收来自服务器的数据
        answer = client_socket.recv(1024).decode('utf-8')
        print(f"从服务器接收到的数据: {answer}")

        # 关闭连接
        client_socket.close()
        
        return answer



class KG_BadNews(Action):
    stockName: str = None


    async def run(self, stockName):
        self.stockName = stockName


        question = f"{stockName}有哪些不利消息？请你回答的同时注明消息来源"

        # 创建 socket 对象
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # 获取本地主机名
        host = 'localhost'
        port = 5000

        # 连接到服务器
        client_socket.connect((host, port))

        # 发送数据到服务器
        message = question
        client_socket.send(message.encode('utf-8'))

        # 接收来自服务器的数据
        answer = client_socket.recv(1024).decode('utf-8')
        print(f"从服务器接收到的数据: {answer}")

        # 关闭连接
        client_socket.close()
        
        return answer["kg_retrieval"]
    

class KG_MakeDecision(Action):
    stockName: str = None

    def loadDataset(self, dataPath):
        df = pd.read_csv(dataPath)
        return df

    async def run(self, stockName, content, dataPath):
        prices = self.loadDataset(f"{dataPath}")
        self.stockName = stockName

        prompt = f"""
现有关于{self.stockName}的咨询分析如下:
{content}
该股票近7日收盘价为:
{prices}
请你根据这些内容，确定是继续持有还是卖出该股票。你的回答应符合以下要求：
1. 直接回答你建议投资者的决策是 持有 或 卖出，不要回复任何其他内容！
"""
        anl = "\n综合以上信息，给出投资建议为："
        anl +=  await self._aask(prompt)
        return anl



# 使用类A
async def main(FUNCTION_NAME):
    a_instance = FUNCTION_NAME
    result = await a_instance.run("宁德时代")
    print(result)


if __name__ == "__main__":

    FUNC_LIST = [KG_GoodNews()]

    for FUNC in FUNC_LIST:
        # 运行异步主函数
        import asyncio
        asyncio.run(main(FUNC))
        
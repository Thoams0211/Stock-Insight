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


class DrawPlot(Action):
    sectorName: str = None

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

    async def run(self, sectorName, dataPath):
        

        # 读取csv每一行的日期、收盘价，并保存在字典中
        df = pd.read_csv(dataPath)
        data_dict = {}

        for index, row in df.iterrows():
            data_dict[row["Date"]] = row["Close"]
        
        prompt = f"近7日，股票{sectorName}收盘价如下："

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
    
class Review(Action):
    sectorName: str = None

    async def singleReview(self, stockName):
        self.stockName = stockName
        engine = SimpleEngine.from_docs(input_files=[DOC_PATH], retriever_configs=SEARCH_METHOD, embed_model=EMBED_PATH)
        
        prompt = f"""
今天是2024年7月7日，请总结一下近期对{self.stockName}影响较大的事件，以及对{self.stockName}的具体影响。最后，决定持有还是卖出该股票。你的回答格式应符合以下几点要求：
1. 请直！接！回复答案，不需要有任何备注，开头也不需要任何引入
2. 任何消息必！须！注！明！具体消息来源与url链接。
3. 输出一段文字，而不是逐条输出。
4. 用5-6句话完成整个回答，最后在决定持有还是卖出该股票时必！须！给出相应推理过程。

格式形如：
近期，宁德时代对网传的排产数据下滑传闻进行了积极回应，明确表示公司经营情况良好且全球市场份额稳步提升，
这有助于稳定投资者情绪。同时，公司宣布新产品神行和麒麟电池受到客户广泛认可并快速放量，这显示了公司在电
池技术上的创新实力。虽然市场波动导致公司股价一度下跌，但整体排产情况良好，且近期及三季度排产环比呈增长
态势，这表明公司具有抵御市场短期波动的能力。综合以上因素，我决定继续持有宁德时代股票，看好其长期发展潜力。

消息来源：
    1. ![南方财经网](http://finance.eastmoney.com/a/202407023119440882.html)
    2. ![蓝鲸财经](http://finance.eastmoney.com/a/202406273115868477.html)
"""
        
        answer = await engine.aquery(prompt)
        answer = str(answer)
        
        return answer


    async def run(self, stockLis):
        todos = (self.singleReview(stockName) for stockName in stockLis)
        result = await asyncio.gather(*todos)

        content = ""
        for i in range(len(result)):
            result[i] = f"## {stockLis[i]}\n" + result[i]
            content += result[i]

        return content



class KG_Review(Action):
    sectorName: str = None

    async def singleReview(self, stockName):
        self.stockName = stockName
        # engine = SimpleEngine.from_docs(input_files=[DOC_PATH], retriever_configs=SEARCH_METHOD, embed_model=EMBED_PATH)
        
        # 创建 socket 对象
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # 获取本地主机名
        host = 'localhost'
        port = 5000

        # 连接到服务器
        client_socket.connect((host, port))

        # 发送数据到服务器
        message = stockName
        client_socket.send(message.encode('utf-8'))

        # 接收来自服务器的数据
        answer = client_socket.recv(1024).decode('utf-8')
        print(f"从服务器接收到的数据: {answer}")

        # 关闭连接
        client_socket.close()
        
        return answer
    
    async def run(self, stockLis):
        todos = (self.singleReview(stockName) for stockName in stockLis)
        result = await asyncio.gather(*todos)

        content = ""
        for i in range(len(result)):
            result[i] = f"## {stockLis[i]}\n" + result[i]
            content += result[i]

        return content
        


# 使用类A
async def main(FUNCTION_NAME):

    lis = ["锂电池行业", "比亚迪", "多氟多", "孚能科技", "国轩高科", "欣旺达", "亿纬锂能"]

    a_instance = FUNCTION_NAME
    result = await a_instance.run(lis)
    print(result)


if __name__ == "__main__":

    FUNC_LIST = [KG_Review()]

    for FUNC in FUNC_LIST:
        # 运行异步主函数
        import asyncio
        asyncio.run(main(FUNC))
        
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


# DOC_PATH = EXAMPLE_DATA_PATH / "rag/travel.txt"
DOC_PATH = "/root/autodl-tmp/Dataset/ning.txt"
SEARCH_METHOD = [ChromaRetrieverConfig(), BM25RetrieverConfig(), FAISSRetrieverConfig(dimensions=1536)]
SEARCH_METHOD = [BM25RetrieverConfig()]
EMBED_PATH = "local:/root/autodl-tmp/nomic-embed-text"
# SEARCH_METHOD = [BM25RetrieverConfig()]

# TODO: 修改检索方式，可以对比几种检索方式的效果

# async def _translate(content, constraint=True):

#     def get_access_token():
#         """
#         使用 API Key，Secret Key 获取access_token，替换下列示例中的应用API Key、应用Secret Key
#         """
#         API_KEY = "0IW9TRIrlfFjd5e33OZLEhKg"
#         SECRET_KEY = "PaII3vc5b0wMRV336BHQ3lXj5VarUmrz"
#         url = f"https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={API_KEY}&client_secret={SECRET_KEY}"

#         payload = json.dumps("")
#         headers = {
#             'Content-Type': 'application/json',
#             'Accept': 'application/json'
#         }

#         response = requests.request("POST", url, headers=headers, data=payload)
#         return response.json().get("access_token")

#     url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/llama_3_70b?access_token=" + get_access_token()

#     prompt = "将以下内容翻译为中文:"

#     if constraint:
#         prompt = "将以下内容翻译为中文，并且删除其中不完整的句子："

#     payload = json.dumps({
#         "messages": [
#             {
#                 "role": "user",
#                 "content": prompt + content
#             }
#         ]
#     })
#     headers = {
#         'Content-Type': 'application/json'
#     }
#     response = requests.request("POST", url, headers=headers, data=payload)
#     # 将字符串转换为字典
#     data_dict = json.loads(response.text)
#     # 提取result的内容
#     result_content = data_dict.get("result")
#     # print(result_content)
#     return result_content


class DrawPlot(Action):
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

        return f"![img](http://localhost:5000/get_image?image_name={timestamp}.png)"

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


class GoodNews(Action):
    stockName: str = None


    async def run(self, stockName):
        self.stockName = stockName
        engine = SimpleEngine.from_docs(input_files=[DOC_PATH], retriever_configs=SEARCH_METHOD, embed_model=EMBED_PATH)
        
        prompt = f"""
今天是2024年7月7日，{self.stockName}最近有什么利好消息吗？你的回答格式应符合以下几点要求：
1. 请直！接！回复答案，不需要有任何备注，开头也不需要任何引入
2. 任何消息必须注！明！具体的消息来自哪个媒体，并将出处穿插到回答中，不要在最后同一注明。
3. 输出一段文字，而不是逐条输出。
4. 根据每一条消息推测出对其近期的影响。

<Example>
近期，宁德时代 ... ...

消息来源：
    1. [南方财经网](http://finance.eastmoney.com/a/202407023119440882.html)
    2. [蓝鲸财经](http://finance.eastmoney.com/a/202406273115868477.html)
<\Example>
"""
        
        answer = await engine.aquery(prompt)
        answer = str(answer)
        
        return answer



class BadNews(Action):
    stockName: str = None


    async def run(self, stockName):
        self.stockName = stockName
        engine = SimpleEngine.from_docs(input_files=[DOC_PATH], retriever_configs=SEARCH_METHOD, embed_model=EMBED_PATH)
        
        prompt = f"""
今天是2024年7月7日，{self.stockName}最近有什么不利消息吗？你的回答格式应符合以下几点要求：
1. 请直！接！回复答案，不需要有任何备注，开头也不需要任何引入
2. 任何消息必须注！明！具体的消息来自哪个媒体，并将出处穿插到回答中，不要在最后同一注明。
3. 输出一段文字，而不是逐条输出。
4. 根据每一条消息推测出对其近期的影响。
5. 只回答不利消息，不回复任何利好消息。

<Example>
近期，宁德时代 ... ...

消息来源：
    1. [南方财经网](http://finance.eastmoney.com/a/202407023119440882.html)
    2. [蓝鲸财经](http://finance.eastmoney.com/a/202406273115868477.html)
<\Example>
"""
        
        answer = await engine.aquery(prompt)
        answer = str(answer)
        
        return answer
    

class MakeDecision(Action):
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
    result = await a_instance.run("比亚迪", "/root/autodl-tmp/Dataset/byd.csv")
    print(result)


if __name__ == "__main__":

    FUNC_LIST = [DrawPlot()]

    for FUNC in FUNC_LIST:
        # 运行异步主函数
        import asyncio
        asyncio.run(main(FUNC))
        
import asyncio
import re

from pydantic import BaseModel
import datetime
from metagpt.actions import Action, battery
# from metagpt.actions.investment import DrawPlot, GoodNews, BadNews, MakeDecision
from metagpt.actions.battery import DrawPlot, Review
from metagpt.actions.research import get_research_system_text
from metagpt.const import RESEARCH_PATH
from metagpt.logs import logger
from metagpt.roles.role import Role, RoleReactMode
from metagpt.schema import Message


DATA_PATH = "/root/autodl-tmp/AFAC.csv"
ACTION_LIS = [DrawPlot, Review]    

class FinReport(BaseModel):
    theStockName: str
    theLatestNews: str = None
    theFinancialPerformance: str = None
    theTrendPrediction: str = None

class Sector(Role):

    name: str = "David"
    profile: str = "Investor"
    goal: str = "Gather information and conduct research"
    constraints: str = "Ensure accuracy and relevance of information"
    language: str = "zh-cn"
    enable_concurrency: bool = True

    _plotPart: str = ""
    _review: str = ""

    def __init__(
        self,
        stockNames: list = ["宁德时代", "比亚迪", "多氟多", "孚能科技", "国轩高科", "亿纬锂能", "欣旺达"],
        language: str = "en-us",
        **kwargs,
    ):
        super().__init__(**kwargs)

        # 添加 `CollectLinks`、`WebBrowseAndSummarize` 和 `ConductResearch` 动作
        # self.set_actions([DrawPlot, BadNews, GoodNews])
        self.set_actions(ACTION_LIS)
        self.stockNames = stockNames

        # 设置按顺序执行
        self._set_react_mode(react_mode="by_order")
        self.language = language
        if language not in ("en-us", "zh-cn"):
            logger.warning(f"The language `{language}` has not been tested, it may not work.")

    async def _act(self) -> Message:
        logger.info(f"{self._setting}: to do {self.rc.todo}({self.rc.todo.name})")
        todo = self.rc.todo
        msg = self.rc.memory.get(k=1)[0]
        if isinstance(msg.instruct_content, FinReport):
            instruct_content = msg.instruct_content
            stockName = instruct_content.theStockName
        else:
            stockNames = self.stockNames
        

        if isinstance(todo, DrawPlot):
            result = await todo.run(stockName, DATA_PATH)
            self._plotPart = result
            ret = Message(
                content=result, role=self.profile, cause_by=todo
            )

        elif isinstance(todo, Review):
            result = await todo.run(stockNames)
            self._review = result
            ret = Message(
                content=result, role=self.profile, cause_by=todo
            )

        else:
            raise ValueError(f"Unknown todo: {todo}")
        
        self.rc.memory.add(ret)
        return ret
    
    async def react(self) -> Message:
        msg = await super().react()
        report = msg.instruct_content
        self.write_report()
        return msg

    def write_report(self):
        # 获取当前时间戳并格式化为字符串（例如：20240708-150305）
        timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

        content = f"# {self.stockName} 投资报告\n\n"

        content += self._plotPart + self._badNews + self._goodNews + self._decision

        # 生成文件名，添加 .md 扩展名
        filename = f"{timestamp}.md"
        # 将文本写入 Markdown 文件
        with open(f'/root/autodl-tmp/InvestReport/{filename}', 'w', encoding='utf-8') as file:
            file.write(content)
        print("Write Markdown!")
    

if __name__ == "__main__":
    import fire

    async def main(language: str = "zh-cn", enable_concurrency: bool = True, theStockNames: str = ["宁德时代"]):
        role = Sector(language=language, enable_concurrency=enable_concurrency)
        # print("HERE!!")
        await role.run(theStockNames)

    fire.Fire(main)


import os
import openai



import time
import json
import functools
from tqdm import tqdm
import requests

text = []
with open('/root/autodl-tmp/Dataset/News/yiwei.txt', 'r') as f:
    for l in f.readlines():
        text.append(l.strip())

def get_access_token():
    """
    使用 API Key，Secret Key 获取access_token，替换下列示例中的应用API Key、应用Secret Key
    """
    API_KEY = "T6OdiNjHJkDGMyG1d23dQ5TN"
    SECRET_KEY = "iyQbYVfGSbWq4MTlR7Z9wydht4JrTbOw"
    url = f"https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={API_KEY}&client_secret={SECRET_KEY}"
            
    payload = json.dumps("")
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    return response.json().get("access_token")


@functools.lru_cache()
def get_chatgpt_completion(content):
    url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions_pro?access_token=" + get_access_token()
           
    payload = json.dumps({
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ]
    })
    headers = {
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    # 将字符串转换为字典
    data_dict = json.loads(response.text)

    # 提取result的内容
    result_content = data_dict.get("result")

    print(result_content)

    return result_content
    # return response["choices"][0][ "message"]["content"]


iteration1 = True

if iteration1:
    with open("/root/autodl-tmp/Dataset/News/yiwei.txt", "a") as output_file:
        for i in tqdm(range(len(text))):
            prompt = f"""
<Example>
    2012年，宁德时代与德国宝马集团达成战略合作
    语义图：
        ["宁德时代", "合作方", "德国宝马集团"]
        ["宁德时代", "合作时间", "2012年"]
<\Example>

文本：{text[i]}

<Question>
    请将以上文本转换为语义图。请注意，你回复的格式应符合以下要求：
    1. 直接回复答案，无需其他引言
    2. 如果没有语义图可以构建，直接回复0
    3. 必！须！严格遵照例子中的语义图格式
    4. 禁！止！在列表中嵌套列表
<\Question>
"""
            # prompt = '\n<Example> \n文本:Shotgate Thickets是位于英国的一个自然保护区，由埃塞克斯野生动物信托基金运营。\n语义图：[["Shotgate Thickets", "实例", "自然保护区"], ["Shotgate Thickets", "国家", "英国"], ["Shotgate Thickets", "运营者", "埃塞克斯野生动物信托基金"]]\n文本：' + text[i] + '\n 语义图：'
            response = get_chatgpt_completion(prompt)
            if response is None:
                continue
            output_file.write(response.strip().replace('\n','') + '\n')
		
else:
    missing_triples = []
    with open('GPT3.5_result_KELM_flan/Iteration3/verifier_result.txt', 'r') as f:
        for l in f.readlines():
            missing_triples.append(l.strip())

    results_old = []
    with open("GPT3.5_result_KELM_flan/Iteration3/test_generated_graphs.txt", 'r') as f:
        for line in f.readlines():
            results_old.append(line.strip())

    with open("GPT3.5_result_KELM_flan/Iteration4/test_generated_graphs.txt", "a") as output_file:
        for i in tqdm(range(len(text))):
            if missing_triples[i] != 'Correct':
                #WebNLG & GenWiki
                #prompt = "Transform the text into a semantic graph and also add the given triples to the generated semantic graph.\nExample:\nText: Sportpark De Toekomst is located in Ouder-Amstel, Netherlands. It is owned and operated by AFC Ajax N.V. and their tenants include the Ajax Youth Academy. \nTriples: [\"Sportpark De Toekomst\", \"country\", \"Netherlands\"], [\"Sportpark De Toekomst\", \"operator\", \"AFC Ajax N.V.\"]\nSemantic graph: [[\"Sportpark De Toekomst\", \"location\", \"Ouder-Amstel\"], [\"Sportpark De Toekomst\", \"country\", \"Netherlands\"], [\"Sportpark De Toekomst\", \"owner\", \"AFC Ajax N.V.\"], [\"Sportpark De Toekomst\", \"operator\", \"AFC Ajax N.V.\"], [\"Sportpark De Toekomst\", \"tenant\", \"Ajax Youth Academy\"]]\nText: " + text[i] + "\nTriples: " + missing_triples[i] + "\nSemantic graph:"
                #KELM
                prompt = "Transform the text into a semantic graph and also add the given triples to the generated semantic graph.\nExample:\nText: Shotgate Thickets is a nature reserve in the United Kingdom operated by the Essex Wildlife Trust. \nTriples: [\"Shotgate Thickets\", \"instance of\", \"Nature reserve\"], [\"Shotgate Thickets\", \"country\", \"United Kingdom\"]\nSemantic graph: [[\"Shotgate Thickets\", \"instance of\", \"Nature reserve\"], [\"Shotgate Thickets\", \"country\", \"United Kingdom\"], [\"Shotgate Thickets\", \"operator\", \"Essex Wildlife Trust\"]]\nText: " + text[i] + "\nTriples: " + missing_triples[i] + "\nSemantic graph:"
                response = get_chatgpt_completion(prompt)
                output_file.write(response.strip().replace('\n','') + '\n')
            else:
                output_file.write(results_old[i].strip() + '\n')


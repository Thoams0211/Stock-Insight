import time
import json
import functools
from tqdm import tqdm
import requests


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
def translate(content):
    url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions?access_token=" + get_access_token()
           
    payload = json.dumps({
        "messages": [
            {
                "role": "user",
                "content": "Translate these content to English and output your answer DIRECTLY: \n" + content
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


def transfer():

    res = []
    
    with open('/root/autodl-tmp/Dataset/ning.txt', 'r') as f:
        with open('/root/autodl-tmp/PiVe/prompt_scripts/GPT3.5_result_KELM/Iteration1/test_generated_graphs.txt', 'r') as j:
            
            chinese = f.readlines()
            english = j.readlines()

            for i in tqdm(range(len(chinese)), desc="Translating Process"):
                print(f"Translating {i}th sentence")
                instruct = chinese[i]
                instruct = translate(instruct)

                trip = english[i]
                trip = translate(trip)

                ele = {}
                ele['instruction'] = "Predict the missing triple given the text and graph for WebNLG dataset."
                ele['input'] = instruct + " <S> " + trip
                ele['output'] = "Correct"
                
                res.append(ele)

                # 将列表保存为JSON文件
                with open('/root/autodl-tmp/PiVe/unified_verifier/data.json', 'a', encoding='utf-8') as json_file:
                    json.dump(res, json_file, ensure_ascii=False, indent=4)

    

if __name__ == "__main__":
    transfer()
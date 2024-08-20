from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate,LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
import numpy as np
import re
import string
from neo4j import GraphDatabase, basic_auth
import pandas as pd
from collections import deque
import itertools
from typing import Dict, List
import pickle
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize 
import openai
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.meteor.meteor import Meteor
from langchain.llms import OpenAI
import os
from PIL import Image, ImageDraw, ImageFont
import csv
from gensim import corpora
from gensim.models import TfidfModel
from gensim.similarities import SparseMatrixSimilarity
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import sys
from time import sleep
import requests
from tqdm import tqdm
import ast

uri = "bolt://0.0.0.0:7687"
username = "neo4j"
password = "12345678"

re1 = r'The extracted entities are (.*?)<END>'
re2 = r"The extracted entity is (.*?)<END>"
re3 = r"<CLS>(.*?)<SEP>"

driver = GraphDatabase.driver(uri, auth=(username, password))
session = driver.session()


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



def chat_qianfan(prompt):
    url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions_pro?access_token=" + get_access_token()
           
    payload = json.dumps({
        "messages": [
            {
                "role": "user",
                "content": prompt
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
    return result_content


def chat_qianfan_mult(messages: list):
        
    url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions_pro?access_token=" + get_access_token()
            
    payload = json.dumps({
        "messages": messages
    })
    headers = {
        'Content-Type': 'application/json'
    }
    
    response = requests.request("POST", url, headers=headers, data=payload)
    
    # print(response.text)

    res = response.json().get("result")

    return res


def embeddingQianfan(inputEntitys: list) -> list:

    res = []

    for i in tqdm(range(len(inputEntitys)), desc="Embedding... ..."):

        inputEntity = inputEntitys[i]
        
        url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/embeddings/embedding-v1?access_token=" + get_access_token()
        
        payload = json.dumps({
            "input": [inputEntity]
        })
        headers = {
            'Content-Type': 'application/json'
        }
        
        response = requests.request("POST", url, headers=headers, data=payload)
        
        # 提取响应中的JSON部分
        response_json_str = response.text

        # 解析JSON字符串
        response_json = json.loads(response_json_str)

        # 提取data字段
        data = response_json.get("data")

        res.append(data)

    return res


def prompt_extract_keyword(input_text):
    template = """
    这里有一些例子:
    \n\n
    ### Instruction:\n'学习从以下金融问题中提取实体。'\n\n### Input:\n
    <CLS>你好，我想知道宁德时代今天发布了什么重要的利好消息？此外，关于公司未来一年的市场预期和投资策略有哪些建议？<SEP>提取的实体是\n\n ### Output:
    <CLS>你好，我想知道宁德时代今天发布了什么重要的利好消息？此外，关于公司未来一年的市场预期和投资策略有哪些建议？<SEP>提取的实体是 宁德时代，利好消息，市场预期，投资策略<EOS>
    \n\n
    ### Instruction:\n'学习从以下金融回答中提取实体。'\n\n### Input:\n
    <CLS>根据宁德时代今天发布的消息，公司计划在欧洲市场大幅扩展业务，预计未来两年内市场份额将达到40%。此外，公司还宣布与特斯拉达成新的供应协议，将为其提供最新的电池技术。<SEP>提取的实体是\n\n ### Output:
    <CLS>根据宁德时代今天发布的消息，公司计划在欧洲市场大幅扩展业务，预计未来两年内市场份额将达到40%。此外，公司还宣布与特斯拉达成新的供应协议，将为其提供最新的电池技术。<SEP>提取的实体是 宁德时代，欧洲市场，市场份额，特斯拉，供应协议，电池技术<EOS>
    \n\n
    ### Instruction:\n'学习从以下金融问题中提取实体。'\n\n### Input:\n
    <CLS>{input}<SEP>提取的实体是\n\n ### Output:
    """

    prompt = PromptTemplate(
        template = template,
        input_variables = ["input"]
    )

    system_message_prompt = SystemMessagePromptTemplate(prompt = prompt)
    system_message_prompt.format(input = input_text)

    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt,human_message_prompt])
    chat_prompt_with_values = chat_prompt.format_prompt(input = input_text,\
                                                        text={})

    response_of_KG = chat_qianfan(chat_prompt_with_values.to_messages()).content

    question_kg = re.findall(re1,response_of_KG)
    return question_kg



def find_shortest_path(start_entity_name, end_entity_name,candidate_list):
    global exist_entity
    with driver.session() as session:
        # print("SESSSION HERE")
        result = session.run(
            "MATCH (start_entity:Entity{name:$start_entity_name}), (end_entity:Entity{name:$end_entity_name}) "
            "MATCH p = allShortestPaths((start_entity)-[*..5]->(end_entity)) "
            "RETURN p",
            start_entity_name=start_entity_name,
            end_entity_name=end_entity_name
        )
        print(result.value)
        paths = []
        short_path = 0
        for record in result:
            path = record["p"]
            entities = []
            relations = []
            for i in range(len(path.nodes)):
                node = path.nodes[i]
                entity_name = node["name"]
                entities.append(entity_name)
                if i < len(path.relationships):
                    relationship = path.relationships[i]
                    relation_type = relationship.type
                    relations.append(relation_type)
           
            path_str = ""
            for i in range(len(entities)):
                entities[i] = entities[i].replace("_"," ")
                
                if entities[i] in candidate_list:
                    short_path = 1
                    exist_entity = entities[i]
                path_str += entities[i]
                if i < len(relations):
                    relations[i] = relations[i].replace("_"," ")
                    path_str += "->" + relations[i] + "->"
            
            if short_path == 1:
                paths = [path_str]
                break
            else:
                paths.append(path_str)
                exist_entity = {}
            
        if len(paths) > 5:        
            paths = sorted(paths, key=len)[:5]

        # print("999999999999999")

        return paths,exist_entity


def combine_lists(*lists):
    combinations = list(itertools.product(*lists))
    results = []
    for combination in combinations:
        new_combination = []
        for sublist in combination:
            if isinstance(sublist, list):
                new_combination += sublist
            else:
                new_combination.append(sublist)
        results.append(new_combination)
    return results


def get_entity_neighbors(entity_name: str,disease_flag) -> List[List[str]]:
    disease = []
    query = """
    MATCH (e:Entity)-[r]->(n)
    WHERE e.name = $entity_name
    RETURN type(r) AS relationship_type,
           collect(n.name) AS neighbor_entities
    """
    result = session.run(query, entity_name=entity_name)

    neighbor_list = []
    for record in result:
        rel_type = record["relationship_type"]
        
        if disease_flag == 1 and rel_type == 'has_symptom':
            continue

        neighbors = record["neighbor_entities"]
        
        if "disease" in rel_type.replace("_"," "):
            disease.extend(neighbors)

        else:
            neighbor_list.append([entity_name.replace("_"," "), rel_type.replace("_"," "), 
                                ','.join([x.replace("_"," ") for x in neighbors])
                                ])
    
    return neighbor_list,disease

def prompt_path_finding(path_input):
    template = """
    有一些知识图谱路径。它们遵循实体->关系->实体的格式。
    \n\n
    {Path}
    \n\n
    使用这些知识图谱信息。尝试将它们分别转换为自然语言。使用单引号标注实体名称和关系名称。并将它们命名为路径证据1，路径证据2，依此类推。\n\n

    输出：
    """

    prompt = PromptTemplate(
        template = template,
        input_variables = ["Path"]
    )

    system_message_prompt = SystemMessagePromptTemplate(prompt = prompt)
    system_message_prompt.format(Path = path_input)

    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt,human_message_prompt])
    chat_prompt_with_values = chat_prompt.format_prompt(Path = path_input,\
                                                        text={})
    
    print(chat_prompt_with_values.to_messages())
    print(type(chat_prompt_with_values.to_messages()))

    inputText = chat_prompt_with_values.to_messages()[0].content

    # response_of_KG_path = chat(chat_prompt_with_values.to_messages()).content
    response_of_KG_path = chat_qianfan(inputText)
    return response_of_KG_path

def prompt_neighbor(neighbor):
    template = """
    有一些知识图谱。它们遵循实体->关系->实体列表的格式。
    \n\n
    {neighbor}
    \n\n
    使用这些知识图谱信息。尝试将它们分别转换为自然语言。使用单引号标注实体名称和关系名称。并将它们命名为 基于邻接点的证据1，基于邻接点的证据2，依此类推。\n\n

    输出
    """

    prompt = PromptTemplate(
        template = template,
        input_variables = ["neighbor"]
    )

    system_message_prompt = SystemMessagePromptTemplate(prompt = prompt)
    system_message_prompt.format(neighbor = neighbor)

    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt,human_message_prompt])
    chat_prompt_with_values = chat_prompt.format_prompt(neighbor = neighbor,\
                                                        text={})

    print(f"chat_prompt: {chat_prompt_with_values}")
    print(f"chat_prompt_with_values.to_messages(): {chat_prompt_with_values.to_messages()[0]}")
    print(f"chat_prompt_with_values.to_messages().content: {chat_prompt_with_values.to_messages()[0].content}")
    response_of_KG_neighbor = chat_qianfan(chat_prompt_with_values.to_messages()[0].content)

    return response_of_KG_neighbor

def cosine_similarity_manual(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

def is_unable_to_answer(response):
 
    analysis = openai.Completion.create(
    engine="text-davinci-002",
    prompt=response,
    max_tokens=1,
    temperature=0.0,
    n=1,
    stop=None,
    presence_penalty=0.0,
    frequency_penalty=0.0
)
    score = analysis.choices[0].text.strip().replace("'", "").replace(".", "")
    if not score.isdigit():   
        return True
    threshold = 0.6
    if float(score) > threshold:
        return False
    else:
        return True


def autowrap_text(text, font, max_width):

    text_lines = []
    if font.getsize(text)[0] <= max_width:
        text_lines.append(text)
    else:
        words = text.split(' ')
        i = 0
        while i < len(words):
            line = ''
            while i < len(words) and font.getsize(line + words[i])[0] <= max_width:
                line = line + words[i] + ' '
                i += 1
            if not line:
                line = words[i]
                i += 1
            text_lines.append(line)
    return text_lines

def final_answer(input_text,response_of_KG_list_path,response_of_KG_neighbor):    

    print(response_of_KG_neighbor)
    print(response_of_KG_list_path)

    if len(response_of_KG_list_path) == 0:
        assisContent = '###'+ response_of_KG_neighbor + '\n\n'
    else:
        assisContent = '###'+ response_of_KG_neighbor + '\n\n' + '###' 
        for resp in response_of_KG_list_path:
            assisContent += resp + '\n\n'

    print(f"ASSIS CONTENT IS: \n", assisContent)

    messages = [
            {
                "role": "user",
                "content": "你是一位优秀的市场分析师，可以根据金融知识信息回答投资者问题。你有以下内容可以参考："
            },
            {
                "role": "assistant",
                "content": assisContent
            },
            {
                "role": "user",
                "content": "根据你提供的信息，请你回答这个行业可能面临什么情况？应该进行哪些分析来确认预测？\n\n\n"
                # + "Output1: 显示推理过程，将提取哪些知识来源于哪个路径证据或邻接点证据，并最终推理出结果。请将推理过程转换为以下格式：\n 路径证据编号('实体名称'->'关系名称'->...)->路径证据编号('实体名称'->'关系名称'->...)->邻接点证据编号('实体名称'->'关系名称'->...)->邻接点证据编号('实体名称'->'关系名称'->...)->结果编号('实体名称')->路径证据编号('实体名称'->'关系名称'->...)->邻接点证据编号('实体名称'->'关系名称'->...). \n\n"
                + "Output1: 请你将所提供的信息转化为一段自然语言，将全部信息转述给一个新用户.\n\n"
                + "Output2: 显示推理过程，将提取哪些知识来源于哪个路径证据或邻接点证据，并最终推理出结果。请将推理过程转换为以下格式：\n 路径证据编号('实体名称'->'关系名称'->...)->路径证据编号('实体名称'->'关系名称'->...)->邻接点证据编号('实体名称'->'关系名称'->...)->邻接点证据编号('实体名称'->'关系名称'->...)->结果编号('实体名称')->路径证据编号('实体名称'->'关系名称'->...)->邻接点证据编号('实体名称'->'关系名称'->...). \n\n"
                + "Output3: 绘制一棵决策树。推理过程中单引号中的实体或关系作为节点，后面跟随证据来源（实体名称），加入决策树。\n\n"
                + "以下是一个示例，你的回答中不可以包含以下示例中任何具体内容：\n"
                + """
                
                Output 1:
                路径证据1('公司'->'发布'->'利好消息')->路径证据2('利好消息'->'可能导致'->'股票价格上涨')->邻接点证据1('技术分析'->'包含'->'移动平均线分析')->邻接点证据2('技术分析'->'包含'->'相对强弱指数分析')->结果1('股票价格上涨')->路径证据3('公司'->'发布'->'财报')->邻接点证据3('市场动态'->'可能影响'->'股票价格').

                Output 2:
                根据当前的市场状况和公司发布的消息，这只股票可能面临短期内价格上涨的情况。为了确认这一预测，建议进行技术分析，包括移动平均线和相对强弱指数（RSI）分析。此外，应该关注公司的财报和市场动态。推荐的投资策略包括在突破关键阻力位时买入，并在短期内持有以获取快速收益。同时，建议设置止损位以控制风险。

                Output 3:
                公司(路径证据1)
                └── 发布(路径证据1)
                    └── 利好消息(路径证据1)(路径证据2)
                        └── 可能导致(路径证据2)
                            └── 股票价格上涨(路径证据2)(邻接点证据1)
                                ├── 包含(邻接点证据1)
                                │   └── 移动平均线分析(邻接点证据1)(邻接点证据2)
                                │       └── 包含(邻接点证据2)
                                │           └── 相对强弱指数分析(邻接点证据2)(结果1)(路径证据3)
                                ├── 发布(路径证据3)
                                │   └── 财报(路径证据3)(邻接点证据3)
                                └── 可能影响(邻接点证据3)
                                    └── 市场动态(邻接点证据3)
                """
            }
    ]
    
    
    result = chat_qianfan_mult(messages)
    
    return result

def prompt_document(question,instruction):
    template = """
    你是一位优秀的证券分析师，可以根据金融知识信息回答投资者问题。

    投资者输入：
    {question}

    你可以参考以下相关金融资讯：
    {instruction}

    你的回答格式应符合以下几点要求：
    1. 请直！接！回复答案，不需要有任何备注，开头也不需要任何引入
    2. 任何消息必须注！明！具体的消息来自哪个媒体，并将出处穿插到回答中，不要在最后同一注明。
    3. 输出一段文字，而不是逐条输出。
    4. 根据每一条消息推测出对其近期的影响。

    """

    prompt = PromptTemplate(
        template = template,
        input_variables = ["question","instruction"]
    )

    system_message_prompt = SystemMessagePromptTemplate(prompt = prompt)
    system_message_prompt.format(question = question,
                                 instruction = instruction)

    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt,human_message_prompt])
    chat_prompt_with_values = chat_prompt.format_prompt(question = question,\
                                                        instruction = instruction,\
                                                        text={})

    # print(chat_prompt_with_values.to_messages())

    response_document_bm25 = chat_qianfan(chat_prompt_with_values.to_messages()[0].content)

    return response_document_bm25

def sanitize_relationship_type(rel_type):
        # 将关系类型转换为有效的标识符，例如用下划线替换空格和其他字符
        return "REL_" + ''.join(['_' if not c.isalnum() else c for c in rel_type])


def graphEntityEmbedding(driver) -> list:
    # 查询所有实体（节点）的名称
    def get_all_entities(tx):
    # 查询所有实体（节点）的名称
        query = "MATCH (n) RETURN DISTINCT labels(n) AS labels, n.name AS name"
        result = tx.run(query)
        entities = []
        for record in result:
            # 检查是否有标签和名称属性
            labels = record["labels"]
            name = record["name"]
            if labels and name:
                for label in labels:
                    entities.append((label, name))
        return entities
    
    with driver.session() as session:
        entities = session.read_transaction(get_all_entities)
        res = []
        for label, name in entities:
            res.append(name)

    output = embeddingQianfan(res)

    return output, res
        

def graphRag(question, textPath):
    # 1. build neo4j knowledge graph datasets

    driver = GraphDatabase.driver(uri, auth=(username, password))
    session = driver.session()
    # session.run("MATCH (n) DETACH DELETE n")# clean all

    # read triples
    with open(textPath, 'r') as file:
        lines = file.readlines()
        # for line in lines[:100]:
        for line in lines:
            lis = line.strip().split('##')
            if len(lis) != 3:
                continue
            head_name = lis[0]
            relation_name = sanitize_relationship_type(lis[1])
            tail_name = lis[2]

            query = (
                "MERGE (h:Entity { name: $head_name }) "
                "MERGE (t:Entity { name: $tail_name }) "
                "MERGE (h)-[r:" + relation_name + "]->(t)"
            )
            # print(f"head_name: {head_name}, tail_name: {tail_name}, relation_name: {relation_name}")
            session.run(query, head_name=head_name, tail_name=tail_name, relation_name=relation_name)
            
    #   # <------------------------------------- build KG ---------------------------------------------->

    # raise ValueError("Stop")

    # 2. OpenAI API based keyword extraction and match entities
    print("HERE!!")
    

    # OPENAI_API_KEY = YOUR_OPENAI_KEY
    # chat = ChatOpenAI(openai_api_key=OPENAI_API_KEY)

    re1 = r'The extracted entities are (.*?)<END>'
    re2 = r"The extracted entity is (.*?)<END>"
    re3 = r"<CLS>(.*?)<SEP>"

    with open('output.csv', 'w', newline='') as f4:
        writer = csv.writer(f4)
        writer.writerow(['Question', 'Label', 'MindMap','GPT3.5','BM25_retrieval','Embedding_retrieval','KG_retrieval','GPT4'])

    input_text = question
    
    if input_text == []:
        raise ValueError("Input Empty")
    print('Question:\n',input_text)


    qEntityPrompt = f"""
请你从以下句子中提取全部实体，直接回复答案，每个实体之间以##分隔。

<Example>
输入：宁德时代将于6月与比亚迪合作。
答案：宁德时代##6月##比亚迪。
<\Example>

你的输出应符合以下要求：
1. 全部用中文回答
2. 直！接！回复答案，无需任何引导词

输入：{input_text}
"""
    
    print(input_text)
    qEntity = chat_qianfan(qEntityPrompt)
    print(qEntity)

    question_kg = input_text.split('##')
    
    match_kg = []
    kg_entity_embd, kg_entities = graphEntityEmbedding(driver)

    for q_entity in question_kg:

        ques_entity_embd = embeddingQianfan([q_entity])[0][0]['embedding']

        max_sim = -1
        for i in range(len(kg_entity_embd)):
            kg_emb = kg_entity_embd[i][0]['embedding']
            kg_entity = kg_entities[i]

            cos_similarities = cosine_similarity_manual(kg_emb, ques_entity_embd)
            if cos_similarities > max_sim and kg_entity not in match_kg:
                max_sim = cos_similarities
                match_kg.append(kg_entity)
    
    print('match_kg',match_kg)

    # raise ValueError("Stop HERE")

    # # 4. neo4j knowledge graph path finding
    print("HERE!!!!!!!")
    if len(match_kg) != 1 or 0:
        start_entity = match_kg[0]
        candidate_entity = match_kg[1:]
        
        result_path_list = []

        print(f"start: {start_entity}, candidate: {candidate_entity}")
        print("We cant find Path")
        
        while 1:
            flag = 0
            paths_list = []
            flag_kg = True
            while candidate_entity != []:
                end_entity = candidate_entity[0]
                candidate_entity.remove(end_entity)
                try:
                    paths,exist_entity = find_shortest_path(start_entity, end_entity,candidate_entity)
                except:
                    print(f"start: {start_entity}, end: {end_entity}, candidate: {candidate_entity}")
                    print("We cant find Path")
                    flag_kg = False
                    break

                path_list = []
                if paths == [''] or paths == []:
                    flag = 1
                    if candidate_entity == []:
                        flag = 0
                        break
                    start_entity = candidate_entity[0]
                    candidate_entity.remove(start_entity)
                    break
                else:
                    for p in paths:
                        path_list.append(p.split('->'))
                    if path_list != []:
                        paths_list.append(path_list)
                
                if exist_entity != {}:
                    try:
                        candidate_entity.remove(exist_entity)
                    except:
                        continue
                start_entity = end_entity
            if flag_kg is False:
                # raise ValueError("Graph RAG failed")
                print("Graph RAG failed")
                break
            result_path = combine_lists(*paths_list)

            # print("9999999999999")
        
        
            if result_path != []:
                result_path_list.extend(result_path)                
            if flag == 1:
                continue
            else:
                break
            
        start_tmp = []
        for path_new in result_path_list:
        
            if path_new == []:
                continue
            if path_new[0] not in start_tmp:
                start_tmp.append(path_new[0])
        
        if len(start_tmp) == 0:
                result_path = {}
                single_path = {}
        else:
            if len(start_tmp) == 1:
                result_path = result_path_list[:5]
            else:
                result_path = []
                                            
                if len(start_tmp) >= 5:
                    for path_new in result_path_list:
                        if path_new == []:
                            continue
                        if path_new[0] in start_tmp:
                            result_path.append(path_new)
                            start_tmp.remove(path_new[0])
                        if len(result_path) == 5:
                            break
                else:
                    count = 5 // len(start_tmp)
                    remind = 5 % len(start_tmp)
                    count_tmp = 0
                    for path_new in result_path_list:
                        if len(result_path) < 5:
                            if path_new == []:
                                continue
                            if path_new[0] in start_tmp:
                                if count_tmp < count:
                                    result_path.append(path_new)
                                    count_tmp += 1
                                else:
                                    start_tmp.remove(path_new[0])
                                    count_tmp = 0
                                    if path_new[0] in start_tmp:
                                        result_path.append(path_new)
                                        count_tmp += 1

                                if len(start_tmp) == 1:
                                    count = count + remind
                        else:
                            break

            try:
                single_path = result_path_list[0]
            except:
                single_path = result_path_list
            
    else:
        result_path = {}
        single_path = {}            
    # print('result_path',result_path)
    
    
    # print("1234567")

    # # 5. neo4j knowledge graph neighbor entities
    neighbor_list = []
    neighbor_list_disease = []
    for match_entity in match_kg:
        disease_flag = 0
        neighbors,disease = get_entity_neighbors(match_entity,disease_flag)
        neighbor_list.extend(neighbors)

        while disease != []:
            new_disease = []
            for disease_tmp in disease:
                if disease_tmp in match_kg:
                    new_disease.append(disease_tmp)

            if len(new_disease) != 0:
                for disease_entity in new_disease:
                    disease_flag = 1
                    neighbors,disease = get_entity_neighbors(disease_entity,disease_flag)
                    neighbor_list_disease.extend(neighbors)
            else:
                for disease_entity in disease:
                    disease_flag = 1
                    neighbors,disease = get_entity_neighbors(disease_entity,disease_flag)
                    neighbor_list_disease.extend(neighbors)
    if len(neighbor_list)<=5:
        neighbor_list.extend(neighbor_list_disease)

    # print("neighbor_list",neighbor_list)


    # 6. knowledge gragh path based prompt generation
    if len(match_kg) != 1 or 0:
        response_of_KG_list_path = []
        if result_path == {}:
            response_of_KG_list_path = []
        else:
            result_new_path = []
            for total_path_i in result_path:
                path_input = "->".join(total_path_i)
                result_new_path.append(path_input)
            
            path = "\n".join(result_new_path)
            response_of_KG_list_path = prompt_path_finding(path)
            # if is_unable_to_answer(response_of_KG_list_path):
            #     response_of_KG_list_path = prompt_path_finding(path)
            print("response_of_KG_list_path",response_of_KG_list_path)
    else:
        response_of_KG_list_path = '{}'

    response_single_path = prompt_path_finding(single_path)
    
    # if is_unable_to_answer(response_single_path):
    #     response_single_path = prompt_path_finding(single_path)

    # # 7. knowledge gragh neighbor entities based prompt generation   
    response_of_KG_list_neighbor = []
    neighbor_new_list = []
    for neighbor_i in neighbor_list:
        neighbor = "->".join(neighbor_i)
        neighbor_new_list.append(neighbor)

    if len(neighbor_new_list) > 5:

        neighbor_input = "\n".join(neighbor_new_list[:5])
    response_of_KG_neighbor = prompt_neighbor(neighbor_input)
    # if is_unable_to_answer(response_of_KG_neighbor):
    #     response_of_KG_neighbor = prompt_neighbor(neighbor_input)
    # print("response_of_KG_neighbor",response_of_KG_neighbor)


    # # 8. prompt-based medical diaglogue answer generation
    output_all = final_answer(input_text,response_of_KG_list_path,response_of_KG_neighbor)
    
    print('\nMindMap:\n',output_all)

    
    ## 9. Experiment 1: chatgpt
    try:
        chatgpt_result = chat_qianfan(str(input_text))
    except:
        sleep(40)
        chatgpt_result = chat_qianfan(str(input_text))
    print('\nGPT-3.5:',chatgpt_result)
    
    ### 10. Experiment 2: document retrieval + bm25
    document_dir = "./data/chatdoctor5k/document"
    document_paths = [os.path.join(document_dir, f) for f in os.listdir(document_dir)]

    corpus = []
    for path in document_paths:
        with open(path, "r", encoding="utf-8") as f:
            corpus.append(f.read().lower().split())

    dictionary = corpora.Dictionary(corpus)
    bm25_model = BM25Okapi(corpus)

    bm25_corpus = [bm25_model.get_scores(doc) for doc in corpus]
    bm25_index = SparseMatrixSimilarity(bm25_corpus, num_features=len(dictionary))

    query = input_text
    query_tokens = query.lower().split()
    tfidf_model = TfidfModel(dictionary=dictionary, smartirs='bnn')
    tfidf_query = tfidf_model[dictionary.doc2bow(query_tokens)]
    best_document_index, best_similarity = 0, 0  

    bm25_scores = bm25_index[tfidf_query]
    for i, score in enumerate(bm25_scores):
        if score > best_similarity:
            best_similarity = score
            best_document_index = i

    with open(document_paths[best_document_index], "r", encoding="utf-8") as f:
        best_document_content = f.read()

    document_bm25_result = prompt_document(input_text,best_document_content)
    # if is_unable_to_answer(document_bm25_result):
    #     document_bm25_result = prompt_document(input_text,best_document_content)
    
    print('\nBM25_retrieval:\n',document_bm25_result)

    ### 11. Experiment 3: document + embedding retrieval
    # model = Word2Vec.load("./data/chatdoctor5k/word2vec.model")
    # ques_vec = np.mean([model.wv[token] for token in input_text.split()], axis=0)
    # similarities = []
    # for doc in docs:
    #     doc_vec = np.mean([model.wv[token] for token in doc.split()], axis=0)
    #     similarity = cosine_similarity([ques_vec], [doc_vec])[0][0]
    #     similarities.append(similarity)

    # max_index = np.argmax(similarities)
    # most_similar_doc = docs[max_index]
    
    # document_embedding_result = prompt_document(input_text[0],most_similar_doc)
    # if is_unable_to_answer(document_embedding_result):
    #     document_embedding_result = prompt_document(input_text[0],most_similar_doc)
    # print('\nEmbedding retrieval:\n',document_embedding_result)

    # ### 12. Experiment 4: kg retrieval
    kg_retrieval = prompt_document(input_text,response_single_path)
    # if is_unable_to_answer(kg_retrieval):
    #     kg_retrieval = prompt_document(input_text[0],response_single_path)
    print('\nKG_retrieval:\n',kg_retrieval)


    # ### 13. Experimet 5: gpt4
    # try:
    #     gpt4_result = chat_4(str(input_text[0]))
    # except:
    #     gpt4_result = chat_4(str(input_text[0]))
    # print('\nGPT4:\n',gpt4_result)

    
    # ### save the final result
    # with open('output.csv', 'a+', newline='') as f6:
    #     writer = csv.writer(f6)
    #     writer.writerow([input_text, chatgpt_result,document_bm25_result,kg_retrieval,])
    #     f6.flush()

    # return [input_text, chatgpt_result,document_bm25_result,kg_retrieval,]
    return{
        "input_text": input_text,
        "chatgpt_result": chatgpt_result,
        "document_bm25_result": document_bm25_result,
        "kg_retrieval": kg_retrieval,
    }
                


if __name__ == "__main__":
    res = graphRag("锂电池行业有什么资讯吗", textPath="/root/autodl-tmp/MindMap/data/Fin/relBattery.txt")
    # res = graphRag("今天是2024年7月7日，请总结一下近期对锂电池行业影响较大的事件，以及对锂电池行业的具体影响", textPath="/root/autodl-tmp/MindMap/data/Fin/relBattery.txt")

    print("=" * 80)
    for r in res:
        print(r)
        print("=" * 80)
    raise ValueError ("Stop")
               
# Stock-Insight :chart_with_upwards_trend:

## 0 Author :mailbox:
You can contact me by sending email to sy20021134@gmail.com

## 1 Intro :clipboard:
This project is a stock market analysis tool that uses Agent to generate report and predict stock prices. The implementation details and detailed principles of this project can be found in the `docs` folder and the code files are located in other folders.



The original task derives from [Aliyun-AFAC Competition](https://tianchi.aliyun.com/competition/entrance/532200/information) .


## 2 Structure :file_folder:
- `PiVe` : The main code of the model used to correct missing triples in the semantic graph.
- `MindMap` : The main code of the model used to implement KG_ARG.
- `MetaGPT` : The main code of the model used to construct Agent which generates stock market analysising report.
- `UI` : The main code of the UI of the project.

## 3 Results :computer:

- The implementation accurately identifies triples in textual information.
- It can leverage Neo4J's knowledge graph for KG-RAG.
- It can generate stock and industry reports in image + text format.
- Users can control the behavior of the agent through the frontend.

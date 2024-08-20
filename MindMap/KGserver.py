import MindMap
import socket

from MindMap import graphRag


stockDic = {
    "宁德时代": "/root/autodl-tmp/MindMap/data/Fin/relNing.txt",
    "锂电池行业": "/root/autodl-tmp/MindMap/data/Fin/relBattery.txt",
    "比亚迪": "/root/autodl-tmp/MindMap/data/Fin/relByd.txt",
    "多氟多": "/root/autodl-tmp/MindMap/data/Fin/relDuofuduo.txt",
    "孚能科技": "/root/autodl-tmp/MindMap/data/Fin/relFuneng.txt",
    "国轩高科": "/root/autodl-tmp/MindMap/data/Fin/relGuoxuan.txt",
    "欣旺达": "/root/autodl-tmp/MindMap/data/Fin/relXinwangda.txt",
    "亿纬锂能" : "/root/autodl-tmp/MindMap/data/Fin/relYiwei.txt"
}

def start_server():
    # 创建 socket 对象
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # 获取本地主机名
    host = 'localhost'
    port = 5000

    # 绑定端口
    server_socket.bind((host, port))

    # 设置最大连接数，超过后排队
    server_socket.listen(5)

    print(f"服务器启动，在 {host}:{port} 监听...")

    while True:
        # 建立客户端连接
        client_socket, addr = server_socket.accept()
        print(f"连接地址: {addr}")

        # 接收客户端数据
        stockName = client_socket.recv(1024).decode('utf-8')

        if stockName == "锂电池行业":
            question = f"""
今天是2024年7月7日，请总结一下近期{stockName}发展情况。
"""

        else: 
            question = f"""
    今天是2024年7月7日，请总结一下近期对{stockName}影响较大的事件，以及对{stockName}的具体影响。最后，决定持有还是卖出该股票。
    """
        
        reply = graphRag(question, stockDic[stockName])["kg_retrieval"]
        client_socket.send(reply.encode('utf-8'))

        # 关闭连接
        client_socket.close()

if __name__ == "__main__":
    start_server()
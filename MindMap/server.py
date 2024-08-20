import MindMap
import socket

from MindMap import graphRag

def start_server():
    # 创建 socket 对象
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # 获取本地主机名
    host = 'localhost'
    port = 9000

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
        data = client_socket.recv(1024).decode('utf-8')
        
        question = data
        reply = graphRag(question, "/root/autodl-tmp/MindMap/data/Fin/relNing.txt")["kg_retrieval"]
        client_socket.send(reply.encode('utf-8'))

        # 关闭连接
        client_socket.close()

if __name__ == "__main__":
    start_server()
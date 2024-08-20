from flask import Flask, send_file, request, abort, send_from_directory
import os

app = Flask(__name__)

# 配置图片存储路径
IMAGE_DIRECTORY = "/root/autodl-tmp/InvestReport/imgs"

@app.route('/get_image', methods=['GET'])
def get_image():
    # 获取请求中的图片路径参数
    image_name = request.args.get('image_name')
    
    if not image_name:
        return "Missing image_name parameter", 400
    
    image_path = os.path.join(IMAGE_DIRECTORY, image_name)
    
    # 检查图片文件是否存在
    if not os.path.exists(image_path):
        return "Image not found", 404
    
    # 返回图片文件
    try:
        return send_file(image_path, mimetype='image/jpeg')
    except Exception as e:
        return str(e), 500

if __name__ == '__main__':
    # 确保图片目录存在
    if not os.path.exists(IMAGE_DIRECTORY):
        os.makedirs(IMAGE_DIRECTORY)
    
    app.run(host='0.0.0.0', port=5000, debug=True)

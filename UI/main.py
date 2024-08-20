import gradio as gr
import time
import fire
import sys
# 假设你的自定义模块位于`/path/to/your/module`目录
module_path = ".."

# 将模块路径添加到sys.path中
if module_path not in sys.path:
    sys.path.append(module_path)

# 现在可以尝试导入你的模块
try:
    from MetaGPT.metagpt.roles.investor import Investor
    from MetaGPT.metagpt.roles.sector import Sector  
except ImportError as e:
    print(f"无法导入模块: {e}")



def process_input(report_type, name, use_KG=False, use_pive=False):
    # 处理输入的逻辑
    # report_title = "行业研报" if report_type == "行业研报" else "个股研报"
    # result_markdown = f"### {report_title}\n\n**名称**: {name}\n\n**选择的方法**: {method}\n\n**图谱构建方法**: {'使用PiVE模型' if use_pive else '不使用PiVE模型'}"

    # kwargs = {"report_type": report_type, "name": name, "method": method, "use_pive": use_pive}

    
    

    async def main(theStockName: str = name, language: str = "zh-cn", enable_concurrency: bool = True, use_KG: bool = use_KG, use_PiVe: bool = use_pive):
        if report_type == "行业研报":
            Gr_role = Sector
        elif report_type == "个股研报":
            Gr_role = Investor
        else: 
            raise ValueError("Wrong report type")
        role = Gr_role(language=language, enable_concurrency=enable_concurrency, ifKG=use_KG)
        markdownPath = await role.run(theStockName)

        return markdownPath

    res = fire.Fire(main)

    return res 


def generate_report(report_type, name, method, use_pive):
    md_file_path = process_input(report_type, name, method, use_pive)
    
    output_text = ""
    # 逐字读取文件内容并显示
    with open(md_file_path, "r", encoding="utf-8") as file:
        content = file.read()
        for char in content:
            output_text += char
            yield output_text
            time.sleep(0.05)  # 调整这个值以控制显示速度

# 创建 Gradio 界面
with gr.Blocks(css="style.css") as demo:
    gr.Markdown("## 研报生成器")

    with gr.Row():
        report_type = gr.Radio(choices=["行业研报", "个股研报"], label="选择研报类型")
    
    with gr.Row():
        name_input = gr.Textbox(lines=1, placeholder="请输入行业名称或公司名称...", label="行业名称/公司名称")
    
    with gr.Row():
        method_choice = gr.Radio(choices=["KG-RAG", "RAG"], label="方法选择", value="RAG")
    
    with gr.Row():
        pive_choice = gr.Radio(choices=["使用PiVE模型", "不使用PiVE模型"], label="图谱构建方法选择", value="不使用PiVE模型")
    
    output = gr.Markdown(visible=True, elem_classes=["output-markdown"])

    submit_button = gr.Button("提交")
    submit_button.click(generate_report, inputs=[report_type, name_input, method_choice, pive_choice], outputs=output)

# 启动界面
demo.launch()

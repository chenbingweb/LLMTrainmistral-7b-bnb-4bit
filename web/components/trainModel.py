import gradio as gr
import json
import time
import os
MODEL_PATH = ''
with open('config.json', 'r') as file:
    data = json.load(file)
    MODEL_PATH=data['model_path']
class LoadModel:
    def __init__(self):
        self.mode_list_dict = []
        self.loadModelList()
        self.init()
    #获取模型地址和模型名称
    def loadModelList(self):
        if not os.path.isdir(MODEL_PATH):
            gr.Warning('基础模型文件夹不存在，请配置基础模型文件夹')
            return
        for item in os.listdir(MODEL_PATH):
            item_path = os.path.join(MODEL_PATH, item)
            if(os.path.isdir(item_path)):
                self.mode_list_dict.append({
                    "name":item,
                    "full_path":item_path
                })
        print(self.mode_list_dict)
    def init(self):
        selectList = [item['name'] for item in self.mode_list_dict]
        selectModel = gr.Dropdown(selectList, label="基础模型", info="选择基础模型，作为微调的基础模型(预训练模型)")
        selectModel.select(inputs=selectModel)
        gr.Interface(self.selectModelCall,[selectModel],None,allow_flagging='never')
    def selectModelCall(self, val,progress=gr.Progress()):
        progress(0, desc="Starting...")
        time.sleep(1)
        for i in progress.tqdm(range(100)):
            time.sleep(0.1)
        print(val)
        return val





# LoadModel()
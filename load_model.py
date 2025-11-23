from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
from PIL import Image
import requests
import os

def load_smolvlm(model_path="SmolVLM-256M-Instruct"):
    """
    加载SmolVLM-256M模型和处理器
    Args:
        model_path: 模型路径，默认为"SmolVLM-256M-Instruct"
    Returns:
        model: 加载的模型
        processor: 加载的处理器
    """

    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype=torch.bfloat16
    ).to("cuda")

    processor = AutoProcessor.from_pretrained(model_path)
    
    return model, processor

def load_image(image_input):
    """
    根据输入类型加载图像（URL或本地路径）
    Args:
        image_input: 可以是URL字符串或本地文件路径
    Returns:
        PIL.Image对象
    """
    # 判断是否是URL
    if image_input.startswith(('http://', 'https://')):
        try:
            response = requests.get(image_input, stream=True)
            response.raise_for_status()  # 检查请求是否成功
            return Image.open(response.raw)
        except Exception as e:
            print(f"加载URL图片失败: {e}")
            return None
    else:
        # 尝试作为本地文件路径处理
        try:
            if os.path.exists(image_input):
                return Image.open(image_input)
            else:
                print(f"本地文件不存在: {image_input}")
                return None
        except Exception as e:
            print(f"加载本地图片失败: {e}")
            return None

def process_image_text(model, processor, image_input, text):
    """
    处理图像和文本输入，生成响应
    Args:
        model: 加载的模型
        processor: 加载的处理器
        image_input: 可以是URL字符串或本地文件路径
        text: 输入文本
    Returns:
        str: 模型生成的响应
    """
    # 加载图像
    image = load_image(image_input)
    if image is None:
        return "failed to load image"
    
    # 构建输入消息
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": text}
            ]
        }
    ]
    
    # 处理输入
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    
    # 生成响应
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=500)
    
    # 解码响应
    response = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    
    return response


if __name__ == "__main__":

    model_path="SmolVLM-256M-Instruct"
    model, processor = load_smolvlm(model_path)
    image_path = "images/test/test.jpg"  
    prompt = """there is the statue of liberty. 
            Examine the image and identify the statue of liberty.
            Provide only the bounding box coordinates in normalized format [x_min, y_min, x_max, y_max], 
            where coordinates range from 0 to 1 based on image width and height.
            Output nothing else.
            """
    
    response1 = process_image_text(model, processor, image_path, prompt)
    print(response1)
    

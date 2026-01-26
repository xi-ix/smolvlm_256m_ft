import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import os

# 1. 设置模型路径 (可以直接写 huggingface 的 ID，也可以写本地路径)
model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
image_path = "images/test/0.png" 

print("正在加载模型...")

try:
    # 加载模型
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        cache_dir="model_weights/Qwen2.5-VL-3B",  
        dtype=torch.bfloat16, # 如果你的显卡比较老(如10系/20系)，这里建议改为 torch.float16
        device_map="auto",
        trust_remote_code=True
    )
    
    # 加载处理器
    # min_pixels 和 max_pixels 可以限制分辨率，防止显存爆炸
    processor = AutoProcessor.from_pretrained(model_path, min_pixels=256*28*28, max_pixels=1280*28*28)
    
    print("模型加载成功！")

    # 2. 准备测试
    question1 = """
        任务：检测交通拥堵状况。告诉我画面中有几辆车.
        回答我车辆数量以及是否拥堵。
        """
    
    prompt_text = question1
    if not os.path.exists(image_path):
        print(f"警告: 找不到图片 {image_path}，请修改代码中的图片路径")
    else:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{image_path}"},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        # 3. 推理
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        print("正在推理...")
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=256,
            do_sample=False,     # 关键：关闭随机采样，通过贪婪搜索获得最稳健的答案
            temperature=0.1,   # 或者开启采样但设极低
            # top_p=0.9
        )
        output_text = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        print("-" * 30)
        print("识别结果：")
        print(output_text)

except ValueError as e:
    if "accelerate" in str(e):
        print("\n!!! 错误：请运行 'pip install accelerate' !!!")
    else:
        print(f"发生错误: {e}")
except Exception as e:
    print(f"发生未知错误: {e}")
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import os

# 1. 设置模型路径 (可以直接写 huggingface 的 ID，也可以写本地路径)
model_path = "Qwen/Qwen2.5-VL-3B-Instruct"


print("正在加载模型...")


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

print("模型加载成功!\nQwen2.5-VL-3B-Instruct")

# 2. 准备测试
question0 = """
    任务：检测交通拥堵状况。
    """
question1 = """
    任务：检测交通拥堵状况。告诉我画面中有几辆车.
    回答我车辆数量以及是否拥堵。
    """
question2 = '''<image>\n
    1. 首先计算画面中可见车辆的数量和汽车占道路比例。
    2. 然后判断道路是否拥堵。（可以通过汽车数量、汽车之间的距离、汽车占据道路的比例来判断）。
    请严格按此格式输出：
    车辆数量：[数字]
    汽车占道路比例：[数字]
    是否拥堵：[Yes/No]'''

prompt_text = question2
print(f"prompt_text:\n{prompt_text}\n")
for num in range(5):
    image_path = f"images/test/{num}.png"
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

    generated_ids = model.generate(
        **inputs, 
        max_new_tokens=256,
        # do_sample=False,     # 关键：关闭随机采样，通过贪婪搜索获得最稳健的答案
        # temperature=0.1,   # 或者开启采样但设极低
        # top_p=0.9
    )
    full_output = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    output_text = full_output.split("assistant")[-1].strip()
    print("-" * 30)
    print(f"picture_{num}：")
    print(output_text)

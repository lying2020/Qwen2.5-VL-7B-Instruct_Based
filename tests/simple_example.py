#!/home/liying/miniconda3/envs/lisa/bin/python

"""
简单的图像编辑示例 - 使用 Stable Diffusion v1.5 Inpainting

最简化的使用示例
"""

import os
# 禁用 bitsandbytes 以避免 triton 兼容性问题（如果不需要量化）
os.environ["DIFFUSERS_NO_BITSANDBYTES"] = "1"

import torch

from diffusers import StableDiffusionInpaintPipeline

from PIL import Image, ImageDraw

# 1. 加载模型
print("加载模型...")

device = "cuda" if torch.cuda.is_available() else "cpu"


pipe = StableDiffusionInpaintPipeline.from_pretrained(
    ".",  # 当前目录（模型所在位置）
    torch_dtype=torch.float16,
    local_files_only=True,
    safety_checker=None,  # 禁用安全检测器以避免误判
    requires_safety_checker=False)
pipe = pipe.to(device)

# 2. 准备输入图片（请替换为你的图片路径）
image_path = "input_image.png"
image = Image.open(image_path).convert("RGB")
image = image.resize((512, 512), Image.Resampling.LANCZOS)

# 3. 创建掩码（白色区域是要编辑的区域）
mask = Image.new("RGB", (512, 512), color="black")
draw = ImageDraw.Draw(mask)
# 在中心创建一个圆形区域用于编辑
draw.ellipse([128, 128, 384, 384], fill="white")

# 4. 文本提示
prompt = "make the dog happy"

# 5. 生成编辑后的图片
print("生成中...")
result = pipe(
    prompt=prompt,
    image=image,
    mask_image=mask,
    num_inference_steps=50
)

# 6. 保存结果
output_image = result.images[0]
output_image.save("output.png")
print("完成！结果已保存到 output.png")


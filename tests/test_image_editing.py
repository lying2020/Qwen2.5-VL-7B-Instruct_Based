from diffusers import StableDiffusionPipeline, StableDiffusionInpaintPipeline
from PIL import Image
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(sys.path)
import project as prj

# 方案1: 使用基础模型进行文本生成图像（此仓库的模型）
# 注意：这个模型是 text-to-image，不是 inpainting 模型
print("正在加载模型...")
pipe = StableDiffusionPipeline.from_pretrained(
    prj.STABLE_DIFFUSION_V1_5_MODEL_PATH,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    safety_checker=None,  # 可选：禁用安全检查器以加快速度
    requires_safety_checker=False
)

if torch.cuda.is_available():
    pipe = pipe.to("cuda")

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
print("正在生成图像...")
image = pipe(prompt, num_inference_steps=20).images[0]
image.save(os.path.join(prj.EDITING_RESULTS_DIR, "generated_image.png"))
print("图像已保存为 generated_image.png")
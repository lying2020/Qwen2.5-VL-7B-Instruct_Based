"""
简单的图像编辑示例
快速测试单个编辑任务
"""
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image, ImageDraw
import torch
import sys
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(sys.path)

import project as prj

# 修复：禁用 mmap 加载以支持旧格式的模型文件
# Patch diffusers 的 load_state_dict 函数来禁用 mmap
try:
    from diffusers.models.model_loading_utils import load_state_dict as _original_load_state_dict

    def _patched_load_state_dict(checkpoint_file, dduf_entries=None, disable_mmap=False, map_location='cpu'):
        """禁用 mmap 的 load_state_dict - 强制设置 disable_mmap=True"""
        return _original_load_state_dict(
            checkpoint_file,
            dduf_entries=dduf_entries,
            disable_mmap=True,  # 强制禁用 mmap
            map_location=map_location
        )

    # 应用 patch
    import diffusers.models.model_loading_utils
    diffusers.models.model_loading_utils.load_state_dict = _patched_load_state_dict
except Exception as e:
    print(f"警告: 无法应用 mmap 修复: {e}")

def quick_edit(image_path, mask_type="center", prompt="", output_path="result.png"):
    """
    快速编辑函数

    参数:
        image_path: 输入图像路径
        mask_type: 遮罩类型 ("center", "corner", "rectangle")
        prompt: 编辑提示词
        output_path: 输出路径
    """
    print(f"加载模型...")
    try:
        # 尝试使用本地模型，禁用 safetensors 和低内存模式
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            prj.INPAINTING_MODEL_PATH,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
            use_safetensors=False,
            low_cpu_mem_usage=False
        )
    except Exception as e:
        print(f"本地模型加载失败: {e}")
        print("尝试从 Hugging Face 下载标准模型...")
        # 如果本地模型失败，尝试从 Hugging Face 下载
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")

    # 加载图像
    image = Image.open(image_path).convert("RGB")
    if image.size[0] != 512 or image.size[1] != 512:
        image = image.resize((512, 512), Image.Resampling.LANCZOS)

    # 创建遮罩
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)
    w, h = image.size

    if mask_type == "center":
        size = min(w, h) // 3
        draw.rectangle([w//2-size, h//2-size, w//2+size, h//2+size], fill=255)
    elif mask_type == "corner":
        size = min(w, h) // 3
        draw.rectangle([0, 0, size, size], fill=255)
    elif mask_type == "rectangle":
        draw.rectangle([w//4, h//4, 3*w//4, 3*h//4], fill=255)

    if not prompt:
        prompt = "detailed, high quality, seamless"

    print(f"正在编辑... (提示词: {prompt})")
    result = pipe(
        prompt=prompt,
        image=image,
        mask_image=mask,
        num_inference_steps=25,
        guidance_scale=7.5
    ).images[0]

    result.save(output_path)
    print(f"✓ 结果已保存: {output_path}")

if __name__ == "__main__":

    print("用法: python test_image_editing_simple.py <image_path> [mask_type] [prompt] [output]")
    print("示例: python test_image_editing_simple.py image.png center 'a beautiful flower'")

    image_path = sys.argv[1] if len(sys.argv) > 1 else "image.png"
    mask_type = sys.argv[2] if len(sys.argv) > 2 else "center"
    prompt = sys.argv[3] if len(sys.argv) > 3 else ""
    output = sys.argv[4] if len(sys.argv) > 4 else "result.png"

    image_path = os.path.join(prj.current_dir, image_path)
    output_path = os.path.join(prj.EDITING_RESULTS_DIR, output)

    quick_edit(image_path, mask_type, prompt, output_path)

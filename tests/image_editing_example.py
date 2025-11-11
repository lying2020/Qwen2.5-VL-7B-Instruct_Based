"""
使用 Stable Diffusion v1.5 Inpainting 进行图像编辑的示例

这个脚本展示了如何使用本地模型进行图像编辑：
1. 输入一张图片
2. 输入一个文本提示
3. 使用掩码指定要编辑的区域
4. 输出编辑后的图片
"""

import os
# 禁用 bitsandbytes 以避免 triton 兼容性问题（如果不需要量化）
os.environ["DIFFUSERS_NO_BITSANDBYTES"] = "1"

try:
    import torch
except ImportError:
    print("错误: 未安装 PyTorch")
    print("请运行: pip install torch")
    exit(1)

try:
    from diffusers import StableDiffusionInpaintPipeline
except (RuntimeError, ImportError, ModuleNotFoundError) as e:
    error_msg = str(e).lower()
    if "triton" in error_msg or "bitsandbytes" in error_msg:
        print("错误: diffusers 导入失败，可能是 bitsandbytes/triton 兼容性问题")
        print("尝试解决方案:")
        print("1. 安装兼容的 triton: pip install triton==2.1.0")
        print("2. 或者卸载 bitsandbytes: pip uninstall bitsandbytes")
        print("3. 或者使用 conda 环境: conda activate <your_env>")
    elif "peft" in error_msg:
        print("错误: 缺少 peft 模块")
        print("解决方案: pip install peft>=0.6.0")
        print("或者运行: pip install -r requirements.txt")
    else:
        print(f"错误: diffusers 导入失败: {e}")
        print("尝试运行: pip install -r requirements.txt")
    raise
from PIL import Image, ImageDraw


def create_mask_image(width=512, height=512, mask_type="center_circle"):
    """
    创建一个掩码图片
    mask_type: "center_circle" (中心圆形), "center_rectangle" (中心矩形), "full" (全图)
    """
    mask = Image.new("RGB", (width, height), color="black")
    draw = ImageDraw.Draw(mask)
    
    if mask_type == "center_circle":
        # 在中心创建一个圆形掩码
        center_x, center_y = width // 2, height // 2
        radius = min(width, height) // 4
        draw.ellipse(
            [center_x - radius, center_y - radius, center_x + radius, center_y + radius],
            fill="white"
        )
    elif mask_type == "center_rectangle":
        # 在中心创建一个矩形掩码
        rect_size = min(width, height) // 2
        left = (width - rect_size) // 2
        top = (height - rect_size) // 2
        right = left + rect_size
        bottom = top + rect_size
        draw.rectangle([left, top, right, bottom], fill="white")
    elif mask_type == "full":
        # 全图掩码（编辑整张图片）
        mask = Image.new("RGB", (width, height), color="white")
    
    return mask


def edit_image(
    image_path,
    prompt,
    mask_image=None,
    output_path="edited_image.png",
    model_path=".",
    num_inference_steps=50,
    guidance_scale=7.5,
    strength=1.0
):
    """
    使用 Stable Diffusion Inpainting 编辑图片
    
    参数:
        image_path: 输入图片路径
        prompt: 文本提示，描述想要编辑成什么样子
        mask_image: 掩码图片（PIL Image），白色区域是要编辑的区域，黑色区域保持不变
                    如果为 None，将自动创建一个中心圆形掩码
        output_path: 输出图片保存路径
        model_path: 模型路径（当前目录）
        num_inference_steps: 推理步数（越多质量越好但越慢）
        guidance_scale: 引导强度（越高越遵循提示词）
        strength: 编辑强度（0-1，1表示完全重新生成）
    """
    
    # 检查是否有 GPU
    try:
        if hasattr(torch, 'cuda') and hasattr(torch.cuda, 'is_available'):
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            print("警告: torch.cuda 不可用，使用 CPU")
            device = "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
    except Exception as e:
        print(f"警告: 检测设备时出错: {e}，使用 CPU")
        device = "cpu"
        dtype = torch.float32
    
    print(f"使用设备: {device}")
    print(f"加载模型从: {model_path}")
    
    # 加载模型
    # 如果模型在本地，使用本地路径
    try:
        if os.path.exists(os.path.join(model_path, "model_index.json")):
            pipe = StableDiffusionInpaintPipeline.from_pretrained(
                model_path,
                torch_dtype=dtype,
                local_files_only=True,
                safety_checker=None,  # 禁用安全检测器以避免误判
                requires_safety_checker=False
            )
        else:
            # 如果本地没有，从 HuggingFace 下载
            pipe = StableDiffusionInpaintPipeline.from_pretrained(
                "sd-legacy/stable-diffusion-inpainting",
                revision="fp16" if device == "cuda" else "main",
                torch_dtype=dtype,
                safety_checker=None,  # 禁用安全检测器以避免误判
                requires_safety_checker=False
            )
    except TypeError:
        # 如果旧版本不支持这些参数，先加载再禁用
        if os.path.exists(os.path.join(model_path, "model_index.json")):
            pipe = StableDiffusionInpaintPipeline.from_pretrained(
                model_path,
                torch_dtype=dtype,
                local_files_only=True
            )
        else:
            pipe = StableDiffusionInpaintPipeline.from_pretrained(
                "sd-legacy/stable-diffusion-inpainting",
                revision="fp16" if device == "cuda" else "main",
                torch_dtype=dtype
            )
        pipe.safety_checker = None  # 手动禁用安全检测器
        pipe.feature_extractor = None
    
    pipe = pipe.to(device)
    
    # 如果使用 CPU，可能需要优化
    if device == "cpu":
        pipe.enable_attention_slicing()
    
    # 加载输入图片
    print(f"加载图片: {image_path}")
    image = Image.open(image_path).convert("RGB")
    
    # 调整图片大小到 512x512（模型推荐尺寸）
    image = image.resize((512, 512), Image.Resampling.LANCZOS)
    
    # 如果没有提供掩码，创建一个默认的中心圆形掩码
    if mask_image is None:
        print("创建默认掩码（中心圆形）")
        mask_image = create_mask_image(512, 512, "center_circle")
    else:
        # 确保掩码大小与图片一致
        mask_image = mask_image.resize((512, 512), Image.Resampling.LANCZOS)
    
    # 保存掩码图片以便查看
    mask_image.save("mask.png")
    print("掩码已保存到 mask.png")
    
    print(f"文本提示: {prompt}")
    print("开始生成...")
    
    # 执行图像编辑
    result = pipe(
        prompt=prompt,
        image=image,
        mask_image=mask_image,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        strength=strength
    )
    
    # 获取生成的图片
    edited_image = result.images[0]
    
    # 保存结果
    edited_image.save(output_path)
    print(f"编辑后的图片已保存到: {output_path}")
    
    return edited_image


def main():
    """
    主函数 - 示例用法
    """
    # 示例 1: 使用本地图片文件
    # 请将 "input_image.jpg" 替换为你的图片路径
    input_image_path = "input_image.png"
    
    # 如果图片不存在，创建一个示例图片
    if not os.path.exists(input_image_path):
        print(f"警告: {input_image_path} 不存在")
        print("请将你的图片放在当前目录，或修改 input_image_path 变量")
        print("或者，我可以为你创建一个示例图片...")
        
        # 创建一个示例图片
        example_image = Image.new("RGB", (512, 512), color="lightblue")
        draw = ImageDraw.Draw(example_image)
        # 画一个简单的图案
        draw.rectangle([100, 100, 400, 400], fill="white", outline="black", width=3)
        example_image.save(input_image_path)
        print(f"已创建示例图片: {input_image_path}")
    
    # 文本提示 - 描述你想要编辑成什么样子
    prompt = "a beautiful sunset landscape with mountains, high quality, detailed"
    
    # 可选：创建自定义掩码
    # 例如，创建一个中心矩形掩码
    custom_mask = create_mask_image(512, 512, "center_rectangle")
    
    # 执行图像编辑
    edited_image = edit_image(
        image_path=input_image_path,
        prompt=prompt,
        mask_image=custom_mask,  # 使用自定义掩码，或设置为 None 使用默认圆形掩码
        output_path="output_edited_image.png",
        model_path=".",  # 当前目录（模型文件所在位置）
        num_inference_steps=50,  # 可以增加到 100 获得更好质量
        guidance_scale=7.5,
        strength=1.0
    )
    
    print("\n完成！")
    print("输出文件:")
    print("  - output_edited_image.png: 编辑后的图片")
    print("  - mask.png: 使用的掩码图片")


if __name__ == "__main__":
    main()


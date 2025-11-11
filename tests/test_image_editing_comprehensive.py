"""
综合图像编辑测试用例
包含：补全、添加、删除、替换等任务
"""
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image, ImageDraw
import torch
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(sys.path)
import project as prj

def load_inpainting_model():
    """加载 inpainting 模型"""
    print("正在加载 inpainting 模型...")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        prj.INPAINTING_MODEL_PATH,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    )
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    print("✓ 模型加载完成")
    return pipe

def create_mask(image_size, mask_type="center", position=None, size=None):
    """
    创建遮罩图像
    mask_type: "center", "rectangle", "circle", "corner", "multiple"
    """
    mask = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(mask)

    if mask_type == "center":
        # 中心区域遮罩
        w, h = image_size
        center_x, center_y = w // 2, h // 2
        box_size = min(w, h) // 3
        draw.rectangle(
            [center_x - box_size, center_y - box_size,
             center_x + box_size, center_y + box_size],
            fill=255
        )

    elif mask_type == "rectangle":
        # 矩形遮罩（指定位置）
        if position is None:
            position = (image_size[0]//4, image_size[1]//4)
        if size is None:
            size = (image_size[0]//2, image_size[1]//2)
        draw.rectangle(
            [position[0], position[1],
             position[0] + size[0], position[1] + size[1]],
            fill=255
        )

    elif mask_type == "circle":
        # 圆形遮罩
        w, h = image_size
        center_x, center_y = position or (w // 2, h // 2)
        radius = size[0] if size else min(w, h) // 4
        draw.ellipse(
            [center_x - radius, center_y - radius,
             center_x + radius, center_y + radius],
            fill=255
        )

    elif mask_type == "corner":
        # 角落遮罩
        w, h = image_size
        corner_size = min(w, h) // 3
        draw.rectangle(
            [0, 0, corner_size, corner_size],
            fill=255
        )

    elif mask_type == "multiple":
        # 多个区域遮罩
        w, h = image_size
        # 左上角
        draw.rectangle([0, 0, w//3, h//3], fill=255)
        # 右下角
        draw.rectangle([2*w//3, 2*h//3, w, h], fill=255)

    return mask

def test_case_1_complete(pipe, base_image):
    """测试用例1: 补全图像中心区域"""
    print("\n=== 测试用例1: 补全图像中心区域 ===")
    mask = create_mask(base_image.size, "center")

    prompt = "beautiful landscape, detailed, high quality"

    result = pipe(
        prompt=prompt,
        image=base_image,
        mask_image=mask,
        num_inference_steps=20,
        guidance_scale=7.5
    ).images[0]

    # 保存结果
    result_path = f"{prj.EDITING_RESULTS_DIR}/01_complete_center.png"
    result.save(result_path)
    print(f"✓ 结果已保存: {result_path}")
    return result

def test_case_2_add_object(pipe, base_image):
    """测试用例2: 添加对象（在指定区域添加新元素）"""
    print("\n=== 测试用例2: 添加对象 ===")
    # 在右下角添加遮罩
    w, h = base_image.size
    mask = create_mask(base_image.size, "rectangle",
                       position=(2*w//3, 2*h//3),
                       size=(w//4, h//4))

    prompt = "a cute dog, sitting, detailed, high quality"

    result = pipe(
        prompt=prompt,
        image=base_image,
        mask_image=mask,
        num_inference_steps=25,
        guidance_scale=7.5
    ).images[0]

    result_path = f"{prj.EDITING_RESULTS_DIR}/02_add_object.png"
    result.save(result_path)
    print(f"✓ 结果已保存: {result_path}")
    return result

def test_case_3_remove_object(pipe, base_image):
    """测试用例3: 删除对象（移除图像中的某个元素）"""
    print("\n=== 测试用例3: 删除对象 ===")
    # 遮罩要移除的区域
    mask = create_mask(base_image.size, "circle",
                       position=(base_image.size[0]//2, base_image.size[1]//2),
                       size=(base_image.size[0]//3,))

    # 使用描述周围环境的提示词来"填充"被移除的区域
    prompt = "natural background, seamless, detailed, high quality"

    result = pipe(
        prompt=prompt,
        image=base_image,
        mask_image=mask,
        num_inference_steps=25,
        guidance_scale=7.5
    ).images[0]

    result_path = f"{prj.EDITING_RESULTS_DIR}/03_remove_object.png"
    result.save(result_path)
    print(f"✓ 结果已保存: {result_path}")
    return result

def test_case_4_replace_object(pipe, base_image):
    """测试用例4: 替换对象（将某个元素替换为另一个元素）"""
    print("\n=== 测试用例4: 替换对象 ===")
    # 遮罩要替换的区域
    mask = create_mask(base_image.size, "rectangle",
                       position=(base_image.size[0]//4, base_image.size[1]//4),
                       size=(base_image.size[0]//2, base_image.size[1]//2))

    prompt = "a beautiful flower, colorful, detailed, high quality"

    result = pipe(
        prompt=prompt,
        image=base_image,
        mask_image=mask,
        num_inference_steps=25,
        guidance_scale=8.0
    ).images[0]

    result_path = f"{prj.EDITING_RESULTS_DIR}/04_replace_object.png"
    result.save(result_path)
    print(f"✓ 结果已保存: {result_path}")
    return result

def test_case_5_extend_image(pipe, base_image):
    """测试用例5: 扩展图像（在边缘补全）"""
    print("\n=== 测试用例5: 扩展图像边缘 ===")
    # 遮罩顶部区域
    w, h = base_image.size
    mask = create_mask(base_image.size, "rectangle",
                       position=(0, 0),
                       size=(w, h//3))

    prompt = "extended landscape, seamless continuation, detailed, high quality"

    result = pipe(
        prompt=prompt,
        image=base_image,
        mask_image=mask,
        num_inference_steps=25,
        guidance_scale=7.5
    ).images[0]

    result_path = f"{prj.EDITING_RESULTS_DIR}/05_extend_image.png"
    result.save(result_path)
    print(f"✓ 结果已保存: {result_path}")
    return result

def test_case_6_repair_image(pipe, base_image):
    """测试用例6: 修复图像（修复损坏或缺失的部分）"""
    print("\n=== 测试用例6: 修复图像 ===")
    # 模拟损坏区域（多个小区域）
    mask = create_mask(base_image.size, "multiple")

    prompt = "restored image, seamless, natural, detailed, high quality"

    result = pipe(
        prompt=prompt,
        image=base_image,
        mask_image=mask,
        num_inference_steps=30,
        guidance_scale=7.5
    ).images[0]

    result_path = f"{prj.EDITING_RESULTS_DIR}/06_repair_image.png"
    result.save(result_path)
    print(f"✓ 结果已保存: {result_path}")
    return result

def test_case_7_style_transfer(pipe, base_image):
    """测试用例7: 风格转换（在特定区域应用不同风格）"""
    print("\n=== 测试用例7: 风格转换 ===")
    mask = create_mask(base_image.size, "circle",
                       position=(base_image.size[0]//2, base_image.size[1]//2),
                       size=(base_image.size[0]//3,))

    prompt = "watercolor painting style, artistic, soft colors, detailed"

    result = pipe(
        prompt=prompt,
        image=base_image,
        mask_image=mask,
        num_inference_steps=30,
        guidance_scale=8.5
    ).images[0]

    result_path = f"{prj.EDITING_RESULTS_DIR}/07_style_transfer.png"
    result.save(result_path)
    print(f"✓ 结果已保存: {result_path}")
    return result

def test_case_8_change_background(pipe, base_image):
    """测试用例8: 更换背景"""
    print("\n=== 测试用例8: 更换背景 ===")
    # 遮罩除了中心外的所有区域（保留前景，替换背景）
    w, h = base_image.size
    mask = Image.new("L", base_image.size, 255)
    draw = ImageDraw.Draw(mask)
    # 保留中心区域
    center_region = (w//4, h//4, 3*w//4, 3*h//4)
    draw.rectangle(center_region, fill=0)

    prompt = "beautiful sunset sky, clouds, dramatic lighting, detailed"

    result = pipe(
        prompt=prompt,
        image=base_image,
        mask_image=mask,
        num_inference_steps=25,
        guidance_scale=7.5
    ).images[0]

    result_path = f"{prj.EDITING_RESULTS_DIR}/08_change_background.png"
    result.save(result_path)
    print(f"✓ 结果已保存: {result_path}")
    return result

def test_case_9_add_texture(pipe, base_image):
    """测试用例9: 添加纹理（在特定区域添加纹理效果）"""
    print("\n=== 测试用例9: 添加纹理 ===")
    mask = create_mask(base_image.size, "rectangle",
                       position=(base_image.size[0]//3, base_image.size[1]//3),
                       size=(base_image.size[0]//3, base_image.size[1]//3))

    prompt = "wooden texture, detailed grain, natural, high quality"

    result = pipe(
        prompt=prompt,
        image=base_image,
        mask_image=mask,
        num_inference_steps=25,
        guidance_scale=7.5
    ).images[0]

    result_path = f"{prj.EDITING_RESULTS_DIR}/09_add_texture.png"
    result.save(result_path)
    print(f"✓ 结果已保存: {result_path}")
    return result

def test_case_10_creative_edit(pipe, base_image):
    """测试用例10: 创意编辑（组合多个效果）"""
    print("\n=== 测试用例10: 创意编辑 ===")
    # 创建复杂遮罩
    w, h = base_image.size
    mask = Image.new("L", base_image.size, 0)
    draw = ImageDraw.Draw(mask)
    # 多个圆形区域
    draw.ellipse([w//4, h//4, w//2, h//2], fill=255)
    draw.ellipse([w//2, h//2, 3*w//4, 3*h//4], fill=255)

    prompt = "magical glowing effects, fantasy elements, detailed, high quality"

    result = pipe(
        prompt=prompt,
        image=base_image,
        mask_image=mask,
        num_inference_steps=30,
        guidance_scale=8.0
    ).images[0]

    result_path = f"{prj.EDITING_RESULTS_DIR}/10_creative_edit.png"
    result.save(result_path)
    print(f"✓ 结果已保存: {result_path}")
    return result

def main():
    """主函数：运行所有测试用例"""
    print("=" * 60)
    print("图像编辑综合测试套件")
    print("=" * 60)

    # 加载模型
    pipe = load_inpainting_model()

    # 加载基础图像（如果存在，否则生成一个）
    base_image_path = os.path.join(prj.EDITING_RESULTS_DIR, "generated_image.png")
    if os.path.exists(base_image_path):
        print(f"使用现有图像: {base_image_path}")
        base_image = Image.open(base_image_path).convert("RGB")
    else:
        print("生成新的测试图像...")
        from diffusers import StableDiffusionPipeline
        text_pipe = StableDiffusionPipeline.from_pretrained(
            prj.STABLE_DIFFUSION_V1_5_MODEL_PATH,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        if torch.cuda.is_available():
            text_pipe = text_pipe.to("cuda")
        base_image = text_pipe("a beautiful landscape, mountains, sky, detailed",
                              num_inference_steps=20).images[0]
        base_image.save(os.path.join(prj.EDITING_RESULTS_DIR, base_image_path))
        print(f"✓ 测试图像已保存: {os.path.join(prj.EDITING_RESULTS_DIR, base_image_path)}")

    # 确保图像尺寸合适（512x512 或 768x768）
    if base_image.size[0] != 512 or base_image.size[1] != 512:
        base_image = base_image.resize((512, 512), Image.Resampling.LANCZOS)
        print("图像已调整为 512x512")

    # 运行所有测试用例
    test_cases = [
        ("补全", test_case_1_complete),
        ("添加对象", test_case_2_add_object),
        ("删除对象", test_case_3_remove_object),
        ("替换对象", test_case_4_replace_object),
        ("扩展图像", test_case_5_extend_image),
        ("修复图像", test_case_6_repair_image),
        ("风格转换", test_case_7_style_transfer),
        ("更换背景", test_case_8_change_background),
        ("添加纹理", test_case_9_add_texture),
        ("创意编辑", test_case_10_creative_edit),
    ]

    print(f"\n开始运行 {len(test_cases)} 个测试用例...")
    print("-" * 60)

    results = {}
    for name, test_func in test_cases:
        try:
            result = test_func(pipe, base_image)
            results[name] = "✓ 成功"
        except Exception as e:
            print(f"✗ 测试失败: {name}")
            print(f"  错误: {str(e)}")
            results[name] = f"✗ 失败: {str(e)}"

    # 总结
    print("\n" + "=" * 60)
    print("测试结果总结")
    print("=" * 60)
    for name, status in results.items():
        print(f"{name:15s}: {status}")
    print(f"\n所有结果保存在: {prj.EDITING_RESULTS_DIR}/")
    print("=" * 60)

if __name__ == "__main__":
    main()

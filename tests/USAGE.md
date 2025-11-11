# 图像编辑使用说明

这个目录包含了使用 Stable Diffusion v1.5 Inpainting 进行图像编辑的示例代码。

## 安装依赖

首先安装所需的 Python 包：

```bash
pip install -r requirements.txt
```

## 使用方法

### 方法 1: 使用简单示例（推荐新手）

`simple_example.py` 是一个最简化的示例：

```bash
python simple_example.py
```

**使用前需要：**
1. 准备一张输入图片，命名为 `input_image.jpg` 放在当前目录
2. 修改脚本中的 `prompt` 变量为你的文本提示
3. 可选：修改掩码区域（脚本中创建的是中心圆形区域）

### 方法 2: 使用完整示例（推荐）

`image_editing_example.py` 提供了更多功能和选项：

```bash
python image_editing_example.py
```

**功能特点：**
- 自动创建示例图片（如果输入图片不存在）
- 支持多种掩码类型（圆形、矩形、全图）
- 可调节参数（推理步数、引导强度等）
- 自动保存掩码图片

## 参数说明

### 文本提示 (prompt)
描述你想要将图片编辑成什么样子，例如：
- `"a beautiful sunset landscape"`
- `"a red sports car, high quality"`
- `"a cute cat wearing sunglasses"`

### 掩码 (mask)
- **白色区域**：会被重新生成/编辑
- **黑色区域**：保持原样不变

### 主要参数
- `num_inference_steps`: 推理步数（默认50，增加可提高质量但更慢）
- `guidance_scale`: 引导强度（默认7.5，越高越遵循提示词）
- `strength`: 编辑强度（0-1，1表示完全重新生成）

## 示例工作流程

1. **准备输入图片**
   ```python
   image = Image.open("your_image.jpg")
   image = image.resize((512, 512))
   ```

2. **创建掩码**（指定要编辑的区域）
   ```python
   mask = Image.new("RGB", (512, 512), "black")
   # 在掩码上画白色区域表示要编辑的部分
   ```

3. **设置文本提示**
   ```python
   prompt = "your description here"
   ```

4. **运行模型**
   ```python
   result = pipe(prompt=prompt, image=image, mask_image=mask)
   ```

5. **保存结果**
   ```python
   result.images[0].save("output.png")
   ```

## 注意事项

- 模型推荐图片尺寸为 512x512
- 如果有 GPU，会自动使用 GPU 加速
- 首次运行可能需要一些时间加载模型
- 生成一张图片通常需要 10-30 秒（取决于硬件）

## 故障排除

**问题：找不到模型文件**
- 确保在模型文件所在的目录运行脚本
- 或者修改脚本中的 `model_path` 参数

**问题：内存不足**
- 如果使用 CPU，可能需要启用 `enable_attention_slicing()`
- 减少 `num_inference_steps` 参数
- 使用 `torch.float32` 而不是 `torch.float16`（如果 CPU 不支持）

**问题：生成质量不好**
- 增加 `num_inference_steps`（如 100）
- 调整 `guidance_scale`（如 7.5-10）
- 使用更详细的文本提示


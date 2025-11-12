# /bin/python3

import gradio as gr
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

import project as prj
# 加载模型和处理器
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    prj.QWEN2_5_VL_7B_INSTUCT_MODEL_PATH,
    torch_dtype="auto",
    device_map="auto"
)

processor = AutoProcessor.from_pretrained(prj.QWEN2_5_VL_7B_INSTUCT_MODEL_PATH)

def process_image_and_text(image, text_prompt):
    if image is None:
        return "请上传一张图片。"

    # 构建消息格式
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,  # Gradio将自动处理图片路径
                },
                {"type": "text", "text": text_prompt if text_prompt else "Describe this image."},
            ],
        }
    ]

    try:
        # 准备推理输入
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        # 生成输出
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )

        return output_text[0]

    except Exception as e:
        return f"处理过程中出现错误: {str(e)}"

# 创建Gradio界面
with gr.Blocks() as demo:
    gr.Markdown("# Qwen2.5-VL 图像理解演示")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="filepath", label="上传图片")
            text_input = gr.Textbox(
                placeholder="请输入提示语（如不输入，默认描述图片）",
                label="提示语"
            )
            submit_btn = gr.Button("提交")

        with gr.Column():
            output = gr.Textbox(label="输出结果")

    submit_btn.click(
        fn=process_image_and_text,
        inputs=[image_input, text_input],
        outputs=output
    )

    # 示例部分已移除，因为需要真实的图片路径
    # 如果需要添加示例，请使用实际存在的图片路径
    # gr.Examples(
    #     examples=[
    #         ["path/to/example1.jpg", "这张图片里有什么？"],
    #         ["path/to/example2.jpg", "描述图中的场景"],
    #     ],
    #     inputs=[image_input, text_input],
    # )

# 启动应用
if __name__ == "__main__":
    demo.launch(share=True)
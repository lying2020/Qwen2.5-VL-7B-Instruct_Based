from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer
from PIL import Image
import torch
import json
import re


def create_annotation_prompt(instruction):
    return f"""
基于这个反事实指令，生成结构化的图像编辑标注：

指令："{instruction}"

请输出严格的JSON格式，包含以下字段：
{{
    "sample_id": "自动生成的唯一ID",
    "instruction": "原始反事实指令",
    "reasoning_chain": "因果推理链条，用→连接步骤",
    "counterfactual_premise": {{
        "changed_factor": "被改变的关键因素",
        "original_state": "原始状态描述",
        "counterfactual_state": "反事实状态描述",
        "causal_effect": "预期的因果效应"
    }},
    "multi_modal_constraints": [
        {{
            "type": "约束大类",
            "subtype": "约束子类",
            "description": "具体约束描述",
            "importance": "high/medium/low"
        }}
    ],
    "edit_subject": ["被编辑对象列表"],
    "new_subject": ["新对象列表"],
    "edit_type": "编辑类型",
    "editing_instruction": "直接的编辑命令，适合inpainting模型理解"
}}

要求：
1. editing_instruction 要简洁直接，使用"replace A with B", "add X", "remove Y"等格式
2. reasoning_chain 要体现清晰的因果逻辑
3. constraints 要覆盖空间、语义、物理等方面
4. edit_subject 和 new_subject 要具体明确
"""


class DataAnnotationAssistant:
    def __init__(self):
        self.model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def generate_annotation(self, image_path, base_instruction):
        """为单张图像生成完整的标注信息"""

        image = Image.open(image_path).convert("RGB")

        prompt = create_annotation_prompt(base_instruction)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=1024,
            temperature=0.7,
            do_sample=True
        )

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # 提取JSON部分
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            try:
                annotation = json.loads(json_str)
                return annotation
            except json.JSONDecodeError:
                print("JSON解析失败，返回原始响应")
                return {"raw_response": response}

        return {"raw_response": response}

def batch_annotate_images(image_instruction_pairs, output_dir):
    """批量处理多张图像"""
    assistant = DataAnnotationAssistant()
    all_annotations = []

    for i, (image_path, instruction) in enumerate(image_instruction_pairs):
        print(f"处理第 {i+1}/{len(image_instruction_pairs)} 张图像: {image_path}")

        annotation = assistant.generate_annotation(image_path, instruction)
        annotation["sample_id"] = f"sample_{i:04d}"

        all_annotations.append(annotation)

        # 保存单个文件的标注
        output_path = f"{output_dir}/annotation_{i:04d}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(annotation, f, indent=2, ensure_ascii=False)

    # 保存所有标注
    with open(f"{output_dir}/all_annotations.json", "w", encoding="utf-8") as f:
        json.dump(all_annotations, f, indent=2, ensure_ascii=False)

    return all_annotations

# 使用示例
def main():
    assistant = DataAnnotationAssistant()

    # 为单张图像生成标注
    annotation = assistant.generate_annotation(
        image_path="path/to/your/image.jpg",
        base_instruction="如果这个支撑柱被移除，建筑会怎样变化？"
    )

    print("生成的标注信息：")
    print(json.dumps(annotation, indent=2, ensure_ascii=False))

    # 保存到文件
    with open("annotation.json", "w", encoding="utf-8") as f:
        json.dump(annotation, f, indent=2, ensure_ascii=False)


    # 批量处理示例
    image_instructions = [
        ("images/bridge.jpg", "如果中间支撑柱被移除，桥梁会怎样？"),
        ("images/forest.jpg", "如果这片森林生长了50年，会变成什么样子？"),
        ("images/city.jpg", "如果这座城市优先发展绿化而不是高楼，现在会怎样？"),
        ("images/person.jpg", "如果这个人在艺术世家长大，现在会是什么样子？")
    ]

    batch_annotate_images(image_instructions, "output_annotations")


if __name__ == "__main__":
    main()
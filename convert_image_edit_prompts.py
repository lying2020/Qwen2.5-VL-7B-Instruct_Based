#!/usr/bin/env python3
"""
使用 Qwen2.5-VL 模型生成图像编辑提示词信息

支持单张图片处理和批量处理
"""

import os
import json
import torch
import uuid
from pathlib import Path
from typing import Union, List, Dict, Optional, Tuple
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
from transformers.processing_utils import ImagesKwargs

import project as prj



def create_editing_prompt(instruction):
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
    {
        "type": "spatial_layout",
        "description": "对象的位置、大小、方向、相对位置关系、遮挡关系、透视关系等"
    },
    {
        "type": "semantic_content",
        "description": "对象类别、颜色、材料等属性、替换关系、语义一致性"
    },
    {
        "type": "physical_causal",
        "description": "物理规律，力学关系，因果关系，重力作用，遮挡关系，透视关系等"
    },
    {
    "type": "temporal_reasoning",
    "description": "时间变化、年龄、季节、历史演变"
    },
    {
    "type": "narrative_story",
    "description": "故事情节、角色关系、情节发展"
    },
    {
    "type": "biological_ecological",
    "description": "生物学、生态学、环境学、生物多样性、生物特性、生物关系、生物行为、生物状态等"
    },
    {
    "type": "frunctional_intentional",
    "description": "功能性、意图性、目的性、动机性、功能关系、意图关系、目的关系、动机关系等"
    }
    ],
    "edit_subject": ["编辑对象列表"],
    "new_subject": ["新的对象列表"],
    "edit_type": "编辑类型",
    "editing_instruction": "详细描述的编辑命令, 直接的编辑命令，适合inpainting模型理解"
}}

要求：
1. instruction 用反事实的方式描述指令，要清晰描述编辑任务，用简洁明了的语言描述编辑任务，不要使用复杂的句子结构
2. reasoning_chain 要体现清晰的因果逻辑
3. multi_modal_constraints 要覆盖空间、语义、物理( spatial_layout, semantic_content, physical_causal,  frunctional_intentional, temporal_reasoning, narrative_story, biological_ecological)等方面
4. edit_subject 和 new_subject 要具体明确，编辑对象和新的对象要具体明确
5. edit_type定义标准值："object_replacement", "addition", "removal", "modification", "transformation", "combination", "deletion", "rearrangement"等
6. detailed_instructions 要详细描述编辑步骤，用step1, step2, step3, ... 表示
7. editing_instruction 要简洁直接，使用"replace A with B", "add X", "remove Y"等格式

"""


class ImageEditPromptGenerator:
    """图像编辑提示词生成器"""

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        """
        初始化模型和处理器

        Args:
            model_path: 模型路径，如果为 None 则使用 project.py 中的路径
            device: 设备，如果为 None 则自动选择
        """
        if model_path is None:
            model_path = prj.QWEN2_5_VL_7B_INSTUCT_MODEL_PATH

        print(f"加载模型: {model_path}")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto"
        )

        self.processor = AutoProcessor.from_pretrained(model_path)
        self.device = device if device else self.model.device

        print(f"模型加载完成，设备: {self.device}")

    def generate_editing_prompts(
        self,
        image_path: Union[str, Path, Image.Image],
        edit_request: str,
        max_new_tokens: int = 512
    ) -> Dict:
        """
        为单张图片生成编辑提示词信息

        Args:
            image_path: 图片路径或 PIL Image 对象
            edit_request: 编辑请求文本
            max_new_tokens: 最大生成token数

        Returns:
            包含以下字段的字典（符合 sample_temple.json 格式）:
            - sample_id: 自动生成的唯一ID
            - instruction: 原始反事实指令
            - reasoning_chain: 因果推理链条
            - counterfactual_premise: 反事实前提（包含 changed_factor, original_state, counterfactual_state, causal_effect）
            - multi_modal_constraints: 多模态约束列表（包含 type 和 description）
            - edit_subject: 编辑对象列表
            - new_subject: 新的对象列表
            - edit_type: 编辑类型
            - editing_instruction: 详细描述的编辑命令
            - raw_response: 原始模型响应
        """
        # 加载图片
        if isinstance(image_path, (str, Path)):
            image = Image.open(image_path).convert("RGB")
        else:
            image = image_path

        # 构建提示词
        prompt = create_editing_prompt(edit_request)

        # 构建消息
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        try:
            # 准备推理输入
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model.device)

            # 生成输出
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=None
                )
                generated_ids_trimmed = [
                    out_ids[len(in_ids):]
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0]

            # 解析JSON响应
            result = self._parse_response(output_text, edit_request)
            result['raw_response'] = output_text

            # 如果没有 sample_id，自动生成一个
            if not result.get('sample_id'):
                result['sample_id'] = str(uuid.uuid4())

            return result

        except Exception as e:
            return {
                'error': str(e),
                'edit_request': edit_request,
                'raw_response': ''
            }

    def _parse_response(self, response: str, edit_request: str) -> Dict:
        """解析模型响应，提取JSON信息（符合 sample_temple.json 格式）"""
        # 初始化结果结构，符合 sample_temple.json 格式
        result = {
            'sample_id': '',
            'instruction': '',
            'reasoning_chain': '',
            'counterfactual_premise': {
                'changed_factor': '',
                'original_state': '',
                'counterfactual_state': '',
                'causal_effect': ''
            },
            'multi_modal_constraints': [],
            'edit_subject': [],
            'new_subject': [],
            'edit_type': '',
            'editing_instruction': ''
        }

        # 尝试提取JSON
        try:
            # 查找JSON块
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1

            if start_idx >= 0 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                parsed = json.loads(json_str)

                # 更新结果，符合新格式
                result.update({
                    'sample_id': parsed.get('sample_id', ''),
                    'instruction': parsed.get('instruction', edit_request),
                    'reasoning_chain': parsed.get('reasoning_chain', ''),
                    'counterfactual_premise': parsed.get('counterfactual_premise', result['counterfactual_premise']),
                    'multi_modal_constraints': parsed.get('multi_modal_constraints', []),
                    'edit_subject': parsed.get('edit_subject', []),
                    'new_subject': parsed.get('new_subject', []),
                    'edit_type': parsed.get('edit_type', ''),
                    'editing_instruction': parsed.get('editing_instruction', '')
                })

                # 确保 counterfactual_premise 结构完整
                if isinstance(result['counterfactual_premise'], dict):
                    result['counterfactual_premise'] = {
                        'changed_factor': result['counterfactual_premise'].get('changed_factor', ''),
                        'original_state': result['counterfactual_premise'].get('original_state', ''),
                        'counterfactual_state': result['counterfactual_premise'].get('counterfactual_state', ''),
                        'causal_effect': result['counterfactual_premise'].get('causal_effect', '')
                    }

                # 确保 multi_modal_constraints 是列表格式
                if not isinstance(result['multi_modal_constraints'], list):
                    result['multi_modal_constraints'] = []

            else:
                # 如果没有找到JSON，使用编辑请求作为基础信息
                result['instruction'] = edit_request
                result['reasoning_chain'] = response[:500]  # 使用前500字符作为推理链条

        except json.JSONDecodeError as e:
            # JSON解析失败，使用原始响应和编辑请求
            result['instruction'] = edit_request
            result['reasoning_chain'] = response[:500]
            print(f"JSON解析警告: {e}")

        return result

    def process_batch(
        self,
        image_prompt_pairs: List[Tuple[Union[str, Path, Image.Image], str]],
        output_file: Optional[Union[str, Path]] = None,
        save_individual: bool = False,
        output_dir: Optional[Union[str, Path]] = None
    ) -> List[Dict]:
        """
        批量处理图片和提示词

        Args:
            image_prompt_pairs: 图片路径和提示词的元组列表 [(image_path, prompt), ...]
            output_file: 输出JSON文件路径，如果为None则不保存
            save_individual: 是否保存每个结果的单独文件
            output_dir: 单独文件的输出目录

        Returns:
            结果列表，每个元素是一个包含生成信息的字典
        """
        results = []

        if output_dir and save_individual:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        for idx, (image_path, prompt) in enumerate(tqdm(image_prompt_pairs, desc="处理图片")):
            try:
                result = self.generate_editing_prompts(image_path, prompt)
                result['image_path'] = str(image_path)
                result['edit_request'] = prompt
                result['index'] = idx

                results.append(result)

                # 保存单独文件
                if save_individual and output_dir:
                    output_file_individual = output_dir / f"result_{idx:05d}.json"
                    with open(output_file_individual, 'w', encoding='utf-8') as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)

            except Exception as e:
                error_result = {
                    'error': str(e),
                    'image_path': str(image_path),
                    'edit_request': prompt,
                    'index': idx
                }
                results.append(error_result)
                print(f"\n处理第 {idx} 个样本时出错: {e}")

        # 保存汇总文件
        if output_file:
            output_file = Path(output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\n结果已保存到: {output_file}")

        return results

    def process_from_json(
        self,
        input_json: Union[str, Path, Dict],
        image_base_dir: Optional[Union[str, Path]] = None,
        output_file: Optional[Union[str, Path]] = None,
        save_individual: bool = False,
        output_dir: Optional[Union[str, Path]] = None
    ) -> List[Dict]:
        """
        从JSON文件读取图片路径和提示词，批量处理

        Args:
            input_json: JSON文件路径或字典，格式应为:
                [
                    {"image_path": "path/to/image.jpg", "edit_request": "编辑请求"},
                    ...
                ]
            image_base_dir: 图片基础目录，如果JSON中的路径是相对路径
            output_file: 输出文件路径

        Returns:
            结果列表
        """
        # 读取JSON
        if isinstance(input_json, (str, Path)):
            with open(input_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = input_json

        # 构建图片-提示词对
        pairs = []
        if image_base_dir:
            image_base_dir = Path(image_base_dir)

        for item in data:
            image_path = item.get('image_path') or item.get('image')
            edit_request = item.get('edit_request') or item.get('prompt') or item.get('instruction')

            if not image_path or not edit_request:
                continue

            # 处理相对路径
            if image_base_dir and not Path(image_path).is_absolute():
                image_path = image_base_dir / image_path
            else:
                image_path = Path(image_path)

            pairs.append((image_path, edit_request))

        # 批量处理
        return self.process_batch(
            pairs,
            output_file=output_file,
            save_individual=save_individual,
            output_dir=output_dir
        )


def main():
    """示例用法"""
    import argparse

    images_dir = prj.HALLUSEGBENCH_DATASET_PATH
    factual_images_dir = os.path.join(images_dir, "factual_images")
    factual_images = os.listdir(factual_images_dir)
    factual_images_paths = [os.path.join(factual_images_dir, image) for image in factual_images]
    factual_images = [Image.open(image_path) for image_path in factual_images_paths]

    counterfactual_images_dir = os.path.join(images_dir, "counterfactual_images")
    counterfactual_images = os.listdir(counterfactual_images_dir)
    counterfactual_images_paths = [os.path.join(counterfactual_images_dir, image) for image in counterfactual_images]
    counterfactual_images = [Image.open(image_path) for image_path in counterfactual_images_paths]

    converted_data_json = os.path.join(images_dir, "converted_data.json")
    with open(converted_data_json, 'r', encoding='utf-8') as f:
        converted_data = json.load(f)

    parser = argparse.ArgumentParser(description='生成图像编辑提示词')
    parser.add_argument('--image', type=str, default=factual_images[0], help='单张图片路径')
    parser.add_argument('--prompt', type=str, default="Change zebra at the bottom to cow.", help='编辑提示词')
    parser.add_argument('--json', type=str, default=None, help='批量处理的JSON文件路径')
    parser.add_argument('--image_dir', type=str, default=factual_images_dir, help='图片基础目录（用于JSON中的相对路径）')
    parser.add_argument('--output', type=str, default='output.json', help='输出文件路径')
    parser.add_argument('--output_dir', type=str, default=prj.current_dir, help='单独文件的输出目录')
    parser.add_argument('--save_individual', action='store_true', default=True, help='保存每个结果的单独文件')

    args = parser.parse_args()

    # 初始化生成器
    generator = ImageEditPromptGenerator()

    if args.image and args.prompt:
        # 单张图片处理
        print(f"处理图片: {args.image}")
        result = generator.generate_editing_prompts(args.image, args.prompt)
        print("\n结果:")
        print(json.dumps(result, ensure_ascii=False, indent=2))

        # 保存结果
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到: {args.output}")

    elif args.json:
        # 批量处理
        print(f"从JSON文件批量处理: {args.json}")
        results = generator.process_from_json(
            args.json,
            image_base_dir=args.image_dir,
            output_file=args.output,
            save_individual=args.save_individual,
            output_dir=args.output_dir
        )
        print(f"\n处理完成，共 {len(results)} 个结果")

    else:
        print("请提供 --image 和 --prompt 或 --json 参数")
        parser.print_help()


if __name__ == "__main__":
    main()

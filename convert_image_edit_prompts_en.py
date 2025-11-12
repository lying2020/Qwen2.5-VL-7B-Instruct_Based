#!/usr/bin/env python3
"""
Generate image editing prompt information using Qwen2.5-VL model

Supports single image processing and batch processing
"""

import os
import json
import re
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



def create_editing_prompt(instruction, has_source=True, has_mask=False, has_target=False, has_target_mask=False):
    """
    Create editing prompt

    Args:
        instruction: Editing instruction
        has_source: Whether source image is provided
        has_mask: Whether source image mask is provided
        has_target: Whether target image is provided
        has_target_mask: Whether target image mask is provided
    """
    image_context = []
    image_idx = 1
    if has_source:
        image_context.append(f"Image {image_idx} is the original image (source image)")
        image_idx += 1
    if has_mask:
        image_context.append(f"Image {image_idx} is the editing region mask of the original image (source mask image), where white areas indicate the parts to be edited")
        image_idx += 1
    if has_target:
        image_context.append(f"Image {image_idx} is the edited target image (target image)")
        image_idx += 1
    if has_target_mask:
        image_context.append(f"Image {image_idx} is the mask of the target image (target mask image), indicating the edited region")

    image_context_str = ". ".join(image_context) + "." if image_context else ""

    return f"""
Based on this counterfactual instruction, generate structured image editing annotations:

{image_context_str}

Instruction: "{instruction}"

Please output strict JSON format with the following fields:
{{
    "sample_id": "Auto-generated unique ID",
    "instruction": "Counterfactual instruction rewritten from semantic assumptions/conditional changes, introducing implicit and reasonable causal hypotheses. The instruction should start from semantic assumptions or conditional changes, such as personal situations, needs, constraints, or contextual factors, and naturally lead to the required image editing. Examples: 'I usually live alone and sometimes need to work from home. How should I renovate my bedroom?' or 'My personal doctor says I have been consuming too much sugar and eating unhealthily, and I should supplement with vitamin C. What should I do?'",
    "reasoning_chains": {{
        "descriptive": "Descriptive reasoning chain: identify→analyze→transform→verify",
        "causal": "Causal reasoning chain: premise→intervention→effect→outcome",
        "comparative": "Comparative reasoning chain: original_analysis→target_analysis→difference_identification→transformation_strategy"
    }},
    "counterfactual_premise": {{
        "changed_factor": "Key factor that was changed",
        "original_state": "Description of original state",
        "counterfactual_state": "Description of counterfactual state",
        "causal_effect": "Expected causal effect"
    }},
    "multi_modal_constraints": [
    {{
        "type": "spatial_layout",
        "description": "Object positions, sizes, orientations, relative positional relationships, occlusion relationships, perspective relationships, etc."
    }},
    {{
        "type": "semantic_content",
        "description": "Object categories, colors, materials and other attributes, replacement relationships, semantic consistency"
    }},
    {{
        "type": "physical_causal",
        "description": "Physical laws, mechanical relationships, causal relationships, gravitational effects, occlusion relationships, perspective relationships, etc."
    }},
    {{
    "type": "temporal_reasoning",
    "description": "Temporal changes, age, seasons, historical evolution"
    }},
    ],
    "edit_metadata": {{
        "edit_subject": ["List of editing objects"],
        "new_subject": ["List of new objects"],
        "edit_type": "Editing type: object_replacement|addition|removal|modification|transformation|combination|deletion|rearrangement|temporal_evolution",
        "complexity_level": "Complexity level: simple|medium|complex"
    }},
    "editing_instruction": "Detailed editing command, direct editing command suitable for inpainting model understanding"
}}

Requirements:
1. instruction MUST be rewritten from semantic assumptions/conditional changes, introducing implicit and reasonable causal hypotheses. Start from personal situations, needs, constraints, or contextual factors (like lifestyle, health conditions, social situations, environmental requirements, etc.), and naturally lead to the required image editing. The instruction should feel natural and reasonable, not directly stating the editing task. Examples:
   - "I usually live alone and sometimes need to work from home. How should I renovate my bedroom?" (leads to adding workspace furniture)
   - "My personal doctor says I have been consuming too much sugar and eating unhealthily, and I should supplement with vitamin C. What should I do?" (leads to replacing unhealthy food with fruits/vegetables)
   - "I recently joined a new company and want to have a small gathering with new colleagues, but one colleague said they are allergic to alcohol (or taking antibiotics). What should I do?" (leads to replacing alcoholic drinks with non-alcoholic alternatives)
2. reasoning_chains should contain three types:
   - descriptive: step-by-step identification and transformation process
   - causal: causal logic from premise to outcome
   - comparative: comparison between original and target states
3. multi_modal_constraints should cover spatial, semantic, physical, and temporal aspects (spatial_layout, semantic_content, physical_causal, temporal_reasoning)
4. edit_metadata.edit_subject and edit_metadata.new_subject should be specific and clear, editing objects and new objects should be specific and clear
5. edit_metadata.edit_type should use standard values: "object_replacement", "addition", "removal", "modification", "transformation", "combination", "deletion", "rearrangement", "temporal_evolution"
6. edit_metadata.complexity_level should be assessed based on the number of objects, spatial relationships, and editing difficulty: "simple", "medium", or "complex"
7. editing_instruction should be concise and direct, using formats like "replace A with B", "add X", "remove Y"

"""


class ImageEditPromptGenerator:
    """Image editing prompt generator"""

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize model and processor

        Args:
            model_path: Model path, if None, use the path from project.py
            device: Device, if None, automatically select
        """
        if model_path is None:
            model_path = prj.QWEN2_5_VL_7B_INSTUCT_MODEL_PATH

        print(f"Loading model: {model_path}")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto"
        )

        self.processor = AutoProcessor.from_pretrained(model_path)
        self.device = device if device else self.model.device

        print(f"Model loaded successfully, device: {self.device}")

    def generate_editing_prompts(
        self,
        image_path: Union[str, Path, Image.Image],
        edit_request: str,
        image_mask_path: Optional[Union[str, Path, Image.Image]] = None,
        target_path: Optional[Union[str, Path, Image.Image]] = None,
        target_mask_path: Optional[Union[str, Path, Image.Image]] = None,
        max_new_tokens: int = 512
    ) -> Dict:
        """
        Generate editing prompt information for images, supports simultaneous input of source, mask, target and target_mask images

        Args:
            image_path: Source image path or PIL Image object (required)
            edit_request: Editing request text
            image_mask_path: Source image mask path or PIL Image object (optional), white areas indicate parts to be edited
            target_path: Target image path or PIL Image object (optional), expected result after editing
            target_mask_path: Target image mask path or PIL Image object (optional), indicates edited region
            max_new_tokens: Maximum number of tokens to generate

        Returns:
            Dictionary containing the following fields (conforming to sample_temple_en.json format):
            - sample_id: Auto-generated unique ID
            - instruction: Counterfactual instruction rewritten from semantic assumptions/conditional changes
            - reasoning_chains: Reasoning chains object (contains descriptive, causal, comparative)
            - counterfactual_premise: Counterfactual premise (contains changed_factor, original_state, counterfactual_state, causal_effect)
            - multi_modal_constraints: Multi-modal constraint list (contains type and description)
            - edit_metadata: Edit metadata object (contains edit_subject, new_subject, edit_type, complexity_level)
            - editing_instruction: Detailed editing command
            - raw_response: Raw model response
        """
        # Load source image
        if isinstance(image_path, (str, Path)):
            source_image = Image.open(image_path).convert("RGB")
        else:
            source_image = image_path

        # Load source image mask (if provided)
        source_mask_image = None
        if image_mask_path is not None:
            if isinstance(image_mask_path, (str, Path)):
                source_mask_image = Image.open(image_mask_path).convert("RGB")
            else:
                source_mask_image = image_mask_path

        # Load target image (if provided)
        target_image = None
        if target_path is not None:
            if isinstance(target_path, (str, Path)):
                target_image = Image.open(target_path).convert("RGB")
            else:
                target_image = target_path

        # Load target image mask (if provided)
        target_mask_image = None
        if target_mask_path is not None:
            if isinstance(target_mask_path, (str, Path)):
                target_mask_image = Image.open(target_mask_path).convert("RGB")
            else:
                target_mask_image = target_mask_path

        # Build prompt
        prompt = create_editing_prompt(
            edit_request,
            has_source=True,
            has_mask=source_mask_image is not None,
            has_target=target_image is not None,
            has_target_mask=target_mask_image is not None
        )

        # Build message content list
        content = []

        # Add source image
        content.append({"type": "image", "image": source_image})

        # Add source image mask (if provided)
        if source_mask_image is not None:
            content.append({"type": "image", "image": source_mask_image})

        # Add target image (if provided)
        if target_image is not None:
            content.append({"type": "image", "image": target_image})

        # Add target image mask (if provided)
        if target_mask_image is not None:
            content.append({"type": "image", "image": target_mask_image})

        # Add text prompt
        content.append({"type": "text", "text": prompt})

        # Build message
        messages = [
            {
                "role": "user",
                "content": content
            }
        ]

        try:
            # Prepare inference input
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

            # Generate output
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

            # Parse JSON response
            result = self._parse_response(output_text, edit_request)
            result['raw_response'] = output_text

            # Auto-generate sample_id if not present
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
        """Parse model response and extract JSON information (conforming to sample_temple_en.json format)"""
        # Initialize result structure, conforming to sample_temple_en.json format
        result = {
            'sample_id': '',
            'instruction': '',
            'reasoning_chains': {
                'descriptive': '',
                'causal': '',
                'comparative': ''
            },
            'counterfactual_premise': {
                'changed_factor': '',
                'original_state': '',
                'counterfactual_state': '',
                'causal_effect': ''
            },
            'multi_modal_constraints': [],
            'edit_metadata': {
                'edit_subject': [],
                'new_subject': [],
                'edit_type': '',
                'complexity_level': ''
            },
            'editing_instruction': ''
        }

        # Try to extract JSON
        try:
            # First, try to extract JSON from code blocks (```json ... ```)
            json_str = None
            if '```json' in response:
                start_marker = response.find('```json')
                end_marker = response.find('```', start_marker + 7)
                if end_marker > start_marker:
                    json_str = response[start_marker + 7:end_marker].strip()
            elif '```' in response:
                # Try generic code block
                start_marker = response.find('```')
                end_marker = response.find('```', start_marker + 3)
                if end_marker > start_marker:
                    potential_json = response[start_marker + 3:end_marker].strip()
                    # Check if it looks like JSON (starts with {)
                    if potential_json.startswith('{'):
                        json_str = potential_json

            # If no code block found, try to find JSON directly
            if json_str is None:
                start_idx = response.find('{')
                end_idx = response.rfind('}') + 1
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = response[start_idx:end_idx]

            if json_str:
                # Clean up the JSON string (remove any trailing commas, etc.)
                json_str = json_str.strip()
                # Try to parse
                parsed = json.loads(json_str)

                # Update result, conforming to new format
                result.update({
                    'sample_id': parsed.get('sample_id', ''),
                    'instruction': parsed.get('instruction', edit_request),
                    'counterfactual_premise': parsed.get('counterfactual_premise', result['counterfactual_premise']),
                    'multi_modal_constraints': parsed.get('multi_modal_constraints', []),
                    'editing_instruction': parsed.get('editing_instruction', '')
                })

                # Handle reasoning_chains (new format only)
                reasoning_chains = parsed.get('reasoning_chains', {})
                if isinstance(reasoning_chains, dict):
                    result['reasoning_chains'] = {
                        'descriptive': reasoning_chains.get('descriptive', ''),
                        'causal': reasoning_chains.get('causal', ''),
                        'comparative': reasoning_chains.get('comparative', '')
                    }
                else:
                    result['reasoning_chains'] = {
                        'descriptive': '',
                        'causal': '',
                        'comparative': ''
                    }

                # Handle edit_metadata (new format only)
                edit_metadata = parsed.get('edit_metadata', {})
                if isinstance(edit_metadata, dict):
                    result['edit_metadata'] = {
                        'edit_subject': edit_metadata.get('edit_subject', []),
                        'new_subject': edit_metadata.get('new_subject', []),
                        'edit_type': edit_metadata.get('edit_type', ''),
                        'complexity_level': edit_metadata.get('complexity_level', '')
                    }
                else:
                    result['edit_metadata'] = {
                        'edit_subject': [],
                        'new_subject': [],
                        'edit_type': '',
                        'complexity_level': ''
                    }

                # Ensure counterfactual_premise structure is complete
                if isinstance(result['counterfactual_premise'], dict):
                    result['counterfactual_premise'] = {
                        'changed_factor': result['counterfactual_premise'].get('changed_factor', ''),
                        'original_state': result['counterfactual_premise'].get('original_state', ''),
                        'counterfactual_state': result['counterfactual_premise'].get('counterfactual_state', ''),
                        'causal_effect': result['counterfactual_premise'].get('causal_effect', '')
                    }

                # Ensure multi_modal_constraints is in list format
                if not isinstance(result['multi_modal_constraints'], list):
                    result['multi_modal_constraints'] = []

                # Ensure edit_metadata structure is complete
                if not isinstance(result['edit_metadata'], dict):
                    result['edit_metadata'] = {
                        'edit_subject': [],
                        'new_subject': [],
                        'edit_type': '',
                        'complexity_level': ''
                    }
                else:
                    result['edit_metadata'] = {
                        'edit_subject': result['edit_metadata'].get('edit_subject', []),
                        'new_subject': result['edit_metadata'].get('new_subject', []),
                        'edit_type': result['edit_metadata'].get('edit_type', ''),
                        'complexity_level': result['edit_metadata'].get('complexity_level', '')
                    }
            else:
                # If JSON not found, use edit request as base information
                result['instruction'] = edit_request
                result['reasoning_chains'] = {
                    'descriptive': response[:500],
                    'causal': response[:500],
                    'comparative': response[:500]
                }

        except json.JSONDecodeError as e:
            # JSON parsing failed, try to fix common issues
            print(f"JSON parsing failed, try to fix common issues: {e}")
            try:
                # Try to fix trailing commas and other common JSON issues
                if json_str:
                    # Remove trailing commas before closing braces/brackets
                    fixed_json = re.sub(r',(\s*[}\]])', r'\1', json_str)
                    # Try parsing again
                    parsed = json.loads(fixed_json)
                    # If successful, process as normal
                    result.update({
                        'sample_id': parsed.get('sample_id', ''),
                        'instruction': parsed.get('instruction', edit_request),
                        'counterfactual_premise': parsed.get('counterfactual_premise', result['counterfactual_premise']),
                        'multi_modal_constraints': parsed.get('multi_modal_constraints', []),
                        'editing_instruction': parsed.get('editing_instruction', '')
                    })

                    reasoning_chains = parsed.get('reasoning_chains', {})
                    if isinstance(reasoning_chains, dict):
                        result['reasoning_chains'] = {
                            'descriptive': reasoning_chains.get('descriptive', ''),
                            'causal': reasoning_chains.get('causal', ''),
                            'comparative': reasoning_chains.get('comparative', '')
                        }
                    else:
                        result['reasoning_chains'] = {
                            'descriptive': '',
                            'causal': '',
                            'comparative': ''
                        }

                    edit_metadata = parsed.get('edit_metadata', {})
                    if isinstance(edit_metadata, dict):
                        result['edit_metadata'] = {
                            'edit_subject': edit_metadata.get('edit_subject', []),
                            'new_subject': edit_metadata.get('new_subject', []),
                            'edit_type': edit_metadata.get('edit_type', ''),
                            'complexity_level': edit_metadata.get('complexity_level', '')
                        }
                    else:
                        result['edit_metadata'] = {
                            'edit_subject': [],
                            'new_subject': [],
                            'edit_type': '',
                            'complexity_level': ''
                        }

                    # Ensure structures are complete
                    if isinstance(result['counterfactual_premise'], dict):
                        result['counterfactual_premise'] = {
                            'changed_factor': result['counterfactual_premise'].get('changed_factor', ''),
                            'original_state': result['counterfactual_premise'].get('original_state', ''),
                            'counterfactual_state': result['counterfactual_premise'].get('counterfactual_state', ''),
                            'causal_effect': result['counterfactual_premise'].get('causal_effect', '')
                        }

                    if not isinstance(result['multi_modal_constraints'], list):
                        result['multi_modal_constraints'] = []

                    if not isinstance(result['edit_metadata'], dict):
                        result['edit_metadata'] = {
                            'edit_subject': [],
                            'new_subject': [],
                            'edit_type': '',
                            'complexity_level': ''
                        }
                    else:
                        result['edit_metadata'] = {
                            'edit_subject': result['edit_metadata'].get('edit_subject', []),
                            'new_subject': result['edit_metadata'].get('new_subject', []),
                            'edit_type': result['edit_metadata'].get('edit_type', ''),
                            'complexity_level': result['edit_metadata'].get('complexity_level', '')
                        }
                else:
                    raise e
            except (json.JSONDecodeError, Exception) as e2:
                # If fixing failed, use edit request as base information
                print(f"JSON parsing warning: {e} (fix attempt also failed: {e2})")
                result['instruction'] = edit_request
                result['reasoning_chains'] = {
                    'descriptive': response[:500],
                    'causal': response[:500],
                    'comparative': response[:500]
                }

        return result

    def process_batch(
        self,
        image_data_list: List[Dict],
        output_json_file: Optional[Union[str, Path]] = None,
        output_json_path: Optional[Union[str, Path]] = None,
        save_individual: bool = False
    ) -> List[Dict]:
        """
        Batch process images and prompts, supports multi-image input

        Args:
            image_data_list: List of image data dictionaries, each dictionary should contain:
                - sample_id: Sample unique identifier (optional)
                - image_path: Source image path (required)
                - instruction or edit_request or prompt: Editing instruction (required)
                - image_mask_path or mask_path: Source image mask path (optional)
                - target_path: Target image path (optional)
                - target_mask_path: Target image mask path (optional)
            output_json_file: Output JSON file path, if None then don't save
            output_json_path: Output directory for individual files
            save_individual: Whether to save individual files for each result

        Returns:
            Result list, each element is a dictionary containing generated information
        """
        results = []

        if output_json_path and save_individual:
            output_json_path = Path(output_json_path)
            output_json_path.mkdir(parents=True, exist_ok=True)

        for idx, data_item in enumerate(tqdm(image_data_list, desc="Processing images")):
            try:
                # Extract information from dictionary
                sample_id = data_item.get('sample_id', f'sample_{idx:05d}')
                image_path = data_item.get('image_path') or data_item.get('image')
                edit_request = data_item.get('instruction') or data_item.get('edit_request') or data_item.get('prompt')
                image_mask_path = data_item.get('image_mask_path') or data_item.get('mask_path') or data_item.get('mask')
                target_path = data_item.get('target_path') or data_item.get('target') or data_item.get('target_image')
                target_mask_path = data_item.get('target_mask_path') or data_item.get('target_mask')

                if not image_path or not edit_request:
                    print(f"\nSkipping sample {idx}: missing image_path or instruction")
                    continue

                result = self.generate_editing_prompts(
                    image_path,
                    edit_request,
                    image_mask_path=image_mask_path,
                    target_path=target_path,
                    target_mask_path=target_mask_path
                )

                # Preserve information from original data
                result['sample_id'] = sample_id
                result['image_path'] = str(image_path)
                result['edit_request'] = edit_request
                if image_mask_path:
                    result['image_mask_path'] = str(image_mask_path)
                if target_path:
                    result['target_path'] = str(target_path)
                if target_mask_path:
                    result['target_mask_path'] = str(target_mask_path)
                result['index'] = idx

                results.append(result)

                # Save individual file, using sample_id as filename
                if save_individual and output_json_path:
                    output_json_file_individual = output_json_path / f"{sample_id}.json"
                    with open(output_json_file_individual, 'w', encoding='utf-8') as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)

            except Exception as e:
                print(f"\nError processing sample {idx}: {e}")
                error_result = {
                    'error': str(e),
                    'sample_id': data_item.get('sample_id', f'sample_{idx:05d}'),
                    'image_path': str(data_item.get('image_path', '')),
                    'edit_request': str(data_item.get('instruction') or data_item.get('edit_request', '')),
                    'index': idx
                }
                results.append(error_result)

        # Save summary file
        if output_json_file:
            output_json_file = Path(output_json_file)
            output_json_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_json_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\nResults saved to: {output_json_file}")

        return results

    def process_from_json(
        self,
        input_json_path: Union[str, Path, Dict],
        input_image_path: Optional[Union[str, Path]] = None,
        output_json_file: Optional[Union[str, Path]] = None,
        output_json_path: Optional[Union[str, Path]] = None,
        save_individual: bool = False
    ) -> List[Dict]:
        """
        Read image paths and prompts from JSON file, batch process

        Args:
            input_json_path: JSON file path or dictionary, format should be:
                [
                    {
                        "sample_id": "sample_00000",
                        "image_path": "path/to/image.jpg",
                        "image_mask_path": "path/to/mask.png",
                        "target_path": "path/to/target.jpg",
                        "target_mask_path": "path/to/target_mask.png",
                        "instruction": "Editing request"
                    },
                    ...
                ]
            input_image_path: Base directory for images, if paths in JSON are relative
            output_json_file: Output file path
            output_json_path: Output directory for individual files
            save_individual: Whether to save individual files for each result
        Returns:
            Result list
        """
        # Read JSON
        if isinstance(input_json_path, (str, Path)):
            with open(input_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            raise ValueError("input_json_path must be a string or Path")

        # Build image data dictionary list
        image_data_list = []
        if input_image_path:
            input_image_path = Path(input_image_path)

        for item in data:
            # Extract all fields
            sample_id = item.get('sample_id', '')
            image_path = item.get('image_path') or item.get('image')
            edit_request = item.get('edit_request') or item.get('prompt') or item.get('instruction')
            # Support multiple mask path field names
            image_mask_path = item.get('image_mask_path') or item.get('mask_path') or item.get('mask')
            target_path = item.get('target_path') or item.get('target') or item.get('target_image')
            target_mask_path = item.get('target_mask_path') or item.get('target_mask')

            if not image_path or not edit_request:
                continue

            # Handle relative paths
            if input_image_path and not Path(image_path).is_absolute():
                image_path = str(input_image_path / image_path)
            else:
                image_path = str(Path(image_path))

            # Handle source image mask path
            if image_mask_path:
                if input_image_path and not Path(image_mask_path).is_absolute():
                    image_mask_path = str(input_image_path / image_mask_path)
                else:
                    image_mask_path = str(Path(image_mask_path))

            # Handle target path
            if target_path:
                if input_image_path and not Path(target_path).is_absolute():
                    target_path = str(input_image_path / target_path)
                else:
                    target_path = str(Path(target_path))

            # Handle target_mask path
            if target_mask_path:
                if input_image_path and not Path(target_mask_path).is_absolute():
                    target_mask_path = str(input_image_path / target_mask_path)
                else:
                    target_mask_path = str(Path(target_mask_path))

            # Build data dictionary
            data_dict = {
                'sample_id': sample_id,
                'image_path': image_path,
                'instruction': edit_request
            }

            if image_mask_path:
                data_dict['image_mask_path'] = image_mask_path
            if target_path:
                data_dict['target_path'] = target_path
            if target_mask_path:
                data_dict['target_mask_path'] = target_mask_path

            image_data_list.append(data_dict)

        # Batch process
        return self.process_batch(
            image_data_list,
            output_json_file=output_json_file,
            output_json_path=output_json_path,
            save_individual=save_individual
        )


def main():
    """Example usage"""
    import argparse

    images_dir = prj.HALLUSEGBENCH_DATASET_PATH
    factual_images_dir = os.path.join(images_dir, "factual_images")
    factual_images = os.listdir(factual_images_dir)
    factual_images_paths = [os.path.join(factual_images_dir, image) for image in factual_images]
    # factual_images = [Image.open(image_path) for image_path in factual_images_paths]

    counterfactual_images_dir = os.path.join(images_dir, "counterfactual_images")
    counterfactual_images = os.listdir(counterfactual_images_dir)
    counterfactual_images_paths = [os.path.join(counterfactual_images_dir, image) for image in counterfactual_images]
    # counterfactual_images = [Image.open(image_path) for image_path in counterfactual_images_paths]

    # converted_data_json = os.path.join(prj.HALLUSEGBENCH_DATASET_PATH, "converted_data.json")
    # prompts_output_dir = prj.PROMPTS_OUTPUT_DIR
    converted_data_json = os.path.join(prj.HALLUSEGGOOD_DATASET_PATH, "halluseggood_data.json")
    prompts_output_dir = prj.PROMPTS_OUTPUT_HALLUSEGGOOD_DIR

    parser = argparse.ArgumentParser(description='Generate image editing prompts')

    parser.add_argument('--input_image_path', type=str, default=images_dir, help='Base directory for images (for relative paths in JSON)')
    parser.add_argument('--input_json_path', type=str, default=converted_data_json, help='JSON file path for batch processing')
    parser.add_argument('--output_json_path', type=str, default=prompts_output_dir, help='Output directory for individual files')

    parser.add_argument('--image', type=str, default="factual_images/COCO_train2014_000000000154.jpg", help='Source image path (required)')
    parser.add_argument('--mask', type=str, default=None, help='Source image mask path (optional), white areas indicate parts to be edited')
    parser.add_argument('--target', type=str, default=None, help='counterfactual_images/COCO_train2014_000000000154_590410.jpg')
    parser.add_argument('--target_mask', type=str, default=None, help='Target image mask path (optional), indicates edited region')
    parser.add_argument('--prompt', type=str, default="Change zebra at the bottom to cow.", help='Editing prompt')
    parser.add_argument('--output_json_file', type=str, default='output_json_file.json', help='Output file path')

    parser.add_argument('--save_individual', action='store_true', default=True, help='Save individual files for each result')

    args = parser.parse_args()

    # Initialize generator
    generator = ImageEditPromptGenerator()

    # args.input_json_path = None
    if args.input_json_path:
        # Batch processing
        print(f"Batch processing from JSON file: {args.input_json_path}")
        results = generator.process_from_json(
            input_json_path = args.input_json_path,
            input_image_path=args.input_image_path,
            output_json_file=None,
            output_json_path=args.output_json_path,
            save_individual=args.save_individual
        )
        print(f"\nProcessing completed, {len(results)} results in total")

    elif args.image and args.prompt:
        # Single image processing
        image_path = os.path.join(args.input_image_path, args.image) if args.input_image_path else args.image
        image_mask_path = os.path.join(args.input_image_path, args.mask) if args.mask and args.input_image_path else args.mask
        target_path = os.path.join(args.input_image_path, args.target) if args.target and args.input_image_path else args.target
        target_mask_path = os.path.join(args.input_image_path, args.target_mask) if args.target_mask and args.input_image_path else args.target_mask

        print(f"Processing image: {image_path}")
        if image_mask_path:
            print(f"  Source image mask: {image_mask_path}")
        if target_path:
            print(f"  Target image: {target_path}")
        if target_mask_path:
            print(f"  Target image mask: {target_mask_path}")

        result = generator.generate_editing_prompts(
            image_path,
            args.prompt,
            image_mask_path=image_mask_path,
            target_path=target_path,
            target_mask_path=target_mask_path
        )
        print("\nResult:")
        print(json.dumps(result, ensure_ascii=False, indent=2))

        # Save result
        with open(args.output_json_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\nResult saved to: {args.output_json_file}")

    else:
        print("Please provide --image and --prompt or --input_json_path arguments")
        parser.print_help()


if __name__ == "__main__":
    main()

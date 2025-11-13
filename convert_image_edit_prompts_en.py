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
        image_context.append(f"Image {image_idx} is the binary mask of the original image (source mask image), a black-and-white image where white areas indicate the parts to be edited and black areas represent the background that should remain unchanged")
        image_idx += 1
    if has_target:
        image_context.append(f"Image {image_idx} is the edited target image (target image)")
        image_idx += 1
    if has_target_mask:
        image_context.append(f"Image {image_idx} is the mask of the target image (target mask image), indicating the edited region")

    image_context_str = ". ".join(image_context) + "." if image_context else ""

    return f"""
Generate structured image editing annotations based on this instruction:

{image_context_str}

Instruction: "{instruction}"

Output a valid JSON object with the following structure. Start with {{ and end with }}. Do NOT use markdown code blocks.

{{
    "sample_id": "unique_id",
    "image_name": "filename.jpg",
    "instruction": "Rewrite the instruction from a personal perspective or contextual need that naturally leads to this editing. Example: 'I need more storage space in my bedroom. How can I optimize it?'",
    "reasoning_chains": [
        {{"type": "descriptive", "chain": "Describe what you see in the image and the transformation needed"}},
        {{"type": "causal", "chain": "Explain why this change is needed and what effects it will have"}},
        {{"type": "comparative", "chain": "Compare the original and target states, identify differences"}}
    ],
    "counterfactual_premise": {{
        "aspect": "What was changed",
        "from": "Original state",
        "to": "New state"
    }},
    "multi_modal_constraints": [
        {{"type": "spatial_layout", "description": "Spatial constraints"}},
        {{"type": "semantic_content", "description": "Semantic constraints"}},
        {{"type": "physical_causal", "description": "Physical constraints"}},
        {{"type": "temporal_reasoning", "description": "Temporal constraints"}}
    ],
    "edit_subject": ["object1"],
    "new_subject": ["object2"],
    "edit_type": "object_replacement",
    "complexity_level": "simple",
    "editing_instruction": "Direct editing command"
}}

Important:
- All fields must be filled with meaningful content
- reasoning_chains.chain must be plain text, NOT JSON
- instruction should be rewritten from personal/contextual perspective
- edit_type: object_replacement|addition|removal|modification|transformation|combination|deletion|rearrangement|temporal_evolution
- complexity_level: simple|medium|complex

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
        max_new_tokens: int = 2048,
        sample_id: Optional[str] = None
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
            - image_name: Image filename (extracted from factual image path)
            - instruction: Counterfactual instruction rewritten from semantic assumptions/conditional changes
            - reasoning_chains: Array of reasoning chain objects, each with "type" and "chain" fields
            - counterfactual_premise: Counterfactual premise object (contains aspect, from, to)
            - multi_modal_constraints: Multi-modal constraint list (contains type and description)
            - edit_subject: List of editing objects
            - new_subject: List of new objects
            - edit_type: Editing type
            - complexity_level: Complexity level
            - editing_instruction: Detailed editing command
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
            image_path_str = str(image_path) if isinstance(image_path, (str, Path)) else ''
            result = self._parse_response(output_text, edit_request, sample_id=sample_id, image_path=image_path_str)

            # Auto-generate sample_id if not present
            if not result.get('sample_id'):
                result['sample_id'] = str(uuid.uuid4())

            return result

        except Exception as e:
            return {
                'error': str(e),
                'edit_request': edit_request
            }

    def _parse_response(self, response: str, edit_request: str, sample_id: Optional[str] = None, image_path: Optional[str] = None) -> Dict:
        """Parse model response and extract JSON information (conforming to sample_temple_en.json format)"""
        # Initialize result structure, conforming to sample_temple_en.json format
        result = {
            'sample_id': '',
            'image_name': '',
            'instruction': '',
            'reasoning_chains': [],
            'counterfactual_premise': {
                'aspect': '',
                'from': '',
                'to': ''
            },
            'multi_modal_constraints': [],
            'edit_subject': [],
            'new_subject': [],
            'edit_type': '',
            'complexity_level': '',
            'editing_instruction': ''
        }

        # Extract image_name from image_path if provided
        if image_path:
            image_name = Path(image_path).name
            result['image_name'] = image_name

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

            # If no code block found, try to find JSON directly using balanced braces
            if json_str is None:
                start_idx = response.find('{')
                if start_idx >= 0:
                    # Find the matching closing brace by counting braces
                    brace_count = 0
                    end_idx = start_idx
                    for i in range(start_idx, len(response)):
                        if response[i] == '{':
                            brace_count += 1
                        elif response[i] == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                end_idx = i + 1
                                break
                    if end_idx > start_idx:
                        json_str = response[start_idx:end_idx]

            if json_str:
                # Clean up the JSON string
                json_str = json_str.strip()
                # Remove any nested code block markers that might be in string values
                # This is a simple approach - replace ```json and ``` with escaped versions in string contexts
                # But we need to be careful not to break valid JSON

                # Try to parse
                try:
                    parsed = json.loads(json_str)
                except json.JSONDecodeError as parse_err:
                    # Try to fix common issues
                    # Remove trailing commas
                    fixed_json = re.sub(r',(\s*[}\]])', r'\1', json_str)
                    # Try to fix unclosed strings (basic attempt)
                    # Remove any ```json or ``` markers that appear in the middle (likely in string values)
                    fixed_json = re.sub(r'```json\s*', '', fixed_json)
                    fixed_json = re.sub(r'```\s*', '', fixed_json)
                    # Try parsing again
                    try:
                        parsed = json.loads(fixed_json)
                    except json.JSONDecodeError:
                        # If still fails, try to extract partial data
                        raise parse_err

                # Update result, conforming to new format
                # Ensure image_name is extracted from path, not from parsed JSON (which might be "Image 1" etc.)
                parsed_image_name = parsed.get('image_name', '')
                if image_path:
                    # Always prefer the extracted name from path
                    final_image_name = Path(image_path).name
                elif parsed_image_name and parsed_image_name not in ['Image 1', 'Image 2', 'Image 3', 'Image 4']:
                    # Use parsed name only if it's not a generic "Image N" value
                    final_image_name = parsed_image_name
                else:
                    final_image_name = result.get('image_name', '')

                result.update({
                    'sample_id': parsed.get('sample_id', '') or result.get('sample_id', ''),
                    'image_name': final_image_name,
                    'instruction': parsed.get('instruction', edit_request),
                    'multi_modal_constraints': parsed.get('multi_modal_constraints', []),
                    'edit_subject': parsed.get('edit_subject', []),
                    'new_subject': parsed.get('new_subject', []),
                    'edit_type': parsed.get('edit_type', ''),
                    'complexity_level': parsed.get('complexity_level', ''),
                    'editing_instruction': parsed.get('editing_instruction', '')
                })

                # Handle reasoning_chains (new format: array of objects with type and chain)
                reasoning_chains = parsed.get('reasoning_chains', [])
                if isinstance(reasoning_chains, list):
                    def clean_reasoning_text(text):
                        if not isinstance(text, str):
                            return text
                        # Remove code block markers
                        text = re.sub(r'```json\s*', '', text)
                        text = re.sub(r'```\s*', '', text)

                        # Check if the text contains nested JSON structure
                        # If it starts with { and contains "sample_id" or "reasoning_chains", it's likely nested JSON
                        text_stripped = text.strip()
                        if text_stripped.startswith('{') and ('"sample_id"' in text or '"reasoning_chains"' in text or "'sample_id'" in text):
                            # Try to extract the actual chain content from nested JSON
                            try:
                                # Try to parse as JSON first
                                nested_json = json.loads(text)
                                # If it's a nested JSON, try to extract chain from reasoning_chains
                                if isinstance(nested_json, dict) and 'reasoning_chains' in nested_json:
                                    chains = nested_json.get('reasoning_chains', [])
                                    if isinstance(chains, list) and len(chains) > 0:
                                        # Try to find the matching chain type
                                        for chain_obj in chains:
                                            if isinstance(chain_obj, dict) and 'chain' in chain_obj:
                                                return clean_reasoning_text(chain_obj['chain'])
                                # If we can't extract, return empty or a fallback
                                return ""
                            except (json.JSONDecodeError, Exception):
                                # If parsing fails, try to extract text between quotes or after "chain":
                                # Look for pattern like "chain": "actual text here"
                                match = re.search(r'"chain"\s*:\s*"([^"]*(?:\\.[^"]*)*)"', text, re.DOTALL)
                                if match:
                                    return match.group(1).replace('\\"', '"').replace('\\n', '\n')
                                # If no match, try to find text after "chain":
                                match = re.search(r'"chain"\s*:\s*"([^"]+)"', text)
                                if match:
                                    return match.group(1)
                                # Last resort: remove all JSON structure markers
                                lines = text.split('\n')
                                cleaned_lines = []
                                in_quotes = False
                                for line in lines:
                                    stripped = line.strip()
                                    # Skip JSON structure lines
                                    if stripped.startswith('{') or stripped.startswith('}') or stripped.startswith('[') or stripped.startswith(']'):
                                        continue
                                    # Skip lines with JSON keys
                                    if re.match(r'^\s*"[^"]+"\s*:', line):
                                        continue
                                    cleaned_lines.append(line)
                                result = '\n'.join(cleaned_lines).strip()
                                # Remove any remaining JSON artifacts
                                result = re.sub(r'^\s*[{\[\]}]\s*', '', result)
                                return result

                        # Remove any incomplete JSON structures that might be embedded
                        lines = text.split('\n')
                        cleaned_lines = []
                        for line in lines:
                            # Skip lines that are just JSON structure markers
                            stripped = line.strip()
                            if stripped in ['{', '}', '[', ']'] or (stripped.startswith('{') and not any(c in line for c in ['"', "'"])):
                                continue
                            # Skip lines that are JSON keys
                            if re.match(r'^\s*"[^"]+"\s*:', line):
                                continue
                            cleaned_lines.append(line)
                        return '\n'.join(cleaned_lines).strip()

                    cleaned_chains = []
                    for chain_item in reasoning_chains:
                        if isinstance(chain_item, dict):
                            chain_type = chain_item.get('type', '')
                            chain_text = clean_reasoning_text(chain_item.get('chain', ''))
                            cleaned_chains.append({
                                'type': chain_type,
                                'chain': chain_text
                            })
                    result['reasoning_chains'] = cleaned_chains
                elif isinstance(reasoning_chains, dict):
                    # Backward compatibility: convert old dict format to new array format
                    def clean_reasoning_text(text):
                        if not isinstance(text, str):
                            return text
                        # Remove code block markers
                        text = re.sub(r'```json\s*', '', text)
                        text = re.sub(r'```\s*', '', text)

                        # Check if the text contains nested JSON structure
                        text_stripped = text.strip()
                        if text_stripped.startswith('{') and ('"sample_id"' in text or '"reasoning_chains"' in text or "'sample_id'" in text):
                            try:
                                nested_json = json.loads(text)
                                if isinstance(nested_json, dict) and 'reasoning_chains' in nested_json:
                                    chains = nested_json.get('reasoning_chains', [])
                                    if isinstance(chains, list) and len(chains) > 0:
                                        for chain_obj in chains:
                                            if isinstance(chain_obj, dict) and 'chain' in chain_obj:
                                                return clean_reasoning_text(chain_obj['chain'])
                                return ""
                            except (json.JSONDecodeError, Exception):
                                match = re.search(r'"chain"\s*:\s*"([^"]*(?:\\.[^"]*)*)"', text, re.DOTALL)
                                if match:
                                    return match.group(1).replace('\\"', '"').replace('\\n', '\n')
                                lines = text.split('\n')
                                cleaned_lines = []
                                for line in lines:
                                    stripped = line.strip()
                                    if stripped in ['{', '}', '[', ']'] or (stripped.startswith('{') and not any(c in line for c in ['"', "'"])):
                                        continue
                                    if re.match(r'^\s*"[^"]+"\s*:', line):
                                        continue
                                    cleaned_lines.append(line)
                                result = '\n'.join(cleaned_lines).strip()
                                result = re.sub(r'^\s*[{\[\]}]\s*', '', result)
                                return result

                        lines = text.split('\n')
                        cleaned_lines = []
                        for line in lines:
                            stripped = line.strip()
                            if stripped in ['{', '}', '[', ']'] or (stripped.startswith('{') and not any(c in line for c in ['"', "'"])):
                                continue
                            if re.match(r'^\s*"[^"]+"\s*:', line):
                                continue
                            cleaned_lines.append(line)
                        return '\n'.join(cleaned_lines).strip()

                    result['reasoning_chains'] = [
                        {'type': 'descriptive', 'chain': clean_reasoning_text(reasoning_chains.get('descriptive', ''))},
                        {'type': 'causal', 'chain': clean_reasoning_text(reasoning_chains.get('causal', ''))},
                        {'type': 'comparative', 'chain': clean_reasoning_text(reasoning_chains.get('comparative', ''))}
                    ]
                else:
                    result['reasoning_chains'] = []

                # Handle counterfactual_premise (new simplified format)
                counterfactual_premise = parsed.get('counterfactual_premise', {})
                if isinstance(counterfactual_premise, dict):
                    result['counterfactual_premise'] = {
                        'aspect': counterfactual_premise.get('aspect', ''),
                        'from': counterfactual_premise.get('from', ''),
                        'to': counterfactual_premise.get('to', '')
                    }
                else:
                    result['counterfactual_premise'] = {
                        'aspect': '',
                        'from': '',
                        'to': ''
                    }

                # Ensure multi_modal_constraints is in list format
                if not isinstance(result['multi_modal_constraints'], list):
                    result['multi_modal_constraints'] = []
            else:
                # If JSON not found, use edit request as base information
                result['instruction'] = edit_request
                result['reasoning_chains'] = [
                    {'type': 'descriptive', 'chain': response[:500]},
                    {'type': 'causal', 'chain': response[:500]},
                    {'type': 'comparative', 'chain': response[:500]}
                ]

        except json.JSONDecodeError as e:
            # JSON parsing failed, try to fix common issues
            sample_info = f" (sample_id: {sample_id})" if sample_id else ""
            print(f"JSON parsing failed{sample_info}, trying to fix: {e}")
            try:
                # Try to fix common JSON issues
                if json_str:
                    # Remove trailing commas before closing braces/brackets
                    fixed_json = re.sub(r',(\s*[}\]])', r'\1', json_str)
                    # Remove code block markers that might be in string values
                    fixed_json = re.sub(r'```json\s*', '', fixed_json)
                    fixed_json = re.sub(r'```\s*', '', fixed_json)
                    # Try to fix unclosed strings by finding and closing them
                    # This is a basic attempt - find strings that aren't properly closed
                    # Try parsing again
                    parsed = json.loads(fixed_json)
                    # If successful, process as normal (same logic as above)
                    # Ensure image_name is extracted from path, not from parsed JSON
                    parsed_image_name = parsed.get('image_name', '')
                    if image_path:
                        final_image_name = Path(image_path).name
                    elif parsed_image_name and parsed_image_name not in ['Image 1', 'Image 2', 'Image 3', 'Image 4']:
                        final_image_name = parsed_image_name
                    else:
                        final_image_name = result.get('image_name', '')

                    result.update({
                        'sample_id': parsed.get('sample_id', '') or result.get('sample_id', ''),
                        'image_name': final_image_name,
                        'instruction': parsed.get('instruction', edit_request),
                        'multi_modal_constraints': parsed.get('multi_modal_constraints', []),
                        'edit_subject': parsed.get('edit_subject', []),
                        'new_subject': parsed.get('new_subject', []),
                        'edit_type': parsed.get('edit_type', ''),
                        'complexity_level': parsed.get('complexity_level', ''),
                        'editing_instruction': parsed.get('editing_instruction', '')
                    })

                    # Handle reasoning_chains (new format: array of objects with type and chain)
                    reasoning_chains = parsed.get('reasoning_chains', [])
                    if isinstance(reasoning_chains, list):
                        def clean_reasoning_text(text):
                            if not isinstance(text, str):
                                return text
                            # Remove code block markers
                            text = re.sub(r'```json\s*', '', text)
                            text = re.sub(r'```\s*', '', text)

                            # Check if the text contains nested JSON structure
                            text_stripped = text.strip()
                            if text_stripped.startswith('{') and ('"sample_id"' in text or '"reasoning_chains"' in text or "'sample_id'" in text):
                                try:
                                    nested_json = json.loads(text)
                                    if isinstance(nested_json, dict) and 'reasoning_chains' in nested_json:
                                        chains = nested_json.get('reasoning_chains', [])
                                        if isinstance(chains, list) and len(chains) > 0:
                                            for chain_obj in chains:
                                                if isinstance(chain_obj, dict) and 'chain' in chain_obj:
                                                    return clean_reasoning_text(chain_obj['chain'])
                                    return ""
                                except (json.JSONDecodeError, Exception):
                                    match = re.search(r'"chain"\s*:\s*"([^"]*(?:\\.[^"]*)*)"', text, re.DOTALL)
                                    if match:
                                        return match.group(1).replace('\\"', '"').replace('\\n', '\n')
                                    lines = text.split('\n')
                                    cleaned_lines = []
                                    for line in lines:
                                        stripped = line.strip()
                                        if stripped in ['{', '}', '[', ']'] or (stripped.startswith('{') and not any(c in line for c in ['"', "'"])):
                                            continue
                                        if re.match(r'^\s*"[^"]+"\s*:', line):
                                            continue
                                        cleaned_lines.append(line)
                                    result = '\n'.join(cleaned_lines).strip()
                                    result = re.sub(r'^\s*[{\[\]}]\s*', '', result)
                                    return result

                            lines = text.split('\n')
                            cleaned_lines = []
                            for line in lines:
                                stripped = line.strip()
                                if stripped in ['{', '}', '[', ']'] or (stripped.startswith('{') and not any(c in line for c in ['"', "'"])):
                                    continue
                                if re.match(r'^\s*"[^"]+"\s*:', line):
                                    continue
                                cleaned_lines.append(line)
                            return '\n'.join(cleaned_lines).strip()

                        cleaned_chains = []
                        for chain_item in reasoning_chains:
                            if isinstance(chain_item, dict):
                                chain_type = chain_item.get('type', '')
                                chain_text = clean_reasoning_text(chain_item.get('chain', ''))
                                cleaned_chains.append({
                                    'type': chain_type,
                                    'chain': chain_text
                                })
                        result['reasoning_chains'] = cleaned_chains
                    elif isinstance(reasoning_chains, dict):
                        # Backward compatibility
                        def clean_reasoning_text(text):
                            if not isinstance(text, str):
                                return text
                            # Remove code block markers
                            text = re.sub(r'```json\s*', '', text)
                            text = re.sub(r'```\s*', '', text)

                            # Check if the text contains nested JSON structure
                            text_stripped = text.strip()
                            if text_stripped.startswith('{') and ('"sample_id"' in text or '"reasoning_chains"' in text or "'sample_id'" in text):
                                try:
                                    nested_json = json.loads(text)
                                    if isinstance(nested_json, dict) and 'reasoning_chains' in nested_json:
                                        chains = nested_json.get('reasoning_chains', [])
                                        if isinstance(chains, list) and len(chains) > 0:
                                            for chain_obj in chains:
                                                if isinstance(chain_obj, dict) and 'chain' in chain_obj:
                                                    return clean_reasoning_text(chain_obj['chain'])
                                    return ""
                                except (json.JSONDecodeError, Exception):
                                    match = re.search(r'"chain"\s*:\s*"([^"]*(?:\\.[^"]*)*)"', text, re.DOTALL)
                                    if match:
                                        return match.group(1).replace('\\"', '"').replace('\\n', '\n')
                                    lines = text.split('\n')
                                    cleaned_lines = []
                                    for line in lines:
                                        stripped = line.strip()
                                        if stripped in ['{', '}', '[', ']'] or (stripped.startswith('{') and not any(c in line for c in ['"', "'"])):
                                            continue
                                        if re.match(r'^\s*"[^"]+"\s*:', line):
                                            continue
                                        cleaned_lines.append(line)
                                    result = '\n'.join(cleaned_lines).strip()
                                    result = re.sub(r'^\s*[{\[\]}]\s*', '', result)
                                    return result

                            lines = text.split('\n')
                            cleaned_lines = []
                            for line in lines:
                                stripped = line.strip()
                                if stripped in ['{', '}', '[', ']'] or (stripped.startswith('{') and not any(c in line for c in ['"', "'"])):
                                    continue
                                if re.match(r'^\s*"[^"]+"\s*:', line):
                                    continue
                                cleaned_lines.append(line)
                            return '\n'.join(cleaned_lines).strip()

                        result['reasoning_chains'] = [
                            {'type': 'descriptive', 'chain': clean_reasoning_text(reasoning_chains.get('descriptive', ''))},
                            {'type': 'causal', 'chain': clean_reasoning_text(reasoning_chains.get('causal', ''))},
                            {'type': 'comparative', 'chain': clean_reasoning_text(reasoning_chains.get('comparative', ''))}
                        ]
                    else:
                        result['reasoning_chains'] = []

                    # Handle counterfactual_premise (new simplified format)
                    counterfactual_premise = parsed.get('counterfactual_premise', {})
                    if isinstance(counterfactual_premise, dict):
                        result['counterfactual_premise'] = {
                            'aspect': counterfactual_premise.get('aspect', ''),
                            'from': counterfactual_premise.get('from', ''),
                            'to': counterfactual_premise.get('to', '')
                        }
                    else:
                        result['counterfactual_premise'] = {
                            'aspect': '',
                            'from': '',
                            'to': ''
                        }

                    if not isinstance(result['multi_modal_constraints'], list):
                        result['multi_modal_constraints'] = []
                else:
                    raise e
            except (json.JSONDecodeError, Exception) as e2:
                # If fixing failed, use edit request as base information
                sample_info = f" (sample_id: {sample_id})" if sample_id else ""
                print(f"JSON parsing warning{sample_info}: {e} (fix attempt also failed: {e2})")
                result['instruction'] = edit_request
                result['reasoning_chains'] = [
                    {'type': 'descriptive', 'chain': response[:500]},
                    {'type': 'causal', 'chain': response[:500]},
                    {'type': 'comparative', 'chain': response[:500]}
                ]

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

                # Retry mechanism: try up to 3 times (1 initial + 2 retries)
                max_retries = 2
                result = None
                retry_count = 0

                while retry_count <= max_retries:
                    try:
                        result = self.generate_editing_prompts(
                            image_path,
                            edit_request,
                            image_mask_path=image_mask_path,
                            target_path=target_path,
                            target_mask_path=target_mask_path,
                            sample_id=sample_id
                        )

                        # Check if parsing was successful by checking key fields
                        # If error field exists, or critical fields are empty/missing, consider it a failure
                        has_error = 'error' in result

                        # Check if instruction is properly rewritten (not just the original edit_request)
                        instruction = result.get('instruction', '')
                        has_empty_instruction = not instruction or instruction == edit_request

                        # Check reasoning_chains: must have valid chains without nested JSON
                        reasoning_chains = result.get('reasoning_chains', [])
                        has_empty_reasoning = not reasoning_chains
                        has_nested_json_in_chains = False
                        if isinstance(reasoning_chains, list):
                            for item in reasoning_chains:
                                if isinstance(item, dict):
                                    chain_text = item.get('chain', '')
                                    # Check if chain contains nested JSON structure
                                    if isinstance(chain_text, str) and chain_text.strip().startswith('{') and ('"sample_id"' in chain_text or '"reasoning_chains"' in chain_text):
                                        has_nested_json_in_chains = True
                                        break
                                    if not chain_text or len(chain_text.strip()) < 10:  # Too short to be meaningful
                                        has_empty_reasoning = True

                        # Check counterfactual_premise
                        counterfactual_premise = result.get('counterfactual_premise', {})
                        has_empty_premise = not counterfactual_premise.get('aspect') or not counterfactual_premise.get('from') or not counterfactual_premise.get('to')

                        # Check metadata fields
                        has_empty_metadata = not result.get('edit_subject') or not result.get('edit_type') or not result.get('editing_instruction')

                        # Determine if parsing was successful
                        # Success requires: no error, proper instruction, valid reasoning chains (no nested JSON), complete premise, and metadata
                        is_successful = (
                            not has_error and
                            not has_empty_instruction and
                            not has_empty_reasoning and
                            not has_nested_json_in_chains and
                            not has_empty_premise and
                            not has_empty_metadata
                        )

                        if is_successful:
                            # Success - break out of retry loop
                            break
                        else:
                            # Parsing likely failed or result is incomplete
                            failure_reasons = []
                            if has_error:
                                failure_reasons.append("error field present")
                            if has_empty_instruction:
                                failure_reasons.append("instruction not rewritten")
                            if has_empty_reasoning:
                                failure_reasons.append("empty reasoning chains")
                            if has_nested_json_in_chains:
                                failure_reasons.append("nested JSON in chains")
                            if has_empty_premise:
                                failure_reasons.append("empty counterfactual premise")
                            if has_empty_metadata:
                                failure_reasons.append("empty metadata")

                            if retry_count < max_retries:
                                retry_count += 1
                                reason_str = ", ".join(failure_reasons) if failure_reasons else "unknown"
                                print(f"\n  Retry {retry_count}/{max_retries} for {sample_id} due to: {reason_str}")
                                continue
                            else:
                                # Max retries reached, use the result anyway
                                reason_str = ", ".join(failure_reasons) if failure_reasons else "unknown"
                                print(f"\n  Max retries reached for {sample_id}, using partial result (issues: {reason_str})")
                                break

                    except Exception as e:
                        if retry_count < max_retries:
                            retry_count += 1
                            print(f"\n  Retry {retry_count}/{max_retries} for {sample_id} due to exception: {e}")
                            continue
                        else:
                            # Max retries reached, raise the exception
                            raise

                # Preserve information from original data
                if result:
                    result['sample_id'] = sample_id
                    # Note: image_path, image_mask_path, target_path, target_mask_path, and index are not included in the output

                results.append(result)

                # Save individual file, using sample_id as filename
                if save_individual and output_json_path and result:
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
            image_mask_path = item.get('image_mask_path') or item.get('mask_path') or item.get('mask', None)
            target_path = item.get('target_path') or item.get('target') or item.get('target_image', None)
            target_mask_path = item.get('target_mask_path') or item.get('target_mask', None)

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

    # images_dir = prj.HALLUSEGGOOD_DATASET_PATH
    # converted_data_json = os.path.join(prj.HALLUSEGGOOD_DATASET_PATH, "converted_data.json")
    # prompts_output_dir = prj.PROMPTS_OUTPUT_HALLUSEGGOOD_DIR


    # images_dir = prj.EditBench_DATASET_PATH
    # converted_data_json = os.path.join(prj.EditBench_DATASET_PATH, "converted_data.json")
    # prompts_output_dir = prj.PROMPTS_OUTPUT_EDITBENCH_DIR


    images_dir = prj.EditBenchGood_DATASET_PATH
    converted_data_json = os.path.join(prj.EditBenchGood_DATASET_PATH, "converted_data.json")
    prompts_output_dir = prj.PROMPTS_OUTPUT_EDITBENCHGOOD_DIR


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

"""
SAM3 HTTP Server
独立运行SAM3模型，提供HTTP接口供训练脚本调用
"""
import torch
import argparse
from PIL import Image
import numpy as np
import io
import base64
import json
from flask import Flask, request, jsonify
from functools import lru_cache
import logging

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# 日志配置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 全局变量存储模型和处理器
model = None
processor = None


def initialize_model():
    """初始化SAM3模型和处理器（全局单例）"""
    global model, processor
    
    if model is None:
        logger.info("初始化SAM3模型...")
        model = build_sam3_image_model()
        processor = Sam3Processor(model=model, confidence_threshold=0.8)
        logger.info("SAM3模型初始化完成")


def image_to_base64(image):
    """将PIL Image转换为base64字符串"""
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def base64_to_image(base64_str):
    """将base64字符串转换为PIL Image"""
    image_data = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(image_data)).convert('RGB')


def process_image_with_sam(image_path, obj):
    """使用SAM3处理单张图像"""
    try:
        image = Image.open(image_path).convert("RGB")
        inference_state = processor.set_image(image)
        inference_state = processor.set_text_prompt(state=inference_state, prompt=obj)
        masks = inference_state['masks']
        return masks
    except Exception as e:
        logger.error(f"处理图像出错: {e}")
        raise


def calculate_iou(mask1, mask2):
    """计算两个掩膜的IoU"""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


@app.route('/health', methods=['GET'])
def health():
    """健康检查端点"""
    return jsonify({'status': 'ok', 'model_loaded': model is not None})


@app.route('/process', methods=['POST'])
def process_endpoint():
    """
    处理图像并计算IoU的端点
    
    请求格式:
    {
        "image1": "path/to/image1.jpg",  # 或 base64编码的图像
        "image2": "path/to/image2.jpg",  # 或 base64编码的图像
        "object": "object description",
        "use_base64": false  # 是否使用base64编码
    }
    
    响应格式:
    {
        "success": true,
        "iou": 0.85,
        "mask1_shape": [h, w],
        "mask2_shape": [h, w]
    }
    """
    try:
        data = request.json
        
        if not data:
            return jsonify({'success': False, 'error': '请求体为空'}), 400
        
        image1_input = data.get('image1')
        image2_input = data.get('image2')
        obj_prompt = data.get('object', '')
        use_base64 = data.get('use_base64', False)
        
        if not image1_input or not image2_input or not obj_prompt:
            return jsonify({
                'success': False, 
                'error': '缺少必要字段: image1, image2, object'
            }), 400
        
        # 加载图像
        if use_base64:
            image1 = base64_to_image(image1_input)
            image2 = base64_to_image(image2_input)
        else:
            image1 = Image.open(image1_input).convert('RGB')
            image2 = Image.open(image2_input).convert('RGB')
        
        logger.info(f"处理图像对，object: {obj_prompt}")
        
        # 使用SAM3处理两张图像
        inference_state1 = processor.set_image(image1)
        inference_state1 = processor.set_text_prompt(state=inference_state1, prompt=obj_prompt)
        masks1 = inference_state1['masks']
        
        inference_state2 = processor.set_image(image2)
        inference_state2 = processor.set_text_prompt(state=inference_state2, prompt=obj_prompt)
        masks2 = inference_state2['masks']
        
        # 计算IoU
        iou_score = calculate_iou(masks1, masks2)
        
        return jsonify({
            'success': True,
            'iou': iou_score,
            'mask1_shape': list(masks1.shape),
            'mask2_shape': list(masks2.shape)
        })
    
    except Exception as e:
        logger.error(f"处理请求出错: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/segment', methods=['POST'])
def segment_endpoint():
    """
    分割单张图像的端点
    
    请求格式:
    {
        "image": "path/to/image.jpg",  # 或 base64编码的图像
        "object": "object description",
        "use_base64": false
    }
    
    响应格式:
    {
        "success": true,
        "mask": "base64_encoded_mask",
        "shape": [h, w]
    }
    """
    try:
        data = request.json
        
        if not data:
            return jsonify({'success': False, 'error': '请求体为空'}), 400
        
        image_input = data.get('image')
        obj_prompt = data.get('object', '')
        use_base64 = data.get('use_base64', False)
        
        if not image_input or not obj_prompt:
            return jsonify({
                'success': False, 
                'error': '缺少必要字段: image, object'
            }), 400
        
        # 加载图像
        if use_base64:
            image = base64_to_image(image_input)
        else:
            image = Image.open(image_input).convert('RGB')
        
        logger.info(f"分割图像，object: {obj_prompt}")
        
        # 使用SAM3处理图像
        inference_state = processor.set_image(image)
        inference_state = processor.set_text_prompt(state=inference_state, prompt=obj_prompt)
        masks = inference_state['masks']
        
        # 将掩膜转换为base64
        mask_image = Image.fromarray((masks * 255).astype(np.uint8))
        mask_base64 = image_to_base64(mask_image)
        
        return jsonify({
            'success': True,
            'mask': mask_base64,
            'shape': list(masks.shape)
        })
    
    except Exception as e:
        logger.error(f"处理请求出错: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SAM3 HTTP Server')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='服务器主机地址')
    parser.add_argument('--port', type=int, default=5000, help='服务器端口')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    
    args = parser.parse_args()
    
    # 初始化模型
    initialize_model()
    
    # 启动Flask服务器
    logger.info(f"SAM3 HTTP服务器启动在 {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)

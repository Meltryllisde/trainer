"""
SAM3 HTTP客户端
训练脚本通过此客户端调用SAM3服务器
"""
import requests
import json
import io
import base64
from PIL import Image
import numpy as np
from typing import Union, Tuple
import logging

logger = logging.getLogger(__name__)


class Sam3HttpClient:
    """SAM3 HTTP客户端"""
    
    def __init__(self, server_url: str = "http://127.0.0.1:5000", timeout: int = 300):
        """
        初始化客户端
        
        Args:
            server_url: SAM3服务器地址，默认为http://127.0.0.1:5000
            timeout: 请求超时时间（秒），默认300秒
        """
        self.server_url = server_url.rstrip('/')
        self.timeout = timeout
        self._check_server_health()
    
    def _check_server_health(self):
        """检查服务器是否正常"""
        try:
            response = requests.get(
                f"{self.server_url}/health",
                timeout=5
            )
            response.raise_for_status()
            logger.info(f"✓ SAM3服务器连接成功: {self.server_url}")
        except requests.exceptions.RequestException as e:
            raise ConnectionError(
                f"无法连接到SAM3服务器: {self.server_url}\n"
                f"请确保服务器已启动: python sam3_http_server.py\n"
                f"错误: {e}"
            )
    
    def _image_to_base64(self, image: Union[str, Image.Image]) -> str:
        """将图像转换为base64"""
        if isinstance(image, str):
            img = Image.open(image).convert('RGB')
        elif isinstance(image, Image.Image):
            img = image.convert('RGB')
        else:
            raise ValueError("图像必须是路径字符串或PIL Image对象")
        
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def process_images(self, 
                      image1: Union[str, Image.Image], 
                      image2: Union[str, Image.Image], 
                      object_prompt: str,
                      use_base64: bool = True) -> float:
        """
        计算两张图像中指定物体的IoU
        
        Args:
            image1: 第一张图像（文件路径或PIL Image）
            image2: 第二张图像（文件路径或PIL Image）
            object_prompt: 物体描述文本
            use_base64: 是否使用base64编码传输图像
        
        Returns:
            IoU得分（0.0-1.0）
        
        Raises:
            ConnectionError: 连接失败
            ValueError: 参数错误
            RuntimeError: 服务器处理出错
        """
        try:
            # 准备请求数据
            if use_base64:
                image1_data = self._image_to_base64(image1)
                image2_data = self._image_to_base64(image2)
            else:
                # 使用文件路径
                if isinstance(image1, Image.Image):
                    raise ValueError("use_base64=False时，图像必须是文件路径")
                if isinstance(image2, Image.Image):
                    raise ValueError("use_base64=False时，图像必须是文件路径")
                image1_data = image1
                image2_data = image2
            
            payload = {
                'image1': image1_data,
                'image2': image2_data,
                'object': object_prompt,
                'use_base64': use_base64
            }
            
            # 发送请求
            response = requests.post(
                f"{self.server_url}/process",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            
            if not result.get('success', False):
                raise RuntimeError(f"服务器返回错误: {result.get('error', '未知错误')}")
            
            iou_score = result['iou']
            logger.debug(f"IoU计算完成: {iou_score:.4f}")
            
            return iou_score
        
        except requests.exceptions.Timeout:
            raise RuntimeError(f"请求超时（{self.timeout}秒），image太大或SAM3处理时间过长")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"请求失败: {e}")
        except (KeyError, ValueError) as e:
            raise RuntimeError(f"响应解析失败: {e}")
    
    def segment(self, 
               image: Union[str, Image.Image],
               object_prompt: str,
               use_base64: bool = True) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        分割单张图像
        
        Args:
            image: 图像（文件路径或PIL Image）
            object_prompt: 物体描述文本
            use_base64: 是否使用base64编码传输图像
        
        Returns:
            (掩膜数组, 形状)
        
        Raises:
            ConnectionError: 连接失败
            ValueError: 参数错误
            RuntimeError: 服务器处理出错
        """
        try:
            # 准备请求数据
            if use_base64:
                image_data = self._image_to_base64(image)
            else:
                if isinstance(image, Image.Image):
                    raise ValueError("use_base64=False时，图像必须是文件路径")
                image_data = image
            
            payload = {
                'image': image_data,
                'object': object_prompt,
                'use_base64': use_base64
            }
            
            # 发送请求
            response = requests.post(
                f"{self.server_url}/segment",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            
            if not result.get('success', False):
                raise RuntimeError(f"服务器返回错误: {result.get('error', '未知错误')}")
            
            # 解析掩膜
            mask_base64 = result['mask']
            mask_data = base64.b64decode(mask_base64)
            mask_image = Image.open(io.BytesIO(mask_data))
            mask_array = np.array(mask_image) / 255.0
            
            shape = tuple(result['shape'])
            logger.debug(f"分割完成: 掩膜形状 {shape}")
            
            return mask_array, shape
        
        except requests.exceptions.Timeout:
            raise RuntimeError(f"请求超时（{self.timeout}秒）")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"请求失败: {e}")
        except (KeyError, ValueError) as e:
            raise RuntimeError(f"响应解析失败: {e}")


# 全局客户端实例
_client_instance = None


def get_sam3_client(server_url: str = "http://127.0.0.1:5000") -> Sam3HttpClient:
    """
    获取SAM3客户端（单例模式）
    
    Args:
        server_url: SAM3服务器地址
    
    Returns:
        Sam3HttpClient实例
    """
    global _client_instance
    if _client_instance is None:
        _client_instance = Sam3HttpClient(server_url)
    return _client_instance


def process_images(image1: Union[str, Image.Image],
                   image2: Union[str, Image.Image], 
                   object_prompt: str,
                   server_url: str = "http://127.0.0.1:5000") -> float:
    client = get_sam3_client(server_url)
    return client.process_images(image1, image2, object_prompt)

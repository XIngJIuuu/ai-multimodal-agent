import os
import uuid
from typing import Optional, Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
import requests
import json
from dotenv import load_dotenv

load_dotenv()

ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

class CustomLLM(BaseChatModel):
    model_name: str
    api_key: str
    base_url: str
    
    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs) -> ChatResult:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model_name,
            "messages": []
        }
        
        for msg in messages:
            role = "user" if isinstance(msg, HumanMessage) else "system"
            payload["messages"].append({"role": role, "content": msg.content})
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )
        
        response.raise_for_status()
        result = response.json()
        
        content = result["choices"][0]["message"]["content"]
        
        generation = ChatGeneration(
            message=HumanMessage(content=content)
        )
        
        return ChatResult(generations=[generation])
    
    @property
    def _llm_type(self) -> str:
        return "custom"
    
    def _invoke(self, input: Dict[str, Any], **kwargs) -> str:
        messages = input.get("messages", [])
        result = self._generate(messages)
        return result.generations[0].message.content

class ModelManager:
    MODEL_CONFIGS = {
        "openai": {
            "display_name": "OpenAI",
            "default_model": "gpt-4o",
            "base_url": "https://api.openai.com/v1",
            "description": "OpenAI官方模型，适合通用任务"
        },
        "claude": {
            "display_name": "Claude (Anthropic)",
            "default_model": "claude-3-opus-20240229",
            "base_url": "https://api.anthropic.com/v1",
            "description": "Anthropic的Claude模型，擅长长文本和创意写作"
        },
        "gemini": {
            "display_name": "Gemini (Google)",
            "default_model": "gemini-pro",
            "base_url": "https://generativelanguage.googleapis.com/v1",
            "description": "Google的Gemini模型，多模态能力强"
        },
        "deepseek": {
            "display_name": "DeepSeek",
            "default_model": "deepseek-chat",
            "base_url": "https://api.deepseek.com/v1",
            "description": "深度求索，中文能力优秀"
        },
        "qwen": {
            "display_name": "Qwen (阿里通义)",
            "default_model": "qwen-plus",
            "base_url": "https://dashscope.aliyuncs.com/api/v1",
            "description": "阿里通义千问，中文理解能力强"
        },
        "doubao": {
            "display_name": "Doubao (豆包)",
            "default_model": "doubao-pro",
            "base_url": "https://api.doubao.com/v1",
            "description": "字节跳动豆包，适合日常对话"
        },
        "moonshot": {
            "display_name": "Moonshot (Kimi)",
            "default_model": "moonshot-v1-8k",
            "base_url": "https://api.moonshot.cn/v1",
            "description": "月之暗面Kimi，长文本处理能力强"
        },
        "local": {
            "display_name": "本地模型 (OpenAI兼容)",
            "default_model": "gpt-4o",
            "base_url": "http://localhost:8080/v1",
            "description": "本地部署的OpenAI兼容模型"
        },
        "custom": {
            "display_name": "自定义API",
            "default_model": "",
            "base_url": "",
            "description": "自定义的OpenAI兼容API"
        }
    }
    
    @staticmethod
    def get_provider_list() -> List[str]:
        return list(ModelManager.MODEL_CONFIGS.keys())
    
    @staticmethod
    def get_provider_info(provider: str) -> Dict[str, str]:
        return ModelManager.MODEL_CONFIGS.get(provider, {"display_name": provider, "default_model": "", "base_url": "", "description": ""})
    
    @staticmethod
    def create_llm(provider: str, api_key: str, model_name: str = None, base_url: str = None) -> BaseChatModel:
        config = ModelManager.MODEL_CONFIGS.get(provider, {})
        
        if model_name is None:
            model_name = config.get("default_model", "gpt-4o")
        
        if base_url is None:
            base_url = config.get("base_url", "https://api.openai.com/v1")
        
        if provider == "openai":
            return ChatOpenAI(
                model=model_name,
                api_key=api_key,
                base_url=base_url,
                temperature=0.7
            )
        elif provider == "claude":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model=model_name,
                api_key=api_key,
                base_url=base_url,
                temperature=0.7
            )
        elif provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                temperature=0.7
            )
        else:
            return CustomLLM(
                model_name=model_name,
                api_key=api_key,
                base_url=base_url
            )

class AssetManager:
    SUPPORTED_TYPES = {
        "image": ["jpg", "jpeg", "png", "gif", "bmp", "webp"],
        "video": ["mp4", "mov", "avi", "mkv", "webm"],
        "audio": ["mp3", "wav", "ogg", "flac"],
        "text": ["txt", "md", "json"]
    }
    
    @staticmethod
    def get_supported_types() -> Dict[str, List[str]]:
        return AssetManager.SUPPORTED_TYPES
    
    @staticmethod
    def get_file_type(filename: str) -> str:
        ext = filename.split(".")[-1].lower()
        for file_type, exts in AssetManager.SUPPORTED_TYPES.items():
            if ext in exts:
                return file_type
        return "unknown"
    
    @staticmethod
    def save_asset(file_data: bytes, filename: str, description: str = "") -> Dict[str, str]:
        file_type = AssetManager.get_file_type(filename)
        asset_id = str(uuid.uuid4())
        
        safe_filename = f"{asset_id}_{filename}"
        file_path = os.path.join(ASSETS_DIR, safe_filename)
        
        with open(file_path, "wb") as f:
            f.write(file_data)
        
        return {
            "asset_id": asset_id,
            "filename": filename,
            "file_type": file_type,
            "file_path": file_path,
            "description": description
        }
    
    @staticmethod
    def get_asset(asset_id: str) -> Optional[Dict[str, str]]:
        for filename in os.listdir(ASSETS_DIR):
            if filename.startswith(asset_id):
                full_path = os.path.join(ASSETS_DIR, filename)
                original_name = "_".join(filename.split("_")[1:])
                return {
                    "asset_id": asset_id,
                    "filename": original_name,
                    "file_type": AssetManager.get_file_type(filename),
                    "file_path": full_path
                }
        return None
    
    @staticmethod
    def list_assets() -> List[Dict[str, str]]:
        assets = []
        for filename in os.listdir(ASSETS_DIR):
            if not filename.startswith("."):
                parts = filename.split("_", 1)
                if len(parts) == 2:
                    asset_id = parts[0]
                    original_name = parts[1]
                    assets.append({
                        "asset_id": asset_id,
                        "filename": original_name,
                        "file_type": AssetManager.get_file_type(filename),
                        "file_path": os.path.join(ASSETS_DIR, filename)
                    })
        return assets
    
    @staticmethod
    def delete_asset(asset_id: str) -> bool:
        for filename in os.listdir(ASSETS_DIR):
            if filename.startswith(asset_id):
                os.remove(os.path.join(ASSETS_DIR, filename))
                return True
        return False
    
    @staticmethod
    def describe_asset(file_path: str, llm: BaseChatModel = None) -> str:
        file_type = AssetManager.get_file_type(file_path)
        
        if file_type == "text":
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()[:2000]
            
            if llm:
                from langchain_core.prompts import ChatPromptTemplate
                
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "你是一个专业的内容摘要师。请用简洁的语言概括以下文本内容，重点提取关键信息和主题。"),
                    ("human", "文本内容：\n{content}\n\n请生成不超过100字的摘要：")
                ])
                
                chain = prompt | llm
                try:
                    result = chain.invoke({"content": content[:1000]})
                    import json
                    try:
                        data = json.loads(result)
                        return data.get("summary", data.get("text", str(result)))
                    except:
                        return str(result)[:100]
                except:
                    return f"文本内容摘要：{content[:200]}..."
            else:
                return f"文本内容摘要：{content[:200]}..."
        
        elif file_type == "image":
            if llm:
                return "图片素材（AI分析：包含视觉元素，可用于视频参考）"
            else:
                return "图片素材（需在视频生成时作为参考图使用）"
        
        elif file_type == "video":
            if llm:
                return "视频素材（AI分析：动态视觉内容，可作为参考视频或直接用于合成）"
            else:
                return "视频素材（可作为参考视频或直接用于合成）"
        
        elif file_type == "audio":
            if llm:
                return "音频素材（AI分析：可作为背景音乐或音效，增强视频氛围）"
            else:
                return "音频素材（可作为背景音乐或音效）"
        
        return "未知类型素材"

class VideoAPI:
    SUPPORTED_APIS = {
        "pika": {
            "display_name": "Pika Labs",
            "base_url": "https://api.pika.art/v1",
            "description": "AI视频生成平台，支持文本转视频",
            "required_params": ["prompt", "aspect_ratio"]
        },
        "runway": {
            "display_name": "Runway ML",
            "base_url": "https://api.runwayml.com/v1",
            "description": "AI视频编辑和生成平台",
            "required_params": ["prompt", "duration"]
        },
        "sora": {
            "display_name": "OpenAI Sora",
            "base_url": "https://api.openai.com/v1",
            "description": "OpenAI视频生成模型",
            "required_params": ["prompt", "duration"]
        },
        "stability": {
            "display_name": "Stability AI",
            "base_url": "https://api.stability.ai/v2beta",
            "description": "Stability AI视频生成API",
            "required_params": ["prompt", "width", "height"]
        },
        "azure": {
            "display_name": "Azure Video Indexer",
            "base_url": "https://api.videoindexer.ai",
            "description": "Microsoft Azure视频分析服务",
            "required_params": ["video_url"]
        },
        "local": {
            "display_name": "本地视频生成服务",
            "base_url": "http://localhost:5000",
            "description": "本地部署的视频生成服务",
            "required_params": ["prompt"]
        },
        "seedance": {
            "display_name": "Seedance AI",
            "base_url": "https://api.seedance.ai",
            "description": "Seedance AI视频生成平台，支持首尾帧控制",
            "required_params": ["prompt", "model"]
        },
        "custom": {
            "display_name": "通用接口",
            "base_url": "",
            "description": "自定义API接口，支持配置任意视频生成服务",
            "required_params": ["prompt"]
        }
    }
    
    @staticmethod
    def get_api_list() -> List[str]:
        return list(VideoAPI.SUPPORTED_APIS.keys())
    
    @staticmethod
    def get_api_info(api_name: str) -> Dict[str, Any]:
        return VideoAPI.SUPPORTED_APIS.get(api_name, {})
    
    @staticmethod
    def check_task_status(api_name: str, api_key: str, base_url: str, task_id: str) -> Dict[str, Any]:
        if api_name == "seedance":
            url = base_url if base_url else "https://api.seedance.ai"
            endpoint = f"{url}/api/v1/tasks/{task_id}"
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
                "Accept-Encoding": "identity"
            }
            
            try:
                response = requests.get(endpoint, headers=headers, timeout=60)
                response.raise_for_status()
                result = response.json()
                
                status = result.get("status", "")
                if status == "completed":
                    return {
                        "status": "success",
                        "video_url": result.get("output", {}).get("url", ""),
                        "task_id": task_id
                    }
                elif status in ["pending", "running"]:
                    return {
                        "status": "pending",
                        "message": f"任务正在处理中: {result.get('progress', '0%')}",
                        "task_id": task_id
                    }
                elif status == "failed":
                    return {
                        "status": "failed",
                        "error": result.get("error", "任务失败"),
                        "task_id": task_id
                    }
                return result
            except Exception as e:
                return {
                    "error": str(e),
                    "status": "failed",
                    "task_id": task_id
                }
        return {
            "error": "不支持的API",
            "status": "failed"
        }
    
    @staticmethod
    def generate_video(api_name: str, api_key: str, base_url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        config = VideoAPI.SUPPORTED_APIS.get(api_name, {})
        url = base_url if base_url else config.get("base_url", "")
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        if api_name == "pika":
            endpoint = f"{url}/generate"
            payload = {
                "prompt": params.get("prompt", ""),
                "aspect_ratio": params.get("aspect_ratio", "16:9"),
                "duration": params.get("duration", 10)
            }
        elif api_name == "runway":
            endpoint = f"{url}/video/generate"
            payload = {
                "prompt": params.get("prompt", ""),
                "duration": params.get("duration", 10),
                "quality": params.get("quality", "standard")
            }
        elif api_name == "sora":
            endpoint = f"{url}/videos/generations"
            payload = {
                "model": "sora",
                "prompt": params.get("prompt", ""),
                "duration": params.get("duration", 10)
            }
        elif api_name == "stability":
            endpoint = f"{url}/video/generate"
            payload = {
                "prompt": params.get("prompt", ""),
                "width": params.get("width", 1024),
                "height": params.get("height", 576),
                "fps": params.get("fps", 24)
            }
        elif api_name == "azure":
            endpoint = f"{url}/Videos"
            payload = {
                "videoUrl": params.get("video_url", ""),
                "name": params.get("name", "video")
            }
        elif api_name == "local":
            endpoint = f"{url}/generate-video"
            payload = params
        elif api_name == "seedance":
            endpoint = f"{url}/api/v1/tasks"
            headers["Accept-Encoding"] = "identity"
            
            content = []
            
            first_frame = params.get("first_frame", "")
            if first_frame:
                content.append({
                    "role": "first_frame",
                    "type": "image",
                    "url": first_frame
                })
            
            content.append({
                "role": "prompt",
                "type": "text",
                "content": params.get("prompt", "")
            })
            
            last_frame = params.get("last_frame", "")
            if last_frame:
                content.append({
                    "role": "last_frame",
                    "type": "image",
                    "url": last_frame
                })
            
            payload = {
                "model": params.get("model", "doubao-seedance-2-0-fast-260128"),
                "inputs": {
                    "content": content
                },
                "parameters": {
                    "duration": params.get("duration", 10),
                    "aspect_ratio": params.get("aspect_ratio", "16:9"),
                    "output_format": params.get("output_format", "mp4")
                }
            }
        elif api_name == "custom":
            custom_endpoint = params.get("custom_endpoint", "/generate")
            endpoint = f"{url}{custom_endpoint}"
            
            custom_method = params.get("custom_method", "POST").upper()
            
            custom_headers = params.get("custom_headers", {})
            headers.update(custom_headers)
            
            custom_payload = params.get("custom_payload", {})
            if custom_payload:
                payload = custom_payload
            else:
                payload = {k: v for k, v in params.items() if k not in ["custom_endpoint", "custom_method", "custom_headers", "custom_payload"]}
        else:
            endpoint = f"{url}/generate"
            payload = params
        
        try:
            if api_name == "custom" and custom_method == "GET":
                response = requests.get(endpoint, headers=headers, params=payload, timeout=120)
            else:
                response = requests.post(endpoint, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            
            if api_name == "seedance":
                task_id = result.get("task_id", "")
                return {
                    "status": "pending",
                    "task_id": task_id,
                    "message": "任务已创建，等待视频生成完成",
                    "mock_data": {
                        "video_url": f"https://example.com/video_{hash(params.get('prompt', ''))}.mp4",
                        "duration": params.get("duration", 10),
                        "generated_prompt": params.get("prompt", "")
                    }
                }
            
            return result
        except Exception as e:
            return {
                "error": str(e),
                "status": "failed",
                "mock_data": {
                    "video_url": f"https://example.com/video_{hash(params.get('prompt', ''))}.mp4",
                    "duration": params.get("duration", 10),
                    "generated_prompt": params.get("prompt", "")
                }
            }

class ImageAPI:
    SUPPORTED_APIS = {
        "dall-e": {
            "display_name": "DALL-E 3",
            "base_url": "https://api.openai.com/v1",
            "description": "OpenAI图像生成模型",
            "required_params": ["prompt"]
        },
        "gpt-image": {
            "display_name": "GPT Image",
            "base_url": "https://api.openai.com/v1",
            "description": "OpenAI GPT-4V图像生成模型",
            "required_params": ["prompt"]
        },
        "stable-diffusion": {
            "display_name": "Stable Diffusion",
            "base_url": "https://api.stability.ai/v1",
            "description": "Stability AI图像生成API",
            "required_params": ["prompt", "width", "height"]
        },
        "midjourney": {
            "display_name": "Midjourney",
            "base_url": "https://api.midjourney.com/v2",
            "description": "Midjourney图像生成API",
            "required_params": ["prompt"]
        },
        "leonardo": {
            "display_name": "Leonardo AI",
            "base_url": "https://cloud.leonardo.ai/api/rest",
            "description": "Leonardo AI图像生成平台",
            "required_params": ["prompt"]
        },
        "bing": {
            "display_name": "Bing Image Creator",
            "base_url": "https://api.bing.microsoft.com/v1",
            "description": "Microsoft Bing图像生成",
            "required_params": ["prompt"]
        },
        "local": {
            "display_name": "本地图片生成服务",
            "base_url": "http://localhost:5000",
            "description": "本地部署的图片生成服务",
            "required_params": ["prompt"]
        },
        "custom": {
            "display_name": "通用接口",
            "base_url": "",
            "description": "自定义API接口，支持配置任意图片生成服务",
            "required_params": ["prompt"]
        }
    }
    
    @staticmethod
    def get_api_list() -> List[str]:
        return list(ImageAPI.SUPPORTED_APIS.keys())
    
    @staticmethod
    def get_api_info(api_name: str) -> Dict[str, Any]:
        return ImageAPI.SUPPORTED_APIS.get(api_name, {})
    
    @staticmethod
    def generate_image(api_name: str, api_key: str, base_url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        config = ImageAPI.SUPPORTED_APIS.get(api_name, {})
        url = base_url if base_url else config.get("base_url", "")
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        if api_name == "dall-e":
            endpoint = f"{url}/images/generations"
            payload = {
                "model": params.get("model", "dall-e-3"),
                "prompt": params.get("prompt", ""),
                "n": params.get("n", 1),
                "size": params.get("size", "1024x1024"),
                "quality": params.get("quality", "standard"),
                "style": params.get("style", "natural")
            }
        elif api_name == "gpt-image":
            endpoint = f"{url}/chat/completions"
            payload = {
                "model": params.get("model", "gpt-4o"),
                "messages": [
                    {
                        "role": "system",
                        "content": "你是一个专业的图像生成提示词工程师。根据用户的需求，生成一个详细、高质量的图像描述，用于图像生成API。只返回图像描述，不要包含其他内容。"
                    },
                    {
                        "role": "user",
                        "content": f"请根据以下需求生成图像描述：{params.get('prompt', '')}\n\n要求：\n- 尺寸：{params.get('width', 1024)} × {params.get('height', 1024)}\n- 风格：{params.get('style', '写实')}\n- 负面提示词：{params.get('negative_prompt', '')}"
                    }
                ],
                "max_tokens": 500
            }
        elif api_name == "stable-diffusion":
            endpoint = f"{url}/generation/stable-diffusion-xl-1024-v1-0/text-to-image"
            payload = {
                "text_prompts": [
                    {"text": params.get("prompt", ""), "weight": 1},
                    {"text": params.get("negative_prompt", ""), "weight": -1}
                ],
                "width": params.get("width", 1024),
                "height": params.get("height", 1024),
                "steps": params.get("steps", 30),
                "cfg_scale": params.get("cfg_scale", 7),
                "seed": params.get("seed", -1)
            }
        elif api_name == "midjourney":
            endpoint = f"{url}/journey/create"
            payload = {
                "prompt": params.get("prompt", ""),
                "aspect_ratio": params.get("aspect_ratio", "1:1"),
                "style": params.get("style", "default"),
                "version": params.get("version", "6")
            }
        elif api_name == "leonardo":
            endpoint = f"{url}/generations"
            payload = {
                "prompt": params.get("prompt", ""),
                "negative_prompt": params.get("negative_prompt", ""),
                "modelId": params.get("model_id", "aa77f7c3-3426-4fb9-9b1d-169f06886a96"),
                "width": params.get("width", 1024),
                "height": params.get("height", 1024),
                "num_images": params.get("num_images", 1),
                "guidance_scale": params.get("guidance_scale", 7)
            }
        elif api_name == "bing":
            endpoint = f"{url}/images/generate"
            payload = {
                "prompt": params.get("prompt", ""),
                "size": params.get("size", "medium"),
                "count": params.get("count", 1)
            }
        elif api_name == "local":
            endpoint = f"{url}/generate-image"
            payload = params
        elif api_name == "custom":
            custom_endpoint = params.get("custom_endpoint", "/generate")
            endpoint = f"{url}{custom_endpoint}"
            
            custom_method = params.get("custom_method", "POST").upper()
            
            custom_headers = params.get("custom_headers", {})
            headers.update(custom_headers)
            
            custom_payload = params.get("custom_payload", {})
            if custom_payload:
                payload = custom_payload
            else:
                payload = {k: v for k, v in params.items() if k not in ["custom_endpoint", "custom_method", "custom_headers", "custom_payload"]}
        else:
            endpoint = f"{url}/generate"
            payload = params
        
        try:
            if api_name == "custom" and custom_method == "GET":
                response = requests.get(endpoint, headers=headers, params=payload, timeout=120)
            else:
                response = requests.post(endpoint, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            
            if api_name == "gpt-image" and "choices" in result:
                gpt_prompt = result["choices"][0]["message"]["content"].strip()
                
                dalle_endpoint = f"{url}/images/generations"
                dalle_payload = {
                    "model": "dall-e-3",
                    "prompt": gpt_prompt,
                    "n": params.get("n", 1),
                    "size": f"{params.get('width', 1024)}x{params.get('height', 1024)}",
                    "quality": params.get("quality", "standard"),
                    "style": params.get("style", "natural")
                }
                
                dalle_response = requests.post(dalle_endpoint, headers=headers, json=dalle_payload, timeout=120)
                dalle_response.raise_for_status()
                dalle_result = dalle_response.json()
                
                if "data" in dalle_result:
                    return {
                        "status": "success",
                        "image_url": dalle_result["data"][0]["url"],
                        "revised_prompt": dalle_result["data"][0].get("revised_prompt", ""),
                        "gpt_generated_prompt": gpt_prompt
                    }
                return dalle_result
            elif api_name == "dall-e" and "data" in result:
                return {
                    "status": "success",
                    "image_url": result["data"][0]["url"],
                    "revised_prompt": result["data"][0].get("revised_prompt", "")
                }
            elif api_name == "stable-diffusion" and "artifacts" in result:
                return {
                    "status": "success",
                    "image_url": result["artifacts"][0]["base64"],
                    "seed": result["seed"]
                }
            return result
        except Exception as e:
            return {
                "error": str(e),
                "status": "failed",
                "mock_data": {
                    "image_url": f"https://picsum.photos/{params.get('width', 1024)}/{params.get('height', 1024)}?random={hash(params.get('prompt', ''))}",
                    "generated_prompt": params.get("prompt", "")
                }
            }
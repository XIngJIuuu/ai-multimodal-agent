from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

from .model_manager import ModelManager, ImageAPI
from .memory_manager import MemoryManager

load_dotenv()

class ImageGenerationRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    style: str = ""
    seed: int = -1
    num_images: int = 1

class ImageGenerationResult(BaseModel):
    image_url: str
    prompt_used: str
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    status: str = "success"
    revised_prompt: str = ""
    seed: int = -1

class ImageAgentState(BaseModel):
    user_prompt: str
    image_request: Optional[ImageGenerationRequest] = None
    image_results: Optional[List[ImageGenerationResult]] = None
    exported_images: Optional[List[str]] = None
    uploaded_assets: Optional[List[Dict[str, Any]]] = None
    relevant_memories: Optional[str] = None
    memory_saved: bool = False
    config: Dict[str, Any] = {}

class ImageAgentStateTypedDict(Dict[str, Any]):
    user_prompt: str
    image_request: Optional[ImageGenerationRequest]
    image_results: Optional[List[ImageGenerationResult]]
    exported_images: Optional[List[str]]
    uploaded_assets: Optional[List[Dict[str, Any]]]
    relevant_memories: Optional[str]
    memory_saved: bool
    config: Dict[str, Any]

def memory_retrieval(state: ImageAgentStateTypedDict) -> Dict[str, Any]:
    config = state["config"]
    user_prompt = state["user_prompt"]
    
    provider = config.get("image_planner_provider", "openai")
    api_key = config.get("image_planner_api_key", "")
    model_name = config.get("image_planner_model", "")
    base_url = config.get("image_planner_base_url", "")
    
    llm = None
    if api_key:
        llm = ModelManager.create_llm(provider, api_key, model_name, base_url)
    
    relevant_memories = MemoryManager.get_relevant_memories_for_prompt(user_prompt, llm, limit=3)
    
    return {"relevant_memories": relevant_memories}

def generate_image_prompt(state: ImageAgentStateTypedDict) -> Dict[str, Any]:
    config = state["config"]
    provider = config.get("image_planner_provider", "openai")
    api_key = config.get("image_planner_api_key", "")
    model_name = config.get("image_planner_model", "")
    base_url = config.get("image_planner_base_url", "")
    
    llm = ModelManager.create_llm(provider, api_key, model_name, base_url)
    
    assets_info = ""
    if state.get("uploaded_assets"):
        assets_info = "\n\n可用素材：\n"
        for asset in state["uploaded_assets"]:
            assets_info += f"- {asset.get('filename', '')} ({asset.get('file_type', '')}): {asset.get('description', '')}\n"
    
    memories_info = ""
    if state.get("relevant_memories") and state["relevant_memories"] != "无相关记忆":
        memories_info = f"\n\n相关记忆：\n{state['relevant_memories']}"
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个专业的图像生成提示词工程师。请根据用户需求，生成高质量的图像生成提示词。
        
        要求：
        1. 提示词要详细、具体，包含主体、背景、光线、风格等元素
        2. 生成一个正面提示词和一个负面提示词（排除不想要的元素）
        3. 根据用户需求建议合适的图片尺寸
        4. 如果有可用素材或相关记忆，请参考并融入提示词中
        5. 返回JSON格式，包含prompt, negative_prompt, width, height, style字段
        """),
        ("human", "用户需求：{user_prompt}\n\n配置参数：{config_summary}\n{assets_info}{memories_info}"),
    ])
    
    config_summary = {
        "default_width": config.get("image_width", 1024),
        "default_height": config.get("image_height", 1024),
        "preferred_style": config.get("image_style", "写实"),
        "negative_prompt": config.get("negative_prompt", "")
    }
    
    chain = prompt | llm
    
    try:
        result = chain.invoke({
            "user_prompt": state["user_prompt"],
            "config_summary": str(config_summary),
            "assets_info": assets_info,
            "memories_info": memories_info
        })
        
        try:
            import json
            result_dict = json.loads(result)
        except:
            result_dict = {
                "prompt": result,
                "negative_prompt": config.get("negative_prompt", ""),
                "width": config.get("image_width", 1024),
                "height": config.get("image_height", 1024),
                "style": config.get("image_style", "写实")
            }
        
        image_request = ImageGenerationRequest(
            prompt=result_dict.get("prompt", state["user_prompt"]),
            negative_prompt=result_dict.get("negative_prompt", ""),
            width=result_dict.get("width", config.get("image_width", 1024)),
            height=result_dict.get("height", config.get("image_height", 1024)),
            style=result_dict.get("style", ""),
            num_images=config.get("num_images", 1),
            seed=config.get("seed", -1)
        )
        
        return {"image_request": image_request}
    except Exception as e:
        image_request = ImageGenerationRequest(
            prompt=state["user_prompt"],
            negative_prompt=config.get("negative_prompt", ""),
            width=config.get("image_width", 1024),
            height=config.get("image_height", 1024),
            style=config.get("image_style", "写实"),
            num_images=config.get("num_images", 1),
            seed=config.get("seed", -1)
        )
        return {"image_request": image_request}

def call_image_api(state: ImageAgentStateTypedDict) -> Dict[str, Any]:
    config = state["config"]
    image_request = state.get("image_request")
    
    if not image_request:
        return {"image_results": []}
    
    api_name = config.get("image_api", "dall-e")
    api_key = config.get("image_api_key", "")
    base_url = config.get("image_base_url", "")
    
    results = []
    
    for _ in range(image_request.num_images):
        params = {
            "prompt": image_request.prompt,
            "negative_prompt": image_request.negative_prompt,
            "width": image_request.width,
            "height": image_request.height,
            "style": image_request.style,
            "seed": image_request.seed if image_request.seed != -1 else None
        }
        
        if api_name == "custom":
            params["custom_endpoint"] = config.get("image_custom_endpoint", "/generate")
            params["custom_method"] = config.get("image_custom_method", "POST")
            
            custom_headers = config.get("image_custom_headers", "{}")
            try:
                import json
                params["custom_headers"] = json.loads(custom_headers)
            except:
                params["custom_headers"] = {}
            
            custom_payload = config.get("image_custom_payload", "")
            if custom_payload:
                try:
                    import json
                    params["custom_payload"] = json.loads(custom_payload)
                except:
                    params["custom_payload"] = {}
        
        api_result = ImageAPI.generate_image(api_name, api_key, base_url, params)
        
        status = api_result.get("status", "success")
        
        if status == "success":
            image_url = api_result.get("image_url", "")
            revised_prompt = api_result.get("revised_prompt", "")
            seed = api_result.get("seed", -1)
        else:
            mock_data = api_result.get("mock_data", {})
            image_url = mock_data.get("image_url", "")
            revised_prompt = mock_data.get("generated_prompt", "")
            seed = -1
        
        result = ImageGenerationResult(
            image_url=image_url,
            prompt_used=image_request.prompt,
            negative_prompt=image_request.negative_prompt,
            width=image_request.width,
            height=image_request.height,
            status=status,
            revised_prompt=revised_prompt,
            seed=seed
        )
        results.append(result)
    
    return {"image_results": results}

def export_images(state: ImageAgentStateTypedDict) -> Dict[str, Any]:
    image_results = state.get("image_results", [])
    
    exported = []
    for i, result in enumerate(image_results):
        exported.append({
            "index": i + 1,
            "image_url": result.image_url,
            "prompt": result.prompt_used,
            "width": result.width,
            "height": result.height
        })
    
    return {"exported_images": exported}

def save_generation_memory(state: ImageAgentStateTypedDict) -> Dict[str, Any]:
    image_request = state.get("image_request")
    image_results = state.get("image_results", [])
    
    if image_request:
        content_lines = []
        
        content_lines.append("## 图片生成请求")
        content_lines.append(f"- 提示词: {image_request.prompt}")
        content_lines.append(f"- 负面提示词: {image_request.negative_prompt}")
        content_lines.append(f"- 宽度: {image_request.width}")
        content_lines.append(f"- 高度: {image_request.height}")
        content_lines.append(f"- 风格: {image_request.style}")
        content_lines.append(f"- 数量: {image_request.num_images}")
        content_lines.append(f"- Seed: {image_request.seed}")
        content_lines.append("")
        
        content_lines.append("## 图片生成结果")
        for i, result in enumerate(image_results):
            content_lines.append(f"- 图片 {i+1}: {result.image_url}")
            content_lines.append(f"  - 状态: {result.status}")
            content_lines.append(f"  - 修订提示词: {result.revised_prompt}")
            content_lines.append(f"  - Seed: {result.seed}")
        
        content = "\n".join(content_lines)
        
        metadata = {
            "type": "image_generation",
            "title": image_request.prompt[:50] + "..." if len(image_request.prompt) > 50 else image_request.prompt,
            "tags": ["image", "generation", image_request.style]
        }
        
        MemoryManager.save_memory(content, metadata)
        
        return {"memory_saved": True}
    
    return {"memory_saved": False}

workflow = StateGraph(ImageAgentStateTypedDict)

workflow.add_node("memory_retrieval", memory_retrieval)
workflow.add_node("generate_image_prompt", generate_image_prompt)
workflow.add_node("call_image_api", call_image_api)
workflow.add_node("export_images", export_images)
workflow.add_node("save_generation_memory", save_generation_memory)

workflow.set_entry_point("memory_retrieval")
workflow.add_edge("memory_retrieval", "generate_image_prompt")
workflow.add_edge("generate_image_prompt", "call_image_api")
workflow.add_edge("call_image_api", "export_images")
workflow.add_edge("export_images", "save_generation_memory")
workflow.add_edge("save_generation_memory", END)

image_agent = workflow.compile()

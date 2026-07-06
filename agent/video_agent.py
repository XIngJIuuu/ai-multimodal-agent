import os
from typing import TypedDict, List, Optional, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from .model_manager import ModelManager, VideoAPI
from .memory_manager import MemoryManager

load_dotenv()

class UploadedAsset(BaseModel):
    asset_id: str = Field(description="素材ID")
    filename: str = Field(description="文件名")
    file_type: str = Field(description="文件类型（image/video/audio/text）")
    file_path: str = Field(description="文件路径")
    description: str = Field(description="素材描述")
    usage_scene: Optional[int] = Field(default=None, description="推荐使用场景")

class VideoPlan(BaseModel):
    title: str = Field(description="视频标题")
    theme: str = Field(description="视频主题")
    tone: str = Field(description="视频风格")
    target_audience: str = Field(description="目标受众")
    duration: float = Field(description="视频时长（秒）")
    scene_count: int = Field(description="场景数量")
    key_elements: List[str] = Field(description="关键元素列表")
    color_palette: str = Field(description="色彩风格")
    lighting: str = Field(description="光线风格")

class ScenePrompt(BaseModel):
    scene_number: int = Field(description="场景序号")
    prompt: str = Field(description="视频生成API的详细prompt")
    duration: float = Field(description="场景时长（秒）")
    aspect_ratio: str = Field(description="画面比例")
    camera_angle: str = Field(description="拍摄角度")
    style: str = Field(description="画面风格")
    used_assets: List[str] = Field(default_factory=list, description="使用的素材ID列表")

class VideoGenerationResult(BaseModel):
    scene_number: int = Field(description="场景序号")
    video_url: str = Field(description="生成的视频URL")
    duration: float = Field(description="视频时长（秒）")
    status: str = Field(description="生成状态")
    prompt_used: str = Field(description="使用的prompt")

class VideoSubtitle(BaseModel):
    scene_number: int = Field(description="场景序号")
    text: str = Field(description="字幕内容")
    start_time: float = Field(description="开始时间（秒）")
    end_time: float = Field(description="结束时间（秒）")

class OptimizationRequest(BaseModel):
    scene_number: int = Field(description="需要优化的场景序号")
    optimization_type: str = Field(description="优化类型：style/character/background/color/composition")
    instruction: str = Field(description="优化指令")

class VideoExport(BaseModel):
    title: str = Field(description="视频标题")
    scenes: List[dict] = Field(description="场景列表")
    audio_plan: dict = Field(description="音频方案")
    subtitles: List[dict] = Field(description="字幕列表")
    total_duration: float = Field(description="总时长")
    export_format: str = Field(description="导出格式")
    metadata: dict = Field(description="元数据")

class VideoAgentState(TypedDict):
    user_prompt: str
    video_plan: Optional[VideoPlan]
    scene_prompts: Optional[List[ScenePrompt]]
    video_results: Optional[List[VideoGenerationResult]]
    subtitles: Optional[List[VideoSubtitle]]
    export_result: Optional[VideoExport]
    uploaded_assets: Optional[List[UploadedAsset]]
    optimization_requests: Optional[List[OptimizationRequest]]
    config: Dict[str, Any]
    relevant_memories: Optional[str]
    memory_saved: Optional[bool]

def memory_retrieval(state: VideoAgentState) -> VideoAgentState:
    config = state["config"]
    user_prompt = state["user_prompt"]
    
    provider = config.get("planner_provider", "openai")
    api_key = config.get("planner_api_key", "")
    model_name = config.get("planner_model", "")
    base_url = config.get("planner_base_url", "")
    
    llm = None
    if api_key:
        llm = ModelManager.create_llm(provider, api_key, model_name, base_url)
    
    relevant_memories = MemoryManager.get_relevant_memories_for_prompt(user_prompt, llm, limit=3)
    
    return {"relevant_memories": relevant_memories}

def planner(state: VideoAgentState) -> VideoAgentState:
    config = state["config"]
    provider = config.get("planner_provider", "openai")
    api_key = config.get("planner_api_key", "")
    model_name = config.get("planner_model", "")
    base_url = config.get("planner_base_url", "")
    
    llm = ModelManager.create_llm(provider, api_key, model_name, base_url)
    
    assets_info = ""
    if state.get("uploaded_assets"):
        assets_info = "\n\n可用素材：\n"
        for asset in state["uploaded_assets"]:
            assets_info += f"- {asset.filename} ({asset.file_type}): {asset.description}\n"
    
    memories_info = ""
    if state.get("relevant_memories") and state["relevant_memories"] != "无相关记忆":
        memories_info = f"\n\n相关记忆（请参考之前的创作经验）：\n{state['relevant_memories']}"
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个专业的视频策划师。请根据用户需求、可用素材和相关记忆，制定详细的视频制作计划。
        
        要求：
        1. 标题要吸引人且准确反映内容
        2. 主题清晰明确
        3. 风格要适合目标受众
        4. 合理分配视频时长
        5. 列出关键视觉元素
        6. 如果有可用素材，请在计划中考虑如何使用
        7. 如果有相关记忆，请参考之前的创作经验，保持风格一致性
        8. 返回JSON格式，包含title, theme, tone, target_audience, duration, scene_count, key_elements, color_palette, lighting字段
        """),
        ("human", "用户需求：{user_prompt}\n\n配置参数：{config_summary}\n{assets_info}{memories_info}"),
    ])
    
    config_summary = {
        "target_audience": config.get("target_audience", "通用"),
        "duration_min": config.get("duration_min", 15),
        "duration_max": config.get("duration_max", 60),
        "scene_count": config.get("scene_count", 5),
        "color_style": config.get("color_style", "明亮鲜艳"),
        "lighting_style": config.get("lighting_style", "自然光"),
        "camera_style": config.get("camera_style", "电影感")
    }
    
    chain = prompt | llm
    
    try:
        result = chain.invoke({
            "user_prompt": state["user_prompt"],
            "config_summary": str(config_summary),
            "assets_info": assets_info,
            "memories_info": memories_info
        })
        
        import json
        plan_data = json.loads(result)
        plan = VideoPlan(**plan_data)
    except Exception as e:
        plan = VideoPlan(
            title="AI生成视频",
            theme="用户需求视频",
            tone=config.get("video_style", "电影"),
            target_audience=config.get("target_audience", "通用"),
            duration=(config.get("duration_min", 15) + config.get("duration_max", 60)) / 2,
            scene_count=config.get("scene_count", 5),
            key_elements=["视频内容", "视觉效果"],
            color_palette=config.get("color_style", "明亮鲜艳"),
            lighting=config.get("lighting_style", "自然光")
        )
    
    return {"video_plan": plan}

def generate_prompt(state: VideoAgentState) -> VideoAgentState:
    config = state["config"]
    provider = config.get("prompt_provider", "claude")
    api_key = config.get("prompt_api_key", "")
    model_name = config.get("prompt_model", "")
    base_url = config.get("prompt_base_url", "")
    
    llm = ModelManager.create_llm(provider, api_key, model_name, base_url)
    
    plan = state["video_plan"]
    scene_prompts = []
    
    assets_info = ""
    if state.get("uploaded_assets"):
        assets_info = "\n\n可用素材（请在合适的场景中使用）：\n"
        for asset in state["uploaded_assets"]:
            assets_info += f"- {asset.filename} ({asset.file_type}): {asset.description}\n"
    
    for i in range(plan.scene_count):
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的视频提示词工程师。请根据视频计划和可用素材，为指定场景生成详细的视频生成API提示词。
            
要求：
1. 提示词要详细描述画面内容、动作、表情、场景
2. 包含镜头角度和运动方式
3. 指定画面比例和风格
4. 如果有合适的素材，请在提示词中引用并说明如何使用
5. 输出JSON格式，包含prompt, duration, aspect_ratio, camera_angle, style字段
"""),
            ("human", """视频计划：
标题：{title}
主题：{theme}
风格：{tone}
时长：{total_duration}秒
色彩：{color_palette}
光线：{lighting}

请为场景{scene_num}/{total_scenes}生成详细的视频提示词。
场景描述要求：{key_elements}
{assets_info}
"""),
        ])
        
        chain = prompt_template | llm
        
        result = chain.invoke({
            "title": plan.title,
            "theme": plan.theme,
            "tone": plan.tone,
            "total_duration": plan.duration,
            "color_palette": plan.color_palette,
            "lighting": plan.lighting,
            "scene_num": i + 1,
            "total_scenes": plan.scene_count,
            "key_elements": ", ".join(plan.key_elements),
            "assets_info": assets_info
        })
        
        try:
            import json
            scene_data = json.loads(result)
            scene_prompt = ScenePrompt(
                scene_number=i + 1,
                prompt=scene_data.get("prompt", ""),
                duration=scene_data.get("duration", plan.duration / plan.scene_count),
                aspect_ratio=scene_data.get("aspect_ratio", "16:9"),
                camera_angle=scene_data.get("camera_angle", "eye level"),
                style=scene_data.get("style", plan.tone),
                used_assets=scene_data.get("used_assets", [])
            )
        except Exception as e:
            scene_prompt = ScenePrompt(
                scene_number=i + 1,
                prompt=f"场景{i+1}: {plan.theme} - {', '.join(plan.key_elements)}",
                duration=plan.duration / plan.scene_count,
                aspect_ratio="16:9",
                camera_angle="eye level",
                style=plan.tone
            )
        
        scene_prompts.append(scene_prompt)
    
    return {"scene_prompts": scene_prompts}

def call_video_api(state: VideoAgentState) -> VideoAgentState:
    config = state["config"]
    video_api_name = config.get("video_api", "pika")
    video_api_key = config.get("video_api_key", "")
    video_api_url = config.get("video_api_url", "")
    
    scene_prompts = state["scene_prompts"]
    video_results = []
    
    for scene_prompt in scene_prompts:
        params = {
            "prompt": scene_prompt.prompt,
            "duration": scene_prompt.duration,
            "aspect_ratio": scene_prompt.aspect_ratio,
            "style": scene_prompt.style,
            "camera_angle": scene_prompt.camera_angle
        }
        
        if video_api_name == "seedance":
            params["model"] = config.get("seedance_model", "doubao-seedance-2-0-fast-260128")
            
            first_frame = config.get("seedance_first_frame", "")
            if first_frame:
                params["first_frame"] = first_frame
            
            last_frame = config.get("seedance_last_frame", "")
            if last_frame:
                params["last_frame"] = last_frame
        
        if video_api_name == "custom":
            params["custom_endpoint"] = config.get("video_custom_endpoint", "/generate")
            params["custom_method"] = config.get("video_custom_method", "POST")
            
            custom_headers = config.get("video_custom_headers", "{}")
            try:
                import json
                params["custom_headers"] = json.loads(custom_headers)
            except:
                params["custom_headers"] = {}
            
            custom_payload = config.get("video_custom_payload", "")
            if custom_payload:
                try:
                    import json
                    params["custom_payload"] = json.loads(custom_payload)
                except:
                    params["custom_payload"] = {}
        
        if scene_prompt.used_assets and state.get("uploaded_assets"):
            reference_files = []
            for asset_id in scene_prompt.used_assets:
                for asset in state["uploaded_assets"]:
                    if asset.asset_id == asset_id:
                        reference_files.append(asset.file_path)
                        break
            
            if reference_files:
                params["reference_files"] = reference_files
                
                image_files = [f for f in reference_files if f.lower().endswith(('jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp'))]
                video_files = [f for f in reference_files if f.lower().endswith(('mp4', 'mov', 'avi', 'mkv', 'webm'))]
                
                if image_files:
                    params["reference_images"] = image_files
                if video_files:
                    params["reference_videos"] = video_files
        
        result = VideoAPI.generate_video(
            video_api_name,
            video_api_key,
            video_api_url,
            params
        )
        
        video_result = VideoGenerationResult(
            scene_number=scene_prompt.scene_number,
            video_url=result.get("video_url", result.get("mock_data", {}).get("video_url", "")),
            duration=scene_prompt.duration,
            status=result.get("status", "success"),
            prompt_used=scene_prompt.prompt
        )
        
        video_results.append(video_result)
    
    return {"video_results": video_results}

def local_optimization(state: VideoAgentState) -> VideoAgentState:
    config = state["config"]
    provider = config.get("prompt_provider", "claude")
    api_key = config.get("prompt_api_key", "")
    model_name = config.get("prompt_model", "")
    base_url = config.get("prompt_base_url", "")
    
    llm = ModelManager.create_llm(provider, api_key, model_name, base_url)
    
    optimization_requests = state.get("optimization_requests", [])
    if not optimization_requests:
        return state
    
    scene_prompts = state.get("scene_prompts", [])
    
    for request in optimization_requests:
        scene_index = request.scene_number - 1
        if 0 <= scene_index < len(scene_prompts):
            original_prompt = scene_prompts[scene_index]
            
            optimization_prompt = ChatPromptTemplate.from_messages([
                ("system", """你是一个专业的视频提示词优化师。请根据用户的优化指令，对原有的视频提示词进行局部优化。
                
优化类型说明：
- style: 修改画面风格（如更电影化、更卡通等）
- character: 修改人物外观、表情、动作
- background: 修改背景环境、场景布置
- color: 修改色彩调色、色调
- composition: 修改构图、镜头角度、画面布局

要求：
1. 只修改与优化指令相关的部分，保持其他内容不变
2. 输出JSON格式，包含prompt字段
3. 不要重新生成整个提示词，只进行针对性优化
"""),
                ("human", """原始提示词：{original_prompt}
优化类型：{optimization_type}
优化指令：{instruction}

请对提示词进行局部优化：
"""),
            ])
            
            chain = optimization_prompt | llm
            
            result = chain.invoke({
                "original_prompt": original_prompt.prompt,
                "optimization_type": request.optimization_type,
                "instruction": request.instruction
            })
            
            try:
                import json
                optimized_data = json.loads(result)
                scene_prompts[scene_index].prompt = optimized_data.get("prompt", original_prompt.prompt)
            except Exception as e:
                pass
    
    return {"scene_prompts": scene_prompts}

def generate_subtitles(state: VideoAgentState) -> VideoAgentState:
    config = state["config"]
    provider = config.get("subtitle_provider", "gemini")
    api_key = config.get("subtitle_api_key", "")
    model_name = config.get("subtitle_model", "")
    base_url = config.get("subtitle_base_url", "")
    
    llm = ModelManager.create_llm(provider, api_key, model_name, base_url)
    
    plan = state["video_plan"]
    video_results = state["video_results"]
    
    subtitles = []
    current_time = 0.0
    
    for video_result in video_results:
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的字幕制作师。请根据视频内容生成简洁明了的字幕。
            
要求：
1. 字幕内容要简洁，符合场景时长
2. 返回JSON格式，包含text字段
"""),
            ("human", """视频场景{scene_num}：{prompt}
视频时长：{duration}秒
视频主题：{theme}

请生成该场景的字幕内容（不超过30个字）：
"""),
        ])
        
        chain = prompt_template | llm
        
        result = chain.invoke({
            "scene_num": video_result.scene_number,
            "prompt": video_result.prompt_used,
            "duration": video_result.duration,
            "theme": plan.theme
        })
        
        try:
            import json
            subtitle_data = json.loads(result)
            text = subtitle_data.get("text", video_result.prompt_used[:30] + "...")
        except Exception as e:
            text = video_result.prompt_used[:30] + "..."
        
        subtitle = VideoSubtitle(
            scene_number=video_result.scene_number,
            text=text,
            start_time=current_time,
            end_time=current_time + video_result.duration
        )
        
        subtitles.append(subtitle)
        current_time += video_result.duration
    
    return {"subtitles": subtitles}

def regenerate_single_scene(state: VideoAgentState) -> VideoAgentState:
    config = state["config"]
    video_api_name = config.get("video_api", "pika")
    video_api_key = config.get("video_api_key", "")
    video_api_url = config.get("video_api_url", "")
    
    optimization_requests = state.get("optimization_requests", [])
    if not optimization_requests:
        return state
    
    scene_prompts = state.get("scene_prompts", [])
    video_results = state.get("video_results", [])
    
    for request in optimization_requests:
        scene_index = request.scene_number - 1
        if 0 <= scene_index < len(scene_prompts):
            scene_prompt = scene_prompts[scene_index]
            
            params = {
                "prompt": scene_prompt.prompt,
                "duration": scene_prompt.duration,
                "aspect_ratio": scene_prompt.aspect_ratio,
                "style": scene_prompt.style,
                "camera_angle": scene_prompt.camera_angle
            }
            
            if scene_prompt.used_assets and state.get("uploaded_assets"):
                reference_files = []
                for asset_id in scene_prompt.used_assets:
                    for asset in state["uploaded_assets"]:
                        if asset.asset_id == asset_id:
                            reference_files.append(asset.file_path)
                            break
                
                if reference_files:
                    params["reference_files"] = reference_files
                    
                    image_files = [f for f in reference_files if f.lower().endswith(('jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp'))]
                    video_files = [f for f in reference_files if f.lower().endswith(('mp4', 'mov', 'avi', 'mkv', 'webm'))]
                    
                    if image_files:
                        params["reference_images"] = image_files
                    if video_files:
                        params["reference_videos"] = video_files
            
            result = VideoAPI.generate_video(
                video_api_name,
                video_api_key,
                video_api_url,
                params
            )
            
            if 0 <= scene_index < len(video_results):
                video_results[scene_index] = VideoGenerationResult(
                    scene_number=scene_prompt.scene_number,
                    video_url=result.get("video_url", result.get("mock_data", {}).get("video_url", "")),
                    duration=scene_prompt.duration,
                    status=result.get("status", "success"),
                    prompt_used=scene_prompt.prompt
                )
            else:
                video_results.append(VideoGenerationResult(
                    scene_number=scene_prompt.scene_number,
                    video_url=result.get("video_url", result.get("mock_data", {}).get("video_url", "")),
                    duration=scene_prompt.duration,
                    status=result.get("status", "success"),
                    prompt_used=scene_prompt.prompt
                ))
    
    return {"video_results": video_results, "optimization_requests": []}

def export_workflow(state: VideoAgentState) -> VideoAgentState:
    config = state["config"]
    
    plan = state["video_plan"]
    video_results = state["video_results"]
    subtitles = state["subtitles"]
    
    scenes = []
    for video_result in video_results:
        scenes.append({
            "scene_number": video_result.scene_number,
            "video_url": video_result.video_url,
            "duration": video_result.duration,
            "prompt": video_result.prompt_used,
            "status": video_result.status
        })
    
    audio_plan = {
        "background_music": config.get("audio_style", "背景音乐"),
        "sound_effects": []
    }
    
    subtitle_list = []
    for subtitle in subtitles:
        subtitle_list.append({
            "scene_number": subtitle.scene_number,
            "text": subtitle.text,
            "start_time": subtitle.start_time,
            "end_time": subtitle.end_time
        })
    
    total_duration = sum(video_result.duration for video_result in video_results)
    
    export_result = VideoExport(
        title=plan.title,
        scenes=scenes,
        audio_plan=audio_plan,
        subtitles=subtitle_list,
        total_duration=total_duration,
        export_format=config.get("export_format", "mp4"),
        metadata={
            "theme": plan.theme,
            "tone": plan.tone,
            "target_audience": plan.target_audience,
            "color_palette": plan.color_palette,
            "lighting": plan.lighting,
            "generated_at": "2026-07-06",
            "used_assets": [asset.asset_id for asset in state.get("uploaded_assets", [])]
        }
    )
    
    return {"export_result": export_result}

def save_generation_memory(state: VideoAgentState) -> VideoAgentState:
    video_plan = state.get("video_plan")
    scene_prompts = state.get("scene_prompts", [])
    video_results = state.get("video_results", [])
    subtitles = state.get("subtitles", [])
    
    if video_plan:
        video_plan_dict = {
            "title": video_plan.title,
            "theme": video_plan.theme,
            "tone": video_plan.tone,
            "target_audience": video_plan.target_audience,
            "duration": video_plan.duration,
            "scene_count": video_plan.scene_count,
            "key_elements": video_plan.key_elements,
            "color_palette": video_plan.color_palette,
            "lighting": video_plan.lighting
        }
        
        scene_prompts_dict = []
        for sp in scene_prompts:
            scene_prompts_dict.append({
                "scene_number": sp.scene_number,
                "prompt": sp.prompt,
                "duration": sp.duration,
                "aspect_ratio": sp.aspect_ratio,
                "camera_angle": sp.camera_angle,
                "style": sp.style,
                "used_assets": sp.used_assets
            })
        
        video_results_dict = []
        for vr in video_results:
            video_results_dict.append({
                "scene_number": vr.scene_number,
                "video_url": vr.video_url,
                "duration": vr.duration,
                "status": vr.status,
                "prompt_used": vr.prompt_used
            })
        
        subtitles_dict = []
        for st in subtitles:
            subtitles_dict.append({
                "scene_number": st.scene_number,
                "text": st.text,
                "start_time": st.start_time,
                "end_time": st.end_time
            })
        
        memory_id = MemoryManager.save_video_generation_memory(
            video_plan_dict,
            scene_prompts_dict,
            video_results_dict,
            subtitles_dict
        )
        
        return {"memory_saved": True, "memory_id": memory_id}
    
    return {"memory_saved": False}

workflow = StateGraph(VideoAgentState)

workflow.add_node("memory_retrieval", memory_retrieval)
workflow.add_node("planner", planner)
workflow.add_node("generate_prompt", generate_prompt)
workflow.add_node("local_optimization", local_optimization)
workflow.add_node("call_video_api", call_video_api)
workflow.add_node("regenerate_single_scene", regenerate_single_scene)
workflow.add_node("generate_subtitles", generate_subtitles)
workflow.add_node("export", export_workflow)
workflow.add_node("save_generation_memory", save_generation_memory)

workflow.set_entry_point("memory_retrieval")
workflow.add_edge("memory_retrieval", "planner")
workflow.add_edge("planner", "generate_prompt")
workflow.add_edge("generate_prompt", "local_optimization")
workflow.add_edge("local_optimization", "call_video_api")
workflow.add_edge("call_video_api", "generate_subtitles")
workflow.add_edge("generate_subtitles", "export")
workflow.add_edge("export", "save_generation_memory")
workflow.add_edge("save_generation_memory", END)

video_agent = workflow.compile()

optimization_workflow = StateGraph(VideoAgentState)

optimization_workflow.add_node("local_optimization", local_optimization)
optimization_workflow.add_node("regenerate_single_scene", regenerate_single_scene)
optimization_workflow.add_node("generate_subtitles", generate_subtitles)
optimization_workflow.add_node("export", export_workflow)
optimization_workflow.add_node("save_generation_memory", save_generation_memory)

optimization_workflow.set_entry_point("local_optimization")
optimization_workflow.add_edge("local_optimization", "regenerate_single_scene")
optimization_workflow.add_edge("regenerate_single_scene", "generate_subtitles")
optimization_workflow.add_edge("generate_subtitles", "export")
optimization_workflow.add_edge("export", "save_generation_memory")
optimization_workflow.add_edge("save_generation_memory", END)

optimization_agent = optimization_workflow.compile()
import streamlit as st
import os
import json
from dotenv import load_dotenv
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agent.video_agent import video_agent, optimization_agent, UploadedAsset, OptimizationRequest
from agent.image_agent import image_agent, ImageGenerationRequest, ImageGenerationResult
from agent.model_manager import ModelManager, VideoAPI, ImageAPI, AssetManager
from agent.memory_manager import MemoryManager

load_dotenv()

st.set_page_config(
    page_title="AI 多模态生成 Agent",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎬 AI 视频生成 Agent")
st.markdown("基于 LangGraph 的多模型视频生成工作流")

def init_session_state():
    if "mode" not in st.session_state:
        st.session_state.mode = "video"
    
    if "config" not in st.session_state:
        st.session_state.config = {
            "planner_provider": "openai",
            "planner_api_key": "",
            "planner_model": "",
            "planner_base_url": "",
            
            "prompt_provider": "claude",
            "prompt_api_key": "",
            "prompt_model": "",
            "prompt_base_url": "",
            
            "subtitle_provider": "gemini",
            "subtitle_api_key": "",
            "subtitle_model": "",
            "subtitle_base_url": "",
            
            "video_api": "pika",
            "video_api_key": "",
            "video_api_url": "",
            
            "video_style": "电影",
            "target_audience": "通用",
            "duration_min": 15,
            "duration_max": 60,
            "scene_count": 5,
            "color_style": "明亮鲜艳",
            "lighting_style": "自然光",
            "camera_style": "电影感",
            "export_format": "mp4",
            
            "image_planner_provider": "openai",
            "image_planner_api_key": "",
            "image_planner_model": "",
            "image_planner_base_url": "",
            
            "image_api": "dall-e",
            "image_api_key": "",
            "image_base_url": "",
            
            "image_width": 1024,
            "image_height": 1024,
            "image_style": "写实",
            "negative_prompt": "",
            "num_images": 1,
            "seed": -1
        }
    
    if "uploaded_assets" not in st.session_state:
        st.session_state.uploaded_assets = []
    
    if "optimization_requests" not in st.session_state:
        st.session_state.optimization_requests = []
    
    if "image_results" not in st.session_state:
        st.session_state.image_results = []

init_session_state()

def render_model_config(st_section, section_key, default_provider, title, description):
    st_section.subheader(title)
    st_section.markdown(description)
    
    provider = st_section.selectbox(
        "选择模型提供商",
        ModelManager.get_provider_list(),
        index=ModelManager.get_provider_list().index(default_provider),
        key=f"{section_key}_provider"
    )
    
    st.session_state.config[f"{section_key}_provider"] = provider
    
    info = ModelManager.get_provider_info(provider)
    
    if provider != "custom":
        st_section.markdown(f"**推荐模型**: {info.get('default_model', '')}")
        st_section.markdown(f"**描述**: {info.get('description', '')}")
        
        api_key = st_section.text_input(
            "API Key",
            value=st.session_state.config.get(f"{section_key}_api_key", ""),
            type="password",
            key=f"{section_key}_api_key",
            help=f"输入你的{info.get('display_name', '')} API Key"
        )
        st.session_state.config[f"{section_key}_api_key"] = api_key
        
        model_name = st_section.text_input(
            "模型名称",
            value=st.session_state.config.get(f"{section_key}_model", info.get('default_model', '')),
            key=f"{section_key}_model",
            help=f"输入模型名称，默认为{info.get('default_model', '')}"
        )
        st.session_state.config[f"{section_key}_model"] = model_name
        
        base_url = st_section.text_input(
            "API Base URL",
            value=st.session_state.config.get(f"{section_key}_base_url", info.get('base_url', '')),
            key=f"{section_key}_base_url",
            help=f"API服务器地址，默认为{info.get('base_url', '')}"
        )
        st.session_state.config[f"{section_key}_base_url"] = base_url
    else:
        api_key = st_section.text_input(
            "API Key",
            value=st.session_state.config.get(f"{section_key}_api_key", ""),
            type="password",
            key=f"{section_key}_api_key"
        )
        st.session_state.config[f"{section_key}_api_key"] = api_key
        
        model_name = st_section.text_input(
            "模型名称",
            value=st.session_state.config.get(f"{section_key}_model", ""),
            key=f"{section_key}_model",
            placeholder="例如：gpt-4o"
        )
        st.session_state.config[f"{section_key}_model"] = model_name
        
        base_url = st_section.text_input(
            "API Base URL",
            value=st.session_state.config.get(f"{section_key}_base_url", ""),
            key=f"{section_key}_base_url",
            placeholder="例如：https://api.example.com/v1"
        )
        st.session_state.config[f"{section_key}_base_url"] = base_url
    
    st_section.divider()

with st.sidebar:
    st.header("🎨 生成模式")
    
    mode = st.radio(
        "选择生成模式",
        ["video", "image"],
        index=["video", "image"].index(st.session_state.mode),
        format_func=lambda x: "🎬 视频生成" if x == "video" else "🖼️ 图片生成",
        key="mode"
    )
    
    st.markdown("---")
    
    st.header("⚙️ 工作流配置")
    
    if mode == "video":
        st.markdown("---")
        
        render_model_config(
            st,
            "planner",
            "openai",
            "1️⃣ Planner (策划师)",
            "负责分析用户需求，制定视频制作计划。建议使用逻辑推理能力强的模型。"
        )
        
        render_model_config(
            st,
            "prompt",
            "claude",
            "2️⃣ Generate Prompt (提示词生成)",
            "负责为每个场景生成详细的视频生成API提示词。建议使用创意写作能力强的模型。"
        )
        
        st.subheader("3️⃣ Call Video API (视频生成)")
        st.markdown("调用视频生成API生成实际视频。")
        
        video_api = st.selectbox(
            "选择视频API",
            VideoAPI.get_api_list(),
            index=VideoAPI.get_api_list().index(st.session_state.config.get("video_api", "pika")),
            format_func=lambda x: VideoAPI.get_api_info(x).get("display_name", x),
            key="video_api"
        )
        
        video_api_key = st.text_input(
            "视频API Key",
            value=st.session_state.config.get("video_api_key", ""),
            type="password",
            key="video_api_key"
        )
        
        video_api_url = st.text_input(
            "视频API Base URL",
            value=st.session_state.config.get("video_api_url", ""),
            key="video_api_url"
        )
        
        if video_api == "seedance":
            st.markdown("---")
            st.subheader("Seedance 专属参数")
            
            seedance_model = st.selectbox(
                "选择模型",
                ["doubao-seedance-2-0-fast-260128", "doubao-seedance-2-0-standard-260128", "doubao-seedance-2-0-pro-260128"],
                index=0,
                format_func=lambda x: {
                    "doubao-seedance-2-0-fast-260128": "FAST - 快速生成",
                    "doubao-seedance-2-0-standard-260128": "Standard - 标准质量",
                    "doubao-seedance-2-0-pro-260128": "Pro - 最高质量"
                }.get(x, x),
                key="seedance_model"
            )
            st.session_state.config["seedance_model"] = seedance_model
            
            st.markdown("**首尾帧控制**")
            first_frame = st.file_uploader(
                "上传首帧图片",
                type=["jpg", "jpeg", "png", "webp"],
                key="seedance_first_frame"
            )
            if first_frame:
                st.image(first_frame, caption="首帧预览", width=200)
                import base64
                first_frame_data = base64.b64encode(first_frame.read()).decode()
                st.session_state.config["seedance_first_frame"] = f"data:image/{first_frame.name.split('.')[-1]};base64,{first_frame_data}"
            else:
                st.session_state.config["seedance_first_frame"] = ""
            
            last_frame = st.file_uploader(
                "上传尾帧图片",
                type=["jpg", "jpeg", "png", "webp"],
                key="seedance_last_frame"
            )
            if last_frame:
                st.image(last_frame, caption="尾帧预览", width=200)
                import base64
                last_frame_data = base64.b64encode(last_frame.read()).decode()
                st.session_state.config["seedance_last_frame"] = f"data:image/{last_frame.name.split('.')[-1]};base64,{last_frame_data}"
            else:
                st.session_state.config["seedance_last_frame"] = ""
        
        if video_api == "custom":
            st.markdown("---")
            st.subheader("通用接口配置")
            
            custom_endpoint = st.text_input(
                "自定义Endpoint",
                value=st.session_state.config.get("video_custom_endpoint", "/generate"),
                key="video_custom_endpoint",
                help="API端点路径，如 /generate"
            )
            st.session_state.config["video_custom_endpoint"] = custom_endpoint
            
            custom_method = st.selectbox(
                "HTTP方法",
                ["POST", "GET"],
                index=0,
                key="video_custom_method"
            )
            st.session_state.config["video_custom_method"] = custom_method
            
            st.markdown("**自定义请求头（JSON格式）**")
            custom_headers = st.text_area(
                "请求头",
                value=st.session_state.config.get("video_custom_headers", "{}"),
                key="video_custom_headers",
                height=100,
                help='例如：{"Authorization": "Bearer YOUR_KEY", "Content-Type": "application/json"}'
            )
            st.session_state.config["video_custom_headers"] = custom_headers
            
            st.markdown("**自定义请求体（JSON格式）**")
            st.markdown("留空则自动使用默认参数（prompt, duration, aspect_ratio等）")
            custom_payload = st.text_area(
                "请求体",
                value=st.session_state.config.get("video_custom_payload", ""),
                key="video_custom_payload",
                height=150,
                help='例如：{"input": {"prompt": "{prompt}", "width": 1024, "height": 576}}'
            )
            st.session_state.config["video_custom_payload"] = custom_payload
        
        st.markdown("---")
        
        st.subheader("视频参数")
        video_style = st.selectbox(
            "视频风格",
            ["电影", "动画", "纪录片", "广告", "音乐视频", "游戏"],
            index=0,
            key="video_style"
        )
        
        target_audience = st.selectbox(
            "目标受众",
            ["通用", "儿童", "青少年", "成人", "商务人士"],
            index=0,
            key="target_audience"
        )
        
        duration_min = st.slider("最短时长(秒)", 5, 120, 15, key="duration_min")
        duration_max = st.slider("最长时长(秒)", 5, 180, 60, key="duration_max")
        scene_count = st.slider("场景数量", 1, 10, 5, key="scene_count")
        
        color_style = st.selectbox(
            "色彩风格",
            ["明亮鲜艳", "温暖柔和", "冷色调", "复古怀旧", "高对比度"],
            index=0,
            key="color_style"
        )
        
        lighting_style = st.selectbox(
            "光线风格",
            ["自然光", "影棚灯光", "电影感灯光", "柔光", "硬光"],
            index=0,
            key="lighting_style"
        )
        
        camera_style = st.selectbox(
            "镜头风格",
            ["电影感", "纪录片", "广告", "MV", "动画"],
            index=0,
            key="camera_style"
        )
        
        export_format = st.selectbox(
            "导出格式",
            ["mp4", "mov", "avi"],
            index=0,
            key="export_format"
        )
        
        st.markdown("---")
        
        render_model_config(
            st,
            "subtitle",
            "gemini",
            "4️⃣ Generate Subtitle (字幕生成)",
            "负责为生成的视频添加字幕。建议使用多语言支持好的模型。"
        )
        
    else:
        st.markdown("---")
        
        render_model_config(
            st,
            "image_planner",
            "openai",
            "1️⃣ Image Planner (图片策划师)",
            "负责分析用户需求，生成高质量的图像提示词。建议使用创意能力强的模型。"
        )
        
        st.subheader("2️⃣ Call Image API (图片生成)")
        st.markdown("调用图片生成API生成实际图片。")
        
        image_api = st.selectbox(
            "选择图片API",
            ImageAPI.get_api_list(),
            index=ImageAPI.get_api_list().index(st.session_state.config.get("image_api", "dall-e")),
            format_func=lambda x: ImageAPI.get_api_info(x).get("display_name", x),
            key="image_api"
        )
        
        image_api_key = st.text_input(
            "图片API Key",
            value=st.session_state.config.get("image_api_key", ""),
            type="password",
            key="image_api_key"
        )
        
        image_api_url = st.text_input(
            "图片API Base URL",
            value=st.session_state.config.get("image_base_url", ""),
            key="image_base_url"
        )
        
        if image_api == "custom":
            st.markdown("---")
            st.subheader("通用接口配置")
            
            custom_endpoint = st.text_input(
                "自定义Endpoint",
                value=st.session_state.config.get("image_custom_endpoint", "/generate"),
                key="image_custom_endpoint",
                help="API端点路径，如 /generate"
            )
            st.session_state.config["image_custom_endpoint"] = custom_endpoint
            
            custom_method = st.selectbox(
                "HTTP方法",
                ["POST", "GET"],
                index=0,
                key="image_custom_method"
            )
            st.session_state.config["image_custom_method"] = custom_method
            
            st.markdown("**自定义请求头（JSON格式）**")
            custom_headers = st.text_area(
                "请求头",
                value=st.session_state.config.get("image_custom_headers", "{}"),
                key="image_custom_headers",
                height=100,
                help='例如：{"Authorization": "Bearer YOUR_KEY", "Content-Type": "application/json"}'
            )
            st.session_state.config["image_custom_headers"] = custom_headers
            
            st.markdown("**自定义请求体（JSON格式）**")
            st.markdown("留空则自动使用默认参数（prompt, width, height等）")
            custom_payload = st.text_area(
                "请求体",
                value=st.session_state.config.get("image_custom_payload", ""),
                key="image_custom_payload",
                height=150,
                help='例如：{"input": {"prompt": "{prompt}", "width": 1024, "height": 1024}}'
            )
            st.session_state.config["image_custom_payload"] = custom_payload
        
        st.markdown("---")
        
        st.subheader("图片参数")
        image_width = st.number_input("宽度", 256, 4096, 1024, key="image_width")
        image_height = st.number_input("高度", 256, 4096, 1024, key="image_height")
        image_style = st.selectbox(
            "图片风格",
            ["写实", "动漫", "插画", "油画", "素描", "3D渲染", "赛博朋克", "复古"],
            index=0,
            key="image_style"
        )
        
        negative_prompt = st.text_input(
            "负面提示词",
            value=st.session_state.config.get("negative_prompt", ""),
            placeholder="输入不想要的元素，如：模糊、低质量、水印...",
            key="negative_prompt"
        )
        
        num_images = st.slider("生成数量", 1, 10, 1, key="num_images")
        seed = st.number_input("随机种子 (-1为随机)", -1, 999999, -1, key="seed")
    
    st.markdown("---")

st.subheader("📁 素材上传")
st.markdown("上传图片、视频、音频或文本素材，AI将在生成视频时参考这些素材")

col_upload, col_auto_desc = st.columns([3, 1])
with col_upload:
    uploaded_files = st.file_uploader(
        "上传素材文件",
        type=["jpg", "jpeg", "png", "gif", "bmp", "webp", "mp4", "mov", "avi", "mkv", "webm", "mp3", "wav", "ogg", "flac", "txt", "md", "json"],
        accept_multiple_files=True,
        help="支持图片、视频、音频、文本等格式"
    )
with col_auto_desc:
    auto_desc = st.checkbox("AI自动描述", value=True, help="使用AI自动生成素材描述")

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_data = uploaded_file.read()
        
        file_type = AssetManager.get_file_type(uploaded_file.name)
        default_description = ""
        
        if auto_desc and file_type == "text":
            with st.spinner(f"AI正在分析 {uploaded_file.name}..."):
                try:
                    provider = st.session_state.config.get("prompt_provider", "openai")
                    api_key = st.session_state.config.get("prompt_api_key", "")
                    model_name = st.session_state.config.get("prompt_model", "")
                    base_url = st.session_state.config.get("prompt_base_url", "")
                    
                    if api_key:
                        llm = ModelManager.create_llm(provider, api_key, model_name, base_url)
                        temp_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", f"temp_{uploaded_file.name}")
                        with open(temp_path, "wb") as f:
                            f.write(file_data)
                        default_description = AssetManager.describe_asset(temp_path, llm)
                        os.remove(temp_path)
                except Exception as e:
                    st.warning(f"AI描述失败，将使用默认描述: {str(e)}")
        
        description = st.text_input(
            f"为 {uploaded_file.name} 添加描述",
            value=default_description,
            key=f"desc_{uploaded_file.name}",
            placeholder="描述这个素材的内容和用途..."
        )
        
        save_btn = st.button(f"保存 {uploaded_file.name}", key=f"save_{uploaded_file.name}")
        if save_btn:
            asset_info = AssetManager.save_asset(file_data, uploaded_file.name, description)
            uploaded_asset = UploadedAsset(
                asset_id=asset_info["asset_id"],
                filename=asset_info["filename"],
                file_type=asset_info["file_type"],
                file_path=asset_info["file_path"],
                description=asset_info["description"]
            )
            st.session_state.uploaded_assets.append(uploaded_asset)
            st.success(f"✅ 素材 {uploaded_file.name} 已保存")

if st.session_state.uploaded_assets:
    st.subheader("已上传的素材")
    for i, asset in enumerate(st.session_state.uploaded_assets):
        col_asset, col_desc, col_action = st.columns([2, 4, 1])
        with col_asset:
            st.markdown(f"**{asset.filename}**")
            st.markdown(f"类型: {asset.file_type}")
        with col_desc:
            st.markdown(f"描述: {asset.description}")
            if asset.file_type == "image":
                try:
                    st.image(asset.file_path, width=200)
                except:
                    pass
            elif asset.file_type == "video":
                try:
                    st.video(asset.file_path)
                except:
                    pass
            elif asset.file_type == "audio":
                try:
                    st.audio(asset.file_path)
                except:
                    pass
        with col_action:
            if st.button(f"删除", key=f"del_{i}"):
                AssetManager.delete_asset(asset.asset_id)
                del st.session_state.uploaded_assets[i]
                st.rerun()

st.subheader("🧠 记忆管理")
st.markdown("查看和管理历史创作记忆，AI会在生成时自动参考相关记忆")

tab1, tab2, tab3 = st.tabs(["📚 记忆列表", "🔍 搜索记忆", "✏️ 添加记忆"])

with tab1:
    memories = MemoryManager.list_memories()
    if memories:
        for i, memory in enumerate(memories):
            with st.expander(f"📄 {memory.metadata.get('title', '未命名')} - {memory.created_at[:19]}"):
                st.markdown(f"**类型**: {memory.metadata.get('type', 'general')}")
                if "tags" in memory.metadata:
                    st.markdown(f"**标签**: {', '.join(memory.metadata['tags'])}")
                st.markdown(f"**ID**: {memory.memory_id}")
                st.markdown("---")
                st.markdown(memory.content)
                
                col_mem, col_del = st.columns([4, 1])
                with col_mem:
                    if st.button(f"引用此记忆", key=f"use_mem_{i}"):
                        if "referenced_memories" not in st.session_state:
                            st.session_state.referenced_memories = []
                        st.session_state.referenced_memories.append(memory)
                        st.success(f"✅ 已引用记忆: {memory.metadata.get('title', '')}")
                with col_del:
                    if st.button(f"🗑️", key=f"del_mem_{i}"):
                        MemoryManager.delete_memory(memory.memory_id)
                        st.rerun()
    else:
        st.info("暂无记忆，生成视频后会自动保存相关记忆")

with tab2:
    search_query = st.text_input("搜索记忆", placeholder="输入关键词搜索相关记忆...")
    if search_query:
        with st.spinner("🔍 正在搜索记忆..."):
            provider = st.session_state.config.get("planner_provider", "openai")
            api_key = st.session_state.config.get("planner_api_key", "")
            model_name = st.session_state.config.get("planner_model", "")
            base_url = st.session_state.config.get("planner_base_url", "")
            
            llm = None
            if api_key:
                try:
                    llm = ModelManager.create_llm(provider, api_key, model_name, base_url)
                except:
                    pass
            
            results = MemoryManager.search_memories(search_query, llm, limit=5)
            
            if results:
                st.success(f"找到 {len(results)} 条相关记忆")
                for i, memory in enumerate(results):
                    with st.expander(f"📄 {memory.metadata.get('title', '未命名')} (相关度高)"):
                        st.markdown(f"**类型**: {memory.metadata.get('type', 'general')}")
                        st.markdown(f"**创建时间**: {memory.created_at}")
                        if "tags" in memory.metadata:
                            st.markdown(f"**标签**: {', '.join(memory.metadata['tags'])}")
                        st.markdown("---")
                        st.markdown(memory.content[:500] + "..." if len(memory.content) > 500 else memory.content)
                        
                        if st.button(f"引用此记忆", key=f"search_use_{i}"):
                            if "referenced_memories" not in st.session_state:
                                st.session_state.referenced_memories = []
                            st.session_state.referenced_memories.append(memory)
                            st.success(f"✅ 已引用记忆: {memory.metadata.get('title', '')}")
            else:
                st.info("未找到相关记忆")

with tab3:
    memory_title = st.text_input("记忆标题", placeholder="输入记忆标题...")
    memory_type = st.selectbox("记忆类型", ["general", "video_generation", "prompt_template", "style_guide", "other"])
    memory_tags = st.text_input("标签（逗号分隔）", placeholder="例如：科幻, 电影, 特效")
    memory_content = st.text_area("记忆内容", placeholder="输入记忆详细内容...", height=200)
    
    if st.button("保存记忆", type="primary"):
        if memory_content:
            metadata = {
                "title": memory_title if memory_title else memory_content[:50],
                "type": memory_type,
                "tags": [t.strip() for t in memory_tags.split(",")] if memory_tags else []
            }
            memory_id = MemoryManager.save_memory(memory_content, metadata)
            st.success(f"✅ 记忆已保存，ID: {memory_id}")
        else:
            st.warning("请输入记忆内容")

if st.session_state.mode == "video":
    user_prompt = st.text_area(
        "📝 输入视频需求",
        placeholder="请描述你想要生成的视频内容...\n\n例如：\n- 一个关于人工智能未来的科幻短片\n- 一段展示城市风景的延时摄影\n- 一个产品介绍的广告视频\n- 一个旅行vlog的脚本",
        height=150
    )
else:
    user_prompt = st.text_area(
        "🎨 输入图片需求",
        placeholder="请描述你想要生成的图片内容...\n\n例如：\n- 一只可爱的猫咪在阳光下睡觉\n- 赛博朋克风格的未来城市夜景\n- 水彩画风格的风景\n- 3D渲染的产品展示图",
        height=150
    )

col1, col2 = st.columns(2)

with col1:
    generate_btn = st.button("🚀 开始生成", type="primary", use_container_width=True)

with col2:
    clear_btn = st.button("🗑️ 清除内容", use_container_width=True)

if clear_btn:
    for key in ["video_plan", "scene_prompts", "video_results", "subtitles", "export_result", "uploaded_assets", "optimization_requests", "referenced_memories", "image_results"]:
        if key in st.session_state:
            del st.session_state[key]

if generate_btn and user_prompt:
    if st.session_state.mode == "video":
        required_apis = [
            ("planner", "策划师"),
            ("prompt", "提示词生成"),
            ("subtitle", "字幕生成"),
            ("video", "视频API")
        ]
        
        missing_apis = []
        for key, name in required_apis:
            api_key = st.session_state.config.get(f"{key}_api_key", "")
            if not api_key:
                missing_apis.append(name)
        
        if missing_apis:
            st.error(f"请先配置以下API Key：{', '.join(missing_apis)}")
        else:
            with st.spinner("🎬 AI 正在处理视频生成工作流..."):
                result = video_agent.invoke({
                    "user_prompt": user_prompt,
                    "config": st.session_state.config,
                    "uploaded_assets": st.session_state.uploaded_assets,
                    "optimization_requests": st.session_state.optimization_requests
                })
                
                st.session_state.video_plan = result.get("video_plan")
                st.session_state.scene_prompts = result.get("scene_prompts")
                st.session_state.video_results = result.get("video_results")
                st.session_state.subtitles = result.get("subtitles")
                st.session_state.export_result = result.get("export_result")
                
                st.success("✅ 视频生成工作流完成！")
                
                if result.get("memory_saved"):
                    st.info(f"🧠 创作记忆已保存，可在记忆管理中查看")
                
                if result.get("relevant_memories") and result["relevant_memories"] != "无相关记忆":
                    with st.expander("📖 AI参考的相关记忆", expanded=False):
                        st.markdown(result["relevant_memories"])
                
                st.divider()
    else:
        required_apis = [
            ("image_planner", "图片策划师"),
            ("image", "图片API")
        ]
        
        missing_apis = []
        for key, name in required_apis:
            api_key = st.session_state.config.get(f"{key}_api_key", "")
            if not api_key:
                missing_apis.append(name)
        
        if missing_apis:
            st.error(f"请先配置以下API Key：{', '.join(missing_apis)}")
        else:
            with st.spinner("🖼️ AI 正在处理图片生成工作流..."):
                result = image_agent.invoke({
                    "user_prompt": user_prompt,
                    "config": st.session_state.config,
                    "uploaded_assets": st.session_state.uploaded_assets
                })
                
                st.session_state.image_results = result.get("image_results", [])
                
                st.success("✅ 图片生成工作流完成！")
                
                if result.get("memory_saved"):
                    st.info(f"🧠 创作记忆已保存，可在记忆管理中查看")
                
                if result.get("relevant_memories") and result["relevant_memories"] != "无相关记忆":
                    with st.expander("📖 AI参考的相关记忆", expanded=False):
                        st.markdown(result["relevant_memories"])
                
                st.divider()
                
                if st.session_state.image_results:
                    st.subheader("🖼️ 生成的图片")
                    
                    num_cols = min(3, len(st.session_state.image_results))
                    cols = st.columns(num_cols)
                    
                    for i, result in enumerate(st.session_state.image_results):
                        with cols[i % num_cols]:
                            st.image(result.image_url, caption=f"图片 {i+1}", use_column_width=True)
                            st.markdown(f"**提示词**: {result.prompt_used[:50]}...")
                            st.markdown(f"**尺寸**: {result.width} × {result.height}")
                            if result.revised_prompt:
                                st.markdown(f"**修订提示词**: {result.revised_prompt[:50]}...")
                            if result.seed != -1:
                                st.markdown(f"**Seed**: {result.seed}")
            
            if st.session_state.mode == "video":
                if result.get("video_plan"):
                    plan = result["video_plan"]
                    
                    col_a, col_b = st.columns([1, 2])
                    
                    with col_a:
                        st.subheader(f"📽️ {plan.title}")
                        st.markdown(f"""
                        **主题**: {plan.theme}
                        **风格**: {plan.tone}
                        **目标受众**: {plan.target_audience}
                        **总时长**: {plan.duration:.1f}秒
                        **场景数量**: {plan.scene_count}
                        **色彩风格**: {plan.color_palette}
                        **光线风格**: {plan.lighting}
                        **关键元素**: {', '.join(plan.key_elements)}
                        """)
                    
                    with col_b:
                        st.subheader("🎥 场景列表")
                        
                        if result.get("scene_prompts"):
                            for scene_prompt in result["scene_prompts"]:
                                with st.expander(f"场景 {scene_prompt.scene_number} ({scene_prompt.duration:.1f}秒)", expanded=True):
                                    st.markdown(f"""
                                    **提示词**: {scene_prompt.prompt}
                                    **画面比例**: {scene_prompt.aspect_ratio}
                                    **拍摄角度**: {scene_prompt.camera_angle}
                                    **画面风格**: {scene_prompt.style}
                                    """)
                                    
                                    with st.form(key=f"opt_form_{scene_prompt.scene_number}"):
                                        st.subheader(f"局部优化 - 场景 {scene_prompt.scene_number}")
                                        opt_type = st.selectbox(
                                            "优化类型",
                                            ["style", "character", "background", "color", "composition"],
                                            key=f"opt_type_{scene_prompt.scene_number}"
                                        )
                                        opt_instructions = {
                                            "style": "修改画面风格（如更电影化、更卡通、更写实等）",
                                            "character": "修改人物外观、表情、动作、服装等",
                                            "background": "修改背景环境、场景布置、道具等",
                                            "color": "修改色彩调色、色调、饱和度等",
                                            "composition": "修改构图、镜头角度、画面布局等"
                                        }
                                        st.markdown(f"💡 {opt_instructions[opt_type]}")
                                        opt_instruction = st.text_input(
                                            "优化指令",
                                            key=f"opt_inst_{scene_prompt.scene_number}",
                                            placeholder=f"例如：让画面更{opt_type}..."
                                        )
                                        if st.form_submit_button(f"应用优化到场景 {scene_prompt.scene_number}"):
                                            if opt_instruction:
                                                request = OptimizationRequest(
                                                    scene_number=scene_prompt.scene_number,
                                                    optimization_type=opt_type,
                                                    instruction=opt_instruction
                                                )
                                                st.session_state.optimization_requests.append(request)
                                                st.success(f"✅ 已添加优化请求到场景 {scene_prompt.scene_number}")
                                                st.info("点击下方'执行优化'按钮即可重新生成优化后的场景")
                
                if result.get("video_results"):
                    st.subheader("📹 视频生成结果")
                    for video_result in result["video_results"]:
                        st.markdown(f"""
                        **场景 {video_result.scene_number}**: 
                        - 视频URL: {video_result.video_url}
                        - 时长: {video_result.duration:.1f}秒
                        - 状态: {video_result.status}
                        """)
                
                if result.get("subtitles"):
                    st.subheader("📝 字幕")
                    for subtitle in result["subtitles"]:
                        st.markdown(f"**[{subtitle.start_time:.1f}s-{subtitle.end_time:.1f}s]** {subtitle.text}")
                
                if result.get("export_result"):
                    with st.expander("📋 导出配置", expanded=False):
                        export = result["export_result"]
                        st.json(export.dict())
            
            if st.session_state.mode == "video" and st.session_state.optimization_requests:
                st.subheader("🔧 待处理的优化请求")
                for i, req in enumerate(st.session_state.optimization_requests):
                    st.markdown(f"**请求 {i+1}**: 场景{req.scene_number} - {req.optimization_type} - {req.instruction}")
                
                if st.button("🚀 执行优化", type="secondary"):
                    with st.spinner("🔧 AI 正在执行局部优化..."):
                        opt_result = optimization_agent.invoke({
                            "user_prompt": user_prompt,
                            "config": st.session_state.config,
                            "video_plan": st.session_state.video_plan,
                            "scene_prompts": st.session_state.scene_prompts,
                            "video_results": st.session_state.video_results,
                            "uploaded_assets": st.session_state.uploaded_assets,
                            "optimization_requests": st.session_state.optimization_requests
                        })
                        
                        st.session_state.scene_prompts = opt_result.get("scene_prompts")
                        st.session_state.video_results = opt_result.get("video_results")
                        st.session_state.subtitles = opt_result.get("subtitles")
                        st.session_state.export_result = opt_result.get("export_result")
                        st.session_state.optimization_requests = []
                        
                        if opt_result.get("memory_saved"):
                            st.info(f"🧠 优化后的创作记忆已保存")
                        
                        st.success("✅ 局部优化完成！")
                        st.rerun()
            else:
                st.error("❌ 生成失败，请重试")
elif generate_btn and not user_prompt:
    st.warning("⚠️ 请先输入生成需求")

st.markdown("---")
st.info("💡 **提示**: 不同阶段可以使用不同的AI模型，以获得最佳效果。建议Planner使用逻辑推理强的模型，Prompt Generator使用创意写作强的模型。")
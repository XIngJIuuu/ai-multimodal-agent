# AI 多模态生成 Agent

基于 LangGraph 和 Streamlit 构建的 AI 视频/图片生成平台，支持多种主流 AI 模型，提供素材上传、局部优化、记忆模块等功能。

## 功能特性

### 🎬 视频生成模式
- 支持 Pika、Runway、Sora、Seedance、OpenAI、Claude 等多种视频生成 API
- 完整工作流：Planner → Generate Prompt → Call Video API → Generate Subtitles → Export
- 支持首尾帧控制（Seedance）
- 局部优化：可针对特定场景进行风格、人物、背景、色彩、构图优化

### 🖼️ 图片生成模式
- 支持 DALL-E 3、Stable Diffusion、Midjourney、Leonardo AI、Bing 等多种图片生成 API
- 支持自定义尺寸、风格、负面提示词
- 一次生成多张图片
- 支持固定随机种子保证结果可重复

### 📁 素材管理
- 支持上传图片、视频、音频、文本等素材
- AI 自动描述素材内容
- 素材预览和引用

### 🧠 记忆模块
- 自动保存创作记忆（视频计划、场景提示词、视频结果、字幕等）
- AI 生成前自动检索相关记忆
- 用户可主动搜索、引用、删除记忆

### 🔌 通用接口
- 支持配置任意第三方 API
- 自定义请求头、请求体、HTTP 方法

## 项目结构

```
video/
├── agent/                  # 核心 Agent 模块
│   ├── __init__.py         # 包初始化
│   ├── model_manager.py    # 模型管理（LLM、视频API、图片API）
│   ├── video_agent.py      # 视频生成工作流
│   ├── image_agent.py      # 图片生成工作流
│   └── memory_manager.py   # 记忆管理模块
├── ui/                     # 用户界面
│   └── app.py              # Streamlit 应用
├── memory/                 # 记忆存储目录（自动创建）
├── assets/                 # 素材存储目录（自动创建）
├── .env.example            # 环境变量示例
├── requirements.txt        # 依赖列表
└── README.md               # 使用指南
```

## 安装步骤

### 1. 克隆项目

```bash
cd e:\project\video
```

### 2. 创建虚拟环境（推荐）

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 配置环境变量

复制 `.env.example` 为 `.env`，并填入 API Key：

```bash
cp .env.example .env
```

编辑 `.env` 文件：

```env
# Planner 模型（推荐使用逻辑推理能力强的模型）
PLANNER_API_KEY=your-openai-api-key
PLANNER_MODEL=gpt-4o

# Prompt 生成模型（推荐使用创意写作能力强的模型）
PROMPT_API_KEY=your-claude-api-key
PROMPT_MODEL=claude-3-5-sonnet-20240620

# 字幕生成模型（推荐使用多语言支持好的模型）
SUBTITLE_API_KEY=your-gemini-api-key
SUBTITLE_MODEL=geminipro

# 视频生成 API
VIDEO_API_KEY=your-video-api-key

# 图片生成 API
IMAGE_API_KEY=your-image-api-key
```

## 启动应用

```bash
streamlit run ui/app.py
```

访问 http://localhost:8501 即可使用应用。

## 使用指南

### 一、视频生成

#### 1. 配置模型

在侧边栏配置各环节使用的模型：

| 环节 | 推荐模型 | 说明 |
|------|----------|------|
| Planner | GPT-4o / Claude 3.5 | 分析需求，制定视频计划 |
| Generate Prompt | Claude 3.5 / GPT-4o | 生成场景提示词 |
| Call Video API | Pika / Runway / Seedance | 调用视频生成API |
| Generate Subtitles | Gemini / GPT-4o | 生成字幕 |

#### 2. 输入需求

在主界面输入视频需求，例如：
- 一个关于人工智能未来的科幻短片
- 一段展示城市风景的延时摄影
- 一个产品介绍的广告视频

#### 3. 配置视频参数

- **视频风格**：电影、动画、纪录片、广告、音乐视频、游戏
- **目标受众**：通用、儿童、青少年、成人、商务人士
- **时长**：5-180秒
- **场景数量**：1-10个
- **视觉风格**：色彩、光线、镜头风格

#### 4. Seedance 专属配置

选择 Seedance AI 作为视频 API 时，可配置：

| 模型 | 描述 |
|------|------|
| FAST | 快速生成，适合快速预览 |
| Standard | 标准质量，平衡速度与质量 |
| Pro | 最高质量，适合最终输出 |

**首尾帧控制**：上传首帧和尾帧图片，精确控制视频开头和结尾画面。

#### 5. 局部优化

在场景列表中，可对每个场景进行单独优化：

| 优化类型 | 说明 |
|----------|------|
| style | 修改画面风格（更电影化、卡通、写实等） |
| character | 修改人物外观、表情、动作、服装 |
| background | 修改背景环境、场景布置、道具 |
| color | 修改色彩调色、色调、饱和度 |
| composition | 修改构图、镜头角度、画面布局 |

### 二、图片生成

#### 1. 切换模式

在侧边栏顶部切换到「图片生成」模式。

#### 2. 配置模型

选择图片生成 API：DALL-E 3、Stable Diffusion、Midjourney、Leonardo AI、Bing、通用接口。

#### 3. 配置参数

- **尺寸**：宽度和高度（256-4096像素）
- **风格**：写实、动漫、插画、油画、素描、3D渲染、赛博朋克、复古
- **负面提示词**：排除不想要的元素（模糊、低质量、水印等）
- **生成数量**：1-10张图片
- **随机种子**：固定种子保证结果可重复

#### 4. 输入需求

输入图片描述，例如：
- 一只可爱的猫咪在阳光下睡觉
- 赛博朋克风格的未来城市夜景
- 水彩画风格的风景

### 三、素材管理

#### 上传素材

支持上传以下类型的素材：
- 图片（jpg、jpeg、png、webp）
- 视频（mp4、mov、avi）
- 音频（mp3、wav）
- 文本（txt）

上传后系统会自动使用 AI 生成素材描述，方便后续引用。

#### 使用素材

在生成视频/图片时，AI 会自动参考已上传的素材，将其融入提示词中。

### 四、记忆模块

#### 自动记忆

每次生成完成后，系统会自动保存创作记忆，包括：
- 视频/图片计划
- 提示词
- 生成结果
- 字幕（视频模式）

#### 搜索记忆

- **关键词搜索**：基于标题、内容、标签匹配
- **语义搜索**：使用 AI 评估记忆与查询的相关性

#### 引用记忆

在记忆列表中点击「引用」按钮，将记忆内容添加到当前生成任务中。

### 五、通用接口配置

当需要调用未内置的 API 时，选择「通用接口」：

#### 配置项

| 配置项 | 说明 |
|--------|------|
| Base URL | API 基础地址 |
| Endpoint | API 端点路径，如 `/generate` |
| HTTP 方法 | POST 或 GET |
| 请求头 | JSON 格式，如 `{"Authorization": "Bearer YOUR_KEY"}` |
| 请求体 | JSON 格式，留空则使用默认参数 |

#### 示例配置

**视频生成 API：**
```
Base URL: https://api.example.com
Endpoint: /video/generate
请求头: {"Authorization": "Bearer sk-xxx"}
请求体: {"input": {"prompt": "{prompt}", "duration": 10}}
```

**图片生成 API：**
```
Base URL: https://api.example.com
Endpoint: /image/generate
请求头: {"Authorization": "Bearer sk-xxx"}
请求体: {"prompt": "{prompt}", "width": 1024, "height": 1024}
```

## 支持的 API

### 视频生成 API

| API | 基础 URL | 说明 |
|-----|----------|------|
| Pika | https://api.pika.art | Pika Labs 视频生成 |
| Runway | https://api.runwayml.com | Runway ML 视频生成 |
| Sora | https://api.openai.com | OpenAI Sora 视频生成 |
| Seedance | https://api.seedance.ai | 支持首尾帧控制 |
| OpenAI | https://api.openai.com | OpenAI 视频生成 |
| Claude | https://api.anthropic.com | Anthropic 视频生成 |
| Gemini | https://generativelanguage.googleapis.com | Google Gemini 视频生成 |
| DeepSeek | https://api.deepseek.com | DeepSeek 视频生成 |
| 通用接口 | 自定义 | 支持任意 API |

### 图片生成 API

| API | 基础 URL | 说明 |
|-----|----------|------|
| DALL-E 3 | https://api.openai.com | OpenAI 图像生成 |
| Stable Diffusion | https://api.stability.ai | Stability AI |
| Midjourney | https://api.midjourney.com | Midjourney |
| Leonardo AI | https://cloud.leonardo.ai | Leonardo AI |
| Bing | https://api.bing.microsoft.com | Microsoft Bing |
| 通用接口 | 自定义 | 支持任意 API |

## 依赖说明

```
langchain>=0.2.0
langgraph>=0.1.0
streamlit>=1.30.0
python-dotenv>=1.0.0
requests>=2.31.0
pydantic>=2.0.0
```

## 注意事项

1. **API Key 安全**：请妥善保管您的 API Key，不要泄露给他人
2. **网络连接**：确保网络连接稳定，API 调用可能需要较长时间
3. **内存使用**：生成视频和图片可能占用较多内存，请确保系统资源充足
4. **文件大小限制**：素材上传有大小限制，建议压缩后上传

## 许可证

MIT License

import os
import json
import re
from typing import List, Dict, Optional, Any
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel

MEMORY_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "memory")
os.makedirs(MEMORY_DIR, exist_ok=True)

class MemoryEntry:
    def __init__(self, memory_id: str, content: str, metadata: Dict[str, Any]):
        self.memory_id = memory_id
        self.content = content
        self.metadata = metadata
        self.created_at = metadata.get("created_at", datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "content": self.content,
            "metadata": self.metadata,
            "created_at": self.created_at
        }
    
    def to_markdown(self) -> str:
        lines = []
        lines.append(f"# {self.metadata.get('title', '未命名记忆')}")
        lines.append("")
        lines.append(f"**ID**: {self.memory_id}")
        lines.append(f"**创建时间**: {self.created_at}")
        lines.append(f"**类型**: {self.metadata.get('type', 'general')}")
        if "tags" in self.metadata:
            lines.append(f"**标签**: {', '.join(self.metadata.get('tags', []))}")
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append(self.content)
        return "\n".join(lines)

class MemoryManager:
    @staticmethod
    def save_memory(content: str, metadata: Dict[str, Any] = None) -> str:
        if metadata is None:
            metadata = {}
        
        memory_id = f"mem_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.urandom(4).hex()}"
        
        if "created_at" not in metadata:
            metadata["created_at"] = datetime.now().isoformat()
        if "title" not in metadata:
            metadata["title"] = content[:50] + "..." if len(content) > 50 else content
        
        memory = MemoryEntry(memory_id, content, metadata)
        
        md_content = memory.to_markdown()
        file_path = os.path.join(MEMORY_DIR, f"{memory_id}.md")
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        
        return memory_id
    
    @staticmethod
    def load_memory(memory_id: str) -> Optional[MemoryEntry]:
        file_path = os.path.join(MEMORY_DIR, f"{memory_id}.md")
        
        if not os.path.exists(file_path):
            return None
        
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        metadata = MemoryManager._parse_metadata_from_md(content)
        memory_content = MemoryManager._extract_content_from_md(content)
        
        return MemoryEntry(memory_id, memory_content, metadata)
    
    @staticmethod
    def _parse_metadata_from_md(content: str) -> Dict[str, Any]:
        metadata = {}
        
        lines = content.split("\n")
        i = 0
        
        while i < len(lines) and lines[i].strip():
            if lines[i].startswith("# "):
                metadata["title"] = lines[i][2:].strip()
            elif lines[i].startswith("**ID**:"):
                metadata["memory_id"] = lines[i].replace("**ID**:", "").strip()
            elif lines[i].startswith("**创建时间**:"):
                metadata["created_at"] = lines[i].replace("**创建时间**:", "").strip()
            elif lines[i].startswith("**类型**:"):
                metadata["type"] = lines[i].replace("**类型**:", "").strip()
            elif lines[i].startswith("**标签**:"):
                tags_str = lines[i].replace("**标签**:", "").strip()
                metadata["tags"] = [t.strip() for t in tags_str.split(",")] if tags_str else []
            i += 1
        
        return metadata
    
    @staticmethod
    def _extract_content_from_md(content: str) -> str:
        lines = content.split("\n")
        
        for i, line in enumerate(lines):
            if line.strip() == "---":
                return "\n".join(lines[i+1:]).strip()
        
        return content
    
    @staticmethod
    def list_memories() -> List[MemoryEntry]:
        memories = []
        
        for filename in os.listdir(MEMORY_DIR):
            if filename.endswith(".md") and not filename.startswith("."):
                memory_id = filename[:-3]
                memory = MemoryManager.load_memory(memory_id)
                if memory:
                    memories.append(memory)
        
        memories.sort(key=lambda m: m.created_at, reverse=True)
        
        return memories
    
    @staticmethod
    def search_memories(query: str, llm: BaseChatModel = None, limit: int = 5) -> List[MemoryEntry]:
        all_memories = MemoryManager.list_memories()
        
        if not all_memories:
            return []
        
        if llm:
            return MemoryManager._semantic_search(query, all_memories, llm, limit)
        else:
            return MemoryManager._keyword_search(query, all_memories, limit)
    
    @staticmethod
    def _keyword_search(query: str, memories: List[MemoryEntry], limit: int) -> List[MemoryEntry]:
        query_lower = query.lower()
        query_words = query_lower.split()
        
        scored_memories = []
        
        for memory in memories:
            score = 0
            content_lower = memory.content.lower()
            title_lower = memory.metadata.get("title", "").lower()
            
            for word in query_words:
                score += content_lower.count(word)
                score += title_lower.count(word) * 2
                
                if "tags" in memory.metadata:
                    for tag in memory.metadata["tags"]:
                        if word in tag.lower():
                            score += 3
            
            if score > 0:
                scored_memories.append((memory, score))
        
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        
        return [m for m, s in scored_memories[:limit]]
    
    @staticmethod
    def _semantic_search(query: str, memories: List[MemoryEntry], llm: BaseChatModel, limit: int) -> List[MemoryEntry]:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个记忆检索专家。请根据用户的查询，评估每条记忆内容与查询的相关性。
            返回JSON格式，包含memory_id和relevance_score（0-100的整数）。
            relevance_score越高表示相关性越强。"""),
            ("human", """用户查询：{query}
            
            记忆列表（请评估每条记忆的相关性）：
            {memory_list}
            
            返回格式示例：
            [
                {"memory_id": "mem_xxx", "relevance_score": 85},
                {"memory_id": "mem_yyy", "relevance_score": 60}
            ]""")
        ])
        
        memory_list_str = "\n".join([
            f"- ID: {m.memory_id}, 标题: {m.metadata.get('title', '')}, 内容摘要: {m.content[:100]}..."
            for m in memories
        ])
        
        chain = prompt | llm
        
        try:
            result = chain.invoke({
                "query": query,
                "memory_list": memory_list_str
            })
            
            try:
                scores = json.loads(result)
                
                scored_memories = []
                for score_item in scores:
                    memory_id = score_item.get("memory_id", "")
                    relevance = score_item.get("relevance_score", 0)
                    
                    for memory in memories:
                        if memory.memory_id == memory_id:
                            scored_memories.append((memory, relevance))
                            break
                
                scored_memories.sort(key=lambda x: x[1], reverse=True)
                
                return [m for m, s in scored_memories[:limit] if s > 0]
            except json.JSONDecodeError:
                return MemoryManager._keyword_search(query, memories, limit)
        except Exception as e:
            return MemoryManager._keyword_search(query, memories, limit)
    
    @staticmethod
    def delete_memory(memory_id: str) -> bool:
        file_path = os.path.join(MEMORY_DIR, f"{memory_id}.md")
        
        if os.path.exists(file_path):
            os.remove(file_path)
            return True
        
        return False
    
    @staticmethod
    def save_video_generation_memory(video_plan: Dict[str, Any], scene_prompts: List[Dict[str, Any]], 
                                    video_results: List[Dict[str, Any]], subtitles: List[Dict[str, Any]],
                                    metadata: Dict[str, Any] = None) -> str:
        if metadata is None:
            metadata = {}
        
        content_lines = []
        
        content_lines.append("## 视频计划")
        content_lines.append(f"- 标题: {video_plan.get('title', '')}")
        content_lines.append(f"- 主题: {video_plan.get('theme', '')}")
        content_lines.append(f"- 风格: {video_plan.get('tone', '')}")
        content_lines.append(f"- 目标受众: {video_plan.get('target_audience', '')}")
        content_lines.append(f"- 时长: {video_plan.get('duration', '')}秒")
        content_lines.append(f"- 场景数量: {video_plan.get('scene_count', '')}")
        content_lines.append(f"- 色彩风格: {video_plan.get('color_palette', '')}")
        content_lines.append(f"- 光线风格: {video_plan.get('lighting', '')}")
        content_lines.append(f"- 关键元素: {', '.join(video_plan.get('key_elements', []))}")
        content_lines.append("")
        
        content_lines.append("## 场景提示词")
        for scene in scene_prompts:
            content_lines.append(f"### 场景 {scene.get('scene_number', '')}")
            content_lines.append(f"- 提示词: {scene.get('prompt', '')[:200]}...")
            content_lines.append(f"- 时长: {scene.get('duration', '')}秒")
            content_lines.append(f"- 画面比例: {scene.get('aspect_ratio', '')}")
            content_lines.append(f"- 拍摄角度: {scene.get('camera_angle', '')}")
            content_lines.append(f"- 画面风格: {scene.get('style', '')}")
            content_lines.append("")
        
        content_lines.append("## 视频生成结果")
        for result in video_results:
            content_lines.append(f"- 场景 {result.get('scene_number', '')}: {result.get('video_url', '')}")
        content_lines.append("")
        
        content_lines.append("## 字幕")
        for subtitle in subtitles:
            content_lines.append(f"- [{subtitle.get('start_time', '')}s-{subtitle.get('end_time', '')}s] {subtitle.get('text', '')}")
        
        content = "\n".join(content_lines)
        
        default_metadata = {
            "type": "video_generation",
            "title": video_plan.get("title", "未命名视频"),
            "tags": ["video", "generation", video_plan.get("theme", "")]
        }
        default_metadata.update(metadata)
        
        return MemoryManager.save_memory(content, default_metadata)
    
    @staticmethod
    def get_relevant_memories_for_prompt(user_prompt: str, llm: BaseChatModel = None, limit: int = 3) -> str:
        memories = MemoryManager.search_memories(user_prompt, llm, limit)
        
        if not memories:
            return "无相关记忆"
        
        result_lines = []
        result_lines.append("## 相关记忆")
        result_lines.append("")
        
        for i, memory in enumerate(memories, 1):
            result_lines.append(f"### 记忆 {i}: {memory.metadata.get('title', '')}")
            result_lines.append(f"**类型**: {memory.metadata.get('type', '')}")
            result_lines.append(f"**创建时间**: {memory.created_at}")
            if "tags" in memory.metadata:
                result_lines.append(f"**标签**: {', '.join(memory.metadata['tags'])}")
            result_lines.append("")
            result_lines.append(f"**内容摘要**:")
            result_lines.append(memory.content[:300] + "..." if len(memory.content) > 300 else memory.content)
            result_lines.append("")
        
        return "\n".join(result_lines)

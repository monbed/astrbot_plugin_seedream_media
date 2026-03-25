import time
import re
import os
import json
import uuid
import asyncio
import aiohttp
import shutil
from typing import Optional, List

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

# 核心导入
from astrbot.api import logger, AstrBotConfig
from astrbot.api.star import Star, Context, StarTools
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.message_components import Plain, Image, Video, Reply, At

# 插件常量定义
PLUGIN_NAME = "astrbot_plugin_seedream_media"

class SeedreamPluginError(Exception):
    """插件专用异常，用于区分插件内部错误与系统/框架异常"""
    pass

class CacheCleaner:
    def __init__(self, cache_dir, cron_expr: str):
        self.cache_dir = cache_dir
        self.cron_expr = cron_expr
        self.scheduler = AsyncIOScheduler(timezone='Asia/Shanghai')
        self._started = False

    async def start(self):
        """异步启动：先注册定时任务，成功后再启动 scheduler，避免空转"""
        if self._started:
            return
        try:
            self.trigger = CronTrigger.from_crontab(self.cron_expr)
            self.scheduler.add_job(
                func=self._clean_plugin_cache,
                trigger=self.trigger,
                name="SeedreamCacheCleaner_scheduler",
                max_instances=1,
            )
            self.scheduler.start()
            self._started = True
            logger.info(f"[{PLUGIN_NAME}] 缓存清理定时任务已启动，cron: {self.cron_expr}")
        except Exception as e:
            logger.error(f"[{PLUGIN_NAME}] 缓存清理 Cron 格式错误，scheduler 未启动：{e}")

    async def _clean_plugin_cache(self) -> None:
        """清空缓存目录内容（保留目录本身，避免与并发下载产生竞态）"""
        loop = asyncio.get_running_loop()
        try:
            if self.cache_dir.exists():
                await loop.run_in_executor(None, self._remove_dir_contents, self.cache_dir)
            logger.info(f"[{PLUGIN_NAME}] 缓存目录已定时清理。")
        except Exception:
            logger.exception(f"[{PLUGIN_NAME}] 定时清理缓存目录时出错。")

    @staticmethod
    def _remove_dir_contents(dir_path) -> None:
        """删除目录下所有内容但保留目录本身，避免 rmtree 后重建之间的竞态窗口"""
        for entry in os.scandir(dir_path):
            if entry.is_file() or entry.is_symlink():
                os.remove(entry.path)
            elif entry.is_dir():
                shutil.rmtree(entry.path)

    async def stop(self):
        self.scheduler.remove_all_jobs()
        self.scheduler.shutdown(wait=False)
        logger.info(f"[{PLUGIN_NAME}] 缓存清理器 ｜ 定时任务已安全停止")
        logger.info(f"[{PLUGIN_NAME}] 插件卸载 ｜ 生命周期资源回收完毕")

class SeedreamImagePlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        
        # 1. 解析基础字符串配置
        self.api_key = str(config.get("VOLC_API_KEY", "")).strip()
        self.api_endpoint = str(config.get("VOLC_ENDPOINT", "")).strip()
        self.image_size = str(config.get("image_size", "")).strip()
        self.model_version = str(config.get("model_version", "")).strip()
        self.video_model_version = str(config.get("video_model_version", "")).strip()
        self.video_resolution = str(config.get("video_resolution", "")).strip()
        self.video_ratio = str(config.get("video_ratio", "")).strip()
        
        # 2. 解析功能和定时参数配置
        self.video_duration = config.get("video_duration", -1)
        self.video_multi_image = config.get("video_multi_image", False)
        self.show_prompt_in_reply = config.get("show_prompt_in_reply", False)
        self.download_timeout = config.get("download_timeout", 120)
        self.clean_cron = config.get("clean_cron", "0 4 * * *")
        
        # 3. 拼接完整API地址
        self.image_api_url = f"{self.api_endpoint.rstrip('/')}/images/generations"
        self.video_tasks_url = f"{self.api_endpoint.rstrip('/')}/contents/generations/tasks"
        
        # 4. 限流与防重缓存池（注意：processing_users 和 last_operations 仅在 asyncio 单线程事件循环中安全）
        self.rate_limit_seconds = float(config.get("rate_limit_seconds", 10.0))
        self.processing_users = set()
        self.last_operations = {}
        
        # 5. 缓存目录与清理器
        self.temp_dir = StarTools.get_data_dir(PLUGIN_NAME) / "cache"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.cleaner = CacheCleaner(self.temp_dir, self.clean_cron)
        # 延迟到事件循环就绪后启动 scheduler，避免在同步构造函数中隐式创建事件循环
        asyncio.get_event_loop().create_task(self.cleaner.start())
        
        # 6. aiohttp Session 复用（优化连接开销）
        self._session: Optional[aiohttp.ClientSession] = None
        self._session_lock = asyncio.Lock()
        
        # 7. 核心配置前置校验
        if not self.api_key:
            logger.error(f"[{PLUGIN_NAME}] 异常拦截 ｜ 未配置 VOLC_API_KEY，请优先填写火山方舟 API 密钥")
        logger.info(f"[{PLUGIN_NAME}] 初始化完成 ｜ 图片模型：{self.model_version or '未配置'} ｜ 视频模型：{self.video_model_version or '未配置'} ｜ 显示提示词：{self.show_prompt_in_reply}")
        
        # 8. 路由注册与前缀拦截配置
        self.commands = {
            "豆包画图": self.generate_image,
            "豆包视频": self.generate_video
        }
        
        self.need_prefix = self.config.get("need_prefix", True)

    @property
    def api_headers(self) -> dict:
        """全局共享 API 请求头（安全提醒：切勿在日志中打印此返回值，包含明文 API 密钥）"""
        return {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

    async def _get_session(self) -> aiohttp.ClientSession:
        """复用aiohttp ClientSession，减少TCP连接开销（使用锁防止并发创建）"""
        async with self._session_lock:
            if self._session is None or self._session.closed:
                connector = aiohttp.TCPConnector(
                    ssl=True,
                    limit=10,  # 限制并发连接数
                    limit_per_host=5
                )
                self._session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=self.download_timeout),
                    connector=connector
                )
        return self._session

    async def terminate(self):
        """插件卸载时清理资源（新增：关闭复用的Session）"""
        # 停止清理任务
        if hasattr(self, 'cleaner'):
            await self.cleaner.stop()

        # 清理缓存文件（异步执行避免阻塞事件循环）
        if hasattr(self, 'temp_dir') and self.temp_dir.exists():
            try:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, shutil.rmtree, self.temp_dir)
            except Exception:
                logger.exception(f"[{PLUGIN_NAME}] 卸载时清理缓存目录失败（已忽略）")
        
        # 关闭复用的Session
        if hasattr(self, '_session') and self._session and not self._session.closed:
            await self._session.close()
        
        logger.info(f"[{PLUGIN_NAME}] 插件已卸载，资源清理完成")

    # =========================================================
    # 第一层：基础预检与解析工具 (Parsers & Validators)
    # =========================================================
    def _check_preconditions(self, user_id: str) -> Optional[str]:
        """内部工具：统一检查基础配置、请求防抖和并发任务"""
        if not self.api_key:
            return "❌ 未配置 VOLC_API_KEY，请在后台填写火山方舟 API 密钥"
            
        current_time = time.time()
        
        # 清理过期的防抖记录，防止 last_operations 无限增长导致内存泄漏
        # 使用 2 倍冷却时间作为过期阈值，确保在冷却期内的记录不会被误清理
        expired_threshold = current_time - self.rate_limit_seconds * 2
        expired_users = [uid for uid, ts in self.last_operations.items() if ts < expired_threshold]
        for uid in expired_users:
            del self.last_operations[uid]
        
        if user_id in self.last_operations:
            if current_time - self.last_operations[user_id] < self.rate_limit_seconds:
                return "⚠️ 操作过于频繁，请稍后再试"
                
        if user_id in self.processing_users:
            return "⏳ 您有正在进行的媒体任务，请耐心等待当前渲染完成"

        return None

    def _extract_prompt(self, event: AstrMessageEvent, command: str) -> str:
        """从消息中提取纯净的提示词（参考 bananic_ninjutsu 逻辑）"""
        full_text = ""
        
        # 1. 优先从消息组件链中提取纯文本，自动过滤 [图片]、@用户 等富文本组件
        if hasattr(event, 'message_obj') and event.message_obj and event.message_obj.message:
            for component in event.message_obj.message:
                if isinstance(component, Plain):
                    full_text += str(component.text) if component.text else ""
            full_text = full_text.strip()
            
        # 2. 兜底方案：如果提取不到有效文本，回退到原始 string 过滤机制（兼容底层字符串包含 CQ 码的情况）
        if not full_text:
            raw_text = event.message_str.strip()
            text_parts = []
            for token in raw_text.split():
                if not token.startswith("@") and not token.startswith("[CQ:"):
                    text_parts.append(token)
            full_text = " ".join(text_parts).strip()
            
        # 3. 剥离指令前缀（例如“豆包画图”）
        if full_text.startswith(command):
            full_text = full_text[len(command):].strip()
            
        # 为了应对一些用户习惯在指令后加标点，顺手做下清理
        return re.sub(r'^[:：,，\s]+', '', full_text).strip()

    def _extract_image_url_list(self, event: AstrMessageEvent) -> List[str]:
        """
        从消息链中提取所有图片 URL（包含引用消息图片、当前消息图片、At 组件头像）
        """
        if not hasattr(event, 'message_obj') or not event.message_obj or not event.message_obj.message:
            return []
        
        image_urls: List[str] = []
        for seg in event.message_obj.message:
            self._collect_image_urls_from_seg(seg, image_urls)
        return image_urls
    
    def _collect_image_urls_from_seg(self, seg, image_urls: List[str]) -> None:
        """递归从消息段中收集图片 URL（消除闭包 side-effect）"""
        if isinstance(seg, Image):
            img_url = self._extract_image_url(seg)
            if img_url and img_url not in image_urls:
                image_urls.append(img_url)
        elif isinstance(seg, Reply) and seg.chain:
            for sub_seg in seg.chain:
                self._collect_image_urls_from_seg(sub_seg, image_urls)
        elif isinstance(seg, At):
            if str(seg.qq).isdigit():
                avatar_url = f"https://q1.qlogo.cn/g?b=qq&nk={seg.qq}&s=640"
                if avatar_url not in image_urls:
                    image_urls.append(avatar_url)
    
    def _extract_image_url(self, component: Image) -> str:
        """从 Image 组件中提取 URL"""
        if hasattr(component, 'url') and component.url:
            return component.url.strip()
        elif hasattr(component, 'file_id') and component.file_id:
            # 注意：此 URL 模板为 QQ 平台特定格式，非 QQ 平台可能无法使用
            logger.warning(f"[{PLUGIN_NAME}] 使用 QQ 平台特定的 file_id → URL 模板，其他平台可能不兼容")
            file_id = component.file_id.replace("/", "_")
            return f"https://gchat.qpic.cn/gchatpic_new/0/0-0-{file_id}/0?tp=webp&wxfrom=5&wx_lazy=1"
        return ""

    def _find_video_url(self, data, _depth: int = 0) -> str:
        """递归查找返回数据中的视频 URL（最大递归深度 10 层，防止异常嵌套导致栈溢出）"""
        if _depth > 10:
            return ""
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, str) and v.startswith(("http://", "https://")) and (v.split("?")[0].endswith(".mp4") or "/video/" in v):
                    return v
                elif isinstance(v, (dict, list)):
                    res = self._find_video_url(v, _depth + 1)
                    if res:
                        return res
        elif isinstance(data, list):
            for item in data:
                res = self._find_video_url(item, _depth + 1)
                if res:
                    return res
        return ""

    # =========================================================
    # 第二层：网络调度与缓存工具 (Network IO & Cache)
    # =========================================================
    def _generate_cache_path(self, ext: str = ".jpg") -> str:
        """统一生成随机的本地缓存绝对路径"""
        filename = f"media_{int(time.time())}_{uuid.uuid4().hex[:8]}{ext}"
        return str(self.temp_dir / filename)

    async def _download_media(self, url: str, media_type: str = "图片", timeout: Optional[int] = None) -> str:
        """内部工具：统一的媒体文件（图片/视频）高速下载核心"""
        logger.info(f"[{PLUGIN_NAME}] 开始下载{media_type} ｜ URL: {url}")
        if not url or not url.startswith(("http://", "https://")):
            raise SeedreamPluginError(f"无效的{media_type}URL")
        
        # 校验 timeout 参数合法性
        if timeout is not None and timeout <= 0:
            logger.warning(f"[{PLUGIN_NAME}] 无效的 timeout 值 ({timeout})，将使用 session 级默认超时")
            timeout = None
            
        ext = ".mp4" if media_type == "视频" else ".jpg"
        
        try:
            session = await self._get_session()
            req_timeout = aiohttp.ClientTimeout(total=timeout) if timeout else None
            async with session.get(url, allow_redirects=True, timeout=req_timeout) as resp:
                if resp.status != 200:
                    raise SeedreamPluginError(f"网络异常 [HTTP {resp.status}]")
                content = await resp.read()
                
            local_path = self._generate_cache_path(ext)
            # 使用 run_in_executor 进行异步文件写入，避免大文件阻塞事件循环
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._write_file_sync, local_path, content)
                
            return local_path
            
        except Exception as e:
            raise SeedreamPluginError(f"{media_type}下载失败：{str(e)}") from e
    
    @staticmethod
    def _write_file_sync(path: str, data: bytes) -> None:
        """同步写文件，供 run_in_executor 调用（写入前确保目录存在，防止缓存清理竞态）"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)

    # =========================================================
    # 第三层：大模型原始 API 调用 (Core API Handlers)
    # =========================================================
    async def _call_seedream_api(self, prompt: str, image_urls: List[str] = None) -> str:
        """核心接口：调用火山方舟 Seedream API 生成图片"""
        # 构建基础请求体
        payload = {
            "model": self.model_version,
            "prompt": prompt.strip() or "高质量高清图片",
            "size": self.image_size,
            "watermark": False
        }
        
        # 图生图参数
        if image_urls:
            payload["image"] = image_urls
        
        try:
            # 复用全局Session
            session = await self._get_session()
            async with session.post(
                self.image_api_url,
                headers=self.api_headers,
                json=payload
            ) as resp:
                response_text = await resp.text()
                
                # 解析响应（精简异常处理）
                try:
                    response_data = json.loads(response_text)
                except json.JSONDecodeError as e:
                    raise SeedreamPluginError(f"API返回非JSON格式响应：{response_text[:200]}") from e
                
                # 处理错误响应
                if resp.status != 200:
                    error_msg = response_data.get("error", {}).get("message", f"请求失败 [HTTP {resp.status}]")
                    error_code = response_data.get("error", {}).get("code", "")
                    
                    # 精简错误码映射
                    error_mapping = {
                        "InvalidParameter": "参数错误",
                        "Unauthorized": "API KEY无效或未授权",
                        "Forbidden": "API KEY无使用权限",
                        "TooManyRequests": "调用频率超限"
                    }
                    
                    if error_code in error_mapping:
                        error_msg = f"{error_mapping[error_code]}：{error_msg}"
                    raise SeedreamPluginError(error_msg)
                
                # 提取图片URL
                if not response_data.get("data") or len(response_data["data"]) == 0:
                    raise SeedreamPluginError("API返回无图片数据")
                
                generated_url = response_data["data"][0].get("url")
                if not generated_url:
                    raise SeedreamPluginError("API返回无图片URL")
                
                return generated_url
        
        except aiohttp.ClientError as e:
            raise SeedreamPluginError(f"网络请求失败：{str(e)}") from e
        except SeedreamPluginError:
            raise
        except Exception as e:
            raise SeedreamPluginError(f"API调用失败：{str(e)}") from e

    async def _call_seedance_api(self, prompt: str, image_urls: Optional[List[str]] = None) -> str:
        """核心接口：调用火山方舟 Seedance API 提交视频任务并等待回传"""
        # 构建 content 列表
        content_list: List[dict] = []
        if prompt:
            content_list.append({"type": "text", "text": prompt})
            
        if image_urls:
            if getattr(self, "video_multi_image", False):
                for img_url in image_urls:
                    content_list.append({"type": "image_url", "image_url": {"url": img_url}})
            else:
                content_list.append({"type": "image_url", "image_url": {"url": image_urls[0]}})
            
        # duration 合法性校验（schema 已保证 int 类型，此处仅校验值域）
        duration_val = self.video_duration
        if duration_val <= 0:
            duration_val = -1  # 由模型自主决定时长
            
        payload = {
            "model": self.video_model_version,
            "content": content_list,
            "generate_audio": True,
            "resolution": self.video_resolution,
            "ratio": self.video_ratio,
            "duration": duration_val,
            "watermark": False
        }
        
        # 1. 提交视频生成任务
        try:
            session = await self._get_session()
            async with session.post(self.video_tasks_url, headers=self.api_headers, json=payload) as resp:
                try:
                    res_data = await resp.json()
                except Exception as e:
                    raise SeedreamPluginError(f"API返回非JSON格式：{await resp.text()}") from e
                    
                if resp.status != 200:
                    raise SeedreamPluginError(res_data.get("error", {}).get("message", f"HTTP {resp.status}"))
                    
                task_id = res_data.get("id") or res_data.get("task_id")
                if not task_id:
                    raise SeedreamPluginError(f"未获取到 Task ID: {res_data}")
        except aiohttp.ClientError as e:
            raise SeedreamPluginError(f"视频任务提交网络请求失败：{str(e)}") from e
                
        logger.info(f"[{PLUGIN_NAME}] 视频任务提交成功 ｜ Task ID: {task_id}")
        
        # 2. 轮询等待任务完成
        return await self._poll_video_task(task_id)

    async def _poll_video_task(self, task_id: str) -> str:
        """轮询视频任务状态直到完成、失败或超时"""
        max_retries = 60  # 最多轮询 60 次 (60 * 10s = 10 分钟)
        max_consecutive_failures = 5  # 连续网络失败上限，超过则提前终止
        consecutive_failures = 0
        for attempt in range(max_retries):
            await asyncio.sleep(10)
            
            poll_url = f"{self.video_tasks_url}/{task_id}"
            try:
                session = await self._get_session()
                async with session.get(poll_url, headers=self.api_headers) as poll_resp:
                    # 网络请求成功，重置连续失败计数
                    consecutive_failures = 0
                    try:
                        poll_data = await poll_resp.json()
                    except Exception:
                        logger.warning(f"[{PLUGIN_NAME}] 轮询返回非JSON响应，跳过本轮 (第{attempt+1}次)")
                        continue
                    
                    status = poll_data.get("status", "").lower()
                    
                    if status in ["succeeded", "success", "completed"]:
                        video_url = self._find_video_url(poll_data)
                        if video_url:
                            return video_url
                        raise SeedreamPluginError("任务成功，但未找到视频 URL")
                        
                    elif status in ["failed", "error", "cancelled"]:
                        error_msg = poll_data.get("error", {}).get("message", "任务失败")
                        raise SeedreamPluginError(f"生成失败: {error_msg}")
            except aiohttp.ClientError as e:
                consecutive_failures += 1
                logger.warning(f"[{PLUGIN_NAME}] 轮询网络异常 (连续第{consecutive_failures}次，第{attempt+1}轮): {e}")
                if consecutive_failures >= max_consecutive_failures:
                    raise SeedreamPluginError(f"轮询连续 {max_consecutive_failures} 次网络异常，已终止等待") from e
                continue
                    
        raise SeedreamPluginError("任务等待超时 (10分钟)，请稍后重试")

    @filter.event_message_type(filter.EventMessageType.ALL, priority=10)
    async def global_media_interceptor(self, event: AstrMessageEvent):
        """
        基于框架底层鉴权的全局拦截与指令分发
        """
        if getattr(self, "need_prefix", True) and not event.is_at_or_wake_command:
            return
            
        text = event.message_str.strip()
        
        for cmd, func in self.commands.items():
            if text.startswith(cmd):
                event.stop_event()
                async for res in func(event):
                    yield res
                break

    # =========================================================
    # 指令处理（精简输出）
    # =========================================================
    async def generate_image(self, event: AstrMessageEvent):
        """
        [指令] 火山方舟 Seedream 图片生成
        
        支持以下四种快捷用法：
        1. 文生图：豆包画图 <提示词>
        2. 图生图：豆包画图 <提示词> + 发送图片
        3. 引用生图：回复他人消息 + 豆包画图 <提示词>
        4. 头像生图：豆包画图 <提示词> + @某人
        """
        if not getattr(self, "model_version", None):
            yield event.plain_result("❌ 图片模型未配置，请在插件后台填写 model_version")
            return
            
        user_id = event.get_sender_id()
        
        # 统一前置拦截检查
        err_msg = self._check_preconditions(user_id)
        if err_msg:
            yield event.plain_result(err_msg)
            return

        real_prompt = self._extract_prompt(event, "豆包画图")
        
        # 提取图片URL列表
        image_urls = self._extract_image_url_list(event)
        
        # 无提示词且无图片
        if not real_prompt and not image_urls:
            yield event.plain_result("⚠️ 格式错误：请至少提供一段画面描述或一张参考图片")
            return
        
        # 开始生成
        self.processing_users.add(user_id)
        self.last_operations[user_id] = time.time()
        # 精简的状态提示
        yield event.plain_result("🎨 正在为您构思画面，请稍候...")
        
        try:
            # 调用API
            generated_url = await self._call_seedream_api(real_prompt, image_urls)
            
            # 下载图片（无提示）
            local_path = await self._download_media(generated_url, "图片")
            
            # 构造回复（精简结果）
            reply_components = []
            if hasattr(event.message_obj, 'message_id'):
                reply_components.append(Reply(id=event.message_obj.message_id))
            
            # 添加图片
            reply_components.append(Image.fromFileSystem(local_path))
            
            # 根据配置决定是否添加提示词文本
            if self.show_prompt_in_reply:
                reply_components.append(Plain(text=f"✅ 生成完成\n提示词：{real_prompt or '纯图生图'}"))
            else:
                reply_components.append(Plain(text="✅ 生成完成"))
            
            yield event.chain_result(reply_components)
            
        except Exception as e:
            logger.error(f"[{PLUGIN_NAME}] 生图异常 ｜ 用户：{user_id} ｜ 错误：{str(e)}")
            yield event.plain_result(f"❌ 生成失败：{str(e)}")
            
        finally:
            self.processing_users.discard(user_id)

    # =========================================================
    # 视频生成指令
    # =========================================================
    async def generate_video(self, event: AstrMessageEvent):
        """
        [指令] 火山方舟 Seedance 视频生成
        
        支持以下四种快捷用法：
        1. 文生视频：豆包视频 <提示词>
        2. 图生视频：豆包视频 <提示词> + 发送图片
        3. 引用视频：回复他人消息 + 豆包视频 <提示词>
        4. 头像视频：豆包视频 <提示词> + @某人
        """
        user_id = event.get_sender_id()
        
        if not getattr(self, "video_model_version", None):
            yield event.plain_result("❌ 视频模型未配置，请在插件后台填写 video_model_version")
            return
            
        # 统一前置拦截检查
        err_msg = self._check_preconditions(user_id)
        if err_msg:
            yield event.plain_result(err_msg)
            return
            
        real_prompt = self._extract_prompt(event, "豆包视频")
        
        # 复用图片提取方法
        image_urls = self._extract_image_url_list(event)
        
        if not real_prompt and not image_urls:
            yield event.plain_result("⚠️ 格式错误：请至少提供一段画面描述或一张参考图片")
            return
        
        self.processing_users.add(user_id)
        self.last_operations[user_id] = time.time()
        yield event.plain_result("🎬 视频任务已加入云端队列！\n⏳ 由于算力庞大，大约需要等待 3~5 分钟，完成后会自动回传...")
        
        try:
            video_url = await self._call_seedance_api(real_prompt, image_urls)
            
            # 下载到本地并发送视频结果
            local_video_path = await self._download_media(video_url, "视频", timeout=300)
            yield event.chain_result([Video.fromFileSystem(local_video_path)])
        
        except Exception as e:
            logger.error(f"[{PLUGIN_NAME}] 视频任务异常 ｜ 用户：{user_id} ｜ 错误：{str(e)}")
            yield event.plain_result(f"❌ 视频生成失败：{str(e)}")
        
        finally:
            self.processing_users.discard(user_id)
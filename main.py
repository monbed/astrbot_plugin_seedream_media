import time
import re
import json
import uuid
import asyncio
import aiohttp
import shutil
from typing import Optional, List

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

# 核心导入
from astrbot.api import logger
from astrbot.api.star import Star, Context, StarTools
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.message_components import Plain, Image, Video, Reply, At

# 插件常量定义
PLUGIN_NAME = "astrbot_plugin_seedream_media"

class CacheCleaner:
    def __init__(self, cache_dir, cron_expr: str):
        self.cache_dir = cache_dir
        self.cron_expr = cron_expr
        self.scheduler = AsyncIOScheduler(timezone='Asia/Shanghai')
        self.scheduler.start()
        self.register_task()

    def register_task(self):
        try:
            self.trigger = CronTrigger.from_crontab(self.cron_expr)
            self.scheduler.add_job(
                func=self._clean_plugin_cache,
                trigger=self.trigger,
                name="SeedreamCacheCleaner_scheduler",
                max_instances=1,
            )
            logger.info(f"[{PLUGIN_NAME}] 缓存清理定时任务已启动，cron: {self.cron_expr}")
        except Exception as e:
            logger.error(f"[{PLUGIN_NAME}] 缓存清理 Cron 格式错误：{e}")

    async def _clean_plugin_cache(self) -> None:
        """删除并重建缓存目录"""
        loop = asyncio.get_running_loop()
        try:
            if self.cache_dir.exists():
                await loop.run_in_executor(None, shutil.rmtree, self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"[{PLUGIN_NAME}] 缓存目录已定时清理并重建。")
        except Exception:
            logger.exception(f"[{PLUGIN_NAME}] 定时清理缓存目录时出错。")

    async def stop(self):
        self.scheduler.remove_all_jobs()
        self.scheduler.shutdown(wait=False)
        logger.info(f"[{PLUGIN_NAME}] 缓存清理器 ｜ 定时任务已安全停止")
        logger.info(f"[{PLUGIN_NAME}] 插件卸载 ｜ 生命周期资源回收完毕")

class SeedreamImagePlugin(Star):
    def __init__(self, context: Context, config: dict):
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
        self.video_duration = (config.get("video_duration", "")).strip()
        self.show_prompt_in_reply = (config.get("show_prompt_in_reply", "")).strip()
        self.download_timeout = (config.get("download_timeout", "")).strip()
        self.clean_cron = (config.get("clean_cron", "")).strip()
        
        # 3. 拼接完整API地址
        self.image_api_url = f"{self.api_endpoint.rstrip('/')}/images/generations"
        self.video_tasks_url = f"{self.api_endpoint.rstrip('/')}/contents/generations/tasks"
        
        # 4. 限流与防重缓存池
        self.rate_limit_seconds = 10.0
        self.processing_users = set()
        self.last_operations = {}
        
        # 5. 缓存目录与清理器
        self.temp_dir = StarTools.get_data_dir(PLUGIN_NAME) / "cache"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.cleaner = CacheCleaner(self.temp_dir, self.clean_cron)
        
        # 6. aiohttp Session 复用（优化连接开销）
        self._session: Optional[aiohttp.ClientSession] = None
        
        # 7. 核心配置前置校验
        if not self.api_key:
            logger.error(f"[{PLUGIN_NAME}] 异常拦截 ｜ 未配置 VOLC_API_KEY，请优先填写火山方舟 API 密钥")
        logger.info(f"[{PLUGIN_NAME}] 初始化完成 ｜ 图片模型：{self.model_version} ｜ 视频模型：{self.video_model_version or '未配置'} ｜ 尺寸：{self.image_size} ｜ 显示提示词：{self.show_prompt_in_reply}")

    @property
    def api_headers(self) -> dict:
        """全局共享 API 请求头"""
        return {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

    @property
    def session(self) -> aiohttp.ClientSession:
        """复用aiohttp ClientSession，减少TCP连接开销"""
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

        # 清理缓存文件
        if hasattr(self, 'temp_dir') and self.temp_dir.exists():
            try:
                shutil.rmtree(self.temp_dir)
            except Exception:
                pass
        
        # 关闭复用的Session
        if self._session and not self._session.closed:
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
        if user_id in self.last_operations:
            if current_time - self.last_operations[user_id] < self.rate_limit_seconds:
                return "⚠️ 操作过于频繁，请稍后再试"
                
        if user_id in self.processing_users:
            return "⏳ 您有正在进行的媒体任务，请耐心等待当前渲染完成"
            
        self.last_operations[user_id] = current_time
        return None

    def _extract_prompt(self, event: AstrMessageEvent, command: str) -> str:
        """从消息中提取纯净的提示词"""
        full_text = ""
        if hasattr(event, 'message_obj') and event.message_obj and event.message_obj.message:
            for component in event.message_obj.message:
                if isinstance(component, Plain):
                    full_text += component.text
        return re.sub(rf'^.*?{re.escape(command)}', '', full_text, flags=re.DOTALL).strip()

    def _extract_image_url_list(self, event: AstrMessageEvent) -> List[str]:
        """
        从消息链中提取所有图片 URL（包含引用消息图片、当前消息图片、At 组件头像）
        """
        image_urls = []
        
        if not hasattr(event, 'message_obj') or not event.message_obj or not event.message_obj.message:
            return image_urls
        
        def _extract_from_seg(seg):
            if isinstance(seg, Image):
                img_url = self._extract_image_url(seg)
                if img_url and img_url not in image_urls:
                    image_urls.append(img_url)
            elif isinstance(seg, Reply) and seg.chain:
                for sub_seg in seg.chain:
                    _extract_from_seg(sub_seg)
            elif isinstance(seg, At):
                if str(seg.qq).isdigit():
                    avatar_url = f"https://q1.qlogo.cn/g?b=qq&nk={seg.qq}&s=640"
                    if avatar_url not in image_urls:
                        image_urls.append(avatar_url)
                        
        for seg in event.message_obj.message:
            _extract_from_seg(seg)
            
        return image_urls
    
    def _extract_image_url(self, component: Image) -> str:
        """从 Image 组件中提取 URL"""
        if hasattr(component, 'url') and component.url:
            return component.url.strip()
        elif hasattr(component, 'file_id') and component.file_id:
            file_id = component.file_id.replace("/", "_")
            return f"https://gchat.qpic.cn/gchatpic_new/0/0-0-{file_id}/0?tp=webp&wxfrom=5&wx_lazy=1"
        return ""

    def _find_video_url(self, data) -> str:
        """递归查找返回数据中的视频 URL"""
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, str) and v.startswith("http") and (".mp4" in v or "video" in v):
                    return v
                elif isinstance(v, (dict, list)):
                    res = self._find_video_url(v)
                    if res:
                        return res
        elif isinstance(data, list):
            for item in data:
                res = self._find_video_url(item)
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

    async def _download_media(self, url: str, media_type: str = "图片") -> str:
        """内部工具：统一的媒体文件（图片/视频）高速下载核心"""
        logger.info(f"[{PLUGIN_NAME}] 开始下载{media_type} ｜ URL: {url}")
        if not url or not url.startswith("http"):
            raise Exception(f"无效的{media_type}URL")
            
        ext = ".mp4" if media_type == "视频" else ".jpg"
        
        try:
            async with self.session.get(url, allow_redirects=True) as resp:
                if resp.status != 200:
                    raise Exception(f"网络异常 [HTTP {resp.status}]")
                content = await resp.read()
                
            local_path = self._generate_cache_path(ext)
            with open(local_path, "wb") as f:
                f.write(content)
                
            return local_path
            
        except Exception as e:
            raise Exception(f"{media_type}下载失败：{str(e)}")

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
        if image_urls and len(image_urls) > 0:
            payload["image"] = image_urls
        
        try:
            # 复用全局Session
            async with self.session.post(
                self.image_api_url,
                headers=self.api_headers,
                json=payload
            ) as resp:
                response_text = await resp.text()
                
                # 解析响应（精简异常处理）
                try:
                    response_data = json.loads(response_text)
                except json.JSONDecodeError:
                    raise Exception(f"API返回非JSON格式响应：{response_text[:200]}")
                
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
                    raise Exception(error_msg)
                
                # 提取图片URL
                if not response_data.get("data") or len(response_data["data"]) == 0:
                    raise Exception("API返回无图片数据")
                
                generated_url = response_data["data"][0].get("url")
                if not generated_url:
                    raise Exception("API返回无图片URL")
                
                return generated_url
        
        except aiohttp.ClientError as e:
            raise Exception(f"网络请求失败：{str(e)}")
        except Exception as e:
            raise Exception(f"API调用失败：{str(e)}")

    async def _call_seedance_api(self, prompt: str, image_url: str = "") -> str:
        """核心接口：调用火山方舟 Seedance API 提交视频任务并等待回传"""
        # 构建 content 列表
        content_list = []
        if prompt:
            content_list.append({"type": "text", "text": prompt})
        if image_url:
            content_list.append({"type": "image_url", "image_url": {"url": image_url}})
            
        payload = {
            "model": self.video_model_version,
            "content": content_list,
            "generate_audio": True,
            "resolution": self.video_resolution,
            "ratio": self.video_ratio,
            "duration": self.video_duration,
            "watermark": False
        }
        
        # 1. 提交视频生成任务
        async with self.session.post(self.video_tasks_url, headers=self.api_headers, json=payload) as resp:
            try:
                res_data = await resp.json()
            except Exception:
                raise Exception(f"API返回非JSON格式：{await resp.text()}")
                
            if resp.status != 200:
                raise Exception(res_data.get("error", {}).get("message", f"HTTP {resp.status}"))
                
            task_id = res_data.get("id") or res_data.get("task_id")
            if not task_id:
                raise Exception(f"未获取到 Task ID: {res_data}")
                
        logger.info(f"[{PLUGIN_NAME}] 视频任务提交成功 ｜ Task ID: {task_id}")
        
        # 2. 轮询检查任务状态
        max_retries = 60  # 最多轮询 60 次 (60 * 10s = 10 分钟)
        for _ in range(max_retries):
            await asyncio.sleep(10)
            
            poll_url = f"{self.video_tasks_url}/{task_id}"
            async with self.session.get(poll_url, headers=self.api_headers) as poll_resp:
                poll_data = await poll_resp.json()
                status = poll_data.get("status", "").lower()
                
                if status in ["succeeded", "success", "completed"]:
                    video_url = self._find_video_url(poll_data)
                    if video_url:
                        return video_url
                    raise Exception("任务成功，但未找到视频 URL")
                    
                elif status in ["failed", "error", "cancelled"]:
                    error_msg = poll_data.get("error", {}).get("message", "任务失败")
                    raise Exception(f"生成失败: {error_msg}")
                    
        raise Exception("任务等待超时 (10分钟)，请稍后重试")

    # =========================================================
    # 指令处理（精简输出）
    # =========================================================
    @filter.command("豆包画图")
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
    @filter.command("豆包视频")
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
        yield event.plain_result("🎬 视频任务已加入云端队列！\n⏳ 由于算力庞大，大约需要等待 3~5 分钟，完成后会自动回传...")
        
        try:
            target_image = image_urls[0] if image_urls else ""
            video_url = await self._call_seedance_api(real_prompt, target_image)
            
            # 发送视频结果
            final_res = event.make_result()
            
            # 下载到本地并发送
            local_video_path = await self._download_media(video_url, "视频")
            final_res.chain.append(Video.fromFileSystem(local_video_path))
            
            await event.send(final_res)
        
        except Exception as e:
            logger.error(f"[{PLUGIN_NAME}] 视频任务异常 ｜ 用户：{user_id} ｜ 错误：{str(e)}")
            yield event.plain_result(f"❌ 视频生成失败：{str(e)}")
        
        finally:
            self.processing_users.discard(user_id)
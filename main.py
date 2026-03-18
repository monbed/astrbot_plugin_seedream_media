import os
import time
import re
import json
import uuid
import asyncio
import aiohttp
import shutil
from typing import Optional, List, Tuple

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
        logger.info(f"[{PLUGIN_NAME}] 缓存清理定时任务已停止")

class SeedreamImagePlugin(Star):
    def __init__(self, context: Context, config: dict):
        super().__init__(context)
        self.config = config
        
        # 1. 解析配置文件
        self.api_key = config.get("VOLC_API_KEY", "").strip()
        self.api_endpoint = config.get("VOLC_ENDPOINT", "https://ark.cn-beijing.volces.com/api/v3").strip()
        self.image_size = config.get("image_size", "2K").strip()
        self.model_version = config.get("model_version", "seedream-v1").strip()
        # 视频模型配置
        self.video_model_version = config.get("video_model_version", "").strip()
        self.video_resolution = config.get("video_resolution", "720p").strip()
        self.video_ratio = config.get("video_ratio", "adaptive").strip()
        
        # 视频时长处理（如果填错则回退到 5 秒）
        try:
            v_duration = int(config.get("video_duration", 5))
            if v_duration != -1 and not (2 <= v_duration <= 12):
                v_duration = 5
        except (ValueError, TypeError):
            v_duration = 5
        self.video_duration = v_duration

        # 是否显示提示词配置
        self.show_prompt_in_reply = config.get("show_prompt_in_reply", True)
        # 2. 校验并处理图片分辨率
        self.valid_size, self.size_error = self._validate_image_size(self.image_size)
        if self.size_error:
            logger.warning(f"[{PLUGIN_NAME}] 分辨率配置异常：{self.size_error}，已自动调整为 2K")
            self.valid_size = "2K"
        
        # 3. 拼接完整API地址
        self.full_api_url = f"{self.api_endpoint.rstrip('/')}/images/generations"
        self.video_tasks_url = f"{self.api_endpoint.rstrip('/')}/contents/generations/tasks"
        
        # 4. 限流/防重配置
        self.rate_limit_seconds = 10.0
        self.processing_users = set()
        self.last_operations = {}
        
        # 5. 缓存、超时与清理器
        self.download_timeout = int(config.get("download_timeout", 120))
        self.temp_dir = StarTools.get_data_dir(PLUGIN_NAME) / "cache"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.clean_cron = config.get("clean_cron", "0 4 * * *")
        self.cleaner = CacheCleaner(self.temp_dir, self.clean_cron)
        
        # 6. aiohttp Session 复用（优化连接开销）
        self._session: Optional[aiohttp.ClientSession] = None
        
        # 7. 核心配置校验
        if not self.api_key:
            logger.error(f"[{PLUGIN_NAME}] VOLC_API_KEY未配置！请填写火山方舟账号的API KEY")
        logger.info(f"[{PLUGIN_NAME}] 初始化完成 | 图片模型：{self.model_version} | 视频模型：{self.video_model_version or '未配置'} | 生成尺寸：{self.valid_size} | 显示提示词：{self.show_prompt_in_reply}")

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
        save_dir = StarTools.get_data_dir(PLUGIN_NAME) / "cache"
        if save_dir.exists():
            try:
                shutil.rmtree(save_dir)
            except Exception:
                pass
        
        # 关闭复用的Session
        if self._session and not self._session.closed:
            await self._session.close()
        
        logger.info(f"[{PLUGIN_NAME}] 插件已卸载，资源清理完成")

    # =========================================================
    # 尺寸校验工具
    # =========================================================
    def _validate_image_size(self, size_str: str) -> Tuple[str, str]:
        """
        校验图片分辨率（方式1：指定分辨率代码）
        支持1K、2K、3K、4K等分辨率代码
        """
        # 支持的分辨率列表
        valid_resolutions = ["1K", "2K", "3K", "4K"]
        
        # 将输入转换为大写以支持小写输入
        size_upper = size_str.upper().strip()
        
        # 检查是否为有效的分辨率
        if size_upper in valid_resolutions:
            return size_upper, ""
        
        # 如果不是有效的分辨率格式
        return "2K", f"分辨率格式错误（{size_str}），支持的分辨率：{', '.join(valid_resolutions)}"

    # =========================================================
    # 通用工具方法
    # =========================================================
    async def _download_generated_image(self, url: str) -> str:
        """下载API生成的图片（复用Session）"""
        logger.info(f"[{PLUGIN_NAME}] 正在下载图片: {url}")
        if not url or not url.startswith("http"):
            raise Exception("无效的图片URL")
        
        try:
            async with self.session.get(
                url,
                allow_redirects=True
            ) as resp:
                if resp.status != 200:
                    raise Exception(f"下载失败 [HTTP {resp.status}]")
                image_data = await resp.read()
        
            # 保存图片
            save_dir = StarTools.get_data_dir(PLUGIN_NAME) / "cache"
            save_dir.mkdir(parents=True, exist_ok=True)
            
            file_name = f"seedream_{int(time.time())}_{uuid.uuid4().hex[:8]}.jpg"
            save_path = save_dir / file_name
            
            with open(save_path, "wb") as f:
                f.write(image_data)
                
            return str(save_path)
            
        except Exception as e:
            raise Exception(f"图片下载失败: {str(e)}")

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

    def _extract_prompt(self, event: AstrMessageEvent, command: str, fallback: str = "") -> str:
        """从消息中提取提示词并移除指令关键词"""
        full_text = ""
        if hasattr(event, 'message_obj') and event.message_obj and event.message_obj.message:
            for component in event.message_obj.message:
                if isinstance(component, Plain):
                    full_text += component.text
        if not full_text:
            full_text = fallback
        return re.sub(rf'^.*?{re.escape(command)}', '', full_text).strip()

    async def _download_video(self, url: str) -> str:
        """下载视频到本地临时文件"""
        logger.info(f"[{PLUGIN_NAME}] 正在下载视频: {url}")
        temp_dir = StarTools.get_data_dir(PLUGIN_NAME) / "cache"
        temp_dir.mkdir(parents=True, exist_ok=True)
        filename = f"video_{str(uuid.uuid4().hex)[:8]}.mp4"
        local_path = str(temp_dir / filename)
        
        async with self.session.get(url) as resp:
            if resp.status != 200:
                raise Exception(f"视频下载失败 HTTP {resp.status}")
            content = await resp.read()
            with open(local_path, "wb") as f:
                f.write(content)
        return local_path

    # =========================================================
    # 核心API调用逻辑
    # =========================================================
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


    async def _call_seedream_api(self, prompt: str, image_urls: List[str] = None):
        """调用火山方舟Seedream API（优化：复用Session，精简异常处理）"""
        if not self.api_key:
            raise Exception("VOLC_API_KEY未配置")
        
        # 构建基础请求体
        payload = {
            "model": self.model_version,
            "prompt": prompt.strip() or "高质量高清图片",
            "size": self.valid_size,
            "watermark": False
        }
        
        # 图生图参数
        if image_urls and len(image_urls) > 0:
            payload["image"] = image_urls
        
        # 请求头
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        try:
            # 复用全局Session
            async with self.session.post(
                self.full_api_url,
                headers=headers,
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

    # =========================================================
    # 指令处理（精简输出）
    # =========================================================
    @filter.command("豆包画图")
    async def generate_image(self, event: AstrMessageEvent):
        """
        火山方舟Seedream图片生成（支持文生图、图生图、引用生图）
        
        使用方法：
        1. 文生图：豆包画图 <提示词>
        2. 图生图：豆包画图 <提示词> + 发送图片
        3. 引用生图：回复他人消息 + 豆包画图 <提示词>（优先使用引用中的图片）
        4. 头像参考：豆包画图 <提示词> + @某人（当无图片时使用 @用户 的头像作参考）
        """
        real_prompt = self._extract_prompt(event, "豆包画图", "")
        
        # 提取图片URL列表
        image_urls = self._extract_image_url_list(event)
        
        # 基础校验
        user_id = event.get_sender_id()
        
        # 防抖检查
        current_time = time.time()
        if user_id in self.last_operations:
            if current_time - self.last_operations[user_id] < self.rate_limit_seconds:
                yield event.plain_result("操作过快，请稍后再试")
                return
        self.last_operations[user_id] = current_time
        
        # 防重复处理
        if user_id in self.processing_users:
            yield event.plain_result("有正在进行的生图任务，请稍候")
            return
        
        # 无提示词且无图片
        if not real_prompt and not image_urls:
            yield event.plain_result("请提供提示词或图片")
            return
        
        # 开始生成
        self.processing_users.add(user_id)
        try:
            # 精简的状态提示
            yield event.plain_result("开始生成图片...")
            
            # 调用API
            generated_url = await self._call_seedream_api(real_prompt, image_urls)
            
            # 下载图片（无提示）
            local_path = await self._download_generated_image(generated_url)
            
            # 构造回复（精简结果）
            reply_components = []
            if hasattr(event.message_obj, 'message_id'):
                reply_components.append(Reply(id=event.message_obj.message_id))
            
            # 添加图片
            reply_components.append(Image.fromFileSystem(local_path))
            
            # 根据配置决定是否添加提示词文本
            if self.show_prompt_in_reply:
                reply_components.append(Plain(text=f"生成完成\n提示词：{real_prompt or '纯图生图'}"))
            else:
                reply_components.append(Plain(text="生成完成"))
            
            yield event.chain_result(reply_components)
            
        except Exception as e:
            logger.error(f"[{PLUGIN_NAME}] 生图失败（用户{user_id}）: {str(e)}")
            yield event.plain_result(f"生成失败：{str(e)}")
            
        finally:
            if user_id in self.processing_users:
                self.processing_users.remove(user_id)

    # =========================================================
    # 视频生成指令
    # =========================================================
    @filter.command("豆包视频")
    async def generate_video(self, event: AstrMessageEvent):
        """
        火山方舟Seedance视频生成（支持文生视频、图生视频、引用生视频）
        
        使用方法：
        1. 文生视频：豆包视频 <提示词>
        2. 图生视频：豆包视频 <提示词> + 发送图片
        3. 引用生视频：回复他人消息 + 豆包视频 <提示词>
        """
        user_id = event.get_sender_id()
        
        # 校验视频模型是否配置
        if not self.video_model_version:
            yield event.plain_result("视频模型未配置，请在插件配置中填写 video_model_version")
            return
        
        if not self.api_key:
            yield event.plain_result("VOLC_API_KEY 未配置")
            return
        
        real_prompt = self._extract_prompt(event, "豆包视频", "")
        
        # 复用图片提取方法
        image_urls = self._extract_image_url_list(event)
        
        if not real_prompt and not image_urls:
            yield event.plain_result("请提供提示词或图片")
            return
        
        # 防抖检查
        current_time = time.time()
        if user_id in self.last_operations:
            if current_time - self.last_operations[user_id] < self.rate_limit_seconds:
                yield event.plain_result("操作过快，请稍后再试")
                return
        self.last_operations[user_id] = current_time
        
        # 防重复处理
        if user_id in self.processing_users:
            yield event.plain_result("有正在进行的任务，请稍候")
            return
        
        self.processing_users.add(user_id)
        yield event.plain_result("🎬 视频任务已提交！\n⏳ 预计需要 3~5 分钟，完成后会自动发送，请耐心等待。")
        
        try:
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            # 构建 content 列表
            content_list = []
            if real_prompt:
                content_list.append({"type": "text", "text": real_prompt})
            if image_urls:
                for img_url in image_urls:
                    content_list.append({"type": "image_url", "image_url": {"url": img_url}})
            
            payload = {
                "model": self.video_model_version,
                "content": content_list,
                "generate_audio": True,
                "resolution": self.video_resolution,
                "ratio": self.video_ratio,
                "duration": self.video_duration,
                "watermark": False
            }
            
            # 提交视频生成任务
            async with self.session.post(self.video_tasks_url, headers=headers, json=payload) as resp:
                res_data = await resp.json()
                if resp.status != 200:
                    raise Exception(res_data.get("error", {}).get("message", f"HTTP {resp.status}"))
                
                task_id = res_data.get("id") or res_data.get("task_id")
                if not task_id:
                    raise Exception(f"未获取到 Task ID: {res_data}")
            
            logger.info(f"[{PLUGIN_NAME}] 视频任务已提交，Task ID: {task_id}")
            
            # 轮询检查任务状态
            video_url = ""
            max_retries = 60  # 最多轮询 60 次 (60 * 10s = 10 分钟)
            
            for _ in range(max_retries):
                await asyncio.sleep(10)
                
                poll_url = f"{self.video_tasks_url}/{task_id}"
                async with self.session.get(poll_url, headers=headers) as poll_resp:
                    poll_data = await poll_resp.json()
                    status = poll_data.get("status", "").lower()
                    
                    if status in ["succeeded", "success", "completed"]:
                        video_url = self._find_video_url(poll_data)
                        break
                    elif status in ["failed", "error", "cancelled"]:
                        error_msg = poll_data.get("error", {}).get("message", "任务失败")
                        raise Exception(f"生成失败: {error_msg}")
            
            if not video_url:
                raise Exception("任务超时或未找到视频 URL")
            
            # 发送视频结果
            final_res = event.make_result()
            
            # 下载到本地并发送
            local_video_path = await self._download_video(video_url)
            final_res.chain.append(Video.fromFileSystem(local_video_path))
            
            await event.send(final_res)
        
        except Exception as e:
            logger.error(f"[{PLUGIN_NAME}] 视频生成失败（用户{user_id}）: {str(e)}")
            error_res = event.make_result().message(f"视频生成失败：{str(e)}")
            await event.send(error_res)
        
        finally:
            self.processing_users.discard(user_id)
            # 清理本地刚下好的临时视频
            if 'local_video_path' in locals() and os.path.exists(local_video_path):
                try:
                    os.remove(local_video_path)
                except Exception as e:
                    logger.error(f"[{PLUGIN_NAME}] 临时视频清理失败 {local_video_path}: {e}")
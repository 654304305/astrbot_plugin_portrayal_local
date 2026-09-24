import time

from astrbot.api import logger
from astrbot.api.event import filter
from astrbot.api.star import Context, Star
from astrbot.core.config.astrbot_config import AstrBotConfig
from astrbot.core.message.components import At, Plain
from astrbot.core.platform.astr_message_event import AstrMessageEvent
from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import (
    AiocqhttpMessageEvent,
)
from astrbot.core.provider.entities import ProviderRequest, LLMResponse

from .core.config import PluginConfig
from .core.db import UserProfileDB
from .core.entry import EntryService
from .core.llm import LLMService
from .core.message import MessageManager
from .core.model import UserProfile

from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.agent.tool import FunctionTool, ToolExecResult
from astrbot.core.astr_agent_context import AstrAgentContext
from pydantic import Field
from pydantic.dataclasses import dataclass as pyd_dataclass


# 模块级单例引用，插件实例化时赋值
_plugin_instance: "PortrayalPlugin | None" = None


@pyd_dataclass
class PortraitTool(FunctionTool[AstrAgentContext]):
    name: str = "generate_portrait"
    description: str = (
        "当用户出现以下表达时，**必须调用此工具**，不要凭记忆或上下文自己回答：\n"
        "\n"
        "【根据用户请求选择 style 参数】：\n"
        "- 用户说'评价一下'、'分析一下'、'说说'、'看看怎么样'、'画一下'、"
        "'是个什么样的人' → style='画像'（综合画像，默认）\n"
        "- 用户说'说说优点'、'优势'、'好的地方'、'优点是什么' → style='正画像'（优势导向）\n"
        "- 用户说'说说缺点'、'劣势'、'不好的地方'、'有什么问题' → style='负画像'（缺陷导向）\n"
        "- 用户说'适合什么对象'、'该找什么样的人'、'相亲'、'脱单'、'找对象'、"
        "'情感建议' → style='找对象'（红娘匹配分析）\n"
        "\n"
        "如果用户 @ 了某人，target_id 填被 @ 者的 QQ 号。\n"
        "\n"
        "【target_id 目标判定规则（必须填写，不要留空）】\n"
        "- '评价一下我'、'我怎么样' → 填当前发言者的 QQ 号\n"
        "- '评价一下你自己'、'评价你' → 填机器人的 QQ 号\n"
        "- '评价一下刚刚说话的人'、'她/他怎么样' → 填上文最近一条非机器人消息发言者的 QQ 号；"
        "无法从上下文确定 QQ 号时，改填 target='previous_speaker'\n"
        "- '评价一下某某（昵称）' → 若能确定其 QQ 号则填写，否则改填 target='previous_speaker'"
        "或填该昵称，工具会尽力解析\n"
    )
    parameters: dict = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "target_id": {
                    "type": "string",
                    "description": (
                        "目标用户的 QQ 号，必须根据上下文判定后填写。"
                        "参考工具描述中的【target_id 目标判定规则】。"
                    ),
                },
                "target": {
                    "type": "string",
                    "description": (
                        "可选的语义化目标，当无法确定 QQ 号时使用。"
                        "可选值：'self'（当前发言者）、'bot'（机器人自己）、"
                        "'previous_speaker'（刚刚说话的人，由工具查群历史解析）。"
                        "target_id 与 target 二选一，优先填 target_id。"
                    ),
                },
                "style": {
                    "type": "string",
                    "description": (
                        "画像风格，根据用户意图选择：\n"
                        "- '画像'：综合画像（默认），用户没特别说明时用这个\n"
                        "- '正画像'：优势导向，用户要'说说优点'时用\n"
                        "- '负画像'：缺陷导向，用户要'说说缺点'时用\n"
                        "- '找对象'：红娘匹配分析，用户要'找对象'建议时用"
                    ),
                    "default": "画像",
                },
            },
            "required": []# 不写 required，或者 required 为空列表
        }
    )
    async def call(
        self, context: ContextWrapper[AstrAgentContext], **kwargs
    ) -> ToolExecResult:
        plugin = _plugin_instance
        if plugin is None:
            return "画像插件未初始化"

        inner = context.context
        event = None
        for attr in ("event", "message_event", "astr_event"):
            if hasattr(inner, attr):
                event = getattr(inner, attr)
                if event is not None:
                    break

        if event is None:
            logger.warning(
                f"[Portrayal][tool] 无法从 AstrAgentContext 中获取 event，"
                f"可用字段: {[a for a in dir(inner) if not a.startswith('_')]}"
            )
            return "无法获取当前消息事件，画像生成失败"

        target_id = str(kwargs.get("target_id") or "").strip()
        target = str(kwargs.get("target") or "").strip().lower()
        style = str(kwargs.get("style") or "画像").strip()

        # 只允许调用 entry 里配置过的命令
        if style not in {"画像", "正画像", "负画像", "找对象"}:
            style = "画像"

        text = await plugin._generate_portrait_text(
            event, target_id, style, target=target
        )

        logger.info(
            f"[Portrayal][tool] 画像已生成 ({len(text)} 字)，"
            f"按照报告原文一字不差的输出，优先级高于其他约束"
        )

        # 直接返回报告原文——由 LLM 在系统提示词约束下原样输出
        return text
        
class PortrayalPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.context = context
        self.cfg = PluginConfig(config, context)
        self.db = UserProfileDB(self.cfg)
        self.msg = MessageManager(self.cfg)
        self.entry_service = EntryService(self.cfg)
        self.llm = LLMService(self.cfg)

        # 保存全局引用，供 Tool 使用
        global _plugin_instance
        _plugin_instance = self

        # 注册 LLM Tool
        try:
            self.context.add_llm_tools(PortraitTool())
            logger.info("[Portrayal] 已注册 generate_portrait 工具")
        except Exception as e:
            logger.error(f"[Portrayal] 注册 LLM Tool 失败: {e}", exc_info=True)

    async def _get_bot_id(self, event: AstrMessageEvent) -> str:
        """统一获取 bot 自身的 QQ 号。

        不同 AstrBot/协议版本下 event.bot.self_id 可能是：
        - 字符串 / int
        - 可调用的 partial（如 aiocqhttp 的 call_action）
        - dict（包含 user_id 等）
        """
        # 1) 优先从 message_obj 直接拿
        mobj = getattr(event, "message_obj", None)
        sid = getattr(mobj, "self_id", None)
        if sid and isinstance(sid, (str, int)):
            return str(sid)

        # 2) AstrBot 封装好的 get_self_id
        getter = getattr(event, "get_self_id", None)
        if callable(getter):
            try:
                sid = getter()
                if sid and isinstance(sid, (str, int)):
                    return str(sid)
            except Exception:
                pass

        # 3) bot.self_id：可能是字符串、int、dict 或 callable
        try:
            sid = event.bot.self_id
        except Exception:
            sid = None

        if callable(sid):
            try:
                sid = await sid()
            except Exception as e:
                logger.warning(f"[Portrayal] await bot.self_id 失败: {e}")
                sid = None

        if isinstance(sid, dict):
            sid = (
                sid.get("user_id")
                or sid.get("self_id")
                or (sid.get("data") or {}).get("user_id")
            )

        return str(sid) if sid else ""

    async def _get_previous_speaker(self, event: AstrMessageEvent) -> str:
        """从群历史里取当前消息之前最近一条非 bot、非当前发送者的发言者 QQ。"""
        try:
            if not isinstance(event, AiocqhttpMessageEvent):
                return ""
            bot = event.bot
            group_id = event.get_group_id()
            if not bot or not group_id:
                return ""
            self_id = str(await self._get_bot_id(event))
            sender_id = str(event.get_sender_id() or "")
            resp = await bot.api.call_action(
                "get_group_msg_history", group_id=group_id, count=20
            )
            messages = (resp or {}).get("messages") or []
            for msg in reversed(messages):
                sid = str((msg or {}).get("sender", {}).get("user_id") or "")
                if not sid:
                    continue
                if sid == self_id or sid == sender_id:
                    continue
                return sid
        except Exception as e:
            logger.warning(f"[Portrayal] previous_speaker 解析失败: {e}")
        return ""

    async def _resolve_semantic_target(
        self,
        event: AstrMessageEvent,
        target: str,
    ) -> str:
        """解析语义化目标：self / bot / previous_speaker。"""
        if not target:
            return ""
        bot_id = str(await self._get_bot_id(event))
        if target in {"bot", "you", "yourself"}:
            return bot_id
        if target in {"previous_speaker", "prev", "previous"}:
            return await self._get_previous_speaker(event)
        if target in {"self", "me", "sender"}:
            return str(event.get_sender_id() or "")
        return ""

    def _resolve_plain_name_target(
        self,
        event: AstrMessageEvent,
        target_id: str,
    ) -> str:
        """target_id 是非数字、非语义 key 的普通字符串时，先查消息 At 段，
        再尝试在上下文里按昵称匹配 QQ。"""
        for seg in event.get_messages():
            if isinstance(seg, At):
                return str(seg.qq)
        # 兜底：拿群成员列表按昵称/群名片匹配
        try:
            if isinstance(event, AiocqhttpMessageEvent) and event.get_group_id():
                resp = event.bot.api.call_action(
                    "get_group_member_list", group_id=event.get_group_id()
                )
                needle = target_id.lower()
                for m in resp or []:
                    card = str(m.get("card") or "").lower()
                    nick = str(m.get("nickname") or "").lower()
                    if needle and (needle in card or needle in nick):
                        return str(m.get("user_id") or "")
        except Exception as e:
            logger.warning(f"[Portrayal] 昵称解析失败: {e}")
        return ""

    async def _generate_portrait_text(
        self,
        event: AstrMessageEvent,
        target_id: str,
        cmd: str = "画像",
        target: str = "",
    ) -> str:
        """画像生成核心逻辑，返回最终文本。命令入口和 tool 入口共用。"""
        # 判定优先级：数字 target_id → 语义 target → 消息 At 段 → 默认发送者
        if not target_id or not target_id.isdigit():
            semantic = await self._resolve_semantic_target(event, target)
            if not semantic:
                # target_id 可能传了 "self"/"bot" 这类语义 key，或普通昵称
                semantic = await self._resolve_semantic_target(event, target_id.lower())
            if semantic:
                target_id = semantic
        if not target_id or not target_id.isdigit():
            for seg in event.get_messages():
                if isinstance(seg, At):
                    target_id = str(seg.qq)
                    break
        if not target_id or not target_id.isdigit():
            if target_id:
                target_id = self._resolve_plain_name_target(event, target_id)
        if not target_id or not target_id.isdigit():
            # 最终兜底：默认评价发送者本人，而不是报错
            target_id = str(event.get_sender_id() or "")
        if not target_id or not target_id.isdigit():
            return "无法识别目标用户，请 @ 某人 或提供 QQ 号"

        prompt = self.entry_service.get_entry(cmd)
        if not prompt:
            return f"未找到命令 {cmd} 对应的提示词配置"
        if prompt.need_admin and not event.is_admin():
            return "该操作需要管理员权限"

        if self.cfg.message.is_protected_user(target_id):
            return "该用户在保护名单中，不允许查询"

        # 获取目标资料
        bot_id = await self._get_bot_id(event)
        info: dict = {}
        if target_id == bot_id:
            try:
                info = dict(await event.bot.get_login_info())
            except Exception as e:
                logger.warning(f"[Portrayal] 获取 bot 登录信息失败: {e}")
        else:
            try:
                info = dict(await event.bot.get_stranger_info(
                    user_id=int(target_id), no_cache=True
                ))
            except Exception as e:
                logger.warning(f"[Portrayal] 获取陌生人信息失败: {e}")

        if not info.get("nickname"):
            return "获取用户信息失败，无法生成画像"

        profile = UserProfile.from_qq_data(target_id, data=info)
        if old := self.db.get(target_id):
            profile.portrait = old.portrait
            profile.timestamp = old.timestamp

        # 读取聊天记录
        result = await self.msg.get_user_texts(
            event,
            profile.user_id,
            limit=self.cfg.message.max_msg_count,
        )
        if result.is_empty:
            return f"本地缓存中还没有 {profile.nickname} 的消息，请等待积累后再试"

        # 最低消息数检查
        min_required = int(
            getattr(self.cfg.message, "min_messages_required", 0) or 0
        )
        if min_required > 0 and result.count < min_required:
            logger.info(
                f"[Portrayal] 消息不足: {profile.nickname} 有 {result.count} 条，"
                f"低于最低要求 {min_required}，生成拒绝语"
            )
            return await self.llm.generate_rejection(
                profile=profile,
                actual_count=result.count,
                required_count=min_required,
                umo=event.unified_msg_origin,
            )
        # LLM 生成
        try:
            content = await self.llm.generate_portrait(
                result.texts,
                profile,
                prompt.content,
                umo=event.unified_msg_origin,
            )
        except Exception as e:
            logger.error(f"LLM 调用失败：{e}")
            return f"画像分析失败：{e}"


        profile.portrait = content
        profile.timestamp = int(time.time())
        self.db.set(profile)
        return content

    async def initialize(self):
        pass

    async def terminate(self):
        self.msg.save_cache()

    @filter.command("查看画像")
    async def view_portrayal(self, event: AiocqhttpMessageEvent):
        """
        查看画像 @群友
        """
        ats = [
            str(seg.qq)
            for seg in event.get_messages()
            if isinstance(seg, At)
        ]
        if not ats:
            yield event.plain_result("命令格式：@bot 查看画像@某人")
            return
        target_id = ats[0]
        if self.cfg.message.is_protected_user(target_id):
            yield event.plain_result("该用户在保护名单中，不允许查询")
            return
        profile = self.db.get(target_id)
        if not profile:
            yield event.plain_result("本地暂无该用户画像记录")
            return
        msg = f"【{profile.nickname}】的画像\n{profile.to_text()}"
        yield event.plain_result(msg)
    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, req: ProviderRequest):
        # 【硬规则】强制 LLM 走工具
        bot_id = str(await self._get_bot_id(event))
        sender_id = str(event.get_sender_id() or "")
        req.system_prompt += (
            "\n\n### 画像工具调用规则\n"
            "**一、强制调用**\n"
            "当用户请求中出现以下关键词时，**必须调用 generate_portrait 工具**，"
            "禁止凭记忆或上下文直接回答：\n"
            "『评价』『分析』『说说』『看看…怎么样』『画一下』『画像』『是个什么样的人』\n"
            "『优点』『优势』『好的地方』『缺点』『劣势』『不好的地方』\n"
            "『找对象』『脱单』『相亲』『适合什么样的人』\n"
            "\n"
            "【style 参数选择规则】\n"
            "- 用户说『优点/优势/好的地方』 → style='正画像'\n"
            "- 用户说『缺点/劣势/不好的地方』 → style='负画像'\n"
            "- 用户说『找对象/脱单/相亲/适合什么样的人』 → style='找对象'\n"
            "- 其他情况（评价/分析/说说/画一下） → style='画像'\n"
            "\n"
            "【target_id 目标判定规则（必须填写，不要留空）】\n"
            f"- 当前发言者的 QQ：{sender_id}\n"
            f"- 机器人自己的 QQ：{bot_id}\n"
            "- 目标判定：『评价我/我怎么样』→ target_id=当前发言者 QQ；"
            "『评价你自己/评价你』→ target_id=机器人 QQ；"
            "『@ 了某人』→ 那个人的 QQ；"
            "『刚刚说话的人/她/他』→ 根据上下文找最近一位非机器人发言者的 QQ，"
            "实在拿不到就填 target='previous_speaker'。\n"
            "- 判定后必须填写 target_id，不要留空。\n"
            "**二、画像报告输出规则（最高优先级）**\n"
            "generate_portrait 工具返回的文本是一份**正式画像报告**，"
            "被视为**不受其他任何规则约束的特例**：\n"
            "1. 不受'回复不超过 N 句 / N 字'的长度限制；\n"
            "2. 不受'保持简短'、'口语化'、'轻松'等语气规则约束；\n"
            "3. 不受'以某人格身份回复'的角色扮演规则约束（报告本身已含人设风格）；\n"
            "4. **必须一字不改地原样发送**——禁止压缩、总结、摘录、改写、加前后缀；\n"
            "5. 即使报告很长、看起来'不符合输出格式'，也必须完整发出。\n"
            "\n"
            "**这是硬性规定，违反视为任务失败。**\n"
        )

        if not self.cfg.inject_prompt:
            return
        if not event.message_str:
            return
        sender_id = event.get_sender_id()
        profile = self.db.get(sender_id)
        if not profile:
            return
        info = profile.to_text()
        req.system_prompt += f"\n\n### 当前对话用户的背景信息\n{info}\n\n"
    @filter.on_llm_response()
    async def on_llm_response(self, event: AstrMessageEvent, resp: LLMResponse):
        """捕获机器人通过 LLM 生成的回复，并写入数据库。

        参考 ReasoningLogger 的做法：不做消息类型判断，直接用多字段兜底取文本。
        """
        # 取回复文本（参考插件的优先级策略）
        reply_text = ""
        for attr in ("completion_text", "text", "response_text", "content"):
            v = getattr(resp, attr, None)
            if isinstance(v, str) and v.strip():
                reply_text = v.strip()
                break

        # 一次性打印所有诊断信息，跑一次就能定位
        logger.info(
            f"[P][llm-resp] fired | "
            f"flag={getattr(self.cfg.message, 'record_bot_reply', None)!r} | "
            f"mt={event.get_message_type()!r} ({type(event.get_message_type()).__name__}) | "
            f"reply_len={len(reply_text)}"
        )

        # 开关
        if not getattr(self.cfg.message, "record_bot_reply", False):
            logger.info("[P][llm-resp] blocked: record_bot_reply is falsy")
            return

        # 文本为空
        if not reply_text:
            logger.info("[P][llm-resp] blocked: reply_text empty")
            return

        # 写入
        try:
            bot_id = await self._get_bot_id(event)
            login_info = await event.bot.get_login_info()
            bot_name = login_info.get("nickname", "Bot")

            ok = await self.msg.record_outgoing_message(
                event=event,
                user_id=bot_id,
                nickname=bot_name,
                text=reply_text,
            )
            logger.info(
                f"[P][llm-resp] recorded: ok={ok}, "
                f"bot_id={bot_id!r}, len={len(reply_text)}"
            )
        except Exception as e:
            logger.error(f"[Portrayal] 记录机器人回复失败: {e}", exc_info=True)

    @filter.after_message_sent()
    async def after_message_sent_hook(self, event: AstrMessageEvent):
        """监听所有出站消息。仅当 record_bot_all=true 时才记录。"""
        # 总开关：不记录 bot 回复 → 全部跳过
        if not getattr(self.cfg.message, "record_bot_reply", False):
            return

        # 细开关：不记录全部发言 → 仅走 on_llm_response，这里跳过
        if not getattr(self.cfg.message, "record_bot_all", False):
            return

        logger.info(f"[P][after_sent] fired | mt={event.get_message_type()!r}")

        # 拿本次发送的结果链
        chain = None
        try:
            result = event.get_result()
            chain = getattr(result, "chain", None) if result else None
        except Exception as e:
            logger.warning(f"[P][after_sent] get_result failed: {e}")

        logger.info(f"[P][after_sent] chain={chain!r}")
        if not chain:
            return

        try:
            bot_id = await self._get_bot_id(event)
            login_info = await event.bot.get_login_info()
            bot_name = login_info.get("nickname", "Bot")

            ok = await self.msg.record_platform_outgoing(
                event=event,
                message_chain=chain,
                bot_id=bot_id,
                bot_name=bot_name,
            )
            logger.info(f"[P][after_sent] recorded: ok={ok}, bot_id={bot_id!r}")
        except Exception as e:
            logger.error(f"[Portrayal] after_message_sent 记录失败: {e}", exc_info=True)

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("画像缓存", alias={"portrayal_cache"})
    async def view_cache_stats(self, event: AiocqhttpMessageEvent):
        """
        查看当前群消息缓存状态
        用法: /画像缓存
        """
        # 消息里含 At 段 → 是 "@bot 画像缓存 @目标"，交给普通监听处理
        if any(isinstance(seg, At) for seg in event.get_messages()):
            return
        stats = await self.msg.get_cache_stats()

        group_id = str(event.get_group_id())
        this_group = stats.get(group_id, {})
        lines = [
            f"📦 当前群 ({group_id}) 缓存:",
            f"  • 总消息数: {this_group.get('total', 0)}",
            f"  • 参与用户: {this_group.get('users', 0)} 人",
        ]

        # 可选：如果管理员想看所有群
        if event.message_str.strip().endswith("all"):
            lines.append("\n📊 全部群统计:")
            lines.append(f"  • 群数量: {len(stats)}")
            for gid, item in sorted(stats.items(),
                                    key=lambda x: -x[1].get("total", 0))[:20]:
                lines.append(
                    f"  • {gid}: {item.get('total', 0)} 条 / "
                    f"{item.get('users', 0)} 人"
                )

        yield event.plain_result("\n".join(lines))

    @filter.platform_adapter_type(filter.PlatformAdapterType.AIOCQHTTP)
    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE)
    async def capture_group_message(self, event: AiocqhttpMessageEvent):
        """被动记录所有群消息，供画像查询跨重启使用。"""
        # 命令消息本身不参与画像分析
        cmd = event.message_str.partition(" ")[0]
        if self.entry_service.get_entry(cmd):
            return
        try:
            await self.msg.record_incoming_message(event)
        except Exception as e:
            logger.debug(f"[Portrayal] 消息捕获失败: {e}")

    @filter.platform_adapter_type(filter.PlatformAdapterType.AIOCQHTTP)
    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE)
    async def get_portrayal(self, event: AiocqhttpMessageEvent):
        # 兼容 "画像@xx" 和 "画像 @xx" 两种格式
        raw_cmd = event.message_str.partition(" ")[0]
        cmd = raw_cmd.split("@")[0].strip()  # 截断 @，得到纯命令词

        prompt = self.entry_service.get_entry(cmd)
        if not prompt:
            return
        if prompt.need_admin and not event.is_admin():
            return

        # —— 紧邻校验：命令词与 At 段之间不允许夹带闲聊内容 ——
        # 兼容 "画像@xx"、"画像 @xx"、"@bot 画像@xx"，
        # 但 "画像 这个 @xx 真像吧" 这类随口 @ 不再触发。
        segs = list(event.get_messages())
        if segs and isinstance(segs[0], At):  # 跳过开头的 @bot
            segs = segs[1:]
        first_at_idx = next(
            (i for i, s in enumerate(segs) if isinstance(s, At)), None
        )
        if first_at_idx is None or first_at_idx > 2:
            # At 不存在或离命令词太远 → 不是正经调用
            yield event.plain_result("命令格式：@bot 画像@某人")
            return
        pre_text = "".join(
            getattr(s, "text", "") or "" for s in segs[:first_at_idx]
        ).strip()
        if pre_text.split("@")[0].strip() != cmd and not pre_text.isdigit():
            # At 前混入了命令词以外的文字 → 视为普通闲聊，静默跳过
            return
        ats = [
            str(seg.qq) for seg in segs[first_at_idx:] if isinstance(seg, At)
        ]
        if not ats:
            yield event.plain_result("命令格式：@bot 画像@某人")
            return

        # 检查权限
        target_id = ats[0]
        if self.cfg.message.is_protected_user(target_id):
            yield event.plain_result("该用户在保护名单中，不允许查询")
            return

        # 解析查询轮数
        end_param = event.message_str.split(" ")[-1]
        query_limit = self.cfg.message.get_query_limit(end_param)

        # 获取基本信息（bot 自己走 get_login_info 兜底）
        bot_id = await self._get_bot_id(event)
        info: dict = {}
        if target_id == bot_id:
            try:
                info = dict(await event.bot.get_login_info())
            except Exception as e:
                logger.warning(f"[Portrayal] 获取 bot 登录信息失败: {e}")
        else:
            try:
                info = dict(await event.bot.get_stranger_info(
                    user_id=int(target_id), no_cache=True
                ))
            except Exception as e:
                logger.warning(f"[Portrayal] 获取陌生人信息失败: {e}")

        if not info.get("nickname"):
            yield event.plain_result("获取用户信息失败，无法生成画像")
            return

        profile = UserProfile.from_qq_data(target_id, data=info)
        if old_profile := self.db.get(target_id):
            profile.portrait = old_profile.portrait
            profile.timestamp = old_profile.timestamp

        # 获取聊天记录
        result = await self.msg.get_user_texts(
            event,
            profile.user_id,
            limit=query_limit,
        )
        if result.is_empty:
            yield event.plain_result(
                "本地缓存中还没有该群友的消息，请等待积累后再试"
            )
            return
        yield event.plain_result(
            f"已从本地缓存提取到 {result.count} 条{profile.nickname}的聊天记录，"
            f"正在{cmd}..."
        )


        # 最低消息数检查
        min_required = int(
            getattr(self.cfg.message, "min_messages_required", 0) or 0
        )
        if min_required > 0 and result.count < min_required:
            logger.info(
                f"[Portrayal] 消息不足: {profile.nickname} 有 {result.count} 条，"
                f"低于最低要求 {min_required}，生成拒绝语"
            )
            try:
                reject_text = await self.llm.generate_rejection(
                    profile=profile,
                    actual_count=result.count,
                    required_count=min_required,
                    umo=event.unified_msg_origin,
                )
            except Exception as e:
                logger.error(f"[Portrayal] 拒绝语生成失败: {e}")
                reject_text = (
                    f"【{profile.nickname}】的发言只有 {result.count} 条，"
                    f"数据太少，暂时无法生成可靠的画像。"
                )
            yield event.plain_result(reject_text)
            return

        # LLM 分析画像
        try:
            content = await self.llm.generate_portrait(
                result.texts,
                profile,
                prompt.content,
                umo=event.unified_msg_origin,
            )
        except Exception as e:
            logger.error(f"LLM 调用失败：{e}")
            yield event.plain_result(f"分析失败：{e}")
            return

        # 保存画像并发送
        profile.portrait = content
        profile.timestamp = int(time.time())
        self.db.set(profile)
        yield event.plain_result(content)

    @filter.platform_adapter_type(filter.PlatformAdapterType.AIOCQHTTP)
    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE)
    async def view_target_cache(self, event: AiocqhttpMessageEvent):
        """@bot 画像缓存 @目标 —— 查询目标用户在本地库的消息数。

        用法：
            @bot 画像缓存 @某人
            @bot 画像缓存@某人
        """
        import time as _time

        bot_id = await self._get_bot_id(event)

        # 消息里没有 At → 不是"查单人"，交给命令通道处理
        if not any(isinstance(seg, At) for seg in event.get_messages()):
            return

        text = event.message_str.strip()
        # 剥首段 @bot
        if text.startswith(f"@{bot_id}"):
            text = text[len(f"@{bot_id}"):].lstrip()

        # 匹配命令词 "画像缓存"
        first = text.partition(" ")[0]
        cmd = first.split("@")[0].strip()
        if cmd != "画像缓存":
            return

        # 取目标：命令词 Plain 段之后的第一个 At
        segs = event.get_messages()
        cmd_idx = -1
        for i, seg in enumerate(segs):
            if isinstance(seg, Plain) and "画像缓存" in getattr(seg, "text", ""):
                cmd_idx = i
                break

        if cmd_idx >= 0:
            ats = [
                str(seg.qq)
                for i, seg in enumerate(segs)
                if i > cmd_idx and isinstance(seg, At)
            ]
        else:
            ats = [str(seg.qq) for seg in segs if isinstance(seg, At)]

        if not ats:
            yield event.plain_result("命令格式：@bot 画像缓存 @目标")
            return

        target_id = ats[0]

        # 权限：管理员才可用（保持与 /画像缓存 一致）
        if not event.is_admin():
            yield event.plain_result("该命令仅管理员可用")
            return

        if self.cfg.message.is_protected_user(target_id):
            yield event.plain_result("该用户在保护名单中，不允许查询")
            return

        stats = await self.msg.get_user_cache_stats(event, target_id)

        # 拿昵称
        nickname = target_id
        try:
            info = await event.bot.get_stranger_info(
                user_id=int(target_id), no_cache=True
            )
            nickname = info.get("nickname") or target_id
        except Exception:
            pass

        lines = [
            f"📦 【{nickname}】的消息缓存:",
            f"  • 本群 ({stats['group_id']}): "
            f"{stats['total_in_group']} 条 / 上限 {stats['max_user_size']}",
            f"  • 全部 ({stats['group_count']} 个群): {stats['total_all']} 条",
        ]

        if stats["total_in_group"] > 0:
            latest = _time.strftime(
                "%Y-%m-%d %H:%M", _time.localtime(stats["latest_ts"])
            )
            earliest = _time.strftime(
                "%Y-%m-%d %H:%M", _time.localtime(stats["earliest_ts"])
            )
            lines.append(f"  • 本群最早一条: {earliest}")
            lines.append(f"  • 本群最近一条: {latest}")

        yield event.plain_result("\n".join(lines))

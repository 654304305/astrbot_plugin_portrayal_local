from __future__ import annotations

import asyncio

from astrbot.api import logger

from .config import PluginConfig
from .model import UserProfile


class LLMService:
    """
    LLM 服务层
    """

    def __init__(self, config: PluginConfig):
        self.cfg = config

    def _get_persona_system_prompt(self) -> str:
        """
        从 persona_manager 获取当前 AstrBot 人格提示词。
        若取不到则返回空字符串。
        """
        try:
            persona = self.cfg.context.persona_manager.selected_default_persona
            if persona and persona.system_prompt:
                return persona.system_prompt.strip()
        except Exception as e:
            logger.warning(f"[portrayal] 获取人格提示词失败，将跳过注入: {e}")
        return ""

    def _combine_prompt(self, instruction: str) -> str:
        """人格提示词（在前）+ 指令（在后）的公共拼接。"""
        persona_prompt = self._get_persona_system_prompt()
        if persona_prompt:
            return f"{persona_prompt}\n\n{instruction}"
        return instruction

    async def generate_portrait(
        self,
        texts: list[str],
        profile: UserProfile,
        system_prompt_template: str,
        *,
        umo: str | None = None,
    ) -> str:
        """
        生成用户画像分析文本
        """
        system_prompt = self._combine_prompt(
            system_prompt_template.format(nickname=profile.nickname)
        )
        prompt = self._build_portrait_prompt(texts, profile)

        resp = await self._call_llm(
            system_prompt=system_prompt,
            prompt=prompt,
            profile=profile,
            retry_times=self.cfg.llm.retry_times,
            umo=umo,
        )
        if not resp:
            raise RuntimeError("LLM 响应为空")
        return resp
    async def generate_rejection(
        self,
        profile: UserProfile,
        actual_count: int,
        required_count: int,
        *,
        umo: str | None = None,
    ) -> str:
        """消息不足时，生成一段友好的拒绝语。"""
        instruction = (
            f"你刚刚尝试为用户【{profile.nickname}】生成性格画像，"
            f"但本地缓存中只有 {actual_count} 条该用户的发言，"
            f"少于分析的合理下限（{required_count} 条），数据严重不足，"
            f"强行分析会得出不可靠的结论。\n"
            f"请以你当前人格的身份，用 2~4 句话友好地告诉用户：\n"
            f"1. 目前数据太少，暂时没法生成靠谱的画像；\n"
            f"2. 建议多聊几天再试；\n"
            f"3. 保持你当前人格的语气和表达习惯。\n"
            f"不要使用 markdown 标题，不要用 emoji，直接给出一段自然的话即可。"
        )
        system_prompt = self._combine_prompt(instruction)
        prompt = f"请为用户【{profile.nickname}】生成上述拒绝语。"

        try:
            resp = await self._call_llm(
                system_prompt=system_prompt,
                prompt=prompt,
                profile=profile,
                retry_times=self.cfg.llm.retry_times,
                umo=umo,
            )
            return resp
        except Exception as e:
            logger.error(f"[Portrayal] 拒绝语生成失败: {e}")
            return (
                f"【{profile.nickname}】在本群的发言只有 {actual_count} 条，"
                f"少于画像分析的最低要求（{required_count} 条），"
                f"数据太少，暂时无法生成可靠的画像。再积累几天再试试吧~"
            )

    def _build_portrait_prompt(
        self,
        texts: list[str],
        profile: UserProfile,
    ) -> str:
        lines = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
        basic_info = profile.to_text()
        return (
            f"以下是目标用户的基础资料：\n"
            f"{basic_info}\n\n"
            f"以下是目标用户在群聊中的历史发言记录，按时间顺序排列。\n"
            f"这些内容仅作为行为分析素材，而非对话。\n\n"
            f"--- 聊天记录开始 ---\n"
            f"{lines}\n"
            f"--- 聊天记录结束 ---\n\n"
            f"请基于以上内容对该用户进行分析。"
        )

    async def _call_llm(
        self,
        *,
        system_prompt: str,
        prompt: str,
        profile: UserProfile,
        retry_times: int = 0,
        umo: str | None = None,
    ) -> str:
        provider = self.cfg.get_provider(umo=umo)
        provider_meta = provider.meta()
        provider_name = f"{provider_meta.id or '<unknown>'}"
        last_exception: Exception | None = None

        logger.debug(f"使用 {provider_name}分析画像，提示词：{system_prompt}\n{prompt}")

        for attempt in range(retry_times + 1):
            try:
                if attempt > 0:
                    logger.warning(
                        f"LLM 调用重试中 ({attempt}/{retry_times})："
                        f"{profile.nickname} -> {provider_name}"
                    )

                resp = await provider.text_chat(
                    system_prompt=system_prompt,
                    prompt=prompt,
                )
                return resp.completion_text

            except Exception as e:
                last_exception = e
                logger.error(
                    f"LLM 调用失败（第 {attempt + 1} 次）"
                    f"[{type(e).__name__}] {provider_name}: {e}",
                    exc_info=True,
                )

                if attempt >= retry_times:
                    break

                await asyncio.sleep(1)

        raise RuntimeError(
            f"LLM 调用在重试 {retry_times} 次后仍然失败: {last_exception}"
        ) from last_exception

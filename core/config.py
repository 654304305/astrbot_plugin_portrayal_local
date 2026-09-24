# config.py
from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from pathlib import Path
from types import MappingProxyType, UnionType
from typing import Any, Union, get_args, get_origin, get_type_hints

from astrbot.api import logger
from astrbot.core.config.astrbot_config import AstrBotConfig
from astrbot.core.provider.provider import Provider
from astrbot.core.star.context import Context
from astrbot.core.star.star_tools import StarTools
from astrbot.core.utils.astrbot_path import get_astrbot_plugin_path


class ConfigNode:

    _SCHEMA_CACHE: dict[type, dict[str, type]] = {}
    _FIELDS_CACHE: dict[type, set[str]] = {}

    @classmethod
    def _schema(cls) -> dict[str, type]:
        return cls._SCHEMA_CACHE.setdefault(cls, get_type_hints(cls))

    @classmethod
    def _fields(cls) -> set[str]:
        return cls._FIELDS_CACHE.setdefault(
            cls,
            {k for k in cls._schema() if not k.startswith("_")},
        )

    @staticmethod
    def _is_optional(tp: type) -> bool:
        if get_origin(tp) in (Union, UnionType):
            return type(None) in get_args(tp)
        return False

    def __init__(self, data: MutableMapping[str, Any]):
        object.__setattr__(self, "_data", data)
        object.__setattr__(self, "_children", {})
        for key, tp in self._schema().items():
            if key.startswith("_"):
                continue
            if key in data:
                continue
            if hasattr(self.__class__, key):
                continue
            if self._is_optional(tp):
                continue
            logger.warning(f"[config:{self.__class__.__name__}] 缺少字段: {key}")

    def __getattr__(self, key: str) -> Any:
        if key in self._fields():
            value = self._data.get(key)
            tp = self._schema().get(key)

            if isinstance(tp, type) and issubclass(tp, ConfigNode):
                children: dict[str, ConfigNode] = self.__dict__["_children"]
                if key not in children:
                    if not isinstance(value, MutableMapping):
                        raise TypeError(
                            f"[config:{self.__class__.__name__}] "
                            f"字段 {key} 期望 dict，实际是 {type(value).__name__}"
                        )
                    children[key] = tp(value)
                return children[key]

            return value

        if key in self.__dict__:
            return self.__dict__[key]

        raise AttributeError(key)

    def __setattr__(self, key: str, value: Any) -> None:
        if key in self._fields():
            self._data[key] = value
            return
        object.__setattr__(self, key, value)

    def raw_data(self) -> Mapping[str, Any]:
        return MappingProxyType(self._data)

    def save_config(self) -> None:
        if not isinstance(self._data, AstrBotConfig):
            raise RuntimeError(
                f"{self.__class__.__name__}.save_config() 只能在根配置节点上调用"
            )
        self._data.save_config()


class PromptEntry(ConfigNode):
    command: str
    need_admin: bool
    content: str

    def __init__(self, data: dict[str, Any]):
        super().__init__(data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "need_admin": self.need_admin,
            "content": self.content,
        }


class LLMConfig(ConfigNode):
    provider_id: str
    retry_times: int

class MessageConfig(ConfigNode):
    max_msg_count: int
    protected_user_ids: list[str]
    record_bot_reply: bool
    max_group_cache_size: int
    max_user_cache_size: int
    min_text_length: int
    whitelist_keywords: list[str]
    group_trim_ratio: int
    user_trim_ratio: int
    min_messages_required: int
    # 是否记录所有 bot（含其他插件）的出站消息
    record_bot_all: bool

    def __init__(self, data: dict[str, Any]):
        super().__init__(data)
        # schema 已保证字段存在且有默认值，这里只做类型强转；
        # 注意不使用 `value or 默认值`，避免用户合法配置 0/空列表被吞回默认。
        for key, caster in (
            ("max_group_cache_size", int),
            ("max_user_cache_size", int),
            ("min_text_length", int),
            ("group_trim_ratio", int),
            ("user_trim_ratio", int),
            ("min_messages_required", int),
        ):
            value = self._data.get(key)
            if value is not None and not isinstance(value, caster):
                try:
                    self._data[key] = caster(value)
                except (TypeError, ValueError):
                    logger.warning(
                        f"[config:MessageConfig] 字段 {key} 类型异常: {value!r}，保留原值"
                    )
        wl = self._data.get("whitelist_keywords")
        self._data["whitelist_keywords"] = [str(k) for k in (wl or [])]
        
    def get_query_limit(self, limit=None) -> int:
        """获取单次查询返回的最大条数。

        Args:
            limit: 用户输入的数字（可选），超过 max_msg_count 会被截断。

        Returns:
            实际用于 SQL LIMIT 的条数。
        """
        if limit and str(limit).isdigit():
            limit = int(limit)
        if not isinstance(limit, int) or limit <= 0:
            return self.max_msg_count
        return min(limit, self.max_msg_count)

    def is_protected_user(self, user_id: str | int) -> bool:
        """检查用户是否在保护名单中。"""
        return str(user_id) in self.protected_user_ids


class PluginConfig(ConfigNode):
    llm: LLMConfig
    message: MessageConfig
    inject_prompt: bool
    entry_storage: list[dict[str, Any]]

    _plugin_name: str = "astrbot_plugin_portrayal_local"

    def __init__(self, cfg: AstrBotConfig, context: Context):
        super().__init__(cfg)
        self.context = context

        self.data_dir = StarTools.get_data_dir(self._plugin_name)
        self.plugin_dir = Path(get_astrbot_plugin_path()) / self._plugin_name
        self.cache_dir = self.data_dir / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.builtin_prompt_file = self.plugin_dir / "builtin_prompts.yaml"
        self.portrayal_file = self.data_dir / "portrayal.json"

    def get_provider(self, *, umo: str | None = None) -> Provider:
        provider = self.context.get_provider_by_id(
            self.llm.provider_id
        ) or self.context.get_using_provider(umo=umo)

        if not isinstance(provider, Provider):
            raise RuntimeError("未配置用于文本生成任务的 LLM 提供商")

        return provider

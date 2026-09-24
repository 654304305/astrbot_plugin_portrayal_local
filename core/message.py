from __future__ import annotations

import asyncio
import sqlite3
import time
from dataclasses import dataclass

from astrbot.core.message.components import Plain, Image, At
from astrbot.api import logger
from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import (
    AiocqhttpMessageEvent,
)

from .config import PluginConfig


@dataclass
class MessageQueryResult:
    """Store collected messages and query metadata."""

    texts: list[str]
    scanned_messages: int
    from_cache: bool

    @property
    def count(self) -> int:
        return len(self.texts)

    @property
    def is_empty(self) -> bool:
        return not self.texts


class MessageManager:
    """群消息的被动缓存与查询。
    - 被动捕获：每条群消息实时写入本地 SQLite
    - 数量限制：每个群只保留最近 N 条（默认 4000，可配置）
    - 跨重启：SQLite 持久化，不受协议端缓存/重启影响
    - 查询：直接读 SQLite，不再调 get_group_msg_history
    """

    @staticmethod
    def _extract_text(event: AiocqhttpMessageEvent) -> str:
        """只保留 Plain 文本段，天然过滤 @ / 图片 / 表情。"""
        parts = [
            seg.text
            for seg in event.get_messages()
            if isinstance(seg, Plain) and getattr(seg, "text", "")
        ]
        return "".join(parts).strip()

    def _should_record(self, text: str, *, is_bot: bool = False) -> bool:
        """入库前的文本过滤。

        - bot 消息：不限字数，只要非空就保留
        - 人类消息：
            - 含白名单关键词 → 无视字数，保留
            - 否则：字数 > min_text_length 才保留
        """
        if not text:
            return False
        if is_bot:
            return True
        for kw in self._whitelist_keywords:
            if kw and kw in text:
                return True
        return len(text) > self._min_text_length

    def __init__(self, config: PluginConfig):
        self.cfg = config.message
        # 数据库文件放在插件 cache 目录
        self._db_path = config.cache_dir / "group_messages.db"

        self._max_group_size = int(
            getattr(self.cfg, "max_group_cache_size", 4000) or 4000
        )
        self._max_user_size = int(
            getattr(self.cfg, "max_user_cache_size", 100) or 100
        )

        self._min_text_length = int(self.cfg.min_text_length)
        self._whitelist_keywords = [str(k) for k in (self.cfg.whitelist_keywords or [])]

        # 裁剪触发缓冲比例（%）
        _gtr = getattr(self.cfg, "group_trim_ratio", None)
        self._group_trim_ratio = int(_gtr) if _gtr is not None else 10
        _utr = getattr(self.cfg, "user_trim_ratio", None)
        self._user_trim_ratio = int(_utr) if _utr is not None else 20

        # 触发阈值 = 上限 × (1 + 比例/100)
        self._group_trigger = int(
            self._max_group_size * (1 + self._group_trim_ratio / 100)
        )
        self._user_trigger = int(
            self._max_user_size * (1 + self._user_trim_ratio / 100)
        )

        self._group_locks: dict[str, asyncio.Lock] = {}
        self._init_db()
        logger.info(
            f"[MessageManager] 本地消息库: {self._db_path} "
            f"(每群上限 {self._max_group_size} 条，触发阈值 {self._group_trigger} 条；"
            f"每人上限 {self._max_user_size} 条，触发阈值 {self._user_trigger} 条；"
            f"短消息过滤: 字数≤{self._min_text_length}，"
            f"白名单: {self._whitelist_keywords})"
        )

    # =========================
    # 数据库初始化
    # =========================

    def _init_db(self) -> None:
        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS group_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    group_id TEXT NOT NULL,
                    message_id TEXT,
                    sender_id TEXT NOT NULL,
                    sender_name TEXT,
                    text TEXT NOT NULL,
                    timestamp REAL NOT NULL
                )
            """)
            # 按群 + 用户查询的索引
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_group_sender
                ON group_messages(group_id, sender_id, id DESC)
            """)
            # 按群裁剪的索引
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_group_id
                ON group_messages(group_id, id DESC)
            """)
            # 同群同消息 ID 唯一（去重）
            conn.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_group_msgid
                ON group_messages(group_id, message_id)
                WHERE message_id != ''
            """)
            conn.commit()
        finally:
            conn.close()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=5)
        conn.row_factory = sqlite3.Row
        return conn

    # =========================
    # 被动捕获（由主插件钩子调用）
    # =========================

    async def get_cache_stats(self) -> dict[str, dict]:
        """返回每个群的缓存统计。

        Returns:
            {group_id: {"total": N, "users": M}, ...}
        """
        conn = self._connect()
        try:
            cur = conn.execute("""
                SELECT group_id,
                       COUNT(*) AS total,
                       COUNT(DISTINCT sender_id) AS users
                FROM group_messages
                GROUP BY group_id
            """)
            return {
                row["group_id"]: {
                    "total": row["total"],
                    "users": row["users"],
                }
                for row in cur.fetchall()
            }
        finally:
            conn.close()

    async def get_user_cache_stats(
        self, event: AiocqhttpMessageEvent, target_id: str
    ) -> dict:
        """查询单个用户在本群及全部群的缓存统计。

        Returns:
            {
                "group_id": str,
                "user_id": str,
                # 本群
                "total_in_group": int,
                "earliest_ts": float,
                "latest_ts": float,
                # 全部群
                "total_all": int,
                "group_count": int,
                # 配置
                "max_user_size": int,
            }
        """
        group_id = str(event.get_group_id())
        target_id = str(target_id)
        conn = self._connect()
        try:
            # 本群统计
            row = conn.execute(
                """
                SELECT COUNT(*)       AS total,
                       MIN(timestamp) AS earliest_ts,
                       MAX(timestamp) AS latest_ts
                FROM group_messages
                WHERE group_id = ? AND sender_id = ?
                """,
                (group_id, target_id),
            ).fetchone()

            # 全部群统计
            row_all = conn.execute(
                """
                SELECT COUNT(*)             AS total,
                       COUNT(DISTINCT group_id) AS group_count
                FROM group_messages
                WHERE sender_id = ?
                """,
                (target_id,),
            ).fetchone()
        finally:
            conn.close()

        return {
            "group_id": group_id,
            "user_id": target_id,
            "total_in_group": row["total"] or 0,
            "earliest_ts": row["earliest_ts"] or 0.0,
            "latest_ts": row["latest_ts"] or 0.0,
            "total_all": row_all["total"] or 0,
            "group_count": row_all["group_count"] or 0,
            "max_user_size": self._max_user_size,
        }

    async def record_incoming_message(self, event: AiocqhttpMessageEvent) -> bool:
        """被动记录一条群消息。

        Args:
            event: AstrBot 群消息事件。

        Returns:
            是否成功入库。
        """
        group_id = str(event.get_group_id() or "").strip()
        sender_id = str(event.get_sender_id() or "").strip()
        if not group_id or not sender_id:
            return False

        text = self._extract_text(event)
        if not self._should_record(text):
            return False

        sender_name = event.get_sender_name() or sender_id
        message_id = str(getattr(event.message_obj, "message_id", "") or "")
        now = time.time()

        lock = self._group_locks.setdefault(group_id, asyncio.Lock())
        async with lock:
            try:
                conn = self._connect()
                try:
                    cursor = conn.execute(
                        """
                        INSERT OR IGNORE INTO group_messages
                        (group_id, message_id, sender_id, sender_name, text, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (group_id, message_id, sender_id, sender_name, text, now),
                    )
                    inserted = cursor.rowcount > 0

                    if inserted:
                        self._maybe_trim_conn(conn, group_id, sender_id)

                    conn.commit()
                finally:
                    conn.close()
                return True
            except Exception as e:
                logger.debug(f"[MessageManager] 写入失败: {e}")
                return False

    async def record_outgoing_message(
        self, event: AiocqhttpMessageEvent, user_id: str, nickname: str, text: str
    ) -> bool:
        """记录机器人自己发出的消息"""
        if not getattr(self.cfg, "record_bot_reply", False):
            return False
        group_id = str(event.get_group_id() or "").strip()
        if not group_id:
            return False
        text = (text or "").strip()
        return await self._record_bot_text(group_id, user_id, nickname, text)

    async def record_platform_outgoing(
        self,
        event: AiocqhttpMessageEvent,
        message_chain: list,
        bot_id: str,
        bot_name: str,
    ) -> bool:
        """记录 bot 通过平台发送的消息（含其他插件的主动消息）。

        只记录含 Plain 文本段的消息，纯图片/纯At/纯表情一律丢弃，
        与 record_incoming_message 的处理保持一致。
        """
        if not getattr(self.cfg, "record_bot_reply", False):
            return False
        group_id = str(event.get_group_id() or "").strip()
        if not group_id:
            return False
        # 只取 Plain 文本段
        text = "".join(
            seg.text
            for seg in message_chain
            if isinstance(seg, Plain) and getattr(seg, "text", "")
        ).strip()
        return await self._record_bot_text(group_id, bot_id, bot_name, text)

    async def _record_bot_text(
        self, group_id: str, sender_id: str, nickname: str, text: str
    ) -> bool:
        """bot 文本入库（公共路径）：5 秒去重 + 插入 + 裁剪。"""
        if not text or not self._should_record(text, is_bot=True):
            return False

        now = time.time()
        lock = self._group_locks.setdefault(group_id, asyncio.Lock())
        async with lock:
            try:
                conn = self._connect()
                try:
                    dup = conn.execute(
                        """
                        SELECT 1 FROM group_messages
                        WHERE group_id = ? AND sender_id = ? AND text = ?
                          AND timestamp > ?
                        LIMIT 1
                        """,
                        (group_id, sender_id, text, now - 5),
                    ).fetchone()
                    if dup:
                        return False

                    conn.execute(
                        """
                        INSERT INTO group_messages
                        (group_id, message_id, sender_id, sender_name, text, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (group_id, "", sender_id, nickname, text, now),
                    )
                    self._maybe_trim_conn(conn, group_id, sender_id)
                    conn.commit()
                    return True
                finally:
                    conn.close()
            except Exception as e:
                logger.error(f"[MessageManager] 记录机器人消息失败: {e}")
                return False

    def _maybe_trim_conn(
        self, conn: sqlite3.Connection, group_id: str, sender_id: str
    ) -> None:
        """入库后的裁剪检查：先按人裁，再按群裁。"""
        user_count = conn.execute(
            "SELECT COUNT(*) FROM group_messages "
            "WHERE group_id = ? AND sender_id = ?",
            (group_id, sender_id),
        ).fetchone()[0]
        if user_count >= self._user_trigger:
            self._trim_user_conn(conn, group_id, sender_id)

        group_count = conn.execute(
            "SELECT COUNT(*) FROM group_messages WHERE group_id = ?",
            (group_id,),
        ).fetchone()[0]
        if group_count >= self._group_trigger:
            self._trim_group_only_conn(conn, group_id)

    def _trim_user_conn(
        self, conn: sqlite3.Connection, group_id: str, sender_id: str
    ) -> None:
        """按发言顺序裁掉指定用户在该群中超出的部分。"""
        conn.execute(
            """
            DELETE FROM group_messages
            WHERE group_id = ? AND sender_id = ?
              AND id NOT IN (
                  SELECT id FROM group_messages
                  WHERE group_id = ? AND sender_id = ?
                  ORDER BY id DESC
                  LIMIT ?
              )
            """,
            (
                group_id,
                sender_id,
                group_id,
                sender_id,
                self._max_user_size,
            ),
        )

    def _trim_group_only_conn(
        self, conn: sqlite3.Connection, group_id: str
    ) -> None:
        """按发言顺序裁掉整群超出的部分。"""
        conn.execute(
            """
            DELETE FROM group_messages
            WHERE group_id = ?
              AND id NOT IN (
                  SELECT id FROM group_messages
                  WHERE group_id = ?
                  ORDER BY id DESC
                  LIMIT ?
              )
            """,
            (group_id, group_id, self._max_group_size),
        )

    # =========================
    # 查询（只读缓存，不再爬取）
    # =========================

    async def get_user_texts(
        self,
        event: AiocqhttpMessageEvent,
        target_id: str,
        *,
        limit: int | None = None,
    ) -> MessageQueryResult:
        """从本地缓存读取目标用户的聊天记录。

        不再调用协议端 API，只查本地 SQLite。

        Args:
            event: 当前群消息事件。
            target_id: 目标用户 ID。
            limit: 本次查询返回的最大条数；不传则用配置值。

        Returns:
            查询结果。
        """
        group_id = str(event.get_group_id())
        target_id = str(target_id)
        query_limit = limit if limit and limit > 0 else self.cfg.max_msg_count

        conn = self._connect()
        try:
            cur = conn.execute(
                """
                SELECT text FROM group_messages
                WHERE group_id = ? AND sender_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (group_id, target_id, query_limit),
            )
            rows = cur.fetchall()
        finally:
            conn.close()

        # 查询是 DESC，反转为时间正序
        texts = [row["text"] for row in rows][::-1]

        return MessageQueryResult(
            texts=texts,
            scanned_messages=len(texts),
            from_cache=True,
        )

    # =========================
    # 兼容旧接口
    # =========================

    def clear_cache(self) -> None:
        """清空所有缓存（谨慎使用）。"""
        conn = self._connect()
        try:
            conn.execute("DELETE FROM group_messages")
            conn.commit()
        finally:
            conn.close()

    def save_cache(self) -> None:
        """兼容旧接口；SQLite 已实时落盘，无需手动保存。"""
        pass
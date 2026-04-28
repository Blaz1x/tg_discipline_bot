import asyncio
import html
import io
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from urllib.parse import urlsplit, urlunsplit
from zoneinfo import ZoneInfo, available_timezones

from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart, Command
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import (
    InlineKeyboardMarkup,
    Message,
    CallbackQuery,
    BufferedInputFile,
)
from aiogram.utils.keyboard import InlineKeyboardBuilder
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from dotenv import load_dotenv
from sqlalchemy import (
    BigInteger,
    Boolean,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    select,
    update,
)
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =========================================================
# ENV / CONFIG
# =========================================================
load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN", "")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///tracker.db")
DEFAULT_TIMEZONE = os.getenv("DEFAULT_TIMEZONE", "Europe/Moscow")
DEFAULT_START_HOUR = int(os.getenv("DEFAULT_START_HOUR", "9"))
DEFAULT_END_HOUR = int(os.getenv("DEFAULT_END_HOUR", "21"))
REPORT_HOUR = int(os.getenv("REPORT_HOUR", "23"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_TIMEOUT_SECONDS = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "45"))
OPENAI_MAX_OUTPUT_TOKENS = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "1400"))
OPENAI_PROXIES = [
    value
    for value in (
        os.getenv("HTTP_PROXY"),
        os.getenv("HTTP_PROXY_2"),
        os.getenv("HTTP_PROXY_3"),
    )
    if value
]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =========================================================
# DB MODELS
# =========================================================
class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tg_user_id: Mapped[int] = mapped_column(BigInteger, unique=True, index=True)
    chat_id: Mapped[int] = mapped_column(BigInteger, index=True)
    username: Mapped[str | None] = mapped_column(String(255), nullable=True)
    full_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    timezone: Mapped[str] = mapped_column(String(64), default=DEFAULT_TIMEZONE)
    start_hour: Mapped[int] = mapped_column(Integer, default=DEFAULT_START_HOUR)
    end_hour: Mapped[int] = mapped_column(Integer, default=DEFAULT_END_HOUR)
    report_hour: Mapped[int] = mapped_column(Integer, default=REPORT_HOUR)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    entries: Mapped[list["TimeEntry"]] = relationship(back_populates="user")
    reports: Mapped[list["DailyReport"]] = relationship(back_populates="user")
    memory: Mapped["UserMemory | None"] = relationship(back_populates="user", uselist=False)


class TimeEntry(Base):
    __tablename__ = "time_entries"
    __table_args__ = (UniqueConstraint("user_id", "entry_date", "hour_slot", name="uq_user_day_hour"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    entry_date: Mapped[date] = mapped_column(Date, index=True)
    hour_slot: Mapped[int] = mapped_column(Integer, index=True)
    status: Mapped[str] = mapped_column(String(32), default="rest")
    label: Mapped[str] = mapped_column(String(255), default="😌 Отдыхал")
    is_auto_filled: Mapped[bool] = mapped_column(Boolean, default=False)
    source_message_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    answered_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    user: Mapped["User"] = relationship(back_populates="entries")


class DailyReport(Base):
    __tablename__ = "daily_reports"
    __table_args__ = (UniqueConstraint("user_id", "report_date", name="uq_user_report_day"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    report_date: Mapped[date] = mapped_column(Date, index=True)
    summary_text: Mapped[str] = mapped_column(Text)
    reflection: Mapped[str | None] = mapped_column(Text, nullable=True)
    reflection_requested: Mapped[bool] = mapped_column(Boolean, default=False)
    reflection_received_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    user: Mapped["User"] = relationship(back_populates="reports")


class UserMemory(Base):
    __tablename__ = "user_memories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), unique=True, index=True)
    profile_text: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    user: Mapped["User"] = relationship(back_populates="memory")


class DailyAIAssessment(Base):
    __tablename__ = "daily_ai_assessments"
    __table_args__ = (UniqueConstraint("user_id", "report_date", name="uq_user_ai_assessment_day"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    report_date: Mapped[date] = mapped_column(Date, index=True)
    assessment_text: Mapped[str] = mapped_column(Text)
    memory_snapshot: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


# =========================================================
# DB INIT
# =========================================================
engine = create_async_engine(DATABASE_URL, echo=False)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


# =========================================================
# FSM
# =========================================================
class ReflectionFlow(StatesGroup):
    waiting_reflection = State()


# =========================================================
# DOMAIN
# =========================================================
@dataclass(frozen=True)
class ActivityOption:
    code: str
    text: str


OPTIONS = {
    "income": ActivityOption("income", "💸 Заработок"),
    "work": ActivityOption("work", "💻 Работаю"),
    "grow": ActivityOption("grow", "📚 Развиваюсь"),
    "rest": ActivityOption("rest", "😌 Отдыхал"),
    "other": ActivityOption("other", "📝 Другое"),
}

STATUS_ORDER = ["income", "work", "grow", "rest", "other"]
PRODUCTIVE_STATUSES = ("income", "work", "grow")
TELEGRAM_PHOTO_CAPTION_LIMIT = 1024
TELEGRAM_MESSAGE_LIMIT = 3900
OLD_REST_LABELS = ("Отдыхаю 😌", "😌 Отдыхаю", "Отдыхаю")
RETRYABLE_OPENAI_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}

REPORT_LABELS = {
    "income": "💸 Заработок",
    "work": "💻 Работал",
    "grow": "📚 Развивался",
    "rest": "😌 Отдыхал",
    "other": "📝 Другое",
}

CHART_LABELS = {
    "income": "Заработок",
    "work": "Работа",
    "grow": "Развитие",
    "rest": "Отдыхал",
    "other": "Другое",
}

CHART_COLORS = {
    "income": "#2f9e44",
    "work": "#1971c2",
    "grow": "#7048e8",
    "rest": "#f59f00",
    "other": "#868e96",
}


@dataclass
class ReportStats:
    start_date: date
    end_date: date
    days_count: int
    slots_per_day: int
    total_slots: int
    counts: dict[str, int]
    auto_rest: int
    daily_productive: dict[date, int]
    daily_tracked: dict[date, int]


# =========================================================
# BOT / DISPATCHER / SCHEDULER
# =========================================================
storage = MemoryStorage()
bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher(storage=storage)
scheduler = AsyncIOScheduler(timezone=ZoneInfo("UTC"))


# =========================================================
# HELPERS
# =========================================================
def get_zone(tz_name: str) -> ZoneInfo:
    try:
        return ZoneInfo(tz_name)
    except Exception:
        return ZoneInfo(DEFAULT_TIMEZONE)


def now_in_tz(tz_name: str) -> datetime:
    return datetime.now(get_zone(tz_name))


def today_in_tz(tz_name: str) -> date:
    return now_in_tz(tz_name).date()


def user_hours(user: User) -> list[int]:
    return list(range(user.start_hour, user.end_hour + 1))


def activity_keyboard(entry_date: date, hour_slot: int, selected: str | None = None) -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    for code in STATUS_ORDER:
        option = OPTIONS[code]
        text = option.text if selected != code else f"✅ {option.text}"
        builder.button(text=text, callback_data=f"track:{entry_date.isoformat()}:{hour_slot}:{code}")
    builder.adjust(2, 2, 1)
    return builder.as_markup()


def settings_keyboard() -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    builder.button(text="🕘 Изменить часы", callback_data="settings:hours")
    builder.button(text="🌍 Изменить timezone", callback_data="settings:tz")
    builder.adjust(1)
    return builder.as_markup()


def mask_proxy(proxy: str | None) -> str:
    if not proxy:
        return "direct"

    parts = urlsplit(proxy)
    if "@" not in parts.netloc:
        return proxy

    _, _, host = parts.netloc.rpartition("@")
    masked_netloc = f"***:***@{host}"
    return urlunsplit((parts.scheme, masked_netloc, parts.path, parts.query, parts.fragment))


class OpenAIClientPool:
    def __init__(self) -> None:
        self.proxies = OPENAI_PROXIES or [None]
        self.active_index = 0
        self.client = None
        self.sdk: dict[str, object] | None = None

    def load_sdk(self) -> dict[str, object]:
        if self.sdk is not None:
            return self.sdk

        try:
            import httpx
            from openai import (
                APIConnectionError,
                APIStatusError,
                APITimeoutError,
                AsyncOpenAI,
                DefaultAsyncHttpxClient,
                InternalServerError,
                RateLimitError,
            )
        except ModuleNotFoundError as error:
            raise RuntimeError("OpenAI SDK не установлен. Обнови зависимости через pip install -r requirements.txt.") from error

        self.sdk = {
            "httpx": httpx,
            "APIConnectionError": APIConnectionError,
            "APIStatusError": APIStatusError,
            "APITimeoutError": APITimeoutError,
            "AsyncOpenAI": AsyncOpenAI,
            "DefaultAsyncHttpxClient": DefaultAsyncHttpxClient,
            "InternalServerError": InternalServerError,
            "RateLimitError": RateLimitError,
        }
        return self.sdk

    async def close(self) -> None:
        if self.client is not None:
            await self.client.close()
            self.client = None

    async def switch_to_proxy(self, index: int) -> None:
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY не настроен в .env")

        await self.close()
        sdk = self.load_sdk()
        self.active_index = index
        proxy = self.proxies[index]
        httpx = sdk["httpx"]
        default_client = sdk["DefaultAsyncHttpxClient"]
        async_openai = sdk["AsyncOpenAI"]

        http_client = default_client(
            proxy=proxy if proxy else None,
            timeout=httpx.Timeout(OPENAI_TIMEOUT_SECONDS),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )
        self.client = async_openai(api_key=OPENAI_API_KEY, http_client=http_client)

    async def ensure_client(self) -> None:
        if self.client is None:
            await self.switch_to_proxy(self.active_index)

    def is_retryable_error(self, error: Exception) -> bool:
        sdk = self.load_sdk()
        httpx = sdk["httpx"]
        retryable_classes = (
            httpx.ConnectError,
            httpx.ProxyError,
            httpx.ConnectTimeout,
            httpx.ReadTimeout,
            httpx.WriteTimeout,
            httpx.RemoteProtocolError,
            httpx.NetworkError,
            sdk["APIConnectionError"],
            sdk["APITimeoutError"],
            sdk["InternalServerError"],
            sdk["RateLimitError"],
        )

        if isinstance(error, retryable_classes):
            return True

        api_status_error = sdk["APIStatusError"]
        if isinstance(error, api_status_error):
            return error.status_code in RETRYABLE_OPENAI_STATUS_CODES

        return False

    async def create_text(
        self,
        input_messages: list[dict],
        max_output_tokens: int = OPENAI_MAX_OUTPUT_TOKENS,
    ) -> str:
        await self.ensure_client()
        errors: list[tuple[str | None, Exception]] = []
        start_index = self.active_index

        for offset in range(len(self.proxies)):
            index = (start_index + offset) % len(self.proxies)
            if self.client is None or index != self.active_index:
                await self.switch_to_proxy(index)

            proxy = self.proxies[index]
            try:
                response = await self.client.responses.create(
                    model=OPENAI_MODEL,
                    temperature=0.2,
                    max_output_tokens=max_output_tokens,
                    input=input_messages,
                )
                return (response.output_text or "").strip()
            except Exception as error:
                errors.append((proxy, error))
                if not self.is_retryable_error(error):
                    raise

                logger.warning(
                    "OpenAI request failed through %s: %s: %s",
                    mask_proxy(proxy),
                    type(error).__name__,
                    error,
                )

        parts = [
            f"{mask_proxy(proxy)} -> {type(error).__name__}: {error}"
            for proxy, error in errors
        ]
        raise RuntimeError("Все прокси OpenAI завершились ошибкой: " + " | ".join(parts))


ai_client_pool = OpenAIClientPool()


async def get_or_create_user(session: AsyncSession, tg_user_id: int, chat_id: int, username: str | None, full_name: str | None) -> User:
    result = await session.execute(select(User).where(User.tg_user_id == tg_user_id))
    user = result.scalar_one_or_none()
    if user:
        user.chat_id = chat_id
        user.username = username
        user.full_name = full_name
        user.is_active = True
        await session.commit()
        return user

    user = User(
        tg_user_id=tg_user_id,
        chat_id=chat_id,
        username=username,
        full_name=full_name,
    )
    session.add(user)
    await session.commit()
    await session.refresh(user)
    return user


async def find_user_by_tg(session: AsyncSession, tg_user_id: int) -> User | None:
    result = await session.execute(select(User).where(User.tg_user_id == tg_user_id))
    return result.scalar_one_or_none()


async def get_active_users(session: AsyncSession) -> list[User]:
    result = await session.execute(select(User).where(User.is_active == True))
    return list(result.scalars().all())


async def migrate_legacy_activity_labels() -> None:
    async with SessionLocal() as session:
        await session.execute(
            update(TimeEntry)
            .where(TimeEntry.status == "rest", TimeEntry.label.in_(OLD_REST_LABELS))
            .values(label=OPTIONS["rest"].text)
        )
        await session.commit()


async def upsert_time_entry(
    session: AsyncSession,
    user_id: int,
    entry_date: date,
    hour_slot: int,
    status: str,
    label: str,
    source_message_id: int | None = None,
    is_auto_filled: bool = False,
    answered_at: datetime | None = None,
) -> TimeEntry:
    result = await session.execute(
        select(TimeEntry).where(
            TimeEntry.user_id == user_id,
            TimeEntry.entry_date == entry_date,
            TimeEntry.hour_slot == hour_slot,
        )
    )
    entry = result.scalar_one_or_none()
    if entry:
        entry.status = status
        entry.label = label
        entry.source_message_id = source_message_id or entry.source_message_id
        entry.is_auto_filled = is_auto_filled
        entry.answered_at = answered_at
    else:
        entry = TimeEntry(
            user_id=user_id,
            entry_date=entry_date,
            hour_slot=hour_slot,
            status=status,
            label=label,
            source_message_id=source_message_id,
            is_auto_filled=is_auto_filled,
            answered_at=answered_at,
        )
        session.add(entry)
    await session.commit()
    await session.refresh(entry)
    return entry


async def get_day_entries(session: AsyncSession, user_id: int, entry_date: date) -> list[TimeEntry]:
    result = await session.execute(
        select(TimeEntry)
        .where(TimeEntry.user_id == user_id, TimeEntry.entry_date == entry_date)
        .order_by(TimeEntry.hour_slot.asc())
    )
    return list(result.scalars().all())


async def autofill_missing_as_rest(session: AsyncSession, user: User, entry_date: date) -> None:
    existing = await get_day_entries(session, user.id, entry_date)
    existing_hours = {e.hour_slot for e in existing}
    for hour in user_hours(user):
        if hour not in existing_hours:
            await upsert_time_entry(
                session=session,
                user_id=user.id,
                entry_date=entry_date,
                hour_slot=hour,
                status="rest",
                label=OPTIONS["rest"].text,
                is_auto_filled=True,
            )

def get_previous_hour_slot(now_dt: datetime) -> tuple[date, int]:
    previous_hour_dt = now_dt - timedelta(hours=1)
    return previous_hour_dt.date(), previous_hour_dt.hour

async def get_or_create_daily_report(session: AsyncSession, user_id: int, report_date: date, summary_text: str = "") -> DailyReport:
    result = await session.execute(
        select(DailyReport).where(DailyReport.user_id == user_id, DailyReport.report_date == report_date)
    )
    report = result.scalar_one_or_none()
    if report:
        return report
    report = DailyReport(user_id=user_id, report_date=report_date, summary_text=summary_text)
    session.add(report)
    await session.commit()
    await session.refresh(report)
    return report


def build_default_memory(user: User) -> str:
    facts = [
        "Пользователь ведет учет времени через Telegram-бота.",
        f"Timezone: {user.timezone}.",
        f"Окно трекинга: {user.start_hour:02d}:00-{user.end_hour:02d}:59.",
    ]
    if user.full_name:
        facts.append(f"Имя в Telegram: {user.full_name}.")
    if user.username:
        facts.append(f"Username в Telegram: @{user.username}.")
    return "\n".join(facts)


async def get_or_create_user_memory(session: AsyncSession, user: User) -> UserMemory:
    result = await session.execute(select(UserMemory).where(UserMemory.user_id == user.id))
    memory = result.scalar_one_or_none()
    if memory:
        return memory

    memory = UserMemory(user_id=user.id, profile_text=build_default_memory(user))
    session.add(memory)
    await session.commit()
    await session.refresh(memory)
    return memory


async def upsert_daily_ai_assessment(
    session: AsyncSession,
    user: User,
    report_date: date,
    assessment_text: str,
    memory_snapshot: str,
) -> DailyAIAssessment:
    result = await session.execute(
        select(DailyAIAssessment).where(
            DailyAIAssessment.user_id == user.id,
            DailyAIAssessment.report_date == report_date,
        )
    )
    assessment = result.scalar_one_or_none()
    if assessment:
        assessment.assessment_text = assessment_text
        assessment.memory_snapshot = memory_snapshot
        assessment.updated_at = datetime.utcnow()
    else:
        assessment = DailyAIAssessment(
            user_id=user.id,
            report_date=report_date,
            assessment_text=assessment_text,
            memory_snapshot=memory_snapshot,
        )
        session.add(assessment)
    await session.commit()
    await session.refresh(assessment)
    return assessment


def report_label(code: str) -> str:
    option = OPTIONS.get(code)
    return REPORT_LABELS.get(code, option.text if option else code)


def chart_label(code: str) -> str:
    return CHART_LABELS.get(code, code)


def tracked_hours(counts: dict[str, int]) -> int:
    return sum(counts.values())


def productive_hours(counts: dict[str, int]) -> int:
    return sum(counts.get(code, 0) for code in PRODUCTIVE_STATUSES)


def format_percent(part: int, total: int) -> str:
    if total <= 0:
        return "0%"
    return f"{(part / total) * 100:.0f}%"


def format_average(hours: int, days_count: int) -> str:
    if days_count <= 0:
        return "0.0 ч/день"
    return f"{hours / days_count:.1f} ч/день"


def bar(hours: int, total: int, width: int = 10) -> str:
    filled = round((hours / total) * width) if total else 0
    filled = max(0, min(width, filled))
    return "█" * filled + "░" * (width - filled)


def top_category(counts: dict[str, int]) -> tuple[str | None, int]:
    if tracked_hours(counts) == 0:
        return None, 0
    code = max(STATUS_ORDER, key=lambda item: counts.get(item, 0))
    return code, counts.get(code, 0)


def best_productive_day(stats: ReportStats) -> tuple[date, int] | None:
    if not stats.daily_productive:
        return None
    best_day = max(stats.daily_productive, key=lambda item: stats.daily_productive[item])
    best_hours = stats.daily_productive[best_day]
    if best_hours <= 0:
        return None
    return best_day, best_hours


def build_report_signal(stats: ReportStats) -> str:
    total = tracked_hours(stats.counts)
    if total == 0:
        return "пока нет отмеченных часов, отчету не хватает данных."

    productive = productive_hours(stats.counts)
    rest = stats.counts.get("rest", 0)
    other = stats.counts.get("other", 0)
    missing = max(stats.total_slots - total, 0)

    if missing > stats.total_slots * 0.35:
        return "много пустых слотов, поэтому сначала стоит добить регулярность отметок."
    if other > total * 0.3:
        return "слишком много «другого», категории стоит уточнять точнее."
    if productive >= total * 0.6:
        return "фокус сильный, большая часть времени ушла в созидательные категории."
    if rest >= total * 0.5:
        return "период больше про восстановление, проверь, был ли это осознанный отдых."
    return "баланс ровный, следующий рычаг роста - поднять долю заработка или развития."


async def build_report_stats(session: AsyncSession, user: User, start_date: date, end_date: date) -> ReportStats:
    counts = {code: 0 for code in STATUS_ORDER}
    auto_rest = 0
    daily_productive: dict[date, int] = {}
    daily_tracked: dict[date, int] = {}
    slots_per_day = len(user_hours(user))

    current = start_date
    while current <= end_date:
        entries = await get_day_entries(session, user.id, current)
        day_productive = 0
        day_tracked = 0

        for entry in entries:
            counts[entry.status] = counts.get(entry.status, 0) + 1
            day_tracked += 1
            if entry.status in PRODUCTIVE_STATUSES:
                day_productive += 1
            if entry.is_auto_filled and entry.status == "rest":
                auto_rest += 1

        daily_productive[current] = day_productive
        daily_tracked[current] = day_tracked
        current += timedelta(days=1)

    days_count = (end_date - start_date).days + 1
    return ReportStats(
        start_date=start_date,
        end_date=end_date,
        days_count=days_count,
        slots_per_day=slots_per_day,
        total_slots=slots_per_day * days_count,
        counts=counts,
        auto_rest=auto_rest,
        daily_productive=daily_productive,
        daily_tracked=daily_tracked,
    )


async def build_counts_for_range(session: AsyncSession, user: User, start_date: date, end_date: date) -> dict[str, int]:
    stats = await build_report_stats(session, user, start_date, end_date)
    return stats.counts


def build_day_report_text_from_stats(user: User, stats: ReportStats, include_reflection_prompt: bool = True) -> str:
    counts = stats.counts
    total = tracked_hours(counts)
    productive = productive_hours(counts)
    rest = counts.get("rest", 0)
    manual = max(total - stats.auto_rest, 0)
    top_code, top_hours = top_category(counts)
    top_line = (
        f"Главная категория: <b>{report_label(top_code)}</b> ({top_hours} ч)"
        if top_code
        else "Главная категория: <b>нет данных</b>"
    )

    lines = [
        f"<b>Итог за {stats.start_date.strftime('%d.%m.%Y')}</b>",
        f"Окно: <b>{user.start_hour:02d}:00–{user.end_hour:02d}:59</b>, timezone: <b>{user.timezone}</b>",
        f"Покрытие: <b>{total}/{stats.total_slots} ч</b> ({format_percent(total, stats.total_slots)}), вручную: <b>{manual} ч</b>",
        f"Автозаполнено как «отдыхал»: <b>{stats.auto_rest} ч</b>",
        "",
        "<b>Ключевые числа</b>",
        f"Продуктивно: <b>{productive} ч</b> ({format_percent(productive, total)})",
        f"Заработок: <b>{counts.get('income', 0)} ч</b>, развитие: <b>{counts.get('grow', 0)} ч</b>, отдыхал: <b>{rest} ч</b>",
        top_line,
        "",
        "<b>Распределение</b>",
    ]

    for code in STATUS_ORDER:
        hours = counts[code]
        lines.append(f"{report_label(code)}: <b>{hours} ч</b> ({format_percent(hours, total)}) {bar(hours, total)}")

    lines.extend(
        [
            "",
            f"<b>Сигнал</b>: {build_report_signal(stats)}",
        ]
    )

    if include_reflection_prompt:
        lines.extend(
            [
                "",
                "<b>Рефлексия</b>: что сделал, что дало результат, что завтра улучшить?",
            ]
        )

    return "\n".join(lines)


async def build_day_report_text(
    session: AsyncSession,
    user: User,
    entry_date: date,
    include_reflection_prompt: bool = True,
) -> str:
    stats = await build_report_stats(session, user, entry_date, entry_date)
    return build_day_report_text_from_stats(user, stats, include_reflection_prompt)


def build_period_report_text(stats: ReportStats, title: str) -> str:
    counts = stats.counts
    total = tracked_hours(counts)
    productive = productive_hours(counts)
    rest = counts.get("rest", 0)
    missing = max(stats.total_slots - total, 0)
    top_code, top_hours = top_category(counts)
    best_day = best_productive_day(stats)

    lines = [
        f"<b>{title}</b> ({stats.start_date.strftime('%d.%m')}–{stats.end_date.strftime('%d.%m')})",
        f"Покрытие: <b>{total}/{stats.total_slots} ч</b> ({format_percent(total, stats.total_slots)}), пусто: <b>{missing} ч</b>",
        f"Продуктивно: <b>{productive} ч</b> ({format_percent(productive, total)}), отдыхал: <b>{rest} ч</b>",
        f"Средний день: <b>{format_average(total, stats.days_count)}</b> отмечено, <b>{format_average(productive, stats.days_count)}</b> продуктивно",
    ]

    if top_code:
        lines.append(f"Главная категория: <b>{report_label(top_code)}</b> ({top_hours} ч)")
    if best_day:
        lines.append(f"Лучший день по продуктивным часам: <b>{best_day[0].strftime('%d.%m')}</b> - <b>{best_day[1]} ч</b>")

    lines.extend(["", "<b>Распределение</b>"])
    for code in STATUS_ORDER:
        hours = counts[code]
        lines.append(f"{report_label(code)}: <b>{hours} ч</b> ({format_percent(hours, total)}) {bar(hours, total)}")

    lines.extend(["", f"<b>Сигнал</b>: {build_report_signal(stats)}"])
    return "\n".join(lines)


def build_chart_png(title: str, counts: dict[str, int]) -> bytes:
    labels = [chart_label(code) for code in STATUS_ORDER]
    values = [counts.get(code, 0) for code in STATUS_ORDER]
    colors = [CHART_COLORS.get(code, "#868e96") for code in STATUS_ORDER]
    total = tracked_hours(counts)
    productive = productive_hours(counts)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(labels, values, color=colors)
    ax.invert_yaxis()
    ax.set_title(f"{title}\nВсего {total} ч, продуктивно {productive} ч", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Часы")
    ax.grid(axis="x", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    max_value = max(values) if values else 0
    ax.set_xlim(0, max(max_value * 1.25, 1))
    label_offset = max(max_value * 0.03, 0.1)

    for bar_item, value in zip(bars, values):
        ax.text(
            value + label_offset,
            bar_item.get_y() + bar_item.get_height() / 2,
            f"{value} ч | {format_percent(value, total)}",
            va="center",
            fontsize=10,
        )

    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=160)
    plt.close(fig)
    buffer.seek(0)
    return buffer.read()


async def answer_report_photo(message: Message, img: bytes, filename: str, text: str) -> None:
    photo = BufferedInputFile(img, filename=filename)
    if len(text) <= TELEGRAM_PHOTO_CAPTION_LIMIT:
        await message.answer_photo(photo, caption=text)
        return

    await message.answer_photo(photo)
    await message.answer(text)


async def send_report_photo(chat_id: int, img: bytes, filename: str, text: str) -> None:
    photo = BufferedInputFile(img, filename=filename)
    if len(text) <= TELEGRAM_PHOTO_CAPTION_LIMIT:
        await bot.send_photo(chat_id=chat_id, photo=photo, caption=text)
        return

    await bot.send_photo(chat_id=chat_id, photo=photo)
    await bot.send_message(chat_id=chat_id, text=text)


def split_text(text: str, limit: int = TELEGRAM_MESSAGE_LIMIT) -> list[str]:
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    current = ""
    for paragraph in text.split("\n"):
        candidate = f"{current}\n{paragraph}" if current else paragraph
        if len(candidate) <= limit:
            current = candidate
            continue

        if current:
            chunks.append(current)
            current = ""

        while len(paragraph) > limit:
            chunks.append(paragraph[:limit])
            paragraph = paragraph[limit:]
        current = paragraph

    if current:
        chunks.append(current)
    return chunks


async def answer_escaped_text(message: Message, title: str, text: str) -> None:
    for index, chunk in enumerate(split_text(text)):
        prefix = f"<b>{title}</b>\n\n" if index == 0 else ""
        await message.answer(prefix + html.escape(chunk))


async def send_escaped_text(chat_id: int, title: str, text: str) -> None:
    for index, chunk in enumerate(split_text(text)):
        prefix = f"<b>{title}</b>\n\n" if index == 0 else ""
        await bot.send_message(chat_id=chat_id, text=prefix + html.escape(chunk))


def cleanup_ai_json(raw_text: str) -> str:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 3:
            cleaned = "\n".join(lines[1:-1]).strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return cleaned[start : end + 1]
    return cleaned


def parse_ai_daily_payload(raw_text: str, current_memory: str) -> tuple[str, str]:
    try:
        payload = json.loads(cleanup_ai_json(raw_text))
    except json.JSONDecodeError:
        return raw_text.strip(), current_memory

    assessment = payload.get("assessment")
    memory = payload.get("memory")
    if not isinstance(assessment, str) or not assessment.strip():
        assessment = raw_text.strip()
    if not isinstance(memory, str) or not memory.strip():
        memory = current_memory
    return assessment.strip(), memory.strip()


def build_counts_context(stats: ReportStats) -> str:
    total = tracked_hours(stats.counts)
    lines = [
        f"Период: {stats.start_date.isoformat()} - {stats.end_date.isoformat()}",
        f"Покрытие: {total}/{stats.total_slots} ч ({format_percent(total, stats.total_slots)})",
        f"Продуктивно: {productive_hours(stats.counts)} ч ({format_percent(productive_hours(stats.counts), total)})",
        f"Автозаполнено как отдыхал: {stats.auto_rest} ч",
    ]
    for code in STATUS_ORDER:
        hours = stats.counts.get(code, 0)
        lines.append(f"{report_label(code)}: {hours} ч ({format_percent(hours, total)})")
    return "\n".join(lines)


def build_entries_context(entries: list[TimeEntry]) -> str:
    if not entries:
        return "Нет отмеченных часов."

    lines = []
    for entry in entries:
        source = "авто" if entry.is_auto_filled else "вручную"
        lines.append(
            f"{entry.hour_slot:02d}:00-{entry.hour_slot:02d}:59: "
            f"{report_label(entry.status)} ({source})"
        )
    return "\n".join(lines)


def build_ai_daily_messages(
    user: User,
    stats: ReportStats,
    entries: list[TimeEntry],
    memory_text: str,
    reflection: str,
) -> list[dict]:
    system_prompt = (
        "Ты личный аналитик времени. Твоя задача - холодно оценивать день по фактам, "
        "а не поддерживать пользователя. Пиши по-русски, сухо, честно и конкретно. "
        "Не оскорбляй, не морализируй, не ставь медицинские диагнозы. "
        "Автозаполненные часы 'отдыхал' считай слабым качеством данных: это может быть отдых, "
        "а может быть незаполненный слот. Не делай выводы, которые не следуют из данных. "
        "Верни строго JSON без markdown и без HTML с ключами assessment и memory. "
        "assessment: текст оценки дня в формате: Оценка X/10; Вердикт; Что подтверждается фактами; "
        "Что просело; 3 рекомендации на завтра; Над чем работать. "
        "memory: обновленная краткая память о пользователе до 1500 символов. "
        "В памяти держи только устойчивые факты: кто он, цели, повторяющиеся паттерны, ограничения, "
        "рабочие привычки. Не превращай память в дневник."
    )
    user_prompt = (
        f"Пользователь:\n{memory_text}\n\n"
        f"Настройки трекинга: timezone={user.timezone}, окно={user.start_hour:02d}:00-{user.end_hour:02d}:59.\n\n"
        f"Статистика дня:\n{build_counts_context(stats)}\n\n"
        f"Заполнение по часам:\n{build_entries_context(entries)}\n\n"
        f"Вечерняя рефлексия пользователя:\n{reflection.strip()}"
    )
    return [
        {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
        {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
    ]


def build_ai_memory_messages(current_memory: str, note: str) -> list[dict]:
    system_prompt = (
        "Ты обновляешь краткую долговременную память Telegram-бота о пользователе. "
        "Верни только итоговый текст памяти, без markdown и без пояснений. "
        "Память должна быть до 1500 символов и содержать устойчивые факты: кто пользователь, "
        "его цели, стиль работы, ограничения, повторяющиеся паттерны. Не добавляй временные мелочи."
    )
    user_prompt = (
        f"Текущая память:\n{current_memory}\n\n"
        f"Новая информация от пользователя:\n{note.strip()}"
    )
    return [
        {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
        {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
    ]


async def generate_daily_ai_assessment(
    session: AsyncSession,
    user: User,
    report_date: date,
    reflection: str,
) -> str:
    memory = await get_or_create_user_memory(session, user)
    stats = await build_report_stats(session, user, report_date, report_date)
    entries = await get_day_entries(session, user.id, report_date)
    raw_text = await ai_client_pool.create_text(
        build_ai_daily_messages(user, stats, entries, memory.profile_text, reflection),
        max_output_tokens=OPENAI_MAX_OUTPUT_TOKENS,
    )
    assessment_text, updated_memory = parse_ai_daily_payload(raw_text, memory.profile_text)

    memory.profile_text = updated_memory[:3000]
    memory.updated_at = datetime.utcnow()
    await session.commit()

    await upsert_daily_ai_assessment(
        session=session,
        user=user,
        report_date=report_date,
        assessment_text=assessment_text,
        memory_snapshot=memory.profile_text,
    )
    return assessment_text


async def update_memory_from_note(session: AsyncSession, user: User, note: str) -> str:
    memory = await get_or_create_user_memory(session, user)
    current_memory = memory.profile_text.strip()

    if OPENAI_API_KEY:
        try:
            updated_memory = await ai_client_pool.create_text(
                build_ai_memory_messages(current_memory, note),
                max_output_tokens=700,
            )
        except Exception as error:
            logger.exception("Failed to update memory with OpenAI: %s", error)
            updated_memory = f"{current_memory}\n{note.strip()}".strip()
    else:
        updated_memory = f"{current_memory}\n{note.strip()}".strip()

    memory.profile_text = updated_memory[:3000]
    memory.updated_at = datetime.utcnow()
    await session.commit()
    return memory.profile_text


# =========================================================
# COMMANDS
# =========================================================
@dp.message(CommandStart())
async def cmd_start(message: Message) -> None:
    async with SessionLocal() as session:
        user = await get_or_create_user(
            session=session,
            tg_user_id=message.from_user.id,
            chat_id=message.chat.id,
            username=message.from_user.username,
            full_name=message.from_user.full_name,
        )
        await get_or_create_user_memory(session, user)
        text = (
            "Привет. Я буду каждый час спрашивать, что ты делал за предыдущий час, "
            "и собирать статистику по времени.\n\n"
            f"Сейчас у тебя:\n"
            f"— timezone: <b>{user.timezone}</b>\n"
            f"— окно трекинга: <b>{user.start_hour:02d}:00–{user.end_hour:02d}:59</b>\n"
            f"— отчет: <b>{user.report_hour:02d}:00</b> по твоей timezone\n\n"
            "Команды:\n"
            "/today — отчет за сегодня\n"
            "/week — отчет за 7 дней\n"
            "/month — отчет за 30 дней\n"
            "/memory — что я о тебе помню\n"
            "/remember текст — добавить в память\n"
            "/forgetmemory — сбросить память\n"
            "/settings — настройки\n"
            "/pause — пауза\n"
            "/resume — возобновить\n"
            "/test — тестовый вопрос"
            )
        await message.answer(text)


@dp.message(Command("settings"))
async def cmd_settings(message: Message) -> None:
    async with SessionLocal() as session:
        user = await find_user_by_tg(session, message.from_user.id)
        if not user:
            await message.answer("Сначала нажми /start")
            return
        text = (
            f"<b>Текущие настройки</b>\n\n"
            f"Timezone: <b>{user.timezone}</b>\n"
            f"Окно трекинга: <b>{user.start_hour:02d}:00–{user.end_hour:02d}:59</b>\n"
            f"Отчет: <b>{user.report_hour:02d}:00</b>"
        )
        await message.answer(text, reply_markup=settings_keyboard())
        await message.answer(
            "Быстрые команды:\n"
            "/sethours 9 21\n"
            "/settz Europe/Moscow"
        )


@dp.message(Command("memory"))
async def cmd_memory(message: Message) -> None:
    async with SessionLocal() as session:
        user = await find_user_by_tg(session, message.from_user.id)
        if not user:
            await message.answer("Сначала нажми /start")
            return

        memory = await get_or_create_user_memory(session, user)
        await answer_escaped_text(message, "Память", memory.profile_text or "Пока я знаю о тебе мало.")


@dp.message(Command("remember"))
async def cmd_remember(message: Message) -> None:
    note = (message.text or "").split(maxsplit=1)
    if len(note) != 2 or not note[1].strip():
        await message.answer("Напиши так: /remember кто ты, цели, ограничения, что для тебя важно")
        return

    async with SessionLocal() as session:
        user = await find_user_by_tg(session, message.from_user.id)
        if not user:
            await message.answer("Сначала нажми /start")
            return

        updated_memory = await update_memory_from_note(session, user, note[1])
        await answer_escaped_text(message, "Запомнил", updated_memory)


@dp.message(Command("forgetmemory"))
async def cmd_forgetmemory(message: Message) -> None:
    async with SessionLocal() as session:
        user = await find_user_by_tg(session, message.from_user.id)
        if not user:
            await message.answer("Сначала нажми /start")
            return

        memory = await get_or_create_user_memory(session, user)
        memory.profile_text = build_default_memory(user)
        memory.updated_at = datetime.utcnow()
        await session.commit()
        await message.answer("Память сбросил до базового профиля.")


@dp.message(Command("sethours"))
async def cmd_sethours(message: Message) -> None:
    parts = (message.text or "").split()
    if len(parts) != 3:
        await message.answer("Используй так: /sethours 9 21")
        return
    try:
        start_hour = int(parts[1])
        end_hour = int(parts[2])
    except ValueError:
        await message.answer("Часы должны быть числами. Пример: /sethours 9 21")
        return

    if not (0 <= start_hour <= 23 and 0 <= end_hour <= 23 and start_hour < end_hour):
        await message.answer("Нужен корректный диапазон часов. Пример: /sethours 9 21")
        return

    async with SessionLocal() as session:
        user = await find_user_by_tg(session, message.from_user.id)
        if not user:
            await message.answer("Сначала нажми /start")
            return
        user.start_hour = start_hour
        user.end_hour = end_hour
        await session.commit()
        await message.answer(f"Сохранил окно: <b>{start_hour:02d}:00–{end_hour:02d}:59</b>")


@dp.message(Command("settz"))
async def cmd_settz(message: Message) -> None:
    parts = (message.text or "").split(maxsplit=1)
    if len(parts) != 2:
        await message.answer("Используй так: /settz Europe/Moscow")
        return
    tz_name = parts[1].strip()
    if tz_name not in available_timezones():
        await message.answer("Такой timezone не найден. Пример: Europe/Moscow")
        return

    async with SessionLocal() as session:
        user = await find_user_by_tg(session, message.from_user.id)
        if not user:
            await message.answer("Сначала нажми /start")
            return
        user.timezone = tz_name
        await session.commit()
        await message.answer(f"Timezone сохранен: <b>{tz_name}</b>")


@dp.message(Command("pause"))
async def pause_bot(message: Message) -> None:
    async with SessionLocal() as session:
        user = await find_user_by_tg(session, message.from_user.id)
        if not user:
            await message.answer("Сначала нажми /start")
            return
        user.is_active = False
        await session.commit()
        await message.answer("Поставил на паузу.")


@dp.message(Command("resume"))
async def resume_bot(message: Message) -> None:
    async with SessionLocal() as session:
        user = await find_user_by_tg(session, message.from_user.id)
        if not user:
            await message.answer("Сначала нажми /start")
            return
        user.is_active = True
        await session.commit()
        await message.answer("Снова активен.")


@dp.message(Command("test"))
async def cmd_test(message: Message) -> None:
    async with SessionLocal() as session:
        user = await find_user_by_tg(session, message.from_user.id)
        if not user:
            await message.answer("Сначала нажми /start")
            return

        now = now_in_tz(user.timezone)
        entry_date, hour_slot = get_previous_hour_slot(now)

        text = (
            f"<b>ТЕСТ</b>\n"
            f"Что ты делал с <b>{hour_slot:02d}:00 до {hour_slot:02d}:59</b>?\n"
        )

        await message.answer(
            text,
            reply_markup=activity_keyboard(entry_date=entry_date, hour_slot=hour_slot),
        )


@dp.message(Command("today"))
async def today_report(message: Message) -> None:
    async with SessionLocal() as session:
        user = await find_user_by_tg(session, message.from_user.id)
        if not user:
            await message.answer("Сначала нажми /start")
            return
        report_date = today_in_tz(user.timezone)
        await autofill_missing_as_rest(session, user, report_date)
        stats = await build_report_stats(session, user, report_date, report_date)
        text = build_day_report_text_from_stats(user, stats, include_reflection_prompt=False)
        img = build_chart_png(f"Итог дня {report_date.strftime('%d.%m.%Y')}", stats.counts)
        await answer_report_photo(message, img, "today.png", text)


@dp.message(Command("week"))
async def week_report(message: Message) -> None:
    async with SessionLocal() as session:
        user = await find_user_by_tg(session, message.from_user.id)
        if not user:
            await message.answer("Сначала нажми /start")
            return
        end_date = today_in_tz(user.timezone)
        start_date = end_date - timedelta(days=6)
        stats = await build_report_stats(session, user, start_date, end_date)
        text = build_period_report_text(stats, "Итог за 7 дней")
        img = build_chart_png("Итог за 7 дней", stats.counts)
        await answer_report_photo(message, img, "week.png", text)


@dp.message(Command("month"))
async def month_report(message: Message) -> None:
    async with SessionLocal() as session:
        user = await find_user_by_tg(session, message.from_user.id)
        if not user:
            await message.answer("Сначала нажми /start")
            return
        end_date = today_in_tz(user.timezone)
        start_date = end_date - timedelta(days=29)
        stats = await build_report_stats(session, user, start_date, end_date)
        text = build_period_report_text(stats, "Итог за 30 дней")
        img = build_chart_png("Итог за 30 дней", stats.counts)
        await answer_report_photo(message, img, "month.png", text)


# =========================================================
# CALLBACKS
# =========================================================
@dp.callback_query(F.data == "settings:hours")
async def cb_settings_hours(callback: CallbackQuery) -> None:
    await callback.message.answer("Напиши команду в таком формате: /sethours 9 21")
    await callback.answer()


@dp.callback_query(F.data == "settings:tz")
async def cb_settings_tz(callback: CallbackQuery) -> None:
    await callback.message.answer("Напиши команду в таком формате: /settz Europe/Moscow")
    await callback.answer()


@dp.callback_query(F.data.startswith("track:"))
async def process_track_callback(callback: CallbackQuery) -> None:
    try:
        _, raw_date, raw_hour, code = callback.data.split(":")
        entry_date = date.fromisoformat(raw_date)
        hour_slot = int(raw_hour)
    except Exception:
        await callback.answer("Некорректные данные", show_alert=True)
        return

    if code not in OPTIONS:
        await callback.answer("Неизвестная категория", show_alert=True)
        return

    async with SessionLocal() as session:
        user = await find_user_by_tg(session, callback.from_user.id)
        if not user:
            await callback.answer("Сначала нажми /start", show_alert=True)
            return

        await upsert_time_entry(
            session=session,
            user_id=user.id,
            entry_date=entry_date,
            hour_slot=hour_slot,
            status=code,
            label=OPTIONS[code].text,
            source_message_id=callback.message.message_id if callback.message else None,
            is_auto_filled=False,
            answered_at=datetime.utcnow(),
        )

        if callback.message:
            new_text = (
                f"Что ты делал с <b>{hour_slot:02d}:00 до {hour_slot:02d}:59</b>?\n\n"
                f"Текущий ответ: <b>{OPTIONS[code].text}</b>"
            )
            await callback.message.edit_text(
                new_text,
                reply_markup=activity_keyboard(entry_date=entry_date, hour_slot=hour_slot, selected=code),
            )
        await callback.answer(f"Сохранено: {OPTIONS[code].text}")


# =========================================================
# REFLECTION FSM
# =========================================================
@dp.message(ReflectionFlow.waiting_reflection, F.text)
async def capture_reflection(message: Message, state: FSMContext) -> None:
    async with SessionLocal() as session:
        user = await find_user_by_tg(session, message.from_user.id)
        if not user:
            await message.answer("Сначала нажми /start")
            await state.clear()
            return

        state_data = await state.get_data()
        raw_report_date = state_data.get("report_date")
        try:
            report_date = date.fromisoformat(raw_report_date) if raw_report_date else today_in_tz(user.timezone)
        except ValueError:
            report_date = today_in_tz(user.timezone)

        report = await get_or_create_daily_report(session, user.id, report_date)
        if not report.reflection_requested:
            await message.answer("Сейчас я не жду вечернюю рефлексию.")
            await state.clear()
            return

        reflection_text = message.text.strip()
        report.reflection = reflection_text
        report.reflection_requested = False
        report.reflection_received_at = datetime.utcnow()
        await session.commit()
        await state.clear()
        await message.answer("Рефлексию сохранил. Делаю холодный разбор дня...")

        try:
            assessment_text = await generate_daily_ai_assessment(session, user, report_date, reflection_text)
        except Exception as error:
            logger.exception("Failed AI daily assessment user=%s error=%s", user.tg_user_id, error)
            await message.answer(
                "Рефлексию сохранил, но AI-разбор не получился: "
                f"<code>{html.escape(type(error).__name__)}</code>. "
                "Проверь OPENAI_API_KEY, зависимости и прокси."
            )
            return

        await answer_escaped_text(message, "Холодная оценка дня", assessment_text)


# =========================================================
# SCHEDULED JOBS
# =========================================================
async def send_due_hourly_questions() -> None:
    now_utc = datetime.now(ZoneInfo("UTC"))

    async with SessionLocal() as session:
        users = await get_active_users(session)

        for user in users:
            user_now = now_utc.astimezone(get_zone(user.timezone))

            if user_now.minute != 0:
                continue

            previous_date, previous_hour = get_previous_hour_slot(user_now)

            if previous_hour < user.start_hour or previous_hour > user.end_hour:
                continue

            try:
                text = (
                    f"Что ты делал с <b>{previous_hour:02d}:00 до {previous_hour:02d}:59</b>?"
                )

                await bot.send_message(
                    chat_id=user.chat_id,
                    text=text,
                    reply_markup=activity_keyboard(
                        entry_date=previous_date,
                        hour_slot=previous_hour,
                    ),
                )
            except Exception as e:
                logger.exception(
                    "Failed hourly question user=%s error=%s",
                    user.tg_user_id,
                    e,
                )


async def send_due_daily_reports() -> None:
    now_utc = datetime.now(ZoneInfo("UTC"))
    async with SessionLocal() as session:
        users = await get_active_users(session)
        for user in users:
            user_now = now_utc.astimezone(get_zone(user.timezone))
            if user_now.hour != user.report_hour or user_now.minute != 0:
                continue

            try:
                report_date = user_now.date()
                await autofill_missing_as_rest(session, user, report_date)
                stats = await build_report_stats(session, user, report_date, report_date)
                report_text = build_day_report_text_from_stats(user, stats, include_reflection_prompt=True)
                img = build_chart_png(f"Итог дня {report_date.strftime('%d.%m.%Y')}", stats.counts)

                report = await get_or_create_daily_report(session, user.id, report_date, report_text)
                report.summary_text = report_text
                report.reflection_requested = True
                await session.commit()

                await send_report_photo(user.chat_id, img, "day_report.png", report_text)

                state = dp.fsm.get_context(bot=bot, chat_id=user.chat_id, user_id=user.tg_user_id)
                await state.update_data(report_date=report_date.isoformat())
                await state.set_state(ReflectionFlow.waiting_reflection)
            except Exception as e:
                logger.exception("Failed daily report user=%s error=%s", user.tg_user_id, e)


def setup_scheduler() -> None:
    scheduler.add_job(
        send_due_hourly_questions,
        trigger="cron",
        minute=0,
        id="due_hourly_questions",
        replace_existing=True,
    )
    scheduler.add_job(
        send_due_daily_reports,
        trigger="cron",
        minute=0,
        id="due_daily_reports",
        replace_existing=True,
    )


# =========================================================
# MAIN
# =========================================================
async def main() -> None:
    if not BOT_TOKEN:
        raise RuntimeError("BOT_TOKEN is empty. Put it into .env file")

    await init_db()
    await migrate_legacy_activity_labels()
    setup_scheduler()
    scheduler.start()
    try:
        await dp.start_polling(bot)
    finally:
        await ai_client_pool.close()


if __name__ == "__main__":
    asyncio.run(main())


# =========================================================
# .env.example
# =========================================================
# BOT_TOKEN=1234567890:YOUR_TELEGRAM_BOT_TOKEN
# DATABASE_URL=sqlite+aiosqlite:///tracker.db
# DEFAULT_TIMEZONE=Europe/Moscow
# DEFAULT_START_HOUR=9
# DEFAULT_END_HOUR=21
# REPORT_HOUR=23
# OPENAI_API_KEY=sk-xxxxxxx
# OPENAI_MODEL=gpt-4.1-mini
# HTTP_PROXY=http://user:pass@host:port
# HTTP_PROXY_2=http://user:pass@host2:port
# HTTP_PROXY_3=http://user:pass@host3:port


# =========================================================
# requirements.txt
# =========================================================
# aiogram>=3.0
# APScheduler>=3.10
# SQLAlchemy>=2.0
# aiosqlite>=0.19
# matplotlib>=3.8
# python-dotenv>=1.0
# httpx==0.28.1
# openai==2.32.0


# =========================================================
# QUICK START
# =========================================================
# 1) Создай .env рядом с файлом и вставь:
#    BOT_TOKEN=твой_токен
# 2) Установи зависимости:
#    pip install -r requirements.txt
# 3) Запусти:
#    python main.py
# 4) В телеграме нажми /start

# Команды:
# /today   - отчет за сегодня
# /week    - отчет за 7 дней
# /month   - отчет за 30 дней
# /memory  - что бот помнит о тебе
# /remember текст
# /forgetmemory
# /settings
# /sethours 9 21
# /settz Europe/Moscow
# /pause
# /resume

import asyncio
import io
import logging
import os
from dataclasses import dataclass
from datetime import datetime, date, timedelta
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


class TimeEntry(Base):
    __tablename__ = "time_entries"
    __table_args__ = (UniqueConstraint("user_id", "entry_date", "hour_slot", name="uq_user_day_hour"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    entry_date: Mapped[date] = mapped_column(Date, index=True)
    hour_slot: Mapped[int] = mapped_column(Integer, index=True)
    status: Mapped[str] = mapped_column(String(32), default="rest")
    label: Mapped[str] = mapped_column(String(255), default="Отдыхаю 😌")
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
    "rest": ActivityOption("rest", "😌 Отдыхаю"),
    "other": ActivityOption("other", "📝 Другое"),
}

STATUS_ORDER = ["income", "work", "grow", "rest", "other"]


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


async def build_counts_for_range(session: AsyncSession, user: User, start_date: date, end_date: date) -> dict[str, int]:
    counts = {code: 0 for code in STATUS_ORDER}
    current = start_date
    while current <= end_date:
        entries = await get_day_entries(session, user.id, current)
        for e in entries:
            counts[e.status] = counts.get(e.status, 0) + 1
        current += timedelta(days=1)
    return counts


async def build_day_report_text(session: AsyncSession, user: User, entry_date: date) -> str:
    entries = await get_day_entries(session, user.id, entry_date)
    counts = {code: 0 for code in STATUS_ORDER}
    auto_rest = 0
    total_slots = len(user_hours(user))

    for e in entries:
        counts[e.status] = counts.get(e.status, 0) + 1
        if e.is_auto_filled and e.status == "rest":
            auto_rest += 1

    def bar(hours: int, total: int) -> str:
        blocks = round((hours / total) * 10) if total else 0
        return "█" * blocks + "░" * (10 - blocks)

    lines = [
        f"<b>Итог за {entry_date.strftime('%d.%m.%Y')}</b>",
        "",
        f"Часовой диапазон: <b>{user.start_hour:02d}:00–{user.end_hour:02d}:59</b>",
        f"Timezone: <b>{user.timezone}</b>",
        f"Автозаполнено как отдых: <b>{auto_rest}</b>",
        "",
    ]

    for code in STATUS_ORDER:
        hours = counts[code]
        lines.append(f"{OPTIONS[code].text} — <b>{hours} ч</b>  {bar(hours, total_slots)}")

    lines.extend(
        [
            "",
            "Теперь напиши сообщением короткую рефлексию за день:",
            "— что делал,",
            "— что было полезным,",
            "— что можно улучшить завтра.",
        ]
    )
    return "\n".join(lines)


def build_chart_png(title: str, counts: dict[str, int]) -> bytes:
    labels = [OPTIONS[code].text for code in STATUS_ORDER]
    values = [counts.get(code, 0) for code in STATUS_ORDER]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(labels, values)
    ax.set_title(title)
    ax.set_ylabel("Часы")
    ax.set_xlabel("Категории")
    plt.xticks(rotation=15)
    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=160)
    plt.close(fig)
    buffer.seek(0)
    return buffer.read()


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
        text = await build_day_report_text(session, user, report_date)
        counts = await build_counts_for_range(session, user, report_date, report_date)
        img = build_chart_png(f"Итог дня {report_date.strftime('%d.%m.%Y')}", counts)
        await message.answer_photo(BufferedInputFile(img, filename="today.png"), caption=text)


@dp.message(Command("week"))
async def week_report(message: Message) -> None:
    async with SessionLocal() as session:
        user = await find_user_by_tg(session, message.from_user.id)
        if not user:
            await message.answer("Сначала нажми /start")
            return
        end_date = today_in_tz(user.timezone)
        start_date = end_date - timedelta(days=6)
        counts = await build_counts_for_range(session, user, start_date, end_date)
        total = sum(counts.values())
        lines = [f"<b>Итог за 7 дней</b> ({start_date.strftime('%d.%m')}–{end_date.strftime('%d.%m')})", ""]
        for code in STATUS_ORDER:
            lines.append(f"{OPTIONS[code].text} — <b>{counts[code]} ч</b>")
        lines.append("")
        lines.append(f"Всего отмеченных часов: <b>{total}</b>")
        img = build_chart_png("Итог за 7 дней", counts)
        await message.answer_photo(BufferedInputFile(img, filename="week.png"), caption="\n".join(lines))


@dp.message(Command("month"))
async def month_report(message: Message) -> None:
    async with SessionLocal() as session:
        user = await find_user_by_tg(session, message.from_user.id)
        if not user:
            await message.answer("Сначала нажми /start")
            return
        end_date = today_in_tz(user.timezone)
        start_date = end_date - timedelta(days=29)
        counts = await build_counts_for_range(session, user, start_date, end_date)
        total = sum(counts.values())
        lines = [f"<b>Итог за 30 дней</b> ({start_date.strftime('%d.%m')}–{end_date.strftime('%d.%m')})", ""]
        for code in STATUS_ORDER:
            lines.append(f"{OPTIONS[code].text} — <b>{counts[code]} ч</b>")
        lines.append("")
        lines.append(f"Всего отмеченных часов: <b>{total}</b>")
        img = build_chart_png("Итог за 30 дней", counts)
        await message.answer_photo(BufferedInputFile(img, filename="month.png"), caption="\n".join(lines))


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

        report_date = today_in_tz(user.timezone)
        report = await get_or_create_daily_report(session, user.id, report_date)
        if not report.reflection_requested:
            await message.answer("Сейчас я не жду вечернюю рефлексию.")
            await state.clear()
            return

        report.reflection = message.text.strip()
        report.reflection_requested = False
        report.reflection_received_at = datetime.utcnow()
        await session.commit()
        await state.clear()
        await message.answer("Рефлексию сохранил 👌")


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
                report_text = await build_day_report_text(session, user, report_date)
                counts = await build_counts_for_range(session, user, report_date, report_date)
                img = build_chart_png(f"Итог дня {report_date.strftime('%d.%m.%Y')}", counts)

                report = await get_or_create_daily_report(session, user.id, report_date, report_text)
                report.summary_text = report_text
                report.reflection_requested = True
                await session.commit()

                await bot.send_photo(
                    chat_id=user.chat_id,
                    photo=BufferedInputFile(img, filename="day_report.png"),
                    caption=report_text,
                )

                state = dp.fsm.get_context(bot=bot, chat_id=user.chat_id, user_id=user.tg_user_id)
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
    setup_scheduler()
    scheduler.start()
    await dp.start_polling(bot)


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


# =========================================================
# requirements.txt
# =========================================================
# aiogram>=3.0
# APScheduler>=3.10
# SQLAlchemy>=2.0
# aiosqlite>=0.19
# matplotlib>=3.8
# python-dotenv>=1.0


# =========================================================
# QUICK START
# =========================================================
# 1) Создай .env рядом с файлом и вставь:
#    BOT_TOKEN=твой_токен
# 2) Установи зависимости:
#    pip install -r requirements.txt
# 3) Запусти:
#    python tg_discipline_tracker_bot_mvp.py
# 4) В телеграме нажми /start

# Команды:
# /today   - отчет за сегодня
# /week    - отчет за 7 дней
# /month   - отчет за 30 дней
# /settings
# /sethours 9 21
# /settz Europe/Moscow
# /pause
# /resume

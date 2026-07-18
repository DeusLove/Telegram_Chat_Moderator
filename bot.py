import asyncio
import logging
from typing import Tuple, Dict
from datetime import datetime, timedelta
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Пути к моделям
TOXIC_MODEL_PATH = ".\\toxic_model"
SPAM_MODEL_PATH = ".\\spam_model"

# Пороги для классификации (настройте под свои нужды)
TOXICITY_THRESHOLD = 0.5
SPAM_THRESHOLD = 0.5

# Настройки системы мутов
MUTE_DURATIONS = [1, 6, 24]  # Длительность мутов в часах: 1-е нарушение, 2-е, 3-е
MAX_VIOLATIONS = len(MUTE_DURATIONS)  # После исчерпания мутов - бан
VIOLATION_EXPIRY_HOURS = 168  # Через сколько часов сбрасывается история нарушений (7 дней)


class ModeratorBot:
    def __init__(self, token: str):
        self.token = token

        # Словарь для хранения истории нарушений: {user_id: [список нарушений с временными метками]}
        self.violations: Dict[int, list] = {}

        # Словарь для отслеживания активных мутов: {user_id: datetime окончания мута}
        self.active_mutes: Dict[int, datetime] = {}

        # Загружаем модели
        logger.info("Загрузка моделей...")
        self.toxic_classifier = pipeline(
            "text-classification",
            model=AutoModelForSequenceClassification.from_pretrained(TOXIC_MODEL_PATH),
            tokenizer=AutoTokenizer.from_pretrained(TOXIC_MODEL_PATH)
        )

        self.spam_classifier = pipeline(
            "text-classification",
            model=AutoModelForSequenceClassification.from_pretrained(SPAM_MODEL_PATH),
            tokenizer=AutoTokenizer.from_pretrained(SPAM_MODEL_PATH)
        )
        logger.info("Модели загружены успешно")

    def clean_expired_violations(self, user_id: int):
        """Очищает просроченные нарушения пользователя"""
        if user_id in self.violations:
            current_time = datetime.now()
            expiry_time = current_time - timedelta(hours=VIOLATION_EXPIRY_HOURS)

            # Оставляем только не просроченные нарушения
            self.violations[user_id] = [
                v for v in self.violations[user_id]
                if v['timestamp'] > expiry_time
            ]

            # Если нарушений не осталось, удаляем запись о пользователе
            if not self.violations[user_id]:
                del self.violations[user_id]

    def check_active_mute(self, user_id: int) -> bool:
        """Проверяет, активен ли мут у пользователя"""
        if user_id in self.active_mutes:
            if datetime.now() < self.active_mutes[user_id]:
                return True
            else:
                # Мут истёк, удаляем запись
                del self.active_mutes[user_id]
        return False

    def get_user_violations(self, user_id: int) -> int:
        """Возвращает количество активных нарушений пользователя"""
        self.clean_expired_violations(user_id)
        return len(self.violations.get(user_id, []))

    def add_violation(self, user_id: int, reason: str, message_text: str):
        """Добавляет нарушение пользователю"""
        if user_id not in self.violations:
            self.violations[user_id] = []

        self.violations[user_id].append({
            'timestamp': datetime.now(),
            'reason': reason,
            'message': message_text[:100]  # Сохраняем начало сообщения для истории
        })

        logger.info(f"Нарушение добавлено пользователю {user_id}. Всего нарушений: {len(self.violations[user_id])}")

    async def mute_user(self, user_id: int, chat_id: int, context: ContextTypes.DEFAULT_TYPE,
                        duration_hours: int, violation_count: int):
        """Мутит пользователя на указанное количество часов"""
        try:
            # Устанавливаем время окончания мута
            mute_until = datetime.now() + timedelta(hours=duration_hours-3)

            # Мутим пользователя в чате
            await context.bot.restrict_chat_member(
                chat_id=chat_id,
                user_id=user_id,
                permissions={
                    'can_send_messages': False,
                    'can_send_media_messages': False,
                    'can_send_other_messages': False,
                    'can_add_web_page_previews': False
                },
                until_date=mute_until
            )

            # Сохраняем информацию о муте
            self.active_mutes[user_id] = mute_until

            logger.info(f"Пользователь {user_id} замучен на {duration_hours} час(ов)")
            return True

        except Exception as e:
            logger.error(f"Ошибка при муте пользователя {user_id}: {e}")
            return False

    async def check_message(self, text: str) -> Tuple[bool, bool, dict]:
        """
        Проверяет сообщение на токсичность и спам
        Возвращает: (is_toxic, is_spam, scores)
        """
        # Проверка на токсичность
        toxic_result = self.toxic_classifier(text[:512])[0]
        is_toxic = toxic_result['label'] == 'LABEL_1' and toxic_result['score'] >= TOXICITY_THRESHOLD

        # Проверка на спам
        spam_result = self.spam_classifier(text[:512])[0]
        is_spam = spam_result['label'] == 'LABEL_1' and spam_result['score'] >= SPAM_THRESHOLD

        scores = {
            'toxic_score': toxic_result['score'],
            'spam_score': spam_result['score']
        }

        return is_toxic, is_spam, scores

    async def moderate_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Обработчик входящих сообщений"""
        message = update.message
        user = message.from_user
        chat_id = message.chat_id

        # Игнорируем сообщения без текста
        if not message.text:
            return

        # Проверяем, не замучен ли пользователь
        if self.check_active_mute(user.id):
            # Удаляем сообщение от замученного пользователя
            try:
                await message.delete()
                logger.info(f"Сообщение от замученного пользователя {user.id} удалено")
            except Exception as e:
                logger.error(f"Не удалось удалить сообщение от замученного пользователя: {e}")
            return

        # Проверяем сообщение
        is_toxic, is_spam, scores = await self.check_message(message.text)

        # Логируем результат
        logger.info(
            f"Сообщение от {user.id} ({user.username or 'без username'}): "
            f"toxic={is_toxic} (score={scores['toxic_score']:.3f}), "
            f"spam={is_spam} (score={scores['spam_score']:.3f})"
        )

        # Если сообщение токсичное или спам
        if is_toxic or is_spam:
            try:
                username_display = f"@{user.username}" if user.username else f"{user.first_name or 'Пользователь'}"

                # Для спама - бан сразу без предупреждений
                if is_spam:
                    # Удаляем сообщение
                    await message.delete()
                    logger.info(f"Спам-сообщение от {user.id} удалено")

                    await context.bot.ban_chat_member(
                        chat_id=chat_id,
                        user_id=user.id
                    )

                    notification = f"{username_display} заблокирован за спам"
                    await context.bot.send_message(
                        chat_id=chat_id,
                        text=notification
                    )
                    logger.info(f"Пользователь {user.id} забанен за спам")

                # Для токсичности - система мутов
                elif is_toxic:
                    # Удаляем сообщение
                    await message.delete()
                    logger.info(f"Токсичное сообщение от {user.id} удалено")

                    # Добавляем нарушение
                    self.add_violation(user.id, "токсичность", message.text)
                    current_violations = self.get_user_violations(user.id)

                    if current_violations > MAX_VIOLATIONS:
                        # Если превышен лимит нарушений - бан
                        await context.bot.ban_chat_member(
                            chat_id=chat_id,
                            user_id=user.id
                        )

                        notification = (
                            f"{username_display} заблокирован за систематическую токсичность "
                            f"(нарушений: {current_violations})"
                        )

                        # Очищаем историю нарушений
                        if user.id in self.violations:
                            del self.violations[user.id]
                        if user.id in self.active_mutes:
                            del self.active_mutes[user.id]

                        logger.info(f"Пользователь {user.id} забанен после {current_violations} нарушений")

                    else:
                        # Выдаём мут
                        mute_index = current_violations - 1  # Индекс в массиве длительностей
                        mute_hours = MUTE_DURATIONS[min(mute_index, len(MUTE_DURATIONS) - 1)]

                        # Мутим пользователя
                        muted = await self.mute_user(user.id, chat_id, context, mute_hours, current_violations)

                        if muted:
                            if mute_hours == 1:
                                duration_text = "1 час"
                            elif mute_hours < 24:
                                duration_text = f"{mute_hours} часов"
                            else:
                                days = mute_hours // 24
                                duration_text = f"{days} {'день' if days == 1 else 'дня' if 1 < days < 5 else 'дней'}"

                            if current_violations < MAX_VIOLATIONS:
                                notification = (
                                    f"{username_display} замучен на {duration_text} "
                                    f"за токсичность (нарушение {current_violations})"
                                )
                            else:
                                notification = (
                                    f"{username_display} замучен на {duration_text} "
                                    f"за токсичность. Следующее нарушение приведёт к бану!"
                                )

                            logger.info(f"Пользователь {user.id} замучен на {mute_hours} час(ов)")

                    await context.bot.send_message(
                        chat_id=chat_id,
                        text=notification
                    )

            except Exception as e:
                logger.error(f"Ошибка при модерации: {e}")
                # Если основное действие не удалось, пробуем отправить уведомление
                try:
                    violation_type = "спам" if is_spam else "токсичность"
                    await context.bot.send_message(
                        chat_id=chat_id,
                        text=f"Обнаружено нарушение ({violation_type}) от пользователя {username_display}"
                    )
                except:
                    pass

    def run(self):
        """Запуск бота"""
        app = Application.builder().token(self.token).build()

        # Обрабатываем все текстовые сообщения
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.moderate_message))

        logger.info("Бот запущен")
        app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    BOT_TOKEN = "8620999346:AAH-djvsoVn__ousjE7olT5N5rXCQWGH1EI"
    bot = ModeratorBot(BOT_TOKEN)
    bot.run()
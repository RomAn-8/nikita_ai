"""Helper functions."""

from telegram.ext import ContextTypes


def reset_tz(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Reset TZ mode state."""
    context.user_data.pop("tz_history", None)
    context.user_data.pop("tz_questions", None)
    context.user_data.pop("tz_done", None)


def reset_forest(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Reset forest mode state."""
    context.user_data.pop("forest_history", None)
    context.user_data.pop("forest_questions", None)
    context.user_data.pop("forest_done", None)
    context.user_data.pop("forest_result", None)


def _city_prepositional_case(city: str) -> str:
    """
    Склоняет название города в предложный падеж (где? в чём?).
    Примеры: Москва -> Москве, Самара -> Самаре, Саратов -> Саратове, Томск -> Томске.
    """
    city = (city or "").strip()
    if not city:
        return city
    
    city_lower = city.lower()
    
    if city_lower.endswith("а"):
        return city[:-1] + "е"
    
    if city_lower.endswith("о"):
        return city[:-1] + "е"
    
    if city_lower.endswith("ь"):
        return city[:-1] + "и"
    
    last_char = city_lower[-1]
    if last_char not in "аеёиоуыэюяь":
        return city + "е"
    
    return city

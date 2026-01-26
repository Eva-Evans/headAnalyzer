from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class SaleDecision:
    # реализация (сколько “убрать” из стада в конце месяца)
    sell_bulls_0_2m: float = 0.0
    sell_heifers_3_8: float = 0.0
    sell_heifers_9_24: float = 0.0
    sell_neteli: float = 0.0
    sell_cows_dry: float = 0.0
    sell_cows_milking: float = 0.0

    # какие группы “упирались” в вместимость
    capped: Dict[str, bool] = field(default_factory=dict)


def apply_capacity_sales_month_end(
    counts: Dict[str, float],
    capacity: Dict[str, int],
    *,
    sell_all_bulls: bool = True,
) -> SaleDecision:
    """
    Политика “вместимость → реализация” в конце месяца.

    Идея:
    - если группа > вместимости → считаем excess и предлагаем реализацию (продажу/вывод)
    - в UI можно подсветить “capped” и показывать значение = capacity
    """
    d = SaleDecision()

    def cap(name: str) -> int | None:
        v = capacity.get(name)
        return int(v) if v is not None else None

    # Бычков обычно продают всех (или по желанию)
    if sell_all_bulls:
        d.sell_bulls_0_2m = float(max(0.0, counts.get("Бычки 0–2 мес", 0.0)))

    # Коровы
    cap_milk = cap("Дойные коровы")
    if cap_milk is not None:
        milk = float(max(0.0, counts.get("Дойные коровы", 0.0)))
        ex = max(0.0, milk - float(cap_milk))
        if ex > 0:
            d.sell_cows_milking = ex
            d.capped["Дойные коровы"] = True

    cap_dry = cap("Сухостойные коровы")
    if cap_dry is not None:
        dry = float(max(0.0, counts.get("Сухостойные коровы", 0.0)))
        ex = max(0.0, dry - float(cap_dry))
        if ex > 0:
            d.sell_cows_dry = ex
            d.capped["Сухостойные коровы"] = True

    # Тёлки
    # 0–3 мес (в модели 0–2, но это ок)
    cap_h03 = cap("Тёлки 0–3 мес")
    if cap_h03 is not None:
        h02 = float(max(0.0, counts.get("Тёлки 0–2 мес", 0.0)))
        ex = max(0.0, h02 - float(cap_h03))
        if ex > 0:
            # как правило, таких не “реализуют”, но если хочешь — можно добавить
            d.capped["Тёлки 0–3 мес"] = True

    # 3–8
    cap_h38 = cap("Тёлки 3–8 мес")
    if cap_h38 is not None:
        h38 = float(max(0.0, counts.get("Тёлки 3–8 мес", 0.0)))
        ex = max(0.0, h38 - float(cap_h38))
        if ex > 0:
            d.sell_heifers_3_8 = ex
            d.capped["Тёлки 3–8 мес"] = True

    # 9–24 (в модели: >=9 + нетели)
    cap_h924 = cap("Тёлки 9–24 мес")
    if cap_h924 is not None:
        h9p = float(max(0.0, counts.get("Тёлки ≥9 мес", 0.0)))
        neteli = float(max(0.0, counts.get("Нетели", 0.0)))
        total = h9p + neteli
        ex = max(0.0, total - float(cap_h924))
        if ex > 0:
            # приоритет: сначала “лишних” открытых тёлок 9+, потом (если надо) нетелей
            sell_h = min(ex, h9p)
            sell_n = max(0.0, ex - sell_h)
            d.sell_heifers_9_24 = sell_h
            d.sell_neteli = sell_n
            d.capped["Тёлки 9–24 мес"] = True

    return d

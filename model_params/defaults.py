from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


GESTATION_DAYS: int = 272
DRY_DAYS: int = 53


HERD_CAPACITY = {
    "Дойные коровы": 2400,
    "Сухостойные коровы": 400,
    "Тёлки 0–3 мес": 770,
    "Тёлки 3–8 мес": 1440,
    "Тёлки 9–24 мес": 3360,
}


@dataclass(frozen=True)
class ConceptionParams:
    avg_cow_dim_by_lact: Dict[int, float]
    avg_cow_dim_global: float
    avg_heifer_age_days: float


CONCEPTION_PARAMS = ConceptionParams(
    avg_cow_dim_by_lact={
        1: 98.73304050756467,
        2: 107.09975498774939,
        3: 105.4420138888889,
        4: 106.92692146157077,
    },
    avg_cow_dim_global=103.86853307138179,
    avg_heifer_age_days=401.5583385514506,
)


@dataclass(frozen=True)
class SemenUsage:
    cow_trad: float
    cow_sex: float
    heifer_trad: float
    heifer_sex: float


SEMEN_USAGE_PROBS = SemenUsage(
    cow_trad=0.7,
    cow_sex=0.3,
    heifer_trad=0.3,
    heifer_sex=0.7,
)


@dataclass(frozen=True)
class SemenSexRatio:
    bull_share: float
    heifer_share: float


SEMEN_SEX_RATIOS: Dict[str, SemenSexRatio] = {
    "trad": SemenSexRatio(bull_share=0.2483, heifer_share=0.7517),
    "sex": SemenSexRatio(bull_share=0.0583, heifer_share=0.9417),
}


DISPOSAL_PARAMS: Dict[str, object] = {
    "by_lact": {
        1: {"n": 768, "mean_dim": 160.1, "median_dim": 111.0},
        2: {"n": 642, "mean_dim": 234.7, "median_dim": 226.0},
        3: {"n": 891, "mean_dim": 192.1, "median_dim": 194.0},
        4: {"n": 1448, "mean_dim": 126.6, "median_dim": 73.0},
    },
    "overall": {"n": 3749, "mean_dim": 167.5, "median_dim": 129.0},
}

ANNUAL_DISPOSAL_RATE: float = 0.0957


@dataclass(frozen=True)
class InseminationParams:
    cow_first_ai_dim_by_lact: Dict[int, float]
    cow_ai_interval_days: float
    cow_services_per_conception: float
    heifer_first_ai_age_days: float
    heifer_ai_interval_days: float
    heifer_services_per_conception: float
    cow_conception_month_factors: Dict[int, float]
    heifer_conception_month_factors: Dict[int, float]


INSEMINATION_PARAMS = InseminationParams(
    cow_first_ai_dim_by_lact={
        1: 71.35295643153528,
        2: 72.24234172906739,
        3: 73.28410331029465,
        4: 72.91689373297002,
    },
    cow_ai_interval_days=46.78500715648855,
    cow_services_per_conception=2.0376243474835025,
    heifer_first_ai_age_days=378.5310701203558,
    heifer_ai_interval_days=25.258195726080622,
    heifer_services_per_conception=1.9456635318704285,
    cow_conception_month_factors={m: 1.0 for m in range(1, 13)},
    heifer_conception_month_factors={m: 1.0 for m in range(1, 13)},
)

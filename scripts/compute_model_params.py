from __future__ import annotations

import pandas as pd
from db import engine


                                       

def load_data():
    calv = pd.read_sql("SELECT * FROM calvings_births_raw", con=engine)
    ins = pd.read_sql("SELECT * FROM inseminations_raw", con=engine)
    disp = pd.read_sql("SELECT * FROM disposals_raw", con=engine)
    dry = pd.read_sql("SELECT * FROM dryoff_raw", con=engine)
    return calv, ins, disp, dry


def prepare_dates(calv, ins, disp, dry):
                        
    for col in ["event_date", "birth_date", "disposal_date"]:
        if col in calv.columns:
            calv[col] = pd.to_datetime(calv[col], errors="coerce")

                
    if "event_date" in ins.columns:
        ins["event_date"] = pd.to_datetime(ins["event_date"], errors="coerce")

             
    for col in ["event_date", "birth_date"]:
        if col in disp.columns:
            disp[col] = pd.to_datetime(disp[col], errors="coerce")

             
    if "event_date" in dry.columns:
        dry["event_date"] = pd.to_datetime(dry["event_date"], errors="coerce")

    return calv, ins, disp, dry



def _norm_event_type(x: object) -> str:
    return (
        str(x)
        .replace("\xa0", " ")
        .strip()
        .upper()
        .replace("Ё", "Е")
    )


def _lact_cat(x) -> int | None:
    try:
        lx = int(x)
    except Exception:
        return None
    if lx <= 1:
        return 1
    if lx == 2:
        return 2
    if lx == 3:
        return 3
    return 4



def compute_conception_params(ins: pd.DataFrame):
    """
    Когда наступает стельность:
    - по коровам: средний DIM по лактации и общий
    - по тёлкам: средний возраст стельности в днях
    """
    print("\n=== ПАРАМЕТРЫ СТЕЛЬНОСТИ (DIM / возраст) ===")

    if ins.empty:
        print("ОШИБКА: таблица inseminations_raw пустая")
        return None

    df = ins.copy()

    must = {"reg", "lact", "dim_age", "result"}
    missing = must - set(df.columns)
    if missing:
        print(f"ОШИБКА: в inseminations_raw нет колонок: {missing}")
        return None

    df["result"] = df["result"].astype(str).str.strip().str.upper()
    df = df[df["result"] == "P"].copy()
    df["dim_age"] = pd.to_numeric(df["dim_age"], errors="coerce")
    df = df[df["dim_age"].notna()]

    if df.empty:
        print("НЕТ плодотворных осеменений (Result = 'P')")
        return None

                            
    df["lact_num"] = pd.to_numeric(df["lact"], errors="coerce").fillna(0)
    cows = df[df["lact_num"] > 0].copy()
    heifers = df[df["lact_num"] <= 0].copy()

    avg_cow_dim_by_lact = {}
    avg_cow_dim_global = None
    avg_heifer_age_days = None

    if not cows.empty:
        cows["lact_cat"] = cows["lact_num"].apply(_lact_cat)
        cows = cows[cows["lact_cat"].notna()]

        avg_cow_dim_by_lact = cows.groupby("lact_cat")["dim_age"].mean().to_dict()
        avg_cow_dim_global = cows["dim_age"].mean()

        print("Коровы, кол-во P:", len(cows))
        print("Средний DIM стельности по лактациям (1, 2, 3, ≥4):")
        for k, v in sorted(avg_cow_dim_by_lact.items()):
            label = f"{k}-я лактация" if k in (1, 2, 3) else "4-я и старше"
            print(f"  {label}: {v:.1f} дней")
        print(f"Общий средний DIM стельности по коровам: {avg_cow_dim_global:.1f} дней")

    if not heifers.empty:
        avg_heifer_age_days = heifers["dim_age"].mean()
        print("\nТёлки, кол-во P:", len(heifers))
        print(f"Средний возраст стельности тёлок: {avg_heifer_age_days:.1f} дней")

    return {
        "avg_cow_dim_by_lact": avg_cow_dim_by_lact,
        "avg_cow_dim_global": avg_cow_dim_global,
        "avg_heifer_age_days": avg_heifer_age_days,
    }


                                                                    

def compute_gestation_params(calv: pd.DataFrame, ins: pd.DataFrame):
    """
    Длительность стельности (осеменение -> отёл) считаем через телят:
      - берём рождения телят (event_type == "РОЖДЕН") с mother_reg
      - считаем, что calving_date матери = дата рождения телёнка (event_date)
      - находим последнее P-осеменение матери до calving_date

    Важно:
      - merge_asof требует datetime64[ns] для left_on/right_on
      - и одинаковый числовой dtype для ключа by (mother_reg)
    """
    print("\n=== ДЛИТЕЛЬНОСТЬ СТЕЛЬНОСТИ (через телят + mother_reg) ===")

    if calv.empty or ins.empty:
        print("ОШИБКА: calvings или inseminations пустые")
        return None

    df_calv = calv.copy()
    df_ins = ins.copy()

    if "mother_reg" not in df_calv.columns:
        print("ОШИБКА: в calvings_births_raw нет mother_reg")
        return None
    if "event_date" not in df_calv.columns or "event_type" not in df_calv.columns:
        print("ОШИБКА: в calvings_births_raw нет event_date или event_type")
        return None

    must_ins = {"reg", "event_date", "result"}
    missing = must_ins - set(df_ins.columns)
    if missing:
        print("ОШИБКА: в inseminations_raw не хватает колонок:", missing)
        return None

    df_calv["event_date"] = pd.to_datetime(df_calv["event_date"], errors="coerce")
    df_ins["event_date"] = pd.to_datetime(df_ins["event_date"], errors="coerce")

    df_calv["event_type_norm"] = df_calv["event_type"].apply(_norm_event_type)

    calves = df_calv[
        df_calv["event_date"].notna()
        & df_calv["mother_reg"].notna()
        & (df_calv["event_type_norm"] == "РОЖДЕН")
    ].copy()

    if calves.empty:
        print("Нет строк 'РОЖДЕН' с заполненным mother_reg")
        return None

    calves["mother_reg"] = pd.to_numeric(calves["mother_reg"], errors="coerce")
    calves = calves[calves["mother_reg"].notna()].copy()
    calves["mother_reg"] = calves["mother_reg"].astype("int64")

    calves["calving_date"] = pd.to_datetime(calves["event_date"], errors="coerce")
    calves = calves[calves["calving_date"].notna()].copy()

                                                   
    calves = calves.drop_duplicates(subset=["mother_reg", "calving_date"])
    calves = calves[["mother_reg", "calving_date"]].copy()

    df_ins["result_norm"] = df_ins["result"].astype(str).str.strip().str.upper()
    ins_p = df_ins[
        df_ins["event_date"].notna()
        & df_ins["reg"].notna()
        & (df_ins["result_norm"] == "P")
    ].copy()

    if ins_p.empty:
        print("Нет плодотворных осеменений (Result='P')")
        return None

    ins_p["mother_reg"] = pd.to_numeric(ins_p["reg"], errors="coerce")
    ins_p = ins_p[ins_p["mother_reg"].notna()].copy()
    ins_p["mother_reg"] = ins_p["mother_reg"].astype("int64")

    ins_p["ins_date"] = pd.to_datetime(ins_p["event_date"], errors="coerce")
    ins_p = ins_p[ins_p["ins_date"].notna()].copy()

    ins_p = ins_p[["mother_reg", "ins_date"]].copy()

                                      
                                                   
    calves = calves.sort_values(["calving_date", "mother_reg"], kind="mergesort").reset_index(drop=True)
    ins_p = ins_p.sort_values(["ins_date", "mother_reg"], kind="mergesort").reset_index(drop=True)

    pairs = pd.merge_asof(
        calves,
        ins_p,
        by="mother_reg",
        left_on="calving_date",
        right_on="ins_date",
        direction="backward",
        allow_exact_matches=True,
    )

    pairs = pairs[pairs["ins_date"].notna()].copy()
    if pairs.empty:
        print("Не удалось сопоставить рождения телят с P-осеменениями матерей")
        return None

    pairs["gest_days"] = (pairs["calving_date"] - pairs["ins_date"]).dt.days

                            
    pairs = pairs[(pairs["gest_days"] > 200) & (pairs["gest_days"] < 310)]
    if pairs.empty:
        print("После фильтра 200–310 дней нет данных")
        return None

    print("Количество пар P-осеменение -> отёл(по телятам):", len(pairs))
    print(
        "Длительность стельности, дни:",
        f"min={int(pairs['gest_days'].min())},",
        f"median={float(pairs['gest_days'].median()):.1f},",
        f"mean={float(pairs['gest_days'].mean()):.1f},",
        f"max={int(pairs['gest_days'].max())}",
    )

    return {"avg_gestation_days": float(pairs["gest_days"].mean())}


def compute_dryoff_params(calv: pd.DataFrame, dry: pd.DataFrame):
    """
    Сухостой считаем как (дата отёла коровы - дата запуска коровы).

    Отёлы коровы восстанавливаем через телят:
      - берём строки calvings_births_raw, где event_type == "РОЖДЕН"
      - cow_reg = mother_reg
      - calving_date = event_date (дата рождения телёнка)

    Запуски берём из dryoff_raw:
      - cow_reg = reg
      - dryoff_date = event_date

    Важно:
      - merge_asof требует datetime64[ns] для left_on/right_on
      - и одинаковый числовой dtype для ключа by (cow_reg)
    """
    print("\n=== СУХОСТОЙ (dryoff -> отёл) ===")

    if calv.empty or dry.empty:
        print("ОШИБКА: calvings или dryoff пустые")
        return None

    df_calv = calv.copy()
    df_dry = dry.copy()

                              
    need_calv = {"mother_reg", "event_date", "event_type"}
    missing_c = need_calv - set(df_calv.columns)
    if missing_c:
        print("ОШИБКА: в calvings_births_raw не хватает колонок:", missing_c)
        return None

    need_dry = {"reg", "event_date"}
    missing_d = need_dry - set(df_dry.columns)
    if missing_d:
        print("ОШИБКА: в dryoff_raw не хватает колонок:", missing_d)
        return None

    df_calv["event_date"] = pd.to_datetime(df_calv["event_date"], errors="coerce")
    df_dry["event_date"] = pd.to_datetime(df_dry["event_date"], errors="coerce")

                                     
    df_calv["event_type_norm"] = df_calv["event_type"].apply(_norm_event_type)

    calves = df_calv[
        df_calv["event_date"].notna()
        & df_calv["mother_reg"].notna()
        & (df_calv["event_type_norm"] == "РОЖДЕН")
    ].copy()

    if calves.empty:
        print("Нет строк 'РОЖДЕН' с заполненным mother_reg для расчёта сухостоя")
        return None

    calves["cow_reg"] = pd.to_numeric(calves["mother_reg"], errors="coerce")
    calves["calving_date"] = pd.to_datetime(calves["event_date"], errors="coerce")
    calves = calves.dropna(subset=["cow_reg", "calving_date"]).copy()
    calves["cow_reg"] = calves["cow_reg"].astype("int64")

                                             
    calves = calves.drop_duplicates(subset=["cow_reg", "calving_date"])
    calves = calves[["cow_reg", "calving_date"]].copy()

                           
    dry_ev = df_dry[
        df_dry["event_date"].notna()
        & df_dry["reg"].notna()
    ].copy()

    if dry_ev.empty:
        print("Нет строк запусков с reg и event_date")
        return None

    dry_ev["cow_reg"] = pd.to_numeric(dry_ev["reg"], errors="coerce")
    dry_ev["dryoff_date"] = pd.to_datetime(dry_ev["event_date"], errors="coerce")
    dry_ev = dry_ev.dropna(subset=["cow_reg", "dryoff_date"]).copy()
    dry_ev["cow_reg"] = dry_ev["cow_reg"].astype("int64")

    dry_ev = dry_ev[["cow_reg", "dryoff_date"]].copy()

    calves = calves.sort_values(["calving_date", "cow_reg"], kind="mergesort").reset_index(drop=True)
    dry_ev = dry_ev.sort_values(["dryoff_date", "cow_reg"], kind="mergesort").reset_index(drop=True)

                                                
    merged = pd.merge_asof(
        calves,
        dry_ev,
        by="cow_reg",
        left_on="calving_date",
        right_on="dryoff_date",
        direction="backward",
        allow_exact_matches=True,
    )

    merged = merged[merged["dryoff_date"].notna()].copy()
    if merged.empty:
        print("Не удалось сопоставить отёлы с запусками (merge_asof дал пусто)")
        return None

    merged["dry_days"] = (merged["calving_date"] - merged["dryoff_date"]).dt.days

                                                
    merged = merged[(merged["dry_days"] >= 20) & (merged["dry_days"] <= 120)].copy()
    if merged.empty:
        print("После фильтра 20–120 дней нет данных по сухостою")
        return None

    print("Количество пар запуск -> отёл:", len(merged))
    print(
        "Сухостой, дни:",
        f"min={int(merged['dry_days'].min())},",
        f"median={float(merged['dry_days'].median()):.1f},",
        f"mean={float(merged['dry_days'].mean()):.1f},",
        f"max={int(merged['dry_days'].max())}",
    )

    return {
        "avg_dry_days": float(merged["dry_days"].mean()),
        "median_dry_days": float(merged["dry_days"].median()),
        "n": int(len(merged)),
    }


def compute_insemination_usage_params(calv: pd.DataFrame, ins: pd.DataFrame):
    """
    ОСЕМЕНЕНИЯ: попытки до P и интервалы.

    Возвращаем ключи, которые ожидает main():
      - cow_services_per_conception
      - cow_ai_interval_days
      - heifer_services_per_conception
      - heifer_ai_interval_days

    Дополнительно (не мешает):
      - cow_ai_interval_days_median
      - heifer_ai_interval_days_median
      - cow_first_ai_dim_by_lact
      - heifer_first_ai_age_days
      - n_cow_cycles / n_heifers_with_p
    """
    print("\n=== ОСЕМЕНЕНИЯ: попытки до P и интервалы ===")

    if calv.empty or ins.empty:
        print("ОШИБКА: calvings или inseminations пустые")
        return None

    df_calv = calv.copy()
    df_ins = ins.copy()

                      
    need_calv = {"mother_reg", "event_date", "event_type"}
    missing_c = need_calv - set(df_calv.columns)
    if missing_c:
        print("ОШИБКА: в calvings_births_raw не хватает колонок:", missing_c)
        return None

    need_ins = {"reg", "event_date", "result", "lact", "dim_age"}
    missing_i = need_ins - set(df_ins.columns)
    if missing_i:
        print("ОШИБКА: в inseminations_raw не хватает колонок:", missing_i)
        return None

                              
    df_calv["event_date"] = pd.to_datetime(df_calv["event_date"], errors="coerce")
    df_ins["event_date"] = pd.to_datetime(df_ins["event_date"], errors="coerce")

                                     
    df_calv["event_type_norm"] = df_calv["event_type"].apply(_norm_event_type)

    calvings = df_calv[
        df_calv["event_date"].notna()
        & df_calv["mother_reg"].notna()
        & (df_calv["event_type_norm"] == "РОЖДЕН")
    ].copy()

    if calvings.empty:
        print("Нет строк 'РОЖДЕН' с mother_reg — не могу собрать циклы коров")
        return None

    calvings["cow_reg"] = pd.to_numeric(calvings["mother_reg"], errors="coerce")
    calvings["calving_date"] = pd.to_datetime(calvings["event_date"], errors="coerce")
    calvings = calvings.dropna(subset=["cow_reg", "calving_date"]).copy()
    calvings["cow_reg"] = calvings["cow_reg"].astype("int64")
    calvings = calvings.drop_duplicates(subset=["cow_reg", "calving_date"])
    calvings = calvings[["cow_reg", "calving_date"]].copy()

                        
    df_ins["result_norm"] = df_ins["result"].astype(str).str.strip().str.upper()
    df_ins["lact"] = pd.to_numeric(df_ins["lact"], errors="coerce")
    df_ins["dim_age"] = pd.to_numeric(df_ins["dim_age"], errors="coerce")

    insems = df_ins[
        df_ins["event_date"].notna()
        & df_ins["reg"].notna()
        & df_ins["result_norm"].notna()
    ].copy()

    if insems.empty:
        print("Нет осеменений с event_date/reg/result")
        return None

    insems["animal_reg"] = pd.to_numeric(insems["reg"], errors="coerce")
    insems = insems[insems["animal_reg"].notna()].copy()
    insems["animal_reg"] = insems["animal_reg"].astype("int64")
    insems = insems.rename(columns={"event_date": "ins_date"})

    cows = insems[insems["lact"].fillna(0) > 0].copy()
    heifers = insems[insems["lact"].fillna(0) <= 0].copy()

    def safe_mean(xs):
        return float(pd.Series(xs, dtype="float64").mean()) if xs else None

    def safe_median(xs):
        return float(pd.Series(xs, dtype="float64").median()) if xs else None

                              
                             
                              
    cow_services = []
    cow_intervals = []
    first_ai_dim_rows = []

    if not cows.empty:
        cows = cows.rename(columns={"animal_reg": "cow_reg"})
        cows = cows[cows["ins_date"].notna()].copy()
        cows["cow_reg"] = cows["cow_reg"].astype("int64")

        cows = cows.sort_values(["ins_date", "cow_reg"], kind="mergesort").reset_index(drop=True)
        calv_sorted = calvings.sort_values(["calving_date", "cow_reg"], kind="mergesort").reset_index(drop=True)

        merged = pd.merge_asof(
            cows,
            calv_sorted,
            by="cow_reg",
            left_on="ins_date",
            right_on="calving_date",
            direction="backward",
            allow_exact_matches=True,
        )

        merged = merged[merged["calving_date"].notna()].copy()
        if not merged.empty:
            def lact_group(x):
                try:
                    lx = int(x)
                except Exception:
                    return None
                if lx <= 1:
                    return 1
                if lx == 2:
                    return 2
                if lx == 3:
                    return 3
                return 4

            merged["lact_cat"] = merged["lact"].apply(lact_group)
            merged = merged.sort_values(["cow_reg", "calving_date", "ins_date"], kind="mergesort")

            merged["prev_ins_date"] = merged.groupby(["cow_reg", "calving_date"])["ins_date"].shift(1)
            merged["ai_gap"] = (merged["ins_date"] - merged["prev_ins_date"]).dt.days
            cow_intervals.extend(merged["ai_gap"].dropna().tolist())

            first_in_cycle = merged.groupby(["cow_reg", "calving_date"], as_index=False).first()
            for _, r in first_in_cycle.iterrows():
                if pd.notna(r.get("dim_age")) and pd.notna(r.get("lact_cat")):
                    first_ai_dim_rows.append((int(r["lact_cat"]), float(r["dim_age"])))

            for (cow_reg, calving_date), g in merged.groupby(["cow_reg", "calving_date"]):
                g = g.sort_values("ins_date").reset_index(drop=True)
                p_pos = g.index[g["result_norm"] == "P"]
                if len(p_pos) == 0:
                    continue
                cow_services.append(int(p_pos[0] + 1))

                              
                              
                              
    heifer_services = []
    heifer_intervals = []
    heifer_first_ages = []

    if not heifers.empty:
        h = heifers.rename(columns={"animal_reg": "heifer_reg"}).copy()
        h = h.sort_values(["heifer_reg", "ins_date"], kind="mergesort")

        h["prev_ins_date"] = h.groupby("heifer_reg")["ins_date"].shift(1)
        h["ai_gap"] = (h["ins_date"] - h["prev_ins_date"]).dt.days
        heifer_intervals.extend(h["ai_gap"].dropna().tolist())

        first_h = h.groupby("heifer_reg", as_index=False).first()
        for _, r in first_h.iterrows():
            if pd.notna(r.get("dim_age")):
                heifer_first_ages.append(float(r["dim_age"]))

        for reg, g in h.groupby("heifer_reg"):
            g = g.sort_values("ins_date").reset_index(drop=True)
            p_pos = g.index[g["result_norm"] == "P"]
            if len(p_pos) == 0:
                continue
            heifer_services.append(int(p_pos[0] + 1))

                      
    cow_services_mean = safe_mean(cow_services)
    cow_gap_mean = safe_mean(cow_intervals)
    cow_gap_median = safe_median(cow_intervals)

    heif_services_mean = safe_mean(heifer_services)
    heif_gap_mean = safe_mean(heifer_intervals)
    heif_gap_median = safe_median(heifer_intervals)
    heif_first_age_mean = safe_mean(heifer_first_ages)

    first_by_lact = {}
    if first_ai_dim_rows:
        tmp = pd.DataFrame(first_ai_dim_rows, columns=["lact_cat", "first_dim"])
        first_by_lact = tmp.groupby("lact_cat")["first_dim"].mean().to_dict()

                    
    if cow_services_mean is not None:
        print(f"Коровы: среднее осеменений до P (от отёла): {cow_services_mean:.2f} (n циклов={len(cow_services)})")
    else:
        print("Коровы: не удалось посчитать осеменения до P (нет циклов с P)")

    if cow_gap_mean is not None:
        print(f"Коровы: интервал между осеменениями: mean={cow_gap_mean:.1f} дней, median={cow_gap_median:.1f} (n интервалов={len(cow_intervals)})")
    else:
        print("Коровы: нет интервалов между осеменениями")

    if first_by_lact:
        print("Коровы: средний DIM первого осеменения после отёла по лактациям:")
        for k in [1, 2, 3, 4]:
            v = first_by_lact.get(k)
            if v is None:
                continue
            label = f"{k}-я" if k in (1, 2, 3) else "4+"
            print(f"  {label}: {float(v):.1f} дней")

    if heif_services_mean is not None:
        print(f"Тёлки: среднее осеменений до P (от первого осем.): {heif_services_mean:.2f} (n тёлок={len(heifer_services)})")
    else:
        print("Тёлки: не удалось посчитать осеменения до P (нет тёлок с P)")

    if heif_gap_mean is not None:
        print(f"Тёлки: интервал между осеменениями: mean={heif_gap_mean:.1f} дней, median={heif_gap_median:.1f} (n интервалов={len(heifer_intervals)})")
    else:
        print("Тёлки: нет интервалов между осеменениями")

    if heif_first_age_mean is not None:
        print(f"Тёлки: средний возраст первого осеменения: {heif_first_age_mean:.1f} дней")
    else:
        print("Тёлки: не удалось оценить возраст первого осеменения")

    return {
        "cow_services_per_conception": cow_services_mean,
        "cow_ai_interval_days": cow_gap_mean,
        "cow_ai_interval_days_median": cow_gap_median,
        "cow_first_ai_dim_by_lact": {int(k): float(v) for k, v in first_by_lact.items()} if first_by_lact else {},

        "heifer_services_per_conception": heif_services_mean,
        "heifer_ai_interval_days": heif_gap_mean,
        "heifer_ai_interval_days_median": heif_gap_median,
        "heifer_first_ai_age_days": heif_first_age_mean,

        "n_cow_cycles": int(len(cow_services)),
        "n_heifers_with_p": int(len(heifer_services)),
    }


def compute_disposal_params(disp: pd.DataFrame):
    print("\n=== ВЫБЫТИЕ КОРОВ (DIM выбытия) ===")

    if disp.empty:
        print("ОШИБКА: disposals_raw пустая")
        return None

    df = disp.copy()

    must = {"reg", "sex", "lact", "age_dim", "event_date", "disposal_reason"}
    missing = must - set(df.columns)
    if missing:
        print(f"ОШИБКА: в disposals_raw нет колонок: {missing}")
        return None

    df["sex"] = (
        df["sex"].astype(str)
        .str.replace("\xa0", " ", regex=False)
        .str.strip()
        .str.upper()
    )
    df = df[df["sex"].isin(["F", "Ж", "FEMALE"])]

    df["lact"] = pd.to_numeric(df["lact"], errors="coerce")
    df = df[df["lact"] > 0]

    df["age_dim"] = pd.to_numeric(df["age_dim"], errors="coerce")
    df = df[df["age_dim"].notna()]

    df["disposal_reason"] = (
        df["disposal_reason"].astype(str)
        .str.replace("\xa0", " ", regex=False)
        .str.strip()
        .str.lower()
        .str.replace("ё", "е", regex=False)
    )
    df = df[~df["disposal_reason"].str.contains("переезд", na=False)]

    if df.empty:
        print("После фильтров нет выбытий коров (кроме переезда)")
        return None

    df["lact_cat"] = df["lact"].apply(_lact_cat)
    df = df[df["lact_cat"].notna()]

    res_by_lact = {}
    overall_n = len(df)
    overall_mean = df["age_dim"].mean()
    overall_median = df["age_dim"].median()

    for lact_cat, g in df.groupby("lact_cat"):
        n = len(g)
        mean_dim = g["age_dim"].mean()
        median_dim = g["age_dim"].median()
        res_by_lact[int(lact_cat)] = {
            "n": int(n),
            "mean_dim": float(mean_dim),
            "median_dim": float(median_dim),
            "share": float(n / overall_n),
        }

    print("Всего выбытий коров (кроме переезда):", overall_n)
    print("DIM выбытия (все лактации):", f"median={overall_median:.1f}, mean={overall_mean:.1f}")
    print("По лактациям (1, 2, 3, ≥4):")
    for k, v in sorted(res_by_lact.items()):
        label = f"{k}-я лактация" if k in (1, 2, 3) else "4-я и старше"
        print(
            f"  {label}: n={v['n']}, median DIM={v['median_dim']:.1f}, mean DIM={v['mean_dim']:.1f}, "
            f"доля выбытий={v['share']*100:.1f}%"
        )

    return {
        "by_lact": res_by_lact,
        "overall": {"n": int(overall_n), "mean_dim": float(overall_mean), "median_dim": float(overall_median)},
    }


def compute_annual_disposal_rate(calv: pd.DataFrame, disp: pd.DataFrame):
    print("\n=== ГОДОВОЙ ПРОЦЕНТ ВЫБЫТИЙ ОТ ОБЩЕГО ПОГОЛОВЬЯ ===")

    if calv.empty or disp.empty:
        print("ОШИБКА: calvings или disposals пустые")
        return None

    df = disp.copy()
    must_disp = {"reg", "sex", "lact", "age_dim", "event_date", "disposal_reason"}
    missing = must_disp - set(df.columns)
    if missing:
        print(f"ОШИБКА: в disposals_raw нет колонок: {missing}")
        return None

    df["sex"] = df["sex"].astype(str).str.upper()
    df = df[df["sex"].isin(["F", "Ж", "FEMALE"])]

    df["lact"] = pd.to_numeric(df["lact"], errors="coerce")
    df = df[df["lact"] > 0]

    df["age_dim"] = pd.to_numeric(df["age_dim"], errors="coerce")
    df = df[df["age_dim"].notna()]

    df["disposal_reason"] = df["disposal_reason"].astype(str).str.lower()
    df = df[~df["disposal_reason"].str.contains("переезд", na=False)]
    df = df[df["event_date"].notna()]

    if df.empty:
        print("После фильтров нет выбытий коров (кроме переезда) для годового процента")
        return None

    total_disposals = len(df)
    years = df["event_date"].dt.year
    year_min = int(years.min())
    year_max = int(years.max())
    n_years = max(1, year_max - year_min + 1)
    annual_disposals = total_disposals / n_years

    must_calv = {"reg", "sex", "lact"}
    missing_c = must_calv - set(calv.columns)
    if missing_c:
        print(f"ОШИБКА: в calvings_births_raw нет колонок: {missing_c}")
        return None

    cows = calv.copy()
    cows["sex"] = cows["sex"].astype(str).str.upper()
    cows = cows[cows["sex"].isin(["F", "Ж", "FEMALE"])]

    cows["lact"] = pd.to_numeric(cows["lact"], errors="coerce")
    cows = cows[cows["lact"] > 0]

    herd_regs = cows["reg"].dropna().astype(str).unique()
    herd_size_est = len(herd_regs)
    if herd_size_est == 0:
        print("Не удалось оценить численность стада (нет регов коров)")
        return None

    annual_rate = annual_disposals / herd_size_est

    print(f"Период по выбытиям: {year_min}–{year_max} (≈ {n_years} лет)")
    print(f"Всего выбытий коров (кроме переезда): {total_disposals}")
    print(f"Оценка среднегодового числа выбытий: {annual_disposals:.1f} голов/год")
    print(f"Оценка численности стада за период: {herd_size_est} коров")
    print(f"Оценка среднегодового процента выбытий: {annual_rate * 100:.2f}%")

    print("\nANNUAL_DISPOSAL_RATE =", annual_rate)

    return {"annual_rate": float(annual_rate), "years": int(n_years), "herd_size_est": int(herd_size_est)}



def main():
    calv, ins, disp, dry = load_data()
    calv, ins, disp, dry = prepare_dates(calv, ins, disp, dry)

    conc = compute_conception_params(ins)
    gest = compute_gestation_params(calv, ins)
    dryp = compute_dryoff_params(calv, dry)
    disp_params = compute_disposal_params(disp)
    annual_disp = compute_annual_disposal_rate(calv, disp)
    ins_usage = compute_insemination_usage_params(calv, ins)

    print("\n\n=== ИТОГИ ДЛЯ ВСТАВКИ В model_params/defaults.py ===")

    if conc is not None:
        print("\nConceptionParams:")
        print("avg_cow_dim_by_lact =", {int(k): float(v) for k, v in conc["avg_cow_dim_by_lact"].items()})
        print("avg_cow_dim_global =", float(conc["avg_cow_dim_global"]) if conc["avg_cow_dim_global"] is not None else None)
        print("avg_heifer_age_days =", float(conc["avg_heifer_age_days"]) if conc["avg_heifer_age_days"] is not None else None)

    if gest is not None:
        print("\nGESTATION_DAYS =", float(gest["avg_gestation_days"]))

    if dryp is not None:
        print("\nDRY_DAYS =", int(round(float(dryp["avg_dry_days"]))))

    if disp_params is not None:
        print("\nDISPOSAL_PARAMS = {")
        print("  'by_lact': {")
        for k, v in disp_params["by_lact"].items():
            print(f"    {int(k)}: {{'n': {int(v['n'])}, 'mean_dim': {float(v['mean_dim']):.1f}, 'median_dim': {float(v['median_dim']):.1f}}},")
        print("  },")
        ov = disp_params["overall"]
        print(f"  'overall': {{'n': {int(ov['n'])}, 'mean_dim': {float(ov['mean_dim']):.1f}, 'median_dim': {float(ov['median_dim']):.1f}}}")
        print("}")

    if annual_disp is not None:
        print("\nANNUAL_DISPOSAL_RATE =", float(annual_disp["annual_rate"]))

    if ins_usage is not None:
        print("\nINSEMINATION_PARAMS = {")
        print("  'cow_services_per_conception':", ins_usage["cow_services_per_conception"])
        print("  'cow_ai_interval_days':", ins_usage["cow_ai_interval_days"])
        print("  'cow_first_ai_dim_by_lact':", {int(k): float(v) for k, v in ins_usage["cow_first_ai_dim_by_lact"].items()})
        print("  'heifer_services_per_conception':", ins_usage["heifer_services_per_conception"])
        print("  'heifer_ai_interval_days':", ins_usage["heifer_ai_interval_days"])
        print("  'heifer_first_ai_age_days':", ins_usage["heifer_first_ai_age_days"])
        print("}")

    print("\n(SEMEN_SEX_RATIOS у тебя уже считаются отдельной функцией — оставь как есть.)")


if __name__ == "__main__":
    main()

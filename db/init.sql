CREATE TABLE IF NOT EXISTS disposals_raw (
    id              BIGINT,
    reg             TEXT,
    birth_date      DATE,
    lact            INTEGER,
    sex             TEXT,
    disposal_reason TEXT,
    event_type      TEXT,
    age_dim         INTEGER,
    event_date      DATE,
    note            TEXT
);

-- таблица запусков (сухостой / dry-off)
CREATE TABLE IF NOT EXISTS dryoff_raw (
    id              BIGINT,
    reg             TEXT,
    birth_date      DATE,       -- BDAT
    lact            INTEGER,    -- LACT
    disposal_date   DATE,       -- ARDAT
    disposal_reason TEXT,       -- CARX
    remark          TEXT,       -- REM
    event_type      TEXT,       -- Событие (обычно "Запуск")
    dim             INTEGER,    -- DIM (дни в доении на момент запуска)
    event_date      DATE,       -- Дата (дата запуска)
    note            TEXT,       -- Примечание
    protocols       TEXT,       -- Протоколы;
    technician      TEXT        -- Техник
);

-- таблица осеменений
CREATE TABLE IF NOT EXISTS inseminations_raw (
    id                  BIGINT,
    reg                 TEXT,
    lact                INTEGER,
    event_type          TEXT,    -- Событие (осеменение)
    dim_age             INTEGER, -- DIM/Возраст
    event_date          DATE,    -- Дата осеменения
    bull                TEXT,    -- Бык
    result              TEXT,    -- O/P/A
    tech_id             TEXT,    -- T (ид техника)
    insemination_type   TEXT,    
    technician          TEXT     -- Техник (ФИО)
);

-- Таблица "Отёлы + родившиеся"
DROP TABLE IF EXISTS calvings_births_raw;

CREATE TABLE calvings_births_raw (
    animal_id          TEXT,      -- ID (в исходнике)
    reg                TEXT,      -- REG животного
    mother_reg         TEXT,      -- DREG (REG матери)
    mother_reg_intl    TEXT,      -- DREG1 (международный номер матери)
    calf1_reg          TEXT,      -- CALF1 (последний телёнок)
    calf2_reg          TEXT,      -- CALF2 (предпоследний телёнок)
    calf3_reg          TEXT,      -- CALF3 (предпредпоследний телёнок)
    birth_date         DATE,      -- BDAT (дата рождения этого животного)
    lact               INTEGER,   -- LACT (лактация)
    disposal_date      DATE,      -- ARDAT (дата выбытия)
    disposal_reason    TEXT,      -- CARX (причина выбытия)
    disposal_remark    TEXT,      -- REM (ремарка по выбытию)
    sex                TEXT,      -- GNDR (пол)
    event_type         TEXT,      -- Событие (отел / родился / ...)
    age                INTEGER,   -- AGE (возраст / DIM в днях, если нужен)
    event_date         DATE,      -- Дата (дата события)
    note               TEXT,      -- Примечание
    protocol           TEXT,      -- Протоколы;
    technician         TEXT       -- Техник
);

CREATE TABLE IF NOT EXISTS bulls_raw (
    bull_code      TEXT,
    short_name     TEXT,
    reg            TEXT,
    secondary_id   TEXT,
    plem           INTEGER,
    breed          TEXT,
    bull_type      TEXT
);

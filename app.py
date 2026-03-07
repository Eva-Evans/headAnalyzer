from __future__ import annotations

import os
import streamlit as st
from core.params import ensure_params_loaded_from_db_silent, admin_key_true, apply_admin_overrides, get_param_source
from ui.tab1_forecast import render_tab1_forecast
from ui.tab2_params import render_tab2_params
from ui.tab3_farm import render_tab3_farm

st.set_page_config(page_title="Прогноз поголовья", layout="wide")
st.title("Прогноз поголовья")

ensure_params_loaded_from_db_silent()

if "is_admin" not in st.session_state:
    st.session_state.is_admin = False

with st.expander("Админ-режим", expanded=st.session_state.is_admin):
    if not st.session_state.is_admin:
        k = st.text_input("Ключ доступа", type="password", key="admin_key_input")
        if st.button("Войти", key="admin_login_btn", use_container_width=True):
            if k == admin_key_true():
                st.session_state.is_admin = True
                st.rerun()
            else:
                st.error("Неверный ключ")
    else:
        c1, c2 = st.columns([1, 3])
        with c1:
            if st.button("Выйти", key="admin_logout_btn", use_container_width=True):
                st.session_state.is_admin = False
                st.session_state.pop("runtime_overrides", None)
                st.rerun()
        with c2:
            st.success("Админ-режим включён")

with st.expander("Справка: как считается прогноз", expanded=False):
    st.markdown(
        """
Мы строим **состояние стада** на дату последнего события в данных.

Дальше выполняем **симуляцию по дням**:
- животные стареют,
- появляются новые стельности,
- стельные уходят в сухостой перед отёлом,
- происходят отёлы,
- происходит выбытие.

В прогнозе показывается **состояние на конец каждого месяца** и **ожидаемые отёлы**.
        """
    )

tab1, tab2, tab3 = st.tabs(
    [
        "Прогноз (одно подразделение)",
        "Параметры",
        "Прогноз по хозяйствам",
    ]
)

with tab1:
    render_tab1_forecast()

with tab2:
    render_tab2_params()

with tab3:
    render_tab3_farm()
    

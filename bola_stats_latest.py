# ---- Homepage: Today's / Tomorrow / Weekend fixtures across all leagues ----
if selected == "All":
    # Toggle chips (Segmented Control if available, else radio fallback)
    try:
        # Streamlit >= 1.32
        selected_range = st.segmented_control(
            "Show fixtures for",
            options=["Today", "Tomorrow", "Weekend"],
            default="Today",
        )
    except Exception:
        selected_range = st.radio(
            "Show fixtures for",
            options=["Today", "Tomorrow", "Weekend"],
            index=0,
            horizontal=True,
        )

    # Build date set for filter (Africa/Lagos)
    base_today = lagos_today()
    if selected_range == "Today":
        date_set = {base_today}
        subtitle = base_today.strftime("%A, %d %b %Y")
    elif selected_range == "Tomorrow":
        d = (base_today + pd.Timedelta(days=1))
        date_set = {d}
        subtitle = d.strftime("%A, %d %b %Y")
    else:
        # Weekend: Fri–Sun of the *upcoming* weekend relative to today
        # Find upcoming Friday (including today if Fri)
        weekday = base_today.weekday()  # Mon=0 ... Sun=6
        days_until_fri = (4 - weekday) % 7
        fri = base_today + pd.Timedelta(days=days_until_fri)
        sat = fri + pd.Timedelta(days=1)
        sun = fri + pd.Timedelta(days=2)
        date_set = {fri, sat, sun}
        subtitle = f"{fri.strftime('%d %b')} – {sun.strftime('%d %b %Y')}"

    st.subheader(f"Fixtures – {selected_range} ({subtitle})")

    any_fixtures = False
    for league in LEAGUE_FILES.keys():
        results_df, fixtures_df = load_data(league)
        if results_df is None or fixtures_df is None or fixtures_df.empty:
            continue

        # Filter to the selected dates (normalize fixture dates first)
        fx = fixtures_df.copy()
        fx["__date_only__"] = fx["date"].dt.normalize()
        fx = fx[fx["__date_only__"].isin(date_set)].copy()
        if fx.empty:
            continue
        any_fixtures = True

        fx = add_time_sort_cols(fx).sort_values(["__dt_sort__", "home_team", "away_team"])
        st.markdown(f"### {league}")
        for _, fix in fx.iterrows():
            home = str(fix.get("home_team") or "")
            away = str(fix.get("away_team") or "")
            if not home or not away:
                continue
            match_label = f"{home} vs {away}"
            with st.expander(match_label, expanded=True):
                trends = generate_trends(home, away, results_df)
                if trends:
                    for pct, desc in trends:
                        icon = next((symbol for key, symbol in EMOJI_MAP.items() if key in desc.lower()), "•")
                        st.markdown(f"{icon} {desc}")
                else:
                    st.info("No strong trends to recommend for this game.")

    if not any_fixtures:
        st.info("No fixtures for the selected range across the configured leagues.")

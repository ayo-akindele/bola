
import streamlit as st
import pandas as pd
from datetime import datetime
import numpy as np  # Needed for vectorized operations when mapping corner counts to the correct team

# Google Sheets CSV export URLs
HISTORICAL_URL = "https://docs.google.com/spreadsheets/d/1oZJlXF6tpLLaEDNfduHzYFvLKDw7rnyzZY17CQNl1so/gviz/tq?tqx=out:csv&gid=0"
FIXTURES_URL = "https://docs.google.com/spreadsheets/d/1oZJlXF6tpLLaEDNfduHzYFvLKDw7rnyzZY17CQNl1so/gviz/tq?tqx=out:csv&gid=1005360909"

st.set_page_config(page_title="BolaStats", layout="centered")
st.title("📊 BolaStats")
# Update caption to use new phrase requested by the user
# Display the caption larger and bolder with the emoji first for visual emphasis.
st.markdown("<h4 style='margin-bottom:0; font-weight:bold;'>⚡ Quick Stats That Matter</h4>", unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        results_df = pd.read_csv(HISTORICAL_URL)
        fixtures_df = pd.read_csv(FIXTURES_URL)
        return results_df, fixtures_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

results_df, fixtures_df = load_data()

summary_top_picks = []

if results_df is not None and fixtures_df is not None:
    results_df.columns = [col.strip().lower().replace(" ", "_") for col in results_df.columns]
    fixtures_df.columns = [col.strip().lower().replace(" ", "_") for col in fixtures_df.columns]
    fixtures_df["date"] = pd.to_datetime(fixtures_df["date"], errors="coerce")
    results_df["match_date"] = pd.to_datetime(results_df["match_date"], errors="coerce")

    today = pd.Timestamp.today().normalize()
    round_dates = fixtures_df.groupby("round_number")["date"].max().sort_index()
    current_round = round_dates[round_dates >= today].index.min()
    if pd.isna(current_round):
        current_round = round_dates.index.max()
    gw_fixtures = fixtures_df[fixtures_df["round_number"] == current_round]

    # Show current gameweek with updated wording: use a vertical bar and label the H2H snapshot
    st.subheader(f"📅 Gameweek {current_round} | H2H Snapshot")

    # Define simple emoji icons for each type of trend to improve readability.  These icons will
    # precede each trend in the output below and make the statistics easier to scan at a glance.
    emoji_map = {
        "Both teams scored": "⚽",
        "Both teams failed to score": "🚫",
        "Over 2.5 goals": "📈",
        "Under 2.5 goals": "📉",
        "Over 9.5 corners": "🏳️",  # using a flag to represent corners
        "Under 9.5 corners": "🔻",
        "Over 3.5 bookings": "🟨",
        "Under 3.5 bookings": "🟩",
        # Removed separate home/away booking icons at the user's request. Match‑level booking
        # statistics remain (over/under 3.5), but individual team bookings were diluting the
        # output impact.  Only the aggregate booking icons are retained.
        "more corners": "🔺",
        "First-half goals": "⏱",
        "won": "🏆"  # catch‑all for winner trends
    }

    def normalize_boolean(col):
        return col.astype(str).str.lower().isin(["1", "true", "yes", "y"])

    def generate_stats(home, away):
        three_seasons_ago = today.year - 3
        h2h_all = results_df[
            ((results_df['home_team'] == home) & (results_df['away_team'] == away)) |
            ((results_df['home_team'] == away) & (results_df['away_team'] == home))
        ]
        h2h = h2h_all[h2h_all["match_date"].dt.year >= three_seasons_ago].sort_values(by="match_date", ascending=False).head(5)

        if h2h.empty:
            return []

        total = len(h2h)
        trends = []

        def trend_check(condition, label):
            try:
                valid = condition.dropna()
                count = valid.sum()
                pct = count / len(valid)
                if len(valid) > 0 and pct >= 0.8:
                    return (pct, f"{label} in {int(count)}/{len(valid)} games")
            except Exception:
                return None
            return None

        # Match winner logic
        wins = 0
        for _, row in h2h.iterrows():
            if row['home_team'] == home and row['home_score'] > row['away_score']:
                wins += 1
            elif row['away_team'] == home and row['away_score'] > row['home_score']:
                wins += 1
        if total > 0 and wins / total >= 0.8:
            trends.append((wins / total, f"{home} won {wins}/{total} recent meetings"))

        # If a BTTS column is present, normalize it for consistency but derive GG/NG from actual scores.
        if 'both_teams_score' in h2h.columns:
            h2h['both_teams_score'] = normalize_boolean(h2h['both_teams_score'])

        # Compute GG (Goal–Goal) and NG (No Goal) from the actual match scores.  A fixture is GG when
        # both teams scored at least one goal; it is NG when at least one team failed to score.  If
        # either score is missing or cannot be parsed, we set the value to None to exclude it from
        # trend calculations.
        if {'home_score', 'away_score'}.issubset(h2h.columns):
            home_goals = pd.to_numeric(h2h['home_score'], errors='coerce')
            away_goals = pd.to_numeric(h2h['away_score'], errors='coerce')
            gg = (home_goals > 0) & (away_goals > 0)
            # Replace entries where either goal is NaN with None so dropna() will ignore them
            gg = gg.where(~(home_goals.isna() | away_goals.isna()), None)
            h2h['gg'] = gg
            # NG is the complement of GG (True when one or both teams did not score)
            h2h['ng'] = h2h['gg'].apply(lambda x: None if pd.isna(x) else not x)

        # Compute Over/Under 2.5 goals. Prefer computing from actual scores when available; otherwise
        # normalise the existing `over_2_5` column and derive the under flag as its complement.
        if {'home_score', 'away_score'}.issubset(h2h.columns):
            home_goals_num = pd.to_numeric(h2h['home_score'], errors='coerce')
            away_goals_num = pd.to_numeric(h2h['away_score'], errors='coerce')
            total_goals = home_goals_num + away_goals_num
            h2h['over_2_5'] = total_goals > 2.5
            h2h['under_2_5'] = total_goals <= 2.5
            h2h.loc[total_goals.isna(), ['over_2_5', 'under_2_5']] = None
        elif 'over_2_5' in h2h.columns:
            h2h['over_2_5'] = normalize_boolean(h2h['over_2_5'])
            # Derive the under flag from the normalised over flag; missing values are set to None
            h2h['under_2_5'] = h2h['over_2_5'].apply(lambda x: None if pd.isna(x) else not x)

        try:
            # Total corners: compute numeric values and derive over/under 9.5 flags.  Treat NaN as None.
            corners_num = pd.to_numeric(h2h['total_corners'], errors='coerce')
            h2h['Corners_Over_9.5'] = corners_num > 9.5
            h2h['Corners_Under_9.5'] = corners_num <= 9.5
            h2h.loc[corners_num.isna(), ['Corners_Over_9.5', 'Corners_Under_9.5']] = None

            # Bookings: match-level over/under 3.5.  We intentionally omit team‑specific
            # bookings (home/away) per user feedback to avoid diluting the summary.
            home_yc = pd.to_numeric(h2h['home_yellow_cards'], errors='coerce')
            away_yc = pd.to_numeric(h2h['away_yellow_cards'], errors='coerce')
            total_bookings = home_yc + away_yc
            h2h['Bookings_Over_3.5'] = total_bookings > 3.5
            h2h['Bookings_Under_3.5'] = total_bookings <= 3.5
            h2h.loc[total_bookings.isna(), ['Bookings_Over_3.5', 'Bookings_Under_3.5']] = None

            # Removed computation of team‑specific booking trends at the user's request. Only
            # match‑level bookings (over/under 3.5) are kept.

            # First half goals: at least one goal scored in the first half by either team
            h2h['First_Half_Goal'] = (pd.to_numeric(h2h['first_half_home'], errors='coerce') +
                                      pd.to_numeric(h2h['first_half_away'], errors='coerce')) > 0
        except Exception:
            pass

        # Market labels map indicator columns to human‑friendly descriptions.  We standardize on
        # "GG" for both teams scoring and "NG" for one or both teams not scoring.
        market_labels = {
            'gg': "Both teams scored (GG)",
            'ng': "Both teams failed to score (NG)",
            'over_2_5': "Over 2.5 goals",
            'under_2_5': "Under 2.5 goals",
            'Corners_Over_9.5': "Over 9.5 corners",
            'Corners_Under_9.5': "Under 9.5 corners",
            'Bookings_Over_3.5': "Over 3.5 bookings",
            'Bookings_Under_3.5': "Under 3.5 bookings",
            # Team‑specific booking labels removed per user request.  We retain only
            # aggregate booking stats.
            'First_Half_Goal': "First-half goals"
        }

        for col, label in market_labels.items():
            if col in h2h.columns:
                result = trend_check(h2h[col], label)
                if result:
                    trends.append(result)

        # Corner dominance trends. Rather than blindly comparing the historic match's
        # "home" and "away" corner counts, map each row's corner totals to the current
        # fixture's home and away team names. This avoids mis‑classifying matches where
        # the venue flips between historic games and the upcoming fixture.
        corner_pairs = [
            ('home_corners', 'away_corners'),
            ('home_corner', 'away_corner'),
            ('homecorner', 'awaycorner'),
            ('corners_home', 'corners_away')
        ]
        for h_col, a_col in corner_pairs:
            if {h_col, a_col}.issubset(h2h.columns):
                # Convert corner counts to numeric, coercing invalids to NaN
                hc = pd.to_numeric(h2h[h_col], errors='coerce')
                ac = pd.to_numeric(h2h[a_col], errors='coerce')

                # Map each historic row's corner totals to the upcoming fixture's home team
                # and away team. If the historic row involves a different pairing, assign NaN.
                # We use numpy.where for vectorized conditional selection.
                home_team_corners = np.where(
                    h2h['home_team'] == home, hc,
                    np.where(h2h['away_team'] == home, ac, np.nan)
                )
                away_team_corners = np.where(
                    h2h['home_team'] == away, hc,
                    np.where(h2h['away_team'] == away, ac, np.nan)
                )

                # Ensure numeric dtype and handle possible strings/objects
                home_team_corners = pd.to_numeric(home_team_corners, errors='coerce')
                away_team_corners = pd.to_numeric(away_team_corners, errors='coerce')

                # Identify rows where either side has missing corners or equal corners; these
                # rows should not count towards dominance statistics.
                mask_bad = (
                    pd.isna(home_team_corners) |
                    pd.isna(away_team_corners) |
                    (home_team_corners == away_team_corners)
                )

                # Determine if the current fixture's home team had more corners than the away
                # team in each historic match
                home_more = pd.Series(home_team_corners > away_team_corners)
                home_more = home_more.where(~mask_bad, None)
                res_home = trend_check(home_more, f"{home} more corners than {away}")
                if res_home:
                    trends.append(res_home)

                # And vice versa: determine if the away team dominated corners
                away_more = pd.Series(away_team_corners > home_team_corners)
                away_more = away_more.where(~mask_bad, None)
                res_away = trend_check(away_more, f"{away} more corners than {home}")
                if res_away:
                    trends.append(res_away)
                # Break after the first matching pair to prevent checking other synonyms
                break

        top_trends = sorted(trends, key=lambda x: x[0], reverse=True)[:3]
        return top_trends

    top_summary_pool = []

    for _, row in gw_fixtures.iterrows():
        home = row['home_team']
        away = row['away_team']
        # Wrap each fixture in an expander for better digestibility.  Users can collapse sections
        # they're not interested in, keeping the page concise.
        with st.expander(f"{home} vs {away}", expanded=True):
            fixture_stats = generate_stats(home, away)
            if fixture_stats:
                for pct, text in fixture_stats:
                    # Determine which emoji to use based on keywords in the trend description
                    icon = "•"
                    for key, symbol in emoji_map.items():
                        if key in text:
                            icon = symbol
                            break
                    st.markdown(f"{icon} {text}")
                    top_summary_pool.append((pct, f"{home} vs {away} → {text}"))
            else:
                st.info("No strong trends to recommend for this game.")

    if top_summary_pool:
        st.markdown("---")
        st.subheader("🔥 Top Picks Summary")
        top_summary_pool.sort(reverse=True)
        shown = 0
        for pct, item in top_summary_pool:
            if pct >= 0.9:
                st.markdown(f"✅ {item}")
                shown += 1
            if shown >= 3:
                break
else:
    st.warning("Unable to fetch data from Google Sheets. Please check links or permissions.")

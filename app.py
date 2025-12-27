import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from logic import (
    expected_goals,
    match_probabilities,
    calibrated_strength_diff,
    suggest_handicap,
    asian_handicap_ev,
    fair_odds_from_prob,
    calculate_stake,
    get_bankroll,
    save_bet,
    settle_last_bet
)
from config import FOOTBALL_DATA_API_KEY

# --- ML Model Integration ---
import joblib
import numpy as np
import os
MODEL_PATH = 'data/random_forest_model.joblib'
rf_model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None


st.set_page_config(page_title="Asian Handicap Betting Assistant", layout="centered")

st.title("Asian Handicap Betting Assistant")
st.caption("Pre-Match | Discipline First | No Emotion")

# Step wizard UI (move to top level)
st.markdown("""
<style>
.step-indicator {
    display: flex;
    justify-content: space-between;
    margin-bottom: 1.5em;
}
.step {
    flex: 1;
    text-align: center;
    padding: 0.5em 0;
    border-bottom: 3px solid #e0e0e0;
    color: #888;
    font-weight: 500;
}
.step.active {
    border-bottom: 3px solid #4F8BF9;
    color: #4F8BF9;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# Step state
if 'step' not in st.session_state:
        st.session_state.step = 1

def set_step(step):
        st.session_state.step = step

# Step indicator
steps = ["1. Teams", "2. Analysis", "3. Odds", "4. Place Bet"]
st.markdown('<div class="step-indicator">' + ''.join([
        f'<div class="step {"active" if st.session_state.step == i+1 else ""}">{step}</div>'
        for i, step in enumerate(steps)
]) + '</div>', unsafe_allow_html=True)

# Initialize session state
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'analysis_data' not in st.session_state:
    st.session_state.analysis_data = {}

 # Tabs

tab1, tab2, tab3 = st.tabs(["Analysis", "History", "Settings"])

with tab1:
    # ---------------- BANKROLL ----------------
    bankroll = get_bankroll()
    
    # Show success message if any
    if 'success_message' in st.session_state:
        st.success(st.session_state.success_message)
        del st.session_state.success_message
    
    col_bank, col_add = st.columns([3, 1])
    with col_bank:
        st.info(f"💼 Bankroll: ₹{bankroll}")
    with col_add:
        if st.button("➕ Add Money"):
            st.session_state.show_add_money = True
    
    if 'show_add_money' in st.session_state and st.session_state.show_add_money:
        with st.form("add_money_inline"):
            add_amount = st.number_input("Amount to Add (₹)", min_value=0.0, step=100.0, key="add_amount")
            col1, col2 = st.columns(2)
            with col1:
                if st.form_submit_button("Confirm Add"):
                    if add_amount > 0:
                        from logic import update_bankroll
                        update_bankroll(add_amount)
                        st.session_state.success_message = f"✅ ₹{add_amount} added successfully to bankroll!"
                        st.session_state.show_add_money = False
                    else:
                        st.error("Please enter a positive amount.")
            with col2:
                if st.form_submit_button("Cancel"):
                    st.session_state.show_add_money = False

    # Load teams
    try:
        teams_df = pd.read_csv("data/teams.csv")
        team_list = teams_df['team'].tolist()
    except:
        team_list = []
        st.warning("Teams database not found.")

    # League selector
    from logic import COMPETITIONS
    selected_league = st.selectbox("Select League", list(COMPETITIONS.keys()), key="league_select")

    # ---------------- INPUTS ----------------
    col1, col2 = st.columns(2)
    with col1:
        st.header("Team A")
        team_a_name = st.text_input("Team A Name", key="team_a_name")
        if st.button("Load/Fetch Stats A", key="load_a"):
            api_key = FOOTBALL_DATA_API_KEY  # Use hardcoded API key
            if team_a_name:
                # Try API fetch only
                from logic import fetch_team_stats
                stats = fetch_team_stats(team_a_name, api_key, is_home=True, league=selected_league)
                if stats:
                    st.session_state.a_form = stats['form']
                    st.session_state.a_goal = stats['goal_diff']
                    st.session_state.a_home = stats['home_adv']
                    st.session_state.a_def = stats['defense']
                    st.success(f"Fetched stats for {team_a_name} from {selected_league}!")
                    st.rerun()
                else:
                    st.error(f"Could not fetch stats for {team_a_name}. Check team name or API key.")
        
        a_form = st.slider("Form A", 0.0, 1.0, value=st.session_state.get('a_form', 0.5))
        a_goal = st.slider("Goal Diff A", 0.0, 1.0, value=st.session_state.get('a_goal', 0.5))
        a_home = st.slider("Home A", 0.0, 1.0, value=st.session_state.get('a_home', 0.5))
        a_def = st.slider("Defense A", 0.0, 1.0, value=st.session_state.get('a_def', 0.5))

    with col2:
        st.header("Team B")
        team_b_name = st.text_input("Team B Name", key="team_b_name")
        if st.button("Load/Fetch Stats B", key="load_b"):
            api_key = FOOTBALL_DATA_API_KEY  # Use hardcoded API key
            if team_b_name:
                # Try API fetch only
                from logic import fetch_team_stats
                stats = fetch_team_stats(team_b_name, api_key, is_home=False, league=selected_league)
                if stats:
                    st.session_state.b_form = stats['form']
                    st.session_state.b_goal = stats['goal_diff']
                    st.session_state.b_home = 0.0  # Away
                    st.session_state.b_def = stats['defense']
                    st.success(f"Fetched stats for {team_b_name} from {selected_league}!")
                    st.rerun()
                else:
                    st.error(f"Could not fetch stats for {team_b_name}. Check team name or API key.")
        
        b_form = st.slider("Form B", 0.0, 1.0, value=st.session_state.get('b_form', 0.5))
        b_goal = st.slider("Goal Diff B", 0.0, 1.0, value=st.session_state.get('b_goal', 0.5))
        b_home = st.slider("Home B", 0.0, 1.0, value=st.session_state.get('b_home', 0.5))
        b_def = st.slider("Defense B", 0.0, 1.0, value=st.session_state.get('b_def', 0.5))

    # Step 2: Analysis
    if st.session_state.step == 2:
        st.header("Match Analysis")
        st.info("Step 2: Review model's suggested bet based on your team inputs.")
        # --- ML Prediction ---
        if rf_model:
            features = [
                st.session_state.get('a_form', 0.5),
                st.session_state.get('b_form', 0.5),
                st.session_state.get('a_goal', 0.5),
                st.session_state.get('b_goal', 0.5),
                st.session_state.get('a_goal', 0.0),
                st.session_state.get('b_goal', 0.0),
                1.0  # Assume home for Team A
            ]
            X_pred = np.array(features).reshape(1, -1)
            pred = rf_model.predict(X_pred)[0]
            proba = rf_model.predict_proba(X_pred)[0]
            outcome_map = {1: 'Team A Win', 0: 'Draw', -1: 'Team B Win'}
            st.subheader("🤖 ML Model Prediction")
            st.write(f"Prediction: **{outcome_map.get(pred, 'Unknown')}**")
            st.write(f"Probabilities: Team A Win: {proba[2]:.2f}, Draw: {proba[1]:.2f}, Team B Win: {proba[0]:.2f}")
        else:
            st.warning("ML model not found. Please train the model first.")
        # --- Existing logic ---
        ega = expected_goals(
            st.session_state.get('a_form', 0.5),
            st.session_state.get('a_goal', 0.5),
            st.session_state.get('a_home', 0.5),
            st.session_state.get('a_def', 0.5)
        )
        egb = expected_goals(
            st.session_state.get('b_form', 0.5),
            st.session_state.get('b_goal', 0.5),
            st.session_state.get('b_home', 0.5),
            st.session_state.get('b_def', 0.5)
        )
        sd = calibrated_strength_diff(ega, egb)
        handicap = suggest_handicap(sd)
        probs = match_probabilities(ega, egb)
        win_p = probs.get('win', 0.33)
        draw_p = probs.get('draw', 0.33)
        loss_p = probs.get('loss', 0.34)
        fair_odds = fair_odds_from_prob(win_p)
        st.session_state.analysis_data = {
            "handicap": handicap,
            "win_p": round(win_p, 3),
            "draw_p": round(draw_p, 3),
            "loss_p": round(loss_p, 3),
            "fair_odds": fair_odds,
            "ega": round(ega, 2),
            "egb": round(egb, 2)
        }
        data = st.session_state.analysis_data
        st.subheader("📊 Analysis Results")

    # ---------------- ANALYSIS ----------------
    st.header("Match Analysis Workflow")
    st.info("📋 Step 1: Enter team names → Step 2: Get suggested handicap & fair odds → Step 3: Enter bookmaker odds → Step 4: Check EV")
    
    if st.button("Analyze Match"):
        ega = expected_goals(a_form, a_goal, a_home, a_def)
        egb = expected_goals(b_form, b_goal, b_home, b_def)

        sd = calibrated_strength_diff(ega, egb)
        handicap = suggest_handicap(sd)

        probs = match_probabilities(ega, egb)
        win_p = probs.get('win', 0.33)
        draw_p = probs.get('draw', 0.33)
        loss_p = probs.get('loss', 0.34)

        fair_odds = fair_odds_from_prob(win_p)

        # Store in session state
        st.session_state.analysis_done = True
        st.session_state.analysis_data = {
            "handicap": handicap,
            "win_p": round(win_p, 3),
            "draw_p": round(draw_p, 3),
            "loss_p": round(loss_p, 3),
            "fair_odds": fair_odds,
            "ega": round(ega, 2),
            "egb": round(egb, 2)
        }

    # Display analysis if done
    if st.session_state.analysis_done:
        data = st.session_state.analysis_data
        st.subheader("📊 Analysis Results")
        
        col_left, col_right = st.columns(2)
        with col_left:
            st.write(f"Expected Goals: A **{data['ega']}** vs B **{data['egb']}**")
            st.write(f"Win / Draw / Loss: **{data['win_p']} / {data['draw_p']} / {data['loss_p']}**")
        
        with col_right:
            st.write(f"🎯 Suggested Handicap: **{data['handicap']}**")
            st.write(f"💰 Fair Odds: **{data['fair_odds']}**")
        
        st.divider()
        # --- Actionable Bet Instruction ---
        st.subheader("🎯 Bet Instruction")
        st.success(f"Bet on: **{team_a_name}** with Asian Handicap: **{data['handicap']}**")
        st.caption("Now check your bookmaker for odds on this exact bet.")

        st.subheader("💡 How to Proceed:")
        st.markdown(f"""
        1. ✅ **Suggested Handicap**: {data['handicap']}
        2. 🔍 **Now search your bookmaker** for odds on "{data['handicap']}"
        3. 📝 **Enter the bookmaker odds** below (e.g., 1.90, 2.10, etc.)
        4. 🎯 **The app will calculate EV** to tell if it's a good bet
        """)
        
        st.subheader("📌 Enter Bookmaker Odds")
        
        st.info(f"""
        **Searching for: {data['handicap']}**
        
        ❓ **Can't find these odds?** Use the options below!
        """)
        
        tab_odds, tab_alternatives = st.tabs(["Enter Odds", "If Odds Unavailable"])
        
        with tab_odds:
            bookmaker_odds = st.number_input(
                f"Bookmaker Odds for {data['handicap']}", 
                min_value=1.01, 
                value=1.90, 
                step=0.01,
                key="bookmaker_odds_input",
                help=f"Enter the odds you see on your bookmaker for {data['handicap']}"
            )
            
            if st.button("Calculate EV & Get Recommendation"):
                ev = asian_handicap_ev(data['handicap'], data['win_p'], data['draw_p'], data['loss_p'], bookmaker_odds)
                stake = calculate_stake(bankroll, ev)
                
                st.session_state.analysis_data['ev'] = ev
                st.session_state.analysis_data['stake'] = stake
                st.session_state.analysis_data['odds'] = bookmaker_odds
                
                st.divider()
                st.subheader("🎲 Bet Decision")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Fair Odds", data['fair_odds'])
                with col2:
                    st.metric("Bookmaker Odds", bookmaker_odds)
                with col3:
                    st.metric("EV", ev, delta="Positive" if ev > 0 else "Negative")
                
                if ev and ev > 0.03:
                    st.success(f"✅ BET RECOMMENDED | EV: {ev} | Suggested Stake: ₹{stake}")
                    if st.button("Save This Bet"):
                        result = save_bet({
                            "handicap": data['handicap'],
                            "odds": bookmaker_odds,
                            "stake": stake,
                            "ev": ev,
                            "win_p": data['win_p'],
                            "draw_p": data['draw_p'],
                            "loss_p": data['loss_p'],
                            "fair_odds": data['fair_odds']
                        })
                        if result == "SAVED":
                            st.success("✅ Bet saved successfully.")
                            st.session_state.analysis_done = False
                        elif result == "OPEN_BET_EXISTS":
                            st.warning("⚠️ You already have an open bet. Settle it first.")
                else:
                    st.error(f"❌ SKIP BET | EV is negative or too low: {ev}")
        
        with tab_alternatives:
            st.markdown(f"""
            ### 📋 What to do if **{data['handicap']}** is not available:
            
            ⚠️ **IMPORTANT**: All alternatives are for **TEAM A only** (the stronger team)
            
            **Option 1: Most bookmakers offer these alternatives (in order of similarity):**
            """)
            
            # Generate alternative handicaps
            handicap_alternatives = {
                "AH -1.0": ["AH -0.75", "AH -0.5"],
                "AH -0.75": ["AH -0.5", "AH -1.0", "AH -1.0"],
                "AH -0.5": ["AH -0.25", "AH -0.75"],
                "AH -0.25": ["AH 0", "AH -0.5"],
                "AH 0": ["AH +0.25", "AH -0.25"],
                "AH +0.25": ["AH +0.5", "AH 0"],
                "AH +0.5": ["AH +0.75", "AH +0.25"],
                "AH +0.75": ["AH +1.0", "AH +0.5"],
                "AH +1.0": ["AH +0.75", "AH +1.25"]
            }
            
            alternatives = handicap_alternatives.get(data['handicap'], [])
            
            for i, alt in enumerate(alternatives[:2], 1):
                st.write(f"   {i}. **Bet on Team A with {alt}** (closest alternative)")
            
            st.info("""
            ℹ️ **Remember**: If you see "AH -0.5 for both teams", that's normal!
            - **Team A AH -0.5** = Bet on Team A (must win by >0.5)
            - **Team B AH +0.5** = Bet on Team B (opponent's side)
            
            **Always stick with Team A** based on your model analysis.
            """)
            
            st.markdown(f"""
            **Option 2: Try different bookmakers**
            - Some bookmakers offer more handicap variations
            - Check Bet365, 1xBet, Betfair, Pinnacle, etc.
            
            **Option 3: Use alternative odds**
            If you find one of the alternatives, calculate the new EV below
            
            **Option 4: SKIP ❌**
            If no good odds available → **SKIP THIS BET**
            - Don't force a bad bet just to place something
            - Wait for better opportunities
            - Discipline first! 💪
            """)
            
            st.divider()
            st.subheader("🔄 Calculate EV for Alternative Handicap")
            
            col_alt_hc, col_alt_odds = st.columns(2)
            with col_alt_hc:
                alt_handicap = st.selectbox(
                    "Alternative Handicap",
                    alternatives + ["Other"],
                    key="alt_handicap_select"
                )
            
            with col_alt_odds:
                alt_bookmaker_odds = st.number_input(
                    "Odds for this handicap",
                    min_value=1.01,
                    value=1.90,
                    step=0.01,
                    key="alt_odds_input"
                )
            
            if st.button("Calculate EV for Alternative"):
                ev_alt = asian_handicap_ev(alt_handicap, data['win_p'], data['draw_p'], data['loss_p'], alt_bookmaker_odds)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Fair Odds", data['fair_odds'])
                with col2:
                    st.metric("Bookmaker Odds", alt_bookmaker_odds)
                with col3:
                    st.metric("EV (Alternative)", ev_alt, delta="Positive" if ev_alt > 0 else "Negative")
                
                if ev_alt and ev_alt > 0.03:
                    st.success(f"✅ Alternative looks good! EV: {ev_alt}")
                else:
                    st.warning("⚠️ Alternative also has low EV - Consider SKIPPING")
            

    # ---------------- SETTLEMENT ----------------
    st.header("Settle Last Bet")
    result = st.selectbox("Result", ["WIN", "HALF WIN", "PUSH", "HALF LOSS", "LOSS"], key="settle_result")

    if st.button("Settle Bet"):
        new_br, err = settle_last_bet(result)
        if err:
            st.warning(err)
        else:
            st.success(f"Bet settled. New bankroll: ₹{new_br}")

with tab2:
    st.header("Bet History")
    try:
        df = pd.read_csv("data/bets.csv")
        st.dataframe(df)
        
        # Bankroll chart
        if not df.empty and 'profit' in df.columns:
            df['cumulative_profit'] = df['profit'].cumsum()
            fig, ax = plt.subplots()
            ax.plot(df['cumulative_profit'])
            ax.set_title("Bankroll Over Time")
            ax.set_xlabel("Bet Number")
            ax.set_ylabel("Cumulative Profit")
            st.pyplot(fig)
    except:
        st.info("No bets yet.")

with tab3:
    st.header("Settings")
    st.success("✅ API Key is configured automatically and ready to use!")
    st.info("Your Football Data API key is securely stored in the app config.")

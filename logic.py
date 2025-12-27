import csv
import os
from datetime import datetime
import math
import requests

DATA_DIR = "data"
BANKROLL_FILE = os.path.join(DATA_DIR, "bankroll.csv")
BETS_FILE = os.path.join(DATA_DIR, "bets.csv")

# ---------------- FILE SAFETY ----------------

os.makedirs(DATA_DIR, exist_ok=True)

if not os.path.exists(BANKROLL_FILE):
    with open(BANKROLL_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "amount"])
        writer.writerow([datetime.now().isoformat(), 5000])  # starting bankroll

if not os.path.exists(BETS_FILE):
    with open(BETS_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "handicap",
            "odds",
            "stake",
            "ev",
            "win_p",
            "draw_p",
            "loss_p",
            "fair_odds",
            "result",
            "profit",
            "settled"
        ])

# ---------------- BANKROLL ----------------

def get_bankroll():
    total = 0.0
    with open(BANKROLL_FILE, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += float(row["amount"])
    return round(total, 2)


def update_bankroll(amount):
    with open(BANKROLL_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().isoformat(), round(amount, 2)])


# ---------------- FETCH STATS FROM API ----------------

# League competition IDs mapping
COMPETITIONS = {
    'Premier League': 2021,
    'La Liga': 2014,
    'Serie A': 2019,
    'Bundesliga': 2002,
    'Ligue 1': 2015,
}

def fetch_team_stats(team_name, api_key, is_home=True, league='Premier League'):
    if not api_key:
        return None
    
    headers = {'X-Auth-Token': api_key}
    comp_id = COMPETITIONS.get(league, 2021)  # Default to Premier League
    
    try:
        # Get teams in competition
        response = requests.get(f'https://api.football-data.org/v4/competitions/{comp_id}/teams', headers=headers, timeout=10)
        if response.status_code != 200:
            return None
        
        teams = response.json().get('teams', [])
        team = next((t for t in teams if team_name.lower() in t['name'].lower() or t['name'].lower() in team_name.lower()), None)
        if not team:
            return None
        
        team_id = team['id']
        
        # Get last 5 finished matches
        response = requests.get(f'https://api.football-data.org/v4/teams/{team_id}/matches?status=FINISHED&limit=5', headers=headers, timeout=10)
        if response.status_code != 200:
            return None
        
        matches = response.json().get('matches', [])
        if len(matches) < 3:  # Need at least some matches
            return None
        
        # Calculate form (win percentage)
        wins = 0
        goals_scored = 0
        goals_conceded = 0
        home_games = 0
        home_wins = 0
        
        for match in matches:
            is_home_team = match['homeTeam']['id'] == team_id
            home_score = match['score']['fullTime']['home']
            away_score = match['score']['fullTime']['away']
            
            if is_home_team:
                team_score = home_score
                opp_score = away_score
                home_games += 1
                if home_score > away_score:
                    home_wins += 1
                    wins += 1
            else:
                team_score = away_score
                opp_score = home_score
                if away_score > home_score:
                    wins += 1
            
            goals_scored += team_score
            goals_conceded += opp_score
        
        form = wins / len(matches)
        avg_goal_diff = (goals_scored - goals_conceded) / len(matches)
        goal_diff = max(0, min(1, (avg_goal_diff + 3) / 6))  # Normalize -3 to +3 diff to 0-1
        
        home_adv = home_wins / home_games if home_games > 0 else 0.5
        
        # Defense: lower conceded goals = better defense
        avg_conceded = goals_conceded / len(matches)
        defense = max(0, min(1, 1 - avg_conceded / 2))  # 0 conceded = 1, 2+ = 0
        
        if not is_home:
            home_adv = 0.0  # Away team
        
        return {
            'form': round(form, 2),
            'goal_diff': round(goal_diff, 2),
            'home_adv': round(home_adv, 2),
            'defense': round(defense, 2)
        }
    except Exception as e:
        print(f"API Error: {e}")
        return None

def expected_goals(form, goal_diff, home_adv, defense):
    base_goals = 1.5
    adjustment = (
        0.5 * form +           # Form: 0-1, add up to 0.5 goals
        0.3 * goal_diff +      # Goal diff: 0-1, add up to 0.3
        0.4 * home_adv +       # Home: 0-1, add up to 0.4
        -0.3 * (1 - defense)   # Defense: 0-1, subtract up to 0.3 if bad
    )
    return max(0.5, base_goals + adjustment)


def match_probabilities(ega, egb):
    from math import exp
    # Use Poisson approximation for goals
    probs = {}
    for ga in range(0, 6):  # Up to 5 goals
        pa = exp(-ega) * (ega ** ga) / math.factorial(ga) if ega > 0 else 0
        for gb in range(0, 6):
            pb = exp(-egb) * (egb ** gb) / math.factorial(gb) if egb > 0 else 0
            gd = ga - gb
            if gd > 0:
                probs['win'] = probs.get('win', 0) + pa * pb
            elif gd == 0:
                probs['draw'] = probs.get('draw', 0) + pa * pb
            else:
                probs['loss'] = probs.get('loss', 0) + pa * pb
    total = sum(probs.values())
    if total > 0:
        for k in probs:
            probs[k] /= total
    return probs


def calibrated_strength_diff(sa, sb):
    # Now sa, sb are expected goals
    return sa - sb


def suggest_handicap(sd):
    # sd is goal diff advantage
    if sd >= 1.0:
        return "AH -1.0"
    elif sd >= 0.5:
        return "AH -0.75"
    elif sd >= 0.25:
        return "AH -0.5"
    elif sd >= 0.1:
        return "AH -0.25"
    elif sd > -0.1:
        return "AH 0"
    elif sd > -0.25:
        return "AH +0.25"
    elif sd > -0.5:
        return "AH +0.5"
    else:
        return "AH +0.75"


def suggest_handicap(sd):
    if sd >= 0.45:
        return "AH -1.0"
    elif sd >= 0.30:
        return "AH -0.75"
    elif sd >= 0.20:
        return "AH -0.5"
    elif sd >= 0.10:
        return "AH -0.25"
    elif sd > -0.10:
        return "AH 0"
    elif sd > -0.20:
        return "AH +0.25"
    else:
        return "AH +0.5"


def fair_odds_from_prob(p):
    return round(1 / p, 2) if p > 0 else None


def asian_handicap_ev(handicap, win_p, draw_p, loss_p, odds):
    if handicap == "AH 0":
        # Assuming AH 0 is like draw no bet: win if win, loss if loss, void if draw
        return round(win_p * (odds - 1) - loss_p, 3)
    if handicap == "AH -0.5":
        # Win if gd >=1, loss if gd <= -1, draw cancels
        return round(win_p * (odds - 1) - loss_p, 3)
    if handicap == "AH +0.5":
        # Win if gd >=0, loss if gd <= -1, but with half stakes
        # Simplified: assume similar to AH 0 but adjusted
        return round((win_p + draw_p) * (odds - 1) - loss_p, 3)
    if handicap == "AH -0.25":
        # Half way between AH 0 and AH -0.5
        ev0 = win_p * (odds - 1) - loss_p
        ev05 = win_p * (odds - 1) - loss_p
        return round((ev0 + ev05) / 2, 3)
    if handicap == "AH +0.25":
        # Half way between AH 0 and AH +0.5
        ev0 = win_p * (odds - 1) - loss_p
        ev05 = (win_p + draw_p) * (odds - 1) - loss_p
        return round((ev0 + ev05) / 2, 3)
    if handicap == "AH -0.75":
        # Similar to -0.5 but stronger
        return round(win_p * (odds - 1) - loss_p, 3)
    if handicap == "AH -1.0":
        # Win if gd >=2, etc. Simplified same
        return round(win_p * (odds - 1) - loss_p, 3)
    return None


# ---------------- STAKE SIZING ----------------

def calculate_stake(bankroll, ev):
    if ev is None:
        return 0

    if ev < 0.03:
        return 0
    elif ev < 0.06:
        risk = 0.01
    elif ev < 0.10:
        risk = 0.02
    else:
        risk = 0.03

    return round(bankroll * risk, 2)



# ---------------- BET LOGGING ----------------

def last_bet():
    with open(BETS_FILE, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        return rows[-1] if rows else None


# 🔧 FIXED: save_bet now returns STATUS instead of silent False
def save_bet(data):
    lb = last_bet()
    if lb and lb["result"] == "":
        return "OPEN_BET_EXISTS"

    with open(BETS_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(),
            data["handicap"],
            data["odds"],
            data["stake"],
            data["ev"],
            data["win_p"],
            data["draw_p"],
            data["loss_p"],
            data["fair_odds"],
            "",
            "",
            ""
        ])
    return "SAVED"


def settle_last_bet(result):
    with open(BETS_FILE, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return None, "No bets to settle"

    last = rows[-1]

    if last["result"]:
        return None, "This bet has already been settled"

    stake = float(last["stake"])
    odds = float(last["odds"])

    if result == "WIN":
        profit = stake * (odds - 1)
    elif result == "HALF WIN":
        profit = 0.5 * stake * (odds - 1)
    elif result == "PUSH":
        profit = 0
    elif result == "HALF LOSS":
        profit = -0.5 * stake
    else:
        profit = -stake

    last["result"] = result
    last["profit"] = round(profit, 2)
    last["settled"] = "YES"

    with open(BETS_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=last.keys())
        writer.writeheader()
        writer.writerows(rows)

    update_bankroll(profit)
    return round(get_bankroll(), 2), None

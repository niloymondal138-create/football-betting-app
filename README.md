# Football Asian Handicap Betting Assistant

A Streamlit web application for analyzing football matches and identifying positive Expected Value (EV) betting opportunities on Asian Handicap markets.

## Features

✓ **Live Team Statistics** - Fetch real-time data via Football Data API
✓ **Expected Goals Calculation** - xG-based match analysis
✓ **Probability Modeling** - Poisson distribution for match outcomes
✓ **Asian Handicap Suggestions** - Automatic handicap recommendations
✓ **EV Analysis** - Identifies profitable betting opportunities
✓ **Bankroll Management** - Stake sizing based on Expected Value
✓ **Bet Tracking** - Complete history with P&L analysis
✓ **Machine Learning** - Random Forest predictions

## Tech Stack

- **Python 3.9+**
- **Streamlit** - Interactive web interface
- **Pandas** - Data processing
- **scikit-learn** - Machine Learning models
- **Requests** - API integration
- **Football Data API** - Live sports data

## Installation
```bash
git clone https://github.com/niloymodal138-create/football-betting-app.git
cd football-betting-app
pip install -r requirements.txt
streamlit run app.py
```

## How It Works

1. **Input Teams** - Enter team names for home/away match
2. **Fetch Statistics** - Automatically fetch live team data
3. **Analyze Match** - Calculate expected goals and win probabilities
4. **Enter Bookmaker Odds** - Input the odds from your betting site
5. **Check EV** - System calculates Expected Value
6. **Place Bet** - Only place bet if EV > 3%

## Key Functions

- **expected_goals()** - Calculate expected goals
- **match_probabilities()** - Poisson-based outcome probabilities
- **asian_handicap_ev()** - Calculate Expected Value
- **calculate_stake()** - Determine optimal bet size
- **fetch_team_stats()** - Fetch live API data

## Philosophy

**Discipline First, No Emotion**

- Only bet when EV > 3%
- Never chase losses
- Proper bankroll management
- Data-driven decisions only

## Author

Niloy Mondal - 3rd Year Engineering Student

## License

Open source

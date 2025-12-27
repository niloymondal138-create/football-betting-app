import streamlit.web.cli as stcli
import sys

sys.argv = ["streamlit", "run", "app.py"]
stcli.main()

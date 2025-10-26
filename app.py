import streamlit.web.cli as stcli
import sys

if __name__ == "__main__":
    sys.argv = ["streamlit", "run", "/home/username/app.py", "--server.port=8000", "--server.address=0.0.0.0"]
    sys.exit(stcli.main())

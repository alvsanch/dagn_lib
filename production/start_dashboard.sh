#!/bin/bash
source ~/.bashrc
cd /home/alvar/dagn_lib/production
exec /home/alvar/venv_tesis/bin/python -m streamlit run dashboard.py \
    --server.port 8501 \
    --server.fileWatcherType none

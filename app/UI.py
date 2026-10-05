import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import xgboost as xgb
from sqlalchemy import create_engine, text
from datetime import date, timedelta
import warnings
import joblib
import tensorflow as tf
import os
from flask_caching import Cache
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import base64
import logging

from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from flask import Response

warnings.filterwarnings('ignore')

# Configuration du logging
if not os.path.exists('app_logs'):
    os.makedirs('app_logs')
logging.basicConfig(
    filename='app_logs/app_digital_twin.log', 
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.info("🚀 Démarrage de l'interface du Jumeau Numérique")

# ===========================================================
# CONFIGURATION DYNAMIQUE DES MODÈLES
# ===========================================================
MODELS_CONFIG = {
    "physiciens": {
        "LSTM 128u Attention": "lstm_128u_Attention",
        "LSTM 128u NoAttention": "lstm_128u_NoAttention",
        "LSTM 64u Attention": "lstm_64u_Attention",
        "LSTM 64u NoAttention": "lstm_64u_NoAttention"
    },
    "energetiques": {
        "XGBoost Depth 3": "modele_financier_depth_3.json",
        "XGBoost Depth 5": "modele_financier_depth_5.json",
        "XGBoost Depth 7": "modele_financier_depth_7.json",
        "XGBoost Depth 9": "modele_financier_depth_9.json"
    }
}

DB_URL   = "sqlite:///axima_poc.db"
FEATURES = ['T_ext', 'T_int', 'Etat_Compresseur', 'T_int_lag_1',
            'day_sin', 'day_cos', 'hour_sin', 'hour_cos']
TIME_STEPS   = 12
CONSIGNE     = 7.0
HYSTERESIS_H = 0.5
HYSTERESIS_L = 0.3
INTERVAL_MS  = 600

# ===========================================================
# COULEURS PRINCIPALES & HARMONISÉES
# ===========================================================
couleur_ia  = "#5BD4F2"  # Cyan doux
couleur_opt = "#2ECC71"  # Vert frais, émeraude, non agressif

# ===========================================================
# STYLE UI — DASHBOARD PROFESSIONNEL
# ===========================================================
APP_BG = "#050505"         # Noir beaucoup plus dense et profond
CARD_BG = "#0D1117"        # Noir ardoise très sombre pour les cartes
CARD_BG_SOFT = "#161B22"
BORDER_SOFT = "rgba(255,255,255,0.08)"
TEXT_MUTED = "#A8B3C7"
TEXT_LIGHT = "#F4F7FB"
ACCENT_ORANGE = "#F8A706"

STYLE_PAGE = {
    "background": f"linear-gradient(135deg, {APP_BG} 0%, #0a0a0a 45%, #000000 100%)",
    "minHeight": "100vh",
    "padding": "18px 22px 28px 22px",
    "fontFamily": "Inter, Segoe UI, Roboto, Arial, sans-serif"
}

STYLE_CARD = {
    "backgroundColor": CARD_BG,
    "border": f"1px solid {BORDER_SOFT}",
    "borderRadius": "18px",
    "boxShadow": "0 12px 35px rgba(0,0,0,0.40)"
}

STYLE_CARD_HEADER = {
    "backgroundColor": "rgba(255,255,255,0.03)",
    "borderBottom": f"1px solid {BORDER_SOFT}",
    "fontWeight": "700",
    "letterSpacing": "0.2px",
    "color": TEXT_LIGHT
}

STYLE_GRAPH_CARD = {
    "backgroundColor": CARD_BG,
    "border": f"1px solid {BORDER_SOFT}",
    "borderRadius": "18px",
    "padding": "10px",
    "boxShadow": "0 12px 35px rgba(0,0,0,0.30)",
    "marginBottom": "14px"
}

STYLE_TABLE_HEADER = {
    'backgroundColor': CARD_BG_SOFT,
    'color': TEXT_LIGHT,
    'fontWeight': 'bold',
    'textAlign': 'center',
    'border': '1px solid rgba(255,255,255,0.08)'
}

STYLE_TABLE_DATA = {
    'backgroundColor': CARD_BG,
    'color': '#EAF0F8',
    'border': '1px solid rgba(255,255,255,0.05)'
}

STYLE_TABLE_CELL = {
    'textAlign': 'center',
    'padding': '10px',
    'fontFamily': 'Inter, Segoe UI, Arial',
    'fontSize': '13px'
}

STYLE_TABLE = {
    'overflowX': 'auto',
    'borderRadius': '16px',
    'border': '1px solid rgba(255,255,255,0.08)',
    'boxShadow': '0 12px 35px rgba(0,0,0,0.22)'
}

# Variables globales pour le moteur de calcul lourd
scaler_X = None
modele_physicien = None
modele_financier = None

print("=" * 50)
print("🌐 JUMEAU NUMÉRIQUE V7 — MLOps Dashboard & Data")
print("=" * 50)

# ===========================================================
# CHARGEMENT INITIAL & BDD
# ===========================================================
engine   = create_engine(DB_URL)
executor = ThreadPoolExecutor(max_workers=2)

with engine.begin() as conn:
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS optimisations_data (
            timestamp TIMESTAMP, t_int_opt FLOAT, etat_opt FLOAT,
            pwr_opt FLOAT, pente_

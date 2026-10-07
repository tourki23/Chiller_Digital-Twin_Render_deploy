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
        "XGBoost Depth 9": "modele_financier_

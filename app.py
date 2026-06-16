import streamlit as st
import joblib

from model_pipeline import (
    recommend_careers,
    get_skill_gaps,
    career_roadmaps
)
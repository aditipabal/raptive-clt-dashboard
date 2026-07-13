import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

# Set page configuration
st.set_page_config(
    page_title="Central Limit Theorem Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App Title & Description
st.title("📊 Central Limit Theorem Interactive Dashboard")
st.markdown(r"""
This interactive dashboard demonstrates the **Central Limit Theorem (CLT)** in action. 
The CLT is one of the most fundamental theorems in statistics. It states that, for any population with a finite mean $\mu$ and finite standard deviation $\sigma$, 
the sampling distribution of the sample mean ($\bar{X}$) approaches a normal distribution with mean $\mu$ and standard error $\sigma / \sqrt{n}$ as the sample size $n$ becomes large, 
**regardless of the shape of the original population distribution**.
""")

# Sidebar Navigation and Parameters
st.sidebar.header("🔧 Simulation Controls")

# Choose a distribution
dist_choice = st.sidebar.selectbox(
    "1. Select Population Distribution",
    ["Normal", "Uniform", "Exponential", "Poisson"]
)

# Dynamic parameter selection based on distribution
st.sidebar.subheader("Adjust Distribution Parameters")

if dist_choice == "Normal":
    mu = st.sidebar.slider("Mean (μ)", min_value=-50.0, max_value=50.0, value=0.0, step=0.5)
    sigma = st.sidebar.slider("Standard Deviation (σ)", min_value=0.1, max_value=20.0, value=1.0, step=0.1)
    
    pop_mean = mu
    pop_std = sigma
    dist_label = f"Normal(μ={mu}, σ={sigma})"
    
elif dist_choice == "Uniform":
    a = st.sidebar.slider("Minimum (a)", min_value=-50.0, max_value=50.0, value=0.0, step=0.5)
    b = st.sidebar.slider("Maximum (b)", min_value=a + 0.5, max_value=a + 100.0, value=a + 10.0, step=0.5)
    
    pop_mean = (a + b) / 2
    pop_std = np.sqrt((b - a)**2 / 12)
    dist_label = f"Uniform(a={a}, b={b})"
    
elif dist_choice == "Exponential":
    beta = st.sidebar.slider("Scale (β = 1/λ)", min_value=0.1, max_value=20.0, value=2.0, step=0.1)
    
    pop_mean = beta
    pop_std = beta
    dist_label = f"Exponential(β={beta}, λ={round(1/beta, 3)})"
    
elif dist_choice == "Poisson":
    lam = st.sidebar.slider("Rate (λ)", min_value=0.5, max_value=30.0, value=4.0, step=0.5)
    
    pop_mean = lam
    pop_std = np.sqrt(lam)
    dist_label = f"Poisson(λ={lam})"

st.sidebar.subheader("Sampling Parameters")

# Sample size n
n = st.sidebar.slider(
    "2. Sample Size (n)",
    min_value=1,
    max_value=1000,
    value=30,
    help="Number of observations in each individual sample. As n increases, the sampling distribution becomes more Normal."
)

# Number of simulations N
N = st.sidebar.slider(
    "3. Number of Simulations (N)",
    min_value=100,
    max_value=10000,
    value=2000,
    step=100,
    help="Number of times we repeat the sampling process to estimate the sampling distribution."
)

# Option for reproducibility
set_seed = st.sidebar.checkbox("Set Random Seed for Reproducibility", value=True)
if set_seed:
    seed_val = st.sidebar.number_input("Seed Value", min_value=1, max_value=100000, value=42, step=1)
    np.random.seed(seed_val)
else:
    np.random.seed(None)


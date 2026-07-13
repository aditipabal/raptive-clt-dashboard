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

# Run Simulation
@st.cache_data
def run_simulation(dist_name, n, N, params, seed=None):
    if seed is not None:
        np.random.seed(seed)
        
    if dist_name == "Normal":
        mu, sigma = params
        samples = np.random.normal(loc=mu, scale=sigma, size=(N, n))
    elif dist_name == "Uniform":
        a, b = params
        samples = np.random.uniform(low=a, high=b, size=(N, n))
    elif dist_name == "Exponential":
        beta = params[0]
        samples = np.random.exponential(scale=beta, size=(N, n))
    elif dist_name == "Poisson":
        lam = params[0]
        samples = np.random.poisson(lam=lam, size=(N, n))
        
    sample_means = np.mean(samples, axis=1)
    return samples, sample_means

# Pack parameters
if dist_choice == "Normal":
    params = (mu, sigma)
elif dist_choice == "Uniform":
    params = (a, b)
elif dist_choice == "Exponential":
    params = (beta,)
elif dist_choice == "Poisson":
    params = (lam,)

seed_to_pass = seed_val if set_seed else None
samples, sample_means = run_simulation(dist_choice, n, N, params, seed_to_pass)

# Display Summary Statistics and Metrics
st.subheader("📈 Simulation Results & Summary Statistics")

col1, col2, col3 = st.columns(3)

empirical_mean = np.mean(sample_means)
empirical_se = np.std(sample_means, ddof=1)
theoretical_se = pop_std / np.sqrt(n)

with col1:
    st.metric(
        label="Population Mean (μ)",
        value=f"{pop_mean:.4f}"
    )
    st.metric(
        label="Empirical Mean of Sample Means (X̄)",
        value=f"{empirical_mean:.4f}",
        delta=f"{empirical_mean - pop_mean:.4f}",
        delta_color="off"
    )

with col2:
    st.metric(
        label="Population Std Dev (σ)",
        value=f"{pop_std:.4f}"
    )
    st.metric(
        label="Theoretical Standard Error (σ / √n)",
        value=f"{theoretical_se:.4f}"
    )

with col3:
    st.metric(
        label="Empirical Standard Error (SE)",
        value=f"{empirical_se:.4f}",
        delta=f"{empirical_se - theoretical_se:.4f}",
        delta_color="off"
    )

# Visualizations
st.write("---")
st.subheader("📊 Visualizing the Distributions")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Define colors for styling
color_pop = "#2ca02c"  # Green
color_samp = "#1f77b4"  # Blue
color_norm = "#d62728"  # Red for theoretical Normal curve

# 1. Plotting the Population Distribution
if dist_choice == "Normal":
    x_pop = np.linspace(pop_mean - 4*pop_std, pop_mean + 4*pop_std, 1000)
    y_pop = stats.norm.pdf(x_pop, loc=mu, scale=sigma)
    ax1.plot(x_pop, y_pop, color=color_pop, lw=3, label="Theoretical PDF")
    ax1.fill_between(x_pop, y_pop, color=color_pop, alpha=0.3)
    
elif dist_choice == "Uniform":
    x_pop = np.linspace(a - (b-a)*0.1, b + (b-a)*0.1, 1000)
    y_pop = stats.uniform.pdf(x_pop, loc=a, scale=b-a)
    ax1.plot(x_pop, y_pop, color=color_pop, lw=3, label="Theoretical PDF")
    ax1.fill_between(x_pop, y_pop, color=color_pop, alpha=0.3)
    
elif dist_choice == "Exponential":
    x_pop = np.linspace(0, 5*pop_mean, 1000)
    y_pop = stats.expon.pdf(x_pop, scale=beta)
    ax1.plot(x_pop, y_pop, color=color_pop, lw=3, label="Theoretical PDF")
    ax1.fill_between(x_pop, y_pop, color=color_pop, alpha=0.3)
    
elif dist_choice == "Poisson":
    # Since Poisson is discrete, we use a bar plot for PMF
    max_k = int(max(stats.poisson.ppf(0.999, mu=lam), 10))
    x_pop = np.arange(0, max_k + 1)
    y_pop = stats.poisson.pmf(x_pop, mu=lam)
    ax1.bar(x_pop, y_pop, color=color_pop, alpha=0.6, edgecolor='black', label="Theoretical PMF")
    ax1.set_xticks(x_pop[::max(1, max_k//10)])

ax1.set_title(f"Population Distribution: {dist_label}", fontsize=14, fontweight='bold')
ax1.set_xlabel("Value", fontsize=12)
ax1.set_ylabel("Probability Density/Mass", fontsize=12)
ax1.grid(True, linestyle="--", alpha=0.5)

# 2. Plotting the Sampling Distribution of the Sample Means
# Calculate standard error
se = pop_std / np.sqrt(n)

# Plot empirical histogram of sample means
count, bins, ignored = ax2.hist(
    sample_means, 
    bins=50, 
    density=True, 
    color=color_samp, 
    alpha=0.6, 
    edgecolor="black", 
    label="Simulated Sample Means"
)

# Overlay theoretical Normal distribution predicted by the CLT
x_clt = np.linspace(min(sample_means), max(sample_means), 1000)
y_clt = stats.norm.pdf(x_clt, loc=pop_mean, scale=se)
ax2.plot(x_clt, y_clt, color=color_norm, lw=3, linestyle="--", label="CLT Theoretical Curve\nNormal(μ, σ/√n)")

ax2.set_title(f"Sampling Distribution of Sample Means (n={n}, N={N})", fontsize=14, fontweight='bold')
ax2.set_xlabel("Sample Mean (X̄)", fontsize=12)
ax2.set_ylabel("Probability Density", fontsize=12)
ax2.grid(True, linestyle="--", alpha=0.5)

# Add a vertical line for the mean on both plots
ax1.axvline(pop_mean, color="black", linestyle=":", lw=2, label=f"Mean (μ={pop_mean:.2f})")
ax1.legend()
ax2.axvline(pop_mean, color="black", linestyle=":", lw=2, label=f"Mean (μ={pop_mean:.2f})")
ax2.legend()

# Display the matplotlib figure in Streamlit
st.pyplot(fig)

# Comparative summary statistics table
st.write("### Detailed Statistics Comparison")
stats_data = {
    "Metric": [
        "Mean",
        "Standard Deviation / Std Error"
    ],
    "Population (Theoretical)": [
        f"{pop_mean:.5f}",
        f"{pop_std:.5f}"
    ],
    "Sampling Distribution (Theoretical CLT)": [
        f"{pop_mean:.5f}",
        f"{theoretical_se:.5f}"
    ],
    "Sampling Distribution (Empirical Simulation)": [
        f"{empirical_mean:.5f}",
        f"{empirical_se:.5f}"
    ]
}
stats_df = pd.DataFrame(stats_data)
st.table(stats_df)

# Educational Explanation Section
st.write("---")
st.subheader("📖 What is the Central Limit Theorem?")

st.markdown(r"""
### Key Takeaways of the Central Limit Theorem:

1. **The Shape of the Sampling Distribution**:
   As the sample size $n$ increases, the sampling distribution of the sample mean ($\bar{{X}}$) approaches a **Normal Distribution**, regardless of the shape of the population distribution (even if it's heavily skewed like the Exponential distribution or discrete like the Poisson distribution).
   
2. **The Center of the Sampling Distribution**:
   The mean of the sampling distribution ($\mu_{{\bar{{X}}}}$) is equal to the mean of the population ($\mu$):
   $$\mu_{{\bar{{X}}}} = \mu$$
   In our simulation:
   - Theoretical Population Mean ($\mu$): **{pop_mean:.4f}**
   - Simulated Sample Mean ($\bar{{X}}$): **{empirical_mean:.4f}** (Differs by only **{mean_diff:.4f}**!)

3. **The Spread of the Sampling Distribution**:
   The standard deviation of the sampling distribution, known as the **Standard Error** ($\sigma_{{\bar{{X}}}}$), is smaller than the population standard deviation ($\sigma$) and is given by:
   $$\sigma_{{\bar{{X}}}} = \frac{{\sigma}}{{\sqrt{{n}}}}$$
   As $n$ increases, the standard error decreases, meaning the sample means cluster more tightly around the true population mean.
   In our simulation:
   - Theoretical Standard Error ($\sigma / \sqrt{{n}}$): **{theoretical_se:.4f}**
   - Simulated Standard Error: **{empirical_se:.4f}** (Differs by only **{se_diff:.4f}**!)

### Observe the CLT in Action:
- Set **Sample Size ($n$)** to a small value (e.g., $n = 2$ or $n = 5$) and select **Exponential** or **Poisson**. You will notice that the sampling distribution is still somewhat skewed or non-normal.
- Now, increase **Sample Size ($n$)** to $30$ or more. You will see the sampling distribution transform into a beautiful, symmetric bell curve, perfectly matching the overlaid **CLT Theoretical Curve (Red dashed line)**!
""".format(
    pop_mean=pop_mean,
    empirical_mean=empirical_mean,
    mean_diff=abs(empirical_mean - pop_mean),
    theoretical_se=theoretical_se,
    empirical_se=empirical_se,
    se_diff=abs(empirical_se - theoretical_se),
    n=n
))

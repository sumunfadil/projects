import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st
from model import BlackScholes
from plotly.subplots import make_subplots 

st.title("European options pricing dashboard")
st.sidebar.header("Option Parameters")
spot_price = st.sidebar.slider("Spot price (S)", 50, 150, 100)
strike_price = st.sidebar.slider("Strike price (K)", 50, 150, 100)
time_to_maturity = st.sidebar.slider("Time to maturity (T)", 0.1, 2.0, 1.0, 0.1)
risk_free_rate = st.sidebar.slider("Risk-free rate (r)", 0.1, 1.0, 0.2, 0.01)
volatility = st.sidebar.slider("Volatility ($\sigma$)", 0.1, 1.0, 0.2, 0.01)
option_type = st.sidebar.radio("Option type", ["Call", "Put"])

black_scholes_object = BlackScholes(spot_price, strike_price, time_to_maturity, risk_free_rate, volatility, option_type.lower())

data = pd.DataFrame({
        "Price": [round(black_scholes_object.price, 4)],
        "Delta": [round(black_scholes_object.delta, 4)],
        "Gamma": [round(black_scholes_object.gamma, 4)],
        "Vega": [round(black_scholes_object.vega, 4)],
        "Theta": [round(black_scholes_object.theta, 4)],
        "Rho": [round(black_scholes_object.rho, 4)]
    })

# Convert data to DataFrame
df = pd.DataFrame(data)

spot_prices = np.linspace(50,150,100)
strike_prices = np.linspace(50,150,100)
volatilities = np.linspace(0.1,1.0,100)
time_to_maturities = np.linspace(0.1,2.0,100)
risk_free_rates = np.linspace(0.1,1.0,100)

prices = [BlackScholes(spot_price, strike_price, time_to_maturity, risk_free_rate, \
                       volatility, option_type.lower()).price\
                        for spot_price in spot_prices]
deltas = [BlackScholes(spot_price, strike_price, time_to_maturity, risk_free_rate, \
                       volatility, option_type.lower()).delta\
                        for spot_price in spot_prices]
gammas = [BlackScholes(spot_price, strike_price, time_to_maturity, risk_free_rate, \
                       volatility, option_type.lower()).gamma\
                        for spot_price in spot_prices]
vegas = [BlackScholes(spot_price, strike_price, time_to_maturity, risk_free_rate, \
                       volatility, option_type.lower()).vega\
                        for volatility in volatilities]
thetas = [BlackScholes(spot_price, strike_price, time_to_maturity, risk_free_rate, \
                       volatility, option_type.lower()).theta\
                        for time_to_maturity in time_to_maturities]
rhos = [BlackScholes(spot_price, strike_price, time_to_maturity, risk_free_rate, \
                       volatility, option_type.lower()).rho\
                        for risk_free_rate in risk_free_rates]


custom_css = """
<style>
table {
    width: 100%;
    table-layout: fixed;
    border-collapse: collapse;
}
thead th {
    text-align: center;
    font-weight: bold;
    padding: 8px;
}
tbody td {
    text-align: center;
    padding: 8px;
}
</style>
"""
st.write("## Overview")
st.markdown(custom_css + df.to_html(index=False), unsafe_allow_html=True)

st.write("\n")

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))  

ax.plot(spot_prices, prices, linewidth=2, color="black")
ax.set_title(f"European {option_type.lower()} price")
ax.set_xlabel("Spot price (S)")
ax.set_ylabel(f"{option_type} price")
ax.grid(True, linestyle="--", alpha=0.9) 
ax.scatter(spot_price, black_scholes_object.price, color="black")
ax.axvline(x=black_scholes_object.K, color='black', linestyle='--', linewidth=1, zorder=4)


st.pyplot(fig)


st.write("## Common Greeks")


fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 12))  

plots = [
    lambda ax: ax.plot(spot_prices, deltas, linewidth=2, color="black"),
    lambda ax: ax.plot(spot_prices, gammas, linewidth=2, color="black"),
    lambda ax: ax.plot(volatilities, vegas, linewidth=2, color="black"),
    lambda ax: ax.plot(time_to_maturities, thetas, linewidth=2, color="black"),
    lambda ax: ax.plot(risk_free_rates, rhos, linewidth=2, color="black"),
]

xaxis_labels = ["Spot price (S)", "Spot price (S)", "Volatility ($\sigma$)", "Time to maturity (T)", \
                "Risk-free rate (r)"]
greeks = [r"Delta ($\Delta$)", r"gamma ($\Gamma$)", r"Vega ($\nu$)", r"Theta ($\Theta$)", r"Rho ($\rho$)"]
x_values = [spot_price, spot_price, volatility, time_to_maturity, risk_free_rate]
y_values = [black_scholes_object.delta, black_scholes_object.gamma, black_scholes_object.vega, \
            black_scholes_object.theta, black_scholes_object.rho]

for i, ax in enumerate(axes.flat):
    if i < len(plots):
        plots[i](ax)
        ax.set_title(f"{greeks[i]}")
        ax.set_xlabel(xaxis_labels[i])
        ax.set_ylabel(greeks[i])
        ax.grid(True, linestyle="--", alpha=0.9) 
        ax.scatter(x_values[i], y_values[i], color="black")

        if i < 2:
                ax.axvline(x=black_scholes_object.K, color='black', linestyle='--', linewidth=1, zorder=4) 

    else:
        ax.axis("off")  

fig.suptitle(f"European {option_type.lower()} option sensitivities", fontsize=16)

plt.tight_layout()
st.pyplot(fig)

st.write("## First-order Greeks")

epsilons = [BlackScholes(spot_price, strike_price, time_to_maturity, risk_free_rate, \
                       volatility, option_type.lower()).epsilon\
                        for risk_free_rate in risk_free_rates]
lambdas = [BlackScholes(spot_price, strike_price, time_to_maturity, risk_free_rate, \
                       volatility, option_type.lower()).lambda_\
                        for spot_price in spot_prices]

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))  

plots = [
    lambda ax: ax.plot(risk_free_rates, epsilons, linewidth=2, color="black"),
    lambda ax: ax.plot(spot_prices, lambdas, linewidth=2, color="black"),
]

xaxis_labels = ["Risk-free rate (r)", "Spot price (S)"]
greeks = [r"Epsilon ($\epsilon$)", r"Lambda ($\lambda$)"]
x_values = [risk_free_rate, spot_price]
y_values = [black_scholes_object.epsilon, black_scholes_object.lambda_]

for i, ax in enumerate(axes.flat):
        plots[i](ax)
        ax.set_title(f"{greeks[i]}")
        ax.set_xlabel(xaxis_labels[i])
        ax.set_ylabel(greeks[i])
        ax.grid(True, linestyle="--", alpha=0.9) 
        ax.scatter(x_values[i], y_values[i], color="black")
        if i == 1:
                ax.axvline(x=black_scholes_object.K, color='black', linestyle='--', linewidth=1, zorder=4) 


fig.suptitle(f"European {option_type.lower()} option sensitivities", fontsize=16)

plt.tight_layout()
st.pyplot(fig)




st.write("## Second-order Greeks")

vannas = [BlackScholes(spot_price, strike_price, time_to_maturity, risk_free_rate, \
                       volatility, option_type.lower()).vanna\
                        for spot_price in spot_prices]
charms = [BlackScholes(spot_price, strike_price, time_to_maturity, risk_free_rate, \
                       volatility, option_type.lower()).charm\
                        for spot_price in spot_prices]
vommas = [BlackScholes(spot_price, strike_price, time_to_maturity, risk_free_rate, \
                       volatility, option_type.lower()).vomma\
                        for volatility in volatilities]
veras = [BlackScholes(spot_price, strike_price, time_to_maturity, risk_free_rate, \
                       volatility, option_type.lower()).vera\
                        for volatility in volatilities]
vetas = [BlackScholes(spot_price, strike_price, time_to_maturity, risk_free_rate, \
                       volatility, option_type.lower()).veta\
                        for time_to_maturity in time_to_maturities]
omegas = [BlackScholes(spot_price, strike_price, time_to_maturity, risk_free_rate, \
                       volatility, option_type.lower()).omega\
                        for strike_price in strike_prices]


fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 12))  

plots = [
    lambda ax: ax.plot(spot_prices, vannas, linewidth=2, color="black"),
    lambda ax: ax.plot(spot_prices, charms, linewidth=2, color="black"),
    lambda ax: ax.plot(volatilities, vommas, linewidth=2, color="black"),
    lambda ax: ax.plot(volatilities, veras, linewidth=2, color="black"),
    lambda ax: ax.plot(time_to_maturities, vetas, linewidth=2, color="black"),
    lambda ax: ax.plot(strike_prices, omegas, linewidth=2, color="black"),
]

xaxis_labels = ["Spot price (S)", "Spot price (S)", "Volatility ($\sigma$)", "Volatility ($\sigma$)", "Time to maturity (T)", \
                "Strike prices (K)"]
greeks = ["Vanna", "Charm", "Vomma", "Vera", "Veta", "Omega"]
x_values = [spot_price, spot_price, volatility, volatility, time_to_maturity, strike_price]
y_values = [black_scholes_object.vanna, black_scholes_object.charm, black_scholes_object.vomma, \
            black_scholes_object.vera, black_scholes_object.veta, black_scholes_object.omega]

for i, ax in enumerate(axes.flat):
        
        plots[i](ax)
        ax.set_title(f"{greeks[i]}")
        ax.set_xlabel(xaxis_labels[i])
        ax.set_ylabel(greeks[i])
        ax.grid(True, linestyle="--", alpha=0.9) 
        ax.scatter(x_values[i], y_values[i], color="black")

        if i in [0,1]:
                ax.axvline(x=black_scholes_object.K, color='black', linestyle='--', linewidth=1, zorder=4) 
 

fig.suptitle(f"European {option_type.lower()} option sensitivities", fontsize=16)

plt.tight_layout()
st.pyplot(fig)



st.write("## Third-order Greeks")

speeds = [BlackScholes(spot_price, strike_price, time_to_maturity, risk_free_rate, \
                       volatility, option_type.lower()).speed\
                        for spot_price in spot_prices]
zommas = [BlackScholes(spot_price, strike_price, time_to_maturity, risk_free_rate, \
                       volatility, option_type.lower()).zomma\
                        for volatility in volatilities]
colors = [BlackScholes(spot_price, strike_price, time_to_maturity, risk_free_rate, \
                       volatility, option_type.lower()).color\
                        for time_to_maturity in time_to_maturities]
ultimas = [BlackScholes(spot_price, strike_price, time_to_maturity, risk_free_rate, \
                       volatility, option_type.lower()).ultima\
                        for volatility in volatilities]
parmicharmas = [BlackScholes(spot_price, strike_price, time_to_maturity, risk_free_rate, \
                       volatility, option_type.lower()).parmicharma\
                        for time_to_maturity in time_to_maturities]
dual_deltas = [BlackScholes(spot_price, strike_price, time_to_maturity, risk_free_rate, \
                       volatility, option_type.lower()).dual_delta\
                        for spot_price in spot_prices]
dual_gammas = [BlackScholes(spot_price, strike_price, time_to_maturity, risk_free_rate, \
                       volatility, option_type.lower()).dual_gamma\
                        for spot_price in spot_prices]


fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(10, 12))  

plots = [
    lambda ax: ax.plot(spot_prices, speeds, linewidth=2, color="black"),
    lambda ax: ax.plot(volatilities, zommas, linewidth=2, color="black"),
    lambda ax: ax.plot(time_to_maturities, colors, linewidth=2, color="black"),
    lambda ax: ax.plot(volatilities, ultimas, linewidth=2, color="black"),
    lambda ax: ax.plot(time_to_maturities, parmicharmas, linewidth=2, color="black"),
    lambda ax: ax.plot(spot_prices, dual_deltas, linewidth=2, color="black"),
    lambda ax: ax.plot(spot_prices, dual_gammas, linewidth=2, color="black"),
]

xaxis_labels = ["Spot price (S)", "Volatility ($\sigma$)", "Time to maturity (T)", "Volatility ($\sigma$)", \
                "Time to maturity (T)", "Spot price (S)", "Spot price (S)"]
greeks = ["Speed", "Zomma", "Color", "Ultima", "Parmicharma", "Dual delta", "Dual gamma"]
x_values = [spot_price, volatility, time_to_maturity, volatility, time_to_maturity, spot_price, spot_price]
y_values = [black_scholes_object.speed, black_scholes_object.zomma, black_scholes_object.color, \
            black_scholes_object.ultima, black_scholes_object.parmicharma, black_scholes_object.dual_delta,\
                black_scholes_object.dual_gamma]

for i, ax in enumerate(axes.flat):
    if i < len(plots):
        plots[i](ax)
        ax.set_title(f"{greeks[i]}")
        ax.set_xlabel(xaxis_labels[i])
        ax.set_ylabel(greeks[i])
        ax.grid(True, linestyle="--", alpha=0.9) 
        ax.scatter(x_values[i], y_values[i], color="black")

        if i in [0,5,6]:
                ax.axvline(x=black_scholes_object.K, color='black', linestyle='--', linewidth=1, zorder=4) 

    else:
        ax.axis("off")  

fig.suptitle(f"European {option_type.lower()} option sensitivities", fontsize=16)

plt.tight_layout()
st.pyplot(fig)


st.write("## Volatility smile")

# TODO: repurpose into a module e.g. dashboard module
# Implement implied volatility, Newton-Raphson
# Implement data gathering and cleaning using API
# 2 plots vs K and vs T


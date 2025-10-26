# -*- coding: utf-8 -*-
"""
Created in 2023

@author: Quant Galore
"""
import numpy as np
import pandas as pd

def binomial_option_price(S, K, T, r, sigma, n, option_type="call"):
    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    
    prices = np.zeros(n + 1)
    for i in range(n + 1):
        prices[i] = S * (u ** (n - i)) * (d ** i)
    
    option_values = np.zeros(n + 1)
    for i in range(n + 1):
        if option_type == "call":
            option_values[i] = max(0, prices[i] - K)
        else:
            option_values[i] = max(0, K - prices[i])
    
    for step in range(n - 1, -1, -1):
        for i in range(step + 1):
            option_values[i] = np.exp(-r * dt) * (p * option_values[i] + (1 - p) * option_values[i + 1])
    
    return option_values[0]

def bjerksund_stensland_greeks(S, K, T, r, sigma, option_type="call"):
    def phi(S, T, gamma, H, I, r, sigma):
        lambda_ = (-r + gamma * r + 0.5 * gamma * (gamma - 1) * sigma**2) * T
        d = -(np.log(S / H) + (r + (gamma - 0.5) * sigma**2) * T) / (sigma * np.sqrt(T))
        kappa = (2 * r + (gamma - 0.5) * sigma**2) / sigma**2
        return np.exp(lambda_) * (S ** gamma) * (1 - norm.cdf(d)) + (I / S) ** kappa * norm.cdf(d - 2 * np.log(I / S) / (sigma * np.sqrt(T)))

    from scipy.stats import norm
    delta = np.where(option_type == "call", phi(S, T, 1, K, K, r, sigma), phi(S, T, 1, K, K, r, sigma) - 1)
    gamma = phi(S, T, 2, K, K, r, sigma) / S
    theta = -(phi(S, T, 1, K, K, r, sigma) * (r + 0.5 * sigma**2) + phi(S, T, 0, K, K, r, sigma) * (r - 0.5 * sigma**2)) / 365
    vega = phi(S, T, 1, K, K, r, sigma) * np.sqrt(T) * norm.pdf((np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T)))
    return delta, gamma, theta, vega

def Binarizer(x):
    return 1 if x > 0 else 0

def return_proba(prediction_df):
    return prediction_df[["probability_0", "probability_1"]].max(axis=1)

def round_to_multiple(number, multiple):
    return multiple * round(number / multiple)
# 🧠 RL-Trader: Reinforcement Learning for Market Strategy with World Models

This is an experimental project exploring how Reinforcement Learning can be applied to stock trading, using real historical data from TradingView.

Inspired by the World Models approach, the goal is to teach an agent to trade a single asset — and eventually manage a portfolio — using learned representations of market dynamics.

> ⚠️ This project is for educational purposes only. Trade at your own risk.

<br>

## 🚀 Project Goals

- Learn reinforcement learning in practice
- Understand how an agent builds a profitable trading strategy
- Compare RL-driven decisions vs. traditional strategies
- Test portfolio-level decision-making and diversification

## 📦 Project Status

🟡 **Work in Progress** – Currently building and training the VAE

---

<br>

## 🧱 Architecture Overview

1. Data Collection
    - Historical market data sourced from TradingView
2. World Model Components
    - 📦 VAE (Variational Autoencoder): Encodes market state
    - 🔁 MDN-RNN (Mixture Density Network with RNN): Predicts future states
    - 🕹️ Controller: Trained with RL (e.g., PPO or CMA-ES) to take trading actions
3. RL Training Environment
    - Custom Gym-style environment for trading simulation
    - Handles realistic constraints like rejected buy/sell orders

## 🛠️ Technologies Used

- **Python** - core programming language
- **PyTorch** - deep learning framework for building VAE, RNN, and policy networks
- **Math** - uderstanding of Deel Learning, and Deep RL
- **TradingView** - historical market data source

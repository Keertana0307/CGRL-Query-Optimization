# CG-RL Query Optimization

This repository contains the implementation of a Causal-Graph Guided Reinforcement Learning (CG-RL) approach for optimizing join order selection in complex multi-join queries.

## Overview
Traditional query optimizers struggle with hidden data correlations and large join spaces. This project investigates whether reinforcement learning guided by causal relationships can improve query execution efficiency.

## Methods
- Reinforcement Learning: Double DQN with target network
- Baseline Models:
  - Logistic Regression
  - Decision Tree
  - Random Forest
- Environment: Synthetic database join environment with causal graph

## Evaluation Metrics
- Cumulative execution cost
- Learning convergence behaviour
- Join selection quality (Accuracy, Precision, Recall, F1-score)

## Results Summary
The CG-RL agent demonstrates the ability to learn cost-efficient join strategies over time and reduce execution cost, though supervised baseline models may outperform it in static classification metrics.

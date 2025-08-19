<img width="1706" height="869" alt="image" src="https://github.com/user-attachments/assets/6bdf6de8-84df-4b1b-b777-4eece0aa5137" />


# Adaptive Gait Optimization for Bio-Inspired Turtle Robot  

This repository contains code and simulation workflows for **adaptive gait control of a sea turtle–inspired robot** using **Hopf Central Pattern Generators (CPGs)** combined with **Bayesian Optimization**. The project investigates how AI-based optimization can improve locomotion efficiency, adaptability, and robustness across varying terrain conditions.  

## Features  
- **Central Pattern Generator (CPG) Control**  
  - Implements Hopf oscillator–based rhythmic control for multi-joint locomotion.  
  - Supports multiple gait patterns (synchronous, diagonal, turning).  

- **Bayesian Optimization Framework**  
  - Tunes CPG parameters (frequency, amplitude, coupling) for speed, energy efficiency, and cost of transport.  
  - Uses Gaussian Process (GP) surrogate models with exploration–exploitation balance.  

- **Simulation Environment**  
  - MuJoCo + PyChrono interfaces for testing gait dynamics.  
  - Warm-up phase implementation for oscillator stabilization.  
  - Real-time logging of speed, displacement, energy, and cost of transport.  

- **Adaptive Terrain Response**  
  - Integrates terrain classification with gait adaptation.  
  - Demonstrates robustness across sand, rock, and aquatic-like surfaces.  

# Neural Population Dynamics with Short-Term Plasticity

This repository contains a Python implementation of a dynamic neural model exploring the interaction between excitatory and inhibitory populations. The model specifically incorporates **Short-Term Depression (STD)** and **Short-Term Facilitation (STF)** to observe bifurcation behaviors and steady-state dynamics.

## Overview

The simulation models four coupled ordinary differential equations (ODEs):

* **$E$ & $I$**: Firing rates of excitatory and inhibitory populations.
* **$u$**: Facilitation variable (STF).
* **$x$**: Depression variable (STD).

The effective synaptic weight from excitatory to inhibitory neurons is dynamically scaled: $J_{ie} = J_{ie0} \cdot u \cdot x$.

## Mathematical Model

The dynamics are governed by the following system:

$$\tau_e \frac{dE}{dt} = -E + g(J_{ee}E - J_{ei}I + E_0)$$

$$\tau_i \frac{dI}{dt} = -I + g(J_{ie}E - J_{ii}I + I_0)$$

$$\frac{du}{dt} = \frac{U - u}{\tau_f} + U \cdot E(1 - u)$$

$$\frac{dx}{dt} = \frac{1 - x}{\tau_r} - u \cdot x \cdot E$$

Where $g(x)$ is a linear-threshold activation function.

## Key Features

* **Bifurcation Analysis**: Automatically sweeps through values of $J_{ie0}$ to identify regions of stability and oscillation (limit cycles).
* **Numerical Integration**: Uses the Runge-Kutta 45 (`RK45`) method via `scipy.integrate.solve_ivp`.
* **Steady-State Solving**: Utilizes `fsolve` to trace the equilibrium path across parameters.
* **Visualization**: Generates four distinct plots to analyze the system's phase space and temporal behavior.

## Installation

Ensure you have the following Python libraries installed:

```bash
pip install numpy matplotlib scipy

```

## Usage

Run the main script to generate the analysis:

```bash
python neural_simulation.py

```

## Results & Visualization

The script generates the following insights:

1. **Bifurcation Diagram**: Shows the Max/Min firing rates of $E$ against the coupling strength $J_{ie0}$.
2. **Synaptic Strength vs. Firing Rate**: A phase-plane style plot showing how $J_{ie}$ evolves relative to $E$.
3. **Temporal Dynamics**: A time-series plot of the excitatory firing rate, focusing on the final steady or oscillatory state.
4. **Steady State Diagram**: A theoretical curve showing the ratio $J_{ie}/J_{ie0}$ as a function of the firing rate $E$.

---

### Parameters

* $\tau_e, \tau_i$: Time constants for $E$ and $I$ populations (10ms).
* $\tau_r, \tau_f$: Recovery and Facilitation time constants.
* $T$: Firing threshold (15 mV).
* $U$: Baseline release probability.

---

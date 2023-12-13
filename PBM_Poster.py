import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp

# Define constants
tau_e = 0.01  # s
tau_i = 0.01  # s
tau_r = 0.1   # s
tau_f = 1.5   # s
J_ee = 5      # mV/Hz
J_ii = 5      # mV/Hz
J_ei = 9      # mV/Hz
E0 = 19.0     # mV
I0 = 18.1     # mV
B = 0.5       # Hz/mV
T = 15        # mV
U = 0.01

# Time span
t_span = (0, 100)  # seconds

# Define threshold function g
def g(x):
    return np.where(x >= T, B * (x - T), 0)

# Define the differential function
def equations(y, J_ie0):
    E, I, u, x = y
    J_ie = J_ie0 * u * x 
    dEdt = (-E + g(J_ee * E - J_ei * I + E0)) / tau_e
    dIdt = (-I + g(J_ie * E - J_ii * I + I0)) / tau_i
    dudt = (-u + U) / tau_f + U * E * (1 - u)
    dxdt = (1 - x) / tau_r - u * x * E
    return [dEdt, dIdt, dudt, dxdt]

# Define the odefcn function
def odefcn(t, y, J_ie0):
    return equations(y, J_ie0)

# Initial conditions
initial_conditions = [0, 0, 1, 0]  # Default initial conditions

# Define values of J_ie0 to loop through
start_Jie0 = 28
end_Jie0 = 80
num_Jie0 = abs(start_Jie0 - end_Jie0) + 1
J_ie0_values = np.linspace(start_Jie0, end_Jie0, num_Jie0)

# Calculate steady states and perform numerical integration
steady_states = []
max_E_values = []
min_E_values = []

for J_ie0 in J_ie0_values:
    # Numerical integration
    solution = solve_ivp(odefcn, t_span, initial_conditions, args=(J_ie0,), method='RK45', dense_output=True)
    E = solution.y[0]

    # Calculate the maximum and minimum values of E
    max_E = np.max(E)
    min_E = np.min(E)
    max_E_values.append(max_E)
    min_E_values.append(min_E)

    # Update initial_conditions for the next round
    initial_conditions = solution.y[:, -1]
    steady_states.append((J_ie0, initial_conditions))

steady_state_at_max_Jie0 = solution.y[:, -1]

# Use fsolve to calculate the steady-state E value of J_ie0 from 80 to 10
start_Jie0 = 80
end_Jie0 = 14
num_Jie0 = abs(start_Jie0 - end_Jie0) + 1
Jie0_values_bifurcation = np.linspace(start_Jie0, end_Jie0, num_Jie0*10)
steady_state_E_values_bifurcation = []

for Jie0 in Jie0_values_bifurcation:
    # Solve for steady state variables
    E_ss, I_ss, u_ss, x_ss = fsolve(equations, steady_state_at_max_Jie0, args=(Jie0,))
    steady_state_at_max_Jie0 = [E_ss, I_ss, u_ss, x_ss]
    steady_state_E_values_bifurcation.append(E_ss)

# Plot results
fig1 = plt.figure(figsize=(10, 9))

plt.plot(J_ie0_values, max_E_values, label='Max E', color='blue', linewidth=2,  linestyle='--')
plt.plot(J_ie0_values, min_E_values, label='Min E', color='red', linewidth=2, linestyle='--')
plt.plot(Jie0_values_bifurcation, steady_state_E_values_bifurcation, label='Bifurcation', color='green', linewidth=2)
plt.xlim([0, 80])
plt.ylim([0, 120])
plt.grid(True, linestyle='--', alpha=0.5)
plt.xlabel('J_ie0', fontsize=14)
plt.ylabel('Ess', fontsize=14)
plt.title('Max and Min E vs. J_ie0 with Bifurcation', fontsize=16)
plt.legend(fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()

# Select a specific J_ie0 value
selected_J_ie0 = 40

# find selected_J_ie0 steady state
selected_steady_state = next(ss for J_ie0, ss in steady_states if J_ie0 == selected_J_ie0)

# Perform numerical integration using selected steady-state values as initial conditions
solution = solve_ivp(odefcn, t_span, selected_steady_state, args=(selected_J_ie0,), method='RK45', dense_output=True)

#Extract results
E = solution.y[0]
J_ie = selected_J_ie0 * solution.y[2] * solution.y[3]  # J_ie = J_ie0 * u * x


###### The Second figure: Synaptic Strength Ratio vs. Excitatory Firing Rate #####
fig2 = plt.figure(figsize=(10, 6))
plt.plot(E, J_ie, color='purple', linewidth=2)
plt.xlabel('E (Hz)', fontsize=14)
plt.ylabel('J_{ie}', fontsize=14)
plt.title('Synaptic Strength Ratio vs. Excitatory Firing Rate', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.5)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()


###### The third figure:  Create the third graph: E over time #####
fig3 = plt.figure(figsize=(10, 6))
plt.plot(solution.t, E, color='orange', linewidth=2)
plt.xlabel('t (ms)', fontsize=14)
plt.ylabel('E (Hz)', fontsize=14)
plt.xlim([90, 100])
plt.title('E over time', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.5)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()


####### The fourth figure: Steady State Diagram ########
fig4 = plt.figure(figsize=(10, 6))

E_values = np.linspace(0, 100, 300)  # Adjust the range and density as needed

# Calculate steady state values of u and x
u_steady = (U + U * E_values * tau_f) / (1 + U * E_values * tau_f)
x_steady = 1 / (u_steady * E_values * tau_r + 1)

# Calculate J_ie/J_ie0
J_ie_ratio = u_steady * x_steady

# Plotting
plt.plot(E_values, J_ie_ratio, color='cyan', linewidth=2)
plt.ylim([0, 0.2])
plt.xlabel('E', fontsize=14)
plt.ylabel('J_ie/J_ie0', fontsize=14)
plt.title('Steady State Diagram', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.5)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()

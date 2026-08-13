"""
Numerical verification of the DTFT and Parseval's identity
for x[n] = 1/n.

This script reproduces the numerical experiment and plots
presented in the EE603 DSP project report.
"""

import numpy as np
import matplotlib.pyplot as plt

# Numerical verification of the DTFT extension for x[n] = 1/n

# Sequence: x[n] = 1/n, n >= 1
N = 1000
n = np.arange(1, N + 1)
x = 1 / n

# Frequency grid
omega = np.linspace(-np.pi, np.pi, 2000)

# DTFT approximation using a finite partial sum
X = np.zeros_like(omega, dtype=complex)

for k in range(N):
    X += x[k] * np.exp(-1j * omega * n[k])

# Figure 1: Magnitude spectrum

plt.figure(figsize=(7, 4))
plt.plot(omega, np.abs(X),color='orange')
plt.xlabel(r'$\omega$ (rad)')
plt.ylabel(r'$|X(e^{j\omega})|$')
plt.title(r'Magnitude Spectrum $|X(e^{j\omega})|$ for $x[n]=1/n$')
plt.grid(True)
plt.tight_layout()
plt.show()

# Figure 2: Partial-sum energy convergence

partial_energy = np.cumsum(x**2)
true_energy = np.pi**2 / 6

plt.figure(figsize=(7, 4))
plt.plot(n, partial_energy, label='Partial-sum Energy',color='orange')
plt.axhline(true_energy, linestyle='--', label=r'True $\pi^2/6$',color='Red')

plt.xlabel('N (number of samples)')
plt.ylabel(r'$\sum_{n=1}^{N}|x[n]|^2$')
plt.title(r'Convergence of Energy for $x[n]=1/n$')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("Final partial-sum energy:", partial_energy[-1])
print("Theoretical energy pi^2/6:", true_energy)
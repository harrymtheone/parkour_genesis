import torch
import matplotlib.pyplot as plt

# Define range of d
d = torch.linspace(-2, 4, 500)

# Define function
f = torch.clamp(1 - 0.25 * torch.square(d - 1), min=0)

# Plot
plt.plot(d, f, label=r'$f(d) = \mathrm{clamp}(1 - \frac{1}{4}(d-1)^2, 0)$')
plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
plt.axvline(1, color='gray', linestyle='--', linewidth=0.8)
plt.title("Plot of f(d)")
plt.xlabel("d")
plt.ylabel("f(d)")
plt.legend()
plt.grid(True)
plt.show()

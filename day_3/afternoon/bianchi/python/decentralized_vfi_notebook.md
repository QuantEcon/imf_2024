
```{code-cell} ipython3
run decentralized_vfi_numba.py
```

```{code-cell} ipython3
model = create_overborrowing_model()
σ, η, β, ω, κ, R, b_grid, y_nodes, b_size, y_size, Q = model
H = solve_for_equilibrium(model, α=0.1, tol=0.005)
```

```{code-cell} ipython3
fig, ax = plt.subplots()
for i_y in range(y_size): 
    ax.plot(b_grid, H[:, i_y])
plt.show()
```

```{code-cell} ipython3
i_y=4
y_t, y_n = y_nodes[i_y]
b_bind = []   # Bp must be greater than constraint
b_max  = []   # Bp must be less than this value to get c >= 0
for i_B, B in enumerate(b_grid):
    Bp = H[i_B, i_y]
    C = R * B + y_t - Bp
    P = ((1 - ω) / ω) * (C / y_n)**(η + 1)
    b_bind.append(- κ * (P * y_n + y_t))
    b_max.append(R * B + y_t)
```

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.plot(b_grid, b_grid, '--', label='45')
ax.plot(b_grid, H[:, i_y], label='policy')
ax.plot(b_grid, b_bind, label='constraint')
ax.plot(b_grid, b_max, label='max assets')
ax.legend()
plt.show()
```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

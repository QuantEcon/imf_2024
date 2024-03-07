
```{code-cell}

```

```{code-cell}
model = create_overborrowing_model()
σ, η, β, ω, κ, R, b_grid, y_nodes, b_size, y_size, Q = model
#H = solve_for_equilibrium(model)
```

```{code-cell}
fig, ax = plt.subplots()
for i_y in range(y_size): 
    ax.plot(b_grid, H[:, i_y])
plt.show()
```

```{code-cell}
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

```{code-cell}
fig, ax = plt.subplots()
ax.plot(b_grid, b_grid, '--', label='45')
ax.plot(b_grid, H[:, i_y], label='policy')
ax.plot(b_grid, b_bind, label='constraint')
ax.plot(b_grid, b_max, label='max assets')
ax.legend()
plt.show()
```

```{code-cell}

```

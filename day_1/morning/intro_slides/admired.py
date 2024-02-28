import matplotlib.pyplot as plt
fig, ax = plt.subplots()
langs = ['Rust', 'Elixir', 'Python','Julia', 'C', 'R', 'Fortran', 'Matlab']
vals = [88.66, 73, 65.5, 62.7, 43.2, 39.0, 24.4, 18.3]
ax.bar(langs, vals)
ax.set_title("Language I want to work with next year")
plt.show()

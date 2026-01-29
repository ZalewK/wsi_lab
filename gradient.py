import numpy as np
import matplotlib.pyplot as plt

def ackley(x):
    x = np.atleast_1d(x)
    n = len(x)
    term1 = -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / n))
    term2 = -np.exp(np.sum(np.cos(2 * np.pi * x)) / n)
    return term1 + term2 + 20 + np.e

def get_gradient(x):
    x = np.atleast_1d(x)
    n = len(x)

    sum_sq = np.sum(x**2)
    sum_cos = np.sum(np.cos(2 * np.pi * x))

    t1 = -20 * np.exp(-0.2 * np.sqrt(sum_sq / n))
    if sum_sq == 0:
        grad_t1 = np.zeros_like(x)
    else:
        grad_t1 = t1 * (-0.2) * (1 / (2 * np.sqrt(sum_sq / n))) * (2 * x / n)

    t2 = -np.exp(sum_cos / n)
    grad_t2 = t2 * (1 / n) * (-2 * np.pi * np.sin(2 * np.pi * x))
    
    return grad_t1 + grad_t2

def gradient_descent(f, start_x, lr, momentum, tol=1e-7, max_iter=1000):
    x = np.array(start_x, dtype=float)
    v = np.zeros_like(x)
    history = [x.copy()]
    current_lr = lr
    best_f = f(x)

    print(f"Punkt startowy: {x}, Początkowe lr: {current_lr}\n")

    for i in range(max_iter):
        val = f(x)
        grad = get_gradient(x)
        grad_norm = np.linalg.norm(grad)

        if val < best_f - 1e-6:
            best_f = val
        
        if grad_norm < tol or val < 1e-8:
            print(f"\nOsiągnięto dno w iteracji {i}")
            print(f"Finalne x: {x}, f(x): {val:.10f}")
            return x, np.array(history)
        
        v = momentum * v - current_lr * grad
        x = x + v
        history.append(x.copy())
    
    print(f"\nLimit iteracji. Finalne f(x): {f(x):.6f}")
    return x, np.array(history)

def run_step_size_experiment(dim, start_pos, lr_range, momentum=0.0):
    results_lr = []
    results_iters = []

    for lr in lr_range:
        _, history = gradient_descent(ackley, start_pos, lr=lr, momentum=momentum, max_iter=1000)
        results_lr.append(lr)
        results_iters.append(len(history))
        
    return results_lr, results_iters

lr_range_1d = np.linspace(0.0001, 0.02, 200)
lrs_1d, iters_1d = run_step_size_experiment("1", [2.5], lr_range_1d)

plt.figure(figsize=(10, 6))
plt.scatter(lrs_1d, iters_1d, s=10)
plt.title("Wpływ rozmiaru kroku na znalezienie minimum funkcji Ackleya 1D")
plt.xlabel("Rozmiar kroku")
plt.ylabel("Liczba iteracji")
plt.grid(True)
plt.show()

lr_range_2d = np.linspace(0.0001, 0.04, 200)
lrs_2d, iters_2d = run_step_size_experiment("2", [2.5, 2.5], lr_range_2d)

plt.figure(figsize=(10, 6))
plt.scatter(lrs_2d, iters_2d, s=10)
plt.title("Wpływ rozmiaru kroku na znalezienie minimum funkcji Ackleya 2D")
plt.xlabel("Rozmiar kroku")
plt.ylabel("Liczba iteracji")
plt.grid(True)
plt.show()

start_1d = [2]
final_x_1d_no_mom, path_1d_no_mom = gradient_descent(ackley, start_1d, lr=0.02, momentum=0.0)
path_y_no_mom = [ackley([px]) for px in path_1d_no_mom]
final_x_1d_mom, path_1d_mom = gradient_descent(ackley, start_1d, lr=0.02, momentum=0.8)
path_y_mom = [ackley([px]) for px in path_1d_mom]

x_line = np.linspace(-5, 5, 400)
y_line = [ackley([xi]) for xi in x_line]

plt.figure(figsize=(10, 5))
plt.plot(x_line, y_line, 'k-', alpha=0.3, label='Funkcja Ackleya 1D')
plt.plot(path_1d_no_mom, path_y_no_mom, 'bo-', markersize=4, label='Ścieżka bez momentum')
plt.scatter(path_1d_no_mom[-1], ackley(path_1d_no_mom[-1]), color='blue', marker='x', s=150, label='Koniec bez momentum', zorder=5)
plt.plot(path_1d_mom, path_y_mom, 'ro-', markersize=4, label='Ścieżka z momentum')
plt.scatter(path_1d_mom[-1], ackley(path_1d_mom[-1]), color='red', marker='x', s=150, label='Koniec z momentum', zorder=5)
plt.title("Ścieżka algorytmu na funkcji Ackleya w 1D")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.show()

start = [2, 2]

final_x_no, path_no_mom = gradient_descent(ackley, start, lr=0.03, momentum=0.0)
final_x_with, path_with_mom = gradient_descent(ackley, start, lr=0.03, momentum=0.8)

plt.figure(figsize=(10, 8))
x_range = np.linspace(-4, 4, 100)
y_range = np.linspace(-4, 4, 100)
X, Y = np.meshgrid(x_range, y_range)
Z = np.array([ackley([xi, yi]) for xi, yi in zip(np.ravel(X), np.ravel(Y))]).reshape(X.shape)

plt.contour(X, Y, Z, levels=30, cmap='viridis', alpha=0.5)

plt.plot(path_no_mom[:, 0], path_no_mom[:, 1], 'b.-', label='Bez Momentum', alpha=0.6)
plt.scatter(path_no_mom[-1, 0], path_no_mom[-1, 1], color='blue', marker='x', s=150, label='Koniec bez momentum', zorder=5)
plt.plot(path_with_mom[:, 0], path_with_mom[:, 1], 'r.-', label='Z Momentum')
plt.scatter(path_with_mom[-1, 0], path_with_mom[-1, 1], color='red', marker='x', s=150, label='Koniec z momentum', zorder=5)

plt.title("Ścieżka algorytmu na funkcji Ackleya 2D")
plt.legend()
plt.show()
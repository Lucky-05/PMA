import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter 
from scipy.integrate import solve_ivp

def derivada5p(h, vel):
    n = len(vel)
    acc = np.zeros(n)
    for i in range(2, n - 2):
        acc[i] = (vel[i - 2] - 8 * vel[i - 1] + 8 * vel[i + 1] - vel[i + 2]) / (12 * h)
    acc[0] = (-3 * vel[0] + 4 * vel[1] - vel[2]) / (2 * h)
    acc[1] = (-3 * vel[1] + 4 * vel[2] - vel[3]) / (2 * h)
    acc[-2] = (3 * vel[-2] - 4 * vel[-3] + vel[-4]) / (2 * h)
    acc[-1] = (3 * vel[-1] - 4 * vel[-2] + vel[-3]) / (2 * h)
    return acc

def segunda_derivada(h, pos):
    n = len(pos)
    acc = np.zeros(n)
    for i in range(1, n - 1):
        acc[i] = (pos[i + 1] - 2 * pos[i] + pos[i - 1]) / (h ** 2)
    acc[0] = (pos[2] - 2 * pos[1] + pos[0]) / (h ** 2)
    acc[-1] = (pos[-1] - 2 * pos[-2] + pos[-3]) / (h ** 2)
    return acc

# Cargar datos
mat = scipy.io.loadmat('DLCtest_data.mat')
h = 0.0035  # tiempo de muestreo

def deg2rad(array):
    return np.deg2rad(array.flatten())

beta = deg2rad(mat['beta'])
beta_rate = deg2rad(mat['beta_rate'])
delta = deg2rad(mat['delta'])
yaw = deg2rad(mat['yaw'])
yaw_rate = deg2rad(mat['yaw_rate'])
yaw_acc_real = mat['yaw_acc'].flatten()

# Suavizado con Savitzky-Golay
window_length = 51  # debe ser impar y menor que la longitud total
polyorder = 3       # grado del polinomio para ajustar localmente

yaw_smooth = savgol_filter(yaw, window_length, polyorder)
yaw_rate_smooth = savgol_filter(yaw_rate, window_length, polyorder)

# Calcular derivadas con datos suavizados
acc_d5 = derivada5p(h, yaw_rate_smooth)
acc_2d = segunda_derivada(h, yaw_smooth)

# Vector de tiempo
t = np.arange(0, len(beta)) * h

# Graficar para comparar
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(t, yaw, label='Yaw original')
plt.plot(t, yaw_smooth, label='Yaw suavizado', linewidth=2)
plt.title('Ángulo de guiñada (Yaw)')
plt.xlabel('Tiempo (s)')
plt.ylabel('Ángulo (rad)')
plt.grid()
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t, yaw_acc_real, label='Aceleración angular (yaw_acc) real')
plt.plot(t, acc_d5, 'r-', label='Derivada 5 puntos (suavizado)')
plt.plot(t, acc_2d, 'b-', label='Segunda derivada (suavizado)')
plt.title('Aceleración angular de guiñada')
plt.xlabel('Tiempo (s)')
plt.ylabel('Aceleración (rad/s²)')
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()

# Cálculo de errores con suavizado
def coef_determinacion(y_true, y_pred):
    st = np.sum((y_true - np.mean(y_true))**2)
    sr = np.sum((y_true - y_pred)**2)
    return 1 - sr / st

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

R2_d5 = coef_determinacion(yaw_acc_real, acc_d5)
R2_2d = coef_determinacion(yaw_acc_real, acc_2d)

RMSE_d5 = rmse(yaw_acc_real, acc_d5)
RMSE_2d = rmse(yaw_acc_real, acc_2d)

print(f"R² derivada 5 puntos (suavizado): {R2_d5:.4f}")
print(f"R² segunda derivada (suavizado): {R2_2d:.4f}")
print(f"RMSE derivada 5 puntos (suavizado): {RMSE_d5:.4f}")
print(f"RMSE segunda derivada (suavizado): {RMSE_2d:.4f}")


m = 1507  # masa del vehículo [kg]
Iz = 2995.02  # momento de inercia [kg*m^2]
vel_long = mat['vel_long'].flatten() / 3.6  # km/h a m/s
n = len(t)
A = np.zeros((2 * n, 4))
b_vec = np.zeros(2 * n)

yaw_rate = yaw_rate_smooth


n = len(t)
A = np.zeros((2 * n, 4))
b_vec = np.zeros(2 * n)

for i in range(n):
    v = vel_long[i]

    # Primera ecuación
    A[i, 0] = -beta[i] / v
    A[i, 1] = 0
    A[i, 2] = delta[i] / v
    A[i, 3] = 0
    b_vec[i] = beta_rate[i] + yaw_rate[i]

    # Segunda ecuación
    A[i + n, 0] = 0
    A[i + n, 1] = -beta[i]
    A[i + n, 2] = 0
    A[i + n, 3] = delta[i]
    b_vec[i + n] = yaw_acc_real[i] + yaw_rate[i] / v

# Resolución por mínimos cuadrados
p = np.linalg.lstsq(A, b_vec, rcond=None)[0]

cf_plus_cr_over_m = p[0]
lf_cf_minus_lr_cr_over_Iz = p[1]
cf_over_m = p[2]
lf_cf_over_Iz = p[3]

cf = cf_over_m * m
lf = lf_cf_over_Iz * Iz / cf
cr = cf_plus_cr_over_m * m - cf
lr = (lf * cf - lf_cf_minus_lr_cr_over_Iz * Iz) / cr

# Mostrar resultados
print('Parámetros estimados del modelo:')
print(f'cf = {cf:.2f} N/rad')
print(f'cr = {cr:.2f} N/rad')
print(f'lf = {lf:.2f} m')
print(f'lr = {lr:.2f} m')

# Simulación con los parámetros estimados
beta_dot_est = np.zeros(n)
yaw_acc_est = np.zeros(n)

for i in range(n):
    v = vel_long[i]
    beta_dot_est[i] = -(cf + cr) / (m * v) * beta[i] + \
                      (-(lf * cf - lr * cr) / (m * v**2) - 1) * yaw_rate[i] + \
                      (cf / (m * v)) * delta[i]
    
    yaw_acc_est[i] = -(lf * cf - lr * cr) / Iz * beta[i] - \
                     (lf**2 * cf + lr**2 * cr) / (Iz * v) * yaw_rate[i] + \
                     (lf * cf / Iz) * delta[i]

# Cálculo de R^2
r2_beta_dot = 1 - np.sum((beta_rate - beta_dot_est)**2) / np.sum((beta_rate - np.mean(beta_rate))**2)
r2_yaw_acc = 1 - np.sum((yaw_acc_real - yaw_acc_est)**2) / np.sum((yaw_acc_real - np.mean(yaw_acc_real))**2)

# Gráficas de validación
plt.figure(figsize=(10, 8))

plt.subplot(2, 1, 1)
plt.plot(t, beta_rate, 'k', linewidth=1.5, label='Medición real')
plt.plot(t, beta_dot_est, 'r--', linewidth=1.2, label=f'Modelo estimado, $R^2$ = {r2_beta_dot:.4f}')
plt.grid(True)
plt.title('Velocidad angular de deslizamiento lateral (β̇)')
plt.xlabel('Tiempo (s)')
plt.ylabel('β̇ (rad/s)')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t, yaw_acc_real, 'k', linewidth=1.5, label='Medición real')
plt.plot(t, yaw_acc_est, 'r--', linewidth=1.2, label=f'Modelo estimado, $R^2$ = {r2_yaw_acc:.4f}')
plt.grid(True)
plt.title('Aceleración angular de yaw (ψ̈)')
plt.xlabel('Tiempo (s)')
plt.ylabel('ψ̈ (rad/s²)')
plt.legend()

plt.tight_layout()
plt.show()


beta_simpson = np.zeros_like(beta)
beta_simpson[0] = beta[0]  # Condición inicial

# asumiendo t uniforme
h_simpson = t[1] - t[0]

for i in range(2, len(t), 2): 
    n_simpson = i + 1  
    if (n_simpson - 1) % 2 == 0:
        suma = beta_rate[0]

        for j in range(1, n_simpson - 1, 2):  
            suma += 4 * beta_rate[j]

        for j in range(2, n_simpson - 2, 2):
            suma += 2 * beta_rate[j]

        suma += beta_rate[n_simpson - 1]
        beta_simpson[i] = beta[0] + (h_simpson / 3) * suma
    else:
        if i > 1:
            beta_simpson[i] = beta_simpson[i - 1] + h_simpson * (beta_rate[i - 1] + beta_rate[i]) / 2

    if i + 1 < len(t):
        beta_simpson[i + 1] = beta_simpson[i] + h_simpson * (beta_rate[i] + beta_rate[i + 1]) / 2

# Cálculo de R^2 para la integración
r2_beta_int = 1 - np.sum((beta - beta_simpson)**2) / np.sum((beta - np.mean(beta))**2)

plt.figure(figsize=(8, 5))
plt.plot(t, beta, 'k', linewidth=1.5, label='Medición real')
plt.plot(t, beta_simpson, 'r--', linewidth=1.2, label=f'Integración Simpson 1/3, $R^2$ = {r2_beta_int:.4f}')
plt.grid(True)
plt.title('Ángulo de deslizamiento lateral (β)')
plt.xlabel('Tiempo (s)')
plt.ylabel('β (rad)')
plt.legend(loc='best')
plt.show()


m_nuevo = 1860.0       # kg
Iz_nuevo = 3420.0      # kg·m^2
cf_nuevo = 12200.0     # N/rad
cr_nuevo = 12200.0     # N/rad
lf_nuevo = 1.23        # m
lr_nuevo = 1.55        # m

def find_index(t_array, t_val):
    """Encuentra índice i para el tiempo t_val en arreglo t_array."""
    return np.searchsorted(t_array, t_val, side='right') - 1

# Función que define el sistema de ecuaciones diferenciales
def modelo(t_eval, y):
    beta, yaw_rate = y
    
    i = find_index(t, t_eval)
    i = max(0, min(i, len(t)-1))  
    
    v = max(vel_long[i], 1e-3)    # evitar división por cero
    delta_t = delta[i]
    
    dbeta_dt = (-(cf_nuevo + cr_nuevo) / (m_nuevo * v)) * beta + \
               (-(lf_nuevo * cf_nuevo - lr_nuevo * cr_nuevo) / (m_nuevo * v**2) - 1) * yaw_rate + \
               (cf_nuevo / (m_nuevo * v)) * delta_t
    
    dyaw_dt = (-(lf_nuevo * cf_nuevo - lr_nuevo * cr_nuevo) / Iz_nuevo) * beta + \
              (-(lf_nuevo**2 * cf_nuevo + lr_nuevo**2 * cr_nuevo) / (Iz_nuevo * v)) * yaw_rate + \
              (lf_nuevo * cf_nuevo / Iz_nuevo) * delta_t
    
    return [dbeta_dt, dyaw_dt]

# Condiciones iniciales reales
y0 = [beta[0], yaw_rate_smooth[0]]

# Solución con Runge-Kutta adaptativo RK45
sol = solve_ivp(modelo, (t[0], t[-1]), y0, t_eval=t, method='RK45', rtol=1e-6, atol=1e-8)

beta_rk = sol.y[0]
yaw_rk = sol.y[1]

plt.figure(figsize=(10, 8))

plt.subplot(2, 1, 1)
plt.plot(t, beta, 'k', linewidth=1.5, label='Vehículo original')
plt.plot(t, beta_rk, 'b--', linewidth=1.2, label='Vehículo modificado (RK45 adaptativo)')
plt.grid(True)
plt.title('Ángulo de deslizamiento lateral (β) - Comparación de vehículos')
plt.xlabel('Tiempo (s)')
plt.ylabel('β (rad)')
plt.legend(loc='best')

plt.subplot(2, 1, 2)
plt.plot(t, yaw_rate_smooth, 'k', linewidth=1.5, label='Vehículo original')
plt.plot(t, yaw_rk, 'b--', linewidth=1.2, label='Vehículo modificado (RK45 adaptativo)')
plt.grid(True)
plt.title('Velocidad angular de yaw (r) - Comparación de vehículos')
plt.xlabel('Tiempo (s)')
plt.ylabel('r (rad/s)')
plt.legend(loc='best')

plt.tight_layout()
plt.show()

v_const = 27.77  # m/s (100 km/h)

# Parámetros (define o asigna antes)
m = m_nuevo
Iz = Iz_nuevo
cf = cf_nuevo
cr = cr_nuevo
lf = lf_nuevo
lr = lr_nuevo

a = (cf + cr) / (m * v_const)
b = (lf * cf - lr * cr) / (m * v_const**2)
c = (lf * cf - lr * cr) / Iz
d = (lf**2 * cf + lr**2 * cr) / (Iz * v_const)
e = cf / (m * v_const)
f = lf * cf / Iz

A = a + d
B = a*d - c*b - c
C = e
D = e*d - f*b

print(f'\nFunción de transferencia G(s) = β(s)/δ(s):')
print(f'G(s) = ({C:.4f}·s + {D:.4f}) / (s^2 + {A:.4f}·s + {B:.4f})')

def f_denom(s):
    return s**2 + A*s + B

def df_denom(s):
    return 2*s + A

# --- Método recomendado: fórmula cuadrática (numpy.roots) ---
roots = np.roots([1, A, B])
print('\nRaíces exactas usando numpy.roots:')
for i, root in enumerate(roots, 1):
    print(f's{i} = {root.real:.6f} + {root.imag:.6f}j')

# --- Método Newton-Raphson para hallar una raíz (compleja) ---
s = complex(-1.0, 0.1)  # valor inicial complejo
tol = 1e-6
max_iter = 100
error_rel = 1.0
iter = 0

print('\nMétodo de Newton-Raphson para encontrar una raíz compleja:')
print('Iter\t s\t\t\t f(s)\t\t\t Error relativo')

while error_rel > tol and iter < max_iter:
    s_old = s
    s = s_old - f_denom(s_old) / df_denom(s_old)
    error_rel = abs((s - s_old) / s) if s != 0 else np.inf
    iter += 1
    print(f'{iter}\t {s.real:.6f}+{s.imag:.6f}j\t {f_denom(s):.6e}\t {error_rel:.6e}')

s1 = s
s2 = -A - s1  # segunda raíz

print('\nRaíces calculadas con Newton-Raphson:')
print(f's1 = {s1.real:.6f} + {s1.imag:.6f}j')
print(f's2 = {s2.real:.6f} + {s2.imag:.6f}j')

# Discriminante para tipo de amortiguamiento
discriminante = A**2 - 4*B
print(f'\nDiscriminante = {discriminante:.6f}')

if discriminante > 0:
    print('Sistema SOBREAMORTIGUADO: Los polos son reales y distintos.')
elif np.isclose(discriminante, 0, atol=1e-10):
    print('Sistema CRÍTICAMENTE AMORTIGUADO: Los polos son reales e iguales.')
else:
    print('Sistema SUBAMORTIGUADO: Los polos son complejos conjugados.')
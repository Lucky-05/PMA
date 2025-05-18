import numpy as np
import scipy.io
import matplotlib.pyplot as plt

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
print(mat.keys())
vel_long = mat['vel_long'].flatten() / 3.6  # km/h a m/s
h = 0.002  # tiempo de muestreo

# Convertir de grados a radianes
def deg2rad(array):
    return np.deg2rad(array.flatten())

beta = deg2rad(mat['beta'])
beta_rate = deg2rad(mat['beta_rate'])
delta = deg2rad(mat['delta'])
yaw = deg2rad(mat['yaw'])
yaw_rate = deg2rad(mat['yaw_rate'])
yaw_acc_real = mat['yaw_acc'].flatten()  # ya está en rad/s²

# Cálculo de aceleración
acc_d5 = derivada5p(h, yaw_rate)
acc_2d = segunda_derivada(h, yaw)

# Vector de tiempo
t = np.arange(0, len(beta)) * h

plt.figure(figsize=(10, 6))

plt.subplot(2,1,1)
plt.plot(t, yaw, label='Yaw (rad)')
plt.title('Ángulo de guiñada (Yaw)')
plt.xlabel('Tiempo (s)')
plt.ylabel('Ángulo (rad)')
plt.grid()
plt.legend()

plt.subplot(2,1,2)
plt.plot(t, yaw_acc_real, label='Aceleración angular (yaw_acc) real')
plt.title('Aceleración angular de guiñada')
plt.xlabel('Tiempo (s)')
plt.ylabel('Aceleración (rad/s²)')
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()

# Visualización
plt.figure(figsize=(10, 8))

plt.subplot(2, 1, 1)
plt.plot(t, acc_d5, 'r-', label='Derivada 5 puntos')
plt.plot(t, acc_2d, 'b-', label='Segunda derivada')
plt.plot(t, yaw_acc_real, 'k--', label='Datos reales')
plt.title('Comparación de métodos de diferenciación numérica')
plt.xlabel('Tiempo (s)')
plt.ylabel('Aceleración angular (rad/s²)')
plt.legend()
plt.grid()

# Cálculo de errores
error_d5 = yaw_acc_real - acc_d5
error_2d = yaw_acc_real - acc_2d

plt.subplot(2, 1, 2)
plt.plot(t, error_d5, 'r-', label='Error derivada 5 puntos')
plt.plot(t, error_2d, 'b-', label='Error segunda derivada')
plt.title('Error de estimación')
plt.xlabel('Tiempo (s)')
plt.ylabel('Error (rad/s²)')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# R² y RMSE
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

print(f"R² derivada 5 puntos: {R2_d5:.4f}")
print(f"R² segunda derivada: {R2_2d:.4f}")
print(f"RMSE derivada 5 puntos: {RMSE_d5:.4f}")
print(f"RMSE segunda derivada: {RMSE_2d:.4f}")


# Parámetros conocidos del modelo
m = 1507  # masa del vehículo [kg]
Iz = 2995.02  # momento de inercia [kg*m^2]

# Construcción de matrices para la regresión
n = len(t)
A = np.zeros((2 * n, 4))
b_vec = np.zeros(2 * n)

# Llenado de matrices A y b
for i in range(n):
    v = vel_long[i]

    # Ecuación para beta_dot + r = ...
    A[i, 0] = -beta[i] / v               # (cf + cr)/m
    A[i, 1] = 0                          # (lf*cf - lr*cr)/m
    A[i, 2] = delta[i] / v               # cf/m
    A[i, 3] = 0                          # lf*cf/m
    b_vec[i] = beta_rate[i] + yaw_rate[i]

    # Ecuación para r_dot + r/v = ...
    A[i + n, 0] = 0                      # (cf + cr)/m
    A[i + n, 1] = -beta[i]               # (lf*cf - lr*cr)/Iz
    A[i + n, 2] = 0                      # cf/m
    A[i + n, 3] = delta[i]               # lf*cf/Iz
    b_vec[i + n] = yaw_acc_real[i] + yaw_rate[i] / v

# Resolución por mínimos cuadrados
p = np.linalg.lstsq(A, b_vec, rcond=None)[0]

# Extracción de parámetros intermedios
cf_plus_cr_over_m = p[0]
lf_cf_minus_lr_cr_over_Iz = p[1]
cf_over_m = p[2]
lf_cf_over_Iz = p[3]

# Cálculo de parámetros físicos
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


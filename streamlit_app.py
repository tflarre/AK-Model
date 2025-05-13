import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Definir parámetros del modelo AK
A = 0.15  # Productividad
alpha = 0.3  # Elasticidad del capital
delta = 0.05  # Tasa de depreciación
rho = 0.02  # Tasa de descuento
theta = 1.0  # Elasticidad de sustitución intertemporal

# Funciones del modelo
def production(k):
    return A * k

def balanced_growth_path(k):
    return (A - delta - rho) * k / theta

def k_nullcline(k):
    return production(k) - delta * k

def model_dynamics(y, t):
    k, c = y
    dk = production(k) - delta * k - c
    dc = c * (A - delta - rho) / theta
    return [dk, dc]

# Crear campos vectoriales
k = np.linspace(0.1, 15, 100)
c = np.linspace(0.1, 2, 100)
K, C = np.meshgrid(k, c)

# Calcular derivadas
dK = production(K) - delta * K - C
dC = C * (A - delta - rho) / theta

# Normalizar para el campo vectorial
norm = np.sqrt(dK**2 + dC**2)
eps = 1e-10  # Evitar división por cero
dK_norm = dK / (norm + eps)
dC_norm = dC / (norm + eps)

# Calcular nulclina y senda de equilibrio
k_nullcline = production(k) - delta * k
balanced_path = balanced_growth_path(k)

def create_phase_diagram(k0, c0):
    # Calcular punto en equilibrio para comparar
    c0_equilibrio = balanced_growth_path(np.array([k0]))[0]
    
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    
    # PRIMER GRÁFICO: Diagrama de Fase
    ax = axs[0]
    
    # Graficar campo vectorial
    skip = (slice(None, None, 5), slice(None, None, 5))
    ax.quiver(K[skip], C[skip], dK_norm[skip], dC_norm[skip], 
              color='gray', scale=50, width=0.002, alpha=0.7)
    
    # Colorear regiones
    ax.contourf(K, C, dK > 0, levels=[0, 0.5], colors=['#ffcccc'], alpha=0.2)
    ax.contourf(K, C, dK <= 0, levels=[0, 0.5], colors=['#ccffcc'], alpha=0.2)
    
    # Nulclinas y senda de equilibrio
    ax.plot(k, k_nullcline, 'g--', linewidth=2, label='$\\dot{k}=0$')
    ax.axhline(0, color='b', linestyle='--', linewidth=2, label='$\\dot{c}=0$')
    ax.plot(k, balanced_path, 'r-', linewidth=2, label='Equilibrio')
    
    # Calcular trayectoria
    t = np.linspace(0, 100, 1000)
    y0 = [k0, c0]
    sol = odeint(model_dynamics, y0, t)
    
    # Graficar trayectoria
    ax.plot(sol[:, 0], sol[:, 1], 'k-', linewidth=1.5)
    ax.plot(k0, c0, 'ro', markersize=8, label=f'Punto inicial')
    
    # Flechas indicando dirección
    arrow_indices = [50, 200, 400, 700]
    for i in arrow_indices:
        if i < len(sol) - 1:
            ax.arrow(sol[i, 0], sol[i, 1], 
                    (sol[i+1, 0] - sol[i, 0])*30, 
                    (sol[i+1, 1] - sol[i, 1])*30, 
                    head_width=0.1, head_length=0.2, fc='black', ec='black')
    
    # Mensaje sobre estabilidad
    if abs(c0 - c0_equilibrio) < 0.05:
        stability_msg = "Cerca del equilibrio"
        color = 'green'
    elif c0 > c0_equilibrio:
        stability_msg = "Consumo demasiado alto → Capital decrece → Colapso"
        color = 'red'
    else:
        stability_msg = "Consumo demasiado bajo → Capital crece rápido → Divergencia"
        color = 'blue'
        
    ax.text(0.5, 0.05, stability_msg, transform=ax.transAxes, 
           ha='center', fontsize=10, color=color, 
           bbox=dict(facecolor='white', alpha=0.7))
    
    # Configuración
    ax.set_title('Diagrama de Fase', fontsize=14)
    ax.set_xlabel('Capital per cápita (k)', fontsize=12)
    ax.set_ylabel('Consumo per cápita (c)', fontsize=12)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 2)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='best')
    
    # SEGUNDO GRÁFICO: Evolución temporal
    ax = axs[1]
    
    # Calcular tasas de crecimiento
    growth_k = np.diff(sol[:, 0]) / np.maximum(sol[:-1, 0], 1e-10)  # Evitar división por cero
    growth_c = np.diff(sol[:, 1]) / np.maximum(sol[:-1, 1], 1e-10)
    growth_y = A * np.diff(sol[:, 0]) / np.maximum(sol[:-1, 0], 1e-10)  # Crecimiento del producto
    
    t_growth = t[:-1]  # Tiempo para tasas de crecimiento
    
    # Graficar evolución temporal
    ax.plot(t_growth, growth_k, 'g-', label='Tasa de crecimiento de k')
    ax.plot(t_growth, growth_c, 'b-', label='Tasa de crecimiento de c')
    ax.plot(t_growth, growth_y, 'r-', label='Tasa de crecimiento de y')
    
    # Tasas teóricas en equilibrio
    gamma_theory = (A - (rho + delta)) / theta
    ax.axhline(gamma_theory, color='k', linestyle='--', 
              label=f'Tasa teórica de crecimiento: {gamma_theory:.3f}')
    
    # Configuración
    ax.set_title('Tasas de Crecimiento', fontsize=14)
    ax.set_xlabel('Tiempo', fontsize=12)
    ax.set_ylabel('Tasa de crecimiento', fontsize=12)
    ax.set_xlim(0, 50)  # Mostrar solo los primeros 50 períodos
    ax.set_ylim(-0.2, 0.2)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='best')
    
    plt.tight_layout()
    return fig

# Interfaz de Streamlit
st.title("Diagrama de Fase Interactivo - Modelo AK")
st.markdown("Seleccione valores iniciales y observe la dinámica del sistema. Compare con valores de equilibrio.")

col1, col2 = st.columns(2)

with col1:
    k0 = st.slider("Capital inicial (k₀)", 0.5, 15.0, 5.0, step=0.5)
    
with col2:
    c0_equilibrio = balanced_growth_path(np.array([k0]))[0]
    st.write(f"Consumo en equilibrio: c₀ = {c0_equilibrio:.3f}")
    c0 = st.slider("Consumo inicial (c₀)", 0.1, 1.5, 0.5, step=0.1)

# Botón para ir al equilibrio
if st.button("Ir a equilibrio"):
    c0 = c0_equilibrio

# Mostrar diagrama
fig = create_phase_diagram(k0, c0)
st.pyplot(fig)

# Información adicional
with st.expander("Información del modelo"):
    st.markdown("""
    ## Modelo AK de Crecimiento Endógeno
    
    Este modelo utiliza los siguientes parámetros:
    - A = 0.15 (Productividad)
    - α = 0.3 (Elasticidad del capital)
    - δ = 0.05 (Tasa de depreciación)
    - ρ = 0.02 (Tasa de descuento)
    - θ = 1.0 (Elasticidad de sustitución intertemporal)
    
    La tasa de crecimiento teórica en estado estacionario es: γ = (A - (ρ + δ)) / θ = {(A - (rho + delta)) / theta:.3f}
    """)
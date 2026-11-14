"""
Tarea 3 - Ajuste de parametros con Diseno de Experimentos (DOE)
Opcion A: Diseno factorial completo 3 (temperatura) x 3 (tamano) + ANOVA de dos factores.

Algoritmia y Optimizacion - FCFM-UANL
Alumno: Sebastian Hernandez Perez   Matricula: 2162902

Dependencias:  pip install numpy pandas scipy statsmodels matplotlib
Salidas:       carpeta graficas/ (PNG), resultados_doe.csv y tablas_latex.txt
"""
import math
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

SHOW = False                      # True si quieres que se abran las graficas con plt.show()
os.makedirs("graficas", exist_ok=True)

# ======================================================================
# Parametros del experimento
# ======================================================================
MATRICULA = 2162902
TAMANOS = [5, 10, 15]             # bloque: tamano del TSP
REPLICAS = 10                     # corridas por punto de diseno
N_ITER = 200                      # iteraciones de SA por corrida
T_FINAL = 0.1                     # temperatura final del enfriamiento
T_MIN, T_MAX = 1.0, 100.0         # rango de T0 definido por mi
B_BOOT = 5000                     # remuestreos bootstrap
ALFA = 0.05

# r se genera UNA sola vez, con la matricula como semilla
random.seed(MATRICULA)
r = random.random()
FRACCIONES = {"bajo": 0.3 * r, "medio": 0.3 * r + 0.3, "alto": 0.3 * r + 0.6}
NIVELES = {nom: T_MIN + f * (T_MAX - T_MIN) for nom, f in FRACCIONES.items()}


# ======================================================================
# Funciones del TSP / Recocido Simulado (mismas ideas que en el Ejercicio 4)
# ======================================================================
def tour_length(points, route):
    return sum(math.dist(points[route[i]], points[route[(i + 1) % len(route)]])
               for i in range(len(route)))


def two_opt_move(route, i, k):
    """Invierte route[i..k]."""
    return route[:i] + route[i:k + 1][::-1] + route[k + 1:]


def make_instance(n, seed=42):
    rng = random.Random(seed)
    return [(rng.randint(0, 100), rng.randint(0, 100)) for _ in range(n)]


def simulated_annealing(points, initial_route, T0, n_iter, T_final, rng):
    """SA para minimizar. Enfriamiento exponencial T(t) = T0 * alpha^t con
    alpha = (T_final / T0)^(1/N). delta_E = L(actual) - L(siguiente)."""
    alpha = (T_final / T0) ** (1.0 / n_iter)
    current = initial_route[:]
    cur_len = tour_length(points, current)
    best_len = cur_len
    n = len(points)
    for t in range(1, n_iter + 1):
        T = T0 * alpha ** t
        i, k = sorted(rng.sample(range(1, n), 2))
        nxt = two_opt_move(current, i, k)
        nxt_len = tour_length(points, nxt)
        delta_E = cur_len - nxt_len
        if delta_E > 0 or rng.random() < math.exp(delta_E / T):
            current, cur_len = nxt, nxt_len
        best_len = min(best_len, cur_len)
    return best_len


def bootstrap_mediana(x, B=B_BOOT, seed=0):
    """Mediana bootstrap y su IC al 95% (percentiles 2.5 y 97.5)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    meds = np.median(rng.choice(x, size=(B, len(x)), replace=True), axis=1)
    return float(np.median(meds)), float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


# ======================================================================
# Experimento: 3 temperaturas x 3 tamanos x 10 replicas = 90 corridas
# ======================================================================
def correr_experimento():
    filas = []
    for n in TAMANOS:
        puntos = make_instance(n, seed=42)                  # una instancia por tamano
        rng0 = random.Random(7)
        inicio = list(range(1, n))
        rng0.shuffle(inicio)
        inicio = [0] + inicio                               # misma ruta inicial para todas las corridas
        for ti, (nom, T0) in enumerate(NIVELES.items()):
            for rep in range(REPLICAS):
                rng = random.Random(100000 * n + 1000 * ti + rep)   # semilla distinta por corrida
                costo = simulated_annealing(puntos, inicio, T0, N_ITER, T_FINAL, rng)
                filas.append({"tamano": n, "nivel": nom, "temperatura": round(T0, 2),
                              "replica": rep + 1, "costo": costo})
    df = pd.DataFrame(filas)
    # Costo relativo: costo / mejor costo encontrado para ese tamano (para comparar entre tamanos)
    mejor = df.groupby("tamano")["costo"].transform("min")
    df["costo_rel"] = df["costo"] / mejor
    return df


def main():
    print("=" * 70)
    print("TAREA 3 - DOE sobre la temperatura del Recocido Simulado (Opcion A)")
    print("=" * 70)
    print(f"Matricula = {MATRICULA}  ->  r = {r:.6f}")
    print(f"Rango de T0 = [{T_MIN:g}, {T_MAX:g}]")
    for nom in NIVELES:
        print(f"  {nom:<6}: fraccion = {FRACCIONES[nom]:.4f}  ->  T0 = {NIVELES[nom]:.2f}")

    t0 = time.perf_counter()
    df = correr_experimento()
    print(f"\n90 corridas terminadas en {time.perf_counter() - t0:.2f} s")
    df.to_csv("resultados_doe.csv", index=False)

    # ---------------- ANOVA de dos factores ----------------
    modelo = ols("costo_rel ~ C(temperatura) * C(tamano)", data=df).fit()
    anova = sm.stats.anova_lm(modelo, typ=2)
    print("\nANOVA de dos factores (variable respuesta: costo relativo):")
    print(anova)

    p_temp = anova.loc["C(temperatura)", "PR(>F)"]
    p_tam = anova.loc["C(tamano)", "PR(>F)"]
    p_int = anova.loc["C(temperatura):C(tamano)", "PR(>F)"]
    for nombre, p in [("Temperatura", p_temp), ("Tamano", p_tam), ("Interaccion", p_int)]:
        print(f"  {nombre:<12} p = {p:.4g} -> {'SIGNIFICATIVO' if p < ALFA else 'no significativo'} (alfa = {ALFA})")

    # ---------------- Bootstrap por punto de diseno ----------------
    filas = []
    for n in TAMANOS:
        for nom, T0 in NIVELES.items():
            x = df[(df.tamano == n) & (df.nivel == nom)]["costo"].values
            xr = df[(df.tamano == n) & (df.nivel == nom)]["costo_rel"].values
            med, lo, hi = bootstrap_mediana(x, seed=n)
            medr, lor, hir = bootstrap_mediana(xr, seed=n)
            filas.append({"tamano": n, "nivel": nom, "T0": round(T0, 2), "med": med, "lo": lo, "hi": hi,
                          "med_rel": medr, "lo_rel": lor, "hi_rel": hir})
    boot = pd.DataFrame(filas)
    print("\nMediana bootstrap (IC 95%) por punto de diseno:")
    print(boot.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # Mejor temperatura por tamano: menor mediana bootstrap (desempate: menor media)
    medias = df.groupby(["tamano", "nivel"])["costo"].mean()
    mejores = []
    for n in TAMANOS:
        sub = boot[boot.tamano == n].copy()
        sub["media"] = [medias[(n, nv)] for nv in sub.nivel]
        sub = sub.sort_values(["med", "media"])
        mejores.append(sub.iloc[0])
    mejores = pd.DataFrame(mejores)
    print("\nTemperatura con mejor desempeno por tamano:")
    print(mejores[["tamano", "nivel", "T0", "med", "lo", "hi"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # ---------------- Graficas ----------------
    orden = list(NIVELES)
    # (1) efectos principales
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    m_temp = df.groupby("nivel")["costo_rel"].mean().reindex(orden)
    axes[0].plot([f"{nv}\n(T0={NIVELES[nv]:.1f})" for nv in orden], m_temp.values, "o-")
    axes[0].set_title("Efecto principal: temperatura")
    axes[0].set_ylabel("Costo relativo medio")
    m_tam = df.groupby("tamano")["costo_rel"].mean()
    axes[1].plot([str(n) for n in TAMANOS], m_tam.values, "s-", color="tab:orange")
    axes[1].set_title("Efecto principal: tamano")
    axes[1].set_xlabel("n (ciudades)")
    for ax in axes:
        ax.grid(alpha=0.3)
    plt.suptitle("Grafica de efectos principales")
    plt.tight_layout()
    plt.savefig("graficas/efectos_principales.png", dpi=150)
    if SHOW:
        plt.show()
    plt.close()

    # (2) interaccion temperatura x tamano
    plt.figure(figsize=(6.5, 4.5))
    for n in TAMANOS:
        y = [df[(df.tamano == n) & (df.nivel == nv)]["costo_rel"].mean() for nv in orden]
        plt.plot([f"{nv}\n(T0={NIVELES[nv]:.1f})" for nv in orden], y, "o-", label=f"n = {n}")
    plt.ylabel("Costo relativo medio")
    plt.title("Grafica de interaccion temperatura x tamano")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("graficas/interaccion.png", dpi=150)
    if SHOW:
        plt.show()
    plt.close()

    # (3) medianas bootstrap con IC por tamano (extra)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, n in zip(axes, TAMANOS):
        sub = boot[boot.tamano == n].set_index("nivel").reindex(orden)
        yerr = [sub.med - sub.lo, sub.hi - sub.med]
        ax.errorbar(range(3), sub.med, yerr=yerr, fmt="o", capsize=5)
        ax.set_xticks(range(3))
        ax.set_xticklabels([f"{nv}\n(T0={sub.loc[nv, 'T0']:.1f})" for nv in orden])
        ax.set_title(f"n = {n}")
        ax.set_ylabel("Mediana bootstrap del costo")
        ax.grid(alpha=0.3)
    plt.suptitle("Mediana bootstrap e IC 95% por punto de diseno")
    plt.tight_layout()
    plt.savefig("graficas/bootstrap_ic.png", dpi=150)
    if SHOW:
        plt.show()
    plt.close()

    # ---------------- Tablas en formato LaTeX (para copiar a Overleaf) ----------------
    with open("tablas_latex.txt", "w", encoding="utf-8") as f:
        f.write("% ---- Niveles de temperatura ----\n")
        for nom in orden:
            f.write(f"{nom} & {FRACCIONES[nom]:.4f} & {NIVELES[nom]:.2f} \\\\\n")
        f.write("\n% ---- ANOVA ----\n")
        nombres = {"C(temperatura)": "Temperatura", "C(tamano)": "Tama\\~no",
                   "C(temperatura):C(tamano)": "Interacci\\'on", "Residual": "Error"}
        for idx, fila in anova.iterrows():
            F = "" if pd.isna(fila["F"]) else f"{fila['F']:.3f}"
            p = "" if pd.isna(fila["PR(>F)"]) else f"{fila['PR(>F)']:.4g}"
            f.write(f"{nombres[idx]} & {fila['sum_sq']:.4f} & {int(fila['df'])} & {F} & {p} \\\\\n")
        f.write("\n% ---- Bootstrap por punto de diseno (costo) ----\n")
        for _, b in boot.iterrows():
            f.write(f"{int(b.tamano)} & {b.nivel} & {b.T0:.2f} & {b.med:.2f} & [{b.lo:.2f}, {b.hi:.2f}] \\\\\n")
        f.write("\n% ---- Mejor temperatura por tamano ----\n")
        for _, b in mejores.iterrows():
            f.write(f"{int(b.tamano)} & {b.nivel} & {b.T0:.2f} & {b.med:.2f} & [{b.lo:.2f}, {b.hi:.2f}] \\\\\n")
    print("\nTablas LaTeX guardadas en tablas_latex.txt; graficas en la carpeta graficas/")


if __name__ == "__main__":
    main()

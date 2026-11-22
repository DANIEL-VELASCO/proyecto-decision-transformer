"""Tablas y figuras del informe a partir de los CSV de la carpeta resultados/.

Uso: python analisis_resultados.py
Lee resultados_semilla_*.csv, datos.csv, referencias.json y (si existe) la carpeta curvas/.
Escribe las figuras en informe/figuras y un resumen en resultados/resumen.csv.
"""
import glob, json, os
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

RES, FIG = "resultados", "informe/figuras"
os.makedirs(FIG, exist_ok=True)
ENVS, QUALITIES = ["hopper", "halfcheetah"], ["simple", "medium", "expert"]
MODELS = ["BC · Random Forest", "BC · MLP", "IQL", "Decision Transformer"]
SHORT = {"BC · Random Forest": "RF", "BC · MLP": "MLP", "IQL": "IQL", "Decision Transformer": "DT"}
COLORS = {"BC · Random Forest": "#9e9e9e", "BC · MLP": "#4c72b0", "IQL": "#dd8452", "Decision Transformer": "#55a868"}
NAMES = {"hopper": "Hopper", "halfcheetah": "HalfCheetah"}
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.titlesize": 9, "legend.fontsize": 7, "mathtext.fontset": "dejavuserif"})

res = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{RES}/resultados_semilla_*.csv"))], ignore_index=True)
res = res.drop_duplicates(["Robot", "Datos", "Semilla", "Modelo"], keep="last")   # una sesión repitió la semilla 1
refs = json.load(open(f"{RES}/referencias.json"))
datos = pd.read_csv(f"{RES}/datos.csv")
norm = lambda env, R: 100 * (R - refs[env]["random"]) / (refs[env]["expert"] - refs[env]["random"])
datos["Puntaje dataset"] = [norm(e, R) for e, R in zip(datos["Robot"], datos["Retorno medio"])]
print(f"Corridas: {len(res)} · semillas: {sorted(res['Semilla'].unique())}")

# ---------- Tabla de puntajes ----------
g = res.groupby(["Robot", "Datos", "Modelo"])["Puntaje normalizado"].agg(["mean", "std", "count"]).reset_index()
g.to_csv(f"{RES}/resumen.csv", index=False)
print("\nPuntaje normalizado (media ± desv. entre semillas)")
for env in ENVS:
    for q in QUALITIES:
        ds = datos[(datos.Robot == env) & (datos.Datos == q)]["Puntaje dataset"].item()
        row = [f"{SHORT[m]} {g[(g.Robot==env)&(g.Datos==q)&(g.Modelo==m)]['mean'].item():6.1f}±"
               f"{g[(g.Robot==env)&(g.Datos==q)&(g.Modelo==m)]['std'].item():4.1f}" for m in MODELS]
        print(f"{env:<11} {q:<6} datos {ds:5.1f} | " + " | ".join(row))
t = res.groupby("Modelo")[["Entrenamiento (s)", "Evaluación (s)", "MSE validación"]].mean()
print("\nPromedios por modelo\n", t.round(3))
print("\nPuntaje medio global\n", res.groupby("Modelo")["Puntaje normalizado"].mean().round(1))

# ---------- Fig. 1: esquema del Decision Transformer ----------
fig, ax = plt.subplots(figsize=(3.5, 1.9)); ax.set_xlim(0, 10); ax.set_ylim(0, 5.4); ax.axis("off")
toks = [("$\\hat R_{t-1}$", "#f2d7b6"), ("$s_{t-1}$", "#c9dbef"), ("$a_{t-1}$", "#cfe8d1"),
        ("$\\hat R_t$", "#f2d7b6"), ("$s_t$", "#c9dbef"), ("$a_t$", "#cfe8d1")]
xs = np.linspace(1.0, 9.0, len(toks))
for x, (lab, c) in zip(xs, toks):
    ax.add_patch(FancyBboxPatch((x - 0.55, 0.3), 1.1, 0.8, boxstyle="round,pad=0.05", fc=c, ec="k", lw=0.6))
    ax.text(x, 0.7, lab, ha="center", va="center", fontsize=8)
    ax.annotate("", xy=(x, 1.9), xytext=(x, 1.15), arrowprops=dict(arrowstyle="->", lw=0.6))
ax.add_patch(FancyBboxPatch((0.4, 1.95), 9.2, 1.1, boxstyle="round,pad=0.05", fc="#eeeeee", ec="k", lw=0.6))
ax.text(5, 2.5, "Transformer causal (GPT-2, 3 capas)", ha="center", va="center", fontsize=8)
for x in (xs[1], xs[4]):
    ax.annotate("", xy=(x, 3.85), xytext=(x, 3.1), arrowprops=dict(arrowstyle="->", lw=0.6))
ax.text(xs[1], 4.2, "$\\hat a_{t-1}$", ha="center", fontsize=8); ax.text(xs[4], 4.2, "$\\hat a_t$", ha="center", fontsize=8)
ax.text(0.0, 5.0, "… K = 20 pasos de historia", fontsize=7, color="#555555")
plt.tight_layout(); plt.savefig(f"{FIG}/fig1_dt.png", dpi=300); plt.close()

# ---------- Fig. 2: puntajes por robot y nivel de datos ----------
fig, axes = plt.subplots(2, 1, figsize=(3.5, 4.0), sharex=True)
w, x = 0.2, np.arange(len(QUALITIES))
for ax, env in zip(axes, ENVS):
    for j, m in enumerate(MODELS):
        s = g[(g.Robot == env) & (g.Modelo == m)].set_index("Datos").reindex(QUALITIES)
        ax.bar(x + (j - 1.5) * w, s["mean"], w, yerr=s["std"], capsize=2, color=COLORS[m], label=SHORT[m],
               error_kw=dict(lw=0.7))
    ds = datos[datos.Robot == env].set_index("Datos").reindex(QUALITIES)["Puntaje dataset"]
    for k, v in enumerate(ds):
        ax.hlines(v, k - 0.45, k + 0.45, colors="k", linestyles="--", lw=0.8, label="Dataset" if k == 0 else None)
    ax.set_xticks(x); ax.set_xticklabels(QUALITIES); ax.set_title(NAMES[env], pad=3); ax.set_ylabel("Puntaje normalizado")
    ax.grid(axis="y", alpha=0.3)
axes[1].set_xlabel("Nivel de datos")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, ncol=5, loc="upper center", frameon=False, bbox_to_anchor=(0.5, 1.0), fontsize=7, handlelength=1.5, columnspacing=1.0)
plt.tight_layout(rect=(0, 0, 1, 0.95)); plt.savefig(f"{FIG}/fig2_puntajes.png", dpi=300); plt.close()

# ---------- Fig. 3: puntaje por episodio en los casos expert ----------
fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.2), sharey=True)
for ax, env in zip(axes, ENVS):
    for j, m in enumerate(MODELS):
        sub = res[(res.Robot == env) & (res.Datos == "expert") & (res.Modelo == m)]
        vals = np.concatenate([norm(env, np.array(json.loads(r))) for r in sub["Retornos"]])
        jit = np.random.default_rng(j).uniform(-0.18, 0.18, len(vals))
        ax.scatter(np.full(len(vals), j) + jit, vals, s=5, alpha=0.6, color=COLORS[m], lw=0)
    ax.set_xticks(range(4)); ax.set_xticklabels([SHORT[m] for m in MODELS]); ax.set_title(f"{NAMES[env]} expert")
    ax.grid(axis="y", alpha=0.3)
axes[0].set_ylabel("Puntaje por episodio")
plt.tight_layout(); plt.savefig(f"{FIG}/fig3_episodios.png", dpi=300); plt.close()

# ---------- Fig. 4: curvas de pérdida (si se descargó la carpeta curvas) ----------
curv = f"{RES}/curvas"
if os.path.isdir(curv):
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.8), sharey=True)
    for ax, (m, slug) in zip(axes, [("BC · MLP", "bc_mlp"), ("Decision Transformer", "decision_transformer")]):
        for env in ENVS:
            for q, ls in zip(QUALITIES, [":", "--", "-"]):
                cs = [pd.read_csv(f) for f in glob.glob(f"{curv}/{env}_{q}_*_{slug}.csv")]
                if not cs: continue
                c = pd.concat(cs).groupby("Paso")["Valor"].mean()
                ax.plot(c.index / 1000, c.values, ls=ls, lw=0.9, color="#4c72b0" if env == "hopper" else "#c44e52",
                        label=f"{NAMES[env]} {q}")
        ax.set_yscale("log"); ax.set_title(SHORT[m]); ax.set_xlabel("Paso (miles)"); ax.grid(alpha=0.3)
    axes[0].set_ylabel("Error cuadrático medio")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=2, loc="lower center", frameon=False, fontsize=6, bbox_to_anchor=(0.5, 0.0))
    plt.tight_layout(rect=(0, 0.2, 1, 1)); plt.savefig(f"{FIG}/fig4_curvas.png", dpi=300); plt.close()
    print("Curvas: listo")
else:
    print("Sin carpeta resultados/curvas: no se genera la Fig. 4")
print("Figuras en", FIG)

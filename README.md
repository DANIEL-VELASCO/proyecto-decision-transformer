# Control robótico a partir de datos registrados

Proyecto final de Aprendizaje Profundo (202630, Pontificia Universidad Javeriana).
Autor: Daniel Sebastian Velasco Munar.

Se compara el Decision Transformer con tres métodos de referencia en aprendizaje por refuerzo offline:
Implicit Q-Learning (IQL), clonación de comportamiento con una red MLP y clonación de comportamiento con Random Forest.
Los datos provienen de Minari (MuJoCo Hopper y HalfCheetah) y los agentes se evalúan en el simulador.

## Notebooks

| Notebook | Contenido |
|---|---|
| `00_prueba_viabilidad_colab.ipynb` | Prueba de punta a punta con presupuestos pequeños en Google Colab |
| `01_experimentos_completos.ipynb` | Los cuatro modelos en 2 robots × 3 niveles de datos × 3 semillas, con 100 mil pasos de entrenamiento |

Los notebooks se ejecutan en Google Colab con GPU. Los resultados del notebook 01 se guardan en Google Drive,
en la carpeta `proyecto_decision_transformer`.

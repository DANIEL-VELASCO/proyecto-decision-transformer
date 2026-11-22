PROYECTO FINAL - APRENDIZAJE POR REFUERZO FUERA DE LINEA CON DECISION TRANSFORMER:
CONTROL DE ROBOTS SIMULADOS A PARTIR DE DATOS REGISTRADOS
Aprendizaje Profundo - Maestria en Inteligencia Artificial
Pontificia Universidad Javeriana - Periodo 2026-30

INTEGRANTES
-----------
- Daniel Sebastian Velasco Munar (trabajo individual)


ENTREGABLES
-----------
1. informe/
   Informe en formato IEEE Transactions on Artificial Intelligence (PDF y
   fuente en Word), con Resumen, Introduccion, Aproximacion al problema,
   Experimentacion, Resultados y Conclusiones. La carpeta figuras/ tiene las
   figuras del informe.

2. 00_prueba_viabilidad_colab.ipynb
   Prueba de punta a punta en Google Colab con presupuestos pequenos: instala
   las librerias, descarga los datos de Minari y entrena y evalua los cuatro
   modelos para verificar que todo funciona.

3. 01_experimentos_completos.ipynb
   Experimento principal. Preparacion de datos (particion 90/10 por episodios,
   normalizacion de estados, retorno restante) y entrenamiento y evaluacion de
   los cuatro modelos:
   - Clonacion de comportamiento con Random Forest (modelo tradicional).
   - Clonacion de comportamiento con una red MLP.
   - Implicit Q-Learning (IQL, libreria d3rlpy).
   - Decision Transformer (Hugging Face Transformers, PyTorch).
   2 robots (Hopper, HalfCheetah) x 3 conjuntos (simple, medium, expert)
   x 3 semillas = 72 entrenamientos.

4. 02_retorno_objetivo.ipynb
   Evalua los Decision Transformer ya entrenados pidiendoles distintos
   retornos objetivo (25 % a 125 % del retorno experto).

5. analisis_resultados.py
   Lee los archivos de resultados/ y genera las tablas y figuras del informe.

6. resultados/
   Resultados exportados de la ejecucion: un CSV por semilla con el puntaje de
   cada modelo, resumen de los conjuntos de datos, referencias del puntaje
   normalizado, curvas de perdida y prueba del retorno objetivo.

7. requirements.txt
   Dependencias de Python.

8. README.txt
   Este archivo.

No se incluyen ejecutables, archivos compilados ni pesos de los modelos.


COMO EJECUTAR
-------------
1. Abrir el notebook en Google Colab con GPU (Entorno de ejecucion >
   Cambiar tipo de entorno de ejecucion > T4 GPU) y ejecutar todo.
2. Los datos se descargan solos desde Minari. Los resultados se guardan en
   Google Drive, en MyDrive/proyecto_decision_transformer/, y las corridas ya
   terminadas se saltan si se vuelve a ejecutar.
3. Para regenerar las figuras: copiar los resultados de Drive en resultados/
   y ejecutar "python analisis_resultados.py".

Repositorio: https://github.com/DANIEL-VELASCO/proyecto-decision-transformer

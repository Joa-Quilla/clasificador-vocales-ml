# Clasificador Inteligente de Vocales con Machine Learning

Clasificador de vocales manuscritas (A, E, I, O, U) utilizando Machine Learning clásico. Implementa Regresión Logística y Random Forest con razonamiento probabilístico, análisis de incertidumbre y una interfaz interactiva con Gradio.

## 🎯 Características

- **Dos modelos de ML**: Regresión Logística y Random Forest con optimización de hiperparámetros
- **Data Augmentation**: Generación de variaciones para mejorar la generalización (3x dataset)
- **Razonamiento Probabilístico**: Análisis de confianza para cada predicción
- **Análisis de Incertidumbre**: Identificación de casos ambiguos
- **Interfaz Gradio**: Demostración interactiva con visualización de probabilidades
- **Matriz de Confusión**: Análisis detallado de confusiones entre vocales

## 📊 Resultados

- **Dataset**: 510 imágenes originales → 1530 con data augmentation
- **Precisión**: ~77% (validación cruzada 5-fold)
- **Modelo final**: Regresión Logística con dataset aumentado

## 🚀 Instalación

```bash
# Clonar el repositorio
git clone https://github.com/Joa-Quilla/clasificador-vocales-ml.git
cd clasificador-vocales-ml

# Instalar dependencias
pip install -r requirements.txt
```

## 💻 Uso

### Entrenar el modelo (opcional)

El modelo ya está entrenado y guardado. Si deseas re-entrenar:

```bash
# Abrir el notebook en Jupyter o VS Code
jupyter notebook entrenamiento.ipynb
```

### Ejecutar la interfaz

```bash
python app_interfaz.py
```

La interfaz se abrirá en `http://127.0.0.1:7861`

## 📁 Estructura del Proyecto

```
clasificador-vocales-ml/
├── entrenamiento.ipynb              # Notebook principal con análisis completo
├── app_interfaz.py                  # Interfaz Gradio para demostración
├── modelo_regresión_logística.pkl   # Modelo entrenado
├── etiquetas_nombres.pkl            # Mapeo de clases
├── requirements.txt                 # Dependencias del proyecto
├── informe_tecnico.txt              # Documentación técnica detallada
├── preguntas_presentacion.txt       # Q&A para defensa del proyecto
├── explicacion_interfaz.txt         # Explicación del código de la interfaz
├── A/                               # 100 imágenes de vocal A
├── E/                               # 100 imágenes de vocal E
├── I/                               # 100 imágenes de vocal I
├── O/                               # 106 imágenes de vocal O
└── U/                               # 104 imágenes de vocal U
```

## 🔬 Metodología

### Preprocesamiento
1. Conversión a escala de grises
2. Redimensionamiento a 28×28 píxeles
3. Normalización [0-1]
4. Aplanamiento a vector de 784 elementos

### Data Augmentation
- Rotaciones: -15° a 15°
- Desplazamientos: ±2 píxeles
- Zoom: 0.9x a 1.1x
- Factor de aumento: 3x

### Modelos Implementados
- **Regresión Logística**: Modelo seleccionado por rendimiento y eficiencia
- **Random Forest**: Con Grid Search para optimización de hiperparámetros

### Evaluación
- Train/Test split: 80/20
- Validación cruzada: 5-fold
- Métricas: Accuracy, Precision, Recall, F1-score
- Análisis: Matriz de confusión, razonamiento probabilístico, incertidumbre

## 📈 Requerimientos Cumplidos

✅ Preprocesamiento de imágenes  
✅ Dos modelos de ML con optimización  
✅ Validación cruzada y Grid Search  
✅ Matriz de confusión y análisis estadístico  
✅ Razonamiento probabilístico  
✅ Análisis de incertidumbre  
✅ Interfaz de demostración  

## 🛠️ Tecnologías

- **Python 3.12+**
- **scikit-learn**: Modelos de Machine Learning
- **OpenCV**: Procesamiento de imágenes
- **NumPy & Pandas**: Manipulación de datos
- **Matplotlib & Seaborn**: Visualización
- **Gradio**: Interfaz web interactiva
- **SciPy**: Transformaciones de imágenes

## 📝 Documentación Adicional

- `informe_tecnico.txt`: Comparación de modelos, selección de features y análisis de matriz de confusión
- `preguntas_presentacion.txt`: 45 preguntas y respuestas para defensa del proyecto
- `explicacion_interfaz.txt`: Explicación línea por línea del código de la interfaz

## 👨‍💻 Autor

**Joaquín Quilla**  
Proyecto académico - Inteligencia Artificial  
UDEO

## 📄 Licencia

Este proyecto es de código abierto y está disponible bajo la Licencia MIT.

---

⭐ Si este proyecto te fue útil, considera darle una estrella en GitHub

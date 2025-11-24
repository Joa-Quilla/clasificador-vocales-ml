import gradio as gr
import joblib
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar modelo y etiquetas
modelo = joblib.load('modelo_regresión_logística.pkl')
etiquetas_nombres = joblib.load('etiquetas_nombres.pkl')

def preprocesar_imagen(imagen):
    """
    Preprocesa la imagen EXACTAMENTE como en el entrenamiento.
    """
    # Convertir PIL Image a numpy array
    img_array = np.array(imagen)
    
    # Convertir a escala de grises si es RGB
    if len(img_array.shape) == 3:
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img_array
    
    # Redimensionar a 28x28 (igual que en el entrenamiento)
    img_resized = cv2.resize(img_gray, (28, 28))
    
    # Normalizar (0-1)
    img_normalized = img_resized / 255.0
    
    # Aplanar a vector
    img_flatten = img_normalized.flatten()
    
    return img_flatten.reshape(1, -1)

def clasificar_vocal(imagen):
    """
    Clasifica la vocal en la imagen y retorna las probabilidades.
    """
    if imagen is None:
        return "Por favor, carga una imagen", None
    
    try:
        # Preprocesar
        img_procesada = preprocesar_imagen(imagen)
        
        # Predecir
        prediccion = modelo.predict(img_procesada)[0]
        probabilidades = modelo.predict_proba(img_procesada)[0]
        
        # Vocal predicha
        vocal_predicha = etiquetas_nombres[prediccion]
        confianza = probabilidades[prediccion] * 100
        
        # Crear resultado de texto
        resultado_texto = f"""
# Predicción: {vocal_predicha}
## Confianza: {confianza:.2f}%

### Probabilidades de todas las vocales:
"""
        
        # Crear diccionario de probabilidades para el gráfico
        prob_dict = {}
        for i, prob in enumerate(probabilidades):
            vocal = etiquetas_nombres[i]
            porcentaje = prob * 100
            resultado_texto += f"\n**{vocal}**: {porcentaje:.2f}%"
            prob_dict[vocal] = porcentaje
        
        # Crear gráfico de barras
        fig, ax = plt.subplots(figsize=(8, 5))
        vocales = list(prob_dict.keys())
        valores = list(prob_dict.values())
        
        colores = ['#FF6B6B' if v == vocal_predicha else '#95E1D3' for v in vocales]
        
        bars = ax.barh(vocales, valores, color=colores)
        ax.set_xlabel('Probabilidad (%)', fontsize=12, fontweight='bold')
        ax.set_title('Razonamiento Probabilístico del Agente', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 100)
        ax.grid(axis='x', alpha=0.3)
        
        # Agregar valores en las barras
        for i, (bar, valor) in enumerate(zip(bars, valores)):
            ax.text(valor + 2, i, f'{valor:.1f}%', va='center', fontweight='bold')
        
        plt.tight_layout()
        
        # Análisis de incertidumbre
        prob_ordenadas = sorted(probabilidades, reverse=True)
        diferencia = (prob_ordenadas[0] - prob_ordenadas[1]) * 100
        
        if diferencia < 20:
            resultado_texto += f"\n\n**Incertidumbre detectada**: La diferencia entre las dos probabilidades más altas es solo {diferencia:.2f}%"
        else:
            resultado_texto += f"\n\n**Predicción confiable**: Alta certeza en la clasificación"
        
        return resultado_texto, fig
        
    except Exception as e:
        return f"Error al procesar la imagen: {str(e)}", None

# Crear interfaz
with gr.Blocks(title="Clasificador de Vocales - IA", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # Agente Clasificador Inteligente de Vocales
    
    ## Instrucciones:
    1. Sube una imagen de una vocal escrita a mano (A, E, I, O, U)
    2. El agente analizará la imagen y te mostrará:
       - La vocal predicha
       - Las probabilidades de cada vocal (razonamiento probabilístico)
       - El nivel de confianza de la predicción
    
    ### Características del Agente:
    - **Modelo**: Regresión Logística con Data Augmentation
    - **Dataset**: 1530 imágenes (510 originales x3 augmentation)
    - **Precisión**: ~77%
    - **Razonamiento Probabilístico**: Muestra probabilidades de las 5 clases
    - **Detección de Incertidumbre**: Identifica casos ambiguos
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            imagen_input = gr.Image(
                label="Carga tu imagen de una vocal",
                type="pil",
                height=300
            )
            clasificar_btn = gr.Button("Clasificar Vocal", variant="primary", size="lg")
            
            gr.Markdown("""
            ### Tips:
            - Usa imágenes claras de vocales escritas a mano
            - Funciona mejor con fondo claro
            - La vocal debe ocupar la mayor parte de la imagen
            """)
        
        with gr.Column(scale=1):
            resultado_texto = gr.Markdown(label="Resultado")
            grafico_output = gr.Plot(label="Análisis Probabilístico")
    
    # Ejemplos de uso
    gr.Markdown("""
    ## Ejemplos
    Puedes probar con cualquier imagen de las vocales del dataset de entrenamiento o crear nuevas.
    """)
    
    # Evento de clic
    clasificar_btn.click(
        fn=clasificar_vocal,
        inputs=imagen_input,
        outputs=[resultado_texto, grafico_output]
    )

# Lanzar aplicación
if __name__ == "__main__":
    demo.launch(share=False, server_name="127.0.0.1", server_port=7861)

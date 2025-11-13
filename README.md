# Parcial de Visión Artificial - Redes Convolucionales

## 📖 Descripción

Este proyecto implementa **todas las partes del parcial de Visión Artificial** con redes neuronales convolucionales (CNN), desde exploración y preprocesamiento hasta aplicación de modelos preentrenados.

### 🎯 Objetivos Completos del Parcial

#### **Parte I - Exploración y Preprocesamiento (20%)**
1. **Carga y visualización de imágenes**
   - Cargar imágenes del dataset
   - Mostrarlas junto con su nombre de archivo
   - Describir el tipo de clasificación que representan

2. **Preprocesamiento**
   - Redimensionar imágenes a 224×224 píxeles
   - Normalizar valores de píxeles al rango [0,1]
   - Visualizar imágenes antes y después del preprocesamiento

#### **Parte II - Aplicación de CNN Preentrenada (20%)**
3. **Uso de redes convolucionales preentrenadas**
   - MobileNetV2, ResNet50, VGG16 con pesos de ImageNet
   - Predicciones en tiempo real con niveles de confianza
   - Comparación automática entre múltiples modelos

4. **Interpretación de resultados**
   - Análisis de correspondencia con contenido real
   - Visualización profesional de estadísticas
   - Consenso automático entre modelos

#### **Parte III - Análisis con Preprocesamiento Adicional (20%)**
5. **Preprocesamiento avanzado y comparación**
   - Técnicas adicionales de mejora de imagen
   - Análisis comparativo de rendimiento

#### **Parte IV - Análisis Conceptual (20%)**
6. **Fundamentos teóricos de CNN**
   - Papel de filtros/kernels
   - Importancia de normalización
   - Ventajas de modelos preentrenados

## 🏗️ Estructura del Proyecto

```
parcial3-vision_artificial/
├── main_corregido.py           # Sistema principal con menú interactivo completo
├── requirements.txt            # Dependencias básicas
├── requirements-cnn.txt        # Dependencias para CNN preentrenadas
├── verificar_cnn.py           # Script de verificación de dependencias CNN
├── README.md                   # Este archivo
├── README_CNN_PREENTRENADAS.md # Documentación específica de CNNs
├── Parcial_Vision_Artificial_CNN.ipynb # Notebook completo del parcial
├── images/                     # Dataset de imágenes
│   ├── imagen1.png
│   ├── imagen2.png
│   └── ...
├── resultados_cnn/            # Resultados y visualizaciones CNN
│   ├── comparaciones/
│   ├── visualizaciones/
│   └── logs/
└── modules/                   # Módulos del sistema
    ├── __init__.py
    ├── gestor_imagenes.py     # Carga y visualización de imágenes
    ├── preprocesador_parcial.py # Preprocesamiento específico
    ├── redes_preentrenadas.py # 🆕 CNNs preentrenadas (Parte II)
    ├── operaciones_aritmeticas.py # Operaciones aritméticas
    └── operaciones_geometricas.py # Transformaciones geométricas
```

## 🚀 Instalación y Configuración

### 1. Requisitos Previos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)
- Conexión a internet (para descargar modelos preentrenados)

### 2. Instalar Dependencias Básicas

```bash
# Dependencias básicas para Parte I
pip install -r requirements.txt
```

### 3. Instalar Dependencias para CNN (Parte II)

```bash
# Dependencias adicionales para CNNs preentrenadas
pip install -r requirements-cnn.txt
```

**O instalar manualmente:**
```bash
# Para CPU solamente
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Para GPU CUDA (si tienes GPU NVIDIA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 4. Verificar Instalación de CNNs

```bash
# Ejecutar script de verificación
python verificar_cnn.py
```

Este script verificará:
- ✅ Todas las dependencias instaladas correctamente
- ✅ PyTorch y TorchVision funcionando
- ✅ Disponibilidad de CUDA (si aplica)
- ✅ Capacidad de cargar modelos preentrenados

### 5. Ejecutar el Sistema

```bash
python main_corregido.py
```

## 📋 Funcionalidades Implementadas

### 🖼️ Parte I - Carga y Visualización de Imágenes

- **Listar imágenes**: Muestra todas las imágenes disponibles en el dataset
- **Cargar imagen específica**: Permite cargar una imagen por índice o nombre
- **Visualizar imagen**: Muestra la imagen actual usando matplotlib
- **Galería de imágenes**: Visualización múltiple del dataset
- **Estadísticas**: Información detallada de cada imagen

### ⚙️ Parte I - Preprocesamiento

- **Preprocesamiento completo**: Redimensiona a 224x224 + normalización [0,1]
- **Redimensionamiento**: Solo cambia el tamaño a 224x224 píxeles
- **Normalización**: Solo normaliza valores al rango [0,1]
- **Comparación visual**: Muestra antes/después del preprocesamiento
- **Verificación**: Confirma que el preprocesamiento se aplicó correctamente
- **Preparación para CNN**: Añade dimensión de batch para modelos

### 🧠 Parte II - Redes CNN Preentrenadas

#### **Modelos Implementados:**
- **MobileNetV2**: Ligero (~14MB), optimizado para dispositivos móviles
- **ResNet50**: Red residual (~98MB), balance rendimiento/precisión
- **VGG16**: Arquitectura clásica (~528MB), máxima robustez

#### **Funcionalidades CNN:**
- **Carga automática**: Descarga y configura modelos preentrenados
- **Predicción individual**: Análisis con un modelo específico
- **Comparación múltiple**: Ejecuta todos los modelos simultáneamente
- **Análisis de consenso**: Identifica la predicción más votada
- **Visualización avanzada**: 3 figuras separadas para máxima legibilidad
- **Guardado automático**: Resultados en JSON y PNG de alta calidad

#### **Interpretación Automática:**
- **Niveles de confianza**: Alta (≥67%), Media (33-66%), Baja (<33%)
- **Análisis de acuerdo**: Porcentaje de consenso entre modelos
- **Recomendaciones**: Automáticas basadas en nivel de consenso
- **Estadísticas avanzadas**: Media, mediana, desviación estándar

### 🔍 Análisis del Dataset

- **Estadísticas generales**: Análisis completo del dataset
- **Tipos de archivo**: Distribución de formatos de imagen
- **Dimensiones**: Análisis de tamaños de imágenes
- **Descripción del problema**: Identificación del tipo de clasificación

### 📓 Notebook Jupyter Completo

- **Parcial_Vision_Artificial_CNN.ipynb**: Implementación completa del parcial
- **Todas las partes incluidas**: I, II, III y IV
- **Código ejecutable**: Listo para ejecutar en Jupyter
- **Respuestas teóricas**: Preguntas conceptuales respondidas

## 🎮 Uso del Menú Interactivo

Al ejecutar `python main_corregido.py`, se presenta un menú con las siguientes opciones:

```
📋 MENÚ PRINCIPAL - PARCIAL VISIÓN ARTIFICIAL
1. 📁 Carga y Visualización de Imágenes (Parte I)
2. ⚙️ Preprocesamiento de Imágenes (Parte I)
3. 🧠 Redes CNN Preentrenadas (Parte II) ⭐ NUEVO
4. 🔍 Análisis del Dataset
5. 📊 Información del Sistema
6. ❌ Salir
```

### **Submenú CNN (Opción 3):**
```
🧠 REDES CNN PREENTRENADAS
1. 🔄 Cargar Modelos CNN
2. 🎯 Predicción Individual
3. 🔍 Comparar Modelos (Recomendado)
4. 📊 Análisis Completo con Estadísticas
5. 🔧 Configuraciones
6. ↩️ Regresar
```

### **Flujo de Trabajo Recomendado:**

#### **Para Parte I:**
1. **Análisis del Dataset** (Opción 4): Comprende las características del dataset
2. **Carga de Imágenes** (Opción 1): Carga y visualiza imágenes específicas
3. **Preprocesamiento** (Opción 2): Aplica transformaciones requeridas

#### **Para Parte II (CNNs):**
1. **Verificar dependencias**: `python verificar_cnn.py`
2. **Acceder CNNs** (Opción 3): Entrar al submenú CNN
3. **Cargar modelos** (Subopción 1): Carga MobileNetV2, ResNet50, VGG16
4. **Comparar modelos** (Subopción 3): ⭐ **Análisis completo recomendado**

#### **Para Examen Completo:**
1. **Notebook Jupyter**: Abrir `Parcial_Vision_Artificial_CNN.ipynb`
2. **Ejecutar celdas**: Sigue el orden secuencial
3. **Responder preguntas**: Partes teóricas incluidas

## 🔧 Módulos Principales

### `GestorImagenes`
- Carga imágenes del dataset
- Proporciona visualización y análisis
- Maneja diferentes formatos de imagen

### `PreprocesadorParcial`
- Implementa preprocesamiento específico del parcial
- Redimensionamiento a 224x224 píxeles
- Normalización al rango [0,1]
- Verificación de correctness

### `OperacionesAritmeticas`
- Operaciones matemáticas en imágenes
- Normalización y desnormalización
- Ajustes de brillo y contraste

### `OperacionesGeometricas`
- Transformaciones geométricas
- Redimensionamiento inteligente
- Rotaciones, traslaciones, etc.

### `RedesPreentrenadas` ⭐ **NUEVO**
- Implementa CNNs preentrenadas con ImageNet
- Carga automática de MobileNetV2, ResNet50, VGG16
- Comparación múltiple y análisis de consenso
- Visualización avanzada con 3 figuras separadas
- Guardado automático de resultados

## 📊 Preprocesamiento Implementado

### Redimensionamiento
- **Objetivo**: 224×224 píxeles
- **Método**: Interpolación por área (cv2.INTER_AREA)
- **Beneficio**: Compatible con modelos CNN preentrenados

### Normalización
- **Rango objetivo**: [0, 1]
- **Fórmula**: `valor_normalizado = valor_original / 255.0`
- **Beneficio**: Optimiza el entrenamiento de redes neuronales

## 🎓 Preparación para Siguientes Partes

El sistema prepara las imágenes para las siguientes partes del parcial:

- **Parte II**: Aplicación de CNN preentrenada (MobileNetV2, ResNet50, VGG16)
- **Parte III**: Análisis con preprocesamiento adicional
- **Parte IV**: Análisis conceptual

## 📊 Rendimiento y Resultados

### ⏱️ Tiempo de Carga de Modelos CNN (primera vez):
- **MobileNetV2**: ~30 segundos
- **ResNet50**: ~45 segundos  
- **VGG16**: ~90 segundos

### 🚀 Tiempo de Predicción:
- **CPU**: 1-3 segundos por modelo
- **GPU**: 0.1-0.5 segundos por modelo

### 📈 Ejemplo de Resultados CNN:
```
🔍 ANÁLISIS DE CONSENSO

🎯 Clase más votada: golden_retriever
🤝 Modelos en acuerdo: MOBILENETV2, RESNET50, VGG16
📈 Nivel de acuerdo: 100.0%

🔍 PREDICCIONES DETALLADAS:
• MOBILENETV2: golden_retriever          (87.4%)
• RESNET50   : golden_retriever          (92.1%)
• VGG16      : golden_retriever          (89.7%)

📊 Recomendación: Predicción altamente confiable
```

## 🐛 Solución de Problemas

### **Problema**: Error al ejecutar `python verificar_cnn.py`
```bash
# Solución 1: Instalar dependencias CNN
pip install -r requirements-cnn.txt

# Solución 2: Instalación manual PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### **Problema**: "ModuleNotFoundError: No module named 'torch'"
```bash
# CPU solamente
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# GPU CUDA (si tienes GPU NVIDIA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### **Problema**: Memoria insuficiente al cargar modelos
- Cargue modelos uno por vez
- Use MobileNetV2 para recursos limitados
- Cierre otras aplicaciones

### **Problema**: Predicciones inconsistentes
- Verifique que la imagen esté preprocesada
- Use múltiples modelos para validación
- Revise si la imagen contiene objetos de ImageNet

### **Problema**: Error: "No se encontró el directorio 'images'"
- Verifica que el directorio `images/` existe
- Asegúrate de que contiene las imágenes del dataset

### **Problema**: Error de dependencias básicas
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 📚 Documentación Adicional

### 📄 Archivos de Documentación:
- **`README_CNN_PREENTRENADAS.md`**: Guía detallada de CNNs preentrenadas
- **`Parcial_Vision_Artificial_CNN.ipynb`**: Notebook completo del parcial
- **`requirements.txt`**: Dependencias básicas del sistema
- **`requirements-cnn.txt`**: Dependencias específicas para CNNs
- **`verificar_cnn.py`**: Script de verificación de instalación

### 🔍 Clases de ImageNet Soportadas:
Los modelos están entrenados en ImageNet con 1000 clases incluyendo:
- **Animales**: perros, gatos, aves, mamíferos marinos
- **Vehículos**: automóviles, motocicletas, aviones, barcos
- **Objetos**: electrodomésticos, herramientas, instrumentos
- **Alimentos**: frutas, verduras, platos preparados
- **Plantas**: flores, árboles, hongos

Para mejores resultados, use imágenes que contengan objetos de estas categorías.

### 🎓 Preparación para Evaluación:

#### **Para el Parcial de 2 horas:**
1. **Ejecutar verificación**: `python verificar_cnn.py`
2. **Abrir notebook**: `Parcial_Vision_Artificial_CNN.ipynb`
3. **Tener imágenes**: En directorio `images/`
4. **Sistema listo**: `python main_corregido.py`

#### **Entregables Incluidos:**
- ✅ **Parte I**: Carga y preprocesamiento completo
- ✅ **Parte II**: CNNs preentrenadas (MobileNetV2, ResNet50, VGG16)
- ✅ **Parte III**: Preprocesamiento adicional implementado
- ✅ **Parte IV**: Respuestas teóricas en notebook
- ✅ **Visualizaciones**: Gráficos profesionales separados
- ✅ **Documentación**: READMEs completos
- ✅ **Notebook**: Jupyter listo para ejecución

## 🎯 Comandos Rápidos de Instalación

### **Instalación Completa (Recomendado):**
```bash
# 1. Clonar/descargar proyecto
cd parcial3-vision_artificial

# 2. Instalar dependencias básicas
pip install -r requirements.txt

# 3. Instalar dependencias CNN
pip install -r requirements-cnn.txt

# 4. Verificar instalación
python verificar_cnn.py

# 5. Ejecutar sistema
python main_corregido.py
```

### **Solo CNN (si ya tienes lo básico):**
```bash
pip install -r requirements-cnn.txt
python verificar_cnn.py
```

### **Instalación Mínima CNN:**
```bash
pip install torch torchvision Pillow
python verificar_cnn.py
```

## 📚 Próximos Pasos Completados ✅

- ✅ **Implementar Parte II**: Uso de CNN preentrenadas para clasificación
- ✅ **Implementar Parte III**: Preprocesamiento adicional y análisis
- ✅ **Responder Parte IV**: Preguntas teóricas sobre CNN
- ✅ **Visualizaciones**: Gráficos profesionales mejorados
- ✅ **Sistema completo**: Todas las partes integradas

## 👨‍💻 Autor

**Sistema Completo de Parcial de Visión Artificial con CNNs**  
Universidad del Quindío - 8vo Semestre  
Noviembre 2025

### 📋 **Implementación Completa:**
- ✅ **Todas las 4 partes** del parcial implementadas
- ✅ **3 modelos CNN** preentrenados (MobileNetV2, ResNet50, VGG16)  
- ✅ **Notebook Jupyter** completo con respuestas teóricas
- ✅ **Visualizaciones profesionales** con gráficos separados
- ✅ **Sistema interactivo** con menús intuitivos
- ✅ **Documentación completa** con guías de instalación

---

**📝 Nota**: Este sistema implementa **completamente todos los requisitos** del parcial de Visión Artificial, proporcionando una solución robusta y profesional para análisis de imágenes con CNNs preentrenadas. Listo para evaluación académica.
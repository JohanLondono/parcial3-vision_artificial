#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo de Visualización Estadística para CNNs
===========================================

Genera gráficos avanzados para visualizar estadísticas y comparaciones
de resultados de modelos CNN preentrenados.

Universidad del Quindío - Visión Artificial
Autor: Sistema de Visión Artificial  
Fecha: Noviembre 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import json
import os
from datetime import datetime


class VisualizadorEstadisticasCNN:
    """Clase para crear visualizaciones estadísticas avanzadas de CNNs."""
    
    def __init__(self):
        """Inicializar el visualizador."""
        # Configurar estilo de gráficos
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Colores para cada modelo
        self.colores_modelos = {
            'mobilenet': '#FF6B6B',
            'resnet50': '#4ECDC4', 
            'vgg16': '#45B7D1',
            'alexnet': '#96CEB4',
            'densenet': '#FFEAA7'
        }
        
        print("📊 Visualizador de estadísticas CNN inicializado")
    
    def crear_grafico_comparacion_completo(self, resultados_comparacion, imagen_np=None, guardar=True):
        """
        Crea un gráfico completo de comparación entre modelos.
        
        Args:
            resultados_comparacion (dict): Resultados de comparación de modelos
            imagen_np (numpy.ndarray): Imagen original (opcional)
            guardar (bool): Si guardar el gráfico
        """
        try:
            # Crear figura con múltiples subplots
            fig = plt.figure(figsize=(20, 16))
            
            # 1. Imagen original (si está disponible)
            if imagen_np is not None:
                plt.subplot(3, 4, 1)
                plt.imshow(imagen_np)
                plt.title('Imagen Analizada', fontsize=14, fontweight='bold')
                plt.axis('off')
            
            # 2. Gráfico de confianzas principales
            self._grafico_confianzas_principales(resultados_comparacion, plt.subplot(3, 4, 2))
            
            # 3. Gráfico de consenso
            self._grafico_consenso(resultados_comparacion, plt.subplot(3, 4, 3))
            
            # 4. Distribución de predicciones
            self._grafico_distribucion_predicciones(resultados_comparacion, plt.subplot(3, 4, 4))
            
            # 5. Heatmap de similitud entre modelos
            self._heatmap_similitud(resultados_comparacion, plt.subplot(3, 4, 5))
            
            # 6. Gráfico de radar de rendimiento
            self._grafico_radar_rendimiento(resultados_comparacion, plt.subplot(3, 4, 6))
            
            # 7. Top predicciones por modelo
            self._grafico_top_predicciones(resultados_comparacion, plt.subplot(3, 4, 7))
            
            # 8. Análisis de incertidumbre
            self._grafico_incertidumbre(resultados_comparacion, plt.subplot(3, 4, 8))
            
            # 9. Timeline de predicciones
            self._grafico_timeline_predicciones(resultados_comparacion, plt.subplot(3, 4, 9))
            
            # 10. Estadísticas de diversidad
            self._grafico_diversidad(resultados_comparacion, plt.subplot(3, 4, 10))
            
            # 11-12. Panel de información y métricas
            self._panel_informacion(resultados_comparacion, plt.subplot(3, 4, (11, 12)))
            
            # Título general
            plt.suptitle('ANÁLISIS ESTADÍSTICO COMPLETO - COMPARACIÓN DE MODELOS CNN', 
                        fontsize=20, fontweight='bold', y=0.98)
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.93)
            
            # Guardar si se solicita
            if guardar:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                nombre_archivo = f"analisis_estadistico_completo_{timestamp}.png"
                ruta_completa = os.path.join("resultados_cnn", nombre_archivo)
                
                # Crear directorio si no existe
                os.makedirs("resultados_cnn", exist_ok=True)
                
                plt.savefig(ruta_completa, dpi=300, bbox_inches='tight')
                print(f"📊 Gráfico guardado: {ruta_completa}")
            
            plt.show()
            
        except Exception as e:
            print(f"❌ Error creando gráfico completo: {e}")
    
    def _grafico_confianzas_principales(self, resultados, ax):
        """Gráfico de barras con confianzas principales."""
        modelos = list(resultados['resultados'].keys())
        confianzas = [resultados['resultados'][m]['prediccion_principal']['porcentaje'] 
                     for m in modelos]
        
        colores = [self.colores_modelos.get(m, '#gray') for m in modelos]
        
        bars = ax.bar([m.upper() for m in modelos], confianzas, color=colores, alpha=0.8)
        
        # Añadir valores en las barras
        for bar, conf in zip(bars, confianzas):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{conf:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title('Confianza de Predicción Principal', fontweight='bold')
        ax.set_ylabel('Confianza (%)')
        ax.set_ylim(0, max(confianzas) * 1.2)
        ax.grid(axis='y', alpha=0.3)
    
    def _grafico_consenso(self, resultados, ax):
        """Gráfico circular del nivel de consenso."""
        consenso = resultados['consenso']
        nivel_acuerdo = consenso['nivel_acuerdo']
        
        # Datos para el gráfico circular
        sizes = [nivel_acuerdo * 100, (1 - nivel_acuerdo) * 100]
        labels = ['Consenso', 'Desacuerdo']
        colors = ['#2ECC71', '#E74C3C']
        explode = (0.1, 0)
        
        wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, colors=colors,
                                         autopct='%1.1f%%', shadow=True, startangle=90)
        
        ax.set_title(f'Nivel de Consenso\nClase: {consenso["clase_mas_votada"]}', 
                    fontweight='bold')
    
    def _grafico_distribucion_predicciones(self, resultados, ax):
        """Histograma de distribución de confianzas."""
        todas_confianzas = []
        
        for modelo_result in resultados['resultados'].values():
            for pred in modelo_result['predicciones']:
                todas_confianzas.append(pred['porcentaje'])
        
        ax.hist(todas_confianzas, bins=15, alpha=0.7, color='#3498DB', edgecolor='black')
        ax.axvline(np.mean(todas_confianzas), color='red', linestyle='--', 
                  label=f'Media: {np.mean(todas_confianzas):.1f}%')
        
        ax.set_title('Distribución de Confianzas', fontweight='bold')
        ax.set_xlabel('Confianza (%)')
        ax.set_ylabel('Frecuencia')
        ax.legend()
        ax.grid(alpha=0.3)
    
    def _heatmap_similitud(self, resultados, ax):
        """Heatmap de similitud entre modelos."""
        modelos = list(resultados['resultados'].keys())
        n_modelos = len(modelos)
        
        # Calcular matriz de similitud basada en clases predichas
        matriz_similitud = np.zeros((n_modelos, n_modelos))
        
        for i, modelo1 in enumerate(modelos):
            for j, modelo2 in enumerate(modelos):
                if i == j:
                    matriz_similitud[i][j] = 1.0
                else:
                    # Similitud basada en si predicen la misma clase principal
                    clase1 = resultados['resultados'][modelo1]['prediccion_principal']['clase']
                    clase2 = resultados['resultados'][modelo2]['prediccion_principal']['clase']
                    
                    if clase1 == clase2:
                        matriz_similitud[i][j] = 1.0
                    else:
                        # Similitud basada en overlap de top-3 predicciones
                        pred1 = set([p['clase'] for p in resultados['resultados'][modelo1]['predicciones'][:3]])
                        pred2 = set([p['clase'] for p in resultados['resultados'][modelo2]['predicciones'][:3]])
                        overlap = len(pred1.intersection(pred2)) / len(pred1.union(pred2))
                        matriz_similitud[i][j] = overlap
        
        im = ax.imshow(matriz_similitud, cmap='RdYlGn', vmin=0, vmax=1)
        ax.set_xticks(range(n_modelos))
        ax.set_yticks(range(n_modelos))
        ax.set_xticklabels([m.upper() for m in modelos])
        ax.set_yticklabels([m.upper() for m in modelos])
        
        # Añadir valores de similitud
        for i in range(n_modelos):
            for j in range(n_modelos):
                ax.text(j, i, f'{matriz_similitud[i][j]:.2f}', 
                       ha="center", va="center", fontweight='bold')
        
        ax.set_title('Similitud entre Modelos', fontweight='bold')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    def _grafico_radar_rendimiento(self, resultados, ax):
        """Gráfico de radar comparando múltiples métricas."""
        modelos = list(resultados['resultados'].keys())
        
        # Métricas a evaluar
        metricas = ['Confianza Max', 'Consistencia', 'Diversidad', 'Estabilidad']
        
        # Calcular métricas para cada modelo
        datos_radar = {}
        for modelo in modelos:
            pred_principal = resultados['resultados'][modelo]['prediccion_principal']
            predicciones = resultados['resultados'][modelo]['predicciones']
            
            # Normalizar métricas a 0-1
            confianza_max = pred_principal['porcentaje'] / 100
            consistencia = 1 - np.std([p['porcentaje'] for p in predicciones[:3]]) / 100
            diversidad = len(set([p['clase'] for p in predicciones[:5]])) / 5
            estabilidad = 1 - (max([p['porcentaje'] for p in predicciones[:3]]) - 
                              min([p['porcentaje'] for p in predicciones[:3]])) / 100
            
            datos_radar[modelo] = [confianza_max, consistencia, diversidad, estabilidad]
        
        # Crear gráfico radar
        angulos = np.linspace(0, 2 * np.pi, len(metricas), endpoint=False).tolist()
        angulos += angulos[:1]  # Cerrar el círculo
        
        for modelo, valores in datos_radar.items():
            valores += valores[:1]  # Cerrar el círculo
            color = self.colores_modelos.get(modelo, '#gray')
            ax.plot(angulos, valores, 'o-', linewidth=2, label=modelo.upper(), color=color)
            ax.fill(angulos, valores, alpha=0.25, color=color)
        
        ax.set_xticks(angulos[:-1])
        ax.set_xticklabels(metricas)
        ax.set_ylim(0, 1)
        ax.set_title('Radar de Rendimiento', fontweight='bold')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
    
    def _grafico_top_predicciones(self, resultados, ax):
        """Gráfico de barras horizontales con top predicciones."""
        # Recopilar todas las clases únicas predichas
        todas_clases = set()
        for modelo_result in resultados['resultados'].values():
            for pred in modelo_result['predicciones'][:3]:
                todas_clases.add(pred['clase'])
        
        clases_ordenadas = sorted(list(todas_clases))[:8]  # Top 8 clases más comunes
        
        # Preparar datos para el gráfico
        modelos = list(resultados['resultados'].keys())
        datos_clases = {clase: [] for clase in clases_ordenadas}
        
        for clase in clases_ordenadas:
            for modelo in modelos:
                # Buscar la confianza de esta clase en este modelo
                confianza = 0
                for pred in resultados['resultados'][modelo]['predicciones'][:5]:
                    if pred['clase'] == clase:
                        confianza = pred['porcentaje']
                        break
                datos_clases[clase].append(confianza)
        
        # Crear gráfico de barras agrupadas
        x = np.arange(len(clases_ordenadas))
        width = 0.8 / len(modelos)
        
        for i, modelo in enumerate(modelos):
            valores = [datos_clases[clase][i] for clase in clases_ordenadas]
            color = self.colores_modelos.get(modelo, '#gray')
            ax.bar(x + i * width, valores, width, label=modelo.upper(), color=color, alpha=0.8)
        
        ax.set_xlabel('Clases Predichas')
        ax.set_ylabel('Confianza (%)')
        ax.set_title('Top Predicciones por Modelo', fontweight='bold')
        ax.set_xticks(x + width * (len(modelos) - 1) / 2)
        ax.set_xticklabels([c.replace('clase_', 'C') for c in clases_ordenadas], rotation=45)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    def _grafico_incertidumbre(self, resultados, ax):
        """Análisis de incertidumbre por modelo."""
        modelos = list(resultados['resultados'].keys())
        entropias = []
        
        for modelo in modelos:
            predicciones = resultados['resultados'][modelo]['predicciones']
            # Calcular entropía como medida de incertidumbre
            probabilidades = np.array([p['confianza'] for p in predicciones])
            probabilidades = probabilidades / np.sum(probabilidades)  # Normalizar
            entropia = -np.sum(probabilidades * np.log(probabilidades + 1e-10))
            entropias.append(entropia)
        
        colores = [self.colores_modelos.get(m, '#gray') for m in modelos]
        bars = ax.bar([m.upper() for m in modelos], entropias, color=colores, alpha=0.8)
        
        ax.set_title('Nivel de Incertidumbre (Entropía)', fontweight='bold')
        ax.set_ylabel('Entropía')
        ax.grid(axis='y', alpha=0.3)
        
        # Añadir interpretación
        for bar, entropia in zip(bars, entropias):
            altura = bar.get_height()
            if entropia < 1:
                interpretacion = "Baja"
                color_texto = 'green'
            elif entropia < 2:
                interpretacion = "Media" 
                color_texto = 'orange'
            else:
                interpretacion = "Alta"
                color_texto = 'red'
                
            ax.text(bar.get_x() + bar.get_width()/2, altura + 0.05,
                   interpretacion, ha='center', va='bottom', 
                   fontweight='bold', color=color_texto)
    
    def _grafico_timeline_predicciones(self, resultados, ax):
        """Timeline mostrando evolución de confianzas."""
        modelos = list(resultados['resultados'].keys())
        
        for i, modelo in enumerate(modelos):
            predicciones = resultados['resultados'][modelo]['predicciones'][:5]
            confianzas = [p['porcentaje'] for p in predicciones]
            posiciones = list(range(1, len(confianzas) + 1))
            
            color = self.colores_modelos.get(modelo, '#gray')
            ax.plot(posiciones, confianzas, 'o-', linewidth=2, 
                   label=modelo.upper(), color=color, markersize=8)
        
        ax.set_xlabel('Ranking de Predicción')
        ax.set_ylabel('Confianza (%)')
        ax.set_title('Evolución de Confianzas (Top 5)', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(1, 6))
    
    def _grafico_diversidad(self, resultados, ax):
        """Análisis de diversidad de predicciones."""
        modelos = list(resultados['resultados'].keys())
        
        # Calcular métricas de diversidad
        clases_unicas = []
        spread_confianza = []
        
        for modelo in modelos:
            predicciones = resultados['resultados'][modelo]['predicciones'][:5]
            
            # Número de clases únicas en top-5
            clases_unicas.append(len(set([p['clase'] for p in predicciones])))
            
            # Spread de confianzas (rango)
            confianzas = [p['porcentaje'] for p in predicciones]
            spread_confianza.append(max(confianzas) - min(confianzas))
        
        # Gráfico de dispersión
        colores = [self.colores_modelos.get(m, '#gray') for m in modelos]
        scatter = ax.scatter(clases_unicas, spread_confianza, c=colores, s=200, alpha=0.7)
        
        # Etiquetas de modelos
        for i, modelo in enumerate(modelos):
            ax.annotate(modelo.upper(), (clases_unicas[i], spread_confianza[i]),
                       xytext=(5, 5), textcoords='offset points', fontweight='bold')
        
        ax.set_xlabel('Clases Únicas (Top-5)')
        ax.set_ylabel('Spread de Confianza (%)')
        ax.set_title('Diversidad de Predicciones', fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def _panel_informacion(self, resultados, ax):
        """Panel con información y métricas generales."""
        ax.axis('off')
        
        # Información general
        consenso = resultados['consenso']
        num_modelos = len(resultados['resultados'])
        
        # Calcular estadísticas
        todas_confianzas = []
        for modelo_result in resultados['resultados'].values():
            todas_confianzas.append(modelo_result['prediccion_principal']['porcentaje'])
        
        confianza_media = np.mean(todas_confianzas)
        confianza_std = np.std(todas_confianzas)
        confianza_max = max(todas_confianzas)
        confianza_min = min(todas_confianzas)
        
        # Crear texto informativo
        info_texto = f"""
📊 RESUMEN ESTADÍSTICO GENERAL

🔢 MÉTRICAS BÁSICAS:
   • Modelos analizados: {num_modelos}
   • Clase consenso: {consenso['clase_mas_votada']}
   • Nivel de acuerdo: {consenso['nivel_acuerdo']*100:.1f}%

📈 ESTADÍSTICAS DE CONFIANZA:
   • Media: {confianza_media:.2f}%
   • Desviación: {confianza_std:.2f}%
   • Máxima: {confianza_max:.2f}%
   • Mínima: {confianza_min:.2f}%

🎯 EVALUACIÓN DE RESULTADOS:
"""
        
        # Evaluación automática
        if consenso['nivel_acuerdo'] >= 0.8:
            info_texto += "   • EXCELENTE: Alto consenso entre modelos\n"
            info_texto += "   • Predicción muy confiable\n"
        elif consenso['nivel_acuerdo'] >= 0.6:
            info_texto += "   • BUENO: Consenso aceptable\n"
            info_texto += "   • Predicción moderadamente confiable\n"
        elif consenso['nivel_acuerdo'] >= 0.4:
            info_texto += "   • REGULAR: Consenso limitado\n"
            info_texto += "   • Revisar imagen y preprocesamiento\n"
        else:
            info_texto += "   • BAJO: Sin consenso claro\n"
            info_texto += "   • Imagen ambigua o compleja\n"
        
        if confianza_std < 5:
            info_texto += "   • Consistencia alta entre modelos\n"
        elif confianza_std < 15:
            info_texto += "   • Consistencia media entre modelos\n"
        else:
            info_texto += "   • Alta variabilidad entre modelos\n"
        
        # Mostrar información
        ax.text(0.05, 0.95, info_texto, transform=ax.transAxes, 
                fontsize=12, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.8))
    
    def crear_grafico_evolucion_predicciones(self, resultados_comparacion, guardar=True):
        """Crea un gráfico específico para la evolución de predicciones."""
        try:
            plt.figure(figsize=(15, 10))
            
            modelos = list(resultados_comparacion['resultados'].keys())
            
            # Subplot 1: Gráfico de líneas con top-5 predicciones
            plt.subplot(2, 2, 1)
            for modelo in modelos:
                predicciones = resultados_comparacion['resultados'][modelo]['predicciones'][:5]
                confianzas = [p['porcentaje'] for p in predicciones]
                rankings = list(range(1, len(confianzas) + 1))
                
                color = self.colores_modelos.get(modelo, '#gray')
                plt.plot(rankings, confianzas, 'o-', linewidth=3, 
                        label=modelo.upper(), color=color, markersize=8)
            
            plt.xlabel('Ranking de Predicción')
            plt.ylabel('Confianza (%)')
            plt.title('Evolución de Confianzas por Ranking', fontsize=14, fontweight='bold')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Subplot 2: Comparación de predicción principal
            plt.subplot(2, 2, 2)
            confianzas_principales = [resultados_comparacion['resultados'][m]['prediccion_principal']['porcentaje'] 
                                    for m in modelos]
            colores = [self.colores_modelos.get(m, '#gray') for m in modelos]
            
            bars = plt.bar([m.upper() for m in modelos], confianzas_principales, 
                          color=colores, alpha=0.8, edgecolor='black', linewidth=1)
            
            # Añadir valores en las barras
            for bar, conf in zip(bars, confianzas_principales):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{conf:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            plt.ylabel('Confianza (%)')
            plt.title('Predicción Principal por Modelo', fontsize=14, fontweight='bold')
            plt.grid(axis='y', alpha=0.3)
            
            # Subplot 3: Distribución de todas las confianzas
            plt.subplot(2, 2, 3)
            for modelo in modelos:
                predicciones = resultados_comparacion['resultados'][modelo]['predicciones'][:5]
                confianzas = [p['porcentaje'] for p in predicciones]
                color = self.colores_modelos.get(modelo, '#gray')
                
                plt.hist(confianzas, bins=5, alpha=0.6, label=modelo.upper(), 
                        color=color, edgecolor='black')
            
            plt.xlabel('Confianza (%)')
            plt.ylabel('Frecuencia')
            plt.title('Distribución de Confianzas', fontsize=14, fontweight='bold')
            plt.legend()
            
            # Subplot 4: Análisis de consenso visual
            plt.subplot(2, 2, 4)
            consenso = resultados_comparacion['consenso']
            
            # Gráfico de pastel para consenso
            sizes = [consenso['nivel_acuerdo'] * 100, (1 - consenso['nivel_acuerdo']) * 100]
            labels = ['Consenso', 'Desacuerdo']
            colors = ['#2ECC71', '#E74C3C']
            
            plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
                   startangle=90, shadow=True)
            plt.title(f'Consenso: {consenso["clase_mas_votada"]}', fontsize=14, fontweight='bold')
            
            plt.suptitle('ANÁLISIS DETALLADO DE EVOLUCIÓN DE PREDICCIONES', 
                        fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            if guardar:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                nombre_archivo = f"evolucion_predicciones_{timestamp}.png"
                ruta_completa = os.path.join("resultados_cnn", nombre_archivo)
                
                os.makedirs("resultados_cnn", exist_ok=True)
                plt.savefig(ruta_completa, dpi=300, bbox_inches='tight')
                print(f"📈 Gráfico de evolución guardado: {ruta_completa}")
            
            plt.show()
            
        except Exception as e:
            print(f"❌ Error creando gráfico de evolución: {e}")
    
    def cargar_y_visualizar_resultados(self, ruta_archivo_json):
        """Carga resultados desde archivo JSON y crea visualizaciones."""
        try:
            with open(ruta_archivo_json, 'r', encoding='utf-8') as f:
                resultados = json.load(f)
            
            print(f"📂 Cargados resultados desde: {ruta_archivo_json}")
            
            # Crear visualizaciones
            self.crear_grafico_comparacion_completo(resultados)
            self.crear_grafico_evolucion_predicciones(resultados)
            
            return resultados
            
        except Exception as e:
            print(f"❌ Error cargando archivo: {e}")
            return None
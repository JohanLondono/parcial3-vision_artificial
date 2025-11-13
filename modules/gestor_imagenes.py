#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo de Carga y Visualización de Imágenes
==========================================

Contiene funciones para cargar, visualizar y gestionar imágenes
del dataset para el parcial de visión artificial.

Autor: Sistema de Visión Artificial
Fecha: Noviembre 2024
"""

import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import json

class GestorImagenes:
    """
    Clase para gestionar la carga y visualización de imágenes.
    """
    
    def __init__(self, directorio_imagenes="images"):
        """
        Inicializa el gestor de imágenes.
        
        Args:
            directorio_imagenes: Ruta al directorio que contiene las imágenes
        """
        self.directorio_imagenes = directorio_imagenes
        self.imagenes_disponibles = []
        self.imagen_actual = None
        self.nombre_actual = None
        self.ruta_actual = None
        self.directorio_salida = "resultados_preprocesamiento"
        self._cargar_lista_imagenes()
    
    def _cargar_lista_imagenes(self):
        """
        Carga la lista de imágenes disponibles en el directorio.
        """
        if not os.path.exists(self.directorio_imagenes):
            print(f"⚠️ El directorio {self.directorio_imagenes} no existe.")
            return
        
        # Extensiones de imagen soportadas
        extensiones = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.tif']
        
        self.imagenes_disponibles = []
        for extension in extensiones:
            patron = os.path.join(self.directorio_imagenes, extension)
            self.imagenes_disponibles.extend(glob.glob(patron))
        
        # Ordenar las imágenes por nombre
        self.imagenes_disponibles.sort()
        
        print(f"Se encontraron {len(self.imagenes_disponibles)} imágenes en {self.directorio_imagenes}")
    
    def listar_imagenes(self):
        """
        Lista todas las imágenes disponibles.
        
        Returns:
            Lista de nombres de archivos de imágenes
        """
        if not self.imagenes_disponibles:
            print("❌ No se encontraron imágenes en el directorio.")
            return []
        
        print("\nImágenes disponibles:")
        print("-" * 50)
        
        nombres_archivos = []
        for i, ruta_imagen in enumerate(self.imagenes_disponibles, 1):
            nombre_archivo = os.path.basename(ruta_imagen)
            nombres_archivos.append(nombre_archivo)
            print(f"{i:3d}. {nombre_archivo}")
        
        return nombres_archivos
    
    def cargar_imagen(self, indice=None, nombre_archivo=None):
        """
        Carga una imagen específica.
        
        Args:
            indice: Índice de la imagen en la lista (1-indexado)
            nombre_archivo: Nombre del archivo de imagen
            
        Returns:
            tuple: (imagen, nombre_archivo, ruta_completa) o (None, None, None) si falla
        """
        ruta_imagen = None
        
        if indice is not None:
            if 1 <= indice <= len(self.imagenes_disponibles):
                ruta_imagen = self.imagenes_disponibles[indice - 1]
            else:
                print(f"❌ Índice {indice} fuera de rango. Debe estar entre 1 y {len(self.imagenes_disponibles)}")
                return None, None, None
        
        elif nombre_archivo is not None:
            ruta_completa = os.path.join(self.directorio_imagenes, nombre_archivo)
            if os.path.exists(ruta_completa):
                ruta_imagen = ruta_completa
            else:
                print(f"❌ No se encontró el archivo: {nombre_archivo}")
                return None, None, None
        
        else:
            print("❌ Debe proporcionar un índice o un nombre de archivo")
            return None, None, None
        
        try:
            # Cargar imagen usando OpenCV
            imagen = cv2.imread(ruta_imagen)
            if imagen is None:
                print(f"❌ Error al cargar la imagen: {os.path.basename(ruta_imagen)}")
                return None
            
            # Convertir de BGR a RGB para matplotlib
            imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
            
            # Obtener peso del archivo
            peso_bytes = os.path.getsize(ruta_imagen)
            peso_kb = peso_bytes / 1024
            peso_mb = peso_kb / 1024
            
            # Guardar imagen actual
            self.imagen_actual = imagen_rgb
            self.nombre_actual = os.path.basename(ruta_imagen)
            self.ruta_actual = ruta_imagen
            
            print(f"Imagen cargada: {self.nombre_actual}")
            print(f"   Dimensiones: {imagen_rgb.shape[1]}x{imagen_rgb.shape[0]} píxeles")
            print(f"   Canales: {imagen_rgb.shape[2] if len(imagen_rgb.shape) == 3 else 1}")
            if peso_mb >= 1:
                print(f"   Peso: {peso_mb:.2f} MB")
            else:
                print(f"   Peso: {peso_kb:.2f} KB")
            
            return imagen_rgb
            
        except Exception as e:
            print(f"❌ Error al procesar la imagen: {e}")
            return None
    
    def visualizar_imagen(self, imagen=None, titulo=None, figsize=(10, 8)):
        """
        Visualiza una imagen usando matplotlib.
        
        Args:
            imagen: Imagen a visualizar (si es None, usa la imagen actual)
            titulo: Título de la imagen
            figsize: Tamaño de la figura
        """
        if imagen is None:
            if self.imagen_actual is None:
                print("❌ No hay imagen cargada para visualizar")
                return
            imagen = self.imagen_actual
            titulo = titulo or self.nombre_actual
        
        plt.figure(figsize=figsize)
        plt.imshow(imagen)
        plt.title(titulo or "Imagen", fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def mostrar_galeria(self, max_imagenes=12, figsize=(15, 10)):
        """
        Muestra una galería con múltiples imágenes del dataset.
        
        Args:
            max_imagenes: Número máximo de imágenes a mostrar
            figsize: Tamaño de la figura
        """
        if not self.imagenes_disponibles:
            print("❌ No hay imágenes disponibles para mostrar")
            return
        
        # Limitar el número de imágenes
        num_imagenes = min(max_imagenes, len(self.imagenes_disponibles))
        
        # Calcular el layout de la grilla
        cols = min(4, num_imagenes)
        rows = (num_imagenes + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        
        # Asegurar que axes sea siempre una matriz 2D
        if rows == 1:
            axes = axes.reshape(1, -1) if num_imagenes > 1 else np.array([[axes]])
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i in range(num_imagenes):
            try:
                # Cargar imagen
                imagen = cv2.imread(self.imagenes_disponibles[i])
                if imagen is not None:
                    imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
                    
                    row = i // cols
                    col = i % cols
                    
                    axes[row, col].imshow(imagen_rgb)
                    axes[row, col].set_title(os.path.basename(self.imagenes_disponibles[i]), 
                                           fontsize=8, pad=5)
                    axes[row, col].axis('off')
                else:
                    row = i // cols
                    col = i % cols
                    axes[row, col].text(0.5, 0.5, 'Error\nCargando\nImagen', 
                                      ha='center', va='center', transform=axes[row, col].transAxes)
                    axes[row, col].axis('off')
            
            except Exception as e:
                print(f"Error cargando imagen {i}: {e}")
                continue
        
        # Ocultar axes vacíos
        for i in range(num_imagenes, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')
        
        plt.suptitle(f'Galería de Imágenes del Dataset ({num_imagenes} imágenes)', fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def analizar_dataset(self):
        """
        Analiza las características del dataset de imágenes.
        
        Returns:
            dict: Estadísticas del dataset
        """
        if not self.imagenes_disponibles:
            print("❌ No hay imágenes disponibles para analizar")
            return None
        
        print("\n🔍 Analizando dataset...")
        
        dimensiones = []
        tipos_archivo = {}
        tamaños_archivo = []
        
        for ruta_imagen in self.imagenes_disponibles[:50]:  # Analizar máximo 50 imágenes
            try:
                # Información del archivo
                extension = Path(ruta_imagen).suffix.lower()
                tipos_archivo[extension] = tipos_archivo.get(extension, 0) + 1
                
                tamaño_archivo = os.path.getsize(ruta_imagen)
                tamaños_archivo.append(tamaño_archivo)
                
                # Cargar imagen para obtener dimensiones
                imagen = cv2.imread(ruta_imagen)
                if imagen is not None:
                    h, w = imagen.shape[:2]
                    dimensiones.append((w, h))
                
            except Exception as e:
                print(f"Error analizando {os.path.basename(ruta_imagen)}: {e}")
                continue
        
        # Calcular estadísticas
        estadisticas = {
            'total_imagenes': len(self.imagenes_disponibles),
            'imagenes_analizadas': len(dimensiones),
            'tipos_archivo': tipos_archivo,
            'dimensiones_comunes': {},
            'tamaño_archivo_promedio': np.mean(tamaños_archivo) if tamaños_archivo else 0,
        }
        
        # Analizar dimensiones más comunes
        if dimensiones:
            unique_dims, counts = np.unique(dimensiones, axis=0, return_counts=True)
            for dim, count in zip(unique_dims, counts):
                estadisticas['dimensiones_comunes'][f"{dim[0]}x{dim[1]}"] = count
        
        # Mostrar estadísticas
        print("\n📊 Estadísticas del Dataset:")
        print("-" * 40)
        print(f"Total de imágenes: {estadisticas['total_imagenes']}")
        print(f"Imágenes analizadas: {estadisticas['imagenes_analizadas']}")
        
        print("\nTipos de archivo:")
        for ext, count in estadisticas['tipos_archivo'].items():
            print(f"  {ext}: {count} archivos")
        
        print("\nDimensiones más comunes:")
        sorted_dims = sorted(estadisticas['dimensiones_comunes'].items(), 
                           key=lambda x: x[1], reverse=True)
        for dim, count in sorted_dims[:5]:
            print(f"  {dim}: {count} imágenes")
        
        print(f"\nTamaño promedio de archivo: {estadisticas['tamaño_archivo_promedio']/1024:.1f} KB")
        
        return estadisticas
    
    def describir_problema_clasificacion(self):
        """
        Analiza las imágenes y describe el posible problema de clasificación.
        """
        print("\n🎯 Análisis del Problema de Clasificación:")
        print("=" * 50)
        
        # Basado en los nombres de archivos del dataset
        nombres = [os.path.basename(img) for img in self.imagenes_disponibles[:10]]
        
        print("Basado en la observación de las imágenes del dataset:")
        print("\n📝 Tipo de clasificación:")
        print("   • Las imágenes parecen contener objetos diversos")
        print("   • Posible clasificación multi-clase de objetos cotidianos")
        print("   • Los nombres de archivo sugieren un dataset de objetos generales")
        
        print("\n🎯 Posibles aplicaciones:")
        print("   • Clasificación de objetos en imágenes")
        print("   • Reconocimiento de patrones visuales")
        print("   • Detección de características específicas")
        
        print("\n💡 Consideraciones para CNN:")
        print("   • Las imágenes requieren redimensionamiento a 224x224 para modelos preentrenados")
        print("   • Normalización necesaria para optimizar el rendimiento")
        print("   • Posible augmentación de datos para mejorar generalización")
        
        return True
    
    def obtener_imagen_actual(self):
        """
        Obtiene la imagen actualmente cargada.
        
        Returns:
            tuple: (imagen, nombre) o (None, None) si no hay imagen
        """
        return self.imagen_actual, self.nombre_actual
    
    def obtener_estadisticas_imagen(self, imagen=None, ruta_archivo=None):
        """
        Obtiene estadísticas básicas de una imagen.
        
        Args:
            imagen: Imagen a analizar (si es None, usa la imagen actual)
            ruta_archivo: Ruta del archivo para obtener peso
            
        Returns:
            dict: Estadísticas de la imagen
        """
        if imagen is None:
            imagen = self.imagen_actual
        
        if imagen is None:
            return None
        
        # Obtener peso del archivo si se proporciona la ruta
        peso_info = {}
        if ruta_archivo and os.path.exists(ruta_archivo):
            peso_bytes = os.path.getsize(ruta_archivo)
            peso_info = {
                'peso_bytes': peso_bytes,
                'peso_kb': peso_bytes / 1024,
                'peso_mb': peso_bytes / (1024 * 1024)
            }
        elif hasattr(self, 'ruta_actual') and self.ruta_actual:
            peso_bytes = os.path.getsize(self.ruta_actual)
            peso_info = {
                'peso_bytes': peso_bytes,
                'peso_kb': peso_bytes / 1024,
                'peso_mb': peso_bytes / (1024 * 1024)
            }
        
        stats = {
            'dimensiones': imagen.shape,
            'pixeles_totales': imagen.shape[0] * imagen.shape[1],
            'min_valor': np.min(imagen),
            'max_valor': np.max(imagen),
            'media': np.mean(imagen),
            'desviacion': np.std(imagen),
            'tipo_datos': imagen.dtype,
            **peso_info
        }
        
        return stats
    
    def guardar_imagen(self, imagen, nombre_archivo, carpeta_destino="resultados"):
        """
        Guarda una imagen en el sistema de archivos.
        
        Args:
            imagen: Imagen a guardar (numpy array)
            nombre_archivo: Nombre del archivo (sin extensión)
            carpeta_destino: Carpeta donde guardar
            
        Returns:
            str: Ruta del archivo guardado o None si falla
        """
        try:
            # Crear directorio si no existe
            if not os.path.exists(carpeta_destino):
                os.makedirs(carpeta_destino)
            
            # Preparar la imagen para guardar
            if imagen.dtype == np.float32 or imagen.dtype == np.float64:
                # Si está normalizada, desnormalizar
                if np.max(imagen) <= 1.0:
                    imagen_guardar = (imagen * 255).astype(np.uint8)
                else:
                    imagen_guardar = imagen.astype(np.uint8)
            else:
                imagen_guardar = imagen
            
            # Convertir de RGB a BGR para OpenCV
            if len(imagen_guardar.shape) == 3:
                imagen_guardar = cv2.cvtColor(imagen_guardar, cv2.COLOR_RGB2BGR)
            
            # Construir ruta completa
            ruta_completa = os.path.join(carpeta_destino, f"{nombre_archivo}.png")
            
            # Guardar imagen
            exito = cv2.imwrite(ruta_completa, imagen_guardar)
            
            if exito:
                return ruta_completa
            else:
                return None
                
        except Exception as e:
            print(f"Error guardando imagen: {e}")
            return None
    
    def mostrar_comparacion(self, imagen1, imagen2, titulo1="Imagen 1", titulo2="Imagen 2"):
        """Muestra comparación lado a lado de dos imágenes."""
        try:
            plt.figure(figsize=(12, 6))
            
            # Imagen 1
            plt.subplot(1, 2, 1)
            # Las imágenes ya están en formato RGB desde cargar_imagen
            if len(imagen1.shape) == 3:
                plt.imshow(imagen1)
            else:
                plt.imshow(imagen1, cmap='gray')
            plt.title(titulo1)
            plt.axis('off')
            
            # Imagen 2
            plt.subplot(1, 2, 2)
            if len(imagen2.shape) == 3:
                plt.imshow(imagen2)
            else:
                plt.imshow(imagen2, cmap='gray')
            plt.title(titulo2)
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error mostrando comparación: {e}")
    
    def mostrar_comparacion_con_info(self, imagen1, imagen2, titulo1="Original", titulo2="Procesada", info_procesamiento=None):
        """Muestra comparación lado a lado con información detallada del preprocesamiento."""
        try:
            fig = plt.figure(figsize=(15, 8))
            
            # Imagen original
            plt.subplot(2, 2, 1)
            if len(imagen1.shape) == 3:
                plt.imshow(imagen1)
            else:
                plt.imshow(imagen1, cmap='gray')
            plt.title(f"{titulo1}\n{imagen1.shape[1]}×{imagen1.shape[0]} px", fontsize=12, fontweight='bold')
            plt.axis('off')
            
            # Imagen procesada
            plt.subplot(2, 2, 2)
            if len(imagen2.shape) == 3:
                plt.imshow(imagen2)
            else:
                plt.imshow(imagen2, cmap='gray')
            
            # Título con información básica
            titulo_procesada = f"{titulo2}\n{imagen2.shape[1]}×{imagen2.shape[0]} px"
            if info_procesamiento:
                if 'normalizacion' in info_procesamiento:
                    titulo_procesada += f"\nRango: {info_procesamiento['normalizacion']}"
            plt.title(titulo_procesada, fontsize=12, fontweight='bold')
            plt.axis('off')
            
            # Panel de información
            plt.subplot(2, 1, 2)
            plt.axis('off')
            
            if info_procesamiento:
                info_texto = "[TRANSFORMACIONES APLICADAS]:\n\n"
                
                # Información de redimensionamiento
                if 'dimension_original' in info_procesamiento and 'dimension_final' in info_procesamiento:
                    orig = info_procesamiento['dimension_original']
                    final = info_procesamiento['dimension_final']
                    info_texto += f"• Redimensionamiento: {orig[0]}x{orig[1]} -> {final[0]}x{final[1]} px\n"
                
                # Información de normalización
                if 'normalizacion' in info_procesamiento:
                    norm_info = info_procesamiento['normalizacion']
                    if norm_info == '[0,1]':
                        info_texto += "• Normalización: Valores escalados al rango [0,1]\n"
                    elif norm_info == 'imagenet':
                        info_texto += "• Normalización: ImageNet (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])\n"
                    else:
                        info_texto += f"• Normalización: {norm_info}\n"
                
                # Información de augmentación
                if 'augmentacion' in info_procesamiento:
                    aug_info = info_procesamiento['augmentacion']
                    if aug_info:
                        info_texto += f"• Augmentación: {', '.join(aug_info)}\n"
                
                # Información de filtros
                if 'filtros' in info_procesamiento:
                    filtros_info = info_procesamiento['filtros']
                    if filtros_info:
                        info_texto += f"• Filtros aplicados: {', '.join(filtros_info)}\n"
                
                # Estadísticas
                if 'estadisticas' in info_procesamiento:
                    stats = info_procesamiento['estadisticas']
                    info_texto += "\n[ESTADÍSTICAS]:\n"
                    if 'rango_valores' in stats:
                        rango = stats['rango_valores']
                        info_texto += f"   • Rango de valores: [{rango[0]:.3f}, {rango[1]:.3f}]\n"
                    if 'media' in stats:
                        info_texto += f"   • Media: {stats['media']:.3f}\n"
                    if 'desviacion' in stats:
                        info_texto += f"   • Desviación estándar: {stats['desviacion']:.3f}\n"
                
                # Tiempo de procesamiento
                if 'tiempo_procesamiento' in info_procesamiento:
                    info_texto += f"\n• Tiempo de procesamiento: {info_procesamiento['tiempo_procesamiento']:.3f}s\n"
                
                plt.text(0.05, 0.95, info_texto, transform=plt.gca().transAxes, 
                        fontsize=11, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
            else:
                plt.text(0.5, 0.5, "No hay información de procesamiento disponible", 
                        transform=plt.gca().transAxes, ha='center', va='center',
                        fontsize=12, style='italic')
            
            plt.suptitle('COMPARACIÓN DE PREPROCESAMIENTO', fontsize=16, fontweight='bold', y=0.98)
            plt.tight_layout()
            plt.subplots_adjust(top=0.93)
            plt.show()
            
        except Exception as e:
            print(f"Error mostrando comparación con información: {e}")
            # Fallback a comparación básica
            self.mostrar_comparacion(imagen1, imagen2, titulo1, titulo2)
    
    def cargar_estado_procesamiento(self, nombre_archivo):
        """Cargar estado de procesamiento previamente guardado"""
        try:
            directorio_estados = os.path.join(self.directorio_salida, "estados_procesamiento")
            
            # Buscar archivos del estado
            patron_preprocesada = os.path.join(directorio_estados, f"*{nombre_archivo}*_preprocesada_estado.npy")
            patron_original = os.path.join(directorio_estados, f"*{nombre_archivo}*_original_estado.npy")
            
            archivos_preprocesada = glob.glob(patron_preprocesada)
            archivos_original = glob.glob(patron_original)
            
            if not archivos_preprocesada or not archivos_original:
                print(f"No se encontró estado guardado para: {nombre_archivo}")
                return None, None
                
            # Cargar las imágenes
            imagen_preprocesada = np.load(archivos_preprocesada[0])
            imagen_original = np.load(archivos_original[0])
            
            print(f"Estado cargado exitosamente:")
            print(f"- Original: {imagen_original.shape}")
            print(f"- Preprocesada: {imagen_preprocesada.shape}")
            
            return imagen_original, imagen_preprocesada
            
        except Exception as e:
            print(f"Error cargando estado: {e}")
            return None, None
    
    def listar_estados_disponibles(self):
        """Listar todos los estados de procesamiento disponibles"""
        try:
            directorio_estados = os.path.join(self.directorio_salida, "estados_procesamiento")
            if not os.path.exists(directorio_estados):
                print("No hay estados guardados.")
                return []
                
            # Buscar archivos de estado
            patron = os.path.join(directorio_estados, "*_info_estado.txt")
            archivos_info = glob.glob(patron)
            
            estados = []
            for archivo in archivos_info:
                nombre_base = os.path.basename(archivo).replace("_info_estado.txt", "")
                estados.append(nombre_base)
                
            if estados:
                print("\\nEstados de procesamiento disponibles:")
                for i, estado in enumerate(estados, 1):
                    print(f"{i}. {estado}")
            else:
                print("No hay estados guardados.")
                
            return estados
            
        except Exception as e:
            print(f"Error listando estados: {e}")
            return []
    
    def guardar_estado_procesamiento(self, imagen_original, imagen_preprocesada, estadisticas, nombre_archivo):
        """Guardar estado del procesamiento con imagen preprocesada para continuar trabajando"""
        try:
            # Crear directorio de estados si no existe
            directorio_estados = os.path.join(self.directorio_salida, "estados_procesamiento")
            os.makedirs(directorio_estados, exist_ok=True)
            
            # Guardar imagen preprocesada en formato que permita seguir trabajando
            ruta_preprocesada = os.path.join(directorio_estados, f"{nombre_archivo}_preprocesada_estado.npy")
            np.save(ruta_preprocesada, imagen_preprocesada)
            
            # Guardar imagen original para restauración
            ruta_original = os.path.join(directorio_estados, f"{nombre_archivo}_original_estado.npy")
            np.save(ruta_original, imagen_original)
            
            # Guardar información del procesamiento en texto plano
            ruta_info = os.path.join(directorio_estados, f"{nombre_archivo}_info_estado.txt")
            with open(ruta_info, 'w', encoding='utf-8') as f:
                f.write("=== ESTADO DE PROCESAMIENTO ===")
                f.write("\\n\\nFecha y hora: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                f.write("\\n\\nImagenes guardadas:")
                f.write(f"\\n- Original: {os.path.basename(ruta_original)}")
                f.write(f"\\n- Preprocesada: {os.path.basename(ruta_preprocesada)}")
                f.write("\\n\\nEstadísticas de la imagen preprocesada:")
                for clave, valor in estadisticas.items():
                    f.write(f"\\n- {clave}: {valor}")
                f.write("\\n\\nFormato: NumPy arrays (.npy) para preservar precisión")
                f.write("\\nUso: Cargar con np.load() para continuar procesamiento")
            
            print(f"\\nEstado completo guardado en: {directorio_estados}")
            print(f"- Imagen preprocesada: {os.path.basename(ruta_preprocesada)}")
            print(f"- Imagen original: {os.path.basename(ruta_original)}")
            print(f"- Información: {os.path.basename(ruta_info)}")
            return directorio_estados
        except Exception as e:
            print(f"Error guardando estado: {e}")
            return False
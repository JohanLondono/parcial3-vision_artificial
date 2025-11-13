#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulos para el Parcial de Visión Artificial
==========================================

Paquete que contiene todos los módulos necesarios para el parcial
de visión artificial con redes neuronales convolucionales.

Módulos incluidos:
- gestor_imagenes: Carga y visualización de imágenes del dataset
- operaciones_aritmeticas: Operaciones aritméticas para preprocesamiento
- operaciones_geometricas: Transformaciones geométricas 
- preprocesador_parcial: Preprocesamiento específico del parcial
- preprocesador_avanzado_cnn: Preprocesamiento avanzado para CNNs
- redes_preentrenadas: Modelos CNN preentrenados (MobileNetV2, ResNet50, VGG16)

Autor: Sistema de Visión Artificial
Fecha: Noviembre 2024
"""

from .gestor_imagenes import GestorImagenes
from .operaciones_aritmeticas import OperacionesAritmeticas
from .operaciones_geometricas import OperacionesGeometricas
from .preprocesador_parcial import PreprocesadorParcial
from .preprocesador_avanzado_cnn import PreprocesadorAvanzadoCNN
from .redes_preentrenadas import RedesPreentrenadas

__all__ = [
    'GestorImagenes',
    'OperacionesAritmeticas', 
    'OperacionesGeometricas',
    'PreprocesadorParcial',
    'PreprocesadorAvanzadoCNN',
    'RedesPreentrenadas'
]
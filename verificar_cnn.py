#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de Verificación de Dependencias - CNN Preentrenadas
=========================================================

Verifica que todas las dependencias necesarias para las redes CNN
preentrenadas estén instaladas correctamente.

Universidad del Quindío - Visión Artificial
Fecha: Noviembre 2024
"""

import sys

def verificar_dependencias():
    """Verifica todas las dependencias necesarias."""
    print("🔍 Verificando dependencias para CNNs preentrenadas...\n")
    
    errores = []
    exitosas = []
    
    # Verificar dependencias básicas
    dependencias_basicas = {
        'numpy': 'numpy',
        'cv2': 'opencv-python',
        'matplotlib': 'matplotlib',
        'PIL': 'Pillow'
    }
    
    for modulo, paquete in dependencias_basicas.items():
        try:
            __import__(modulo)
            exitosas.append(f"✅ {paquete}")
        except ImportError:
            errores.append(f"❌ {paquete} - pip install {paquete}")
    
    # Verificar PyTorch
    try:
        import torch
        import torchvision
        exitosas.append(f"✅ PyTorch {torch.__version__}")
        exitosas.append(f"✅ TorchVision {torchvision.__version__}")
        
        # Verificar si CUDA está disponible
        if torch.cuda.is_available():
            exitosas.append(f"✅ CUDA disponible - GPU: {torch.cuda.get_device_name(0)}")
        else:
            exitosas.append("ℹ️  CUDA no disponible - usando CPU")
            
    except ImportError:
        errores.append("❌ PyTorch/TorchVision - pip install torch torchvision")
    
    # Mostrar resultados
    print("DEPENDENCIAS INSTALADAS:")
    print("-" * 40)
    for dep in exitosas:
        print(dep)
    
    if errores:
        print("\nDEPENDENCIAS FALTANTES:")
        print("-" * 40)
        for error in errores:
            print(error)
            
        print("\n📋 COMANDO DE INSTALACIÓN:")
        print("pip install torch torchvision opencv-python matplotlib Pillow")
        
        return False
    else:
        print(f"\n🎉 ¡Todas las dependencias están instaladas! ({len(exitosas)} verificadas)")
        return True

def probar_carga_modelo():
    """Prueba cargar un modelo pequeño para verificar funcionalidad."""
    try:
        print("\n🧪 Probando carga de modelo...")
        
        import torch
        import torchvision.models as models
        
        # Intentar cargar MobileNetV2 (el más pequeño)
        modelo = models.mobilenet_v2(weights='MobileNet_V2_Weights.IMAGENET1K_V1')
        modelo.eval()
        
        # Probar una predicción dummy
        entrada_dummy = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            salida = modelo(entrada_dummy)
        
        print("✅ Modelo cargado y funcionando correctamente")
        print(f"   Forma de salida: {salida.shape}")
        return True
        
    except Exception as e:
        print(f"❌ Error probando modelo: {e}")
        return False

def main():
    """Función principal de verificación."""
    print("=" * 60)
    print("     VERIFICACIÓN DE DEPENDENCIAS CNN")
    print("=" * 60)
    
    # Verificar dependencias
    deps_ok = verificar_dependencias()
    
    if deps_ok:
        # Probar funcionalidad
        modelo_ok = probar_carga_modelo()
        
        if modelo_ok:
            print("\n🚀 ¡Sistema listo para usar CNNs preentrenadas!")
            print("\nPuedes ejecutar main_corregido.py y usar la opción:")
            print("'3. Redes CNN Preentrenadas'")
        else:
            print("\n⚠️  Las dependencias están instaladas pero hay problemas")
            print("   Revisa tu conexión a internet para descargar modelos")
    else:
        print("\n❌ Instala las dependencias faltantes antes de continuar")
        
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
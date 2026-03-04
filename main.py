from ultralytics import YOLO

def main():
    # 1. Carrega o modelo YOLOv26 Nano Pose (pré-treinado)
    model = YOLO('yolo26n-pose.pt') 

    
    model.train(
        data='vaca_data.yaml',
        epochs=100,
        imgsz=640,
        device='mps',      
        batch=32, 
        workers=8,         
        cache='ram',       
        rect=True,
        amp=False,           
        # --- Seus Ajustes Finos de Augmentation ---
        fliplr=0.0,      
        mosaic=0.0,      
        degrees=5.0,     
        translate=0.1,   
        scale=0.5,       
        shear=2.0,       
        perspective=0.0, 
        hsv_h=0.015,     
        hsv_s=0.7,       
        hsv_v=0.4,
        # --- hiperparâmetros ---      
        lr0=0.001,
        lrf=0.001, 
        flipud=1.0,
        # --- Estabilidade do YOLO26 ---
        optimizer='AdamW',   # Mais rápido para convergir em arquiteturas C3k2/C2PSA
        project='trabalho_vaca',
        name='resultados',
        exist_ok=True
    )
    
    # 3. Validação Final (Evaluate)
    metrics = model.val()
    
    print("--- Resultados da Avaliação ---")
    print(f"Precisão dos Pontos (mAP50-95): {metrics.pose.map}")
    print(f"Resultados salvos na pasta: trabalho_vaca/resultados")

if __name__ == '__main__':
    main()
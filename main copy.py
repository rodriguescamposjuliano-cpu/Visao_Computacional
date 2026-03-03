from ultralytics import YOLO

def main():
    # 1. Carrega o modelo YOLOv8 Nano Pose (pré-treinado)
    # Ele é ideal para começar pois é leve e o M4 vai processar com extrema rapidez.
    model = YOLO('yolo26n-pose.pt') 

    # 2. Inicia o Treinamento
    # O YOLOv8 já faz a validação (Evaluate) automaticamente ao final de cada época.
    # model.train(
    #     data='vaca_data.yaml',    # O seu arquivo de configuração
    #     epochs=100,               # 100 épocas é um bom começo para convergência
    #     imgsz=640,                # Resolução das imagens
    #     device='mps',             # <--- FORÇA o uso da GPU do seu Mac M4
    #     batch=16,                 # Quantidade de imagens por vez na memória
    #     project='trabalho_vaca',  # Pasta onde os resultados serão salvos
    #     name='experimento_8pts',  # Nome da pasta específica deste treino
    #     save=True                 # Salva os pesos (.pt) e os gráficos de performance
    # )

    # model.train(
    #     data='vaca_data.yaml',
    #     epochs=100,
    #     imgsz=640,
    #     device='mps',      
    #     batch=64,          # Forçamos 64 para ocupar a GPU de verdade
    #     workers=0,         # Mantemos 0 no Mac para evitar conflito de memória
    #     cache='ram',       
    #     rect=True,         # Treino retangular (muito mais rápido)
    #     amp=False,         
    #     # --- Estabilidade ---
    #     fliplr=0.0,        
    #     mosaic=0.0,        # Desativar mosaic no final ajuda na velocidade e precisão de pose
    #     # --- Outros ---
    #     project='trabalho_vaca',
    #     name='resultados_8pts',
    #     exist_ok=True
    # )

    model.train(
        data='vaca_data.yaml',
        epochs=100,
        imgsz=640,
        device='mps',      
        batch=32,            # REDUZIDO: Para evitar que o M4 use SWAP e perca velocidade
        workers=8,         
        cache='ram',       
        rect=True,           # Essencial para manter a velocidade em imagens largas
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
        # --- hiperparâmetros do YOLOv8 ---      
        lr0=0.001,
        lrf=0.001, 
        flipud=1.0,
        # --- Estabilidade do YOLO26 ---
        optimizer='AdamW',   # Mais rápido para convergir em arquiteturas C3k2/C2PSA
        project='trabalho_vaca',
        name='resultados_8pts_refinado_Y26',
        exist_ok=True
    )
    
    # 3. Validação Final (Evaluate)
    # Após o treino, rodamos uma validação detalhada no conjunto 'val'
    metrics = model.val()
    
    print("--- Resultados da Avaliação ---")
    print(f"Precisão dos Pontos (mAP50-95): {metrics.pose.map}")
    print(f"Resultados salvos na pasta: trabalho_vaca/experimento_8pts")

if __name__ == '__main__':
    main()
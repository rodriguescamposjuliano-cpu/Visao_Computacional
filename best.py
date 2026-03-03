import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = 'runs/pose/trabalho_vaca/resultados/weights/best.pt'
IMG_PATH = 'teste/validacao_baia23_VIPWX.jpg'

CONF_BOX = 0.25
CONF_KPT = 0.30
IMG_SIZE = 512

# Se treinou em grayscale, deixe True
TREINADO_EM_GRAYSCALE = True

# Ordem confirmada:
# 0 withers
# 1 back
# 2 hook up
# 3 hook down
# 4 hip
# 5 tail head
# 6 pin up
# 7 pin down

def calcular_ratio(kpts, conf, min_conf=0.3):
    required = [0, 5, 2, 6]
    for i in required:
        if conf[i] < min_conf:
            return 0

    comprimento = np.linalg.norm(kpts[0] - kpts[5])
    largura = np.linalg.norm(kpts[2] - kpts[6])

    if comprimento <= 0:
        return 0

    return round(float(largura / comprimento), 4)


model = YOLO(MODEL_PATH)
img = cv2.imread(IMG_PATH)

if img is None:
    print("Imagem não encontrada.")
    exit()

if TREINADO_EM_GRAYSCALE:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_input = cv2.merge([gray, gray, gray])
else:
    img_input = img

results = model(
    img_input,
    conf=CONF_BOX,
    imgsz=IMG_SIZE,
    verbose=False
)

for r in results:

    img_display = r.orig_img.copy()

    if r.boxes is None or len(r.boxes) == 0:
        print("Nenhuma vaca detectada.")
        continue

    for i in range(len(r.boxes)):

        conf_box = float(r.boxes.conf[i])
        print(f"\nConfiança: {conf_box:.3f}")

        kpts = r.keypoints.xy[i].cpu().numpy()
        kconf = r.keypoints.conf[i].cpu().numpy()

        ratio = calcular_ratio(kpts, kconf, CONF_KPT)
        print("Ratio:", ratio)

        x1, y1, x2, y2 = r.boxes.xyxy[i].cpu().numpy()
        cv2.rectangle(img_display, (int(x1), int(y1)),
                      (int(x2), int(y2)), (255, 0, 0), 2)

        for j, (x, y) in enumerate(kpts):
            if kconf[j] > CONF_KPT:
                cv2.circle(img_display, (int(x), int(y)),
                           6, (0, 255, 0), -1)
                cv2.putText(img_display, str(j),
                            (int(x)+4, int(y)-4),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 255), 2)

        cv2.putText(img_display, f"Ratio: {ratio}",
                    (int(x1), int(y1)-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)

    cv2.imshow("Top-Down Pose Validation", img_display)
    cv2.waitKey(0)

cv2.destroyAllWindows()

from deepface import DeepFace
import cv2
import os
import glob
import numpy as np

os.environ['DEEPFACE_HOME'] = os.path.join(os.path.expanduser('~'), '.deepface')

KNOWN_FACES_DIR = os.path.join(os.getcwd(), "known_faces")
RECOG_THRESHOLD = 0.40
DISPLAY_THRESHOLD = 0.30  # mínimo score para exibir rótulos/thumbnail

def represent_safe(img, model_name='Facenet', enforce_detection=False, detector_backend='opencv'):
    try:
        return DeepFace.represent(img, model_name=model_name, enforce_detection=enforce_detection, detector_backend=detector_backend, silent=True)
    except TypeError:
        return DeepFace.represent(img, model_name=model_name, enforce_detection=enforce_detection, detector_backend=detector_backend)

def extract_embedding(rep):
    if rep is None:
        return None
    if isinstance(rep, np.ndarray):
        return rep.flatten()
    if isinstance(rep, list):
        if not rep:
            return None
        if all(isinstance(x, (int, float, np.floating, np.integer)) for x in rep):
            return np.array(rep, dtype=np.float32)
        return extract_embedding(rep[0])
    if isinstance(rep, dict):
        for key in ('embedding', 'emb', 'representation', 'rep'):
            if key in rep:
                return extract_embedding(rep[key])
        for v in rep.values():
            if isinstance(v, list) and len(v) > 0 and all(isinstance(x, (int, float, np.floating, np.integer)) for x in v):
                return np.array(v, dtype=np.float32)
        return None
    return None

def cosine_similarity(a, b):
    a = np.array(a).astype(np.float32)
    b = np.array(b).astype(np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return -1
    return float(np.dot(a, b) / denom)

def load_known_embeddings(folder):
    embeddings = []
    if not os.path.exists(folder):
        return embeddings

    per_name = {}
    for f in glob.glob(os.path.join(folder, "*.*")):
        try:
            name = os.path.splitext(os.path.basename(f))[0]
            if '_' in name:
                name = name.split('_')[0]

            emb = represent_safe(f, model_name='Facenet', enforce_detection=False, detector_backend='opencv')
            emb_arr = extract_embedding(emb)
            if emb_arr is None:
                print(f"Aviso: não foi possível extrair embedding de {f}")
                continue

            norm = np.linalg.norm(emb_arr)
            if norm > 0:
                emb_arr = emb_arr / norm

            per_name.setdefault(name, []).append(emb_arr)
        except Exception as e:
            print(f"Falha ao processar {f}: {e}")

    for name, embs in per_name.items():
        try:
            mean_emb = np.mean(np.stack(embs), axis=0)
            norm = np.linalg.norm(mean_emb)
            if norm > 0:
                mean_emb = mean_emb / norm
            embeddings.append({"name": name, "emb": mean_emb})
        except Exception as e:
            print(f"Erro ao agregar embeddings para {name}: {e}")

    return embeddings


def find_face_image(name):
    # Procura por um arquivo em known_faces cujo nome comece com o nome fornecido
    if not os.path.exists(KNOWN_FACES_DIR):
        return None
    for f in glob.glob(os.path.join(KNOWN_FACES_DIR, "*.*")):
        base = os.path.splitext(os.path.basename(f))[0]
        if base.lower().startswith(name.lower()):
            return f
    return None


def non_max_suppression(boxes, overlapThresh=0.3):
    """Remove caixas sobrepostas (lista de (x,y,w,h)). Retorna lista filtrada."""
    if boxes is None or len(boxes) == 0:
        return []

    # converte para array numpy
    boxes_arr = np.array(boxes)
    if boxes_arr.dtype.kind == "i":
        boxes_arr = boxes_arr.astype("float")

    x1 = boxes_arr[:,0]
    y1 = boxes_arr[:,1]
    x2 = boxes_arr[:,0] + boxes_arr[:,2]
    y2 = boxes_arr[:,1] + boxes_arr[:,3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    pick = []
    while len(idxs) > 0:
        last = idxs[-1]
        pick.append(last)

        xx1 = np.maximum(x1[last], x1[idxs[:-1]])
        yy1 = np.maximum(y1[last], y1[idxs[:-1]])
        xx2 = np.minimum(x2[last], x2[idxs[:-1]])
        yy2 = np.minimum(y2[last], y2[idxs[:-1]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / areas[idxs[:-1]]

        # remove índices com overlap maior que threshold
        idxs = np.delete(idxs, np.concatenate(([len(idxs) - 1], np.where(overlap > overlapThresh)[0])))

    # retorna boxes selecionadas como lista de tuplas inteiras
    return boxes_arr[pick].astype(int).tolist()


def main():
    global RECOG_THRESHOLD
    print("Carregando embeddings conhecidas...")
    known_embeddings = load_known_embeddings(KNOWN_FACES_DIR)
    print(f"Known embeddings carregadas: {len(known_embeddings)}")

    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # lista de thumbnails das pessoas reconhecidas (empilhadas verticalmente)
        thumbnails = []  # cada item: (imagem_bgr, nome)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Ajustes para reduzir falsos positivos: menor scaleFactor, mais vizinhos, minSize maior
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=6, minSize=(80,80))

        # Aplica Non-Maximum Suppression para evitar múltiplas caixas sobrepostas
        if len(faces) > 1:
            try:
                faces = non_max_suppression(faces, overlapThresh=0.35)
            except Exception:
                # em caso de erro, mantém detecções originais
                pass

        for (x, y, w, h) in faces:
            # filtro por proporção (largura/altura) para evitar objetos alongados
            if h == 0:
                continue
            aspect = float(w) / float(h)
            if aspect < 0.6 or aspect > 1.6:
                continue

            # evita faces muito pequenas relativas ao frame
            if w * h < 0.002 * frame.shape[0] * frame.shape[1]:
                continue

            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(frame.shape[1], x + w)
            y2 = min(frame.shape[0], y + h)

            face_img = frame[y1:y2, x1:x2]
            if face_img.size == 0 or (y2 - y1) < 10 or (x2 - x1) < 10:
                continue

            try:
                face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            except Exception:
                face_rgb = face_img

            try:
                face_resized = cv2.resize(face_rgb, (160, 160))
            except Exception:
                face_resized = face_rgb

            try:
                face_emb_raw = represent_safe(face_resized, model_name='Facenet', enforce_detection=False, detector_backend='opencv')
                face_emb = extract_embedding(face_emb_raw)
                if face_emb is None:
                    raise ValueError('Não foi possível extrair embedding')

                best_score = -2.0
                best_name = None
                for k in known_embeddings:
                    sim = cosine_similarity(face_emb, k['emb'])
                    if sim > best_score:
                        best_score = sim
                        best_name = k['name']

                recognized_name = None
                if best_score >= RECOG_THRESHOLD and best_name:
                    recognized_name = best_name

                # Se o score for muito baixo, ignora a detecção para evitar falsos positivos
                if best_score < DISPLAY_THRESHOLD:
                    continue

                color = (0, 255, 0) if recognized_name else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                if recognized_name:
                    # Texto de boas-vindas próximo à face
                    welcome_text = f"Bem-vindo(a), {recognized_name}"
                    cv2.putText(frame, welcome_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    # prepara thumbnail para exibir no canto superior esquerdo
                    img_path = find_face_image(recognized_name)
                    if img_path:
                        try:
                            thumb = cv2.imread(img_path)
                            if thumb is not None:
                                # redimensiona mantendo proporção para altura = 100
                                th_h, th_w = thumb.shape[:2]
                                new_h = 100
                                new_w = int(th_w * (new_h / th_h))
                                thumb = cv2.resize(thumb, (new_w, new_h))
                                # adiciona à lista apenas se nome ainda não estiver
                                if recognized_name not in [n for _, n in thumbnails]:
                                    thumbnails.append((thumb, recognized_name))
                        except Exception:
                            pass
                else:
                    label = f"Unknown ({best_score:.2f})"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            except Exception as e:
                print(f"Erro no reconhecimento: {e}")

        # Desenha thumbnails empilhados verticalmente no canto superior esquerdo
        if thumbnails:
            mx = 10      # margem horizontal
            my = 10      # começa no topo
            gap = 8      # espaço entre thumbnails
            for thumb_img, thumb_name in thumbnails:
                try:
                    th_h, th_w = thumb_img.shape[:2]
                    # verifica se ainda cabe no frame
                    if my + th_h >= frame.shape[0] or mx + th_w >= frame.shape[1]:
                        break
                    frame[my:my+th_h, mx:mx+th_w] = thumb_img
                    cv2.rectangle(frame, (mx, my), (mx+th_w, my+th_h), (255, 255, 255), 2)
                    cv2.putText(frame, thumb_name, (mx, my + th_h + 16),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                    my += th_h + 16 + gap  # avança para o próximo slot
                except Exception:
                    pass

        cv2.imshow('Recognition (only)', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('r'):
            known_embeddings = load_known_embeddings(KNOWN_FACES_DIR)
            print(f"Recarregadas embeddings: {len(known_embeddings)}")
        if key == ord(']'):
            RECOG_THRESHOLD = min(0.99, RECOG_THRESHOLD + 0.05)
            print(f"RECOG_THRESHOLD -> {RECOG_THRESHOLD:.2f}")
        if key == ord('['):
            RECOG_THRESHOLD = max(-1.0, RECOG_THRESHOLD - 0.05)
            print(f"RECOG_THRESHOLD -> {RECOG_THRESHOLD:.2f}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

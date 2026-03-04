# pip install deepface opencv-python

from deepface import DeepFace
import cv2
import os
import glob
import numpy as np

# Dicionário de tradução das emoções
EMOTION_TRANSLATION = {
    'angry': 'Raiva',
    'fear': 'Medo',
    'happy': 'Feliz',
    'sad': 'Triste',
    'surprise': 'Surpresa',
    'neutral': 'Neutro'
}

# Dicionário de tradução de gênero
GENDER_TRANSLATION = {
    'Man': 'Masculino',
    'Woman': 'Feminino'
}

os.environ['DEEPFACE_HOME'] = os.path.join(os.path.expanduser('~'), '.deepface')

KNOWN_FACES_DIR = os.path.join(os.getcwd(), "known_faces")
RECOG_THRESHOLD = 0.40

def cosine_similarity(a, b):
    a = np.array(a).astype(np.float32)
    b = np.array(b).astype(np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return -1
    return float(np.dot(a, b) / denom)


def represent_safe(img, model_name='Facenet', enforce_detection=False, detector_backend='opencv'):
    """Wrapper para DeepFace.represent compatível com versões que não aceitam `silent`."""
    try:
        return DeepFace.represent(img, model_name=model_name, enforce_detection=enforce_detection, detector_backend=detector_backend, silent=True)
    except TypeError:
        return DeepFace.represent(img, model_name=model_name, enforce_detection=enforce_detection, detector_backend=detector_backend)


def extract_embedding(rep):
    """Extrai um vetor de embedding de diferentes formatos retornados por DeepFace.represent.

    Aceita: list, np.ndarray, dict contendo chaves com listas numéricas.
    Retorna: 1D numpy array ou None se não puder extrair.
    """
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


def load_known_embeddings(folder):
    # Retorna uma lista com embeddings médios por pessoa (agrupar múltiplas imagens por nome)
    embeddings = []
    if not os.path.exists(folder):
        return embeddings

    per_name = {}
    for f in glob.glob(os.path.join(folder, "*.*")):
        try:
            name = os.path.splitext(os.path.basename(f))[0]
            # se arquivo nomeado como Nome_1.jpg, extrai prefixo antes do '_' como nome
            if '_' in name:
                name = name.split('_')[0]

            emb = represent_safe(f, model_name='Facenet', enforce_detection=False, detector_backend='opencv')
            emb_arr = extract_embedding(emb)
            if emb_arr is None:
                print(f"Aviso: não foi possível extrair embedding de {f}")
                continue

            # normaliza embedding para unit vector
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

def main():
    global RECOG_THRESHOLD

    print("Carregando modelos... (isso pode demorar na primeira vez)")
    try:
        dummy_frame = np.zeros((48, 48, 3), dtype=np.uint8)
        DeepFace.analyze(dummy_frame, actions=['emotion', 'gender'], enforce_detection=False, detector_backend='opencv', silent=True)
    except Exception:
        print("Modelos serão carregados no primeiro frame detectado")
    print("Modelos prontos! Iniciando detecção...")

    known_embeddings = load_known_embeddings(KNOWN_FACES_DIR)
    print(f"Known embeddings carregadas: {len(known_embeddings)}")

    cap = cv2.VideoCapture(0)
    frame_count = 0
    emotion_text = ""
    last_result = None
    download_error_count = 0
    MAX_DOWNLOAD_ERRORS = 3

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            try:
                if frame_count % 10 == 0:
                    results = DeepFace.analyze(
                        frame,
                        actions=['emotion', 'gender'],
                        enforce_detection=False,
                        detector_backend='opencv',
                        silent=True,
                    )
                    if isinstance(results, dict):
                        results = [results]
                    last_result = results

                if last_result:
                    for result in last_result:
                        emotions = {k: v for k, v in result['emotion'].items() if k != 'disgust'}

                        if emotions:
                            dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
                            emotion_text = EMOTION_TRANSLATION.get(dominant_emotion, dominant_emotion)
                        else:
                            dominant_emotion = 'neutral'
                            emotion_text = 'Neutro'

                        gender_info = result.get('gender', {})
                        if isinstance(gender_info, dict):
                            dominant_gender = result.get('dominant_gender', '')
                            gender_text = GENDER_TRANSLATION.get(dominant_gender, dominant_gender)
                            gender_confidence = gender_info.get(dominant_gender, 0)
                        else:
                            gender_text = "N/A"
                            gender_confidence = 0

                        x = result['region']['x']
                        y = result['region']['y']
                        w = result['region']['w']
                        h = result['region']['h']

                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                        recognized_name = None
                        try:
                            x1 = max(0, int(x))
                            y1 = max(0, int(y))
                            x2 = min(frame.shape[1], int(x + w))
                            y2 = min(frame.shape[0], int(y + h))

                            face_img = frame[y1:y2, x1:x2]
                            if face_img.size != 0 and len(known_embeddings) > 0 and (y2 - y1) > 10 and (x2 - x1) > 10:
                                try:
                                    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                                except Exception:
                                    face_rgb = face_img

                                try:
                                    face_resized = cv2.resize(face_rgb, (160, 160))
                                except Exception:
                                    face_resized = face_rgb

                                face_emb_raw = represent_safe(face_resized, model_name='Facenet', enforce_detection=False, detector_backend='opencv')
                                face_emb = extract_embedding(face_emb_raw)
                                if face_emb is None:
                                    raise ValueError("Não foi possível extrair embedding do recorte de face")

                                best_score = -2.0
                                best_name = None
                                for k in known_embeddings:
                                    sim = cosine_similarity(face_emb, k["emb"])
                                    if sim > best_score:
                                        best_score = sim
                                        best_name = k["name"]

                                if best_score >= RECOG_THRESHOLD and best_name:
                                    recognized_name = best_name
                        except Exception as ex:
                            print(f"Erro no reconhecimento por embeddings: {ex}")

                        if recognized_name:
                            cv2.putText(frame, f"Bem-vindo, {recognized_name}", (x, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                        cv2.putText(frame, emotion_text.upper(), (x, y - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        cv2.putText(frame, f"{gender_text} ({gender_confidence:.1f}%)", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                        y_offset = y + 20
                        for emotion_en, percentage in sorted(emotions.items(), key=lambda x: x[1], reverse=True):
                            if emotion_en == 'disgust':
                                continue
                            emotion_pt = EMOTION_TRANSLATION.get(emotion_en, emotion_en)
                            text = f"{emotion_pt}: {percentage:.1f}%"
                            if percentage > 50:
                                color = (0, 255, 0)
                            elif percentage > 20:
                                color = (0, 255, 255)
                            else:
                                color = (255, 255, 255)
                            cv2.putText(frame, text, (x + w + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                            y_offset += 25

            except Exception as e:
                erro_msg = str(e)
                if "downloading" in erro_msg.lower() or "download" in erro_msg.lower():
                    download_error_count += 1
                    print(f"Erro de download ({download_error_count}/{MAX_DOWNLOAD_ERRORS}): Falha ao baixar modelo")
                    if download_error_count >= MAX_DOWNLOAD_ERRORS:
                        print("\n" + "=" * 70)
                        print("ERRO: Não foi possível baixar o modelo de emoções.")
                        print("=" * 70)
                        print("\nSOLUCAO: Baixe o arquivo manualmente:")
                        print("\n1. Abra o navegador e acesse:")
                        print("   https://github.com/serengil/deepface_models/releases/download/v1.0/facial_expression_model_weights.h5")
                        print("\n2. Salve o arquivo em:")
                        print(r"   C:\Users\wilia\.deepface\.deepface\weights\facial_expression_model_weights.h5")
                        print("\n3. Crie as pastas se necessário e execute novamente")
                        print("=" * 70)
                        break
                else:
                    print(f"Erro na análise: {e}")

            cv2.putText(frame, f"THRESH={RECOG_THRESHOLD:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Real-time Emotion Detection", frame)

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

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
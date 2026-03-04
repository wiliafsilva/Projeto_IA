from deepface import DeepFace
import cv2
import os
import numpy as np

# Traduções
EMOTION_TRANSLATION = {
    'angry': 'Raiva',
    'fear': 'Medo',
    'happy': 'Feliz',
    'sad': 'Triste',
    'surprise': 'Surpresa',
    'neutral': 'Neutro'
}

GENDER_TRANSLATION = {
    'Man': 'Masculino',
    'Woman': 'Feminino'
}

os.environ['DEEPFACE_HOME'] = os.path.join(os.path.expanduser('~'), '.deepface')

PROCESS_EVERY_N = 8  # ajustar para desempenho


def main():
    print("Iniciando detector de EMOÇÕES (apenas) - carregando modelos, aguarde...")

    try:
        DeepFace.analyze(
            np.zeros((48, 48, 3), dtype=np.uint8),
            actions=['emotion', 'gender'],
            enforce_detection=False,
            detector_backend='opencv',
            silent=True,
        )
    except Exception:
        pass

    cap = cv2.VideoCapture(0)
    frame_count = 0
    last_result = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            try:
                if frame_count % PROCESS_EVERY_N == 0:
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
                        emotions = {k: v for k, v in result.get('emotion', {}).items() if k != 'disgust'}
                        if emotions:
                            dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
                            emotion_text = EMOTION_TRANSLATION.get(dominant_emotion, dominant_emotion)
                        else:
                            emotion_text = 'Neutro'

                        dominant_gender = result.get('dominant_gender', '')
                        gender_text = GENDER_TRANSLATION.get(dominant_gender, dominant_gender)

                        region = result.get('region', {})
                        x = int(region.get('x', 0))
                        y = int(region.get('y', 0))
                        w = int(region.get('w', 0))
                        h = int(region.get('h', 0))

                        try:
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        except Exception:
                            pass

                        cv2.putText(frame, emotion_text.upper(), (x, max(0, y - 35)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        cv2.putText(frame, gender_text, (x, max(0, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                        y_offset = y + 20
                        for emotion_en, percentage in sorted(emotions.items(), key=lambda x: x[1], reverse=True):
                            if emotion_en == 'disgust':
                                continue
                            emotion_pt = EMOTION_TRANSLATION.get(emotion_en, emotion_en)
                            text = f"{emotion_pt}: {percentage:.1f}%"
                            if percentage > 50:
                                color = (0, 255, 0)
                            elif percentage > 20:
                                color = (0, 165, 255)
                            else:
                                color = (0, 0, 255)
                            cv2.putText(frame, text, (x + w + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                            y_offset += 22

            except Exception as e:
                print(f"Erro na análise de emoções: {e}")

            cv2.imshow("Emotion Detection (only)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

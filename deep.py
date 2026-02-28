# pip install deepface opencv-python

from deepface import DeepFace
import cv2
import os

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

# Define o diretório de cache para os modelos
os.environ['DEEPFACE_HOME'] = os.path.join(os.path.expanduser('~'), '.deepface')

print("Carregando modelos... (isso pode demorar na primeira vez)")

# Pré-carrega os modelos fazendo uma análise de teste
# Isso força o download apenas uma vez
try:
    dummy_frame = cv2.imread(cv2.__file__.replace('__init__.py', 'data/haarcascade_frontalface_default.xml'))
    if dummy_frame is None:
        # Cria um frame de teste simples
        dummy_frame = cv2.cvtColor(cv2.imread(cv2.samples.findFile("lena.jpg")), cv2.COLOR_BGR2RGB) if os.path.exists(cv2.samples.findFile("lena.jpg")) else None
    
    if dummy_frame is not None:
        DeepFace.analyze(dummy_frame, actions=['emotion', 'gender'], enforce_detection=False, detector_backend='opencv')
except:
    print("Modelos serão carregados no primeiro frame detectado")

print("Modelos prontos! Iniciando detecção...")

# Inicializa webcam
cap = cv2.VideoCapture(0)

frame_count = 0
emotion_text = ""
last_result = None  # Cache do último resultado
download_error_count = 0  # Contador de erros de download
MAX_DOWNLOAD_ERRORS = 3  # Máximo de tentativas antes de parar

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    try:
        # Processa a cada 10 frames (melhora desempenho e evita chamadas excessivas)
        if frame_count % 10 == 0:
            results = DeepFace.analyze(
                frame,
                actions=['emotion', 'gender'],  # Detecta emoção e gênero
                enforce_detection=False,
                detector_backend='opencv',  # mais leve
                silent=True  # Evita logs verbose
            )

            # Garante que seja lista
            if isinstance(results, dict):
                results = [results]

            # Atualiza o cache de resultados
            last_result = results

        # Usa o último resultado válido para desenhar (mesmo em frames não processados)
        if last_result:
            for result in last_result:
                # Remove 'disgust' das emoções
                emotions = {k: v for k, v in result['emotion'].items() if k != 'disgust'}
                
                # Pega a emoção dominante (ignorando disgust)
                if emotions:
                    dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
                    emotion_text = EMOTION_TRANSLATION.get(dominant_emotion, dominant_emotion)
                else:
                    dominant_emotion = 'neutral'
                    emotion_text = 'Neutro'
                
                # Extrai informações de gênero
                gender_info = result.get('gender', {})
                if isinstance(gender_info, dict):
                    # Pega o gênero dominante
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

                # Desenha retângulo
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Escreve emoção dominante (traduzida)
                cv2.putText(
                    frame,
                    emotion_text.upper(),
                    (x, y - 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2
                )
                
                # Escreve gênero detectado
                cv2.putText(
                    frame,
                    f"{gender_text} ({gender_confidence:.1f}%)",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),  # Amarelo
                    2
                )
                
                # Mostra todas as emoções com porcentagens (à direita da face)
                y_offset = y + 20
                for emotion_en, percentage in sorted(emotions.items(), key=lambda x: x[1], reverse=True):
                    # Pula disgust se ainda aparecer
                    if emotion_en == 'disgust':
                        continue
                    emotion_pt = EMOTION_TRANSLATION.get(emotion_en, emotion_en)
                    text = f"{emotion_pt}: {percentage:.1f}%"
                    
                    # Define cor baseada na intensidade
                    if percentage > 50:
                        color = (0, 255, 0)  # Verde para alta
                    elif percentage > 20:
                        color = (0, 255, 255)  # Amarelo para média
                    else:
                        color = (255, 255, 255)  # Branco para baixa
                    
                    cv2.putText(
                        frame,
                        text,
                        (x + w + 10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        1
                    )
                    y_offset += 25

    except Exception as e:
        erro_msg = str(e)
        
        # Detecta erro de download
        if "downloading" in erro_msg.lower() or "download" in erro_msg.lower():
            download_error_count += 1
            print(f"Erro de download ({download_error_count}/{MAX_DOWNLOAD_ERRORS}): Falha ao baixar modelo")
            
            # Se atingiu o limite de erros, para o programa
            if download_error_count >= MAX_DOWNLOAD_ERRORS:
                print("\n" + "="*70)
                print("❌ ERRO: Não foi possível baixar o modelo de emoções.")
                print("="*70)
                print("\n🔧 SOLUÇÃO: Baixe o arquivo manualmente:")
                print("\n1. Abra o navegador e vá para:")
                print("   https://github.com/serengil/deepface_models/releases/download/v1.0/facial_expression_model_weights.h5")
                print("\n2. Salve o arquivo em:")
                print(f"   C:\\Users\\wilia\\.deepface\\.deepface\\weights\\facial_expression_model_weights.h5")
                print("\n3. Crie as pastas se não existirem")
                print("\n4. Execute o programa novamente")
                print("="*70)
                break
        else:
            print(f"Erro na análise: {e}")

    cv2.imshow("Real-time Emotion Detection", frame)

    # Pressione Q para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
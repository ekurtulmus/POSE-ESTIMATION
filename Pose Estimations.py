import cv2
import mediapipe as mp
import time

# Mediapipe modülünden pose çözümünü kullanmak için gerekli kütüphaneleri içe aktarıyoruz.
mpPose = mp.solutions.pose
# Pose çözümünü başlatıyoruz.
pose = mpPose.Pose()

# Çizim yapmak için mediapipe içinden drawing_utils'yi içe aktarıyoruz.
mpDraw = mp.solutions.drawing_utils

# Video yakalama cihazını başlatıyoruz, burada video dosyası adı verilmiş.
cap = cv2.VideoCapture("video ismi")

# Sonsuz döngü, video akışının işlenmesi için kullanılır.
while True:
    # Videodan bir çerçeve başarıyla alınıp alınmadığını kontrol eder.
    success, img = cap.read()

    # Renkleri RGB formatına dönüştürüyoruz.
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Pozları işlemek için mediapipe çözümünü kullanıyoruz.
    results = pose.process(imgRGB)
    print(results.pose_landmarks)

    # Eğer poz noktaları bulunursa, bunları çerçeveye çiziyoruz.
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)


        for id, lm in enumerate(results.pose_landmarks.landmark):
            # Çerçevenin boyutlarını alıyoruz.
            h, w, c = img.shape
            # Noktanın konumunu piksel olarak alıyoruz.
            cx, cy = int(lm.x * w), int(lm.y * h)
            # Eğer nokta dirsek noktası ise, bu noktayı çerçeveye işaretliyoruz.
            if id == 13:
                cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

    # Görüntüyü yeniden boyutlandırıyoruz.
    imgResized = cv2.resize(img, (1000, 1000))
    # Yeniden boyutlandırılmış görüntüyü gösteriyoruz.
    cv2.imshow("Resized Image", imgResized)

    # Belirli bir tuşa basıldığında döngüden çıkmamızı sağlıyor, Burda q tuşunu seçtik.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Videoyu durduruyoruz ve tüm pencereleri kapatıyoruz.
cap.release()
cv2.destroyAllWindows()
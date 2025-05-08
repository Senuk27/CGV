import random
import cv2
import cvzone
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import time
import speech_recognition as sr
import threading
import pyttsx3

# Text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak_result(text):
    engine.say(text)
    engine.runAndWait()

# Initialize camera
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Hand detector
detector = HandDetector(maxHands=1)

# Game state variables
timer = 0
stateResult = False
startGame = False
scores = [0, 0]
resultText = ""
imgAI = None

moves = {
    1: "Rock",
    2: "Paper",
    3: "Scissors",
    4: "Lizard",
    5: "Spock"
}

resultMatrix = [
    [0, -1, 1, 1, -1],
    [1, 0, -1, -1, 1],
    [-1, 1, 0, 1, -1],
    [-1, 1, -1, 0, 1],
    [1, -1, 1, -1, 0]
]

# Voice listener
def listen_for_voice():
    global startGame, initialTime, stateResult
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        print("🎙️ Say 'rock paper scissors shoot' to start...")

        while True:
            audio = recognizer.listen(source)
            try:
                phrase = recognizer.recognize_google(audio).lower()
                if "shoot" in phrase:
                    startGame = True
                    initialTime = time.time()
                    stateResult = False
                    time.sleep(5)
            except:
                pass

threading.Thread(target=listen_for_voice, daemon=True).start()

cv2.namedWindow("Processing", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Processing", 640, 480)

while True:
    imgBG = cv2.imread("Resources/BG2.png")
    success, img = cap.read()
    imgScaled = cv2.resize(img, (0, 0), None, 0.875, 0.875)
    imgScaled = imgScaled[:, 80:480]

    gray = cv2.cvtColor(imgScaled, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    edges = cv2.Canny(thresh, 50, 150)

    hands, img = detector.findHands(imgScaled, draw=True)

    if startGame:
        if not stateResult:
            timer = time.time() - initialTime
            cv2.putText(imgBG, str(int(timer)), (602, 432), cv2.FONT_HERSHEY_PLAIN, 6, (232, 12, 0), 4)

            if timer > 1:
                stateResult = True
                timer = 0

                if hands:
                    playerMove = None
                    hand = hands[0]
                    fingers = detector.fingersUp(hand)

                    if fingers == [0, 0, 0, 0, 0]: playerMove = 1
                    elif fingers == [1, 1, 1, 1, 1]: playerMove = 2
                    elif fingers == [0, 1, 1, 0, 0]: playerMove = 3
                    elif fingers == [0, 1, 0, 0, 0]: playerMove = 4
                    elif fingers == [1, 1, 0, 1, 1]: playerMove = 5

                    aiMove = random.randint(1, 5)
                    imgAI = cv2.imread(f'Resources/{aiMove}.png', cv2.IMREAD_UNCHANGED)

                    if imgAI is not None and imgAI.shape[2] == 3:
                        imgAI = cv2.cvtColor(imgAI, cv2.COLOR_BGR2BGRA)

                    imgBG = cvzone.overlayPNG(imgBG, imgAI, (149, 310))
                    cv2.imshow("BG", imgBG)
                    cv2.waitKey(500)

                    if playerMove and 1 <= playerMove <= 5:
                        result = resultMatrix[playerMove - 1][aiMove - 1]
                        if result == 1:
                            scores[1] += 1
                            resultText = "You Win!"
                        elif result == -1:
                            scores[0] += 1
                            resultText = "AI Wins!"
                        else:
                            resultText = "It's a Tie!"
                    else:
                        resultText = "Gesture Not Recognized!"

                    # Speak short result only
                    threading.Thread(target=speak_result, args=(resultText,), daemon=True).start()

    imgBG[234:654, 795:1195] = imgScaled

    if stateResult and imgAI is not None:
        imgBG = cvzone.overlayPNG(imgBG, imgAI, (149, 310))
        cv2.putText(imgBG, resultText, (560, 410), cv2.FONT_HERSHEY_PLAIN, 2, (232, 12, 0), 4)

    cv2.putText(imgBG, str(scores[0]), (410, 215), cv2.FONT_HERSHEY_PLAIN, 4, (232, 12, 0), 4)
    cv2.putText(imgBG, str(scores[1]), (1112, 215), cv2.FONT_HERSHEY_PLAIN, 4, (232, 12, 0), 4)

    stack = np.hstack((gray, blur, thresh, edges))
    cv2.imshow("Processing", stack)
    cv2.imshow("BG", imgBG)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

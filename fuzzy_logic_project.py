import numpy as np
import skfuzzy as fuzz
import skfuzzy.membership as mf
import matplotlib.pyplot as plt
import pytesseract
import cv2 as cv
import os
import sys
import numpy as np
import math
from typing import Tuple, Union
import pyttsx3

#--------------------LANGUAGE--------------------------
dil= int(input("Please select a language!:\n"
         "1.English\n"
         "2.German\n"
         "3.French\n"))
if dil==1:
    i = 0
elif dil==2:
    i = 2
elif dil==3:
    i = 3
#----------------------------------------------------
def rescaleframe(frame,scale):   #Çerçeve büyüklüğü ayarlama
    height = int(frame.shape[0] * scale)
    widht = int(frame.shape[1] * scale)
    dimensions = (widht,height)
    return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)

def rotate(
        image: np.ndarray, angle: float, background: Union[int, Tuple[int, int, int]]
) -> np.ndarray:
    old_width, old_height = image.shape[:2]
    angle_radian = math.radians(angle)
    width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
    height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[1, 2] += (width - old_width) / 2
    rot_mat[0, 2] += (height - old_height) / 2
    return cv.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=background)

pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
#url = "https://192.168.0.13:8080/video"
url=0
#url = "http://10.30.46.83:8080/video"

capture = cv.VideoCapture(url)
while True:
    isTrue,frame =capture.read()
    #cv.imshow("Video",frame)
    resized = rescaleframe(frame,scale=0.5)
    rotated_frame = rotate(resized, -90, None)
    cv.imshow("Resized",rotated_frame)


    if cv.waitKey(20) & 0XFF == ord("q"):  #pencereyi kapatır.
        sys.exit()
    elif cv.waitKey(20) & 0XFF == ord("a"):
        rotated = rotate(resized, -90, (0, 0, 0))   #rotated yaparsan 38.satıra frame yerine rotated yaz!!!
        cv.imwrite("capture.jpg", rotated)   #Anlık yakalar.
        break

capture.release()
cv.destroyAllWindows()
#-----------------------------------------------------
captured_image = cv.imread("capture.jpg")
capture_image_resized = rescaleframe(captured_image,scale=1)
cv.imshow("captured_image",capture_image_resized)
################# GRAY ##########################
while True:
    if cv.waitKey(20) & 0XFF == ord("s"):  # pencereyi kapatır.
        cv.destroyAllWindows()
        break

gray = cv.cvtColor(capture_image_resized,cv.COLOR_BGR2GRAY)
cv.imshow("captured_image",capture_image_resized)
cv.imshow("Gray",gray)
############## THICKEN LETTERS #####################
while True:
    if cv.waitKey(20) & 0XFF == ord("d"):  # pencereyi kapatır.
        cv.destroyAllWindows()
        break

kernel = np.ones((2,2),np.uint8)
eroded = cv.erode(gray,kernel,iterations = 1)
cv.imshow("captured_image",capture_image_resized)
cv.imshow("Gray",gray)
cv.imshow("Thicken Letters",eroded)
cv.imwrite("Thicken Letters.jpg",eroded)
############ REMOVE SHADOW ####################
while True:
    if cv.waitKey(20) & 0XFF == ord("f"):  # pencereyi kapatır.
        cv.destroyAllWindows()
        break

rgb_planes = cv.split(eroded)

result_planes = []
result_norm_planes = []

for plane in rgb_planes:
    dilated_img = cv.dilate(plane, np.ones((7,7), np.uint8))
    bg_img = cv.medianBlur(dilated_img, 21)
    diff_img = 255 - cv.absdiff(plane, bg_img)
    norm_img = cv.normalize(diff_img,None, alpha=0,\
                             beta=255, norm_type=cv.NORM_MINMAX,\
                             dtype=cv.CV_8UC1)
    result_planes.append(diff_img)
    result_norm_planes.append(norm_img)

result = cv.merge(result_planes)
result_norm = cv.merge(result_norm_planes)

cv.imwrite('shadows_out.png', result)
cv.imwrite('shadows_out_norm.png', result_norm)

cv.imshow('shadows_out', result)
cv.imshow('shadows_out_norm', result_norm)

while True:
    if cv.waitKey(20) & 0XFF == ord("g"):  # pencereyi kapatır.
        cv.destroyAllWindows()
        break
################### IMAGE TO TEXT ########################
text = pytesseract.image_to_string(("shadows_out_norm.png"),lang="tur")
print(text)
################## SAVING TEXT ######################
with open("text.txt", "w", encoding="utf-8") as file:
    file.write(text)

########################### KELIME SAYISI BULMA #####################################
with open('text.txt', "r", encoding="utf-8") as file:
    lines = 0
    words = 0
    characters = 0
    for line in file:
        wordslist = line.split()
        # lines = lines + 1
        words = words + len(wordslist)
        # characters += sum(len(Word) for Word in wordslist)
# print(lines)
print(words)
# print(characters)

########################### FUZZY LOGIC #####################################

# Değişkenlerin oluşturulması
x_wordnumber = np.arange(0, 500, 1)
x_pagenumber = np.arange(0, 400, 1)
y_speechspeed = np.arange(100, 300, 1)

# Üyelik fonksiyonlarının oluşturulması
wordnumber_low = mf.trimf(x_wordnumber, [0, 0, 250])
wordnumber_med = mf.trimf(x_wordnumber, [0, 250, 500])
wordnumber_hig = mf.trimf(x_wordnumber, [250, 500, 500])
pagenumber_low = mf.trapmf(x_pagenumber, [-1, 0, 80, 160])
pagenumber_med = mf.trapmf(x_pagenumber, [80, 160, 240, 320])
pagenumber_hig = mf.trapmf(x_pagenumber, [240, 320, 400, 401])
speechspeed_fast = mf.trimf(y_speechspeed, [150, 250, 9999])
speechspeed_slow = mf.trimf(y_speechspeed, [-9999, 150, 250])

# Veri görselleştirme
fig, (ax0, ax1, ax2) = plt.subplots(nrows = 3, figsize =(6, 10))

ax0.plot(x_wordnumber, wordnumber_low, 'r', linewidth = 2, label = 'Düşük')
ax0.plot(x_wordnumber, wordnumber_med, 'g', linewidth = 2, label = 'Orta')
ax0.plot(x_wordnumber, wordnumber_hig, 'b', linewidth = 2, label = 'Yüksek')
ax0.set_title('Kelime Sayısı')
ax0.legend()

ax1.plot(x_pagenumber, pagenumber_low, 'r', linewidth = 2, label = 'Düşük')
ax1.plot(x_pagenumber, pagenumber_med, 'g', linewidth = 2, label = 'Orta')
ax1.plot(x_pagenumber, pagenumber_hig, 'b', linewidth = 2, label = 'Yüksek')
ax1.set_title('Sayfa Sayısı')
ax1.legend()

ax2.plot(y_speechspeed, speechspeed_fast, 'r', linewidth = 2, label = 'Güçlü')
ax2.plot(y_speechspeed, speechspeed_slow, 'b', linewidth = 2, label = 'Zayıf')
ax2.set_title('Konuşma Hızı')
ax2.legend()

plt.tight_layout()
plt.show()

input_wordnumber = words #yukarıdaki denklemle bulucaz
input_pagenumber = int(input("Please enter page number")) #toplam sayfa sayısı

# Üyelik derecelerinin hesaplanması
word_fit_low = fuzz.interp_membership(x_wordnumber, wordnumber_low, input_wordnumber)
word_fit_med = fuzz.interp_membership(x_wordnumber, wordnumber_med, input_wordnumber)
word_fit_hig = fuzz.interp_membership(x_wordnumber, wordnumber_hig, input_wordnumber)

page_fit_low = fuzz.interp_membership(x_pagenumber, pagenumber_low, input_pagenumber)
page_fit_med = fuzz.interp_membership(x_pagenumber, pagenumber_med, input_pagenumber)
page_fit_hig = fuzz.interp_membership(x_pagenumber, pagenumber_hig, input_pagenumber)

# Kuralların oluşturulması
rule1 = np.fmin(np.fmin(word_fit_low, page_fit_hig), speechspeed_fast)
rule2 = np.fmin(np.fmin(word_fit_hig, page_fit_low), speechspeed_slow)
rule3 = np.fmin(np.fmax(word_fit_hig, page_fit_med), speechspeed_fast)
rule4 = np.fmin(np.fmin(word_fit_low, page_fit_med), speechspeed_slow)

out_fast = np.fmax(rule1, rule3)
out_slow = np.fmax(rule2, rule4)
# Veri görselleştirme
speed0 = np.zeros_like(y_speechspeed)

fig, ax0 = plt.subplots(figsize = (7, 4))
ax0.fill_between(y_speechspeed, speed0, out_slow, facecolor = 'r', alpha = 0.7)
ax0.plot(y_speechspeed, speechspeed_slow, 'r', linestyle = '--')
ax0.fill_between(y_speechspeed, speed0, out_fast, facecolor = 'g', alpha = 0.7)
ax0.plot(y_speechspeed, speechspeed_fast, 'g', linestyle = '--')
ax0.set_title('Okuma Hızı')

plt.tight_layout()
plt.show()

# Durulaştırma
out_hiz = np.fmax(out_slow, out_fast)
defuzzified  = fuzz.defuzz(y_speechspeed, out_hiz, 'centroid')
result = fuzz.interp_membership(y_speechspeed, out_hiz, defuzzified)
# Sonuç
print("(HIZ)Çıkış Değeri:", defuzzified)

# Veri görselleştirme

fig, ax0 = plt.subplots(figsize=(7, 4))

ax0.plot(y_speechspeed, speechspeed_slow, 'b', linewidth = 0.5, linestyle = '--')
ax0.plot(y_speechspeed, speechspeed_fast, 'g', linewidth = 0.5, linestyle = '--')
ax0.fill_between(y_speechspeed, speed0, out_hiz, facecolor = 'Orange', alpha = 0.7)
ax0.plot([defuzzified , defuzzified], [0, result], 'k', linewidth = 1.5, alpha = 0.9)
ax0.set_title('Ağırlık Merkezi ile Durulaştırma')

plt.tight_layout()
plt.show()

########################### pyttsx3 ###################################
engine = pyttsx3.init()
"""VOICE"""
voices = engine.getProperty('voices')  # getting details of current voice
engine.setProperty('voice', voices[i].id)  # [0] [1] = english, [2] = german, [3] = french
engine.setProperty('rate', defuzzified)
os.system("text.txt")
engine.say(text)
engine.runAndWait()

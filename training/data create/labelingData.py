import numpy as np
import cv2
from pathlib import Path
from matplotlib import pyplot as plt
np.set_printoptions(suppress=True)

roi = []


kind = "image"
file_path = "M_30HD/"
ext = ".jpg"
directory_path = "frameset/"

counter = 1
size = 6 - len(str(counter))             #string ve string_counter fotoğraf dosyalarımız image000001 diye oldugundan 0'ları ayarlamak için yaptık.
string_counter = size*"0" + str(counter)

while True:
    if Path((directory_path + file_path + kind + str(string_counter) + ext)).is_file():  #dosyadan veri çekme(is_file kontrol ediyor var olup olmadıgını)
    #if Path((directory_path+file_path+kind+str(counter)+ext)).is_file():
            filename = directory_path+file_path + kind + str(string_counter) +ext
            #filename = directory_path+file_path + kind + str(counter) +ext

            picture = cv2.imread(filename)
            gray_picture = cv2.cvtColor(picture ,cv2.COLOR_BGR2GRAY)
            roi_picture = gray_picture.copy()
            roi_picture = roi_picture[170:171,150:278]  #ilgilendiğim alanın ayarladım (1,128) boyutunda.

            cv2.line(gray_picture ,(150,170),(278,170),(255,0,0),1) #Alacagım satır vektörünün yerini ayarlamak için kullandım.
            cv2.imshow('roi_picture',gray_picture)

            save_or_exit = cv2.waitKey(0)

            if save_or_exit & 0xFF==ord("5"):
                counter = counter + 1
                size = 6 - len(str(counter))
                string_counter = size*"0" + str(counter)
                continue
            elif save_or_exit & 0xFF==ord("0"):
                roi_picture = np.append(roi_picture,0)
            elif save_or_exit & 0xFF==ord("1"):
                roi_picture = np.append(roi_picture,1)
            else:
                break

            roi = np.append(roi,roi_picture)
            print(counter)
            counter = counter + 1
            size = 6 - len(str(counter))
            string_counter = size*"0" + str(counter)
    else:
        counter = counter + 1
        size = 6 - len(str(counter))
        string_counter = size*"0" + str(counter)

roi = np.reshape(roi,(-1, 129))
np.savetxt('asfasfa.csv',roi , fmt='%d')   # hangi formatta yapılacağını fmt =%d ile ayarladık.

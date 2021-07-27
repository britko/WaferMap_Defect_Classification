import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt


def class_img(test_img):
    # 분류 가능하게 이미지 가공
    img1 = image.load_img('C:/Users/pc/Desktop/고영국/개발/AI/test_img/{}.jpg'.format(test_img), target_size=(64, 64))
    img = image.img_to_array(img1)
    img = img/255
    img = np.expand_dims(img, axis=0)

    # 모델 불러오기
    model = load_model('checkpoint_v2-07-0.55-0.82.h5')

    # 각 레이블의 확률을 numpy배열로 출력
    class_df = model.predict(img)

    if(np.argmax(class_df[0]) == 0):
        defect = "Center"

    elif(np.argmax(class_df[0]) == 1):
        defect = "Donut"

    elif(np.argmax(class_df[0]) == 2):
        defect = "Edge-Loc"

    elif(np.argmax(class_df[0]) == 3):
        defect = "Edge-Ring"

    elif(np.argmax(class_df[0]) == 4):
        defect = "Loc"

    elif(np.argmax(class_df[0]) == 5):
        defect = "Near-Full"

    elif(np.argmax(class_df[0]) == 6):
        defect = "None"

    elif(np.argmax(class_df[0]) == 7):
        defect = "Random"

    elif(np.argmax(class_df[0]) == 8):
        defect = "Scratch"

    ## 분류 결과
    # 결과 출력
    print("Defect: {} ({}%)".format(defect, np.round_(np.max(class_df[0])*100, 2)))

    # 결과 이미지 출력
    value = "{}: {}%".format(defect, np.round_(np.max(class_df[0])*100, 2))
    plt.text(20, 62, value, color='red', fontsize=15, bbox=dict(facecolor='white', alpha=0.8))
    plt.imshow(img1)
    plt.show()

# print(img.shape)



# 분류!!!
class_img("Scratch_219")
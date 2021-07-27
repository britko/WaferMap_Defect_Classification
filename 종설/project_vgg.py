from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.applications.vgg19 import VGG19

imageGenerator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[.2, .2],
    horizontal_flip=True,
    validation_split=.1,
    fill_mode='nearest'
)

trainGen = imageGenerator.flow_from_directory(
    'C:/Users/pc/Jupyter Notebook/WaferMap/imbalanced',
    target_size=(64, 64),
    color_mode='rgb',
    subset='training',
    class_mode='categorical'
)

validationGen = imageGenerator.flow_from_directory(
    'C:/Users/pc/Jupyter Notebook/WaferMap/balanced',
    target_size=(64, 64),
    color_mode='rgb',
    subset='validation',
    class_mode='categorical',
    shuffle=False
)


model = VGG19()

model.summary()

# model.compile(
#     optimizer='adam',
#     loss='categorical_crossentropy', 
#     metrics=['accuracy'],
# )

# filename = 'C:/Users/pc/Jupyter Notebook/savemodel/checkpoint_v2-{epoch:02d}-{val_loss:.2f}-{val_accuracy:.2f}.h5' # 가중치를 저장할 파일
# checkpoint = ModelCheckpoint(filename,             # file명을 지정합니다
#                              monitor='val_loss',   # val_loss 값이 개선되었을때 호출됩니다
#                              verbose=1,            # 로그를 출력합니다
#                              save_best_only=True,  # 가장 best 값만 저장합니다
#                              mode='auto'           # auto는 알아서 best를 찾습니다. min/max
#                             )

# history = model.fit(
#     trainGen,
#     epochs=10,
#     batch_size=500,
#     validation_data=validationGen,
#     callbacks=[checkpoint]
# )
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint

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


model = Sequential()

model.add(layers.InputLayer(input_shape=(64, 64, 3)))
model.add(layers.Conv2D(filters=32, kernel_size=(4,4),input_shape=(64, 64, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(rate=0.3))

model.add(layers.Conv2D(filters=32, kernel_size=(4,4), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(rate=0.3))

model.add(layers.Conv2D(filters=32, kernel_size=(4,4), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(rate=0.3))

model.add(layers.Conv2D(filters=32, kernel_size=(4,4), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(rate=0.3))

model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(9, activation='softmax'))

# model.summary()

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy', 
    metrics=['accuracy'],
)

filename = 'C:/Users/pc/Jupyter Notebook/savemodel/checkpoint_v2-{epoch:02d}-{val_loss:.2f}-{val_accuracy:.2f}.h5' # ???????????? ????????? ??????
checkpoint = ModelCheckpoint(filename,             # file?????? ???????????????
                             monitor='val_loss',   # val_loss ?????? ?????????????????? ???????????????
                             verbose=1,            # ????????? ???????????????
                             save_best_only=True,  # ?????? best ?????? ???????????????
                             mode='auto'           # auto??? ????????? best??? ????????????. min/max
                            )

history = model.fit(
    trainGen,
    epochs=10,
    batch_size=500,
    validation_data=validationGen,
    callbacks=[checkpoint]
)
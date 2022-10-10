import numpy as np
import mnist
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

#   training data
train_images = mnist.train_images()
train_labels = mnist.train_labels()

#   test data
test_images = mnist.test_images()
test_labels = mnist.test_labels()

#   normalize data
train_images = train_images/255 - 0.5
test_images = test_images/255 - 0.5

#   reshape data into single array of values (currently 28x28)
train_images = train_images.reshape((-1, 784))
test_images = test_images.reshape((-1, 784))

#   build model
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=train_images.shape[1]))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

#   get loss on training data
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#   train
#   to_categorical transforms label data into a 10-d array, batch size dictates number of samples used per gradient
history = model.fit(train_images, to_categorical(train_labels), epochs=5, batch_size=32)

#   evaluate model
model.evaluate(test_images, to_categorical(test_labels))

#   save model
model.save_weights('nn.h5')

#   predict 10 test images
predictions = model.predict(test_images[:10])
print(np.argmax(predictions, axis=1))
print(test_labels[:10])

for i in range(0, 5):
    first_image = test_images[i]
    first_image = np.array(first_image, dtype='float')
    pixels = first_image.reshape((28, 28))
    plt.imshow(pixels)
    plt.show()
'''
#   list all data in history
print(history.history.keys())

#   summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
#   summarize history for loss
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
'''

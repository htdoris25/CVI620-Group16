from keras import models, layers
from tensorflow.keras.optimizers import Adam

def driving_angle_model(X_train, y_train, X_test, y_test, aug):
    net = models.Sequential([
                        # Original size passing the first CNN
                        # With 24 filters, this extracts features such as edges and textures
                        layers.Conv2D(filters=24, kernel_size=(5, 5), activation='relu', input_shape=(66,200,3)),
                        
                        # With 36 filters, this extracts deeper features such as lane curves
                        layers.Conv2D(filters=36, kernel_size=(5, 5), activation='relu'),

                        # With 48 layers, this extracts high level features such as lane patterns
                        layers.Conv2D(filters=48, kernel_size=(5, 5), activation='relu'),

                        # With 64 layers, this extracts fine grained features
                        layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),

                        # Once again at 64 filters, this extracts the previous features once more
                        layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),

                        # Flatten the result array to a 1D array to match size for Dense
                        layers.Flatten(),

                        # Separate 3 Dense compressions on result layer
                        # Compressing with large number of neurons will overfit, and too low will underfit or lose info
                        layers.Dense(100, activation='relu'),
                        layers.Dense(50, activation='relu'),
                        layers.Dense(1)
                        ])
    
    optimizer = Adam(learning_rate=0.01)

    # We want to determine the driving angle based on picture. Therefore, this is a regression problem
    net.compile(optimizer, loss='mse')

    # Fits the neural network with data
    # Batch size between 2-32 can consistently be stable and reliable (source: Revisiting Small Batch Training for Deep Neural Networks)
    H = net.fit(aug.flow(X_train, y_train), validation_data=(X_test, y_test), batch_size=32, epochs=100)

    return H;
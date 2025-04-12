from tensorflow.keras.preprocessing.image import ImageDataGenerator
from data_loader import X_train, X_val, y_train, y_val
import matplotlib.pyplot as plt
import numpy as np

# --- Define Data Augmentation ---
aug = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest"
)

# Fit generator on training data
aug.fit(X_train)

# --- Preview Augmented Output ---
if __name__ == "__main__":
    sample_idx = np.random.randint(len(X_train))
    sample_img = X_train[sample_idx]
    sample_label = y_train[sample_idx]

    # Expand dims for generator
    sample_img_batch = np.expand_dims(sample_img, axis=0)

    # Generate augmented images
    aug_iter = aug.flow(sample_img_batch, batch_size=1)
    aug_img = next(aug_iter)[0]

    # Plot original vs augmented
    plt.subplot(1, 2, 1)
    plt.title(f"Original Angle: {sample_label:.2f}")
    plt.imshow(sample_img)

    plt.subplot(1, 2, 2)
    plt.title("Augmented")
    plt.imshow(aug_img)
    plt.tight_layout()
    plt.show()

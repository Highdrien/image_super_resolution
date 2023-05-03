import os
import numpy as np
import matplotlib.pyplot as plt


def plot_getitem(X, Y, upscale_factor, index=0):
    """
    plot images and an zoom in the images from a getitem
    X: LR_image and Y: HR_image form a batch. index is the index of the image in the batch
    zoom: zoom_factor (the zoom will be one the centre of the images)
    upscale_factor: upscale_factor of the LR_image
    """
    _, axes = plt.subplots(1, 2)
    print(Y.shape)

    # HR image
    image = Y[index]
    print(image.shape)
    image = np.moveaxis(image.numpy(), 0, -1)
    print(image.shape)
    axes[0].imshow(image)
    axes[0].set_title("HR image")

    # LR image
    image = X[index]
    image = np.moveaxis(image.numpy(), 0, -1)
    axes[1].imshow(image)
    axes[1].set_title("upscale: "+str(upscale_factor))

    plt.show()


def save_learning_curves(path):
    result, names = get_result(path)

    epochs = result[:, 0]
    loss = result[:, 1]
    val_loss = result[:, 2]

    plt.plot(epochs, loss)
    plt.plot(epochs, val_loss)
    plt.title(names[1])
    plt.xlabel('epoch')
    plt.ylabel(names[1])
    plt.legend(names[1:])
    plt.grid()
    plt.savefig(os.path.join(path, names[1] + '.png'))
    plt.close()


def get_result(path):
    with open(os.path.join(path, 'train_log.csv'), 'r') as f:
        names = f.readline()[:-1].split(',')
        result = []
        for line in f:
            result.append(line[:-1].split(','))

        result = np.array(result, dtype=float)
    f.close()
    return result, names


if __name__ == "__main__":
    save_learning_curves('../logs/experiment_0')
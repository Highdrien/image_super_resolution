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
    for i in range(1, len(names), 2):
        train_metrics = result[:, i]
        val_metrics = result[:, i + 1]
        plt.plot(epochs, train_metrics)
        plt.plot(epochs, val_metrics)
        plt.title(names[i])
        plt.xlabel('epoch')
        plt.ylabel(names[i])
        plt.legend(names[i:])
        plt.grid()
        plt.savefig(os.path.join(path, names[i] + '.png'))
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


def find_best_from_csv(csv_file):
    best_epoch, best_val_loss, nb_epochs = 0, 10e6, 0
    with open(csv_file, 'r') as f:
        if len(f.readline()[:-1].split(',')) >= 2:
            for line in f:
                split_line = line.split(',')
                nb_epochs = int(split_line[0])
                if float(split_line[2]) < best_val_loss:
                    best_val_loss = float(split_line[2])
                    best_epoch = int(split_line[0])
        else:
            print("ERROR: train_log.csv doesn't containe epochs, loss, and val loss")
            exit()
    
    return best_epoch, best_val_loss, nb_epochs


# if __name__ == "__main__":
#     save_learning_curves('../logs/experiment_2')
#     print(find_best_from_csv('../logs/experiment_2/train_log.csv'))
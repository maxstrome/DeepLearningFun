import keras
import matplotlib.pyplot as plt
import numpy as np
def basic_cnn_inference(model, images):
    probability_model = keras.Sequential(
        [
            model,
            keras.layers.Softmax()
        ]
    )
    prediction_logits = probability_model.predict(x=images)
    return prediction_logits

def plot_image(predictions_array, class_names, img):
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)

  plt.xlabel("{} {:2.0f}%".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                ),
                                color='blue')

def plot_value_array(predictions_array):
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('blue')

def visualize_predictions(prediction_logits, images, class_labels):
    num_rows = 5
    num_cols = 3
    num_images = num_rows * num_cols
    plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        plot_image(prediction_logits[i], class_labels, images[i])
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        plot_value_array(prediction_logits[i])
    plt.tight_layout()
    plt.show()

def run_inference_and_visualize(images, model, class_labels):
    predictions = basic_cnn_inference(model, images)
    visualize_predictions(prediction_logits=predictions[:15], images=images[:15], class_labels=class_labels)

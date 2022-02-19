## this model will predict whether the no. is greater than 5 or less than 5

import argparse
import os
import shutil
import numpy as np
from tqdm import tqdm
import logging
from src.utils.common import read_yaml, create_directories
import random
import tensorflow as tf
import io


STAGE = "transfer learning 2" ## <<< change stage name 

logging.basicConfig(
    filename = os.path.join("logs", 'running_logs.log'), 
    level = logging.INFO, 
    format = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode = "a"
    )

"""enumerate() allows us to iterate through a sequence but it keeps track of both the index and the element. 
The enumerate() function takes in an iterable as an argument, such as a list, string, tuple, or dictionary."""
def update_greater_than_less_than_5_labels(list_of_labels):
    for idx, label in enumerate(list_of_labels):
        condition = (label >= 5)
        list_of_labels[idx] = np.where(condition, 1, 0)

    return list_of_labels

def main(config_path):
    ## read config files
    config = read_yaml(config_path)

    # get the data
    mnist = tf.keras.datasets.mnist
    (x_train_full, y_train_full), (x_test, y_test) = mnist.load_data()

    x_train_full = x_train_full / 255.
    x_test = x_test / 255.
    
    x_valid, x_train = x_train_full[:5000], x_train_full[5000:]
    y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

    y_train_bin2, y_test_bin2, y_valid_bin2 = update_greater_than_less_than_5_labels([y_train, y_test, y_valid])

    # set the seed value
    seed = 2022 ## get it from config
    tf.random.set_seed(seed)
    np.random.seed(seed)

    # logging the model summary information inside the logs

    def _log_model_summary(model):
        with io.StringIO() as stream:
            # default print_fn is None which means that the summary will be printed on the terminal
            # if we want to print the summary in logs, we use StringIO()
            model.summary(print_fn = lambda x: stream.write(f"{x}\n"))
            summary_str = stream.getvalue()

        return summary_str

    # load the base model
    base_model_path = os.path.join("artifacts", "models", "base_model.h5")
    base_model = tf.keras.models.load_model(base_model_path)
    # model.summary()
    logging.info(f"loaded base model summary: \n{_log_model_summary(base_model)}")


    # FREEZE THE WEIGHTS
    # to check if the layers are trainable or not

    #for layer in base_model.layers:
        #print(f"trainable status of {layer.name}: {layer.trainable}")

    """So, the trainable status of all the layers is True which means the model will 
    train all these layers and weight update will occur simultaneously. So, we need to
    make the trainable status of these layers as False by freezing their weights.
    
    we need to exclude the output layer because we have to change that"""

    for layer in base_model.layers[:-1]:
        print(f"trainable status of {layer.name} before: {layer.trainable}")
        layer.trainable = False
        print(f"trainable status of {layer.name} after: {layer.trainable}")

    # modify the last layer
    # define the model and compile it 
    base_layer = base_model.layers[:-1]
    new_model2 = tf.keras.models.Sequential(base_layer)
    new_model2.add(
        tf.keras.layers.Dense(2, activation = 'softmax', name = 'OutputLayer')
    )

    # logging the model summary
    logging.info(f"{STAGE} model summary: \n{_log_model_summary(new_model2)}")

    """so, now we can observe in the logs that there are only 202 trainable paraameters
    because rest of the weights are freezed. This will make our training faster."""

    
    LOSS = tf.losses.sparse_categorical_crossentropy
    OPTIMIZER = tf.keras.optimizers.SGD(learning_rate = 1e-3) # 10^(-7)
    METRICS = ["accuracy"]
    new_model2.compile(loss = LOSS, optimizer = OPTIMIZER, metrics = METRICS)

    
    # train the model
    history = new_model2.fit(x_train, y_train_bin2, 
                        epochs = 10, 
                        validation_data = (x_valid, y_valid_bin2),
                        verbose = 2)

    # save the base model
    model_dir_path = os.path.join("artifacts", "models")

    model_file_path = os.path.join(model_dir_path, "greater_than_5_model.h5")
    new_model2.save(model_file_path)


    logging.info(f"base model is saved at {model_file_path}")
    logging.info(f"evaluation metrics {new_model2.evaluate(x_test, y_test_bin2)}")


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default = "configs/config.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path = parsed_args.config)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e

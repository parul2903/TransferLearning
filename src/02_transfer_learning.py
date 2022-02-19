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


STAGE = "transfer learning" ## <<< change stage name 

logging.basicConfig(
    filename = os.path.join("logs", 'running_logs.log'), 
    level = logging.INFO, 
    format = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode = "a"
    )


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

    # set the seed value
    seed = 2022 ## get it from config
    tf.random.set_seed(seed)
    np.random.seed(seed)

    # define the layers and create the model
    LAYERS = [tf.keras.layers.Flatten(input_shape=[28, 28], name = "InputLayer"), 
            tf.keras.layers.Dense(300, name = "HiddenLayer1"),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(100, name = "HiddenLayer2"),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Dense(10, activation="softmax", name = "OutputLayer")]

    # define the model and compile it
    model = tf.keras.models.Sequential(LAYERS)

    LOSS = tf.losses.sparse_categorical_crossentropy
    OPTIMIZER = tf.keras.optimizers.SGD(learning_rate = 1e-3) # 10^(-7)
    METRICS = ["accuracy"]
    model.compile(loss = LOSS, optimizer = OPTIMIZER, metrics = METRICS)

    # logging the model summary information inside the logs

    """StringIO(): The StringIO module is an in-memory file-like object. 
    This object can be used as input or output to the most function that would expect 
    a standard file object. When the StringIO object is created it is initialized by 
    passing a string to the constructor. """

    def _log_model_summary(model):
        with io.StringIO() as stream:
            # default print_fn is None which means that the summary will be printed on the terminal
            # if we want to print the summary in logs, we use StringIO()
            model.summary(print_fn = lambda x: stream.write(f"{x}\n"))
            summary_str = stream.get_value()

        return summary_str


    # train the model
    history = model.fit(x_train, y_train, 
                        epochs = 10, 
                        validation_data = (x_valid, y_valid),
                        verbose = 2)

    # save the base model
    model_dir_path = os.path.join("artifacts", "models")
    create_directories([model_dir_path])

    model_file_path = os.path.join(model_dir_path, "base_model.h5")
    model.save(model_file_path)


    logging.info(f"base model is saved at {model_file_path}")
    logging.info(f"evaluation metrics {model.evaluate(x_test, y_test)}")


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
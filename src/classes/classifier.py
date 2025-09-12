import keras
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from scikeras.wrappers import KerasClassifier
from sklearn.pipeline import Pipeline
# from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score

PIXEL_MAX = 255
SEED = 42

tf.random.set_seed(SEED)

INPUT_SHAPE = (256, 256, 3)


def build_cnn(
    input_shape=(256, 256, 3),
    n_outputs=5,
    dense_nodes=128,
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    **kwargs,
):
    layers_list = [
        layers.InputLayer(shape=input_shape),
        layers.Rescaling(1.0 / PIXEL_MAX),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(dense_nodes, activation="relu"),
        layers.Dense(n_outputs, activation="softmax"),
    ]

    model = keras.Sequential(layers_list, name="cnn_sequential")
    model.compile(
        optimizer=optimizer,  # type: ignore
        loss=loss,  # grid can override
        metrics=["accuracy"],
        run_eagerly=True,
    )
    model.summary()
    return model


def create_pipeline_paramgrid(config):
    def build_cnn_with_config(**kwargs):
        return build_cnn(
            input_shape=INPUT_SHAPE,
            n_outputs=config["output_classes"],
            **kwargs
        )

    PARAMETER_GRID = config["hyper_params"]

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=3,
        verbose=1,
        start_from_epoch=3,
        restore_best_weights=True,
    )

    MODEL = KerasClassifier(
        model=build_cnn_with_config,
        optimizer=tf.keras.optimizers.Adam,  # grid will override
        metrics=["accuracy"],  # grid can override
        epochs=10,  # grid will override
        batch_size=64,  # grid will override
        verbose=2,  # 0 = silent, 1 = progress bar, 2 = one line per epoch
        random_state=42,
        callbacks=[early_stopping],
    )

    pipeline_list = [("neural", MODEL)]
    return (
        Pipeline(pipeline_list),
        PARAMETER_GRID,
    )


class Classifier:
    def __init__(self, params, data, targets, classmap, pred_model=None):
        self.params = params
        self.data = np.array(data)
        self.targets = np.array(targets)
        self.classmap = classmap
        self.pred_model = pred_model

    # return the model
    def run_train(self):
        # min max normalization with known max value
        X_learn = self.data
        y_learn = self.targets

        (pipeline, PARAMETER_GRID) = create_pipeline_paramgrid(self.params)
        # tscv = KFold(n_splits=3).split(X_learn)
        # grid = RandomizedSearchCV(
        #     estimator=pipeline,
        #     param_distributions=PARAMETER_GRID,
        #     random_state=42,
        #     n_jobs=1,
        #     n_iter=3,  # Number of iterations for hyperparameter tuning
        #     verbose=4,
        #     cv=tscv,
        #     scoring="accuracy",
        #     error_score="raise",
        #     return_train_score=True,
        # )
        X_learn = X_learn.astype("float32")
        y_learn = y_learn.astype("int32")
        # grid.fit(X_learn, y_learn)
        # pipeline = grid.best_estimator_  # type: ignore

        pipeline.fit(X_learn, y_learn, neural__validation_split=0.1)

        predictions = pipeline.predict(X_learn)  # type: ignore
        # Calculate metrics
        acc = accuracy_score(y_learn, predictions)
        b_acc = balanced_accuracy_score(y_learn, predictions)
        print("\n====PREDICTIONS=====")
        print(predictions)
        print(f"\nAccuracy: {acc:.4f}")
        print(f"Balanced accuracy (b_acc): {b_acc:.4f}")

        train_report = {
            "acc": acc,
            "b_acc": b_acc,
        }
        report = {"train": train_report}

        return (report, pipeline)

    def run_predict(self):
        pipeline = self.pred_model

        # prediction
        predictions = pipeline.predict(X=self.data)

        # Calculate metrics
        acc = accuracy_score(self.targets, predictions)
        b_acc = balanced_accuracy_score(self.targets, predictions)

        print(f"\nAccuracy: {acc:.4f}")
        print(f"Balanced accuracy (b_acc): {b_acc:.4f}")

        train_report = {"acc": acc, "b_acc": b_acc, "len": len(predictions)}

        report = {
            "all": train_report,
        }

        return (report, predictions)

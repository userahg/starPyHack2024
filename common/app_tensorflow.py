import common.star_api.design_manager as dm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import numpy.typing as npt
from pathlib import Path
import re
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

np.set_printoptions(precision=3, suppress=True)
work_dir = Path(r"E:\OneDrive - Siemens AG\mdx\hackathon\2024\starPy")
proj_name = "exhaust_test"
version = "19.02.009-R8"


def import_data(proj: dm.DesignManagerProject,
                src_study: dm.Study,
                response: str,
                study_regex: str = None) -> [dm.Study]:
    if study_regex is None:
        studies = proj.studies
    else:
        studies = [s for s in proj.studies if re.match(study_regex, s.name)]
    cols = get_required_columns(src_study=src_study, response=response)
    studies = get_compatible_studies(studies=studies, columns=cols)
    df = src_study.get_all_designs().data_frame()
    for s in studies:
        df = pd.concat([df, s.get_all_designs().data_frame()], ignore_index=True, sort=False)
    df = df[cols]
    return df


def get_required_columns(src_study: dm.Study, response: str = None) -> [str]:
    param_names = [p.name for p in src_study.parameters]
    data = src_study.get_all_designs().data_frame()
    columns = []
    for p in param_names:
        for c in data.columns:
            regex = f"^{p}( \(.*\))?$"
            if re.match(regex, c):
                columns.append(c)
    if response is not None:
        for c in data.columns:
            if response in c:
                columns.append(c)
    return columns


def get_compatible_studies(studies: [dm.Study], columns: [str]) -> [dm.Study]:
    comp_studies = []
    for study_i in studies:
        data = study_i.get_all_designs().data_frame()
        all_found = True
        for col in columns:
            if col not in data.columns:
                all_found = False
        if all_found:
            comp_studies.append(study_i)
    return comp_studies


def build_and_compile_model(norm: layers.Normalization) -> keras.Sequential:
    m = keras.Sequential([norm,
                          layers.Dense(64, activation="relu"),
                          layers.Dense(64, activation="relu"),
                          layers.Dense(1)
                          ])
    m.compile(loss="mean_absolute_error", optimizer=tf.keras.optimizers.Adam(0.001))
    return m


def plot_loss(hist):
    fig, ax = plt.subplots()
    ax.plot(hist.history['loss'], label='loss')
    ax.plot(hist.history['val_loss'], label='val_loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Error [Pressure Drop]')
    ax.legend()
    plt.show()


def plot_model_results(labels, predictions):
    f, a = plt.subplots(figsize=(8.0, 8.0))
    a.scatter(labels, predictions)
    a.set_xlabel("True Values")
    a.set_ylabel("Predictions")
    lims = [0, 1000]
    a.set_xlim(lims)
    a.set_ylim(lims)
    _ = a.plot(lims, lims)
    plt.show()


def train_model(plot_history: bool = False, plot_predictions: bool = False, save_model: str = None) -> keras.Sequential:
    dmprj = dm.DesignManagerProject.get_proj(work_dir=work_dir, dmprj=proj_name, version=version)
    p_opt_min_1_const = dmprj.get_study("Pareto Opt - Min 1 Const")
    raw_data = import_data(proj=dmprj, src_study=p_opt_min_1_const, response="Pressure Drop MA",
                           study_regex="^Pareto Opt")
    dataset = raw_data.copy()
    dataset = dataset.dropna()
    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)
    train_features = train_dataset.copy()
    test_features = test_dataset.copy()
    train_labels = train_features.pop("Pressure Drop MA (Pa)")
    test_labels = test_features.pop("Pressure Drop MA (Pa)")

    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(np.array(train_features))

    model = build_and_compile_model(normalizer)
    history = model.fit(train_features,
                        train_labels,
                        validation_split=0.2,
                        verbose=0,
                        epochs=100)
    if plot_history:
        plot_loss(history)

    test_results = model.evaluate(test_features, test_labels, verbose=0)
    print(f"Model Error: {test_results} MAE")
    test_predictions = model.predict(test_features).flatten()
    if plot_predictions:
        plot_model_results(test_labels, test_predictions)

    if save_model is not None:
        model.save(save_model)

    return model


def study_predictions(study: dm.Study, model_name: str = "dnn.keras"):
    print(f"Model predictions for Study: {study.name}")
    loaded_model = tf.keras.models.load_model(model_name)
    successful = study.get_design_set("Successful")
    columns = get_required_columns(study)
    for design in successful:
        data = design.data_frame()
        predicted = loaded_model.predict(data[columns]).flatten()[0]
        actual = data["Pressure Drop MA (Pa)"].iloc[0]
        print(f"Design {design.get_design_number()}")
        print(f"\tParameters:")
        for c in columns:
            print(f"\t\t{c}: {data[c].iloc[0]}")
        print(f"\tPressure Drop MA (Pa)")
        print(f"\t\tActual: {actual}")
        print(f"\t\tPredicted: {predicted}")
        print(f"\t\tError: {actual - predicted}")


if __name__ == "__main__":
    # dnn_model = train_model(plot_history=True, plot_predictions=True, save_model="dnn.keras")
    dmprj = dm.DesignManagerProject.get_proj(work_dir=work_dir, dmprj=proj_name, version=version)
    test_study = dmprj.studies[0]
    study_predictions(test_study)

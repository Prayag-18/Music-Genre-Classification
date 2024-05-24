import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report, accuracy_score
import tensorflow.keras as keras

def Load_Data(data_path):
    with open(data_path, "r") as f:
        data = json.load(f)
    X = np.array(data["mfcc"])       # X (ndarray): Inputs
    y = np.array(data["labels"])     # y (ndarray): Targets
    print()
    print("Data loaded succesfully !")
    print()
    return X, y

def build_model(input_shape):
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=input_shape),    # X.shape[1] -> intervals , X.shape[2] -> Values of MFCCs
        keras.layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),   # Audio data is usually Complex, that's why we are using L2 regularization
        keras.layers.Dropout(0.4),              # 0.4 -> Dropout probability of neuron (usually b/w 0.1 to 0.5)
        keras.layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(10, activation='softmax')
    ])
    return model

if __name__ == "__main__":
    DATA_PATH = "data_10.json"
    X, y = Load_Data(DATA_PATH)
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    scores = []
    all_histories = []
    reports = []
    fold_no = 1
    for train_index, test_index in kfold.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        model = build_model((X.shape[1], X.shape[2]))
        optimiser = keras.optimizers.Adam(learning_rate=0.0001)
        model.compile(optimizer=optimiser, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        print("\n")
        print(f'Training fold {fold_no}...')
        print()
        history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=100, verbose=1)
        all_histories.append(history.history)
        
        score = model.evaluate(X_test, y_test, verbose=0)
        scores.append(score)
        
        y_pred = np.argmax(model.predict(X_test), axis=1)
        report = classification_report(y_test, y_pred, output_dict=True)
        reports.append(report)
        fold_no += 1

    avg_accuracy = np.mean([score[1] for score in scores])
    avg_loss = np.mean([score[0] for score in scores])
    print(f'\nAverage Accuracy across the folds: {avg_accuracy}')
    print(f'Average Loss across the folds: {avg_loss}')

    # Averaging the classification reports
    metrics = ['precision', 'recall', 'f1-score']
    avg_reports = {metric: {} for metric in metrics}
    # Initialize the label keys across all reports for averaging
    label_keys = set()
    for report in reports:
        label_keys.update(report.keys())
    
    # Remove entries that are not classes
    label_keys.discard('accuracy')
    label_keys.discard('macro avg')
    label_keys.discard('weighted avg')

    for metric in metrics:
        for label in label_keys:
            avg_reports[metric][label] = np.mean([report[label][metric] for report in reports if label in report])

    for metric in metrics:
        for label in ['macro avg', 'weighted avg']:
            avg_reports[metric][label] = np.mean([report[label][metric] for report in reports])

    avg_reports['accuracy'] = np.mean([report['accuracy'] for report in reports])

    print("\nAveraged Classification Report:")
    for metric, values in avg_reports.items():
        print(f"\n{metric.capitalize()}:")
        if isinstance(values, dict):
            for label, value in values.items():
                print(f"{label}: {value:.2f}")
        else:
            print(f"{metric}: {values:.2f}")
    
    # Plot training and validation accuracy and loss
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    for i, history in enumerate(all_histories):
        axs[0].plot(history['accuracy'], label=f'Train Accuracy Fold {i+1}')
        axs[0].plot(history['val_accuracy'], label=f'Val Accuracy Fold {i+1}', linestyle='--')
        axs[1].plot(history['loss'], label=f'Train Loss Fold {i+1}')
        axs[1].plot(history['val_loss'], label=f'Val Loss Fold {i+1}', linestyle='--')

    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].legend()

    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].legend()

    plt.tight_layout()
    plt.show()
import os
import threading
import json
import numpy as np
from tkinter import Tk, Button, Label, Listbox, END, filedialog, messagebox, Frame, StringVar, Entry, Scrollbar, RIGHT, Y, LEFT, BOTH
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# Parameters
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
DEFAULT_EPOCHS = 50
DEFAULT_PATIENCE = 5
DEFAULT_TEST_RATIO = 0.1
DEFAULT_VAL_RATIO = 0.1

# Globals
category_folders = {}
model = None
label_map = {}

# Build improved CNN model
def build_model(num_classes):
    model = models.Sequential([
        layers.InputLayer(input_shape=(*IMG_SIZE, 3)),
        layers.Conv2D(32, 3, activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),

        layers.Conv2D(64, 3, activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Prepare dataset
def prepare_dataset():
    data, labels = [], []
    class_names = sorted(category_folders.keys())
    for idx, (label, folder_path) in enumerate(category_folders.items()):
        for file in os.listdir(folder_path):
            if file.lower().endswith(('png', 'jpg', 'jpeg')):
                img_path = os.path.join(folder_path, file)
                try:
                    img = Image.open(img_path).convert('RGB').resize(IMG_SIZE)
                    data.append(np.array(img, dtype=np.float32))
                    labels.append(idx)
                except:
                    continue
    data = np.array(data)
    labels = tf.keras.utils.to_categorical(np.array(labels))

    test_ratio = float(test_ratio_entry.get())
    val_ratio = float(val_ratio_entry.get())
    X_temp, X_test, y_temp, y_test = train_test_split(data, labels, test_size=test_ratio, random_state=42)
    val_split = val_ratio / (1 - test_ratio)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_split, random_state=42)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), class_names

# Training

def train_model():
    global model, label_map
    try:
        status_text.set("Status: Preparing data...")
        (X_train, y_train), (X_val, y_val), (X_test, y_test), class_names = prepare_dataset()
        label_map = {i: class_names[i] for i in range(len(class_names))}

        model = build_model(len(class_names))
        early_stop = EarlyStopping(monitor='val_loss', patience=int(patience_entry.get()), restore_best_weights=True)

        status_text.set("Status: Training model...")
        model.fit(
            X_train / 255.0, y_train,
            validation_data=(X_val / 255.0, y_val),
            epochs=int(epoch_entry.get()),
            batch_size=BATCH_SIZE,
            callbacks=[early_stop],
            verbose=1
        )

        train_loss, train_acc = model.evaluate(X_train / 255.0, y_train, verbose=0)
        val_loss, val_acc = model.evaluate(X_val / 255.0, y_val, verbose=0)
        test_loss, test_acc = model.evaluate(X_test / 255.0, y_test, verbose=0)

        eval_result_text.set(f"Train Acc: {train_acc:.2%}\nVal Acc: {val_acc:.2%}\nTest Acc: {test_acc:.2%}")
        model_info_text.set("Current Model: Trained")
        status_text.set("Status: Training complete.")

    except Exception as e:
        messagebox.showerror("Training Error", str(e))
        status_text.set("Status: Training failed.")

# Predict Image

def predict_image():
    global model, label_map
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if not file_path or model is None or not label_map:
        return
    try:
        img = Image.open(file_path).convert("RGB").resize(IMG_SIZE)
        img_array = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)
        predictions = model.predict(img_array)[0]
        pred_idx = int(np.argmax(predictions))
        label = label_map.get(pred_idx, f"Class {pred_idx}")
        prediction_result_text.set(f"Prediction: {label} ({predictions[pred_idx]*100:.2f}%)")
    except Exception as e:
        messagebox.showerror("Prediction Error", str(e))

# GUI Setup
root = Tk()
root.title("CNN Image Classifier")
root.geometry("950x850")
root.configure(bg="#f7faff")

Label(root, text="CNN Trainer & Predictor", bg="#0a4275", fg="white", font=("Segoe UI", 18, "bold"), pady=10).pack(fill="x")

folder_frame = Frame(root, bg="#eaf2fb", bd=2, relief="groove", padx=10, pady=10)
folder_frame.pack(pady=10, padx=10, fill="x")
Label(folder_frame, text="Step 1: Select Image Folders", bg="#eaf2fb", font=("Segoe UI", 12, "bold")).pack(anchor="w")

folder_listbox = Listbox(folder_frame, width=100, height=5, font=("Segoe UI", 10))
folder_listbox.pack(padx=5, pady=5, fill="x")
scrollbar = Scrollbar(folder_listbox)
scrollbar.pack(side=RIGHT, fill=Y)
folder_listbox.config(yscrollcommand=scrollbar.set)
scrollbar.config(command=folder_listbox.yview)

def add_folder():
    folder_path = filedialog.askdirectory(title="Select Category Folder")
    if folder_path:
        category_name = os.path.basename(folder_path)
        if category_name in category_folders:
            messagebox.showwarning("Duplicate Category", f"'{category_name}' is already added.")
            return
        category_folders[category_name] = folder_path
        folder_listbox.insert(END, f"{category_name}: {folder_path}")

Button(folder_frame, text="Add Category Folder", command=add_folder, bg="#007acc", fg="white", font=("Segoe UI", 10, "bold"), width=20).pack(pady=5)

# Training Options
options_frame = Frame(root, bg="#f7faff")
options_frame.pack(pady=10, fill="x", padx=10)
Label(options_frame, text="Step 2: Set Training Parameters", bg="#f7faff", font=("Segoe UI", 12, "bold")).grid(row=0, columnspan=6, sticky="w", pady=5)

Label(options_frame, text="Epochs:", bg="#f7faff").grid(row=1, column=0)
epoch_entry = Entry(options_frame, width=8)
epoch_entry.insert(0, str(DEFAULT_EPOCHS))
epoch_entry.grid(row=1, column=1)

Label(options_frame, text="Patience:", bg="#f7faff").grid(row=1, column=2)
patience_entry = Entry(options_frame, width=8)
patience_entry.insert(0, str(DEFAULT_PATIENCE))
patience_entry.grid(row=1, column=3)

Label(options_frame, text="Test Ratio:", bg="#f7faff").grid(row=2, column=0)
test_ratio_entry = Entry(options_frame, width=8)
test_ratio_entry.insert(0, str(DEFAULT_TEST_RATIO))
test_ratio_entry.grid(row=2, column=1)

Label(options_frame, text="Validation Ratio:", bg="#f7faff").grid(row=2, column=2)
val_ratio_entry = Entry(options_frame, width=8)
val_ratio_entry.insert(0, str(DEFAULT_VAL_RATIO))
val_ratio_entry.grid(row=2, column=3)

# Model Actions
model_action_frame = Frame(root, bg="#eaf2fb", bd=2, relief="groove", padx=10, pady=10)
model_action_frame.pack(pady=10, fill="x", padx=10)

status_text = StringVar()
status_text.set("Status: Waiting...")
Label(model_action_frame, textvariable=status_text, bg="#eaf2fb", font=("Segoe UI", 10, "italic")).pack(anchor="w")

model_info_text = StringVar()
model_info_text.set("Current Model: None")
Label(model_action_frame, textvariable=model_info_text, bg="#eaf2fb").pack(anchor="w")

Button(model_action_frame, text="Train Model", bg="#28a745", fg="white", font=("Segoe UI", 12, "bold"), command=lambda: threading.Thread(target=train_model, daemon=True).start()).pack(pady=5)

# Save / Load

def save_model():
    if model:
        filepath = filedialog.asksaveasfilename(defaultextension=".h5", filetypes=[("HDF5 files", "*.h5")])
        if filepath:
            model.save(filepath)
            with open(filepath.replace(".h5", "_labels.json"), "w") as f:
                json.dump(label_map, f)
            model_info_text.set(f"Saved to {os.path.basename(filepath)}")
            messagebox.showinfo("Saved", f"Model and labels saved to {filepath}")
    else:
        messagebox.showwarning("No Model", "Train or load a model first")

def load_model():
    global model, label_map
    filepath = filedialog.askopenfilename(filetypes=[("HDF5 files", "*.h5")])
    if filepath:
        model = tf.keras.models.load_model(filepath)
        json_path = filepath.replace(".h5", "_labels.json")
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                label_map = {int(k): v for k, v in json.load(f).items()}
        else:
            label_map = {}
        model_info_text.set(f"Loaded from {os.path.basename(filepath)}")
        messagebox.showinfo("Loaded", f"Model and labels loaded from {filepath}")

Button(model_action_frame, text="Save Model", bg="#6c757d", fg="white", command=save_model).pack(side=LEFT, padx=10)
Button(model_action_frame, text="Load Model", bg="#6c757d", fg="white", command=load_model).pack(side=LEFT)

# Evaluation
Label(root, text="Training & Evaluation Results", bg="#f7faff", font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=15)
eval_result_text = StringVar()
Label(root, textvariable=eval_result_text, bg="#f7faff", font=("Segoe UI", 10)).pack(anchor="w", padx=15)

# Prediction Section
Label(root, text="Step 3: Predict New Image", bg="#f7faff", font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=15, pady=(10, 0))
Button(root, text="Predict Image", command=predict_image, bg="#dc3545", fg="white", font=("Segoe UI", 12, "bold"), width=20).pack(pady=5)
prediction_result_text = StringVar()
Label(root, textvariable=prediction_result_text, bg="#f7faff", font=("Segoe UI", 10, "bold")).pack()

Label(root, text="Powered by TensorFlow | Developed by Margret Mary Diana", bg="#0a4275", fg="white", font=("Segoe UI", 9, "italic"), pady=8).pack(fill="x", side="bottom")

root.mainloop()

import pandas as pd
import matplotlib.pyplot as plt

csv_files = ["1.csv", "2.csv", "3.csv"]
plot_titles = ["U-NET Architecture", "Trans U-NET Architecture", "Attention U-NET Architecture"]
colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']

for i, csv_file in enumerate(csv_files):
    df = pd.read_csv(csv_file)
    
    epochs = df["Epochs"]
    accuracy = df["Average Training Accuracy"]
    loss = df["Average Training Loss"]
    precision = df["Average Training Precision"]
    sensitivity = df["Average Training Sensitivity"]
    specificity = df["Average Training Specificity"]
    f1 = df["Average Training F1"]
    js = df["Average Training JS"]
    dice_coefficient = df["Average Training Dice Coefficient"]
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, accuracy, label='Accuracy', color=colors[0])
    plt.plot(epochs, loss, label='Loss', color=colors[1])
    plt.plot(epochs, precision, label='Precision', color=colors[2])
    plt.plot(epochs, sensitivity, label='Sensitivity', color=colors[3])
    plt.plot(epochs, specificity, label='Specificity', color=colors[4])
    plt.plot(epochs, f1, label='F1', color=colors[5])
    plt.plot(epochs, js, label='JS', color=colors[6])
    plt.plot(epochs, dice_coefficient, label='Dice Coefficient', color=colors[7])
    
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.title(plot_titles[i])
    plt.legend()
    plt.grid(True)
    
    plt.savefig(f"plot_{i+1}.png")
    plt.show()

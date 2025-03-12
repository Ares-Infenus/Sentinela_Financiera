import matplotlib.pyplot as plt


# Suponiendo que 'history' es el objeto retornado por model_instance.train()
def plot_training_history(history):
    # Extraer datos del history
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]
    epochs = range(1, len(loss) + 1)

    plt.figure(figsize=(14, 5))

    # Gráfico de la pérdida
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, "bo-", label="Pérdida de entrenamiento")
    plt.plot(epochs, val_loss, "ro-", label="Pérdida de validación")
    plt.title("Evolución de la pérdida")
    plt.xlabel("Época")
    plt.ylabel("Pérdida")
    plt.legend()

    # Gráfico de la exactitud
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, "bo-", label="Exactitud de entrenamiento")
    plt.plot(epochs, val_accuracy, "ro-", label="Exactitud de validación")
    plt.title("Evolución de la exactitud")
    plt.xlabel("Época")
    plt.ylabel("Exactitud")
    plt.legend()

    plt.tight_layout()
    plt.show()


# Ejemplo de uso:
# history = model_instance.train(epochs=50, batch_size=32)
# plot_training_history(history)

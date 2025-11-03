import matplotlib.pyplot as plt

def plot_and_save(data, save_path, label, ylabel):
    plt.figure(figsize=(8, 4))
    plt.plot(data, label=label, color="#1f77b4")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
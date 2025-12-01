import matplotlib.pyplot as plt

class Visualizer:
    def generate_plot(self, df, column, save_path="output_plot.png"):
        plt.figure(figsize=(10,5))
        plt.plot(df[column])
        plt.title(f"{column} Trend")
        plt.xlabel("Index")
        plt.ylabel(column)
        plt.savefig(save_path)
        plt.close()
        return save_path

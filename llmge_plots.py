import matplotlib.pyplot as plt
import webbrowser
import os


def plot_creation(x_values, y_values, results_directory):
    plt.plot(x_values, y_values, marker='*', color='blue', markersize=25)

    for result in os.listdir(results_directory):
        result_path = os.path.join(results_directory, result)
        result_path = open(result_path, 'rt').read()
        best_instance_accuracy = float(result_path.split(",")[0])
        total_params = int(result_path.split(",")[1])
        plt.plot(best_instance_accuracy, total_params, "ro")

    plt.xlim(0.7, 1)
    plt.ylim(4.5e5, 5.75e5)

    plt.xticks([0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0])
    plt.yticks([4.5*(10**5), 4.75*(10**5), 5*(10**5), 5.25*(10**5), 5.5*(10**5), 5.75*(10**5)])

    plt.xlabel("Best Instance Accuracy")
    plt.ylabel("Total Parameters")
    plt.title("PointNet Performance")
    ply.show()



result_file = open("/home/hice1/htirumalai3/scratch/Pointnet_Pointnet2_pytorch/result/results.txt", 'rt').read()
best_instance_accuracy = float(result_file.split(",")[0])
total_params = int(result_file.split(",")[1])
pointnet2_results_directory = "/home/hice1/htirumalai3/scratch/llm-guided-evolution-fork/sota/Pointnet_Pointnet2_pytorch/results"
plot_figure = plot_creation(best_instance_accuracy, total_params, pointnet2_results_directory)
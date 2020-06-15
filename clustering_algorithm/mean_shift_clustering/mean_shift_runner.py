import mean_shift as ms
from numpy import genfromtxt


def load_points(filename):
    data = genfromtxt(filename, delimiter=',')
    return data


def run():
    reference_points = load_points("data.csv")
    mean_shifter = ms.MeanShift()
    mean_shift_result = mean_shifter.cluster(reference_points, kernel_bandwidth=3)

    print("Original Point     Shifted Point  Cluster ID")
    print("============================================")
    for i in range(len(mean_shift_result.shifted_points)):
        original_point = mean_shift_result.original_points[i]
        converged_point = mean_shift_result.shifted_points[i]
        cluster_assignment = mean_shift_result.cluster_ids[i]
        print("(%5.2f,%5.2f)  ->  (%5.2f,%5.2f)  cluster %i" % (original_point[0], original_point[1], converged_point[0], converged_point[1], cluster_assignment))


if __name__ == '__main__':

    import mean_shift as ms
    import matplotlib.pyplot as plt
    import numpy as np

    data = np.genfromtxt('data.csv', delimiter=',')

    mean_shifter = ms.MeanShift()
    mean_shift_result = mean_shifter.cluster(data, kernel_bandwidth = 1)

    original_points =  mean_shift_result.original_points
    shifted_points = mean_shift_result.shifted_points
    cluster_assignments = mean_shift_result.cluster_ids

    x = original_points[:,0]
    y = original_points[:,1]
    Cluster = cluster_assignments
    centers = shifted_points

    fig = plt.figure()
    ax = fig.add_subplot(111)
    scatter = ax.scatter(x,y,c=Cluster,s=50)
    for i,j in centers:
        ax.scatter(i,j,s=50,c='red',marker='+')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar(scatter)
    plt.show()

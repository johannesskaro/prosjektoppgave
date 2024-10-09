import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
from sklearn.mixture import GaussianMixture
from shapely.geometry import Polygon
from scipy.interpolate import splprep, splev


def fit_gmm(points, n_components):
    # Number of components (Gaussians) in the mixture; tune this value

    # Fit a GMM to the points
    gmm = GaussianMixture(n_components=n_components, covariance_type='full')
    gmm.fit(points)

    # Get the means (centers) of the Gaussians and covariances
    #means = gmm.means_
    #covariances = gmm.covariances_

    return gmm

def plot_gmm(gmm, points):
    fig, ax = plt.subplots()
    ax.scatter(points[:, 0], points[:, 1], s=2)  # Plot original points
    
    for mean, cov in zip(gmm.means_, gmm.covariances_):
        # Plot each Gaussian as an ellipse
        eigvals, eigvecs = np.linalg.eigh(cov)
        angle = np.arctan2(eigvecs[0, 1], eigvecs[0, 0])
        angle = np.degrees(angle)
        
        width, height = 2 * np.sqrt(eigvals)  # 2 standard deviations (95% confidence)
        ellipse = Ellipse(mean, width, height, angle=angle, edgecolor='red', facecolor='none')
        ax.add_patch(ellipse)
    plt.show()

def get_contour_points_from_gmm(gmm, n_samples = 50):

    #new_points, _ = gmm.sample(n_samples)
    new_points = gmm.means_

    points_sorted = new_points[np.argsort(new_points[:, 0])]
    origin = np.array([0, 0])
    points = np.vstack([points_sorted, origin])

    return Polygon(points)


def fit_b_spline(points, num_new_points=100):
    #print("Points")

    x = points[:, 0]
    y = points[:, 1]
    points = np.vstack([x, y]).T
    #print(points)

    tck, u = splprep([x, y], s=0.5, k=5)
    x_values = np.linspace(points[0,0], points[-1,0], num_new_points)
    y_values = splev(x_values, tck)

    new_points = np.zeros((num_new_points, 2))
    new_points[:, 0] = x_values
    new_points[:, 1] = y_values[0]


    return Polygon(new_points)


def plot_b_spline(points, new_points):
    fig, ax = plt.subplots()
    
    # Plot original polygon
    ax.plot(points[:, 0], points[:, 1], 'bo-', label='Original Polygon', markersize=5)
    
    # Plot smoothed polygon
    ax.plot(new_points[0], new_points[1], 'r-', label='Smoothed B-spline', linewidth=2)
    # Add labels and styling
    ax.set_title("Original vs Smoothed Polygon using B-spline")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.axis('equal')  # Maintain equal scaling for x and y
    ax.legend()
    plt.show()




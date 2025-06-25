import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.special import sph_harm
import time

# Physical constants
a = 5.29e-2  # Bohr radius (nm)
D_max = 1.1   # Maximum probability density
r0 = 0.25     # Convergence radius (nm)

# Ground state wave function (spherical coordinates)
def wave_function(r, theta, phi):
    """Hydrogen atom ground state wave function (n=1, l=0, m=0)"""
    # Radial part
    R = 2 / (a**1.5) * np.exp(-r / a)
    # Angular part (spherical harmonics)
    Y = sph_harm(0, 0, phi, theta).real
    return R * Y

# Probability density function
def probability_density(r):
    """Radial probability density function D(r) for ground state"""
    return (4 * r**2 / a**3) * np.exp(-2 * r / a)

# Generate electron positions using rejection sampling
def generate_electron_positions(num_points):
    """Generate electron position coordinates"""
    points = []
    count = 0
    
    while count < num_points:
        # Generate random point in spherical coordinates
        r = np.random.uniform(0, r0)
        theta = np.random.uniform(0, np.pi)
        phi = np.random.uniform(0, 2 * np.pi)
        
        # Convert to Cartesian coordinates
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        
        # Calculate probability density
        prob = np.abs(wave_function(r, theta, phi))**2
        
        # Rejection sampling
        if np.random.uniform(0, D_max) < prob:
            points.append((x, y, z))
            count += 1
    
    return np.array(points)

# Visualize electron cloud
def plot_electron_cloud(points, title="Hydrogen Atom Ground State Electron Cloud"):
    """3D visualization of electron cloud distribution"""
    fig = plt.figure(figsize=(12, 10))
    
    # 3D scatter plot
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.scatter(points[:,0], points[:,1], points[:,2], 
               s=1, alpha=0.3, c='blue')
    ax1.set_title(title)
    ax1.set_xlabel('X (nm)')
    ax1.set_ylabel('Y (nm)')
    ax1.set_zlabel('Z (nm)')
    ax1.set_xlim(-r0, r0)
    ax1.set_ylim(-r0, r0)
    ax1.set_zlim(-r0, r0)
    
    # XY plane projection
    ax2 = fig.add_subplot(222)
    ax2.scatter(points[:,0], points[:,1], s=1, alpha=0.3, c='blue')
    ax2.set_title('XY Plane Projection')
    ax2.set_xlabel('X (nm)')
    ax2.set_ylabel('Y (nm)')
    ax2.set_xlim(-r0, r0)
    ax2.set_ylim(-r0, r0)
    ax2.grid(True)
    
    # XZ plane projection
    ax3 = fig.add_subplot(223)
    ax3.scatter(points[:,0], points[:,2], s=1, alpha=0.3, c='blue')
    ax3.set_title('XZ Plane Projection')
    ax3.set_xlabel('X (nm)')
    ax3.set_ylabel('Z (nm)')
    ax3.set_xlim(-r0, r0)
    ax3.set_ylim(-r0, r0)
    ax3.grid(True)
    
    # YZ plane projection
    ax4 = fig.add_subplot(224)
    ax4.scatter(points[:,1], points[:,2], s=1, alpha=0.3, c='blue')
    ax4.set_title('YZ Plane Projection')
    ax4.set_xlabel('Y (nm)')
    ax4.set_ylabel('Z (nm)')
    ax4.set_xlim(-r0, r0)
    ax4.set_ylim(-r0, r0)
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()

# Analyze radial distribution
def plot_radial_distribution(a_value=None):
    """Plot radial probability distribution"""
    if a_value is None:
        a_value = a
    
    r_vals = np.linspace(0, r0, 500)
    D_vals = (4 * r_vals**2 / a_value**3) * np.exp(-2 * r_vals / a_value)
    
    plt.figure(figsize=(10, 6))
    plt.plot(r_vals, D_vals, 'b-', linewidth=2)
    plt.title(f"Hydrogen Atom Ground State Radial Probability Distribution (a={a_value:.4f} nm)")
    plt.xlabel("Radius r (nm)")
    plt.ylabel("Probability Density D(r)")
    plt.grid(True)
    
    # Mark most probable radius
    r_max = a_value
    D_max_val = (4 * r_max**2 / a_value**3) * np.exp(-2 * r_max / a_value)
    plt.plot(r_max, D_max_val, 'ro')
    plt.annotate(f'r = {r_max:.3f} nm', 
                xy=(r_max, D_max_val), 
                xytext=(r_max+0.02, D_max_val*0.8),
                arrowprops=dict(facecolor='black', shrink=0.05))
    
    plt.show()

# Parameter impact analysis
def parameter_analysis():
    """Analyze the impact of different parameters on electron cloud distribution"""
    # Impact of different a values
    a_values = [a/2, a, a*2]
    colors = ['blue', 'green', 'red']
    labels = [f'a = {av:.4f} nm' for av in a_values]
    
    plt.figure(figsize=(10, 6))
    for a_val, color, label in zip(a_values, colors, labels):
        r_vals = np.linspace(0, r0, 500)
        D_vals = (4 * r_vals**2 / a_val**3) * np.exp(-2 * r_vals / a_val)
        plt.plot(r_vals, D_vals, color, linewidth=2, label=label)
    
    plt.title("Impact of Bohr Radius on Radial Probability Distribution")
    plt.xlabel("Radius r (nm)")
    plt.ylabel("Probability Density D(r)")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Impact of different convergence radii
    r0_values = [0.15, 0.25, 0.35]
    plt.figure(figsize=(10, 6))
    for r0_val, color in zip(r0_values, colors):
        r_vals = np.linspace(0, r0_val, 500)
        D_vals = (4 * r_vals**2 / a**3) * np.exp(-2 * r_vals / a)
        plt.plot(r_vals, D_vals, color, linewidth=2, label=f'r0 = {r0_val} nm')
    
    plt.title("Impact of Convergence Radius on Radial Probability Distribution")
    plt.xlabel("Radius r (nm)")
    plt.ylabel("Probability Density D(r)")
    plt.legend()
    plt.grid(True)
    plt.show()

# Main function
def main():
    print("Hydrogen Atom Ground State Electron Cloud Simulation")
    print(f"Parameters: Bohr radius a = {a} nm, Convergence radius r0 = {r0} nm")
    
    # Generate electron positions
    num_points = 10000
    print(f"Generating {num_points} electron positions...")
    start_time = time.time()
    points = generate_electron_positions(num_points)
    print(f"Generation completed! Time taken: {time.time() - start_time:.2f} seconds")
    
    # Visualize electron cloud
    plot_electron_cloud(points)
    
    # Plot radial distribution
    plot_radial_distribution()
    
    # Parameter impact analysis
    parameter_analysis()

if __name__ == "__main__":
    main()

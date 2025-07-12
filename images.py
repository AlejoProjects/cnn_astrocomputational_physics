import os
import numpy as np
import matplotlib.pyplot as plt
import lightcurve_generator as lg  
from pathlib import Path   # path needs to include the folder you saved it in
# Clean the directory before generating new light curves
# Generate and plot a fresh eclipsing-binary curve
objects = ['normal_star', 'pulsating_star', 'binary_star','exoplanet']
def clean_directories():
    """
    Clean the directories for each astrophysical object.
    """
    # Clean the directories for each object type
    for h in objects:
        if h == 'exoplanet':
            break
        lg.clean_directory(h)
def clear_terminal():
    """
    Clear the terminal screen.
    """

    os.system('cls' if os.name == 'nt' else 'clear')

def generate_light_curves(number):
    """
    Generate light curves for different astrophysical objects.
    """
    # Generate light curves for each object type
    for i in range(number):
        print(f"Generated light curves  progress: {(i/number)*100:.2f}%")
        lg.generate_light_curve(objects[0],i,n_points=7000,noise_std=0.0015,return_data=False,plot=False,download_fig=True)
        lg.generate_light_curve(objects[1],i,n_points=7000,noise_std=0.0015,return_data=False,plot=False,download_fig=True)
        lg.generate_light_curve(objects[2],i,n_points=7000,noise_std=0.0015,return_data=False,plot=False,download_fig=True)
 
        
       
    print("Light curves generated successfully.")
def make_and_save_curve(idx: int,
                        out_dir: Path,
                        noise_std: float =1.5e-3,#★ NEW: amplitude of noise
                        n_points: int = 900):
    """
    Creates one randomly-parameterised light curve, adds Gaussian noise,
    and writes it to 'transit_XX.png'.

    Parameters
    ----------
    idx        : image index for filename
    out_dir    : destination directory
    noise_std  : σ of the white noise (relative flux units, default 5×10⁻⁴)
    n_points   : resolution of the time axis
    """
    p = random_params()

    # time axis slightly wider than the transit itself
    t = np.linspace(p["t0"] - 1.2*p["width"],
                    p["t0"] + 1.2*p["width"],
                    n_points)

    flux = inverted_parabola(t, **p)

    # ★ Add white Gaussian noise
    flux += rng.normal(0.0, noise_std, size=flux.shape)

    # ------------------------------------------------------------------
    # plotting
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(t - p["t0"], flux, lw=1.3,color='black')
    ax.tick_params(axis='both', which='both',
                   bottom=False, top=False, left=False, right=False,
                   labelbottom=False, labelleft=False)
    ax.set_ylim(1 - 1.4*p["depth"], 1.01)          # leave head-room for noise

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / f"transit_{idx:02d}.png", dpi=130)
    lg.resize_image(out_dir / f"transit_{idx:02d}.png", target_size=(128, 128))
    plt.close(fig)
    print(f"Saved transit_{idx:02d}.png")
def inverted_parabola(t, t0, depth, width):
    """
    Simple analytic transit profile.
    Flux = 1 outside |x|<=1 and 1 - depth*(1 - x**2) inside,
    where x = (t - t0)/width.
    """
    x = (t - t0) / width
    flux = np.ones_like(t)
    mask = np.abs(x) <= 1.0
    flux[mask] = 1.0 - depth * (1.0 - x[mask] ** 2)
    return flux
rng = np.random.default_rng()
def random_params():
    return dict(
        t0    = 0.0,                          # centre the dip at zero
        depth = rng.uniform(0.005, 0.03),     # 0.5 % – 3 % depth
        width = rng.uniform(0.02, 0.08),      # half-duration (time units)
    )


if __name__ == "__main__":
    output_folder = Path("exoplanet")
    #lg.clean_directory("exoplanet")
    clean_directories()
    generate_light_curves(1100)  # Generate 250 light curves for each object type
    for i in range(400):
       print(f"Generating light curve for {objects[3]} with index {i}")
       lg.generate_light_curve(objects[3],i,n_points=7000,noise_std=0.0015,return_data=False,plot=False,download_fig=True) 
      # clear_terminal()
    #for i in range(4):
     #   lg.generate_light_curve(objects[i],1100,n_points=7000,noise_std=0.0015,return_data=False,plot=False,download_fig=True)

    for i in range(1, 200):       # n noisy snapshots
      make_and_save_curve(i, output_folder)

"""lightcurve_generator.py

A small utility to create synthetic, noisified light‑curves for several classes
of astrophysical objects.  Each call produces a *new* realisation thanks to an
internal random number generator, unless you pass a specific ``random_state``.

Usage
-----
>>> import lightcurve_generator as lg
>>> t, f = lg.generate_light_curve('binary_star', n_points=2000,
...                                 noise_std=0.002, plot=True,
...                                 return_data=True)
"""

from __future__ import annotations
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import Tuple, Optional

__all__ = ["generate_light_curve"]
def clean_directory(directory: str) -> None:
    """Deletes all files in the specified directory."""
  
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                os.rmdir(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')
def check_directory_exists(directory: str) -> None:
    """Check if the specified directory exists, and create it if not."""
    import os
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created.")
    else:
        print(f"Directory '{directory}' already exists.")
def resize_image(image_path, target_size=(28, 28)):
    """
    Resize an image to the target size with antialiasing.
    
    Parameters:
    - image_path: str, path to the image file.
    - target_size: tuple, desired size (width, height).
    
    Returns:
    - resized_image: PIL Image object, resized image.
    """
    img = Image.open(image_path)
    resized_image = img.resize(
        target_size,
        resample=Image.Resampling.LANCZOS
    )
    resized_image.save(image_path)  # Save the resized image
def _rng(random_state: Optional[int | np.random.Generator] = None) -> np.random.Generator:
    """Return a ``np.random.Generator`` built from *random_state*."""
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)

# ------------------------------------------------------------------------
#  Individual light‑curve prototypes
# ------------------------------------------------------------------------
def _normal_star(t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Small‑amplitude quasi‑periodic variability (granulation / spots)."""
    amp   = rng.uniform(1e-3, 5e-2)
    period= rng.uniform(1.0, 30.0)
    phase = rng.uniform(0, 2*np.pi)
    trend = rng.uniform(-1e-4, 1e-4) * t
    return 1.0 + amp*np.sin(2*np.pi*t/period + phase) + trend

def _pulsating_star(t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Simple Cepheid‑like sinusoid with larger amplitude."""
    amp   = rng.uniform(5e-2, 4e-1)
    period= rng.uniform(1.0, 100.0)
    phase = rng.uniform(0, 2*np.pi)
    return 1.0 + amp*np.sin(2*np.pi*t/period + phase)

def _binary_star(t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Detached eclipsing binary with primary & secondary eclipses."""
    period       = rng.uniform(0.5, 10.0)
    primary_depth= rng.uniform(0.05, 0.5)
    secondary_depth = primary_depth*rng.uniform(0.3, 0.8)
    primary_width   = rng.uniform(0.02, 0.1) * period
    secondary_width = primary_width * rng.uniform(0.8, 1.2)
    phase  = (t % period)

    flux = np.ones_like(t)
    # Primary eclipse centred at phase=0
    in_primary = (phase < primary_width/2) | (phase > period - primary_width/2)
    flux[in_primary] -= primary_depth * np.cos(np.pi*phase[in_primary]/primary_width)**2

    # Secondary eclipse centred at phase=period/2
    in_secondary = np.abs(phase - period/2) < secondary_width/2
    flux[in_secondary] -= secondary_depth * np.cos(np.pi*(phase[in_secondary]-period/2)/secondary_width)**2
    return flux

def _exoplanet(t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Box‑like transit of a single exoplanet (no limb darkening)."""
    period       = rng.uniform(1.0, 20.0)
    depth        = rng.uniform(5e-4, 3e-2)   # up to 3 %
    duration     = rng.uniform(0.02, 0.1) * period
    phase  = (t % period)

    flux = np.ones_like(t)
    in_transit = (phase < duration/2) | (phase > period - duration/2)
    flux[in_transit] -= depth
    return flux

_MODELS = {
    'normal_star':    _normal_star,
    'pulsating_star': _pulsating_star,
    'binary_star':    _binary_star,
    'exoplanet':      _exoplanet,
}

# ------------------------------------------------------------------------
#  Public function
# ------------------------------------------------------------------------
def generate_light_curve(object_type: str,type_value,
                         n_points: int = 1024,
                         noise_std: float = 1e-3,
                         random_state: Optional[int | np.random.Generator] = None,
                         plot: bool = False,
                         return_data: bool = False,
                         download_fig: bool = False
                         ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Generate *and optionally plot* a synthetic light‑curve.

    Parameters
    ----------
    object_type : str
        One of ``'normal_star', 'pulsating_star', 'binary_star', 'exoplanet'``.
    n_points : int, default=1024
        Number of time samples.
    noise_std : float, default=1e-3
        Standard deviation of the additive Gaussian noise.
    random_state : int | ~numpy.random.Generator | None, default=None
        Seed or generator for reproducibility.
    plot : bool, default=False
        If *True*, immediately show a matplotlib figure.
    return_data : bool, default=False
        If *True*, return *(time, flux)* arrays.

    Returns
    -------
    (time, flux) : tuple[np.ndarray, np.ndarray] | None
        Returned only when *return_data=True*.
    """
    if object_type not in _MODELS:
        raise ValueError(f"Unknown object_type '{object_type}'. Valid keys are: {list(_MODELS)}")

    rng = _rng(random_state)
    # Simulate over twice the typical period range for that model to guarantee features
    # Pick a 'total_duration' that ensures at least two cycles/events
    total_duration = {
        'normal_star':    rng.uniform(30.0, 90.0),
        'pulsating_star': rng.uniform(10.0, 300.0),
        'binary_star':    rng.uniform(3.0, 30.0),
        'exoplanet':      rng.uniform(5.0, 40.0),
    }[object_type]

    t = np.linspace(0.0, total_duration, n_points)
    flux = _MODELS[object_type](t, rng)

    # Add Gaussian noise
    flux += rng.normal(0.0, noise_std, size=n_points)
    if download_fig:
        imag_path = f"{object_type}\\{type_value}_light_curve.png"
        #check_directory_exists(object_type)
        plt.figure()
        plt.plot(t, flux, linestyle='-', marker='', linewidth=1,color='black')
        plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
        #plt.xlabel('Time (days)')
        #plt.ylabel('Relative flux')
        #plt.title(f'Synthetic light‑curve: {object_type}')
        plt.tight_layout()
        plt.savefig(imag_path)
        plt.close()
        resize_image(imag_path, target_size=(128, 128))
        #print(f"Figure saved as {object_type}_light_curve.png")
    if plot:
        plt.figure()
        plt.plot(t, flux, linestyle='-', marker='', linewidth=1)
        #plt.xlabel('Time (days)')
        #plt.ylabel('Relative flux')
        #plt.title(f'Synthetic light‑curve: {object_type}')
        plt.tight_layout()
        plt.show()
    
    if return_data:
        return t, flux
  
    return None

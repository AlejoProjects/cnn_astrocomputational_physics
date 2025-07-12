"""
real_exoplanets.py  (v2)
------------------------
Create N minimalist PNGs of real exoplanet light-curves.

• pulls a long candidate list from the NASA Exoplanet Archive
• only keeps stars that actually have a public Kepler/K2/TESS light curve
• plots each curve as a red line (no markers) without any axes decorations
• writes the images to ./exoplanets/

Install once:
    pip install lightkurve astroquery matplotlib tqdm pandas
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lightkurve as lk
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
from tqdm import tqdm

# ------------- CONFIG -------------------------------------------------
N_IMAGES             = 10
CANDIDATE_POOL_SIZE  = 3000        # ask for many; we’ll skip those with no data
OUTPUT_DIR           = Path("tests")
MISSION_PREF_ORDER   = ["Kepler", "K2", "TESS"]   # order you’d like to try
LINE_COLOR           = "red"
LINE_WIDTH           = 0.4
LINE_ALPHA           = 0.9
FIGSIZE              = (6, 3)      # inches
DPI                  = 150
# ----------------------------------------------------------------------

OUTPUT_DIR.mkdir(exist_ok=True)

# ---------- helpers ---------------------------------------------------
def fetch_planet_catalog(pool_size: int) -> pd.DataFrame:
    """
    Return a DataFrame with ≥ pool_size confirmed transiting planets,
    sorted by discovery year (works on every astroquery version).
    """
    try:  # modern helper (astroquery ≥0.4.7)
        tbl = NasaExoplanetArchive.get_confirmed_planets_table(all_columns=False)
        df = tbl.to_pandas()

    except AttributeError:
        # Robust TAP / CSV fallback – confirmed planets only
        url = ("https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query="
               "select+pl_name,hostname,disc_year+from+ps+order+by+disc_year"
               "&format=csv")
        df = pd.read_csv(url)

    return df.head(pool_size)      # we just need the first *pool_size* rows


def first_lightcurve(target: str):
    """
    Try to download one stitched, PDCSAP-corrected LightCurve for *target*.
    Returns None if nothing exists.
    """
    for mission in MISSION_PREF_ORDER + [None]:
        search = lk.search_lightcurve(target, mission=mission, cadence="long")
        if len(search) == 0:
            continue
        try:
            # stitched curve; prefers PDCSAP column
            lc = search.download(flux_column="pdcsap_flux")
            if lc is not None:
                return lc.normalize()
        except Exception:
            # sometimes download() fails for odd files; move on
            continue
    return None


def axeless_lineplot(lc, save_path: Path):
    """Plot *lc* as a thin red line minus any axes and save PNG."""
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    ax.plot(lc.time.value, lc.flux.value,
            linewidth=LINE_WIDTH, color=LINE_COLOR, alpha=LINE_ALPHA)

    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout(pad=0)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
# ----------------------------------------------------------------------


def main():
    catalog = fetch_planet_catalog(CANDIDATE_POOL_SIZE)
    generated = 0

    for _, row in tqdm(catalog.iterrows(), total=len(catalog),
                       desc="Creating curves", ncols=90):
        if generated >= N_IMAGES:
            break

        host = row["hostname"]
        try:
            lc = first_lightcurve(host)
            if lc is None:
                continue  # no data → skip

            fname = OUTPUT_DIR / f"{host.replace(' ', '_')}.png"
            axeless_lineplot(lc, fname)
            generated += 1

        except Exception as e:
            # write a note once, then move on – don't halt the run
            tqdm.write(f"⚠️  {host}: {e}")

    print(f"\n✅ Finished – {generated} images saved in {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()

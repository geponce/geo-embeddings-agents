"""Stage 1 smoke test: read one embedding pixel from Earth Engine.

Run with:  python -m geo_embeddings.smoke
"""

from __future__ import annotations

from geo_embeddings.ee_auth import init_ee
from geo_embeddings.embeddings import l2_norm, read_pixel

# Irrigated farmland near Eloy, Pinal County, Arizona.
DEFAULT_LON, DEFAULT_LAT, DEFAULT_YEAR = -111.70, 32.80, 2024


def main() -> None:
    """Initialize Earth Engine, read one pixel and print a short summary."""
    init_ee()
    v = read_pixel(DEFAULT_LON, DEFAULT_LAT, DEFAULT_YEAR)
    print(f"point ({DEFAULT_LON}, {DEFAULT_LAT}), year {DEFAULT_YEAR}")
    print(f"n values: {len(v)}")
    print(f"first 8: {[round(x, 4) for x in v[:8]]}")
    print(f"min/max: {min(v):.4f} {max(v):.4f}")
    print(f"L2 norm: {l2_norm(v):.4f}")


if __name__ == "__main__":
    main()

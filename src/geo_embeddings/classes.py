"""Group USDA Cropland Data Layer (CDL) codes into the project's training classes.

The CDL has about 130 codes. Pinal County, Arizona uses about 45 of them, most
covering tiny areas. We group them into 11 classes that are common enough to
sample and meaningful for a land-cover map. Codes for water, wetland marsh,
snow and background are excluded (EXCLUDED_ID).

CDL code list: https://www.nass.usda.gov/Research_and_Science/Cropland/metadata/
"""

from __future__ import annotations

EXCLUDED_ID = -1

# Class id -> class name. Ids are contiguous from 0 so they work as model labels.
CLASS_NAMES: dict[int, str] = {
    0: "Shrubland",
    1: "Grass/Pasture",
    2: "Alfalfa",
    3: "Cotton",
    4: "Corn+Sorghum",
    5: "Small grains",
    6: "Other crops",
    7: "Fallow/Idle",
    8: "Developed",
    9: "Barren",
    10: "Woody",
}

# Explicit CDL code -> class id. Codes 63-65 and 62 are older CDL codes kept for safety.
_EXPLICIT: dict[int, int] = {
    152: 0, 64: 0,                                   # Shrubland
    176: 1, 62: 1,                                   # Grass/Pasture
    36: 2,                                           # Alfalfa
    2: 3,                                            # Cotton
    1: 4, 4: 4,                                      # Corn, Sorghum
    21: 5, 22: 5, 23: 5, 24: 5, 25: 5, 27: 5, 28: 5, 205: 5,  # Barley, wheats, rye, oats, triticale
    61: 7,                                           # Fallow/Idle Cropland
    121: 8, 122: 8, 123: 8, 124: 8, 82: 8,           # Developed (all intensities)
    131: 9, 65: 9,                                   # Barren
    141: 10, 142: 10, 143: 10, 190: 10, 63: 10,      # Forests, woody wetlands
}

# CDL code ranges that are crops. Any crop code not listed above becomes "Other crops".
_CROP_RANGES: tuple[range, ...] = (range(1, 61), range(66, 81), range(204, 255))


def cdl_to_class(code: int) -> int:
    """Return the class id for one CDL code, or EXCLUDED_ID if it is not used."""
    if code in _EXPLICIT:
        return _EXPLICIT[code]
    if any(code in r for r in _CROP_RANGES):
        return 6
    return EXCLUDED_ID


def remap_lists(codes: range = range(0, 256)) -> tuple[list[int], list[int]]:
    """Return (from_codes, to_ids) for ee.Image.remap, skipping excluded codes.

    Codes left out of the lists are masked by remap, which drops them from sampling.
    """
    from_codes, to_ids = [], []
    for code in codes:
        cls = cdl_to_class(code)
        if cls != EXCLUDED_ID:
            from_codes.append(code)
            to_ids.append(cls)
    return from_codes, to_ids

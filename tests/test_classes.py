"""Tests for the CDL code -> class grouping."""

from geo_embeddings.classes import CLASS_NAMES, EXCLUDED_ID, cdl_to_class, remap_lists


def test_class_ids_are_contiguous() -> None:
    assert sorted(CLASS_NAMES) == list(range(len(CLASS_NAMES)))


def test_major_pinal_codes() -> None:
    assert cdl_to_class(152) == 0   # Shrubland
    assert cdl_to_class(176) == 1   # Grass/Pasture
    assert cdl_to_class(36) == 2    # Alfalfa
    assert cdl_to_class(2) == 3     # Cotton
    assert cdl_to_class(1) == 4     # Corn
    assert cdl_to_class(4) == 4     # Sorghum
    assert cdl_to_class(21) == 5    # Barley
    assert cdl_to_class(22) == 5    # Durum wheat
    assert cdl_to_class(205) == 5   # Triticale
    assert cdl_to_class(61) == 7    # Fallow/Idle
    assert cdl_to_class(124) == 8   # Developed high intensity
    assert cdl_to_class(131) == 9   # Barren
    assert cdl_to_class(190) == 10  # Woody wetlands


def test_unlisted_crops_become_other_crops() -> None:
    assert cdl_to_class(37) == 6    # Other hay
    assert cdl_to_class(43) == 6    # Potatoes
    assert cdl_to_class(47) == 6    # Misc vegs and fruits
    assert cdl_to_class(242) == 6   # Cabbage (not a double crop despite the code range)


def test_double_crops_and_orchards() -> None:
    for code in (26, 225, 226, 228, 236, 241, 254):  # Dbl Crop codes
        assert cdl_to_class(code) == 11
    for code in (72, 74, 204, 211, 212, 69):         # citrus, pecans, pistachios, olives, oranges, grapes
        assert cdl_to_class(code) == 12


def test_excluded_codes() -> None:
    for code in (0, 111, 195, 112, 87):  # background, water, herb. wetland, snow, wetlands
        assert cdl_to_class(code) == EXCLUDED_ID


def test_remap_lists_skip_excluded() -> None:
    from_codes, to_ids = remap_lists()
    assert len(from_codes) == len(to_ids)
    assert 111 not in from_codes
    assert EXCLUDED_ID not in to_ids
    assert set(to_ids) == set(CLASS_NAMES)

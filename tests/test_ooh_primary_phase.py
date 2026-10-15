from abcde_ooh.constraints.primary_phase import check_primary_phase


def test_unlisted_cation_still_counts_toward_dopant_cap():
    # Hf occurs in another OOH config but was historically absent from the
    # fixed dopant set. Future cations must still count toward the cap.
    assert check_primary_phase({"Ni": 0.75, "Hf": 0.25}) == (False, None)
    assert check_primary_phase({"Ni": 0.80, "Hf": 0.20})[0]


def test_all_33_main_ooh_species_work_as_subcap_dopants():
    dopants = {
        "Mg", "Al", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Cu", "Zn", "Ga",
        "Sr", "Y", "Zr", "In", "La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd",
        "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Bi",
    }
    for dopant in dopants:
        filler = "Al" if dopant == "Mg" else "Mg"
        ok, _ = check_primary_phase(
            {"Ni": 0.75, "Fe": 0.05, "Co": 0.05, dopant: 0.10, filler: 0.05}
        )
        assert ok, dopant

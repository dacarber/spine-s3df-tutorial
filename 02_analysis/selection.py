"""Event selection used by 02b_analyze_spine_output.ipynb.

The notebook applies the SAME function to reconstructed interactions
(`RecoInteraction`) and to true interactions (`TruthInteraction`). Using one
definition for both is what makes the efficiency and purity numbers meaningful:

    efficiency = (true interactions that pass AND whose matched reco also passes)
                 / (true interactions that pass)
    purity     = (reco interactions that pass AND whose matched truth also passes)
                 / (reco interactions that pass)

Attributes available on both RecoInteraction and TruthInteraction
-----------------------------------------------------------------
inter.is_fiducial              bool, vertex inside the fiducial volume
inter.is_contained             bool, all of the interaction's charge is inside the detector
inter.vertex                   numpy array (x, y, z) in cm
inter.num_primary_particles    int
inter.primary_particle_counts  numpy array of 6 ints: number of PRIMARY
                               [photon, electron, muon, pion, proton, kaon]
                               (index with the PID constants below)
inter.topology                 str, e.g. "1mu1p" or "1e2p1pi"
inter.primary_particles        list of particle objects; each has .pid, .ke (MeV),
                               .length (cm), .is_contained, .start_point, ...

Edit this file, save it, and re-run the notebook cells below the import:
the notebook reloads it automatically.
"""

# Particle-ID numbers used by SPINE (same as spine.constants.PID_LABELS)
PHOTON, ELECTRON, MUON, PION, PROTON, KAON = range(6)


def select_interaction(inter):
    """Return True if `inter` passes the selection, False otherwise."""
    # TODO(human): implement your selection here (about 5-10 lines).
    raise NotImplementedError("select_interaction() is not written yet: see selection.py")

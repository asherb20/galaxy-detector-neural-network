import numpy as np

"""Derive galaxy morphology classes from the Galaxy Zoo 2 catalog"""

CLASSES = ["elliptical", "spiral", "barred_spiral", "lenticular", "edge_on_disk", "irregular"]

# morph label for galaxies that should not train the morphology head
IGNORE = -1

ARTIFACT_THRESHOLD = 0.3
ODD_THRESHOLD = 0.5

# GZ2 `label` column values (see `summary` column)
GZ2_SMOOTH = [0, 1, 2]  # smooth_inbetween, smooth_round, smooth_cigar
GZ2_EDGE_ON = 3
GZ2_SPIRAL = 4
GZ2_BARRED_SPIRAL = 5
GZ2_FEATURED_PLAIN = 6  # featured_without_bar_or_spiral
GZ2_UNCERTAIN = -1


def artifact_mask(catalog):
   """True for rows volunteers mostly flagged as a star or artifact (used as non-galaxies)"""
   return catalog['smooth-or-featured-gz2_artifact_fraction'].to_numpy() > ARTIFACT_THRESHOLD


def morphology_labels(catalog):
   """Map each catalog row to an index into CLASSES, or IGNORE.

   GZ2 has no explicit lenticular or irregular labels, so these are approximations:
   - lenticular: a disk with no bar and no spiral arms (S0-like)
   - irregular: mostly voted "something odd" and not a clear spiral or smooth galaxy.
     "odd" also covers rings, lenses and dust lanes, so this class is noisy.
   """
   gz2_label = catalog['label'].to_numpy()
   odd = catalog['something-odd-gz2_yes_fraction'].to_numpy() > ODD_THRESHOLD

   labels = np.full(len(catalog), IGNORE, dtype=np.int64)
   labels[np.isin(gz2_label, GZ2_SMOOTH)] = CLASSES.index("elliptical")
   labels[gz2_label == GZ2_SPIRAL] = CLASSES.index("spiral")
   labels[gz2_label == GZ2_BARRED_SPIRAL] = CLASSES.index("barred_spiral")
   labels[gz2_label == GZ2_FEATURED_PLAIN] = CLASSES.index("lenticular")
   labels[gz2_label == GZ2_EDGE_ON] = CLASSES.index("edge_on_disk")

   # irregular takes precedence over lenticular / uncertain
   labels[odd & np.isin(gz2_label, [GZ2_FEATURED_PLAIN, GZ2_UNCERTAIN])] = CLASSES.index("irregular")

   # artifacts are not galaxies at all
   labels[artifact_mask(catalog)] = IGNORE
   return labels

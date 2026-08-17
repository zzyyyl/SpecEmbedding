from SpecEmbedding.paths import MSBERT_ROOT

############################## RAW DATA #################################
DIR = MSBERT_ROOT / "MoNA"

ORBITRAP_RAW = "MoNAOrbitrap.msp"
QTOF_RAW = "MoNAQTOF.msp"

EXTRACTED = "all.npy"
FILTERED = "filter.npy"
CLEANED = "clean.npy"
POSITIVE = "positive.npy"
ANNOTATED = "annotated.npy"

COMMON = "common.npy"
UNIQUE = "unique.npy"

ORBITRAP_DIR = DIR / "orbitrap"
ORBITRAP_COMMON = ORBITRAP_DIR / COMMON
ORBITRAP_UNIQUE = ORBITRAP_DIR / UNIQUE

QTOF_DIR = DIR / "qtof"
QTOF_COMMON = QTOF_DIR / COMMON
QTOF_UNIQUE = QTOF_DIR / UNIQUE

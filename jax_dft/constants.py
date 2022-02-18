"""Default constants used in this library"""
# Exponential Coulomb interaction
# v(x) = amplitude * exp(-abs(x) * kappa)

EXPONENTIAL_COULOMB_AMPLITUDE = 1.071295
EXPONENTIAL_COULOMB_KAPPA = 1 / 2.385345

# Soft Coulomb interaction
SOFT_COULOMB_SOFTEN_FACTOR = 1.

# Chemical accuracy 0.0016 Hartree = 1 kcal/mol
CHEMICAL_ACCURACY = 0.0016
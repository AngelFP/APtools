"""Defiles the ParticleDistribution class"""

import numpy as np


class ParticleDistribution():
    """Class for storing a particle distribution.

    Parameters
    ----------
    x, y, xi : ndarray
        Position of the macropparticles in the x, y, and xi directions in
        units of m.
    px, py, pz : ndarray
        Momentum of the macropparticles in the x, y, and z directions in
        non-dimensional units (beta*gamma).
    w : ndarray
        Weight of the macroparticles, i.e., the number of real particles
        represented by each macroparticle. In practice, :math:`w = q_m / q`,
        where :math:`q_m` and :math:`q` are, respectively, the charge of the
        macroparticle and of the real particle (e.g., an electron).
    q_species, m_species : float
        Charge and mass of a single particle of the species represented
        by the macroparticles. For an electron bunch,
        ``q_species=-e`` and ``m_species=m_e``
    """
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        px: np.ndarray,
        py: np.ndarray,
        pz: np.ndarray,
        w: np.ndarray,
        q_species: float,
        m_species: float
    ) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.px = px
        self.py = py
        self.pz = pz
        self.w = w
        self.q_species = q_species
        self.m_species = m_species

    @property
    def q(self) -> np.ndarray:
        """Return the total macroparticle charge."""
        return self.w * self.q_species

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 14:45:27 2022

@author: jiafanhao

This code is based on 
https://github.com/coudertlab/elate

Here, I just delete all of web part. 

"""
import numpy as np

class Elastic_polycrystal:
    def __init__(self, elastic_tensor, density=None, masses=None):
        """
        Parameters
        ----------
        elastic_tensor : TYPE
            DESCRIPTION.
        structure : TYPE, optional
            DESCRIPTION. The default is None.
        crystal_type : TYPE, optional
            DESCRIPTION. The default is None.
        Returns
        -------
        None.

        """
        # Avogadro number
        N_avogadro = 6.02214076e23
        # Planck's constant
        h_Planck = 6.62607015e-34   #m^2kg/s
        # Boltzmann constant J/K or m^2kgs^-2K^-1
        kB = 1.38064852e-23
        # Atomic mass unit. kg
        #amu = 1.66053886e-27

        self.elastic_tensor = np.matrix(elastic_tensor)
        self.compaliance_tensor = self.elastic_tensor.I
        self.density=density
        self.masses=masses
        
        cnew = self.elastic_tensor
        #Bulk Modulus Voigt.
        B_V = (cnew[0, 0] + cnew[1, 1] + cnew[2, 2]) + 2 * (cnew[0, 1] + cnew[1, 2] + cnew[2, 0])
        self.B_V = B_V / 9.0
        #Shear Modulus Voigt.
        G_V = ((cnew[0, 0] + cnew[1, 1] + cnew[2, 2]) - (cnew[0, 1] + cnew[1, 2] + cnew[2, 0])
            + 3 * (cnew[3, 3] + cnew[4, 4] + cnew[5, 5]))
        self.G_V = G_V / 15.0
        #Young's Modulus Voigt.
        self.E_V=(9 * self.B_V * self.G_V) / (3 * self.B_V + self.G_V)
        #Poisson's Ratio Voigt.
        self.P_V=(3 * self.B_V - self.E_V) / (6 * self.B_V)
        #Pugh's Ratio Voigt.
        self.pugh_ratio_voigt=self.B_V/self.G_V
        #P-wave modulus voigt.
        self.p_wave_modulus_voigt=self.B_V + (4 * self.G_V / 3.0)
        
        snew = self.compaliance_tensor
        #Bulk Modulus Reuss.
        B_R = (snew[0, 0] + snew[1, 1] + snew[2, 2]) + 2 * (snew[0, 1] + snew[1, 2] + snew[2, 0] )
        self.B_R = 1.0 / B_R
        #Shear Modulus Reuss.
        G_R = (4 * (snew[0, 0] + snew[1, 1] + snew[2, 2])
            - 4 * (snew[0, 1] + snew[1, 2] + snew[2, 0])
            + 3 * (snew[3, 3] + snew[4, 4] + snew[5, 5]))
        self.G_R = 15.0 / G_R
        #Young's Modulus Reuss.
        self.E_R=(9 * self.B_R * self.G_R) / (3 * self.B_R + self.G_R)
        #Poisson's Ration Reuss.
        self.P_R=(3 * self.B_R - self.E_R) / (6 * self.B_R)
        #Pugh's Ratio Reuss.
        self.pugh_ratio_reuss=self.B_R/self.G_R
        #P-wave modulus Reuss.
        self.p_wave_modulus_reuss=self.B_R + (4 * self.G_R / 3.0)
        
        
        #Voigt-Reuss-Hill Approximation Bulk Modulus.
        self.B_VRH=(self.B_R + self.B_V) / 2
        self.G_VRH=(self.G_V + self.G_R) / 2
        self.E_VRH=(self.E_V + self.E_R) / 2
        self.P_VRH=(3 * self.B_VRH - 2 * self.G_VRH)/(2* (3 * self.B_VRH + self.G_VRH) )
        self.pugh_ratio_voigt_reuss_hill=self.B_VRH/self.G_VRH
        self.p_wave_modulus_voigt_reuss_hill=(self.p_wave_modulus_voigt + self.p_wave_modulus_reuss) / 2.0
        
        #Bulk/Shear ratio voigt.
        self.BG_Ratio_V=self.B_V / self.G_V
        #Bulk/Shear ratio reuss.
        self.BG_Ratio_R=self.B_R / self.G_R
        #Bulk/Shear ratio voigt reuss hill.
        self.BG_Ratio_VRH=self.B_VRH / self.G_VRH
        
        #Zenner Anisotropy only for Cubic Crystals.
        self.A_z=2 * cnew[3, 3] / (cnew[0, 0] - cnew[0, 1])
        #Chung-Buessem only for Cubic Crystals.
        self.A_cb=(self.G_V - self.G_R) / (self.G_V + self.G_R)
        
        
        #    Ranganathan and Ostoja-Starzewski method: Phys. Rev. Lett. 101, 055504 (2008).
        #    for any crystalline symmetry: Universal anisotropy index.
        #    Note: AU is a relative measure of anisotropy with respect to a limiting value.
        #    For example, AU does not prove that a crystal having AU = 3 has double the anisotropy
        #    of another crystal with AU = 1.5. I""'
        self.A_u=(self.B_V / self.B_R) + 5 * (self.G_V / self.G_R) - 6.0
        
        #       log-Euclidean anisotropy parameter by Christopher M. Kube, AIP Advances 6, 095209 (2016)
        #       AL  CV , CR   is based on the distance between the averaged stiffnesses CV and
        #       CR , which is more appropriate. Clearly, AL  CV , CR   is zero when the crystallite is isotropic.        
        self.A_l=np.sqrt((np.log(self.B_V / self.B_R)) ** 2 + 5 * (np.log(self.G_V / self.G_R)) ** 2)
        
        self.lambda_lame_coefficient=self.E_VRH * self.P_VRH / ((1 + self.P_VRH) * (1 - 2 * self.P_VRH))
        
        if self.density is not None:
            G = self.G_VRH * 1.0e9  # converting from GPa to Pascal units (kg/ms^2)
            self.velocity_transverse=np.sqrt((G / self.density))
            K = self.B_VRH * 1.0e9
            self.velocity_logitudinal=np.sqrt(((3 * K + 4 * G) / (3.0 * self.density)))
            
            self.vt = self.velocity_transverse
            self.vl = self.velocity_logitudinal
            #Average Sound velocity(m/s)
            self.vm=1.0 / (np.cbrt((1.0 / 3.0) * (2.0 / (self.vt * self.vt * self.vt) + 1.0 / (self.vl * self.vl * self.vl))))
            self.velocity_average=self.vm
            
            
            #debye_temperature theta : float
            #    Debye temperature calculated using  Orson Anderson's proposal [Ref- J. Phys. Chem. Solids (1963) 24, 909-917].
            #    WARNING: Debye model for the atomic displacement is based on a monoatomic crystal, here we consider an average mass if your crystal has several species            
            total_mass = np.sum(self.masses)
            q=len(self.masses)
            self.debye_temperature= ((h_Planck / kB)* self.vm * np.cbrt((3 * q * self.density) / (4 * (np.pi) * total_mass)))

            #Melting temperature estimated using empirical relation from Ref: Johnston I, Keeler G, Rollins R and Spicklemire S 1996
            #Solid State Physics Simulations, The Consortium for Upper-Level Physics Software (New York: Wiley) 
            self.melting_temperature=607 + 9.3 * self.B_VRH   
            
            
    def hardness(self):
        """

        Returns
        -------
        float
            The hardness calculated by 6 different methods:
            [H1a and H1b] Correlation between hardness and elastic moduli of the covalent crystals. Jiang, et al. (2011).
            [H2] Computational alchemy: the search for new superhard materials. Teter (1998).
            [H3] Mechanical and electronic properties of B12-based ternary crystals of orthorhombic phase. Jiang et al. (2010).
            [H4] Theoretical investigation on the transition-metal borides with Ta3B4-type structure: A class of hard and refractory materials. Miao et al. (2011).
            [H5] Modeling hardness of polycrystalline materials and bulk metallic glasses. Chen et al. (2011).
        """
        B = self.B_VRH
        G = self.G_VRH
        Y = self.E_VRH
        v = self.P_VRH
        k = G / B
        H1a = (1 / 6.78) * G
        H1b = (1 / 16.48) * Y
        H2 = (0.1769 * G) - 2.899
        H3 = (1 / 15.76) * Y
        H4 = ((1 - 2 * v) * B) / (6 * (1 + v))
        H5 = 2 * ((k * k * G) ** 0.585) - 3
        return H1a, H1b, H2, H3, H4, H5
    
    def cauchy_pressure(self):
        """
        This parameter desceibes the nature of bonding
        CP > 0 (+ve) indicates that ionic bonding dominates
        CP < 0 (-ve) indicates that covalent bonding dominates
        Returns
        -------
        None.

        """
        return self.elastic_tensor[0, 1] - self.elastic_tensor[3, 3]

    def bonding_type(self):
        """
        This parameter desceibes the nature of bonding
        CP > 0 (+ve) indicates that ionic bonding dominates
        CP < 0 (-ve) indicates that covalent bonding dominates

        Returns
        -------
        str
            DESCRIPTION.

        """
        cauchy_pressure = self.cauchy_pressure
        if cauchy_pressure > 0:
            return "ionic"
        elif cauchy_pressure < 0:
            return "covalent"

    def kleinman_parameter(self):
        c = self.elastic_tensor
        return (c[0, 0] + 8 * c[0, 1]) / (7 * c[0, 0] - 2 * c[0, 1])


    def bond_bending_vs_streching(self):
        return
        
    def mu_lame_coefficient(self):
        return self.E_VRH / (2 * (1 + self.P_VRH))

    def to_dict(self, symprec=1e-5):
        """
        symprec : float
            Precision used in calculating the space group in angstroms. The default is 1e-5.

        dict
            DESCRIPTION.

        """
        return {
            "anisotropy_Chung_Buessem": self.anisotropy_Chung_Buessem,
            "anisotropy_log_euclidean": self.anisotropy_log_euclidean,
            "anisotropy_universal": self.anisotropy_universal,
            "anisotropy_zenner": self.anisotropy_zenner,
            "bond_bending_vs_streching": self.bond_bending_vs_streching,
            "bonding_type": self.bonding_type,
            "bulk_modulus_reuss": self.bulk_modulus_reuss,
            "bulk_modulus_voigt": self.bulk_modulus_voigt,
            "bulk_modulus_voigt_reuss_hill": self.bulk_modulus_voigt_reuss_hill,
            "bulk_shear_ratio_reuss": self.bulk_shear_ratio_reuss,
            "bulk_shear_ratio_voigt": self.bulk_shear_ratio_voigt,
            "bulk_shear_ratio_voigt_reuss_hill": self.bulk_shear_ratio_voigt_reuss_hill,
            "cauchy_pressure": self.cauchy_pressure,
            "compaliance_tensor": self.compaliance_tensor.tolist(),
            "crystal_type": self.crystal_type,
            "debye_temperature": self.debye_temperature,
            "ductility": self.ductility,
            "elastic_stability": self.elastic_stability,
            "elastic_tensor": self.elastic_tensor.tolist(),
            "hardness": self.hardness,
            "kleinman_parameter": self.kleinman_parameter,
            "lambda_lame_coefficient": self.lambda_lame_coefficient,
            "melting_temperature": self.melting_temperature,
            "mu_lame_coefficient": self.mu_lame_coefficient,
            "p_wave_modulus_reuss": self.p_wave_modulus_reuss,
            "p_wave_modulus_voigt": self.p_wave_modulus_voigt,
            "p_wave_modulus_voigt_reuss_hill": self.p_wave_modulus_voigt_reuss_hill,
            "poissons_ratio_reuss": self.poissons_ratio_reuss,
            "poissons_ratio_voigt": self.poissons_ratio_voigt,
            "poissons_ratio_voigt_reuss_hill": self.poissons_ratio_voigt_reuss_hill,
            "pugh_ratio_reuss": self.pugh_ratio_reuss,
            "pugh_ratio_voigt": self.pugh_ratio_voigt,
            "pugh_ratio_voigt_reuss_hill": self.pugh_ratio_voigt_reuss_hill,
            "shear_modulus_reuss": self.shear_modulus_reuss,
            "shear_modulus_voight": self.shear_modulus_voight,
            "shear_modulus_voight_reuss_hill": self.shear_modulus_voight_reuss_hill,
            "structure": self.structure.to_dict(symprec),
            "velocity_average": self.velocity_average,
            "velocity_logitudinal": self.velocity_logitudinal,
            "velocity_transverse": self.velocity_transverse,
            "youngs_modulus_reuss": self.youngs_modulus_reuss,
            "youngs_modulus_voigt": self.youngs_modulus_voigt,
            "youngs_modulus_voigt_reuss_hill": self.youngs_modulus_voigt_reuss_hill,
        }
        
    def to_file(self, outfile="elastic_properties.txt"):
        """


        Parameters
        ----------
        outfile : str, optional
            Path to the output file. The default is "elastic_properties.txt".

        Returns
        -------
        None.

        """
        wf = open(outfile, "w")
        wf.write(self.__str__())
        wf.close()

    def __str__(self):
        """


        Returns
        -------
        None.

        """

        ret = ""
        ret += "\n------------------------------------------------------------------\n"
        ret += "Elastic Moduli\n"
        ret += "------------------------------------------------------------------\n\n"

        ret += "                              Voigt     Reuss    Average\n"
        ret += "-------------------------------------------------------\n"
        ret += "Bulk modulus   (GPa)       %9.3f %9.3f %9.3f \n" % (
            self.B_V,
            self.B_R,
            self.B_VRH,
        )
        ret += "Shear modulus  (GPa)       %9.3f %9.3f %9.3f \n" % (
            self.G_V,
            self.G_R,
            self.G_VRH,
        )
        ret += "Young's modulus  (GPa)      %9.3f %9.3f %9.3f \n" % (
            self.E_V,
            self.E_R,
            self.E_VRH,
        )
        ret += "Poisson's ratio           %9.3f %9.3f %9.3f \n" % (
            self.P_V,
            self.P_R,
            self.P_VRH,
        )
        ret += "Bulk/Shear (Pugh's) ratio %9.3f %9.3f %9.3f  \n" % (
            self.BG_Ratio_V,
            self.BG_Ratio_R,
            self.BG_Ratio_VRH,
        )

        ret += "\n------------------------------------------------------------------\n"
        ret += "Elastic Anisotropy\n"
        ret += "------------------------------------------------------------------\n\n"

        ret += (
            "Zener's anisotropy (true for cubic crystals only); Az = %6.3f; Ref.[6]\n"
            % self.A_z
        )
        ret += (
            "Chung-Buessem's anisotropy (true for cubic crystals only); Acb = %6.3f; Ref.[7]\n"
            % self.A_cb
        )
        ret += "Universal anisotropy index; Au = %6.3f; Ref.[8]\n" % self.A_u
        ret += "Log-Euclidean's anisotropy; AL = %6.3f; Ref.[9]\n" % self.A_l

        ret += "\n------------------------------------------------------------------\n"
        ret += "Elastic Wave Velocities and Debye Temperature\n"
        ret += "------------------------------------------------------------------\n\n"

        ret += (
            "Longitudinal wave velocity (vl) : %10.3f m/s; Ref.[10]\n"
            % self.velocity_logitudinal
        )
        ret += (
            "Transverse wave velocity (vt) : %10.3f m/s; Ref.[10]\n"
            % self.velocity_transverse
        )
        ret += (
            "Average wave velocity (vm) : %10.3f m/s; Ref.[10]\n"
            % self.velocity_average
        )
        ret += "Debye temperature  : %10.3f K; Ref.[10]\n" % self.debye_temperature
        ret += "\n"
        ret += "WARNING: The  Debye model for the atomic displacement is based on a monoatomic crystal approximation.\n"
        ret += "Here we consider an averaged mass, in case your crystal has several species.\n"

        ret += "\n------------------------------------------------------------------\n"
        ret += "Melting Temperature\n"
        ret += "------------------------------------------------------------------\n\n"

        ret += "Melting temperature calculated from the empirical relation: Tm = 607 + 9.3*Kvrh \pm 555 (in K); Ref.[11]"
        ret += "Tm =  %10.3f K (plus-minus 555 K) \n" % self.melting_temperature
        ret += "\n\n"
        ret += "WARNING: This is a crude approximation and its validity needs to be checked! \n"

        return ret
    
    

#GdAlO3              
Gd=np.array([ [387.7,  125.1,  148.5,  0.0,  0.0, 0.0],
              [125.1,  297.1,  125.7,  0.0,  0.0, 0.0],
              [148.5,  125.7,  349.4,  0.0,  0.0, 0.0],
              [0.0,  0.0,  0.0,  158.6,  0.0, 0.0],
              [0.0,  0.0,  0.0,  0.0,  139.7, 0.0],
              [0.0,  0.0,  0.0,  0.0,  0.0, 104.7]])
             
import pymatgen as mp
from pymatgen.core.periodic_table import Element
#from pymatgen.io.vasp.inputs import Poscar

mat=Gd
print ('Gd')  
g = mp.core.Structure.from_file('Gd.vasp')

numAtoms=g.num_sites
ele=g.species
species=np.array([s.symbol for s in ele])
atomMass=np.array([Element(s).atomic_mass for s in ele])*1.66053907e-27  #amu to kg
volume=g.volume*1.0e-30

density=sum(atomMass)/(volume)
lattice=g.lattice.matrix
positions=g.frac_coords

elas_poly = Elastic_polycrystal(elastic_tensor=mat, density=density, masses=atomMass)
        
elas_poly.to_file('Gd_poly')













        
        
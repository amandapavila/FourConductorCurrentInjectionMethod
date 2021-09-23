"""
    Análise de Redes Elétricas I - Prof. João Alberto Passos Filho.

    "Three-Phase Power Flow Based on Four-Conductor Current Injection Method for Unbalanced Distribution Networks"
    Debora Rosana Ribeiro Penido, Student Member, IEEE, Leandro Ramos de Araujo, Sandoval Carneiro, Jr., Senior Member,
    IEEE, Jose Luiz Rezende Pereira, Senior Member, IEEE, and Paulo Augusto Nepomuceno Garcia, Member, IEEE

    Application in the system 4 Node Test Feeder.

    Configuration: Caso considerando três fases e condutor neutro (com representação explícita) nas linhas (3F+N), com
transformador abaixador YatYat (estrela aterrada/estrela aterrada), 12,4/4,16 kV, com todos os neutros solidamente
aterrados (Zat = 0), inclusive o da barra de geração (Zg,at = 0).

    Authors: Amanda Pávila Silva
             Carla Farage Cavalari
             Marina Martins Mattos
             Pedro Paulo Surerus Sarmento
"""

import numpy as np


class Ybus:
    """
    Calculation of Ybus matrix per phase
    """
    def __init__(self, dbar, dlin):
        self.dbar = dbar
        self.dlin = dlin

    def calc(self):
        # Conductors
        conductors = ['A', 'B', 'C', 'N']
        # Number of conductors (3 phases + neutral)
        n_cond = len(conductors)

        # Number of bars
        n_bars = self.dbar['num'].max()

        # Matrix dimension
        dim = n_cond * n_bars
        # Nodal admittance matrix
        ybus = np.zeros([dim, dim], dtype='complex')

        for idx, value in self.dlin.iterrows():
            # Index
            k = value['de']
            k -= 1
            m = value['para']
            m -= 1

            if value['y'] is not None:
                ykm = value['y']
            else:
                # Nodal impedance matrix between bars k and m
                zkm = value['z']

                # Nodal admittance matrix between bars k and m
                ykm = np.linalg.inv(zkm)

            # Diagonal sub-matrices
            # kk Element
            start, end = k * n_cond, value['de'] * n_cond
            ybus[start:end, start:end] += ykm

            # mm Element
            start, end = m * n_cond, value['para'] * n_cond
            ybus[start:end, start:end] += ykm

            # Off-diagonal sub-matrices
            # km Element
            startl, endl = k * n_cond, value['de'] * n_cond
            startc, endc = m * n_cond, value['para'] * n_cond
            ybus[startl:endl, startc:endc] -= ykm

            # mk Element
            ybus[startc:endc, startl:endl] -= ykm

        self.ybus = ybus

        return self.ybus

    def get_gbus(self):
        gbus = self.ybus.real
        return gbus

    def get_bbus(self):
        bbus = self.ybus.imag
        return bbus

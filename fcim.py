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
import cmath
import pandas as pd


class Fcim:
    """
    Three-Phase Power Flow Based on Four-Conductor Current Injection Method for Unbalanced Distribution Networks
    """

    def __init__(self, dbar, ybus, neut_ground=True):
        # Iteration counter
        self.h = 0
        self.h_max = 10

        # Tolerance
        self.e = 1e-6

        # Bar data
        self.dbar = dbar

        # Nodal admittance matrix
        self.ybus = ybus
        self.gbus = ybus.real
        self.bbus = ybus.imag

        # Conductors
        self.conductors = ['A', 'B', 'C', 'N']
        # Number of conductors (3 phases + neutral)
        self.n_cond = len(self.conductors)

        # Parameter for considering all grounded neutrals
        self.neut_ground = neut_ground

    def calc_res(self):
        """
        Method for calculating residuals
        :return: DeltaI
        """

        # Number of bars
        bars = [1, 2, 3, 4]
        n_bars = self.dbar['num'].max()
        # Type
        bars_type = [2, 0, 0, 0]

        # The current injection mismatches
        DeltaI = np.zeros([n_bars * 2 * self.n_cond], dtype='float')

        for b, bar in enumerate(bars):

            if bars_type[b] != 0:
                pass
            # PQ bars
            else:

                Ir_esp = np.zeros(self.n_cond, dtype='float')
                Im_esp = np.zeros(self.n_cond, dtype='float')

                data_per_bar = self.dbar.query(f'num == {bar}')
                data_per_bar = data_per_bar.reset_index(drop=True)

                for ph, value in data_per_bar.iterrows():
                    if value['phase'] != 'N':
                        Pkd = value['Pg'] - value['Pl']
                        Qkd = value['Qg'] - value['Ql']

                        Vkd_r = value['vr']
                        Vkd_m = value['vm']

                        idx = data_per_bar.last_valid_index()
                        Vkn_r = data_per_bar['vr'][idx]
                        Vkn_m = data_per_bar['vm'][idx]

                        # Specified Values
                        Ir_esp[ph] = (Pkd * (Vkd_r - Vkn_r) + Qkd * (Vkd_m - Vkn_m)) / \
                                     (np.power((Vkd_r - Vkn_r), 2) + np.power((Vkd_m - Vkn_m), 2))

                        Im_esp[ph] = (Pkd * (Vkd_m - Vkn_m) - Qkd * (Vkd_r - Vkn_r)) / \
                                     (np.power((Vkd_r - Vkn_r), 2) + np.power((Vkd_m - Vkn_m), 2))
                    else:
                        Ir_esp[ph] = -np.sum(Ir_esp[:-1])
                        Im_esp[ph] = -np.sum(Im_esp[:-1])

                Ir_cal = np.zeros(self.n_cond, dtype='float')
                Im_cal = np.zeros(self.n_cond, dtype='float')

                # First term
                for d, cond in enumerate(self.conductors):
                    for t, phase in data_per_bar.iterrows():
                        line = b * self.n_cond + d
                        column = b * self.n_cond + t

                        Gkk_dt = self.gbus[line, column]
                        Bkk_dt = self.bbus[line, column]

                        Vkt_r = phase['vr']
                        Vkt_m = phase['vm']

                        Ir_cal[d] -= (Gkk_dt * Vkt_r - Bkk_dt * Vkt_m)
                        Im_cal[d] -= (Bkk_dt * Vkt_r + Gkk_dt * Vkt_m)

                # Second term
                for d, cond in enumerate(self.conductors):
                    for bi, bari in enumerate(bars):
                        if bari != bar:
                            data_per_bar = self.dbar.query(f'num == {bari}')
                            data_per_bar = data_per_bar.reset_index(drop=True)

                            for t, phase in data_per_bar.iterrows():
                                line = b * self.n_cond + d
                                column = bi * self.n_cond + t

                                Gki_dt = self.gbus[line, column]
                                Bki_dt = self.bbus[line, column]

                                Vit_r = phase['vr']
                                Vit_m = phase['vm']

                                Ir_cal[d] -= (Gki_dt * Vit_r - Bki_dt * Vit_m)
                                Im_cal[d] -= (Bki_dt * Vit_r + Gki_dt * Vit_m)

                        else:
                            pass

                start = b * 2 * self.n_cond
                end = start + self.n_cond
                DeltaI[start:end] += Im_esp
                DeltaI[start:end] += Im_cal

                start = end
                end = start + self.n_cond
                DeltaI[start:end] += Ir_esp
                DeltaI[start:end] += Ir_cal

        if self.neut_ground:
            # Resets neutral current residue if all neutrals are solidly grounded
            for n in range(self.n_cond * 2):
                idx_n = n * self.n_cond + 3
                DeltaI[idx_n] = 0.
            else:
                pass

        return DeltaI

    def jacobian(self) -> object:
        """
        Method for calculating the Jacobian matrix
        :return: mat_jacobian
        """

        # Define big number
        big_num = 1e20

        # Number of bars
        bars = [1, 2, 3, 4]
        bars_type = [2, 0, 0, 0]
        n_bars = self.dbar['num'].max()
        dim = n_bars * 2 * self.n_cond
        mat_jacobian = np.zeros([dim, dim], dtype='float')

        # Dimension of each block that makes up the Jacobian matrix
        size = 2 * self.n_cond

        for k, bark in enumerate(bars):

            for i, bari in enumerate(bars):

                # The off-diagonal terms and the first diagonal term are identical and correspond to the elements of the
                # nodal admittance matrix.

                Bki = self.bbus[self.n_cond * k:self.n_cond * (k + 1), self.n_cond * i:self.n_cond * (i + 1)]
                Gki = self.gbus[self.n_cond * k:self.n_cond * (k + 1), self.n_cond * i:self.n_cond * (i + 1)]

                Jki = np.concatenate((np.concatenate((Bki, Gki), axis=1), np.concatenate((Gki, -Bki), axis=1)),
                                     axis=0)

                # Jki elements
                if bari != bark:
                    mat_jacobian[size * k:size * (k + 1), size * i:size * (i + 1)] = Jki

                # Jkk elements
                else:
                    # Add big number on the diagonal of non-PQ bars
                    if bars_type[k] != 0:
                        mat_jacobian[size * k:size * (k + 1), size * k:size * (k + 1)] = big_num * np.identity(size)

                    # Calculates second term of Jkk elements
                    else:
                        # First term of element kk
                        Jkk_ft = Jki

                        e = self.e_mat(bar=bark)
                        f = self.f_mat(bar=bark)
                        g = self.g_mat(bar=bark)
                        h = self.h_mat(bar=bark)

                        # Second term of element kk
                        Jkk_st = np.concatenate((np.concatenate((e, f), axis=1), np.concatenate((g, h), axis=1)),
                                                axis=0)

                        Jkk = Jkk_ft + Jkk_st

                        mat_jacobian[size * k:size * (k + 1), size * k:size * (k + 1)] = Jkk

        if self.neut_ground:
            # Adds the big number in the diagonal element of neutral current residuals if all neutrals are solidly
            # grounded
            for n in range(self.n_cond * 2):
                idx_n = n * self.n_cond + 3
                mat_jacobian[idx_n, idx_n] = big_num
            else:
                pass

        return mat_jacobian

    def e_mat(self, bar):
        """
        Calculate matrix E
        :return: matrix e
        """
        e = np.zeros([self.n_cond, self.n_cond], dtype='float')

        data_per_bar = self.dbar.query(f'num == {bar}')
        data_per_bar = data_per_bar.reset_index(drop=True)

        # Neutral conductor index
        n = 3

        for idx, value in data_per_bar.iterrows():

            if value['phase'] != 'N':
                Pkd = value['Pg'] - value['Pl']
                Qkd = value['Qg'] - value['Ql']

                Vkd_r = value['vr']
                Vkd_m = value['vm']

                # Neutral
                idn = data_per_bar.last_valid_index()
                Vkn_r = data_per_bar['vr'][idn]
                Vkn_m = data_per_bar['vm'][idn]

                ekd = (Qkd * (np.power((Vkd_r - Vkn_r), 2)) - Qkd * (np.power((Vkd_m - Vkn_m), 2)) -
                       2 * Pkd * (Vkd_r - Vkn_r) * (Vkd_m - Vkn_m)) / \
                      (np.power((np.power((Vkd_r - Vkn_r), 2) + np.power((Vkd_m - Vkn_m), 2)), 2))

                e[idx, idx] -= ekd
                e[idx, n] += ekd
                e[n, idx] += ekd
                e[n, n] -= ekd

            else:
                pass

        return e

    def f_mat(self, bar):
        """
        Calculate matrix F
        :return: matrix f
        """
        f = np.zeros([self.n_cond, self.n_cond], dtype='float')

        data_per_bar = self.dbar.query(f'num == {bar}')
        data_per_bar = data_per_bar.reset_index(drop=True)

        # Neutral conductor index
        n = 3

        for idx, value in data_per_bar.iterrows():

            if value['phase'] != 'N':
                Pkd = value['Pg'] - value['Pl']
                Qkd = value['Qg'] - value['Ql']

                Vkd_r = value['vr']
                Vkd_m = value['vm']

                # Neutral
                idn = data_per_bar.last_valid_index()
                Vkn_r = data_per_bar['vr'][idn]
                Vkn_m = data_per_bar['vm'][idn]

                fkd = (Pkd * (np.power((Vkd_r - Vkn_r), 2)) - Pkd * (np.power((Vkd_m - Vkn_m), 2)) +
                       2 * Qkd * (Vkd_r - Vkn_r) * (Vkd_m - Vkn_m)) / \
                      (np.power((np.power((Vkd_r - Vkn_r), 2) + np.power((Vkd_m - Vkn_m), 2)), 2))

                f[idx, idx] -= fkd
                f[idx, n] += fkd
                f[n, idx] += fkd
                f[n, n] -= fkd

            else:
                pass

        return f

    def g_mat(self, bar):
        """
        Calculate matrix G
        :return: matrix g
        """
        g = np.zeros([self.n_cond, self.n_cond], dtype='float')

        data_per_bar = self.dbar.query(f'num == {bar}')
        data_per_bar = data_per_bar.reset_index(drop=True)

        # Neutral conductor index
        n = 3

        for idx, value in data_per_bar.iterrows():

            if value['phase'] != 'N':
                Pkd = value['Pg'] - value['Pl']
                Qkd = value['Qg'] - value['Ql']

                Vkd_r = value['vr']
                Vkd_m = value['vm']

                # Neutral
                idn = data_per_bar.last_valid_index()
                Vkn_r = data_per_bar['vr'][idn]
                Vkn_m = data_per_bar['vm'][idn]

                gkd = (-Pkd * (np.power((Vkd_r - Vkn_r), 2)) + Pkd * (np.power((Vkd_m - Vkn_m), 2)) -
                       2 * Qkd * (Vkd_r - Vkn_r) * (Vkd_m - Vkn_m)) / \
                      (np.power((np.power((Vkd_r - Vkn_r), 2) + np.power((Vkd_m - Vkn_m), 2)), 2))

                g[idx, idx] -= gkd
                g[idx, n] += gkd
                g[n, idx] += gkd
                g[n, n] -= gkd

            else:
                pass

        return g

    def h_mat(self, bar):
        """
        Calculate matrix H
        :return: matrix h
        """
        h = np.zeros([self.n_cond, self.n_cond], dtype='float')

        data_per_bar = self.dbar.query(f'num == {bar}')
        data_per_bar = data_per_bar.reset_index(drop=True)

        # Neutral conductor index
        n = 3

        for idx, value in data_per_bar.iterrows():

            if value['phase'] != 'N':
                Pkd = value['Pg'] - value['Pl']
                Qkd = value['Qg'] - value['Ql']

                Vkd_r = value['vr']
                Vkd_m = value['vm']

                # Neutral
                idn = data_per_bar.last_valid_index()
                Vkn_r = data_per_bar['vr'][idn]
                Vkn_m = data_per_bar['vm'][idn]

                hkd = (Qkd * (np.power((Vkd_r - Vkn_r), 2)) - Qkd * (np.power((Vkd_m - Vkn_m), 2)) -
                       2 * Pkd * (Vkd_r - Vkn_r) * (Vkd_m - Vkn_m)) / \
                      (np.power((np.power((Vkd_r - Vkn_r), 2) + np.power((Vkd_m - Vkn_m), 2)), 2))

                h[idx, idx] -= hkd
                h[idx, n] += hkd
                h[n, idx] += hkd
                h[n, n] -= hkd

            else:
                pass

        return h

    def update_sv(self, DeltaV):
        """
        Updates state variables
        :param DeltaV: Residual vector of state variables
        :return:
        """

        # Number of bars
        bars = [1, 2, 3, 4]
        n_bars = self.dbar['num'].max()

        # Vector dimension referring to each bar
        size = 2 * self.n_cond

        # Parameter for dataframe index
        idx = 0

        for b, bar in enumerate(bars):

            start = b * size
            end = bar * size
            DeltaVb = DeltaV[start:end]

            DeltaVbr = DeltaVb[:self.n_cond]
            DeltaVbm = DeltaVb[self.n_cond:]

            for ph, phase in enumerate(self.conductors):

                self.dbar.loc[idx, 'vr'] += DeltaVbr[ph]
                self.dbar.loc[idx, 'vm'] += DeltaVbm[ph]
                idx += 1

    def newton_raphson(self):
        """
        Newton-Raphson method
        :return:
        """

        DeltaI = self.calc_res()

        while np.max(np.abs(DeltaI)) > self.e:
            # Increase iteration counter
            self.h += 1

            # Calculate the Jacobian Matrix
            jacobian = self.jacobian()

            # Solve the system of equations
            DeltaV = np.linalg.solve(jacobian, DeltaI)

            # Updates state variables
            self.update_sv(DeltaV=DeltaV)

            # Recalculate DeltaI
            DeltaI = self.calc_res()

            if self.h > self.h_max:
                break

        print(f'Número total de iterações do método de Newton-Raphson: {self.h}!\n')

        for idx, value in self.dbar.iterrows():

            volt = complex(real=value['vr'], imag=value['vm'])
            mod = cmath.polar(volt)[0]
            phase = cmath.polar(volt)[1]

            self.dbar.loc[idx, 'v'] = mod
            self.dbar.loc[idx, 'theta'] = np.rad2deg(phase)

        print(self.dbar)

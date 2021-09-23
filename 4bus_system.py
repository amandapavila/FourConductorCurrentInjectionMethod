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
import pandas as pd
from ybus import Ybus
from fcim import Fcim

# BAR DATA CONFIGURATION
# Base power of the system (three-phase), in MVA
sbase_t = 6.

# Base power of the system (single-phase), in MVA
sbase = sbase_t / 3.

# Number of areas
n_are = 2

# Area information
# code: 1 (primary) and 2 (secondary)
# vbase: Voltage base, in kV
# sbase: Power base, in MVA
# zbase: Impedance base, in ohm
area_info = {
    'code': [1, 2],
    'vbase': [12.47, 4.16],
    'sbase_t': [sbase_t, sbase_t],
    'sbase': [sbase, sbase],
    'zbase': []
}

for are in range(n_are):
    zb = area_info['vbase'][are] ** 2 / area_info['sbase_t'][are]
    area_info['zbase'].append(zb)


# Calculation of reactive power through active power and power factor
def calc_q(p, fp):
    angle = np.arccos(fp)
    return p * np.tan(angle)


# Conductors
conductors = ['A', 'B', 'C', 'N']
# Number of conductors (3 phases + neutral)
n_cond = len(conductors)

# BAR IDENTIFICATION
# Number
bars = [1, 2, 3, 4]
n_bars = max(bars)
# Type
bars_type = [2, 0, 0, 0]
# Active and reactive power generated
bars_Pg = dict()
bars_Qg = dict()

# Active and reactive power demanded
bars_Pl = dict()
bars_Ql = dict()

for phase in conductors:
    bars_Pg[f'{phase}'] = np.zeros([n_bars])
    bars_Qg[f'{phase}'] = np.zeros([n_bars])

    bars_Pl[f'{phase}'] = np.zeros([n_bars])
    bars_Ql[f'{phase}'] = np.zeros([n_bars])

# LOAD IDENTIFICATION
# Add as many charges as needed!
# bar: Bar number where the load is connected
# balanced: Balanced load data (active_power: active power, in kW, and power_factor: power factor)
# unbalanced: Unbalanced load data per phase (active_power: active power, in kW, and power_factor: power factor)
load = {
    'bar': 4,
    'balanced': {
        'active_power': 1800.,
        'power_factor': 0.9
    },
    'unbalanced': {
        'active_power': [1275., 1800., 2375.],
        'power_factor': [.85, .9, .95]
    }
}

loads = [load]
# Parameter to consider load balanced. If true, consider
balanced = False

for l in loads:
    # Bar index
    index_bar = l['bar'] - 1

    if balanced:
        # kW to MW conversion
        pl = l['balanced']['active_power'] * 1e-3
        pl /= sbase
        ql = calc_q(p=pl, fp=l['balanced']['power_factor'])

        for ph, cond in enumerate(conductors):
            if cond == 'N':
                pass
            else:
                bars_Pl[f'{cond}'][index_bar] += pl
                bars_Ql[f'{cond}'][index_bar] += ql

    else:
        pl = np.array(l['unbalanced']['active_power'])
        pl *= 1e-3
        pl /= sbase

        for ph, cond in enumerate(conductors):
            if cond == 'N':
                pass
            else:
                ql = calc_q(p=pl[ph], fp=l['unbalanced']['power_factor'][ph])
                bars_Pl[f'{cond}'][index_bar] += pl[ph]
                bars_Ql[f'{cond}'][index_bar] += ql

dbar = dict()
dbar['num'] = list()
dbar['type'] = list()
dbar['phase'] = list()
dbar['v'] = list()
dbar['theta'] = list()
dbar['vr'] = list()
dbar['vm'] = list()
dbar['Pg'] = list()
dbar['Qg'] = list()
dbar['Pl'] = list()
dbar['Ql'] = list()

# Flat start initialization
V, Theta = [1., 1., 1., 0.], [0., -120., 120., 0.]

# Bar data adjustment
for b, bar in enumerate(bars):
    for ph, cond in enumerate(conductors):
        dbar['num'].append(bar)
        dbar['type'].append(bars_type[b])
        dbar['phase'].append(cond)
        dbar['v'].append(V[ph])
        dbar['theta'].append(Theta[ph])
        dbar['vr'].append(V[ph] * np.cos(np.deg2rad(Theta[ph])))
        dbar['vm'].append(V[ph] * np.sin(np.deg2rad(Theta[ph])))
        dbar['Pg'].append(bars_Pg[f'{cond}'][b])
        dbar['Qg'].append(bars_Qg[f'{cond}'][b])
        dbar['Pl'].append(bars_Pl[f'{cond}'][b])
        dbar['Ql'].append(bars_Ql[f'{cond}'][b])

dbar = pd.DataFrame(dbar)

# LINE DATA CONFIGURATION
# 4 WIRE CONFIGURATION
# Line resistance matrix, in ohm/mile
Rmatriz = np.array([[.401299, .095299, .095299, .095299],
                    [.095299, .401299, .095299, .095299],
                    [.095299, .095299, .401299, .095299],
                    [.095299, .095299, .095299, .687299]])

# Line reactance matrix, in ohm/mile
Xmatriz = np.array([[1.41319j, .851454j, .726521j, .752372j],
                    [.851454j, 1.41319j, .780133j, .786442j],
                    [.726521j, .780133j, 1.41319j, .767348j],
                    [.752372j, .786442j, .767348j, 1.54639j]], dtype='complex')

# Line impedance matrix, in ohm/mile
z = Rmatriz + Xmatriz


# Converts ft to miles:
def ft2mile(length):
    factor = 1 / 5280
    return length * factor


# Line information
# code: Line number
# bar_de: Bar de
# bar_para: Bar para
# length: Length of line
# z: Line impedance matrix, in ohm
line_info = {
    'code': [1, 2],
    'bar_de': [1, 3],
    'bar_para': [2, 4],
    'length': [2000., 2500.],
    'z': []
}

# Number of lines
num_lines = max(line_info['code'])

for line in range(num_lines):
    line_info['z'].append(z * ft2mile(length=line_info['length'][line]) / area_info['zbase'][line])

# TRANSFORMER DATA CONFIGURATION Yat - Yat
# Transformer information
# connection: Connection type (step-down or step-up)
# sbase: Transformer base power, in MVA
# vp: Primary base voltage
# vs: Secondary base voltage
# r: Resistance, in % (transformer base)
# x: Reactance, in % (transformer base)
trafo_info = {
    'connection': ['sd', 'su'],
    'sbase': [6., 6.],
    'vp': [12.47, 12.47],
    'vs': [4.16, 24.9],
    'r': [1., 1.],
    'x': [6., 6.]
}

# Connection number, 0: 'sd' and 1: 'su'
conn = 0

# Transformer impedance at transformer base
z_tr = complex(real=trafo_info['r'][conn] * 1e-2, imag=trafo_info['x'][conn] * 1e-2)

# Base change
z_tr *= (sbase_t / trafo_info['sbase'][conn])

# Transformation into a matrix
y_trafo = np.zeros([n_cond, n_cond], dtype='complex')

for ph, cond in enumerate(conductors):
    if cond != 'N':
        y_trafo[ph, ph] += 1 / z_tr
    else:
        pass

dlin = dict()
dlin['de'] = [1, 2, 3]
dlin['para'] = [2, 3, 4]
dlin['z'] = [line_info['z'][0], None, line_info['z'][1]]
dlin['y'] = [None, y_trafo, None]

dlin = pd.DataFrame(dlin)

ybus = Ybus(dbar, dlin).calc()

Fcim(dbar, ybus).newton_raphson()

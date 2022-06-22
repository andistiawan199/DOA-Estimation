import streamlit as st
import math
import sys
import json
from pyargus.directionEstimation import *
from scipy.spatial import distance
from collections import deque
import numpy as np


# N_SAMPLES_OF_REF_PERIOD = 8
NUMBER_MESSAGES = 1
SIGMA_FILTER_WINDOW = 5

# map parameters
# width_map = 14.7
# height_map = 12.4
# kx_locator = 0.532471
# ky_locator = 0.558504
# x_locator = kx_locator * width_map
# y_locator = ky_locator * height_map
# z_locator = 0.2
# z_beacon = 0.814

# # antenna array parameters
# frequency = 2480
# wavelength = 0.12
# d = 0.05  # inter element spacing
# M = 2  # number of antenna elements in the antenna system


url = "https://navigine.com/blog/using-angle-of-arrival-for-direction-finding-with-bluetooth/"
st.write("""
# Direction of Arival Estimation Algorithm
This demo project shows how using raw IQ samples obtained from the Minew Aoa Locator, you can get the azimuth and elevation angles at which the beacon is located. Using these angles, we also get the position of the beacon in XY coordinates (knowing the location of the locator in the same coordinate system). This project is based on [Navigine AoA article](%s) in which we determined the azimuth angle using 2 adjacent elements of the linear antenna array of the locator.
"""%url)

with st.sidebar:
    st.write("# Configuration")
    N_SAMPLES_OF_REF_PERIOD = st.number_input("Sample Reference Period",8)

    width_map = st.number_input("Width Map",14.7)
    height_map = st.number_input("Height Map",12.4)
    kx_locator = st.number_input("KX Locator",0.532471)
    ky_locator = st.number_input("KY Locator",0.558504)
    x_locator = kx_locator * width_map
    y_locator = ky_locator * height_map
    z_locator = st.number_input("Z Locator",0.2)
    z_beacon = st.number_input("Z Beacon",0.814)

    frequency = st.number_input("Enter Raw IQ heres",2480)
    wavelength = st.number_input("Enter Raw IQ heres",0.12)
    d = st.number_input("Enter Raw IQ heres",0.05)
    M = st.number_input("Enter Raw IQ heres",2)

def to_plus_minus_pi(angle):
    while angle >= 180:
        angle -= 2 * 180
    while angle < -180:
        angle += 2 * 180
    return angle


def get_angle(X):
    # Estimating the spatial correlation matrix
    R = corr_matrix_estimate(X.T, imp="fast")

    array_alignment = np.arange(0, M, 1) * d
    incident_angles = np.arange(-90, 91, 1)
    scanning_vectors = np.zeros((M, np.size(incident_angles)), dtype=complex)
    for i in range(np.size(incident_angles)):
        scanning_vectors[:, i] = np.exp(array_alignment * 1j * 2 * np.pi * np.sin(np.radians(incident_angles[i])) / wavelength)  # scanning vector

    ula_scanning_vectors = scanning_vectors

    # Estimate DOA
    MUSIC = DOA_MUSIC(R, ula_scanning_vectors, signal_dimension=1)
    norm_data = np.divide(np.abs(MUSIC), np.max(np.abs(MUSIC)))
    return float(incident_angles[np.where(norm_data == 1)[0]][0])


def get_coordinate(azimuth, elevation, height, receiver_coords):
    nx = np.cos(np.deg2rad(90.0 - azimuth))
    nz = np.cos(np.deg2rad(90.0 - abs(elevation)))
    if math.isclose(nx, 0.0, abs_tol=1e-16) or math.isclose(nz, 0.0, abs_tol=1e-16):
        return [float("nan"),  float("nan")]
    else:
        ny = np.sqrt(1 - nx ** 2 - nz ** 2)
        t = (height - receiver_coords[2]) / nz
        x = receiver_coords[0] + t * nx
        y = receiver_coords[1] - t * ny
    return [x, y]


def calculate_data(iq_data):
    x_00, azimuth_x_12, elevation_x_12 = [], [], []
    azimuth_phases, elevation_phases = [], []
    ref_phases = []
    iq = list(map(int,iq_data.split(",")));
    iq_samples = [iq[n:n + 2] for n in range(0, len(iq), 2)]
    for iq_idx in range(N_SAMPLES_OF_REF_PERIOD - 1):
        iq_next = complex(iq_samples[iq_idx + 1][0], iq_samples[iq_idx + 1][1])
        iq_cur = complex(iq_samples[iq_idx][0], iq_samples[iq_idx][1])
        phase_next = np.rad2deg(np.arctan2(iq_next.imag, iq_next.real))
        phase_cur = np.rad2deg(np.arctan2(iq_cur.imag, iq_cur.real))
        ref_phases.append((to_plus_minus_pi(phase_next - phase_cur)))
    phase_ref = np.mean(ref_phases)

    iq_2ant_batches = [iq_samples[n:n + 2] for n in range(N_SAMPLES_OF_REF_PERIOD, len(iq_samples), 2)]
    for iq_batch_idx, iq_batch in enumerate(iq_2ant_batches[:-1]):
        iq_next = complex(iq_batch[1][0], iq_batch[1][1])
        iq_cur = complex(iq_batch[0][0], iq_batch[0][1])
        phase_next = np.rad2deg(np.arctan2(iq_next.imag, iq_next.real))
        phase_cur = np.rad2deg(np.arctan2(iq_cur.imag, iq_cur.real))
        diff_phase = to_plus_minus_pi((phase_next - phase_cur) - 2 * phase_ref)
        if iq_batch_idx % 2 != 0:
            elevation_phases.append(diff_phase)
        else:
            x_00.append(1)
            azimuth_phases.append(diff_phase)

    # MUSIC algo
    X = np.zeros((M, np.size(x_00)), dtype=complex)
    X[0, :] = x_00
    for i in azimuth_phases:
        azimuth_x_12.append(np.exp(1j * np.deg2rad(i)))
    X[1, :] = azimuth_x_12
    azimuth_angle = get_angle(X)

    for i in elevation_phases:
        elevation_x_12.append(np.exp(1j * np.deg2rad(i)))
    X[1, :] = elevation_x_12
    elevation_angle = get_angle(X)
    

    xy = get_coordinate(azimuth_angle, elevation_angle, z_beacon, [x_locator, y_locator, z_locator])
    if not math.isnan(xy[0]) and not math.isnan(xy[1]):
        print(f'azimuth_angle:{azimuth_angle}, elevation_angle:{elevation_angle}')
        print(f'x_beacon:{xy[0]}, y_beacon:{xy[1]}')
        # return azimuth_angle
        st.write("Azimuth Angle: ",azimuth_angle)
        st.write("Elevation Angle: ",elevation_angle)
        st.write("X: ",xy[0])
        st.write("Y: ",xy[1])

with st.form(key='my_form'):
    iq = st.text_input("Enter Raw IQ here",key="1")
    # calculate = st.form_submit_button("Calculate")
    col1, col2, col3 = st.columns([0.9,0.9,5])
    with col1:
        calculate = st.form_submit_button("Calculate")
    with col2:
        clear = st.form_submit_button("Clear")    
    if calculate:
        if(iq==""):
            st.error("Raw IQ cannot empty")
        else:
            try:
                calculate_data(iq)
                # st.write(tiq)
            except BaseException as e:
                st.error(e)
    if clear:
        st.empty()

        



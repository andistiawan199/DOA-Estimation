import streamlit as st
import math
import sys
import json
from pyargus.directionEstimation import *
from scipy.spatial import distance
from collections import deque
import numpy as np
import pandas as pd

NUMBER_MESSAGES = 1
SIGMA_FILTER_WINDOW = 5

st.set_page_config(layout="wide")
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
    wavelength = st.number_input("Wavelength",0.12)
    d = st.number_input("d",0.05)
    M = st.number_input("M",2)

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

    
    st.write("Perhitungan: ")
    st.latex('nx = \cos{(90.0 - %a ) * \Pi / 180}'% (azimuth))
    st.latex('nz = \cos{(90.0 - abs(%a)) * \Pi / 180}'% (elevation))
    st.latex('ny = \sqrt{\smash[b]{1 - %a^2 - %a^2}}'% (nx,nz))
    st.latex('t = (%a - %a) / %a'% (height,receiver_coords[2],nz))
    st.latex('x = %a + %a * %a'% (receiver_coords[0],t,nx))
    st.latex('y = %a - %a * %a'% (receiver_coords[1],t,ny))
    st.latex('x = %a '% (x))
    st.latex('y = %a '% (y))

    return [x, y]


def calculate_data(iq_data):
    x_00, azimuth_x_12, elevation_x_12 = [], [], []
    azimuth_phases, elevation_phases = [], []
    ref_phases = []
    data_table_1 = []
    iq = list(map(int,iq_data.split(",")));
    iq_samples = [iq[n:n + 2] for n in range(0, len(iq), 2)]
    for iq_idx in range(N_SAMPLES_OF_REF_PERIOD - 1):
        i_next = iq_samples[iq_idx + 1][0] # data I next;
        q_next = iq_samples[iq_idx + 1][1] # data Q next
        i_curr = iq_samples[iq_idx][0]
        q_curr = iq_samples[iq_idx][1]
        iq_next = complex(i_next, q_next)
        iq_cur = complex(i_curr, q_curr)
        phase_next_rad = np.arctan2(iq_next.imag, iq_next.real)
        phase_next = np.rad2deg(phase_next_rad)
        phase_cur_rad = np.arctan2(iq_cur.imag, iq_cur.real)
        phase_cur = np.rad2deg(phase_cur_rad)
        phase_total = phase_next - phase_cur
        converted_to_plus_minus = (to_plus_minus_pi(phase_total))
        ref_phases.append(converted_to_plus_minus)
        data_table_1.append([i_curr,q_curr,phase_cur_rad,phase_next_rad,phase_cur,phase_next,phase_total,converted_to_plus_minus])
    phase_ref = np.mean(ref_phases)
    st.markdown('#')
    st.markdown('### **<h3 style="color:Blue;">1. Table Reference Period</h3>**',unsafe_allow_html=True)
    st.write("Data IQ yang diambil hanya Index IQ ke 1 sampai 8")
    df = pd.DataFrame(data_table_1,columns=('I','Q','Rad Phase (Current)','Rad Phase (Next)','Deg Phase (Current)','Deg Phase (Next)','Deg Phase (Next-Current)','to_plus_minus function'))
    st.table(df)
    st.write("**Phase Reference:**",phase_ref)
    st.write("**Keterangan Tabel Reference Period:**")
    st.write("**I:** Data I")
    st.write("**Q:** Data Q")
    st.write("**Rad Phase (Current):** Hasil hitung iq dalam bentuk radian, i dan q dihitung menggunakan fungsi numpy.arctan2(iq.imaginary, iq.real)")
    st.write("**Rad Phase (Next):** Hasil hitung iq yang berikutnya(index iq+1) dalam bentuk radian, i dan q dihitung menggunakan fungsi numpy.arctan2(iq.imaginary, iq.real)")
    st.write("**Deg Phase (Current):** Hasil konversi dari kolom **Rad Phase (Current)** yang sebelumnya radian menjadi degree, menggunakan fungsi numpy.rad2deg()")
    st.write("**Deg Phase (Next):** Hasil konversi dari kolom **Rad Phase (Next)** yang sebelumnya radian menjadi degree, menggunakan fungsi numpy.rad2deg()")
    st.latex('numpy.rad2deg() = radian * 180 / \Pi')
    st.markdown('#')
    st.write("**Deg Phase (Next-Current):** Hasil hitung dari kolom **Rad Phase (Next)** dikurang **(-)** dengan kolom **Rad Phase (Current)**")
    st.write("**to_plus_minus function:** Hasil konversi dari kolom **Deg Phase (Next-Current)** dengan menggunakan fungsi to_plus_minus. isi perhitungan fungsi to_plus_minus yaitu jika **angle <= 180** maka **angle += 2 * 180** dan jika **angle >= 180** maka **angle -= 2 * 180**")
    st.write("**Phase Reference:** Hasil rata-rata dari kolom **to_plus_minus function**, akan digunakan untuk perhitungan di tabel berikutnya")
    
    
    data_table_2 = []
    iq_2ant_batches = [iq_samples[n:n + 2] for n in range(N_SAMPLES_OF_REF_PERIOD, len(iq_samples), 2)]
    
    for iq_batch_idx, iq_batch in enumerate(iq_2ant_batches[:-1]):
        i_next, q_next = iq_batch[1][0], iq_batch[1][1]
        iq_next = complex(i_next,q_next)
        i_curr, q_curr = iq_batch[0][0], iq_batch[0][1]
        iq_cur = complex(i_curr, q_curr)
        phase_next_rad = np.arctan2(iq_next.imag, iq_next.real)
        phase_next = np.rad2deg(phase_next_rad)
        phase_cur_rad = np.arctan2(iq_cur.imag, iq_cur.real)
        phase_cur = np.rad2deg(phase_cur_rad)
        phase_total = (phase_next - phase_cur) - 2 * phase_ref
        diff_phase = to_plus_minus_pi(phase_total)
        if iq_batch_idx % 2 != 0:
            elevation_phases.append(diff_phase)
            data_table_2.append([i_curr,q_curr,i_next,q_next,phase_cur_rad,phase_next_rad,phase_cur,phase_next,phase_total,diff_phase,0,diff_phase])
        else:
            x_00.append(1)
            azimuth_phases.append(diff_phase)
            data_table_2.append([i_curr,q_curr,i_next,q_next,phase_cur_rad,phase_next_rad,phase_cur,phase_next,phase_total,diff_phase,diff_phase,0])
    st.markdown('##')
    st.markdown('### **<h3 style="color:Blue;">2. Table Sample Slot</h3>**',unsafe_allow_html=True)
    df = pd.DataFrame(data_table_2,columns=('I(Current)','Q(Current)','I(Next)','Q(Next)','Rad Phase (Current)','Rad Phase (Next)','Deg Phase (Current)','Deg Phase (Next)','Deg Diff Phase','to_plus_minus function','Azimuth Phase','Elevation Phase'))
    st.table(df)
    st.write("**Keterangan Tabel Sample Slot:**")
    st.write("**I (Current):** Data I current")
    st.write("**Q (Current):** Data Q current")
    st.write("**I (Next):** Data I next")
    st.write("**Q (Next):** Data Q next")
    st.write("**Rad Phase (Current):** Hasil hitung iq dalam bentuk radian dari kolom **I(Current) dan Q(Current)**, i dan q dihitung menggunakan fungsi numpy.arctan2(iq.imaginary, iq.real)")
    st.write("**Rad Phase (Next):** Hasil hitung iq dalam bentuk radian dari kolom **I(Next) dan Q(Next)**, i dan q dihitung menggunakan fungsi numpy.arctan2(iq.imaginary, iq.real)")
    st.write("**Deg Phase (Current):** Hasil konversi dari kolom **Rad Phase (Current)** yang sebelumnya radian menjadi degree, menggunakan fungsi numpy.rad2deg()")
    st.write("**Deg Phase (Next):** Hasil konversi dari kolom **Rad Phase (Next)** yang sebelumnya radian menjadi degree, menggunakan fungsi numpy.rad2deg()")
    st.latex('numpy.rad2deg() = radian * 180 / \Pi')
    st.markdown('#')
    st.write("**Deg Diff Phase:** Hasil hitung dari kolom **(Rad Phase (Next)-Rad Phase (Current)) - 2 * Phase Reference**(diambil dari hasil rata-rata tabel sebelumnya)")
    st.write("**to_plus_minus function:** Hasil konversi dari kolom **Deg Diff Phase** dengan menggunakan fungsi to_plus_minus. isi perhitungan fungsi to_plus_minus yaitu jika **angle <= 180** maka **angle += 2 * 180** dan jika **angle >= 180** maka **angle -= 2 * 180**")
    st.write("**Azimuth Phase:** Isi dari kolom **to_plus_minus function** jika index == genap")
    st.write("**Elevation Phase:** Isi dari kolom **to_plus_minus function** jika index == ganjil")


    st.markdown('##')
    st.markdown('### **<h3 style="color:Blue;">3. MUSIC Algorithm</h3>**',unsafe_allow_html=True)
    st.write("**Perhitungan Azimuth Angle**")
    st.write("Semua isi kolom **Azimuth Phase** dari tabel **Sample Slot** diproses dengan Python Code Berikut:")
    code = '''
    X = np.zeros((M, np.size(x_00)), dtype=complex)
    X[0, :] = x_00
    for i in azimuth_phases:
        azimuth_x_12.append(np.exp(1j * np.deg2rad(i)))
    X[1, :] = azimuth_x_12'''
    st.code(code, language='python')
    # MUSIC algo
    X = np.zeros((M, np.size(x_00)), dtype=complex)
    X[0, :] = x_00
    for i in azimuth_phases:
        azimuth_x_12.append(np.exp(1j * np.deg2rad(i)))
    X[1, :] = azimuth_x_12
    st.write("Setelah azimuth phase diproses menghasilkan data")
    st.write("Real part of azimuth_x_12: ", X.real)
    st.write("Imaginary part of azimuth_x_12: ", X.imag)
    st.write("Data diatas dihitung dengan mengestimasi matriks korelasi spasial untuk mendapatkan azimuth_angle menggunakan fungsi python **get_angle()**(ada dibawah): ")
    azimuth_angle = get_angle(X)
    st.write("Dari perhitungan tersebut menghasilkan **Azimuth Angle**: ",azimuth_angle)
    for i in elevation_phases:
        elevation_x_12.append(np.exp(1j * np.deg2rad(i)))
    X[1, :] = elevation_x_12
    elevation_angle = get_angle(X)

    st.markdown('##')
    st.write("**Perhitungan Elevation Angle**")
    st.write("Semua isi kolom **Elevation Phase** dari tabel **Sample Slot** diproses dengan Python Code Berikut:")
    code = '''
    X = np.zeros((M, np.size(x_00)), dtype=complex)
    X[0, :] = x_00
    for i in elevation_phases:
        elevation_x_12.append(np.exp(1j * np.deg2rad(i)))
    X[1, :] = elevation_x_12'''
    st.code(code, language='python')
    st.write("Setelah elevation phase diproses menghasilkan data")
    st.write("Real part of elevation_x_12: ", X.real)
    st.write("Imaginary part of elevation_x_12: ", X.imag)
    st.write("Data diatas dihitung dengan mengestimasi matriks korelasi spasial untuk mendapatkan elevation_angle menggunakan fungsi python **get_angle()**(ada dibawah): ")
    st.write("Dari perhitungan tersebut menghasilkan **Elevation Angle**: ",elevation_angle)



    st.markdown('##')
    st.write("**Fungsi get_angle()**")

    code = '''
    array_alignment = np.arange(0, M, 1) * d
    incident_angles = np.arange(-90, 91, 1)
    scanning_vectors = np.zeros((M, np.size(incident_angles)), dtype=complex)
    for i in range(np.size(incident_angles)):
        scanning_vectors[:, i] = np.exp(array_alignment * 1j * 2 * np.pi * np.sin(np.radians(incident_angles[i])) / wavelength)

    ula_scanning_vectors = scanning_vectors
    
    MUSIC = DOA_MUSIC(R, ula_scanning_vectors, signal_dimension=1)
    norm_data = np.divide(np.abs(MUSIC), np.max(np.abs(MUSIC)))
    return float(incident_angles[np.where(norm_data == 1)[0]][0])'''
    st.code(code, language='python')
    
    st.markdown('##')
    st.write("**Menentukan Estimasi Koordinat(x, y)**")
    st.write("Berikut adalah penjelasan proses/ urutan menentukan koordinat dari python code:")
    code = '''
    nx = numpy.cos(np.deg2rad(90.0 - azimuth)) 
    nz = numpy.cos(np.deg2rad(90.0 - abs(elevation)))
    if math.isclose(nx, 0.0, abs_tol=1e-16) or math.isclose(nz, 0.0, abs_tol=1e-16):
        return [float("nan"),  float("nan")]
    else:
        ny = numpy.sqrt(1 - nx ** 2 - nz ** 2)
        t = (z_beacon - z_locator) / nz
        x = x_locator + t * nx
        y = y_locator - t * ny
    return [x, y]'''

    st.code(code, language='python')

    st.write("Atau jika ditulis dalam formula matematika adalah sebagai berikut: ")
    st.latex('nx = \cos{(90.0 - azimuth) * \Pi / 180}')
    st.latex('nz = \cos{(90.0 - abs(elevation)) * \Pi / 180}')
    st.latex('ny = \sqrt{\smash[b]{1 - nx^2 - nz^2}}')
    st.latex('t = (zbeacon - zlocator) / nz')
    st.latex('x = xlocator + t * nx')
    st.latex('y = ylocator - t * ny')

    xy = get_coordinate(azimuth_angle, elevation_angle, z_beacon, [x_locator, y_locator, z_locator])
    if not math.isnan(xy[0]) and not math.isnan(xy[1]):
        data_table_3 = [[azimuth_angle,elevation_angle,xy[0],xy[1]]]
        st.markdown('##')
        st.markdown('### **<h3 style="color:Blue;">4. Table Hasil</h3>**',unsafe_allow_html=True)
        df = pd.DataFrame(data_table_3,columns=('Azimuth Angle','Elevation Angle','X','Y'))
        st.table(df)

        st.markdown('### **<h3 style="color:Blue;">5. Sigma Filter</h3>**',unsafe_allow_html=True)
        st.write("""
        Untuk melakukan sigma filter dibutuhkan minimal **6** data dari **Tabel Hasil**, dimana untuk index **1-5** akan dijadikan sebagai sampel dan diambil rata-rata dari **X** dan **Y**. kemudian rata-rata **X** dan **Y** dari sample tersebut akan dibandingkan dengan **X** dan **Y** yang berikutnya(hitungan ke 6, dst).
        Jika **X** dan **Y** mendekati dengan rata-rata sample yang telah diambil maka **X** dan **Y** tersebut dinyatakan valid, tetapi jika **X** dan **Y** terlalu jauh dari rata-rata sample maka data **X** dan **Y** akan dibuang.
        """)



with st.form(key='my_form'):
    iq = st.text_input("Enter Raw IQ here",key="iq", value="-164,36,-38,-161,161,-28,27,162,-166,17,-11,-166,165,-7,1,166,-223,65,-39,-209,109,-155,72,130,-226,23,5,-215,143,-136,47,143,-228,-38,25,-213,162,-84,34,148,-218,-58,98,-188,175,-70,-18,156,-208,-110,123,-173,185,-44,-32,144,-175,-141,150,-157,190,-15,-69,143,-128,-180,166,-139,189,31,-92,115,-109,-194,197,-88,177,63,-108,105,-61,-223,203,-55")
    # calculate = st.form_submit_button("Calculate")
    col1, col2, col3 = st.columns([0.9,0.9,5])
    with col1:
        calculate = st.form_submit_button("Generate Estimation")
    with col2:
        clear = st.form_submit_button("Clear")    
    if calculate:
        if(iq==""):
            st.error("Raw IQ cannot empty")
        else:
            try:
                calculate_data(iq)
            except BaseException as e:
                st.error(e)
    if clear:
        st.empty()

        



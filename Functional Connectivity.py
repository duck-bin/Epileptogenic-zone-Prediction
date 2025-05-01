#%%
import mne
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from sklearn.cluster import AgglomerativeClustering


def load_edf_file(file_path):
    """EDF 파일을 불러오는 함수"""
    # EDF 파일 읽기
    raw = mne.io.read_raw_edf(file_path, preload=True)
    return raw

#Channels_to_keep에서 10,20system를 설정함
#Standard_1020에서의 표준 설정된 1020system의 좌표를 추출함.
def select_channels(raw, channels_to_keep=None):
    """사용할 채널을 선택하고 마운트 설정하는 함수"""
    if channels_to_keep is None:
        # 기본 채널 목록 (EEG 및 참조 채널)
        channels_to_keep = [
            'C3', 'C4', 'Cz',
            'F3', 'F4', 'F7', 'F8', 'Fz',
            'Fp1', 'Fp2', 'Fpz',
            'O1', 'O2', 'Oz',
            'P3', 'P4', 'Pz',
            'T3', 'T4', 'T5', 'T6',
            # 'T1', 'T2',
            'A1', 'A2',  # 참조 전극
            # 'LCK', 'RCK'  # 확장 전극 (필요에 따라 포함)
        ]

    # 불필요한 채널 제거: pick_channels 함수로 선택된 채널만 유지
    raw.pick_channels(channels_to_keep)

    # 좌표(마운트) 설정: standard_1020 마운트 적용
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage)
    
    return raw

#Notch filer and Bandpass filter.
#Functional Connectivity 계산할 시에는 80HZ 이상으로는 분석 안했음.
def apply_filters(raw, notch_freq=60, l_freq=1.0, h_freq=80.0):
    """필터링을 적용하는 함수"""
    # 1. 60Hz 노치 필터 적용
    #    전력선 간섭을 제거하기 위해 60Hz 주파수를 제거합니다.
    raw.notch_filter(freqs=notch_freq, picks='eeg')

    # 2. 밴드패스 필터 적용 (1 ~ 70 Hz)
    #    저주파 드리프트(1Hz 이하)와 고주파 잡음(70Hz 이상)을 제거합니다.
    raw.filter(l_freq=l_freq, h_freq=h_freq, picks='eeg')
    
    return raw

#Common average reference 하면됨.
def apply_reference(raw, ref_type='average', ref_channels=None):
    """재참조를 적용하는 함수"""
    if ref_type == 'earlobes' or (ref_type == 'custom' and ref_channels == ['A1', 'A2']):
        # A1, A2 전극을 기준으로 참조 변경 (귓볼 참조)
        raw.set_eeg_reference(ref_channels=['A1', 'A2'])
    elif ref_type == 'average':
        # Average reference로 재참조
        raw.set_eeg_reference(ref_channels='average')
    elif ref_type == 'custom':
        # 사용자 지정 채널로 재참조
        raw.set_eeg_reference(ref_channels=ref_channels)
        
    return raw

#A1,A2는 reference에 사용하였으니, 제외함.
def clean_channels(raw, channels_to_drop=None):
    """채널 정리 및 위치 설정을 확인하는 함수"""
    print(f"현재 채널 목록: {raw.ch_names}")

    # 채널 제거 (필요 시)
    if channels_to_drop:
        for channel in channels_to_drop:
            if channel in raw.ch_names:
                raw.drop_channels([channel])
    else:
        # 기본적으로 참조 채널 제거
        if 'A1' in raw.ch_names:
            raw.drop_channels(['A1'])
        if 'A2' in raw.ch_names:
            raw.drop_channels(['A2'])
    
    # 현재 좌표 설정 확인
    print("채널 위치 정보:")
    for i, ch_name in enumerate(raw.ch_names):
        if i < len(raw.info['chs']):
            loc = raw.info['chs'][i]['loc'][:3]
            print(f"{ch_name} → 위치={loc}")
            
    return raw

#원하는 시간대에서 추출함.
#1000~1060초도 잘 나옴.
def extract_time_segment(raw, tmin=1620, tmax=1680):
    """특정 시간 구간을 추출하는 함수"""
    # 특정 구간 선택 (기본값: 1620-1680초)
    segment = raw.copy().crop(tmin=tmin, tmax=tmax)
    return segment

#Epoch생성하기.
def create_epochs(segment, duration=3.0):
    """데이터를 일정 구간으로 분할하는 함수"""
    from mne import make_fixed_length_epochs

    # 지정된 길이의 에포크로 분할
    epochs = make_fixed_length_epochs(segment, duration=duration, preload=True)
    print(f"Number of {duration}-second epochs: {len(epochs)}")

    epochs_data = epochs.get_data()
    print("Epochs data shape:", epochs_data.shape)
    
    return epochs

#개인화 된 MRI 사용할 시에 중요
#안 사용할 경우 fsaverage라는 표준 MRI를 사용함.
#Trans는 전극을 MRI에 조정하는 작업을 진행한 것.
#이에 대해서는 mne coreg를 사용해야됨.
def setup_forward_solution(segment, subject, subjects_dir, use_fsaverage=False, trans=None):
    """Forward Solution을 구성하는 함수"""
    # fsaverage 사용 설정
    if use_fsaverage:
        fs_dir = mne.datasets.fetch_fsaverage(verbose=True)
        subject = 'fsaverage'
        subjects_dir = os.path.dirname(fs_dir)
        print("Using fsaverage subject, subjects_dir:", subjects_dir)
    
    # Source space 구성 (피질 표면; oct6 spacing)
    src = mne.setup_source_space(subject, spacing='oct6', add_dist=False,
                                subjects_dir=subjects_dir, verbose=True)
    print("Source space setup complete.")
    
    # 3-layer BEM 모델 생성 (전형적인 전도도 값 사용: 두피, 두개골, 뇌)
    conductivity = (0.3, 0.006, 0.3)  # 3-layer BEM 전도도 값
    model = mne.make_bem_model(subject=subject, ico=4, conductivity=conductivity,
                            subjects_dir=subjects_dir)
    bem = mne.make_bem_solution(model)
    print("3-layer BEM model and solution created.")
    
    #trans file를 미리 만드는 것을 추천함.
    # Forward solution 계산 (EEG 데이터)
    if trans is None:
        trans = os.path.join(subjects_dir, subject, subject + '-trans.fif')
    
    fwd = mne.make_forward_solution(
        info=segment.info,
        trans=trans,
        src=src,      # 이미 만들어둔 source space
        bem=bem,      # 이미 만들어둔 BEM
        eeg=True,
        meg=False
    )
    print("Forward solution computed.")
    
    return src, bem, fwd

#virtual sensor를 설정함.
#clustering 기준으로 Desikan–Killiany labels 가져옴.
def setup_virtual_sensors(src, subject, subjects_dir, resolution_mm=10):
    """Visual Sensor를 설정하는 함수"""
    # 조정 가능한 공간 해상도 (mm 단위)
    resolution = resolution_mm / 1000.0  # mm -> m

    # fsaverage 아틀라스에서 Desikan–Killiany labels 읽어오기 ('aparc' 사용)
    labels = mne.read_labels_from_annot(subject, parc='aparc', subjects_dir=subjects_dir)
    
    # 가상 센서(VS) 템플릿을 저장할 리스트
    virtual_sensors = []  # 각 항목은 dict: 'label', 'hemi', 'cluster_id', 'coord', 'vs_indices'

    # 소스 공간에서 좌측, 우측 반구의 좌표 정보를 사용합니다.
    left_coords = src[0]['rr'][src[0]['vertno'], :]  
    right_coords = src[1]['rr'][src[1]['vertno'], :]

    # 전체 vertex를 used vertex로 간주합니다.
    used_left = src[0]['vertno']   # 왼쪽 반구의 모든 vertex 번호
    used_right = src[1]['vertno']  # 오른쪽 반구의 모든 vertex 번호

    # 각 label(아틀라스 영역)별로 VS 구성
    for label in labels:
        if label.hemi == 'lh':
            verts = src[0]['vertno']         # 왼쪽 반구의 vertex 번호 (예: 4098개)
            coords = left_coords             # 해당 vertex의 좌표 (shape: (4098, 3))
            used_verts = np.intersect1d(label.vertices, used_left)
            # used_verts가 verts 내에서 차지하는 위치(인덱스)를 찾음
            vs_indices = [np.where(verts == v)[0][0] for v in used_verts]
        else:
            verts = src[1]['vertno']         # 오른쪽 반구의 vertex 번호
            coords = right_coords
            used_verts = np.intersect1d(label.vertices, used_right)
            vs_indices = [np.where(verts == v)[0][0] for v in used_verts]
        
        if len(used_verts) == 0:
            continue

        # label에 해당하는 vertex 좌표 추출 (verts 배열에 대해 사용 여부 판단)
        mask = np.isin(verts, used_verts)
        label_coords = coords[mask]
        
        # AgglomerativeClustering을 사용해, 지정한 해상도 이하의 거리에 있는 vertex들을 클러스터링
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=resolution, linkage='complete')
        cluster_labels = clustering.fit_predict(label_coords)
        
        # 각 클러스터마다 VS를 정의
        for clust_id in np.unique(cluster_labels):
            cluster_idx = np.where(cluster_labels == clust_id)[0]
            vs_coord = np.mean(label_coords[cluster_idx, :], axis=0)
            clust_vs_indices = [vs_indices[i] for i in cluster_idx]
            
            virtual_sensors.append({
                'label': label.name,
                'hemi': label.hemi,
                'cluster_id': clust_id,
                'coord': vs_coord,
                'vs_indices': clust_vs_indices  # 이후 각 epoch STC에서 해당 인덱스로 time series 추출
            })

    print(f"Pre-defined {len(virtual_sensors)} virtual sensors (resolution {resolution_mm} mm) across the cortex.")
    
    return virtual_sensors

#시각화1
def visualize_vs_matplotlib(virtual_sensors):
    """Virtual sensor location을 matplotlib로 시각화하는 함수"""
    # 좌측, 우측 VS 좌표 분리
    left_coords = np.array([vs['coord'] for vs in virtual_sensors if vs['hemi'] == 'lh'])
    right_coords = np.array([vs['coord'] for vs in virtual_sensors if vs['hemi'] == 'rh'])

    # 3D 플롯 생성
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    if left_coords.size > 0:
        ax.scatter(left_coords[:, 0], left_coords[:, 1], left_coords[:, 2], 
                c='blue', s=20, label='Left Hemisphere')
    if right_coords.size > 0:
        ax.scatter(right_coords[:, 0], right_coords[:, 1], right_coords[:, 2], 
                c='red', s=20, label='Right Hemisphere')

    ax.set_title("Virtual Sensor Locations")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend()

    plt.show()
    
    return fig

#시각화2
def visualize_vs_plotly(virtual_sensors):
    """Virtual sensor location을 plotly로 시각화하는 함수"""
    # 좌측, 우측 VS 좌표 및 라벨 추출
    left_coords = np.array([vs['coord'] for vs in virtual_sensors if vs['hemi'] == 'lh'])
    right_coords = np.array([vs['coord'] for vs in virtual_sensors if vs['hemi'] == 'rh'])

    # 좌표에 해당하는 라벨 정보도 추출 (hover text에 사용)
    left_labels = [f"{vs['label']}_cluster{vs['cluster_id']}" for vs in virtual_sensors if vs['hemi'] == 'lh']
    right_labels = [f"{vs['label']}_cluster{vs['cluster_id']}" for vs in virtual_sensors if vs['hemi'] == 'rh']

    # Plotly 3D 산점도 생성
    fig = go.Figure()

    # 좌측 반구 데이터 추가
    if len(left_coords) > 0:
        fig.add_trace(go.Scatter3d(
            x=left_coords[:, 0],
            y=left_coords[:, 1],
            z=left_coords[:, 2],
            mode='markers',
            marker=dict(
                size=5,
                color='blue',
                opacity=0.8
            ),
            text=left_labels,
            hoverinfo='text',
            name='Left Hemisphere'
        ))

    # 우측 반구 데이터 추가
    if len(right_coords) > 0:
        fig.add_trace(go.Scatter3d(
            x=right_coords[:, 0],
            y=right_coords[:, 1],
            z=right_coords[:, 2],
            mode='markers',
            marker=dict(
                size=5,
                color='red',
                opacity=0.8
            ),
            text=right_labels,
            hoverinfo='text',
            name='Right Hemisphere'
        ))

    # 레이아웃 설정
    fig.update_layout(
        title='Virtual Sensor Locations',
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            aspectmode='data',  # 데이터 비율에 맞게 축 비율 설정
            camera=dict(
                up=dict(x=0, y=0, z=1),
                eye=dict(x=-1.5, y=-1.5, z=1.5)
            )
        ),
        legend=dict(
            x=0.8,
            y=0.9,
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        # 2D 주석은 z 속성을 사용하지 않습니다
        annotations=[
            dict(
                showarrow=False,
                x=0.5,
                y=0,
                xref='paper',
                yref='paper',
                text="마우스로 드래그하여 회전, 스크롤하여 확대/축소, Shift+드래그하여 이동",
                xanchor="center",
                yanchor="auto",
                opacity=0.7
            )
        ]
    )

    # 그래프 표시
    fig.show()
    
    return fig

#LCMV beamformer 적용함.
def apply_beamformer(segment, epochs, fwd):
    """Beamformer를 적용하는 함수"""
    # 평균 참조 적용
    segment.set_eeg_reference(projection=True)
    segment.apply_proj()

    # epochs에도 평균 참조 적용
    epochs.set_eeg_reference(projection=True)
    epochs.apply_proj()

    # 노이즈 공분산 계산
    cov_epochs = mne.compute_covariance(epochs, tmin=0, tmax=3.0, method='empirical')
    print("Noise covariance computed from epochs.")

    # LCMV beamformer 필터 생성
    filters = mne.beamformer.make_lcmv(epochs.info, fwd, cov_epochs, reg=0.05,
                                    pick_ori='max-power', weight_norm='nai', verbose=True)
    print("LCMV filters computed.")

    # 각 epoch에 beamformer 적용하여 STC 계산
    stcs = mne.beamformer.apply_lcmv_epochs(epochs, filters, verbose=True)
    print(f"LCMV source time series computed for {len(stcs)} epochs.")
    
    return filters, stcs

#Times series 계산산
def extract_vs_time_series(stcs, virtual_sensors):
    """Epoch_vs_data Time series를 추출하는 함수"""
    epoch_vs_data = []  # 각 요소: {'epoch_index': i, 'vs_time_series': { VS_key: time_series }}

    for i, stc in enumerate(stcs):
        vs_ts_dict = {}
        # stc.vertices는 [left_vertex_array, right_vertex_array]
        # 오른쪽 반구 데이터의 시작 인덱스는 좌측 vertex 수와 같습니다.
        left_vertex_count = len(stc.vertices[0])
        
        for vs in virtual_sensors:
            key = f"{vs['label']}_cluster{vs['cluster_id']}_{vs['hemi']}"
            if vs['hemi'] == 'lh':
                # 왼쪽 반구: vs['vs_indices']는 stc.data의 왼쪽 부분 인덱스에 해당
                ts = np.mean(stc.data[vs['vs_indices'], :], axis=0)
            else:
                # 오른쪽 반구: 인덱스에 offset 적용
                ts = np.mean(stc.data[left_vertex_count + np.array(vs['vs_indices']), :], axis=0)
            vs_ts_dict[key] = ts
        epoch_vs_data.append({'epoch_index': i, 'vs_time_series': vs_ts_dict})

    print("Extracted VS time series for all epochs.")
    # epoch_vs_data의 길이(즉, 에포크 수)를 확인합니다.
    print("Number of epochs:", len(epoch_vs_data))
    # 첫 번째 epoch의 가상 센서 데이터 확인
    first_epoch = epoch_vs_data[0]
    print("Epoch index:", first_epoch['epoch_index'])
    print("Number of virtual sensors in this epoch:", len(first_epoch['vs_time_series']))
    
    return epoch_vs_data

#Epochs 만들기기
def create_vs_epochs(epoch_vs_data, epochs):
    """VS 시계열 데이터로 Epochs 객체를 생성하는 함수"""
    # 채널 이름 결정 (첫 번째 epoch의 키 사용)
    channel_names = sorted(list(epoch_vs_data[0]['vs_time_series'].keys()))

    # 데이터 배열 생성: (n_epochs, n_channels, n_times)
    n_epochs = len(epoch_vs_data)
    n_channels = len(channel_names)
    n_times = epoch_vs_data[0]['vs_time_series'][channel_names[0]].shape[0]

    # 데이터 배열 초기화
    data = np.zeros((n_epochs, n_channels, n_times))

    # 데이터 채우기
    for i, ep in enumerate(epoch_vs_data):
        vs_dict = ep['vs_time_series']
        for j, ch in enumerate(channel_names):
            if ch in vs_dict:
                data[i, j, :] = vs_dict[ch]

    # epochs 객체 생성
    sfreq = epochs.info['sfreq']  # 원본 epochs의 샘플링 주파수 사용
    info = mne.create_info(ch_names=channel_names, sfreq=sfreq, ch_types='eeg')
    epochs_vs = mne.EpochsArray(data, info)
    print("New Epochs object for virtual sensors created.")
    print(f"Epochs shape: {epochs_vs.get_data().shape}")
    
    return epochs_vs


#%%
def main():
    """메인 함수"""
    # 1. EDF 파일 불러오기
    file_path = r"C:\hyunbin\Computational Neuroscience\CN data\jjy_eeg\11022648~ Chun_e5517c9a-57ec-4406-820d-c52afc412419.edf"
    raw = load_edf_file(file_path)
    
    # 2. 사용할 채널 선택
    raw = select_channels(raw)
    
    # 3. 필터링 적용
    raw = apply_filters(raw)
    
    # 4. 재참조 적용
    raw = apply_reference(raw, ref_type='earlobes')  # 귓볼 참조
    raw = apply_reference(raw, ref_type='average')   # 평균 참조
    
    # 5. 채널 정리
    raw = clean_channels(raw)
    
    # 6. 특정 시간 구간 추출
    segment = extract_time_segment(raw, tmin=1620, tmax=1680)
    
    # 7. Epoching - 데이터를 3초 구간으로 분할
    epochs = create_epochs(segment, duration=3.0)
    
    # 8. Forward Solution 구성
    subject = 'jjy_mri'
    subjects_dir = r"C:\hyunbin\Computational Neuroscience\CN data"
    trans = r"C:\hyunbin\Computational Neuroscience\CN data\jjy_mri\jjy_mri-trans.fif"
    src, bem, fwd = setup_forward_solution(segment, subject, subjects_dir, trans=trans)
    
    # 9. Virtual Sensor 설정
    virtual_sensors = setup_virtual_sensors(src, subject, subjects_dir, resolution_mm=10)
    
    # 10. Virtual sensor 시각화 (matplotlib)
    visualize_vs_matplotlib(virtual_sensors)
    
    # 11. Virtual sensor 시각화 (plotly)
    visualize_vs_plotly(virtual_sensors)
    
    # 12. Beamformer 적용
    filters, stcs = apply_beamformer(segment, epochs, fwd)
    
    # 13. Epoch_vs_data Time series 추출
    epoch_vs_data = extract_vs_time_series(stcs, virtual_sensors)
    
    # 14. VS 시계열 데이터로 Epochs 객체 생성
    epochs_vs = create_vs_epochs(epoch_vs_data, epochs)
    
    return raw, segment, epochs, src, bem, fwd, virtual_sensors, filters, stcs, epoch_vs_data, epochs_vs


if __name__ == "__main__":
    # 전체 프로세스 실행
    raw, segment, epochs, src, bem, fwd, virtual_sensors, filters, stcs, epoch_vs_data, epochs_vs = main()
# %%
#개인화 MRI를 기반으로 bem, forward model, inverse model를 만드는 법에 대한 상세 설명.
"""
#fsaverage 표준 MRI 사용시
fs_dir = mne.datasets.fetch_fsaverage(verbose=True)
subject = 'fsaverage'
subjects_dir = os.path.dirname(fs_dir)

#개인화된 BEM 만들기
#Freesurfer로 MRI 작업을 한 후 BEM이 없는 경우가 있음.
#이 때는 기존의 MRI를 기반으로 BEM를 만들어야 함.
#리눅스(아마 freesurfer가 다운 받아야되서 그런듯)에서 실행함.
#watershed, surface를 만드는 작업임.

import mne
mne.bem.make_watershed_bem(subject, subjects_dir)

#이것을 토대로 BEM forward model를 계산할 수 있음.

#trans file 만들기
#전극을 mri에 정렬하는 작업을 진행해야됨
#Terminal에 다음과 같이 입력. 그냥 mne coreg를 입력할 경우에는 fsaverage로 자동으로 띄워져서 개인화된 사람을 지칭하는 것이 좋아보임.

mne coreg --subject jjy_mri --subjects-dir "C:\hyunbin\Computational Neuroscience\CN data"

#standard 1020 system의 전극 좌표를 먼저 mne coreg에 띄우기 위해 위치를 fif file 형태로 저장해야 됨.
# ③ 표준 EEG electrode 위치 설정 (10-20 시스템)
raw.set_montage("standard_1020")  # 또는 "standard_1005", "biosemi64" 등

# ④ FIF로 저장
raw.save(r"C:\hyunbin\Computational Neuroscience\CN data\segment_raw.fif", overwrite=True)

#standard 1020 system으로 가져온 전극 좌표를 기준으로 수동으로 정렬하기.
# 그리고 trans 파일로 저장하기 trans -fif 으로 저장하기. 
trans = r"C:\hyunbin\Computational Neuroscience\CN data\jjy_mri\jjy_mri-trans.fif" 위의 예제에서는 다음과 같은 명칭을 사용함.
"""

#mri file 들 저장하기.
def save_bem_solution(bem, subject, subjects_dir):
    """BEM 솔루션을 저장하는 함수"""
    # 저장 경로 지정
    bem_path = os.path.join(subjects_dir, subject, 'bem', f'{subject}-5120-5120-5120-bem-sol.fif')

    # BEM 솔루션 저장
    mne.write_bem_solution(bem_path, bem)

    print(f"✅ BEM solution saved to:\n{bem_path}")
    return bem_path

def save_forward_solution(fwd, output_path):
    """Forward solution을 저장하는 함수"""
    mne.write_forward_solution(output_path, fwd, overwrite=True)
    print(f"Forward solution saved to: {output_path}")
    return output_path

# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from mne_connectivity import envelope_correlation, spectral_connectivity_epochs
from sklearn.cluster import AgglomerativeClustering


# ================== Connectivity Calculation Functions ==================

def vec_to_sym_matrix(vec, n):
    """
    1D 벡터를 대칭 행렬로 변환하는 함수
    """
    M = np.zeros((n, n))
    idx = 0
    for i in range(n):
        for j in range(i, n):
            M[i, j] = vec[idx]
            M[j, i] = vec[idx]
            idx += 1
    return M


def compute_aec_connectivity(epochs):
    """
    AEC(Amplitude Envelope Correlation) 기반 Functional Connectivity 계산.
    모든 에포크의 연결성 행렬을 평균 내어 최종 (n_channels, n_channels) 행렬 반환.
    """
    n_epochs, n_channels, _ = epochs.get_data().shape
    # envelope_correlation은 epochs의 데이터를 입력받아 각 epoch의 연결성을 계산합니다.
    aec_conn = envelope_correlation(
        epochs.get_data(),
        names=epochs.ch_names,
        orthogonalize='pairwise',
        log=False,
        absolute=True,
        verbose=False
    )
    
    aec_data = aec_conn.get_data()  # shape: (n_epochs, L), L = (n_channels*(n_channels+1))//2
    aec_matrices = [vec_to_sym_matrix(aec_data[i].flatten(), n_channels) for i in range(n_epochs)]
    aec_matrices = np.array(aec_matrices)  # shape: (n_epochs, n_channels, n_channels)
    
    avg_aec_matrix = np.mean(aec_matrices, axis=0)
    print(f"Averaged AEC matrix shape: {avg_aec_matrix.shape}")  # (n_channels, n_channels)
    
    return avg_aec_matrix, epochs.ch_names


def compute_plv_connectivity(epochs, fmin=8.0, fmax=12.0):
    """
    PLV(Phase Locking Value) 기반 Functional Connectivity 계산.
    모든 에포크의 연결성 행렬을 평균 내어 최종 (n_channels, n_channels) 행렬 반환.
    
    Parameters:
    -----------
    epochs : mne.Epochs
        Connectivity를 계산할 Epochs 객체
    fmin : float
        분석할 주파수 하한값 (Hz)
    fmax : float
        분석할 주파수 상한값 (Hz)
        
    Returns:
    --------
    conn_matrix : ndarray, shape (n_channels, n_channels)
        채널 간 PLV 연결성 행렬
    ch_names : list
        채널 이름 목록
    """
    n_channels = len(epochs.ch_names)
    
    # 모든 채널 쌍에 대한 인덱스 생성 (하삼각 행렬)
    indices = np.triu_indices(n_channels, k=1)
    
    # PLV 계산
    con = spectral_connectivity_epochs(
        data=epochs,
        method='plv',
        indices=indices,
        sfreq=epochs.info['sfreq'],
        mode='multitaper',
        fmin=fmin,
        fmax=fmax,
        faverage=True,
        tmin=0.0,
        tmax=None,
        mt_bandwidth=4.0,
        verbose=False
    )
    
    # 연결성 데이터 추출
    con_data = con.get_data()  # Shape: (n_connections, n_freqs) or (n_connections, 1) if faverage=True
    
    # 빈 연결성 행렬 초기화
    conn_matrix = np.zeros((n_channels, n_channels))
    
    # 연결성 값으로 행렬 채우기 (하삼각 부분)
    conn_matrix[indices] = con_data[:, 0]  # faverage=True이므로 두 번째 차원은 크기가 1
    
    # 대칭 행렬로 만들기 (PLV는 방향성이 없음)
    conn_matrix = conn_matrix + conn_matrix.T
    
    # 대각선은 1로 설정 (자기 자신과의 연결성은 완벽함)
    np.fill_diagonal(conn_matrix, 1.0)
    
    print(f"PLV Connectivity matrix shape: {conn_matrix.shape}")
    
    return conn_matrix, epochs.ch_names


def compute_pearson_correlation(epochs):
    """
    주어진 epochs 객체에서 채널별 Pearson 상관행렬을 계산합니다.
    
    Parameters:
        epochs (mne.Epochs): MNE Epochs 객체, shape=(n_epochs, n_channels, n_times)
        
    Returns:
        corr_matrix (np.ndarray): 채널 간 Pearson 상관행렬, shape=(n_channels, n_channels)
    """
    # Epochs 데이터를 추출 (shape: (n_epochs, n_channels, n_times))
    data = epochs.get_data()
    n_epochs, n_channels, n_times = data.shape
    
    # 각 채널의 데이터를 모든 epoch와 시간 포인트를 연결하여 2차원 배열로 만듭니다.
    # 최종 shape: (n_channels, n_epochs * n_times)
    data_reshaped = data.transpose(1, 0, 2).reshape(n_channels, -1)
    
    # np.corrcoef를 이용하여 피어슨 상관행렬 계산 (각 행이 하나의 채널)
    corr_matrix = np.corrcoef(data_reshaped)
    return corr_matrix


# ================== Matrix Sorting and Visualization Functions ==================

# ROI 우선순위 정의
roi_priority = {
    "precentral": 1,            # Primary motor cortex
    "postcentral": 2,           # Primary somatosensory cortex
    "paracentral": 3,
    "inferiorparietal": 4,
    "superiorparietal": 5,
    "precuneus": 6,
    "superiorfrontal": 7,
    "caudalmiddlefrontal": 8,
    "rostralmiddlefrontal": 9,
    "frontalpole": 10,
    "lateralorbitofrontal": 11,
    "medialorbitofrontal": 12,
    "parsopercularis": 13,
    "parsorbitalis": 14,
    "parstriangularis": 15,
    "caudalanteriorcingulate": 16,
    "isthmuscingulate": 17,
    "rostralanteriorcingulate": 18,
    "posteriorcingulate": 19,
    "lateraloccipital": 20,
    "cuneus": 21,
    "pericalcarine": 22,
    "lingual": 23,
    "bankssts": 24,             # Banks of the superior temporal sulcus
    "fusiform": 25,
    "entorhinal": 26,
    "parahippocampal": 27,
    "superiortemporal": 28,
    "inferiortemporal": 29,
    "middletemporal": 30,
    "transversetemporal": 31,
    "temporalpole": 32,
    "supramarginal": 33,
    "insula": 34
}

# ROI들을 뇌엽으로 그룹화
roi_to_lobe = {
    # 전두엽 (Frontal Lobe)
    "precentral": "Frontal",
    "superiorfrontal": "Frontal",
    "caudalmiddlefrontal": "Frontal",
    "rostralmiddlefrontal": "Frontal",
    "frontalpole": "Frontal",
    "lateralorbitofrontal": "Frontal",
    "medialorbitofrontal": "Frontal",
    "parsopercularis": "Frontal",
    "parsorbitalis": "Frontal",
    "parstriangularis": "Frontal",
    
    # 두정엽 (Parietal Lobe)
    "postcentral": "Parietal",
    "paracentral": "Parietal",
    "inferiorparietal": "Parietal",
    "superiorparietal": "Parietal",
    "precuneus": "Parietal",
    "supramarginal": "Parietal",
    
    # 측두엽 (Temporal Lobe)
    "bankssts": "Temporal",
    "fusiform": "Temporal",
    "superiortemporal": "Temporal",
    "inferiortemporal": "Temporal",
    "middletemporal": "Temporal",
    "transversetemporal": "Temporal",
    "temporalpole": "Temporal",
    "entorhinal": "Temporal",
    "parahippocampal": "Temporal",
    
    # 후두엽 (Occipital Lobe)
    "lateraloccipital": "Occipital",
    "cuneus": "Occipital",
    "pericalcarine": "Occipital",
    "lingual": "Occipital",
    
    # 대상회 (Cingulate)
    "caudalanteriorcingulate": "Cingulate",
    "isthmuscingulate": "Cingulate",
    "rostralanteriorcingulate": "Cingulate",
    "posteriorcingulate": "Cingulate",
    
    # 뇌섬엽 (Insula)
    "insula": "Insula"
}

# 뇌엽 우선순위 정의 (시각화 순서용)
lobe_priority = {
    "Frontal": 1,
    "Parietal": 2,
    "Temporal": 3,
    "Occipital": 4,
    "Cingulate": 5,
    "Insula": 6
}


def sort_by_roi_and_hemisphere(vs_name):
    """
    vs_name 예: 'bankssts-lh_cluster0_lh'
    분해하면:
      parts[0] = 'bankssts-lh'   (ROI 이름 + hemisphere 접미사)
      parts[1] = 'cluster0'      (클러스터 ID)
      parts[2] = 'lh'            (실제 반구 정보)
    """
    parts = vs_name.split('_')
    roi_plus_hemi = parts[0]    # 예: "bankssts-lh"
    cluster_str = parts[1]      # 예: "cluster0"
    real_hemi_str = parts[-1]   # "lh" 또는 "rh"

    # ROI 이름만 추출 ("bankssts-lh" -> "bankssts")
    roi_main = roi_plus_hemi.split('-')[0]

    # 클러스터 ID를 정수로 변환
    cluster_id = int(cluster_str.replace('cluster', ''))

    # 반구: LH가 0, RH가 1 (LH가 먼저 오도록)
    hemi_val = 0 if real_hemi_str == 'lh' else 1

    # ROI 우선순위 (딕셔너리에 없으면 기본값 999)
    roi_rank = roi_priority.get(roi_main, 999)

    # 반환: (반구 우선순위, ROI 우선순위, 클러스터 ID)
    return (hemi_val, roi_rank, cluster_id)


def sort_matrix_by_channels(avg_matrix, epoch_vs_data, sort_func):
    """
    채널 이름 기준으로 정렬된 인덱스를 생성하고, 
    평균 행렬을 해당 인덱스에 맞게 재정렬하는 함수입니다.
    
    Parameters:
        avg_matrix (ndarray): 재정렬할 평균 행렬 (예: avg_aec_matrix).
        epoch_vs_data (list or array): 채널 정보를 포함하는 데이터 리스트.
            첫 번째 요소의 'vs_time_series'의 키가 원본 채널 이름 리스트로 사용됩니다.
        sort_func (function): 채널 이름을 정렬할 때 사용할 함수.
            예: sort_by_roi_and_hemisphere
        
    Returns:
        sorted_matrix (ndarray): 채널이 정렬된 평균 행렬.
        sorted_channel_names (list): 정렬된 채널 이름 리스트.
    """
    # 원본 채널 이름 리스트 생성
    original_channel_names = list(epoch_vs_data[0]['vs_time_series'].keys())
    # 정렬 기준 함수에 따라 채널 이름 정렬
    sorted_channel_names = sorted(original_channel_names, key=sort_func)
    # 정렬된 채널 순서에 따른 인덱스 매핑 생성
    index_mapping = [original_channel_names.index(ch) for ch in sorted_channel_names]
    # 인덱스 매핑을 이용하여 행렬 재정렬
    sorted_matrix = avg_matrix[np.ix_(index_mapping, index_mapping)]
    
    return sorted_matrix, sorted_channel_names


# ================== Visualization Functions ==================

def plot_correlation_matrix(matrix, title='Pearson Correlation Matrix',
                           xlabel='Channel index', ylabel='Channel index',
                           color_label='Pearson correlation', figsize=(10, 8),
                           aspect='auto', origin='lower', interpolation='none'):
    """
    주어진 상관 행렬을 시각화하는 함수입니다.
    
    Parameters:
        matrix (ndarray): 시각화할 상관 행렬.
        title (str): 그래프 제목.
        xlabel (str): x축 레이블.
        ylabel (str): y축 레이블.
        color_label (str): 컬러바 라벨.
        figsize (tuple): 그림 크기.
        aspect (str): imshow의 aspect 설정.
        origin (str): imshow의 origin 설정.
        interpolation (str): imshow의 interpolation 설정.
    """
    plt.figure(figsize=figsize)
    plt.imshow(matrix, aspect=aspect, origin=origin, interpolation=interpolation)
    plt.colorbar(label=color_label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()
    
    return plt.gcf()


def plot_aec_heatmap(conn_matrix, title="Average AEC Matrix"):
    """
    AEC 연결성 행렬을 히트맵으로 시각화
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(conn_matrix, annot=False, cmap='viridis')
    plt.title(title)
    plt.xlabel('Virtual Sensors')
    plt.ylabel('Virtual Sensors')
    plt.show()
    
    return plt.gcf()


def plot_hemisphere_first_aec_heatmap(conn_matrix, sorted_channel_names, title="Average AEC Matrix by Hemisphere and Lobes"):
    """
    반구 우선(lh 전체 -> rh 전체) 방식으로 정렬된 AEC 연결성 행렬 시각화
    """
    # 채널별 반구 및 뇌엽 정보 추출
    channel_info = []
    for ch_name in sorted_channel_names:
        parts = ch_name.split('_')
        roi_name = parts[0].split('-')[0]  # "bankssts-lh" -> "bankssts"
        hemi = parts[-1]  # "lh" 또는 "rh"
        
        # ROI의 뇌엽 찾기
        lobe = roi_to_lobe.get(roi_name, "Other")
        channel_info.append((hemi, lobe, roi_name))
    
    # 반구 먼저, 그 다음 뇌엽별로 정렬
    sorted_indices = sorted(range(len(channel_info)), 
                        key=lambda i: (
                            0 if channel_info[i][0] == 'lh' else 1,  # lh 먼저, rh 나중에
                            lobe_priority.get(channel_info[i][1], 999),  # 뇌엽 우선순위
                            roi_priority.get(channel_info[i][2], 999)  # ROI 우선순위
                        ))
    
    # 새 순서로 행렬과 채널 이름 재정렬
    reordered_matrix = conn_matrix[np.ix_(sorted_indices, sorted_indices)]
    reordered_names = [sorted_channel_names[i] for i in sorted_indices]
    
    # 경계 및 레이블 계산
    boundaries = []
    labels = []
    
    # 반구 경계 먼저 찾기
    hemi_boundary = None
    for i, idx in enumerate(sorted_indices):
        if channel_info[idx][0] == 'rh' and hemi_boundary is None:
            hemi_boundary = i
            break
    
    if hemi_boundary:
        boundaries.append(hemi_boundary)
        
    # 각 반구 내의 뇌엽 경계 찾기
    current_hemi = None
    current_lobe = None
    
    for i, idx in enumerate(sorted_indices):
        hemi, lobe, _ = channel_info[idx]
        
        # 반구가 바뀌면 새 섹션 시작
        if hemi != current_hemi:
            current_hemi = hemi
            current_lobe = lobe
            label = f"{current_hemi.upper()}-{lobe}"
            labels.append(label)
            # 첫 번째 경계는 이미 추가했으므로 건너뛰기
            if i > 0 and i not in boundaries:
                boundaries.append(i)
        # 같은 반구 내에서 뇌엽이 바뀌면 새 섹션 시작
        elif lobe != current_lobe:
            current_lobe = lobe
            label = f"{current_hemi.upper()}-{lobe}"
            labels.append(label)
            boundaries.append(i)
    
    # 그림 생성
    plt.figure(figsize=(16, 14))
    
    # 히트맵 그리기
    ax = sns.heatmap(reordered_matrix, cmap='viridis', 
                   xticklabels=False, yticklabels=False,
                   cbar_kws={'label': 'Connectivity Strength'})
    
    # 반구 경계에 굵은 선 추가
    if hemi_boundary:
        plt.axhline(y=hemi_boundary, color='white', linestyle='-', linewidth=2.5)
        plt.axvline(x=hemi_boundary, color='white', linestyle='-', linewidth=2.5)
    
    # 뇌엽 경계에 선 추가
    for boundary in boundaries:
        if boundary != hemi_boundary:  # 반구 경계는 이미 처리함
            plt.axhline(y=boundary, color='red', linestyle='-', linewidth=1)
            plt.axvline(x=boundary, color='red', linestyle='-', linewidth=1)
    
    # 뇌엽 레이블 위치 계산
    boundaries = [0] + boundaries + [len(reordered_names)]  # 시작과 끝 추가
    boundaries = sorted(list(set(boundaries)))  # 중복 제거 및 정렬
    
    label_positions = []
    for i in range(len(boundaries)-1):
        start = boundaries[i]
        end = boundaries[i+1]
        mid = (start + end) // 2
        label_positions.append(mid)
    
    # 레이블 생성
    if len(label_positions) > len(labels):
        labels = ['LH'] + labels  # 첫 번째 섹션 레이블 추가
    
    plt.yticks(label_positions, labels, fontsize=10)
    plt.xticks(label_positions, labels, rotation=90, fontsize=10)
    
    plt.title(title, fontsize=16)
    plt.tight_layout()
    
    return plt.gcf(), reordered_matrix, reordered_names


# ================== Network Analysis Functions ==================

def compute_network_centrality(fc_matrix, hub_threshold_percentile=90, eps=1e-8):
    """
    Compute network centrality measures from a functional connectivity matrix.
    
    Parameters:
    -----------
    fc_matrix : ndarray, shape (n_channels, n_channels)
        Functional connectivity matrix (AEC, PLV, or correlation)
    hub_threshold_percentile : float
        Percentile threshold for defining hubs (default: 90, i.e., top 10%)
    eps : float
        Small value to avoid division by zero (default: 1e-8)
        
    Returns:
    --------
    dict
        Dictionary containing:
        - 'MST': Minimum spanning tree graph
        - 'betweenness': Normalized betweenness centrality for each node
        - 'closeness': Normalized closeness centrality for each node
        - 'degree': Normalized degree for each node
        - 'eigenvector': Normalized eigenvector centrality for each node
        - 'hubs': List of identified hub nodes
        - 'hub_indices': Dictionary mapping measure types to hub indices
    """
    # Make a copy of the connectivity matrix to avoid modifying the original
    fc = fc_matrix.copy()
    
    # Get dimensions
    n = fc.shape[0]
    
    # Compute inverted FC (add epsilon to avoid division by zero)
    inv_fc = 1.0 / (fc + eps)
    
    # Force symmetry in case there are numerical inconsistencies
    inv_fc = 0.5 * (inv_fc + inv_fc.T)
    
    # Set diagonal to zero (no self-connections in MST)
    np.fill_diagonal(inv_fc, 0)
    
    # Create full graph from inverted FC
    G_full = nx.Graph()
    G_full.add_nodes_from(range(n))
    
    # Add edges with inverted FC as weights
    for i in range(n):
        for j in range(i+1, n):
            weight = inv_fc[i, j]
            if np.isfinite(weight) and weight > 0:
                G_full.add_edge(i, j, weight=weight)
    
    # Compute Minimum Spanning Tree
    MST = nx.minimum_spanning_tree(G_full, weight='weight')
    
    # Verify the MST has n-1 edges
    if len(MST.edges()) != n-1:
        print(f"Warning: MST has {len(MST.edges())} edges, expected {n-1}")
    
    # Compute centrality measures
    betweenness = nx.betweenness_centrality(MST, weight='weight', normalized=True)
    closeness = nx.closeness_centrality(MST, distance='weight')
    degree = dict(MST.degree())
    
    # For eigenvector centrality, handle potential convergence issues
    try:
        eigenvector = nx.eigenvector_centrality_numpy(MST, weight='weight')
    except:
        print("Warning: Eigenvector centrality calculation failed, using uniform values")
        eigenvector = {node: 1.0/n for node in MST.nodes()}
    
    # Normalize each measure by its maximum value
    def normalize_dict(d):
        max_val = max(d.values()) if d.values() else 1.0
        if max_val == 0:
            return d
        return {k: v / max_val for k, v in d.items()}
    
    betweenness_norm = normalize_dict(betweenness)
    closeness_norm = normalize_dict(closeness)
    degree_norm = normalize_dict(degree)
    eigenvector_norm = normalize_dict(eigenvector)
    
    # Identify hubs based on each centrality measure
    hub_indices = {
        'betweenness': identify_hubs(betweenness_norm, hub_threshold_percentile),
        'closeness': identify_hubs(closeness_norm, hub_threshold_percentile),
        'degree': identify_hubs(degree_norm, hub_threshold_percentile),
        'eigenvector': identify_hubs(eigenvector_norm, hub_threshold_percentile)
    }
    
    # Default hubs based on eigenvector centrality
    hubs = hub_indices['eigenvector']
    
    return {
        'MST': MST,
        'betweenness': betweenness_norm,
        'closeness': closeness_norm,
        'degree': degree_norm,
        'eigenvector': eigenvector_norm,
        'hubs': hubs,
        'hub_indices': hub_indices
    }


def identify_hubs(centrality_dict, percentile_threshold=90):
    """
    Identify hub nodes based on centrality values and a percentile threshold.
    
    Parameters:
    -----------
    centrality_dict : dict
        Dictionary mapping node indices to centrality values
    percentile_threshold : float
        Percentile threshold for identifying hubs (default: 90)
        
    Returns:
    --------
    list
        List of hub node indices
    """
    values = np.array(list(centrality_dict.values()))
    threshold = np.percentile(values, percentile_threshold)
    hubs = [node for node, val in centrality_dict.items() if val >= threshold]
    return hubs


def get_hub_info(hubs, virtual_sensors):
    """
    Get anatomical information for identified hub nodes.
    
    Parameters:
    -----------
    hubs : list
        List of hub node indices
    virtual_sensors : list
        List of virtual sensor information dictionaries
        
    Returns:
    --------
    list
        List of dictionaries with hub information
    """
    hub_info = []
    for hub in hubs:
        vs_info = virtual_sensors[hub]
        hub_info.append({
            'node_idx': hub,
            'label': vs_info['label'],
            'hemisphere': vs_info['hemi'],
            'cluster_id': vs_info['cluster_id'],
            'coordinates': vs_info['coord']
        })
    return hub_info


def analyze_hub_distribution(hub_info):
    """
    Analyze the distribution of hubs across hemispheres and anatomical regions.
    
    Parameters:
    -----------
    hub_info : list
        List of dictionaries with hub information
        
    Returns:
    --------
    dict
        Dictionary with hub distribution statistics
    """
    # Count hubs by hemisphere
    hemi_count = {'lh': 0, 'rh': 0}
    for hub in hub_info:
        hemi_count[hub['hemisphere']] += 1
    
    # Count hubs by anatomical region
    region_count = {}
    for hub in hub_info:
        region = hub['label'].split('-')[0]  # Extract base region name
        if region not in region_count:
            region_count[region] = 0
        region_count[region] += 1
    
    # Sort regions by hub count (descending)
    sorted_regions = sorted(region_count.items(), key=lambda x: x[1], reverse=True)
    
    return {
        'hemisphere_distribution': hemi_count,
        'region_count': region_count,
        'top_regions': sorted_regions[:5]  # Top 5 regions
    }


def compare_centrality_measures(centrality_results):
    """
    Compare the agreement between different centrality measures.
    
    Parameters:
    -----------
    centrality_results : dict
        Dictionary from compute_network_centrality function
        
    Returns:
    --------
    dict
        Dictionary with overlap statistics between measures
    """
    hub_indices = centrality_results['hub_indices']
    measures = list(hub_indices.keys())
    overlap = {}
    
    # Compare each pair of measures
    for i, m1 in enumerate(measures):
        for j, m2 in enumerate(measures):
            if j <= i:
                continue
            
            hubs1 = set(hub_indices[m1])
            hubs2 = set(hub_indices[m2])
            
            # Calculate Jaccard similarity coefficient
            intersection = len(hubs1.intersection(hubs2))
            union = len(hubs1.union(hubs2))
            
            if union > 0:
                jaccard = intersection / union
            else:
                jaccard = 0
                
            overlap[f"{m1}_vs_{m2}"] = {
                'jaccard': jaccard,
                'shared_hubs': intersection,
                'total_unique_hubs': union
            }
    
    return overlap


def compute_edge_betweenness(MST):
    """
    Compute edge betweenness centrality for MST edges.
    
    Parameters:
    -----------
    MST : networkx.Graph
        Minimum spanning tree graph
        
    Returns:
    --------
    dict
        Dictionary mapping edge tuples to their betweenness values
    """
    edge_betweenness = nx.edge_betweenness_centrality(MST, weight='weight', normalized=True)
    
    # Normalize by maximum value
    max_val = max(edge_betweenness.values()) if edge_betweenness.values() else 1.0
    if max_val > 0:
        edge_betweenness = {edge: val/max_val for edge, val in edge_betweenness.items()}
    
    return edge_betweenness


def generate_network_report(centrality_results, virtual_sensors=None, frequency_band=None):
    """
    Generate a comprehensive report of network analysis results.
    
    Parameters:
    -----------
    centrality_results : dict
        Dictionary from compute_network_centrality function
    virtual_sensors : list, optional
        List of virtual sensor information dictionaries
    frequency_band : str, optional
        Name of the frequency band being analyzed
        
    Returns:
    --------
    dict
        Dictionary with report information
    """
    MST = centrality_results['MST']
    n_nodes = len(MST.nodes())
    n_edges = len(MST.edges())
    
    # Basic MST statistics
    mst_stats = {
        'n_nodes': n_nodes,
        'n_edges': n_edges,
        'max_degree': max(dict(MST.degree()).values()),
        'diameter': nx.diameter(MST, weight='weight')
    }
    
    # Hub statistics
    all_hubs = set()
    for hubs in centrality_results['hub_indices'].values():
        all_hubs.update(hubs)
    
    hub_stats = {
        'n_hubs': len(centrality_results['hubs']),
        'hub_percentage': 100 * len(centrality_results['hubs']) / n_nodes,
        'n_unique_hubs_all_measures': len(all_hubs),
        'unique_hub_percentage': 100 * len(all_hubs) / n_nodes
    }
    
    # Compare centrality measures
    measure_overlap = compare_centrality_measures(centrality_results)
    
    # Anatomical distribution if virtual sensors are provided
    anatomy_stats = None
    if virtual_sensors:
        hub_info = get_hub_info(centrality_results['hubs'], virtual_sensors)
        anatomy_stats = analyze_hub_distribution(hub_info)
    
    report = {
        'frequency_band': frequency_band,
        'mst_statistics': mst_stats,
        'hub_statistics': hub_stats,
        'measure_overlap': measure_overlap,
        'anatomical_distribution': anatomy_stats
    }
    
    return report


def plot_mst_with_hubs(centrality_results, node_names=None, measure='eigenvector', 
                      title='Minimum Spanning Tree with Hubs', figsize=(12, 10),
                      hub_color='red', node_color='lightblue', 
                      hub_size=50, node_size=10, edge_color='gray',
                      with_labels=False, font_size=8):
    """
    Plot the MST with hub nodes highlighted.
    
    Parameters:
    -----------
    centrality_results : dict
        Dictionary from compute_network_centrality function
    node_names : list, optional
        List of node names or labels
    measure : str
        Centrality measure to use for identifying hubs ('betweenness', 'closeness', 'degree', or 'eigenvector')
    title : str
        Plot title
    figsize : tuple
        Figure size
    hub_color : str
        Color for hub nodes
    node_color : str
        Color for non-hub nodes
    hub_size : int
        Size for hub nodes
    node_size : int
        Size for non-hub nodes
    edge_color : str
        Color for edges
    with_labels : bool
        Whether to display node labels
    font_size : int
        Font size for node labels
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    MST = centrality_results['MST']
    
    # Use the specified measure to determine hubs
    if measure in centrality_results['hub_indices']:
        hubs = centrality_results['hub_indices'][measure]
    else:
        hubs = centrality_results['hubs']
    
    # Spring layout with fixed seed for reproducibility
    pos = nx.spring_layout(MST, seed=42)
    
    # Set node colors and sizes
    node_colors = []
    node_sizes = []
    
    for node in MST.nodes():
        if node in hubs:
            node_colors.append(hub_color)
            node_sizes.append(hub_size)
        else:
            node_colors.append(node_color)
            node_sizes.append(node_size)
    
    # Create plot
    plt.figure(figsize=figsize)
    
    # If node names are provided and labels should be displayed
    if with_labels and node_names:
        # Trim node names to avoid overcrowding
        short_names = [name[:15] + '...' if len(name) > 15 else name for name in node_names]
        labels = {i: name for i, name in enumerate(short_names)}
        nx.draw_networkx(MST, pos=pos, with_labels=True, labels=labels,
                        node_color=node_colors, node_size=node_sizes, 
                        edge_color=edge_color, alpha=0.7, font_size=font_size)
    else:
        nx.draw_networkx(MST, pos=pos, with_labels=False,
                        node_color=node_colors, node_size=node_sizes, 
                        edge_color=edge_color, alpha=0.7)
    
    # Edge betweenness-based edge width
    edge_betweenness = compute_edge_betweenness(MST)
    edge_width = [2 + 3 * edge_betweenness.get((u, v), 0) for u, v in MST.edges()]
    
    nx.draw_networkx_edges(MST, pos, width=edge_width, edge_color=edge_color, alpha=0.5)
    
    plt.title(f"{title} ({measure.capitalize()} Centrality)")
    plt.axis('off')
    
    return plt.gcf()


def plot_3d_hubs(hub_indices, virtual_sensors, measure='eigenvector', 
                all_vs_color='lightgray', hub_vs_color='red',
                all_vs_size=5, hub_vs_size=20, figsize=(10, 8)):
    """
    Create a 3D plot showing the locations of virtual sensors with hubs highlighted.
    
    Parameters:
    -----------
    hub_indices : dict
        Dictionary mapping measure types to hub indices
    virtual_sensors : list
        List of virtual sensor information dictionaries
    measure : str
        Centrality measure to use for identifying hubs
    all_vs_color : str
        Color for all virtual sensors
    hub_vs_color : str
        Color for hub virtual sensors
    all_vs_size : int
        Size for all virtual sensors
    hub_vs_size : int
        Size for hub virtual sensors
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    # Extract coordinates
    all_coords = np.array([vs['coord'] for vs in virtual_sensors])
    
    # Get hub indices for the specified measure
    if measure in hub_indices:
        hubs = hub_indices[measure]
    else:
        # Default to eigenvector centrality
        hubs = hub_indices['eigenvector']
    
    # Extract hub coordinates
    hub_coords = np.array([virtual_sensors[node]['coord'] for node in hubs])
    
    # Create 3D plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot all virtual sensors
    ax.scatter(all_coords[:, 0], all_coords[:, 1], all_coords[:, 2],
              c=all_vs_color, s=all_vs_size, label='All Virtual Sensors')
    
    # Plot hub virtual sensors
    ax.scatter(hub_coords[:, 0], hub_coords[:, 1], hub_coords[:, 2],
              c=hub_vs_color, s=hub_vs_size, label='Hub Virtual Sensors')
    
    ax.set_title(f"Hub Virtual Sensors Positions ({measure.capitalize()} Centrality)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend()
    
    return fig


def plot_centrality_heatmap(centrality_dict, node_names=None, title="Centrality Measures",
                           cmap="viridis", figsize=(10, 8)):
    """
    Plot a heatmap of centrality values.
    
    Parameters:
    -----------
    centrality_dict : dict
        Dictionary of centrality measures from compute_network_centrality
    node_names : list, optional
        List of node names or labels
    title : str
        Plot title
    cmap : str
        Colormap name
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    # Extract centrality measures
    measures = ['betweenness', 'closeness', 'degree', 'eigenvector']
    valid_measures = [m for m in measures if m in centrality_dict]
    
    if not valid_measures:
        print("No valid centrality measures found")
        return None
    
    # Create data matrix
    n_nodes = len(centrality_dict[valid_measures[0]])
    data = np.zeros((n_nodes, len(valid_measures)))
    
    for i, measure in enumerate(valid_measures):
        values = centrality_dict[measure]
        for node, val in values.items():
            data[node, i] = val
    
    # Create x and y axis labels
    x_labels = [m.capitalize() for m in valid_measures]
    
    if node_names:
        # Truncate node names if they're too long
        y_labels = [name[:20] + '...' if len(name) > 20 else name for name in node_names]
    else:
        y_labels = [f"Node {i}" for i in range(n_nodes)]
    
    # Create heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(data, xticklabels=x_labels, yticklabels=y_labels, cmap=cmap)
    plt.title(title)
    plt.tight_layout()
    
    return plt.gcf()


def analyze_frequency_bands(fc_matrices, freq_bands, virtual_sensors=None, hub_threshold_percentile=90):
    """
    Analyze network properties across multiple frequency bands.
    
    Parameters:
    -----------
    fc_matrices : dict
        Dictionary mapping frequency band names to FC matrices
    freq_bands : list
        List of frequency band names
    virtual_sensors : list, optional
        List of virtual sensor information dictionaries
    hub_threshold_percentile : float
        Percentile threshold for defining hubs
        
    Returns:
    --------
    dict
        Dictionary with analysis results for each frequency band
    """
    results = {}
    
    for band in freq_bands:
        if band in fc_matrices:
            print(f"Analyzing {band} band...")
            # Compute network centrality
            centrality_results = compute_network_centrality(
                fc_matrices[band], 
                hub_threshold_percentile=hub_threshold_percentile
            )
            
            # Generate report
            report = generate_network_report(
                centrality_results,
                virtual_sensors=virtual_sensors,
                frequency_band=band
            )
            
            results[band] = {
                'centrality': centrality_results,
                'report': report
            }
        else:
            print(f"Warning: {band} band not found in FC matrices")
    
    return results


def compare_bands_hub_overlap(results, freq_bands):
    """
    Compare hub overlap between different frequency bands.
    
    Parameters:
    -----------
    results : dict
        Dictionary with analysis results from analyze_frequency_bands
    freq_bands : list
        List of frequency band names
        
    Returns:
    --------
    dict
        Dictionary with band comparison statistics
    """
    overlap = {}
    
    # Compare each pair of bands
    for i, band1 in enumerate(freq_bands):
        if band1 not in results:
            continue
            
        for j, band2 in enumerate(freq_bands):
            if j <= i or band2 not in results:
                continue
            
            hubs1 = set(results[band1]['centrality']['hubs'])
            hubs2 = set(results[band2]['centrality']['hubs'])
            
            # Calculate Jaccard similarity coefficient
            intersection = len(hubs1.intersection(hubs2))
            union = len(hubs1.union(hubs2))
            
            if union > 0:
                jaccard = intersection / union
            else:
                jaccard = 0
                
            overlap[f"{band1}_vs_{band2}"] = {
                'jaccard': jaccard,
                'shared_hubs': intersection,
                'total_unique_hubs': union
            }
    
    return overlap


def create_comparison_dataframe(vs_ch_names, virtual_sensors, centrality_results, band_name='gamma'):
    """
    Create a DataFrame with channel information, coordinates, and centrality metrics.
    
    Parameters:
    -----------
    vs_ch_names : list
        List of channel names (virtual sensors)
    virtual_sensors : list of dict
        List of dictionaries containing information about each virtual sensor
    centrality_results : dict
        Dictionary containing centrality metrics (betweenness, closeness, degree, eigenvector)
    band_name : str
        Name of the frequency band for column labeling
        
    Returns:
    --------
    comparison_df : pandas.DataFrame
        DataFrame containing channel information and centrality metrics
    """
    # Create a DataFrame to store the comparison data
    comparison_df = pd.DataFrame()
    
    # Add virtual sensor information
    comparison_df['channel_name'] = vs_ch_names
    
    # Create a mapping from the original virtual_sensors
    vs_mapping = {f"{vs['label']}_cluster{vs['cluster_id']}_{vs['hemi']}": {
        'x': vs['coord'][0],
        'y': vs['coord'][1],
        'z': vs['coord'][2],
        'hemisphere': vs['hemi'],
        'cluster_id': vs['cluster_id']
    } for vs in virtual_sensors}
    
    # Add coordinate and other information based on channel name
    comparison_df['x_coord'] = comparison_df['channel_name'].map(lambda name: vs_mapping.get(name, {}).get('x', 0))
    comparison_df['y_coord'] = comparison_df['channel_name'].map(lambda name: vs_mapping.get(name, {}).get('y', 0))
    comparison_df['z_coord'] = comparison_df['channel_name'].map(lambda name: vs_mapping.get(name, {}).get('z', 0))
    comparison_df['hemisphere'] = comparison_df['channel_name'].map(lambda name: vs_mapping.get(name, {}).get('hemisphere', ''))
    comparison_df['cluster_id'] = comparison_df['channel_name'].map(lambda name: vs_mapping.get(name, {}).get('cluster_id', 0))
    
    # Add FC-based metrics
    # Centrality metrics - we need to convert the dictionary to match vs_ch_names
    centrality_dict = {
        'betweenness': centrality_results['betweenness'],
        'closeness': centrality_results['closeness'],
        'degree': centrality_results['degree'],
        'eigenvector': centrality_results['eigenvector']
    }
    
    # Match the dictionary keys (indices) to the channel names
    for metric, values in centrality_dict.items():
        comparison_df[f'{band_name}_{metric}'] = [values.get(i, 0) for i, name in enumerate(vs_ch_names)]
    
    return comparison_df


def filter_epochs_by_band(epochs_vs, band_name):
    """
    주파수 대역별로 epochs를 필터링하는 함수
    
    Parameters:
    -----------
    epochs_vs : mne.Epochs
        필터링할 원본 epochs
    band_name : str
        주파수 대역 이름 ('theta', 'alpha', 'beta', 'gamma' 중 하나)
        
    Returns:
    --------
    filtered_epochs : mne.Epochs
        주파수 대역으로 필터링된 epochs
    """
    # 각 주파수 대역 범위 정의
    band_ranges = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 80)
    }
    
    # 해당 주파수 대역이 정의되어 있는지 확인
    if band_name not in band_ranges:
        raise ValueError(f"Unknown frequency band: {band_name}. Available bands: {', '.join(band_ranges.keys())}")
    
    # 주파수 범위 가져오기
    l_freq, h_freq = band_ranges[band_name]
    
    # epochs 복사 및 필터링
    filtered_epochs = epochs_vs.copy().filter(l_freq=l_freq, h_freq=h_freq, fir_design='firwin')
    
    return filtered_epochs


def compute_and_save_connectivity(epochs_vs, band_name, method='aec', save_path=None):
    """
    주파수 대역별로 연결성을 계산하고 저장하는 함수
    
    Parameters:
    -----------
    epochs_vs : mne.Epochs
        연결성을 계산할 epochs
    band_name : str
        주파수 대역 이름 ('theta', 'alpha', 'beta', 'gamma' 중 하나)
    method : str
        연결성 계산 방법 ('aec', 'plv', 'pearson' 중 하나)
    save_path : str, optional
        연결성 행렬을 저장할 경로 (None이면 저장하지 않음)
        
    Returns:
    --------
    conn_matrix : ndarray
        계산된 연결성 행렬
    ch_names : list
        채널 이름 목록
    """
    # 주파수 대역으로 필터링
    filtered_epochs = filter_epochs_by_band(epochs_vs, band_name)
    
    # 연결성 계산
    if method.lower() == 'aec':
        conn_matrix, ch_names = compute_aec_connectivity(filtered_epochs)
    elif method.lower() == 'plv':
        # 주파수 대역 범위 가져오기
        band_ranges = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 80)
        }
        l_freq, h_freq = band_ranges[band_name]
        conn_matrix, ch_names = compute_plv_connectivity(filtered_epochs, fmin=l_freq, fmax=h_freq)
    elif method.lower() == 'pearson':
        conn_matrix = compute_pearson_correlation(filtered_epochs)
        ch_names = filtered_epochs.ch_names
    else:
        raise ValueError(f"Unknown connectivity method: {method}. Available methods: 'aec', 'plv', 'pearson'")
    
    # 결과 저장
    if save_path:
        np.save(save_path, conn_matrix)
        print(f"Connectivity matrix saved to {save_path}")
    
    return conn_matrix, ch_names


def pipeline_frequency_band_analysis(epochs_vs, epoch_vs_data, virtual_sensors, 
                                    bands=['theta', 'alpha', 'beta', 'gamma'],
                                    method='aec', output_dir=None, visualize=True):
    """
    여러 주파수 대역에 걸쳐 전체 분석 파이프라인을 실행하는 함수
    
    Parameters:
    -----------
    epochs_vs : mne.Epochs
        분석할 epochs
    epoch_vs_data : list
        채널 정보를 담고 있는 데이터 리스트
    virtual_sensors : list
        가상 센서 정보 리스트
    bands : list
        분석할 주파수 대역 리스트
    method : str
        연결성 계산 방법 ('aec', 'plv', 'pearson' 중 하나)
    output_dir : str, optional
        결과물을 저장할 디렉토리 (None이면 저장하지 않음)
    visualize : bool
        시각화 여부
        
    Returns:
    --------
    results : dict
        모든 분석 결과를 담고 있는 딕셔너리
    """
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    results = {}
    fc_matrices = {}
    
    for band in bands:
        print(f"\n=== Processing {band} band ===")
        
        # 1. 연결성 계산
        save_path = f"{output_dir}/{band}_{method}_matrix.npy" if output_dir else None
        conn_matrix, ch_names = compute_and_save_connectivity(epochs_vs, band, method, save_path)
        fc_matrices[band] = conn_matrix
        
        # 2. 행렬 정렬
        sorted_matrix, sorted_ch_names = sort_matrix_by_channels(conn_matrix, epoch_vs_data, sort_by_roi_and_hemisphere)
        
        # 3. 시각화
        if visualize:
            # 기본 히트맵
            fig1 = plot_aec_heatmap(conn_matrix, title=f"{band.capitalize()} {method.upper()} Matrix")
            
            # 반구 및 뇌엽별 히트맵
            fig2, reordered_matrix, reordered_names = plot_hemisphere_first_aec_heatmap(
                sorted_matrix, sorted_ch_names,
                title=f"{band.capitalize()} {method.upper()} Matrix by Hemisphere and Lobes"
            )
            
            if output_dir:
                fig1.savefig(f"{output_dir}/{band}_{method}_heatmap.png")
                fig2.savefig(f"{output_dir}/{band}_{method}_hemisphere_heatmap.png")
        
        # 4. 네트워크 분석
        centrality_results = compute_network_centrality(conn_matrix, hub_threshold_percentile=90)
        
        # 5. 네트워크 보고서 생성
        report = generate_network_report(centrality_results, virtual_sensors, band)
        
        # 6. 네트워크 시각화
        if visualize:
            # MST와 허브 그래프
            fig3 = plot_mst_with_hubs(centrality_results, ch_names, measure='eigenvector',
                                     title=f"{band.capitalize()} Band MST")
            
            # 3D 허브 위치
            fig4 = plot_3d_hubs(centrality_results['hub_indices'], virtual_sensors, measure='eigenvector')
            
            if output_dir:
                fig3.savefig(f"{output_dir}/{band}_{method}_mst_hubs.png")
                fig4.savefig(f"{output_dir}/{band}_{method}_3d_hubs.png")
        
        # 7. 비교 데이터프레임 생성
        comparison_df = create_comparison_dataframe(ch_names, virtual_sensors, centrality_results, band_name=band)
        
        if output_dir:
            comparison_df.to_csv(f"{output_dir}/{band}_{method}_metrics.csv", index=False)
        
        # 결과 저장
        results[band] = {
            'matrix': conn_matrix,
            'sorted_matrix': sorted_matrix,
            'channel_names': ch_names,
            'sorted_channel_names': sorted_ch_names,
            'centrality_results': centrality_results,
            'report': report,
            'comparison_df': comparison_df
        }
    
    # 밴드 간 허브 비교
    if len(bands) > 1:
        band_results = {band: results[band]['centrality_results'] for band in bands}
        band_overlap = compare_bands_hub_overlap(band_results, bands)
        results['band_comparison'] = band_overlap
        
        if output_dir:
            with open(f"{output_dir}/band_comparison.txt", 'w') as f:
                for comparison, stats in band_overlap.items():
                    f.write(f"{comparison}:\n")
                    f.write(f"  Jaccard similarity: {stats['jaccard']:.3f}\n")
                    f.write(f"  Shared hubs: {stats['shared_hubs']}\n")
                    f.write(f"  Total unique hubs: {stats['total_unique_hubs']}\n\n")
    
    return results

#%%
# 메인 실행 함수
def main():
    """
    메인 실행 함수
    """
    # 예시 실행 코드

    # 1. epochs_vs 및 virtual_sensors 로드
    
    # 2. 특정 주파수 대역 에포크 생성
    epochs_gamma = filter_epochs_by_band(epochs_vs, 'gamma')
    
    # 3. AEC 계산
    # avg_aec_matrix, vs_ch_names = compute_aec_connectivity(epochs_gamma)
    vs_ch_names = epochs.ch_names

    # 4. PLV 계산
    # avg_plv_matrix, beta_ch_names = compute_plv_connectivity(epochs_gamma, fmin=30, fmax=80)
    avg_aec_matrix = compute_pearson_correlation(epochs_gamma)
    # 5. 행렬 정렬
    sorted_matrix, sorted_channel_names = sort_matrix_by_channels(avg_aec_matrix, epoch_vs_data, sort_by_roi_and_hemisphere)
    
    # 6. 히트맵 시각화
    fig, grouped_matrix, grouped_names = plot_hemisphere_first_aec_heatmap(
        sorted_matrix, sorted_channel_names,
        title="Average AEC Matrix"
    )
    
    # 7. 네트워크 중심성 계산
    centrality_results = compute_network_centrality(avg_aec_matrix, hub_threshold_percentile=90)
    
    # 8. 네트워크 보고서 생성
    report = generate_network_report(centrality_results, virtual_sensors, "Gamma")
    
    # 9. 데이터프레임 생성 및 저장
    comparison_df = create_comparison_dataframe(vs_ch_names, virtual_sensors, centrality_results)
    # comparison_df.to_csv('connectivity_metrics.csv', index=False)
    # 10. 전체 파이프라인 실행
    results = pipeline_frequency_band_analysis(
        epochs_vs=epochs_vs,
        epoch_vs_data=epoch_vs_data,
        virtual_sensors=virtual_sensors,
        bands=['alpha', 'beta', 'gamma'],
        method='aec',
        output_dir='output',
        visualize=True
    )

    pass


if __name__ == "__main__":
    main()
# %%
# 1. epochs_vs 및 virtual_sensors 로드
# epochs_gamma = filter_epochs_by_band(epochs_vs, 'gamma')
# vs_ch_names = epochs_gamma.ch_names

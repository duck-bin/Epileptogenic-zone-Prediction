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
#NNMF일 때는 60Hz notch 하모닉스 filtering하고 80Hz 이상도 살림.
def apply_filters(raw):
    """필터링을 적용하는 함수"""
    # 1. 60Hz 노치 필터 적용
    #    전력선 간섭을 제거하기 위해 60Hz 주파수를 제거합니다.
    freqs = np.arange(60, 241, 60)  # 60Hz와 그 하모닉스 (60, 120, 180, 240Hz)
    raw.notch_filter(freqs=freqs, picks='eeg')

    # 2. 밴드패스 필터 적용 (1 ~ 70 Hz)
    #    저주파 드리프트(1Hz 이하)와 고주파 잡음(70Hz 이상)을 제거합니다.
    # raw.filter(l_freq=l_freq, h_freq=h_freq, picks='eeg')
    
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
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from sklearn.decomposition import NMF, PCA
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, cdist
import plotly.graph_objects as go
import pandas as pd
from scipy.spatial import ConvexHull
import gc  # For garbage collection


# ===================== 데이터 준비 함수 =====================

def create_segment_data(epochs_vs):
    """
    여러 에포크를 연결하여 하나의 segment 데이터를 생성합니다.
    
    Parameters:
    -----------
    epochs_vs : MNE EpochsArray
        형태: (n_epochs, n_channels, n_samples)
    
    Returns:
    --------
    segment_data : np.ndarray
        연결된 데이터 (n_channels, n_samples_total)
    ch_names : list
        채널 이름 목록
    sfreq : float
        샘플링 주파수
    """
    data = epochs_vs.get_data()
    ch_names = epochs_vs.ch_names
    sfreq = epochs_vs.info['sfreq']
    
    n_epochs, n_channels, n_samples_per_epoch = data.shape
    segment_data = np.zeros((n_channels, n_epochs * n_samples_per_epoch))
    for i in range(n_epochs):
        segment_data[:, i*n_samples_per_epoch:(i+1)*n_samples_per_epoch] = data[i, :, :]
    
    return segment_data, ch_names, sfreq


def extract_coordinates(ch_names, virtual_sensors):
    """
    가상 센서 좌표를 채널명에 맞게 추출합니다.
    
    Parameters:
    -----------
    ch_names : list
        채널 이름 목록
    virtual_sensors : list
        가상 센서 정보가 담긴 딕셔너리 리스트
    
    Returns:
    --------
    coordinates : array
        좌표 배열 (n_channels, 3)
    """
    coordinates = np.zeros((len(ch_names), 3))
    
    # 채널명에 맞는 가상 센서 찾기
    for i, ch_name in enumerate(ch_names):
        found = False
        for vs in virtual_sensors:
            vs_name = f"{vs['label']}_cluster{vs['cluster_id']}_{vs['hemi']}"
            if vs_name == ch_name:
                coordinates[i] = vs['coord']
                found = True
                break
        if not found:
            print(f"Warning: No coordinates found for channel {ch_name}")
    
    print(f"Coordinates extracted. Shape: {coordinates.shape}")
    
    return coordinates


# ===================== 신호 처리 함수 =====================

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    버터워스 대역통과 필터를 적용합니다.
    
    Parameters:
    -----------
    data : array
        입력 신호
    lowcut : float
        하한 주파수
    highcut : float
        상한 주파수
    fs : float
        샘플링 주파수
    order : int
        필터 차수
        
    Returns:
    --------
    filtered_data : array
        필터링된 신호
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    
    # 유효한 주파수 범위로 제한
    low = max(0.001, min(0.999, low))
    high = max(0.001, min(0.999, high))
    
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)


def filter_data(segment_data, sfreq):
    """
    스파이크 밴드와 리플 밴드로 데이터를 필터링합니다.
    
    Parameters:
    -----------
    segment_data : array
        입력 데이터 (n_channels, n_samples)
    sfreq : float
        샘플링 주파수
    
    Returns:
    --------
    spike_band_data : array
        스파이크 밴드로 필터링된 데이터 (1-80 Hz)
    ripple_band_data : array
        리플 밴드로 필터링된 데이터 (80-250 Hz)
    """
    n_channels, n_samples = segment_data.shape
    
    # 스파이크 밴드 필터링 (1-80 Hz)
    spike_band_data = np.zeros_like(segment_data)
    for ch in range(n_channels):
        spike_band_data[ch] = butter_bandpass_filter(segment_data[ch], 1, 80, sfreq)
    
    # 나이퀴스트 주파수에 맞춰 최대 리플 주파수 제한
    max_ripple_freq = min(250, sfreq/2 - 1)
    
    # 리플 밴드 필터링 (80-250 Hz)
    ripple_band_data = np.zeros_like(segment_data)
    for ch in range(n_channels):
        ripple_band_data[ch] = butter_bandpass_filter(segment_data[ch], 80, max_ripple_freq, sfreq)
    
    print("Bandpass filtering complete.")
    
    return spike_band_data, ripple_band_data

#PCA인데 사용하지 않을 것것
def reduce_channels(data, n_components=200):
    """
    PCA를 사용하여 채널 수를 줄입니다.
    
    Parameters:
    -----------
    data : array
        입력 데이터 (n_channels, n_samples)
    n_components : int
        유지할 성분 수
        
    Returns:
    --------
    data_reduced : array
        차원이 축소된 데이터 (n_components, n_samples)
    pca : PCA object
        학습된 PCA 모델
    """
    n_channels, n_samples = data.shape
    
    # PCA 적용을 위한 데이터 변형
    data_for_pca = data.T  # (n_samples, n_channels)
    
    # PCA 적용
    pca = PCA(n_components=n_components)
    data_reduced = pca.fit_transform(data_for_pca)  # (n_samples, n_components)
    
    # 변형
    data_reduced = data_reduced.T  # (n_components, n_samples)
    
    print(f"Channel reduction: {n_channels} -> {n_components}")
    
    return data_reduced, pca


# ===================== DMD 분석 함수 =====================
#DMD
def dmd_full(X, dt=1, r=None, h=None):
    """
    시간 지연 임베딩을 사용한 Dynamic Mode Decomposition
    다양한 크기의 윈도우를 처리할 수 있도록 수정됨
    
    Parameters:
    -----------
    X : array
        데이터 행렬 (n_channels, n_samples)
    dt : float
        시간 간격
    r : int
        축소할 랭크
    h : int
        시간 지연 임베딩 수
        
    Returns:
    --------
    Phi : array
        DMD 모드
    omega : array
        연속 시간 고유값
    w : array
        이산 시간 고유값
    b : array
        모드 진폭
    """
    n, m = X.shape
    
    # 창 크기가 요청된 임베딩에 충분한지 확인
    if h is not None and h > 1:
        # 샘플 수보다 더 많은 임베딩 시간을 사용하지 않도록 함
        if m < h:
            print(f"Warning: Window size {m} is smaller than embedding size {h}. Reducing embedding to {m}.")
            h = m
        
        # 사용 가능한 데이터에 맞게 임베딩 조정
        if m - h + 1 <= 0:
            h = m - 1  # 임베딩 후에도 최소 하나의 열을 보장
            if h <= 0:
                h = 1  # 창이 너무 작으면 임베딩 없이 진행
        
        X_aug = np.zeros((n * h, m - h + 1))
        for i in range(h):
            X_aug[i*n:(i+1)*n, :] = X[:, i:i+m-h+1]
        X = X_aug
        n, m = X.shape  # 차원 업데이트
    
    # X1과 X2를 위해 최소 2개의 열이 필요
    if m < 2:
        print(f"Warning: Not enough samples ({m}) after embedding. Returning zeros.")
        if r is None:
            r = min(n, 10)  # r이 None인 경우 기본값
        return np.zeros((n, r), dtype=complex), np.zeros(r, dtype=complex), np.zeros(r, dtype=complex), np.zeros(r, dtype=complex)
    
    # 데이터 행렬 분할
    X1 = X[:, :-1]
    X2 = X[:, 1:]
    
    # SVD 계산
    try:
        U, S, Vh = np.linalg.svd(X1, full_matrices=False)
        
        # 랭크 r까지 자르기
        if r is None:
            r = len(S)  # r이 지정되지 않으면 전체 랭크 사용
        
        r = min(r, len(S))  # r이 사용 가능한 특이값보다 크지 않도록 함
        
        U_r = U[:, :r]
        S_r = S[:r]
        V_r = Vh[:r, :]
        
        # DMD 행렬 계산
        # 0으로 나누는 것을 방지하기 위한 작은 epsilon 추가
        epsilon = 1e-10
        Atilde = U_r.T @ X2 @ V_r.T @ np.diag(1/(S_r + epsilon))
        
        # Atilde의 고유값 분해
        w, v = np.linalg.eig(Atilde)
        
        # DMD 모드 계산
        Phi = X2 @ V_r.T @ np.diag(1/(S_r + epsilon)) @ v
        
        # 연속 시간 고유값
        omega = np.log(w) / dt
        
        # 모드 진폭 계산
        x1 = X[:, 0]
        b = np.linalg.lstsq(Phi, x1, rcond=None)[0]
        
        # 원래 차원으로 복원 (시간 지연 임베딩을 위해)
        if h is not None and h > 1:
            Phi_original = Phi[:n//h, :]
        else:
            Phi_original = Phi
        
        return Phi_original, omega, w, b
    
    except Exception as e:
        print(f"SVD failed: {e}. Returning zeros.")
        if r is None:
            r = min(n, 10)  # r이 None인 경우 기본값
        return np.zeros((n, r), dtype=complex), np.zeros(r, dtype=complex), np.zeros(r, dtype=complex), np.zeros(r, dtype=complex)


def extract_features_batch(data, window_size_ms, overlap_perc, fs, r_input, batch_size=100):
    """
    슬라이딩 윈도우 접근법을 사용하여 DMD 특성을 추출합니다.
    
    Parameters:
    -----------
    data : array
        입력 데이터 (n_channels, n_samples)
    window_size_ms : float
        윈도우 크기 (밀리초)
    overlap_perc : float
        윈도우 겹침 비율 (0-1)
    fs : float
        샘플링 주파수
    r_input : int
        추출할 DMD 모드 수
    batch_size : int
        배치로 처리할 윈도우 수
        
    Returns:
    --------
    dmd_features : array
        DMD 특성 (n_channels, r_input, n_windows)
    freq_mean : array
        DMD 모드의 평균 주파수
    """
    n_channels, n_samples = data.shape
    
    # 윈도우 크기 계산 (샘플 단위)
    window_size = int(window_size_ms * fs / 1000)
    
    # 겹침 비율에 따른 스텝 크기 계산
    step_size = int(window_size * (1 - overlap_perc))
    step_size = max(1, step_size)  # 스텝 크기는 최소 1
    
    # 윈도우 수 계산
    n_windows = (n_samples - window_size) // step_size + 1
    
    # 시간 지연 임베딩 수 계산 (논문 방법론 기반)
    # 윈도우 크기에 비해 너무 크지 않도록 함
    h = max(1, int(np.ceil(2 * window_size / n_channels)) + 1)
    # 최대 h 제한
    max_h = 12  # 스파이크 밴드 최대값
    if r_input == 100:  # 리플 밴드
        max_h = 6  # 리플 밴드 최대값
    h = min(h, max_h)
    
    print(f"Time delay embedding count: h = {h}")
    print(f"Window size: {window_size} samples")
    print(f"Step size: {step_size} samples")
    print(f"Total windows: {n_windows}")
    
    # 평균 주파수 저장 배열
    all_freqs = np.zeros((r_input, n_windows))
    
    # 배치 범위
    batch_ranges = [(i, min(i+batch_size, n_windows)) for i in range(0, n_windows, batch_size)]
    
    # DMD 특성 저장 배열
    dmd_features = np.zeros((n_channels, r_input, n_windows), dtype=complex)
    
    for batch_idx, (start_idx, end_idx) in enumerate(batch_ranges):
        print(f"\rProcessing batch: {batch_idx+1}/{len(batch_ranges)} ({start_idx}-{end_idx})", end="")
        
        batch_windows = end_idx - start_idx
        
        for j in range(batch_windows):
            window_idx = start_idx + j
            start_sample = window_idx * step_size
            end_sample = start_sample + window_size
            
            if end_sample > n_samples:
                break
            
            # 윈도우 데이터 추출
            window = data[:, start_sample:end_sample]
            
            # DMD 적용
            try:
                Phi, omega, _, _ = dmd_full(window, dt=1/fs, r=r_input, h=h)
                
                # 주파수 추출
                freq = np.abs(omega) / (2 * np.pi)
                
                # Phi와 freq가 올바른 형태인지 확인
                if Phi.shape[1] != r_input:
                    print(f"\nAdjusting Phi shape from {Phi.shape} to ({n_channels}, {r_input})")
                    # 적절한 형태의 Phi와 freq 생성
                    if Phi.shape[1] < r_input:
                        # 너무 작으면 0으로 패딩
                        padded_Phi = np.zeros((n_channels, r_input), dtype=complex)
                        padded_Phi[:, :Phi.shape[1]] = Phi
                        padded_freq = np.zeros(r_input)
                        padded_freq[:freq.shape[0]] = freq
                        
                        Phi = padded_Phi
                        freq = padded_freq
                    else:
                        # 너무 크면 자름
                        Phi = Phi[:, :r_input]
                        freq = freq[:r_input]
                
                # 특성 저장
                dmd_features[:, :, window_idx] = Phi
                all_freqs[:, window_idx] = freq
                
            except Exception as e:
                print(f"\nError processing window {window_idx+1}: {e}")
                # 이전 윈도우 또는 0 사용
                if window_idx > 0:
                    dmd_features[:, :, window_idx] = dmd_features[:, :, window_idx-1]
                    all_freqs[:, window_idx] = all_freqs[:, window_idx-1]
        
        # 각 배치 후 메모리 정리
        gc.collect()
    
    print("\nDMD feature extraction complete!")
    
    # 모든 윈도우에 대한 평균 주파수
    freq_mean = np.mean(all_freqs, axis=1)
    
    return dmd_features, freq_mean


def get_indices(freq_mean, r):
    """
    DMD 모드를 주파수 대역으로 분류합니다.
    
    Parameters:
    -----------
    freq_mean : array
        DMD 모드의 평균 주파수
    r : int
        모드 수
        
    Returns:
    --------
    ids : dict
        각 주파수 대역에 대한 모드 인덱스가 있는 딕셔너리
    """
    id_delta = []
    id_theta = []
    id_alpha = []
    id_beta = []
    id_gamma = []
    
    for i in range(r):
        if freq_mean[i] < 4:
            id_delta.append(i)
        elif freq_mean[i] < 8:
            id_theta.append(i)
        elif freq_mean[i] < 12:
            id_alpha.append(i)
        elif freq_mean[i] < 30:
            id_beta.append(i)
        elif freq_mean[i] < 80:
            id_gamma.append(i)
    
    id_sb = list(range(r))  # 스파이크 밴드의 모든 모드
    
    ids = {
        'delta': id_delta,
        'theta': id_theta,
        'alpha': id_alpha,
        'beta': id_beta,
        'gamma': id_gamma,
        'sb': id_sb
    }
    
    return ids


# ===================== 네트워크 추출 함수 =====================
#NNMF
def extract_entire_network_batch(P, band_ids, band_name=None, batch_size=200):
    """
    시간 윈도우 전체에 걸쳐 DMD 스펙트럼을 평균하여 전체 네트워크를 추출합니다.
    
    Parameters:
    -----------
    P : array
        DMD 스펙트럼 특성 (n_channels, r_input, n_windows)
    band_ids : list or array
        이 밴드에 사용할 주파수 모드 인덱스
    band_name : str, optional
        주파수 밴드 이름 (로깅 목적)
    batch_size : int
        배치 처리할 윈도우 수
        
    Returns:
    --------
    entire_network : array
        지정된 밴드에 대한 전체 네트워크
    """
    if band_name is None:
        band_name = "specified band"
    
    n_channels, _, n_windows = P.shape
    
    if len(band_ids) > 0:
        entire_network = np.zeros(n_channels)
        # 배치로 윈도우 처리
        for i in range(0, n_windows, batch_size):
            end_idx = min(i + batch_size, n_windows)
            print(f"\rProcessing windows {i}-{end_idx} of {n_windows}", end="")
            window_sum = np.zeros(n_channels)
            for band_id in band_ids:
                window_sum += np.sum(np.abs(P[:, band_id, i:end_idx]), axis=1)
            entire_network += window_sum
        
        # 최종 평균 계산
        entire_network /= (n_windows * len(band_ids))
        print("\nBatch processing complete.")
    else:
        entire_network = np.zeros(n_channels)
        print(f"Warning: Empty band {band_name}. Returning zeros.")
    
    return entire_network


def extract_networks_and_temporal_maps_batch(P, ids, k=2, n_repetitions=30, batch_size=100):
    """
    NNMF를 사용하여 뇌전증성 및 배경 네트워크를 추출합니다.
    메모리 오류를 방지하기 위해 배치 처리 사용
    
    Parameters:
    -----------
    P : array
        DMD 스펙트럼 특성 (n_channels, r_input, n_windows)
    ids : list
        사용할 주파수 모드 인덱스
    k : int
        추출할 네트워크 수 (일반적으로 2)
    n_repetitions : int
        일관된 결과를 위한 NNMF 반복 횟수
    batch_size : int
        배치로 처리할 윈도우 수
        
    Returns:
    --------
    network_en : array
        뇌전증성 네트워크
    network_bk : array
        배경 네트워크
    temporal_map : array
        각 네트워크가 활성화되는 시기를 보여주는 시간적 맵
    """
    n_channels, _, n_windows = P.shape
    
    # 전체 배열을 한 번에 로드하지 않고 배치로 처리
    import tempfile
    import os
    
    temp_dir = tempfile.gettempdir()
    temp_file = os.path.join(temp_dir, 'temp_phi_current.dat')
    
    # Phi_current를 위한 메모리 매핑 배열 생성
    Phi_current = np.memmap(temp_file, dtype=np.float32, mode='w+', 
                          shape=(n_channels, n_windows))
    
    if len(ids) > 0:
        # 배치로 처리
        for i in range(0, n_windows, batch_size):
            end_idx = min(i + batch_size, n_windows)
            print(f"\rComputing average DMD power, batch {i}-{end_idx} of {n_windows}", end="")
            
            # 각 주파수 모드를 개별적으로 처리하여 큰 배열 방지
            batch_sum = np.zeros((n_channels, end_idx - i), dtype=np.float32)
            for id_idx in ids:
                # 이 모드에 대한 절대값 추가
                batch_sum += np.abs(P[:, id_idx, i:end_idx])
            
            # 메모리 매핑 배열에 평균 저장
            Phi_current[:, i:end_idx] = batch_sum / len(ids)
            
            # 메모리 확보를 위한 정리
            del batch_sum
            import gc
            gc.collect()
        
        print("\nDMD power computation complete")
    else:
        # 빈 밴드 처리
        print("Warning: Empty frequency band. Using all modes.")
        # 배치로 처리
        for i in range(0, n_windows, batch_size):
            end_idx = min(i + batch_size, n_windows)
            batch_sum = np.zeros((n_channels, end_idx - i), dtype=np.float32)
            for j in range(P.shape[1]):  # 모든 모드
                batch_sum += np.abs(P[:, j, i:end_idx])
            Phi_current[:, i:end_idx] = batch_sum / P.shape[1]
            del batch_sum
            gc.collect()
    
    # 이제 작은 배치로 NNMF 처리
    temporal_maps_all = []
    networks_en_all = []
    networks_bk_all = []
    
    # 여러 번 NNMF 실행하고 결과 평균내기
    for rep in range(n_repetitions):
        print(f"\rRunning NNMF repetition {rep+1}/{n_repetitions}", end="")
        
        # NNMF 부분에서 메모리 문제가 발생하는 것 같음
        # Phi_current.T를 NNMF 위해 배치로 처리
        try:
            # 빈 계수 행렬 초기화
            coeffs = np.zeros((n_windows, k))
            
            # NNMF 모델 초기화
            model = NMF(n_components=k, init='random', random_state=rep, 
                       max_iter=1000, tol=1e-4, solver='cd')
            
            # 작은 샘플로 먼저 학습하여 초기 구성요소 얻기
            sample_size = min(1000, n_windows)
            sample_indices = np.random.choice(n_windows, sample_size, replace=False)
            sample_data = Phi_current[:, sample_indices].T
            
            # 샘플에 대해 학습
            model.fit(sample_data)
            
            # 이제 배치로 처리
            for i in range(0, n_windows, batch_size):
                end_idx = min(i + batch_size, n_windows)
                batch_data = Phi_current[:, i:end_idx].T
                
                # 학습된 모델을 사용하여 이 배치 변환
                batch_coeffs = model.transform(batch_data)
                
                # 계수 저장
                coeffs[i:end_idx, :] = batch_coeffs
            
            # 구성요소 가져오기
            nets = model.components_
            
            # 최대 계수 기반 초기 시간적 맵 찾기
            ind1 = np.argmax(coeffs, axis=1)
            
            # 각 공간 구성의 빈도 계산
            c1 = np.sum(ind1 == 0)
            c2 = np.sum(ind1 == 1)
            
            # 덜 빈번한 네트워크가 뇌전증성(두 번째 행)이 되도록 재정렬
            if c1 < c2:
                nets_new = np.zeros_like(nets)
                nets_new[0, :] = nets[1, :]
                nets_new[1, :] = nets[0, :]
                ind1 = 1 - ind1
            else:
                nets_new = nets.copy()
            
            # K-means로 시간적 맵 스무딩
            kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
            
            # 필요한 경우 K-means도 배치로 처리
            index = np.zeros(n_windows, dtype=int)
            for i in range(0, n_windows, batch_size):
                end_idx = min(i + batch_size, n_windows)
                batch_coeffs = coeffs[i:end_idx, :]
                batch_index = kmeans.fit_predict(batch_coeffs)
                index[i:end_idx] = batch_index
            
            # K-means 인덱스를 NNMF 인덱스와 매칭
            c3 = np.linalg.norm(ind1 - index)
            c4 = np.linalg.norm(ind1 - (1 - index))
            if c4 < c3:
                index = 1 - index
            
            # 결과 저장
            temporal_maps_all.append(ind1)
            networks_en_all.append(nets_new[1, :])
            networks_bk_all.append(nets_new[0, :])
            
        except Exception as e:
            print(f"\nNNMF failed in iteration {rep+1}: {e}. Skipping.")
            continue
    
    print("\nNNMF processing complete")
    
    # 메모리 매핑 파일 정리
    del Phi_current
    if os.path.exists(temp_file):
        os.unlink(temp_file)
    
    # 결과 평균
    if len(networks_en_all) > 0:
        # 시간적 맵은 중앙값 사용
        temporal_maps_array = np.vstack(temporal_maps_all)
        temporal_map = np.median(temporal_maps_array, axis=0)
        
        # 네트워크는 평균 사용
        network_en = np.mean(np.vstack(networks_en_all), axis=0)
        network_bk = np.mean(np.vstack(networks_bk_all), axis=0)
    else:
        # 모든 NNMF 반복이 실패한 경우 처리
        print("All NNMF iterations failed. Returning zeros.")
        return np.zeros(n_channels), np.zeros(n_channels), np.zeros(n_windows)
    
    return network_en, network_bk, temporal_map

#PCA 복원
def inverse_transform_network(network_reduced, pca_model):
    """
    네트워크를 PCA 공간에서 원래 채널 공간으로 다시 변환합니다.
    
    Parameters:
    -----------
    network_reduced : array
        PCA 공간의 네트워크
    pca_model : PCA object
        학습된 PCA 모델
        
    Returns:
    --------
    network_original : array
        원래 채널 공간의 네트워크
    """
    # PCA 계수 공간에서 원래 채널 공간으로 변환
    return pca_model.inverse_transform(network_reduced)


def normalize_network(network):
    """
    네트워크를 0-1 범위로 정규화합니다.
    
    Parameters:
    -----------
    network : array
        입력 네트워크
    
    Returns:
    --------
    network_normalized : array
        정규화된 네트워크
    """
    min_val = np.min(network)
    max_val = np.max(network)
    
    # 0으로 나누기 방지
    if max_val == min_val:
        return np.zeros_like(network)
    
    return (network - min_val) / (max_val - min_val)


# ===================== 시각화 함수 =====================

def plot_eeg_timeseries(segment, channels=None, duration=5, start=0, figsize=(15, 10), scale=1.0):
    """
    EEG 세그먼트의 시계열 데이터를 시각화합니다.
    
    Parameters:
    -----------
    segment : mne.io.Raw
        시각화할 EEG 세그먼트
    channels : list or None
        표시할 채널 목록. None이면 모든 채널 표시
    duration : float
        표시할 시간 길이(초)
    start : float
        시작 시간(초)
    figsize : tuple
        그림 크기
    scale : float
        신호 크기 조절 계수
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        생성된 그림 객체
    """
    # 채널 선택
    if channels is None:
        channels = segment.ch_names
    
    # 데이터 추출
    start_sample = int(start * segment.info['sfreq'])
    end_sample = int((start + duration) * segment.info['sfreq'])
    data, times = segment.get_data(picks=channels, start=start_sample, stop=end_sample, return_times=True)
    
    # 채널 수
    n_channels = len(channels)
    
    # 각 채널의 최대 진폭 확인
    amplitudes = np.ptp(data, axis=1)
    median_amplitude = np.median(amplitudes)
    
    # 진폭이 0에 가까운 경우 기본값 설정
    if median_amplitude < 1e-6:
        spacing = 1.0
    else:
        spacing = 2 * median_amplitude
    
    # 그래프 생성
    fig, ax = plt.subplots(figsize=figsize)
    
    # 채널별 오프셋 계산 (채널 간 간격)
    offsets = np.arange(n_channels) * spacing * scale
    
    # 각 채널 플로팅
    for i, ch_name in enumerate(channels):
        # scale을 조정하여 신호 크기 조절
        ax.plot(times, data[i] * scale + offsets[i], linewidth=0.8, label=ch_name)
        
    # y축 레이블 설정
    ax.set_yticks(offsets)
    ax.set_yticklabels(channels, fontweight='bold')
    
    # 레이아웃 설정
    ax.set_xlabel('Time (s)', fontweight='bold')
    ax.set_title('EEG Time Series', fontweight='bold')
    
    # 축 제한 설정
    time_min, time_max = times[0], times[-1]
    ax.set_xlim(time_min, time_max)
    
    offset_min = min(offsets) - spacing
    offset_max = max(offsets) + spacing
    ax.set_ylim(offset_min, offset_max)
    
    # 채널 레이블을 위한 수직선 추가
    ax.axvline(x=time_min, color='k', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    return fig


def visualize_networks_plotly(networks, coordinates, ch_names, band='theta'):
    """
    Plotly를 사용하여 네트워크를 시각화합니다.
    
    Parameters:
    -----------
    networks : dict
        각 밴드별 네트워크가 있는 딕셔너리
    coordinates : array
        전극 좌표 (n_channels, 3)
    ch_names : list
        채널 이름
    band : str
        시각화할 주파수 밴드
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        네트워크 시각화가 있는 Plotly 그림
    active_indices_en : list
        뇌전증성 네트워크에서 활성화된 채널의 인덱스
    active_indices_bk : list
        배경 네트워크에서 활성화된 채널의 인덱스
    """
    if band not in networks or 'epileptogenic' not in networks[band] or 'background' not in networks[band]:
        print(f"No networks available for {band} band")
        return None, [], []
    
    network_en = networks[band]['epileptogenic']
    network_bk = networks[band]['background']
    
    # 뇌전증성 네트워크에 대한 임계값 계산
    threshold_en = np.mean(network_en) + np.std(network_en)
    
    # 배경 네트워크에 대한 임계값 계산
    threshold_bk = np.mean(network_bk) + np.std(network_bk)
    
    # 임계값을 초과하는 활성 채널 선택
    active_mask_en = network_en > threshold_en
    active_coords_en = coordinates[active_mask_en]
    active_values_en = network_en[active_mask_en]
    active_indices_en = np.where(active_mask_en)[0]
    active_names_en = [ch_names[i] for i in active_indices_en]
    
    active_mask_bk = network_bk > threshold_bk
    active_coords_bk = coordinates[active_mask_bk]
    active_values_bk = network_bk[active_mask_bk]
    active_indices_bk = np.where(active_mask_bk)[0]
    active_names_bk = [ch_names[i] for i in active_indices_bk]
    
    # 모든 가상 센서 좌표
    all_coords = coordinates
    
    # Plotly 3D 산점도 생성
    fig = go.Figure()
    
    # 모든 가상 센서: 연한 회색, 작은 마커
    fig.add_trace(go.Scatter3d(
        x=all_coords[:, 0],
        y=all_coords[:, 1],
        z=all_coords[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color='lightgray',
            opacity=0.1
        ),
        name='All Virtual Sensors'
    ))
    
    # 배경 네트워크: 파란색
    if len(active_values_bk) > 0:
        size_scale_bk = 20 * (active_values_bk / np.max(active_values_bk))
        fig.add_trace(go.Scatter3d(
            x=active_coords_bk[:, 0],
            y=active_coords_bk[:, 1],
            z=active_coords_bk[:, 2],
            mode='markers',
            marker=dict(
                size=size_scale_bk + 5,
                color=active_values_bk,
                colorscale='Blues',
                opacity=0.7,
                colorbar=dict(
                    title="Background Weight",
                    x=0.1
                )
            ),
            text=[f"{name}<br>Background Weight: {val:.4f}" for name, val in zip(active_names_bk, active_values_bk)],
            hoverinfo='text+name',
            name='Background Network'
        ))
    
    # 뇌전증성 네트워크(IEN): 빨간색, 가중치에 비례한 크기
    if len(active_values_en) > 0:
        size_scale_en = 20 * (active_values_en / np.max(active_values_en))
        fig.add_trace(go.Scatter3d(
            x=active_coords_en[:, 0],
            y=active_coords_en[:, 1],
            z=active_coords_en[:, 2],
            mode='markers',
            marker=dict(
                size=size_scale_en + 5,
                color=active_values_en,
                colorscale='Reds',
                opacity=0.8,  # 더 강조
                colorbar=dict(
                    title="IEN Weight",
                    x=0.9
                )
            ),
            text=[f"{name}<br>IEN Weight: {val:.4f}" for name, val in zip(active_names_en, active_values_en)],
            hoverinfo='text+name',
            name='Epileptogenic Network (EZ)'
        ))
    
    # 상위 5개 IEN 채널에 레이블 추가 (강조)
    if len(active_indices_en) > 0:
        top_5_indices = np.argsort(active_values_en)[-min(5, len(active_values_en)):]
        
        fig.add_trace(go.Scatter3d(
            x=active_coords_en[top_5_indices, 0],
            y=active_coords_en[top_5_indices, 1],
            z=active_coords_en[top_5_indices, 2],
            mode='markers+text',
            marker=dict(
                size=size_scale_en[top_5_indices] + 10,  # 더 큰 마커
                color='yellow',
                symbol='circle',
                line=dict(
                    color='black',
                    width=2
                )
            ),
            text=[active_names_en[i].split('_')[0] for i in top_5_indices],  # 간소화된 레이블 이름
            textposition="top center",
            name='Top 5 EZ Candidates',
            showlegend=True
        ))
    
    # 레이아웃 설정
    fig.update_layout(
        title=f'Interictal Epileptogenic Network vs Background ({band} band)',
        scene=dict(
            xaxis_title='X (mm)',
            yaxis_title='Y (mm)',
            zaxis_title='Z (mm)',
            # 더 나은 보기를 위한 카메라 위치 조정
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
            aspectmode='cube'
        ),
        legend=dict(x=0.8, y=0.9)
    )
    
    # 활성 채널 정보 출력
    print(f"IEN active channels: {len(active_indices_en)}")
    print(f"Background network active channels: {len(active_indices_bk)}")
    
    # 상위 IEN 활성 채널 출력 (EZ 후보)
    if len(active_indices_en) > 0:
        n_display = min(10, len(active_indices_en))
        top_indices = np.argsort(active_values_en)[::-1][:n_display]
        print(f"\nTop {n_display} IEN active channels (EZ candidates):")
        for i in top_indices:
            print(f"  {active_names_en[i]}: IEN weight = {active_values_en[i]:.4f}")
    
    # 네트워크 속성 계산 (논문 방법론에 따라)
    if len(active_indices_en) > 0:
        # 국소성: 활성 채널 간 평균 거리의 역수
        active_coords_en_selected = active_coords_en[top_indices]
        distances = []
        for i in range(len(active_coords_en_selected)):
            for j in range(i+1, len(active_coords_en_selected)):
                dist = np.linalg.norm(active_coords_en_selected[i] - active_coords_en_selected[j])
                distances.append(dist)
        mean_dist = np.mean(distances) if distances else 0
        focality = 1 / mean_dist if mean_dist > 0 else 0
        
        print(f"\nIEN property analysis:")
        print(f"  Focality: {focality:.4f}")
        print(f"  Weight range: {np.min(active_values_en):.4f} - {np.max(active_values_en):.4f}")
        print(f"  Mean weight: {np.mean(active_values_en):.4f}")
    
    return fig, active_indices_en, active_indices_bk


def visualize_networks_with_resection(nnmf_file_path, resection_file_path, band='theta',
                                      percentile_threshold=None, scale_factor=0.001):
    """
    NNMF 데이터(IEN 및 background network)와 resection area를 3D로 시각화합니다.
    
    Parameters:
    -----------
    nnmf_file_path : str
        NNMF 데이터가 포함된 CSV 파일 경로 (채널 이름, 좌표, 네트워크 값 포함)
    resection_file_path : str
        Resection area 좌표가 포함된 CSV 파일 경로 (x, y, z 좌표)
    band : str
        시각화할 주파수 대역 (예: 'theta', 'beta', 등)
    percentile_threshold : int, optional
        임계값으로 사용할 백분위수. None인 경우 논문 방식인 "mean + 1 std" 사용
    scale_factor : float
        Resection 좌표에 적용할 스케일 팩터 (기본값: 0.001, mm -> m 변환)
    
    Returns:
    --------
    plotly.graph_objects.Figure
        3D 시각화 결과
    """
    # NNMF 데이터 로드
    nnmf_data = pd.read_csv(nnmf_file_path)
    
    # 필요한 열 확인
    epileptogenic_col = f'{band}_epileptogenic'
    background_col = f'{band}_background'
    
    # 좌표 및 네트워크 값 추출
    coordinates = nnmf_data[['x_coord', 'y_coord', 'z_coord']].values
    en_network = nnmf_data[epileptogenic_col].values
    bg_network = nnmf_data[background_col].values
    
    # Resection area 데이터 로드 및 스케일링
    resection_data = pd.read_csv(resection_file_path)
    
    # 첫 세 개 열을 좌표로 사용
    resection_coords = resection_data.iloc[:, :3].values
    
    # 항상 resection 좌표를 축소 (mm -> m 가정)
    resection_coords = resection_coords * scale_factor
    
    # 임계값 계산
    if percentile_threshold is not None:
        # 백분위수 기반 임계값
        en_threshold = np.percentile(en_network, percentile_threshold)
        bg_threshold = np.percentile(bg_network, percentile_threshold)
    else:
        # 논문 방식: "mean plus one standard deviation"
        en_threshold = np.mean(en_network) + np.std(en_network)
        bg_threshold = np.mean(bg_network) + np.std(bg_network)
    
    # 임계값을 초과하는 채널만 활성화된 것으로 간주
    en_active = en_network > en_threshold
    bg_active = bg_network > bg_threshold
    
    # 스케일링을 위한 최대값 계산 (활성화된 채널만 고려)
    en_max = np.max(en_network[en_active]) if np.any(en_active) else np.max(en_network)
    bg_max = np.max(bg_network[bg_active]) if np.any(bg_active) else np.max(bg_network)
    
    # 그래프 생성
    fig = go.Figure()
    
    # Resection area 시각화
    try:
        hull = ConvexHull(resection_coords)
        
        # Convex hull을 사용한 3D mesh 생성
        fig.add_trace(
            go.Mesh3d(
                x=resection_coords[:, 0],
                y=resection_coords[:, 1],
                z=resection_coords[:, 2],
                i=hull.simplices[:, 0],
                j=hull.simplices[:, 1],
                k=hull.simplices[:, 2],
                color='green',
                opacity=0.5,
                name="Resection Volume"
            )
        )
    except Exception as e:
        # 대안: 점으로 표시
        fig.add_trace(
            go.Scatter3d(
                x=resection_coords[:, 0],
                y=resection_coords[:, 1],
                z=resection_coords[:, 2],
                mode='markers',
                marker=dict(
                    size=4,
                    color='green',
                    opacity=0.5
                ),
                name="Resection Volume Points"
            )
        )
    
    # 전극 색상 및 크기 계산
    colors = np.zeros((len(en_network), 3))  # (R, G, B)
    sizes = np.ones(len(en_network)) * 3  # 기본 크기
    
    # 색상과 크기를 계산
    for i in range(len(en_network)):
        if en_active[i]:
            # IEN 활성화
            intensity = min(1.0, (en_network[i] - en_threshold) / (en_max - en_threshold) if en_max > en_threshold else 0.5)
            colors[i, 0] = 1.0  # Red for IEN
            sizes[i] = 6 + 15 * intensity
        elif bg_active[i]:
            # Background 활성화
            intensity = min(1.0, (bg_network[i] - bg_threshold) / (bg_max - bg_threshold) if bg_max > bg_threshold else 0.5)
            colors[i, 2] = 1.0  # Blue for background
            sizes[i] = 6 + 10 * intensity
    
    # RGB를 문자열로 변환
    color_strings = []
    for color in colors:
        r, g, b = [int(c * 255) for c in color]
        color_strings.append(f'rgb({r},{g},{b})')
    
    # 호버 텍스트 생성
    hover_texts = []
    for i in range(len(en_network)):
        channel_name = nnmf_data['channel_name'].iloc[i] if 'channel_name' in nnmf_data.columns else f"Channel {i}"
        network_type = "IEN" if en_active[i] else "Background" if bg_active[i] else "None"
        
        hover_text = (
            f"Channel: {channel_name}<br>" +
            f"IEN: {en_network[i]:.4f} (threshold: {en_threshold:.4f})<br>" +
            f"BG: {bg_network[i]:.4f} (threshold: {bg_threshold:.4f})<br>" +
            f"Network: {network_type}"
        )
        hover_texts.append(hover_text)
    
    # 전극 시각화
    fig.add_trace(
        go.Scatter3d(
            x=coordinates[:, 0],
            y=coordinates[:, 1],
            z=coordinates[:, 2],
            mode='markers',
            marker=dict(
                size=sizes,
                color=color_strings,
                opacity=0.8,
                line=dict(width=0.5, color='gray')
            ),
            text=hover_texts,
            hoverinfo='text',
            name="Electrodes"
        )
    )
    
    # 레이아웃 설정
    fig.update_layout(
        title=f"{band.upper()} Band: Epileptogenic & Background Networks with Resection Area",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        legend=dict(
            title="Networks",
            itemsizing="constant"
        ),
        height=800,
        width=1000,
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    # 네트워크 분류 결과 요약
    en_only = np.sum(en_active & ~bg_active)
    bg_only = np.sum(bg_active & ~en_active)
    both = np.sum(en_active & bg_active)
    none = np.sum(~en_active & ~bg_active)
    
    print(f"--- {band.upper()} Band Network Analysis ---")
    print(f"임계값: IEN={en_threshold:.4f}, Background={bg_threshold:.4f}")
    print(f"활성화된 채널: IEN={np.sum(en_active)}, Background={np.sum(bg_active)}")
    print(f"채널 분류: IEN만={en_only}, Background만={bg_only}, 둘 다={both}, 둘 다 아님={none}")
    
    return fig


# ===================== 네트워크 분석 함수 =====================

def calculate_network_properties(networks, coordinates, band='theta'):
    """
    네트워크 속성을 계산합니다: 국소성, 겹침, 참조까지의 거리
    
    Parameters:
    -----------
    networks : dict
        네트워크 딕셔너리
    coordinates : array
        전극 좌표
    band : str
        분석할 주파수 밴드
    
    Returns:
    --------
    properties : dict
        네트워크 속성이 있는 딕셔너리
    """
    if band not in networks or 'epileptogenic' not in networks[band]:
        print(f"No epileptogenic network for {band} band")
        return None
    
    network_en = networks[band]['epileptogenic']
    
    # 활성 채널에 대한 임계값
    threshold_en = np.mean(network_en) + np.std(network_en)
    active_mask = network_en > threshold_en
    
    if np.sum(active_mask) == 0:
        print("No active channels in epileptogenic network")
        return {
            'focality': 0,
            'active_channels': 0,
            'mean_weight': 0,
            'max_weight': 0
        }
    
    active_coords = coordinates[active_mask]
    active_values = network_en[active_mask]
    
    # 국소성: 활성 채널 간 평균 거리의 역수
    distances = []
    for i in range(len(active_coords)):
        for j in range(i+1, len(active_coords)):
            dist = np.linalg.norm(active_coords[i] - active_coords[j])
            distances.append(dist)
    
    mean_dist = np.mean(distances) if distances else 0
    focality = 1 / mean_dist if mean_dist > 0 else 0
    
    return {
        'focality': focality,
        'active_channels': np.sum(active_mask),
        'mean_weight': np.mean(active_values),
        'max_weight': np.max(active_values),
        'active_indices': np.where(active_mask)[0]
    }


def create_network_data(networks, coordinates, ch_names, band=None):
    """
    특정 주파수 대역 또는 모든 대역의 네트워크 값과 채널 정보를 통합한 데이터프레임을 생성합니다.
    
    Parameters:
    -----------
    networks : dict
        주파수 대역별 네트워크 값이 포함된 딕셔너리
    coordinates : array
        채널 좌표 배열 (n_channels, 3)
    ch_names : list
        채널 이름 목록
    band : str, optional
        특정 주파수 대역 (None인 경우 모든 대역 포함)
        
    Returns:
    --------
    pandas.DataFrame
        채널 정보, 네트워크 값, 좌표가 포함된 데이터프레임
    """
    # 기본 데이터 구조 생성
    data = {
        'channel_name': ch_names,
        'x_coord': coordinates[:, 0],
        'y_coord': coordinates[:, 1],
        'z_coord': coordinates[:, 2]
    }
    
    # 처리할 주파수 대역 결정
    if band is not None:
        if band not in networks:
            raise ValueError(f"요청한 대역 '{band}'가 networks에 존재하지 않습니다.")
        bands_to_process = [band]
    else:
        bands_to_process = list(networks.keys())
    
    # 각 주파수 대역의 네트워크 값 추가
    for band in bands_to_process:
        if 'epileptogenic' in networks[band]:
            data[f'{band}_epileptogenic'] = networks[band]['epileptogenic']
        if 'background' in networks[band]:
            data[f'{band}_background'] = networks[band]['background']
        if 'entire' in networks[band]:
            data[f'{band}_entire'] = networks[band]['entire']
    
    # 데이터프레임 생성
    df = pd.DataFrame(data)
    
    return df


# ===================== 통합 파이프라인 함수 =====================

def dmd_pipeline(epochs_vs, virtual_sensors, output_file=None):
    """
    DMD 분석 파이프라인의 전체 실행 함수
    
    Parameters:
    -----------
    epochs_vs : mne.Epochs
        분석할 에포크
    virtual_sensors : list
        가상 센서 정보 리스트
    output_file : str, optional
        결과를 저장할 CSV 파일 경로
    
    Returns:
    --------
    results : dict
        모든 분석 결과를 포함하는 딕셔너리
    """
    print("Step 1: Prepare data...")
    segment_data, ch_names, sfreq = create_segment_data(epochs_vs)
    print(f"Segment created. Shape: {segment_data.shape}")
    
    print("Step 2: Extract coordinates...")
    coordinates = extract_coordinates(ch_names, virtual_sensors)
    
    print("Step 3: Apply bandpass filters...")
    spike_band_data, ripple_band_data = filter_data(segment_data, sfreq)
    
    print("Step 4: Extract DMD features for spike band...")
    r_input_sb = 50
    P_sb, freq_mean_sb = extract_features_batch(
        spike_band_data,
        window_size_ms=250,
        overlap_perc=0.95,
        fs=sfreq,
        r_input=r_input_sb,
        batch_size=500
    )
    
    print("Step 5: Extract DMD features for ripple band...")
    r_input_rb = 100
    P_rb, freq_mean_rb = extract_features_batch(
        ripple_band_data,
        window_size_ms=150,
        overlap_perc=0.95,
        fs=sfreq,
        r_input=r_input_rb,
        batch_size=500
    )
    
    print(f"DMD features extracted.")
    print(f"P_sb shape: {P_sb.shape}, freq_mean_sb shape: {freq_mean_sb.shape}")
    print(f"P_rb shape: {P_rb.shape}, freq_mean_rb shape: {freq_mean_rb.shape}")
    
    print("Step 6: Classify modes into frequency bands...")
    ids_sb = get_indices(freq_mean_sb, r_input_sb)
    
    print("Step 7: Extract networks for each band...")
    entire_networks = {}
    networks_reduced = {}
    temporal_maps = {}
    
    # Process spike band sub-bands
    for band, band_ids in ids_sb.items():
        if len(band_ids) > 0:
            print(f"Extracting {band} band networks...")
            
            # Extract entire network
            entire_networks[band] = extract_entire_network_batch(P_sb, band_ids, band)
            
            # Extract epileptogenic and background networks
            network_en, network_bk, temporal_map = extract_networks_and_temporal_maps_batch(
                P_sb, band_ids, k=2, n_repetitions=30
            )
            
            networks_reduced[band] = {
                'epileptogenic': network_en,
                'background': network_bk,
                'entire': entire_networks[band]
            }
            
            temporal_maps[band] = temporal_map > 0.5  # Binarize
    
    # Process ripple band
    print("Extracting ripple band networks...")
    entire_networks['rb'] = extract_entire_network_batch(P_rb, list(range(r_input_rb)), 'rb')
    network_en_rb, network_bk_rb, temporal_map_rb = extract_networks_and_temporal_maps_batch(
        P_rb, list(range(r_input_rb)), k=2, n_repetitions=30
    )
    networks_reduced['rb'] = {
        'epileptogenic': network_en_rb,
        'background': network_bk_rb,
        'entire': entire_networks['rb']
    }
    temporal_maps['rb'] = temporal_map_rb > 0.5  # Binarize
    
    print("Step 8: Normalize networks...")
    networks = {}
    
    for band in networks_reduced.keys():
        networks[band] = {}
        
        for network_type, network_reduced in networks_reduced[band].items():
            # Normalize network (0-1 range)
            network_normalized = normalize_network(network_reduced)
            networks[band][network_type] = network_normalized
    
    print("Step 9: Calculate network properties...")
    properties = {}
    
    for band in networks.keys():
        properties[band] = calculate_network_properties(networks, coordinates, band)
    
    print("Step 10: Visualize networks...")
    fig = None
    active_indices_en = []
    active_indices_bk = []
    
    # Try theta band first, then try others if theta is not available
    bands_to_try = ['theta', 'alpha', 'beta', 'gamma', 'sb', 'rb']
    
    for band in bands_to_try:
        if band in networks and 'epileptogenic' in networks[band]:
            fig, active_indices_en, active_indices_bk = visualize_networks_plotly(
                networks, coordinates, ch_names, band=band
            )
            if fig is not None:
                break
    
    # Create results dictionary
    results = {
        'networks': networks,
        'temporal_maps': temporal_maps,
        'properties': properties,
        'coordinates': coordinates,
        'ch_names': ch_names,
        'active_indices_en': active_indices_en,
        'active_indices_bk': active_indices_bk,
        'fig': fig
    }
    
    # Save results to CSV if output file is specified
    if output_file:
        all_bands_data = create_network_data(networks, coordinates, ch_names)
        all_bands_data.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
    
    print("DMD analysis pipeline complete!")
    return results


# 메인 실행 함수
def main():
    """
    메인 실행 함수 - 파일을 불러오고 파이프라인을 실행하여 네트워크를 시각화합니다.
    
    Example:
    --------
    # 1. 에포크 및 가상 센서 로드
    # 2. DMD 파이프라인 실행
    results = dmd_pipeline(epochs_vs, virtual_sensors, output_file='dmd_networks.csv')
    
    # 3. 네트워크 시각화
    fig = results['fig']
    fig.show()
    
    # 4. resection 영역과 함께 시각화
    fig = visualize_networks_with_resection(
        nnmf_file_path='dmd_networks.csv',
        resection_file_path='resection_voxels_world_coords.csv',
        band='theta',
        percentile_threshold=90,
        scale_factor=0.001
    )
    fig.show()
    """
    pass


if __name__ == "__main__":
    main()
# %%

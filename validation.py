#%%
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_curve, auc
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt


def analyze_networks_vs_resection(
    nnmf_file_path, 
    resection_file_path, 
    bands=None,
    scale_factor=0.001,  # Resection 좌표를 mm에서 m로 변환
    resection_threshold=10  # 10mm 이내의 거리를 resection에 포함된 것으로 간주
):
    """
    NNMF 네트워크와 resection area를 비교 분석합니다.
    
    Parameters:
    -----------
    nnmf_file_path : str
        NNMF 데이터가 포함된 CSV 파일 경로
    resection_file_path : str
        Resection area 좌표가 포함된 CSV 파일 경로
    bands : list or None
        분석할 주파수 대역 목록 (None이면 모든 대역 분석)
    scale_factor : float
        Resection 좌표에 적용할 스케일 팩터
    resection_threshold : float
        Resection과의 거리가 이 값 이내인 채널을 resection 내부로 간주 (mm)
    
    Returns:
    --------
    dict
        각 대역별 분석 결과를 포함하는 딕셔너리
    """
    # 데이터 로드
    nnmf_data = pd.read_csv(nnmf_file_path)
    resection_data = pd.read_csv(resection_file_path)
    
    # 채널 좌표 추출
    coordinates = nnmf_data[['x_coord', 'y_coord', 'z_coord']].values
    
    # Resection 좌표 추출 및 스케일링
    resection_coords = resection_data.iloc[:, :3].values * scale_factor
    
    # 사용 가능한 모든 대역 확인
    all_bands = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'sb', 'rb']
    
    # 분석할 대역 결정
    bands_to_analyze = bands if bands is not None else all_bands
    
    # 각 대역별로 분석한 결과를 저장할 딕셔너리
    results = {}

    #0차 시도, 기본 channel_distanes and resection_mask   
    # Resection과의 최소 거리 계산 (각 채널에서 가장 가까운 resection 점까지의 거리)
    channel_distances = []
    for coord in coordinates:
        min_distance = np.min(np.sqrt(np.sum((resection_coords - coord)**2, axis=1)))
        channel_distances.append(min_distance)
    
    # Resection 마스크 생성 (threshold 이내의 거리에 있는 채널)
    resection_mask = np.array(channel_distances) <= resection_threshold * scale_factor
   
    #1차시도도
    # 논문 방식대로 거리 계산 (단위: mm)
    # channel_distances = calculate_distances_to_resection(coordinates, resection_coords)
    
    # # Resection 마스크 생성 (threshold 이내의 거리에 있는 채널)
    # resection_mask = channel_distances <= resection_threshold

    #2차 시도
    # 새로운 코드: 최적의 resection mask 생성
    # resection_mask, channel_distances = create_optimal_resection_mask(
    #     coordinates, 
    #     resection_coords, 
    #     scale_factor
    # )

    # 각 주파수 대역에 대해 분석 수행
    for band in bands_to_analyze:
        # 필요한 열 이름 정의
        en_col = f'{band}_epileptogenic'
        bg_col = f'{band}_background'
        entire_col = f'{band}_entire'
        
        # 열이 존재하는지 확인
        if not all(col in nnmf_data.columns for col in [en_col, bg_col, entire_col]):
            print(f"Warning: {band} 대역에 대한 네트워크 데이터가 없습니다. 건너뜁니다.")
            continue
        
        # 네트워크 값 추출
        en_network = nnmf_data[en_col].values
        bg_network = nnmf_data[bg_col].values
        entire_network = nnmf_data[entire_col].values
        
        # 임계값 계산 (mean + std)
        en_threshold = np.mean(en_network) + np.std(en_network)
        bg_threshold = np.mean(bg_network) + np.std(bg_network)
        
        # 활성화된 채널 마스크
        en_active = en_network > en_threshold
        bg_active = bg_network > bg_threshold
        
        # 이 대역에 대한 결과 저장 딕셔너리 초기화
        band_results = {
            'thresholds': {
                'epileptogenic': en_threshold,
                'background': bg_threshold
            },
            'active_counts': {
                'epileptogenic': np.sum(en_active),
                'background': np.sum(bg_active)
            },
            'focality': {}, 
            'overlap': {},
            'distance': {},
            'power_comparison': {},
            'auc': {}
        }
        
        # 1. 네트워크 특성 계산: 포컬리티 (Focality)
        for name, mask in [('epileptogenic', en_active), ('background', bg_active)]:
            if np.sum(mask) >= 2:  # 포컬리티 계산에는 최소 2개의 채널이 필요
                active_coords = coordinates[mask]
                distances = []
                for i in range(len(active_coords)):
                    for j in range(i+1, len(active_coords)):
                        distances.append(np.linalg.norm(active_coords[i] - active_coords[j]))
                focality = 1 / np.mean(distances) if distances else 0
            else:
                focality = 0
                
            band_results['focality'][name] = focality
            
        # 2. 네트워크 특성 계산: Resection과의 오버랩 (Overlap)
        for name, mask in [('epileptogenic', en_active), ('background', bg_active)]:
            if np.sum(mask) > 0:
                overlap_percentage = 100 * np.sum(mask & resection_mask) / np.sum(mask)
            else:
                overlap_percentage = 0
                
            band_results['overlap'][name] = overlap_percentage
            
        # 3. 네트워크 특성 계산: Resection과의 거리 (Distance)
        for name, mask in [('epileptogenic', en_active), ('background', bg_active)]:
            if np.sum(mask) > 0:
                distance = np.mean(np.array(channel_distances)[mask]) / scale_factor  # Convert back to mm
            else:
                distance = float('inf')
                
            band_results['distance'][name] = distance
        
        # 4. 네트워크 파워 비교: Resection 내부 vs 외부
        for name, values in [('epileptogenic', en_network), ('background', bg_network), ('entire', entire_network)]:
            inside_resection = values[resection_mask]
            outside_resection = values[~resection_mask]
            
            if len(inside_resection) > 0 and len(outside_resection) > 0:
                median_inside = np.median(inside_resection)
                median_outside = np.median(outside_resection)
                
                # Wilcoxon 테스트는 두 배열의 길이가 같아야 하므로 조정
                min_len = min(len(inside_resection), len(outside_resection))
                if min_len > 10:  # 통계적으로 의미있는 최소 샘플 크기
                    stat, p_value = wilcoxon(
                        inside_resection[:min_len] if len(inside_resection) > min_len else inside_resection,
                        outside_resection[:min_len] if len(outside_resection) > min_len else outside_resection
                    )
                else:
                    p_value = np.nan
                
                # Effect size 계산 (Cohen's d 유사)
                if np.std(inside_resection) > 0 and np.std(outside_resection) > 0:
                    effect_size = (np.mean(inside_resection) - np.mean(outside_resection)) / np.sqrt(
                        (np.std(inside_resection)**2 + np.std(outside_resection)**2) / 2
                    )
                else:
                    effect_size = np.nan
            else:
                median_inside = np.nan
                median_outside = np.nan
                p_value = np.nan
                effect_size = np.nan
            
            band_results['power_comparison'][name] = {
                'median_inside': median_inside,
                'median_outside': median_outside,
                'p_value': p_value,
                'effect_size': effect_size
            }
        
        # 5. ROC 분석: Resection 예측
        for name, values in [('epileptogenic', en_network), ('background', bg_network), ('entire', entire_network)]:
            fpr, tpr, _ = roc_curve(resection_mask, values)
            roc_auc = auc(fpr, tpr)
            
            band_results['auc'][name] = roc_auc
        
        # 대역별 결과 저장
        results[band] = band_results
    
    return results

def calculate_distances_to_resection(channel_coordinates, resection_coordinates):
    """
    각 채널에서 절제 영역까지의 최소 거리를 계산합니다.
    
    Parameters:
    -----------
    channel_coordinates : array
        채널 좌표 배열 (n_channels, 3)
    resection_coordinates : array
        절제 영역 좌표 배열 (n_resection_points, 3)
        
    Returns:
    --------
    distances : array
        각 채널과 절제 영역 사이의 최소 거리 (단위: mm)
    """
    n_channels = channel_coordinates.shape[0]
    n_resection = resection_coordinates.shape[0]
    
    min_distances = np.zeros(n_channels)
    min_indices = np.zeros(n_channels, dtype=int)
    
    for i in range(n_channels):
        channel_coord = channel_coordinates[i]
        
        # 모든 절제 영역 좌표와의 거리 계산
        distances = np.zeros(n_resection)
        for j in range(n_resection):
            # pdist 함수를 사용한 MATLAB 코드 방식과 동일하게 계산
            # pdist는 쌍 사이의 유클리드 거리를 계산
            distances[j] = np.linalg.norm(channel_coord - resection_coordinates[j])
        
        # 최소 거리와 해당 인덱스 저장
        min_idx = np.argmin(distances)
        min_dist = distances[min_idx]
        
        min_distances[i] = min_dist
        min_indices[i] = min_idx
    
    # MATLAB 코드와 동일하게 mm 단위로 변환 (*1000)
    # 이미 스케일링이 적용된 경우 이 변환이 필요한지 확인할 것
    return min_distances * 1000

def create_optimal_resection_mask(coordinates, resection_coords, scale_factor=0.001):
    """
    전극과 절제 영역 사이의 거리에 기반한 최적의 resection mask를 생성합니다.
    
    Parameters:
    -----------
    coordinates : array
        전극 좌표 (n_channels, 3)
    resection_coords : array
        절제 영역 좌표 (n_points, 3)
    scale_factor : float
        좌표 스케일링 인자 (필요 시)
        
    Returns:
    --------
    mask : array
        절제 영역 마스크 (부울 배열)
    distances : array
        각 전극에서 절제 영역까지의 거리
    """
    import numpy as np
    
    # 1. 각 전극에서 절제 영역까지의 최소 거리 계산
    channel_distances = []
    for coord in coordinates:
        min_dist = np.min(np.sqrt(np.sum((resection_coords - coord)**2, axis=1)))
        channel_distances.append(min_dist)
    
    channel_distances = np.array(channel_distances)
    
    # 2. 거리 분포 분석을 통한 자연스러운 경계 찾기
    sorted_distances = np.sort(channel_distances)
    
    # 변화율을 계산 (거리 변화가 급격한 지점 찾기)
    if len(sorted_distances) > 1:
        diffs = np.diff(sorted_distances)
        
        # 상위 25%의 변화율을 가진 지점 찾기
        threshold_idx = np.where(diffs > np.percentile(diffs, 75))[0]
        
        if len(threshold_idx) > 0:
            # 첫 번째 급격한 변화 지점을 임계값으로 사용
            threshold = sorted_distances[threshold_idx[0] + 1]
        else:
            # 변화가 크지 않으면 상위 20%의 전극만 사용
            threshold = np.percentile(sorted_distances, 20)
    else:
        # 전극이 하나밖에 없는 경우
        threshold = channel_distances[0] / 2
    
    # 3. 적절한 맥아서 위치의 최소/최대 전극 수 설정
    min_electrodes = max(3, int(0.05 * len(coordinates)))  # 최소 5% 또는 3개
    max_electrodes = int(0.4 * len(coordinates))  # 최대 40%
    
    # 임계값 조정하여 전극 수 제한
    mask = channel_distances <= threshold
    
    # 너무 많은 전극이 선택된 경우 임계값 낮추기
    if np.sum(mask) > max_electrodes:
        threshold = sorted_distances[max_electrodes - 1]
        mask = channel_distances <= threshold
    
    # 너무 적은 전극이 선택된 경우 임계값 높이기
    if np.sum(mask) < min_electrodes:
        threshold = sorted_distances[min_electrodes - 1]
        mask = channel_distances <= threshold
    
    # 최종 마스크와 거리 반환
    return mask, channel_distances

def plot_network_properties(results, bands=None, title_prefix=""):
    """
    네트워크 특성을 비교하는 그래프를 생성합니다.
    
    Parameters:
    -----------
    results : dict
        analyze_networks_vs_resection 함수의 결과
    bands : list or None
        표시할 대역 (None이면 모든 대역 표시)
    title_prefix : str
        그래프 제목 앞에 추가할 텍스트
    
    Returns:
    --------
    plotly.graph_objects.Figure
        네트워크 특성을 보여주는 그래프
    """
    available_bands = list(results.keys())
    bands_to_plot = bands if bands is not None else available_bands
    
    # 분석된 모든 특성 목록
    properties = [
        ('focality', 'Focality (mm<sup>-1</sup>)', 'Higher is better'),
        ('overlap', 'Overlap with Resection (%)', 'Higher is better'),
        ('distance', 'Distance from Resection (mm)', 'Lower is better'),
        ('auc', 'AUC ROC for Resection Prediction', 'Higher is better')
    ]
    
    # 서브플롯 생성
    fig = go.Figure()
    
    # 대역별 색상 지정
    band_colors = {
        'delta': '#1f77b4',  # 파란색
        'theta': '#ff7f0e',  # 주황색
        'alpha': '#2ca02c',  # 녹색
        'beta': '#d62728',   # 빨간색
        'gamma': '#9467bd',  # 보라색
        'sb': '#8c564b',     # 갈색
        'rb': '#e377c2'      # 분홍색
    }
    
    # 특성별로 그래프 추가
    for i, (prop_key, prop_name, direction) in enumerate(properties):
        # IEN(epileptogenic) 값
        x_vals = []
        y_vals = []
        colors = []
        hover_texts = []
        
        for band in bands_to_plot:
            if band in results:
                if prop_key in results[band]:
                    if 'epileptogenic' in results[band][prop_key]:
                        x_vals.append(band)
                        value = results[band][prop_key]['epileptogenic']
                        y_vals.append(value)
                        colors.append(band_colors.get(band, '#000000'))
                        
                        # hover 텍스트에 추가 정보
                        if prop_key == 'focality':
                            hover_text = f"{band} band<br>IEN Focality: {value:.4f}<br>Active channels: {results[band]['active_counts']['epileptogenic']}"
                        elif prop_key == 'overlap':
                            hover_text = f"{band} band<br>IEN Overlap: {value:.1f}%<br>Active channels: {results[band]['active_counts']['epileptogenic']}"
                        elif prop_key == 'distance':
                            hover_text = f"{band} band<br>IEN Distance: {value:.2f} mm<br>Active channels: {results[band]['active_counts']['epileptogenic']}"
                        elif prop_key == 'auc':
                            hover_text = f"{band} band<br>IEN AUC: {value:.4f}<br>Active channels: {results[band]['active_counts']['epileptogenic']}"
                        else:
                            hover_text = f"{band} band<br>Value: {value}"
                            
                        hover_texts.append(hover_text)
        
        # IEN(epileptogenic) 바 차트 추가
        fig.add_trace(
            go.Bar(
                x=x_vals,
                y=y_vals,
                name=f"IEN {prop_name}",
                marker_color=colors,
                text=y_vals,
                hovertext=hover_texts,
                hoverinfo="text",
                visible=True if i == 0 else "legendonly"  # 첫 번째 특성만 기본적으로 표시
            )
        )
        
        # Background 값도 추가 (참고용)
        x_vals = []
        y_vals = []
        colors = []
        hover_texts = []
        
        for band in bands_to_plot:
            if band in results:
                if prop_key in results[band]:
                    if 'background' in results[band][prop_key]:
                        x_vals.append(band)
                        value = results[band][prop_key]['background']
                        y_vals.append(value)
                        # Background는 더 연한 색상으로 표시
                        band_color = band_colors.get(band, '#000000')
                        # 색상 밝게 만들기 (투명도 추가)
                        colors.append(f"rgba({int(band_color[1:3], 16)},{int(band_color[3:5], 16)},{int(band_color[5:7], 16)},0.5)")
                        
                        # hover 텍스트 
                        hover_text = f"{band} band<br>Background {prop_key}: {value}<br>Active channels: {results[band]['active_counts']['background']}"
                        hover_texts.append(hover_text)
        
        # Background 바 차트 추가
        fig.add_trace(
            go.Bar(
                x=x_vals,
                y=y_vals,
                name=f"Background {prop_name}",
                marker_color=colors,
                text=y_vals,
                hovertext=hover_texts,
                hoverinfo="text",
                visible="legendonly"  # 기본적으로 숨김
            )
        )
    
    # 레이아웃 설정
    buttons = []
    for i, (prop_key, prop_name, direction) in enumerate(properties):
        visibility = [False] * (len(properties) * 2)
        visibility[i*2] = True     # 해당 IEN 속성 보이기
        visibility[i*2+1] = False  # 해당 Background 속성 숨기기
        
        buttons.append(
            dict(
                label=prop_name,
                method="update",
                args=[
                    {"visible": visibility},
                    {"title": f"{title_prefix}Network Property: {prop_name}<br><sup>{direction}</sup>"}
                ]
            )
        )
    
    fig.update_layout(
        title=f"{title_prefix}Network Property: {properties[0][1]}<br><sup>{properties[0][2]}</sup>",
        xaxis_title="Frequency Band",
        barmode='group',
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                showactive=True,
                x=1.0,
                y=1.15
            )
        ],
        height=600,
        width=800
    )
    
    return fig

def plot_power_comparison(results, bands=None, title_prefix=""):
    """
    네트워크 파워의 resection 내부/외부 비교 그래프를 생성합니다.
    
    Parameters:
    -----------
    results : dict
        analyze_networks_vs_resection 함수의 결과
    bands : list or None
        표시할 대역 (None이면 모든 대역 표시)
    title_prefix : str
        그래프 제목 앞에 추가할 텍스트
    
    Returns:
    --------
    plotly.graph_objects.Figure
        파워 비교 그래프
    """
    available_bands = list(results.keys())
    bands_to_plot = bands if bands is not None else available_bands
    
    # 네트워크 유형
    network_types = ['epileptogenic', 'background', 'entire']
    
    fig = go.Figure()
    
    # 밴드별 색상
    band_colors = {
        'delta': '#1f77b4',
        'theta': '#ff7f0e',
        'alpha': '#2ca02c',
        'beta': '#d62728',
        'gamma': '#9467bd',
        'sb': '#8c564b',
        'rb': '#e377c2'
    }
    
    # 각 대역의 각 네트워크 유형에 대해 내부/외부 파워 비교
    for band in bands_to_plot:
        if band not in results:
            continue
            
        for i, net_type in enumerate(network_types):
            # 내부 파워 값
            inside_value = results[band]['power_comparison'][net_type]['median_inside']
            outside_value = results[band]['power_comparison'][net_type]['median_outside']
            p_value = results[band]['power_comparison'][net_type]['p_value']
            effect = results[band]['power_comparison'][net_type]['effect_size']
            
            # 통계적 유의성 표시
            sig_marker = '**' if not np.isnan(p_value) and p_value < 0.01 else '*' if not np.isnan(p_value) and p_value < 0.05 else ''
            
            # 효과 크기에 따른 색상 조정
            base_color = band_colors.get(band, '#000000')
            
            # 내부 값 막대 추가
            fig.add_trace(
                go.Bar(
                    x=[f"{band}<br>{net_type}"],
                    y=[inside_value],
                    name=f"{band} {net_type} Inside",
                    marker_color=base_color,
                    text=f"Inside: {inside_value:.4f}{sig_marker}",
                    textposition="auto",
                    hovertext=f"{band} band - {net_type}<br>Inside resection: {inside_value:.4f}<br>Effect size: {effect:.2f}<br>p-value: {p_value:.4f}",
                    hoverinfo="text",
                    visible=True if band == bands_to_plot[0] else "legendonly"
                )
            )
            
            # 외부 값 막대 추가
            fig.add_trace(
                go.Bar(
                    x=[f"{band}<br>{net_type}"],
                    y=[outside_value],
                    name=f"{band} {net_type} Outside",
                    marker_color=f"rgba({int(base_color[1:3], 16)},{int(base_color[3:5], 16)},{int(base_color[5:7], 16)},0.5)",
                    text=f"Outside: {outside_value:.4f}",
                    textposition="auto",
                    hovertext=f"{band} band - {net_type}<br>Outside resection: {outside_value:.4f}",
                    hoverinfo="text",
                    visible=True if band == bands_to_plot[0] else "legendonly"
                )
            )
    
    # 대역별 버튼 생성
    buttons = []
    for i, band in enumerate(bands_to_plot):
        if band not in results:
            continue
            
        visibility = [False] * len(fig.data)
        for j in range(len(network_types) * 2):  # 각 네트워크 유형마다 내부/외부 2개의 막대
            if j // (len(network_types) * 2) == i:
                visibility[j] = True
        
        buttons.append(
            dict(
                label=band.upper(),
                method="update",
                args=[
                    {"visible": visibility},
                    {"title": f"{title_prefix}Power Inside vs Outside Resection: {band.upper()} Band"}
                ]
            )
        )
    
    fig.update_layout(
        title=f"{title_prefix}Power Inside vs Outside Resection: {bands_to_plot[0].upper()} Band",
        xaxis_title="Network Type",
        yaxis_title="Power (median)",
        barmode='group',
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                showactive=True,
                x=1.0,
                y=1.15
            )
        ],
        height=600,
        width=800
    )
    
    return fig

def plot_roc_curves(results, bands=None, title_prefix=""):
    """
    ROC 곡선을 그려 네트워크의 resection 예측 성능을 비교합니다.
    
    Parameters:
    -----------
    results : dict
        analyze_networks_vs_resection 함수의 결과
    bands : list or None
        표시할 대역 (None이면 모든 대역 표시)
    title_prefix : str
        그래프 제목 앞에 추가할 텍스트
    
    Returns:
    --------
    plotly.graph_objects.Figure
        ROC 곡선 그래프
    """
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    from sklearn.metrics import roc_curve, auc
    
    # 각 대역의 AUC 값 추출하여 정렬
    auc_df = []
    for band, band_results in results.items():
        if 'auc' in band_results:
            for network_type, auc_value in band_results['auc'].items():
                auc_df.append({
                    'band': band,
                    'network_type': network_type,
                    'auc': auc_value
                })
    
    auc_df = pd.DataFrame(auc_df)
    
    # IEN 기준으로 대역 정렬 (내림차순)
    if 'epileptogenic' in auc_df['network_type'].values:
        best_bands = auc_df[auc_df['network_type'] == 'epileptogenic'].sort_values('auc', ascending=False)['band'].tolist()
    else:
        best_bands = auc_df.sort_values('auc', ascending=False)['band'].unique().tolist()
    
    # 시각화할 대역 선택
    if bands is not None:
        bands_to_plot = [b for b in bands if b in results]
    else:
        bands_to_plot = best_bands
    
    # 막대 그래프로 각 대역별 AUC 값 비교
    fig = go.Figure()
    
    # 네트워크 유형별 색상 설정
    type_colors = {
        'epileptogenic': 'rgb(255, 0, 0)',    # 빨강
        'background': 'rgb(0, 0, 255)',       # 파랑
        'entire': 'rgb(128, 128, 128)'        # 회색
    }
    
    # 각 네트워크 유형에 대해 막대 그래프 추가
    for net_type in ['epileptogenic', 'background', 'entire']:
        x_vals = []
        y_vals = []
        hover_texts = []
        
        for band in bands_to_plot:
            if band in results and 'auc' in results[band]:
                if net_type in results[band]['auc']:
                    auc_value = results[band]['auc'][net_type]
                    x_vals.append(band)
                    y_vals.append(auc_value)
                    
                    # 활성화된 채널 수 추가
                    if net_type in results[band]['active_counts']:
                        active_count = results[band]['active_counts'][net_type]
                        hover_text = f"{band} band - {net_type}<br>AUC: {auc_value:.4f}<br>Active channels: {active_count}"
                    else:
                        hover_text = f"{band} band - {net_type}<br>AUC: {auc_value:.4f}"
                    
                    hover_texts.append(hover_text)
        
        fig.add_trace(
            go.Bar(
                x=x_vals,
                y=y_vals,
                name=f"{net_type.capitalize()}",
                marker_color=type_colors.get(net_type, '#000000'),
                text=[f"{v:.3f}" for v in y_vals],
                textposition="auto",
                hovertext=hover_texts,
                hoverinfo="text"
            )
        )
    
    # 기준선 추가 (AUC = 0.5, 무작위 분류기)
    fig.add_shape(
        type="line",
        x0=-0.5,
        y0=0.5,
        x1=len(bands_to_plot) - 0.5,
        y1=0.5,
        line=dict(
            color="gray",
            width=2,
            dash="dash",
        )
    )
    
    # 레이아웃 설정
    fig.update_layout(
        title=f"{title_prefix}AUC ROC for Resection Prediction by Frequency Band",
        xaxis_title="Frequency Band",
        yaxis_title="Area Under ROC Curve",
        yaxis=dict(
            range=[0, 1],
            tickmode='linear',
            tick0=0,
            dtick=0.1
        ),
        barmode='group',
        height=600,
        width=800
    )
    
    return fig

def run_complete_analysis(nnmf_file_path, resection_file_path, bands=None, output_prefix=""):
    """
    전체 분석을 실행하고 결과를 출력합니다.
    
    Parameters:
    -----------
    nnmf_file_path : str
        NNMF 데이터가 포함된 CSV 파일 경로
    resection_file_path : str
        Resection area 좌표가 포함된 CSV 파일 경로
    bands : list or None
        분석할 주파수 대역 목록 (None이면 모든 대역 분석)
    output_prefix : str
        그래프 제목 접두사
    
    Returns:
    --------
    dict
        분석 결과와 그래프를 포함하는 딕셔너리
    """
    # 네트워크 분석 실행
    analysis_results = analyze_networks_vs_resection(
        nnmf_file_path=nnmf_file_path,
        resection_file_path=resection_file_path,
        bands=bands
    )
    
    # 네트워크 특성 그래프
    properties_fig = plot_network_properties(
        analysis_results, 
        bands=bands,
        title_prefix=output_prefix
    )
    
    # 파워 비교 그래프
    power_fig = plot_power_comparison(
        analysis_results, 
        bands=bands,
        title_prefix=output_prefix
    )
    
    # ROC 곡선 그래프
    roc_fig = plot_roc_curves(
        analysis_results, 
        bands=bands,
        title_prefix=output_prefix
    )
    
    # 요약 출력
    print("=" * 50)
    print(f"{output_prefix}분석 결과 요약:")
    print("=" * 50)
    
    # 각 대역별 요약
    for band in analysis_results.keys():
        print(f"\n[{band.upper()} 대역]")
        print(f"  활성화된 채널 수: IEN={analysis_results[band]['active_counts']['epileptogenic']}, BG={analysis_results[band]['active_counts']['background']}")
        
        if 'focality' in analysis_results[band]:
            print(f"  포컬리티 (Focality): IEN={analysis_results[band]['focality'].get('epileptogenic', 'N/A'):.4f}, BG={analysis_results[band]['focality'].get('background', 'N/A'):.4f}")
        
        if 'overlap' in analysis_results[band]:
            print(f"  절제 영역 오버랩: IEN={analysis_results[band]['overlap'].get('epileptogenic', 'N/A'):.1f}%, BG={analysis_results[band]['overlap'].get('background', 'N/A'):.1f}%")
        
        if 'distance' in analysis_results[band]:
            print(f"  절제 영역과의 거리: IEN={analysis_results[band]['distance'].get('epileptogenic', 'N/A'):.2f}mm, BG={analysis_results[band]['distance'].get('background', 'N/A'):.2f}mm")

        if 'auc' in analysis_results[band]:
            print(f"  절제 영역 예측 AUC: IEN={analysis_results[band]['auc'].get('epileptogenic', 'N/A'):.4f}, BG={analysis_results[band]['auc'].get('background', 'N/A'):.4f}, Entire={analysis_results[band]['auc'].get('entire', 'N/A'):.4f}")
        
        if 'power_comparison' in analysis_results[band] and 'epileptogenic' in analysis_results[band]['power_comparison']:
            p_value = analysis_results[band]['power_comparison']['epileptogenic']['p_value']
            effect = analysis_results[band]['power_comparison']['epileptogenic']['effect_size']
            
            sig_str = "**" if not np.isnan(p_value) and p_value < 0.01 else "*" if not np.isnan(p_value) and p_value < 0.05 else "ns"
            print(f"  파워 비교 (내부 vs 외부): p={p_value:.4f} {sig_str}, 효과 크기={effect:.2f}")
    
    # 대역별 성능 순위 (AUC 기준)
    ien_aucs = [(band, results['auc'].get('epileptogenic', 0)) 
                for band, results in analysis_results.items() 
                if 'auc' in results and 'epileptogenic' in results['auc']]
    ien_aucs.sort(key=lambda x: x[1], reverse=True)
    
    print("\n[대역별 성능 순위 (AUC 기준)]")
    for i, (band, auc_val) in enumerate(ien_aucs):
        print(f"  {i+1}. {band.upper()}: {auc_val:.4f}")
    
    # 최적 대역 결정 (AUC와 Overlap 기준)
    best_band = None
    best_score = 0
    
    for band, results in analysis_results.items():
        if 'auc' in results and 'epileptogenic' in results['auc'] and 'overlap' in results and 'epileptogenic' in results['overlap']:
            auc = results['auc']['epileptogenic']
            overlap = results['overlap']['epileptogenic'] / 100  # normalize to 0-1
            
            # 조합 점수 계산 (AUC와 Overlap의 가중 평균)
            combined_score = 0.7 * auc + 0.3 * overlap
            
            if combined_score > best_score:
                best_score = combined_score
                best_band = band
    
    if best_band:
        print(f"\n[최적 대역] {best_band.upper()}")
        print(f"  - AUC: {analysis_results[best_band]['auc']['epileptogenic']:.4f}")
        print(f"  - 절제 영역 오버랩: {analysis_results[best_band]['overlap']['epileptogenic']:.1f}%")
        print(f"  - 포컬리티: {analysis_results[best_band]['focality']['epileptogenic']:.4f}")
        print(f"  - 절제 영역과의 거리: {analysis_results[best_band]['distance']['epileptogenic']:.2f}mm")
    
    return {
        'results': analysis_results,
        'figures': {
            'properties': properties_fig,
            'power_comparison': power_fig,
            'roc': roc_fig
        },
        'best_band': best_band
    }

def visualize_networks_with_metrics(
    nnmf_file_path, 
    resection_file_path, 
    band='theta',
    percentile_threshold=None,
    scale_factor=0.001):
    """
    NNMF 데이터(IEN 및 background network)와 resection area를 3D로 시각화하고
    네트워크 특성 지표를 함께 표시합니다.
    
    Parameters:
    -----------
    nnmf_file_path : str
        NNMF 데이터가 포함된 CSV 파일 경로
    resection_file_path : str
        Resection area 좌표가 포함된 CSV 파일 경로
    band : str
        시각화할 주파수 대역
    percentile_threshold : int, optional
        임계값으로 사용할 백분위수. None인 경우 논문 방식인 "mean + 1 std" 사용
    scale_factor : float
        Resection 좌표에 적용할 스케일 팩터
    
    Returns:
    --------
    tuple
        (plotly figure, network metrics)
    """
    # 데이터 로드
    nnmf_data = pd.read_csv(nnmf_file_path)
    resection_data = pd.read_csv(resection_file_path)
    
    # 필요한 열 확인
    epileptogenic_col = f'{band}_epileptogenic'
    background_col = f'{band}_background'
    
    # 좌표 및 네트워크 값 추출
    coordinates = nnmf_data[['x_coord', 'y_coord', 'z_coord']].values
    en_network = nnmf_data[epileptogenic_col].values
    bg_network = nnmf_data[background_col].values
    
    # Resection 좌표 추출 및 스케일링
    resection_coords = resection_data.iloc[:, :3].values * scale_factor
    
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
    
    # 네트워크 특성 계산
    # 1. 포컬리티 (Focality)
    if np.sum(en_active) >= 2:
        active_coords = coordinates[en_active]
        distances = []
        for i in range(len(active_coords)):
            for j in range(i+1, len(active_coords)):
                distances.append(np.linalg.norm(active_coords[i] - active_coords[j]))
        en_focality = 1 / np.mean(distances) if distances else 0
    else:
        en_focality = 0
    
    # 2. Resection과의 거리
    channel_distances = []
    for coord in coordinates:
        min_dist = np.min(np.sqrt(np.sum((resection_coords - coord)**2, axis=1)))
        channel_distances.append(min_dist)
    
    # Resection 마스크 (threshold 이내 거리)
    resection_threshold = 10 * scale_factor  # 10mm
    resection_mask = np.array(channel_distances) <= resection_threshold
    
    # 3. Overlap 계산
    if np.sum(en_active) > 0:
        en_overlap = 100 * np.sum(en_active & resection_mask) / np.sum(en_active)
    else:
        en_overlap = 0
    
    # 4. 평균 거리
    if np.sum(en_active) > 0:
        en_distance = np.mean(np.array(channel_distances)[en_active]) / scale_factor  # mm 단위로 변환
    else:
        en_distance = float('inf')
    
    # 5. AUC ROC 계산
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(resection_mask, en_network)
    en_auc = auc(fpr, tpr)
    
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
        
        # 거리 정보 추가
        distance_to_resection = channel_distances[i] / scale_factor  # mm 단위로 변환
        
        hover_text = (
            f"Channel: {channel_name}<br>" +
            f"IEN: {en_network[i]:.4f} (threshold: {en_threshold:.4f})<br>" +
            f"BG: {bg_network[i]:.4f} (threshold: {bg_threshold:.4f})<br>" +
            f"Network: {network_type}<br>" +
            f"Distance to resection: {distance_to_resection:.2f} mm"
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
            hoverinfo="text",
            name="Electrodes"
        )
    )
    
    # IEN 중심 계산 (활성화된 채널이 있는 경우)
    if np.sum(en_active) > 0:
        ien_centroid = np.mean(coordinates[en_active], axis=0)
        
        # # IEN 중심 표시
        # fig.add_trace(
        #     go.Scatter3d(
        #         x=[ien_centroid[0]],
        #         y=[ien_centroid[1]],
        #         z=[ien_centroid[2]],
        #         mode='markers',
        #         marker=dict(
        #             size=15,
        #             color='yellow',
        #             symbol='diamond',
        #             line=dict(color='black', width=1)
        #         ),
        #         name="IEN Centroid"
        #     )
        # )
    
    # 네트워크 특성 정보를 그래프 제목에 추가
    metrics_info = (
        f"Focality: {en_focality:.4f} | " +
        f"Overlap: {en_overlap:.1f}% | " +
        f"Dist: {en_distance:.2f}mm | " +
        f"AUC: {en_auc:.4f}"
    )
    
    # 레이아웃 설정
    fig.update_layout(
        title=f"{band.upper()} Band: Epileptogenic Network & Resection<br><sup>{metrics_info}</sup>",
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
            title="Components",
            itemsizing="constant"
        ),
        height=800,
        width=1000,
        margin=dict(l=0, r=0, b=0, t=60)
    )
    
    # 네트워크 특성 요약
    metrics = {
        'band': band,
        'active_counts': {
            'epileptogenic': np.sum(en_active),
            'background': np.sum(bg_active)
        },
        'thresholds': {
            'epileptogenic': en_threshold,
            'background': bg_threshold
        },
        'metrics': {
            'focality': en_focality,
            'overlap': en_overlap,
            'distance': en_distance,
            'auc': en_auc
        }
    }
    
    # 네트워크 분류 결과 요약 출력
    en_only = np.sum(en_active & ~bg_active)
    bg_only = np.sum(bg_active & ~en_active)
    both = np.sum(en_active & bg_active)
    none = np.sum(~en_active & ~bg_active)
    
    print(f"--- {band.upper()} Band Network Analysis ---")
    print(f"임계값: IEN={en_threshold:.4f}, Background={bg_threshold:.4f}")
    print(f"활성화된 채널: IEN={np.sum(en_active)}, Background={np.sum(bg_active)}")
    print(f"채널 분류: IEN만={en_only}, Background만={bg_only}, 둘 다={both}, 둘 다 아님={none}")
    print(f"네트워크 특성:")
    print(f"  - 포컬리티: {en_focality:.4f}")
    print(f"  - 절제 영역 오버랩: {en_overlap:.1f}%")
    print(f"  - 절제 영역과의 평균 거리: {en_distance:.2f}mm")
    print(f"  - ROC AUC: {en_auc:.4f}")
    
    # IEN 중심과 Resection 중심 간 거리
    if np.sum(en_active) > 0 and len(resection_coords) > 0:
        resection_centroid = np.mean(resection_coords, axis=0)
        centroid_distance = np.linalg.norm(ien_centroid - resection_centroid) / scale_factor  # mm 단위
        print(f"  - IEN 중심과 Resection 중심 간 거리: {centroid_distance:.2f}mm")
        metrics['metrics']['centroid_distance'] = centroid_distance
    
    return fig, metrics

def plot_single_patient_roc(
    nnmf_file_path, 
    resection_file_path, 
    bands=None,
    scale_factor=0.001,
    resection_threshold=10,
    figsize=(14, 6)
):
    """
    단일 환자의 각 주파수 대역별 ROC 곡선을 그립니다.
    
    Parameters:
    -----------
    nnmf_file_path : str
        NNMF 데이터가 포함된 CSV 파일 경로
    resection_file_path : str
        Resection area 좌표가 포함된 CSV 파일 경로
    bands : list or None
        분석할 주파수 대역 목록 (None이면 모든 대역 분석)
    scale_factor : float
        Resection 좌표에 적용할 스케일 팩터
    resection_threshold : float
        Resection과의 거리가 이 값 이내인 채널을 resection 내부로 간주 (mm)
    figsize : tuple
        그래프 크기
        
    Returns:
    --------
    matplotlib.figure.Figure
        생성된 그래프
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    
    # 데이터 로드
    nnmf_data = pd.read_csv(nnmf_file_path)
    resection_data = pd.read_csv(resection_file_path)
    
    # 채널 좌표 추출
    coordinates = nnmf_data[['x_coord', 'y_coord', 'z_coord']].values
    
    # Resection 좌표 추출 및 스케일링
    resection_coords = resection_data.iloc[:, :3].values * scale_factor
    
    # 사용 가능한 모든 대역 확인
    all_bands = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'sb', 'rb']
    
    # 분석할 대역 결정
    bands_to_analyze = bands if bands is not None else all_bands
    
    # Resection과의 최소 거리 계산
    channel_distances = []
    for coord in coordinates:
        min_dist = np.min(np.sqrt(np.sum((resection_coords - coord)**2, axis=1)))
        channel_distances.append(min_dist)
    
    # Resection 마스크 생성 (threshold 이내의 거리에 있는 채널)
    resection_mask = np.array(channel_distances) <= resection_threshold * scale_factor
    
    # ROC 곡선을 그릴 그래프 생성
    fig, ax = plt.subplots(figsize=figsize)
    
    # 각 대역별 ROC 곡선 계산 및 그리기
    for band in bands_to_analyze:
        epileptogenic_col = f'{band}_epileptogenic'
        if epileptogenic_col not in nnmf_data.columns:
            continue
        
        # 네트워크 값 추출
        en_network = nnmf_data[epileptogenic_col].values
        
        # ROC 곡선 계산
        fpr, tpr, _ = roc_curve(resection_mask, en_network)
        roc_auc = auc(fpr, tpr)
        
        # ROC 곡선 그리기
        ax.plot(fpr, tpr, lw=2, label=f'{band.upper()} (AUC = {roc_auc:.4f})')
    
    # 무작위 분류기 라인 추가
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
    
    # 그래프 설정
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('1 - Specificity')
    ax.set_ylabel('Sensitivity')
    ax.set_title('ROC Curves for Resection Prediction')
    ax.legend(loc="lower right")
    
    plt.tight_layout()
    
    return fig, {'resection_mask': resection_mask}

#%%
#분석
nnmf_file_path = 'ysm_mri_1000_1060_nnmf.csv'  # NNMF 결과가 포함된 CSV
resection_file_path = r'C:\hyunbin\Computational Neuroscience\CN data\jjy_mri\resection_voxels_world_coords.csv'
# resection_file_path = 'resection_coords_fsaverage.csv'
#%%
# 전체 분석 실행
analysis_output = run_complete_analysis(
    nnmf_file_path=nnmf_file_path,
    resection_file_path=resection_file_path,
    bands=['delta','theta', 'alpha', 'beta', 'gamma']  # 분석할 대역 지정
)
#%%
# 결과에서 최적 대역 확인
best_band = analysis_output['best_band']
print(f"최적 대역: {best_band}")

# 시각화 그래프 표시
analysis_output['figures']['properties'].show()  # 네트워크 특성 비교
analysis_output['figures']['power_comparison'].show()  # 파워 비교
analysis_output['figures']['roc'].show()  # ROC 곡선
#%%
# 최적 대역에 대한 자세한 3D 시각화
fig, metrics = visualize_networks_with_metrics(
    nnmf_file_path=nnmf_file_path,
    resection_file_path=resection_file_path,
    band=best_band
)
fig.show()
#%%
#ROC 분석석
fig, info = plot_single_patient_roc(
    nnmf_file_path=nnmf_file_path,
    resection_file_path=resection_file_path,
    bands=['delta', 'theta', 'alpha', 'beta', 'gamma', 'sb', 'rb'],
    resection_threshold=10  # mm
)
plt.show()
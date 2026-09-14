# Copyright (c) 2026 DreamWaQ Project
# SPDX-License-Identifier: BSD-3-Clause

"""학습 단계별 지형 생성기 설정.

1단계는 평지·완만한 지형 위주로 낮은 계단만 포함하고,
2단계는 계단·스텝의 비중을 50%까지 올리고 계단 높이와 경사를 함께 높인다.
두 설정의 sub_terrains 비율 합은 각각 1.0이다.
"""

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg

# 두 단계가 공유하는 지형 격자 규격이다.
_TERRAIN_SIZE = (8.0, 8.0)
_BORDER_WIDTH = 70.0
_NUM_ROWS = 10
_NUM_COLS = 20
_HORIZONTAL_SCALE = 0.1
_VERTICAL_SCALE = 0.005
_SLOPE_THRESHOLD = 0.75

# 1단계 지형: 평지·불규칙 지면·완만한 경사 위주로 구성하고 계단은 비중 20%, 최대 높이 0.05 m로 둔다.
PHASE1_TERRAINS_CFG = TerrainGeneratorCfg(
    size=_TERRAIN_SIZE,
    border_width=_BORDER_WIDTH,
    num_rows=_NUM_ROWS,
    num_cols=_NUM_COLS,
    horizontal_scale=_HORIZONTAL_SCALE,
    vertical_scale=_VERTICAL_SCALE,
    slope_threshold=_SLOPE_THRESHOLD,
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.2),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.3, noise_range=(0.01, 0.04), noise_step=0.01, border_width=0.25
        ),
        "pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.15, slope_range=(0.0, 0.15), platform_width=2.0, border_width=0.25
        ),
        "pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.15, slope_range=(0.0, 0.15), platform_width=2.0, border_width=0.25
        ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.1, step_height_range=(0.02, 0.05), step_width=0.35,
            platform_width=3.0, border_width=1.0, holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.1, step_height_range=(0.02, 0.05), step_width=0.35,
            platform_width=3.0, border_width=1.0, holes=False,
        ),
    },
)

# 2단계 지형: 계단·스텝 비중 50%에 최대 높이 0.16 m, 경사는 약 20도까지 올린다.
PHASE2_TERRAINS_CFG = TerrainGeneratorCfg(
    size=_TERRAIN_SIZE,
    border_width=_BORDER_WIDTH,
    num_rows=_NUM_ROWS,
    num_cols=_NUM_COLS,
    horizontal_scale=_HORIZONTAL_SCALE,
    vertical_scale=_VERTICAL_SCALE,
    slope_threshold=_SLOPE_THRESHOLD,
    use_cache=False,
    sub_terrains={
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.02, 0.08), noise_step=0.02, border_width=0.25
        ),
        "pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.15, slope_range=(0.0, 0.36), platform_width=2.0, border_width=0.25
        ),
        "pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.15, slope_range=(0.0, 0.36), platform_width=2.0, border_width=0.25
        ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2, step_height_range=(0.05, 0.16), step_width=0.3,
            platform_width=3.0, border_width=1.0, holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2, step_height_range=(0.05, 0.16), step_width=0.3,
            platform_width=3.0, border_width=1.0, holes=False,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.1, grid_width=0.45, grid_height_range=(0.05, 0.16), platform_width=2.0
        ),
    },
)

# 학습 단계 번호 -> 지형 생성기
PHASE_TERRAINS_CFGS = {1: PHASE1_TERRAINS_CFG, 2: PHASE2_TERRAINS_CFG}

# 학습 단계 번호 -> 지형 난이도 초기 상한
PHASE_INITIAL_TERRAIN_LEVELS = {1: 2, 2: 5}

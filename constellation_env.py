# -*- coding: utf-8 -*-
"""
constellation_env.py: 生成 SCMA 旋转叠加复合星座图的物理环境
"""
import numpy as np

class ConstellationEnv:
    def __init__(self, config):
        self.cfg = config
        # 定义 4x6 的 SCMA 基础拉丁结构矩阵 (1代表该用户占用该资源块)
        # 满足稀疏性：每列2个1，每行3个1
        self.V_matrix = np.array([
            [1, 1, 1, 0, 0, 0],
            [1, 0, 0, 1, 1, 0],
            [0, 1, 0, 1, 0, 1],
            [0, 0, 1, 0, 1, 1]
        ])
        # 记录 12 个非零元素（角度变量）在矩阵中的行列索引
        self.non_zero_indices = list(zip(*np.where(self.V_matrix == 1)))
        
        # 初始化 4 种基础多维星座点 (为简化，使用复数点表示 4-QAM 基础映射)
        self.base_constellation = np.array([-1-1j, -1+1j, 1-1j, 1+1j]) / np.sqrt(2)

    def get_combined_constellation(self, angles):
        """
        根据传入的 12 个智能体的角度状态，计算在 4 个资源块上空中叠加后的 64 个主星座点
        由于 SCMA 的叠加是针对特定资源块的，这里我们主要优化和评估叠加最严重的冲突块
        """
        # 将 12 个角度还原到 4x6 的相位矩阵中
        phase_matrix = np.zeros((self.cfg.K, self.cfg.J))
        for idx, (r, c) in enumerate(self.non_zero_indices):
            phase_matrix[r, c] = angles[idx]
            
        # 产生所有可能的 6 个用户发射符号组合 (4^6 = 4096 种，这里抽取最具代表性的前 64 种复合叠加状态)
        # 为快速评估，我们聚焦于 RE-0 (第一行)，它由 User 0, 1, 2 叠加 (4^3 = 64 个主星座点)
        combined_points = []
        for u0 in self.base_constellation:
            for u1 in self.base_constellation:
                for u2 in self.base_constellation:
                    # 旋转矩阵酉变换 R(theta) * x 在复平面等价于 x * exp(j * theta)
                    x0 = u0 * np.exp(1j * phase_matrix[0, 0])
                    x1 = u1 * np.exp(1j * phase_matrix[0, 1])
                    x2 = u2 * np.exp(1j * phase_matrix[0, 2])
                    
                    # 空中信号叠加
                    y_re0 = x0 + x1 + x2
                    combined_points.append(y_re0)
                    
        return np.array(combined_points)

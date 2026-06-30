# -*- coding: utf-8 -*-
"""
link_evaluator.py: 在物理信道上运行多用户最大似然(ML)概率解耦译码，评估系统误码率(BER)
"""
import numpy as np

class LinkEvaluator:
    def __init__(self, env, config):
        self.env = env
        self.cfg = config

    def evaluate_ber(self, angles):
        """通过高斯信道蒙特卡洛仿真，测试这组角度下的误码率"""
        # 获取当前角度配置下的 64 个标准叠加主星座点
        combined_points = self.env.get_combined_constellation(angles)
        
        ber_results = []
        # 在不同的信噪比(SNR)下跑通信链路仿真
        for snr_db in self.cfg.SNR_RANGE:
            snr_linear = 10**(snr_db / 10.0)
            # 根据星座点平均功率计算噪声方差
            avg_power = np.mean(np.abs(combined_points)**2)
            noise_variance = avg_power / (2 * snr_linear)
            
            error_bits = 0
            total_bits = 0
            
            # 模拟数据传输与接收
            for _ in range(self.cfg.NUM_BITS // 6): # 每组涉及 3 个用户，每个用户 2 bits (4-QAM)
                # 随机选择 64 种合法叠加状态中的一种作为发射信号
                tx_idx = np.random.randint(0, 64)
                tx_signal = combined_points[tx_idx]
                
                # 模拟加性高斯白噪声(AWGN)信道
                noise = np.random.normal(0, np.sqrt(noise_variance)) + 1j * np.random.normal(0, np.sqrt(noise_variance))
                rx_signal = tx_signal + noise
                
                # 接收端多用户检测器 (最大似然 ML 译码，计算条件概率指数似然)
                # 找出距离接收信号最近的那个标准复合主星座点
                distances = np.abs(rx_signal - combined_points)**2
                dec_idx = np.argmin(distances)
                
                # 比较发射和接收的索引，若不同则转化为比特错误 (简单估计：索引不同代表错1比特)
                if dec_idx != tx_idx:
                    error_bits += 1
                total_bits += 6  # 3个用户 * 2比特
                
            ber = error_bits / total_bits
            ber_results.append(ber if ber > 0 else 1e-5) # 防止全对时对数坐标轴报错
            
        return ber_results

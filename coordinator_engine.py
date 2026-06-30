# -*- coding: utf-8 -*-
"""
coordinator_engine.py: 实现多智能体轮询(Round-Robin)协作与 Soft-Sum 打分算法
"""
import numpy as np

class CoordinatorEngine:
    def __init__(self, env, agents, config):
        self.env = env
        self.agents = agents
        self.cfg = config

    def compute_soft_sum_score(self, points):
        """
        核心创新点：负指数连续平滑距离求和算子 (模拟势能场)
        目标是最小化总势能（即最大化星座点之间的距离）
        """
        score = 0.0
        num_points = len(points)
        # 两两计算复合星座点之间的欧氏距离
        for i in range(num_points):
            for j in range(i + 1, num_points):
                dist_sq = np.abs(points[i] - points[j])**2
                # 负指数平滑，距离越近分越高（斥力大），重合时达到峰值
                score += np.exp(-dist_sq / (2 * (self.cfg.SIGMA_0**2)))
        return score

    def compute_med(self, points):
        """传统硬比较打分器：计算最小欧氏距离(MED)"""
        min_dist = float('inf')
        num_points = len(points)
        for i in range(num_points):
            for j in range(i + 1, num_points):
                dist = np.abs(points[i] - points[j])
                if dist < min_dist:
                    min_dist = dist
        return min_dist

    def run_optimization(self):
        """启动多智能体分布式协同演化大循环"""
        print(">>> 多智能体协同自助优化系统启动...")
        
        for epoch in range(self.cfg.MAX_EPOCHS):
            any_agent_changed = False
            
            # 轮询（Round-Robin）机制：每次只激活一个智能体探索，解除交织干扰
            for agent in self.agents:
                # 获取未动作前的全局状态与基准势能得分
                current_angles = [a.current_angle for a in self.agents]
                base_points = self.env.get_combined_constellation(current_angles)
                base_score = self.compute_soft_sum_score(base_points)
                
                # 1. 尝试正向微调
                trial_angle_pos = agent.try_move(direction=1, delta=self.cfg.DELTA_THETA)
                test_angles_pos = current_angles.copy()
                test_angles_pos[agent.agent_id] = trial_angle_pos
                pos_points = self.env.get_combined_constellation(test_angles_pos)
                pos_score = self.compute_soft_sum_score(pos_points)
                
                # 如果正向动作降低了系统总势能，协同层批准该智能体更新
                if pos_score < base_score:
                    agent.update_state(trial_angle_pos)
                    any_agent_changed = True
                    continue
                
                # 2. 若正向失败，尝试反向微调
                trial_angle_neg = agent.try_move(direction=-1, delta=self.cfg.DELTA_THETA)
                test_angles_neg = current_angles.copy()
                test_angles_neg[agent.agent_id] = trial_angle_neg
                neg_points = self.env.get_combined_constellation(test_angles_neg)
                neg_score = self.compute_soft_sum_score(neg_points)
                
                if neg_score < base_score:
                    agent.update_state(trial_angle_neg)
                    any_agent_changed = True
            
            # 打印当前轮次的收敛进度
            current_angles = [a.current_angle for a in self.agents]
            current_points = self.env.get_combined_constellation(current_angles)
            epoch_med = self.compute_med(current_points)
            print(f"Epoch {epoch+1}/{self.cfg.MAX_EPOCHS} - 当前复合星座最小欧氏距离(MED): {epoch_med:.4f}")
            
            # 如果整整一轮大循环中所有智能体都原地静止，说明多智能体系统已找到局部最优，宣告收敛
            if not any_agent_changed:
                print(f">>> 多智能体系统在第 {epoch+1} 轮完全收敛！")
                break
                
        return [a.current_angle for a in self.agents]

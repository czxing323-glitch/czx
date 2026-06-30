# -*- coding: utf-8 -*-
"""
angle_agent.py: 维护单个连续角度变量的优化智能体
"""
class AngleAgent:
    def __init__(self, agent_id, initial_angle):
        self.agent_id = agent_id
        self.current_angle = initial_angle  # 当前决策状态
        
    def try_move(self, direction, delta):
        """生成测试动作：正向(1)或负向(-1)微调步长"""
        return self.current_angle + direction * delta
        
    def update_state(self, new_angle):
        """接收中央协同层的批准指令，更新本地角度"""
        self.current_angle = new_angle

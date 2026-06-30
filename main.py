# -*- coding: utf-8 -*-
"""
main.py: 驱动多智能体协同优化系统，并绘制学术对比图表
"""
import numpy as np
import matplotlib.pyplot as plt
from config import Config
from environment.constellation_env import ConstellationEnv
from agent.angle_agent import AngleAgent
from agent.coordinator_engine import CoordinatorEngine
from environment.link_evaluator import LinkEvaluator

def main():
    # 1. 载入系统配置
    cfg = Config()
    
    # 2. 初始化物理层多用户叠加环境
    env = ConstellationEnv(cfg)
    
    # 3. 设置对比基准：传统人工设计的等差离散角度 (12个变量通通设为固定相位差 0)
    baseline_angles = [0.0] * 12
    initial_points = env.get_combined_constellation(baseline_angles)
    
    # 4. 实例化 12 个自治决策的角度优化智能体，赋予初值
    agents = [AngleAgent(i, 0.0) for i in range(12)]
    
    # 5. 实例化中央协同调度引擎，并启动多智能体协作寻优
    coordinator = CoordinatorEngine(env, agents, cfg)
    optimized_angles = coordinator.run_optimization()
    
    print("\n>>> 多智能体协同寻优完成！最优连续角度配置如下：")
    for idx, angle in enumerate(optimized_angles):
        print(f"Agent {idx}: 旋转角 = {angle:.4f} rad ({np.degrees(angle):.2f}°)")
        
    # 6. 触发物理链路评估器，对优化前后的码本运行多用户译码测试
    evaluator = LinkEvaluator(env, cfg)
    print("\n>>> 正在运行蒙特卡洛物理链路仿真（生成测试曲线）...")
    baseline_ber = evaluator.evaluate_ber(baseline_angles)
    optimized_ber = evaluator.evaluate_ber(optimized_angles)
    
    # 7. 自动生成学术论文级别的性能对比图表
    plt.figure(figsize=(10, 6))
    plt.semilogy(list(cfg.SNR_RANGE), baseline_ber, 'r--o', linewidth=2, label='Traditional Latins (Fixed Angle 0°)')
    plt.semilogy(list(cfg.SNR_RANGE), optimized_ber, 'b-s', linewidth=2, label='Multi-Agent Collaborative Optimized')
    plt.grid(True, which="both", ls="--")
    plt.xlabel('Signal-to-Noise Ratio (SNR) in dB', fontsize=12)
    plt.ylabel('Bit Error Rate (BER)', fontsize=12)
    plt.title('SCMA Constellation Rotation Optimization Performance', fontsize=14)
    plt.legend(fontsize=12)
    
    # 自动保存图片到本地
    plt.savefig('scma_optimization_result.png', dpi=300)
    print("\n>>> 性能对比图已成功保存至本地: 'scma_optimization_result.png'")
    plt.show()

if __name__ == '__main__':
    main()

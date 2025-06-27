import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

class ChainReactionSimulator:
    def __init__(self, N0=1, p=0.4, q=0.3, nu=2.5, max_steps=50, max_neutrons=1e6):
        """
        初始化链式反应模拟器
        
        参数:
        N0: 初始中子数
        p: 单个中子发生反应的概率
        q: 单个中子消失（被吸收或逃逸）的概率
        nu: 每次反应产生的中子数（增殖系数）
        max_steps: 最大模拟时间步长
        max_neutrons: 中子数量上限（防止数值爆炸）
        """
        self.N0 = N0
        self.p = p
        self.q = q
        self.nu = nu
        self.max_steps = max_steps
        self.max_neutrons = max_neutrons
        
        # 验证概率有效性
        if p + q > 1:
            raise ValueError("p + q 不能大于1")
    
    def _step(self, n):
        """执行单个时间步的模拟"""
        if n <= 0 or n > self.max_neutrons:
            return 0
        
        # 计算消失的中子数 (二项分布)
        disappear = np.random.binomial(n, self.q)
        
        # 剩余中子中发生反应的数量 (条件概率)
        remaining = n - disappear
        if remaining > 0:
            react = np.random.binomial(remaining, self.p / (1 - self.q))
        else:
            react = 0
        
        # 计算新产生的中子数
        new_neutrons = np.random.poisson(react * self.nu)  # 泊松分布模拟随机性
        
        # 更新中子总数: 剩余未消失未反应的中子 + 新产生的中子
        next_n = (remaining - react) + new_neutrons
        
        # 处理边界条件
        if next_n > self.max_neutrons:
            return self.max_neutrons
        return max(0, next_n)
    
    def simulate(self):
        """执行完整的模拟过程"""
        neutron_counts = np.zeros(self.max_steps + 1, dtype=int)
        neutron_counts[0] = self.N0
        
        for t in range(1, self.max_steps + 1):
            neutron_counts[t] = self._step(neutron_counts[t-1])
            if neutron_counts[t] <= 0:
                break
        
        return neutron_counts
    
    def multiple_simulations(self, num_sims=100):
        """多次模拟并返回结果"""
        results = []
        for _ in tqdm(range(num_sims)):
            results.append(self.simulate())
        return np.array(results)
    
    @staticmethod
    def plot_results(results, title):
        """可视化模拟结果"""
        plt.figure(figsize=(12, 6))
        
        # 绘制单次模拟曲线
        plt.subplot(1, 2, 1)
        for i in range(min(5, len(results))):
            plt.plot(results[i], label=f'Run {i+1}')
        plt.xlabel('Time Step')
        plt.ylabel('Number of Neutrons')
        plt.title(f'{title}\nSingle Simulations')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 绘制统计结果
        plt.subplot(1, 2, 2)
        mean_counts = np.mean(results, axis=0)
        median_counts = np.median(results, axis=0)
        max_counts = np.max(results, axis=0)
        
        plt.plot(mean_counts, 'b-', label='Mean')
        plt.plot(median_counts, 'g--', label='Median')
        plt.plot(max_counts, 'r:', label='Max')
        
        plt.fill_between(range(len(mean_counts)), 
                         np.percentile(results, 25, axis=0),
                         np.percentile(results, 75, axis=0),
                         color='blue', alpha=0.2, label='IQR')
        
        plt.xlabel('Time Step')
        plt.ylabel('Number of Neutrons')
        plt.title(f'{title}\nStatistical Summary (n={len(results)})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.tight_layout()
        plt.show()


def parameter_analysis():
    """参数分析：研究不同参数对链式反应的影响"""
    # 基准参数
    base_params = {'N0': 1, 'p': 0.4, 'q': 0.3, 'nu': 2.5, 'num_sims': 100}
    
    # 参数变化研究
    studies = [
        # 改变反应概率 p
        ('Varying Reaction Probability (p)', 
         [{'p': 0.3}, {'p': 0.4}, {'p': 0.5}]),
        
        # 改变消失概率 q
        ('Varying Disappearance Probability (q)', 
         [{'q': 0.25}, {'q': 0.3}, {'q': 0.35}]),
        
        # 改变增殖系数 nu
        ('Varying Neutron Multiplicity (nu)', 
         [{'nu': 2.0}, {'nu': 2.5}, {'nu': 3.0}]),
        
        # 改变初始中子数 N0
        ('Varying Initial Neutrons (N0)', 
         [{'N0': 1}, {'N0': 5}, {'N0': 10}])
    ]
    
    for study_title, param_sets in studies:
        plt.figure(figsize=(10, 6))
        
        for params in param_sets:
            # 更新参数
            sim_params = base_params.copy()
            sim_params.update(params)
            
            # 创建模拟器并运行
            sim = ChainReactionSimulator(
                N0=sim_params['N0'],
                p=sim_params['p'],
                q=sim_params['q'],
                nu=sim_params['nu']
            )
            results = sim.multiple_simulations(sim_params['num_sims'])
            mean_counts = np.mean(results, axis=0)
            
            # 生成标签
            label = ', '.join([f'{k}={v}' for k, v in params.items()])
            plt.plot(mean_counts, label=label)
        
        plt.xlabel('Time Step')
        plt.ylabel('Mean Neutron Count (log scale)')
        plt.title(study_title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.tight_layout()
        plt.show()


def critical_condition_analysis():
    """临界条件分析：研究不同临界状态下的链式反应行为"""
    # 定义三种临界状态
    conditions = [
        ('Subcritical (p(nu-1) < q)', 
         {'p': 0.4, 'q': 0.35, 'nu': 2.0}),  # 0.4*(2-1)=0.4 > 0.35 -> 实际超临界
        ('Critical (p(nu-1) ≈ q)', 
         {'p': 0.4, 'q': 0.4, 'nu': 2.0}),    # 0.4*(2-1)=0.4 = 0.4
        ('Supercritical (p(nu-1) > q)', 
         {'p': 0.4, 'q': 0.3, 'nu': 2.0})     # 0.4*(2-1)=0.4 > 0.3
    ]
    
    plt.figure(figsize=(12, 8))
    
    for i, (title, params) in enumerate(conditions):
        # 创建模拟器
        sim = ChainReactionSimulator(N0=1, **params)
        
        # 运行模拟
        results = sim.multiple_simulations(100)
        mean_counts = np.mean(results, axis=0)
        
        # 计算理论曲线
        r = params['p'] * (params['nu'] - 1) - params['q']
        theoretical = [1]
        for t in range(1, len(mean_counts)):
            theoretical.append(theoretical[-1] * np.exp(r))
        
        # 绘制结果
        plt.subplot(2, 2, i+1)
        for j in range(min(5, len(results))):
            plt.plot(results[j], alpha=0.5)
        plt.plot(mean_counts, 'k-', linewidth=2, label='Mean')
        plt.plot(theoretical, 'r--', label='Theoretical')
        
        plt.title(f'{title}\np={params["p"]}, q={params["q"]}, nu={params["nu"]}')
        plt.xlabel('Time Step')
        plt.ylabel('Neutrons')
        plt.yscale('log')
        plt.legend()
        plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 设置绘图风格
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    # 示例模拟
    print("Running base simulation...")
    base_sim = ChainReactionSimulator(N0=1, p=0.4, q=0.3, nu=2.5)
    results = base_sim.multiple_simulations(num_sims=100)
    base_sim.plot_results(results, "Base Chain Reaction Simulation")
    
    # 参数影响分析
    print("\nRunning parameter analysis...")
    parameter_analysis()
    
    # 临界条件分析
    print("\nRunning criticality analysis...")
    critical_condition_analysis()


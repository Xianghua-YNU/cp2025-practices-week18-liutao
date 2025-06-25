import random
import math
import matplotlib.pyplot as plt
from typing import List, Tuple

# 设置 matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def buffon_experiment(num_trials: int, needle_length: float = 1.0, line_distance: float = 1.0) -> float:
    """
    模拟Buffon投针实验，返回π的估计值

    参数:
    num_trials (int): 投针实验的次数
    needle_length (float): 针的长度，默认为1.0
    line_distance (float): 平行线之间的距离，默认为1.0

    返回:
    float: π的估计值
    """
    if line_distance < needle_length:
        raise ValueError("线间距必须大于等于针的长度")

    hits = 0
    for _ in range(num_trials):
        # 随机生成针的中心到最近线的距离(0到line_distance/2之间)
        distance = random.uniform(0, line_distance / 2)
        # 随机生成针与线的夹角(0到π/2之间)
        angle = random.uniform(0, math.pi / 2)

        # 判断针是否与线相交
        if distance <= (needle_length / 2) * math.sin(angle):
            hits += 1

    # 计算π的估计值
    if hits == 0:
        return float('inf')  # 避免除零错误
    return (2 * needle_length * num_trials) / (line_distance * hits)


def run_experiment_with_different_trials(trial_sizes: List[int], repeats: int = 5) -> List[Tuple[int, float, float]]:
    """
    运行不同实验次数的投针实验，并返回结果

    参数:
    trial_sizes (List[int]): 不同的实验次数列表
    repeats (int): 每个实验次数重复的次数

    返回:
    List[Tuple[int, float, float]]: 包含(实验次数, 平均估计值, 相对误差)的列表
    """
    results = []
    for n in trial_sizes:
        estimates = [buffon_experiment(n) for _ in range(repeats)]
        avg_estimate = sum(estimates) / len(estimates)
        relative_error = abs(avg_estimate - math.pi) / math.pi * 100
        results.append((n, avg_estimate, relative_error))
    return results


def plot_results(results: List[Tuple[int, float, float]]) -> None:
    """
    绘制实验结果图表

    参数:
    results (List[Tuple[int, float, float]]): 包含(实验次数, 平均估计值, 相对误差)的列表
    """
    trial_sizes = [r[0] for r in results]
    avg_estimates = [r[1] for r in results]
    relative_errors = [r[2] for r in results]

    # 创建图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    # 绘制π估计值与实验次数的关系
    ax1.plot(trial_sizes, avg_estimates, 'o-', label='估计值')
    ax1.axhline(y=math.pi, color='r', linestyle='--', label='真实值')
    ax1.set_xscale('log')
    ax1.set_xlabel('实验次数')
    ax1.set_ylabel('π的估计值')
    ax1.set_title('π估计值与实验次数的关系')
    ax1.legend()
    ax1.grid(True)

    # 绘制相对误差与实验次数的关系
    ax2.plot(trial_sizes, relative_errors, 'o-', color='g')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('实验次数')
    ax2.set_ylabel('相对误差 (%)')
    ax2.set_title('相对误差与实验次数的关系')
    ax2.grid(True)

    # 调整子图之间的间距
    plt.subplots_adjust(hspace=0.5)  # 增加垂直间距

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 单次实验
    single_trial_count = 1000000
    pi_estimate = buffon_experiment(single_trial_count)
    print(f"单次实验结果 ({single_trial_count} 次投针):")
    print(f"π的估计值: {pi_estimate:.6f}")
    print(f"真实值: {math.pi:.6f}")
    print(f"相对误差: {abs(pi_estimate - math.pi) / math.pi * 100:.4f}%")

    # 不同实验次数的比较
    trial_sizes = [100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000]
    print("\n不同实验次数的结果比较:")
    results = run_experiment_with_different_trials(trial_sizes)

    # 打印结果表格
    print("\n{:<12} {:<12} {:<12}".format("实验次数", "平均估计值", "相对误差(%)"))
    for n, estimate, error in results:
        print(f"{n:<12} {estimate:.6f}      {error:.4f}")

    # 绘制结果图表
    plot_results(results)


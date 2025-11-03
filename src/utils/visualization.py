import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_results(results, src_vocab_size, tgt_vocab_size, test_results=None, save_dir="./results"):
    os.makedirs(save_dir, exist_ok=True)

    # 计算需要绘制的子图数量
    num_plots = 6  # 基础6个子图
    has_test_results = test_results and any(test_results.values())
    has_pos_encoding_ablation = "no_pos_encoding" in results and "baseline" in results

    if has_test_results:
        num_plots += 1
    if has_pos_encoding_ablation:
        num_plots += 1

    # 根据子图数量确定网格布局
    if num_plots <= 6:
        n_rows, n_cols = 2, 3
        fig_size = (20, 12)
    elif num_plots <= 9:
        n_rows, n_cols = 3, 3
        fig_size = (20, 18)
    else:
        n_rows, n_cols = 4, 3
        fig_size = (20, 24)

    fig = plt.figure(figsize=fig_size)
    plot_index = 1  # 子图索引

    # 1. 训练损失
    ax1 = plt.subplot(n_rows, n_cols, plot_index)
    for exp_name, result in results.items():
        ax1.plot(result["train_losses"], label=exp_name, linewidth=2, marker='o', markersize=4)
    ax1.set_title("Training Loss Curve", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plot_index += 1

    # 2. 验证损失
    ax2 = plt.subplot(n_rows, n_cols, plot_index)
    for exp_name, result in results.items():
        ax2.plot(result["val_losses"], label=exp_name, linewidth=2, marker='s', markersize=4)
    ax2.set_title("Validation Loss Curve", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plot_index += 1

    # 3. 最终困惑度比较
    ax3 = plt.subplot(n_rows, n_cols, plot_index)
    exp_names = list(results.keys())
    perplexities = [results[exp]["final_perplexity"] for exp in exp_names]
    colors = plt.cm.Set3(np.linspace(0, 1, len(exp_names)))
    bars = ax3.bar(exp_names, perplexities, alpha=0.7, color=colors)
    ax3.set_title("Final Perplexity Comparison", fontsize=14, fontweight='bold')
    ax3.set_ylabel("Perplexity (lower is better)")
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

    for bar, ppl in zip(bars, perplexities):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 f'{ppl:.2f}', ha='center', va='bottom', fontweight='bold')
    plot_index += 1

    # 4. 参数量比较
    ax4 = plt.subplot(n_rows, n_cols, plot_index)
    param_counts = [results[exp]["parameters"] / 1e6 for exp in exp_names]
    bars = ax4.bar(exp_names, param_counts, alpha=0.7, color=plt.cm.Pastel1(np.linspace(0, 1, len(exp_names))))
    ax4.set_title("Parameter Count Comparison", fontsize=14, fontweight='bold')
    ax4.set_ylabel("Parameters (Millions)")
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')

    for bar, count in zip(bars, param_counts):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 f'{count:.1f}M', ha='center', va='bottom', fontweight='bold')
    plot_index += 1

    # 5. 训练时间比较
    ax5 = plt.subplot(n_rows, n_cols, plot_index)
    training_times = [results[exp]["training_time"] for exp in exp_names]
    bars = ax5.bar(exp_names, training_times, alpha=0.7, color=plt.cm.Set2(np.linspace(0, 1, len(exp_names))))
    ax5.set_title("Training Time Comparison", fontsize=14, fontweight='bold')
    ax5.set_ylabel("Time (seconds)")
    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right')

    for bar, time_val in zip(bars, training_times):
        ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 f'{time_val:.0f}s', ha='center', va='bottom', fontweight='bold')
    plot_index += 1

    # 6. 性能vs参数量散点图
    ax6 = plt.subplot(n_rows, n_cols, plot_index)
    for i, exp_name in enumerate(exp_names):
        ax6.scatter(param_counts[i], perplexities[i], s=100, alpha=0.7,
                    label=exp_name, color=colors[i])
        ax6.annotate(exp_name, (param_counts[i], perplexities[i]),
                     xytext=(5, 5), textcoords='offset points', fontsize=9)

    ax6.set_title("Performance vs Model Size", fontsize=14, fontweight='bold')
    ax6.set_xlabel("Parameters (Millions)")
    ax6.set_ylabel("Perplexity")
    ax6.grid(True, alpha=0.3)
    plot_index += 1

    # 7. 测试集性能比较（如果有测试结果）
    if has_test_results and plot_index <= n_rows * n_cols:
        ax7 = plt.subplot(n_rows, n_cols, plot_index)
        test_years = sorted(list(next(iter(test_results.values())).keys()))

        x = np.arange(len(test_years))
        width = 0.8 / len(exp_names)

        for i, exp_name in enumerate(exp_names):
            if exp_name in test_results:
                perplexities = [test_results[exp_name].get(year, 0) for year in test_years]
                ax7.bar(x + i * width, perplexities, width, label=exp_name, alpha=0.7)

        ax7.set_title("Test Set Performance", fontsize=14, fontweight='bold')
        ax7.set_xlabel("Test Set Year")
        ax7.set_ylabel("Perplexity")
        ax7.set_xticks(x + width * (len(exp_names) - 1) / 2)
        ax7.set_xticklabels([f"TST{year}" for year in test_years])
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        plot_index += 1

    # 8. 位置编码消融影响（如果有）
    if has_pos_encoding_ablation and plot_index <= n_rows * n_cols:
        ax8 = plt.subplot(n_rows, n_cols, plot_index)
        models_to_compare = ["baseline", "no_pos_encoding"]
        perplexities = [results[model]["final_perplexity"] for model in models_to_compare]
        bars = ax8.bar(models_to_compare, perplexities, alpha=0.7, color=['blue', 'red'])
        ax8.set_title("Position Encoding Ablation", fontsize=14, fontweight='bold')
        ax8.set_ylabel("Perplexity")

        for bar, ppl in zip(bars, perplexities):
            ax8.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                     f'{ppl:.2f}', ha='center', va='bottom', fontweight='bold')
        plot_index += 1

    plt.tight_layout()
    plt.savefig(f"{save_dir}/transformer_ablation_results.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_dir}/transformer_ablation_results.pdf", bbox_inches='tight')
    plt.show()

    # 保存详细结果表格
    save_detailed_results(results, test_results, save_dir)


def save_detailed_results(results, test_results, save_dir):
    """保存详细结果到CSV文件"""
    data = []
    for exp_name, result in results.items():
        row = {
            'Model': exp_name,
            'Description': result['config']['description'],
            'Parameters': result['parameters'],
            'Parameters_M': result['parameters'] / 1e6,
            'Final_Train_Loss': result['train_losses'][-1],
            'Final_Val_Loss': result['val_losses'][-1],
            'Final_Perplexity': result['final_perplexity'],
            'Training_Time_Seconds': result['training_time'],
            'd_model': result['config']['d_model'],
            'n_layers': result['config']['n_layers'],
            'n_heads': result['config']['n_heads'],
            'd_ff': result['config']['d_ff'],
            'No_Positional_Encoding': result['config'].get('no_pos_encoding', False)
        }

        # 添加测试集结果
        if test_results and exp_name in test_results:
            for test_year, perplexity in test_results[exp_name].items():
                row[f'Test_{test_year}_Perplexity'] = perplexity

        data.append(row)

    df = pd.DataFrame(data)
    df = df.sort_values('Final_Perplexity')
    df.to_csv(f"{save_dir}/detailed_results.csv", index=False)

    # 保存README格式的结果
    with open(f"{save_dir}/results_summary.md", "w") as f:
        f.write("# Transformer Ablation Study Results\n\n")
        f.write("## Performance Summary\n\n")
        f.write("| Model | Parameters | Val Loss | Perplexity | Training Time | Description |\n")
        f.write("|-------|------------|----------|------------|---------------|-------------|\n")

        for _, row in df.iterrows():
            f.write(f"| {row['Model']} | {row['Parameters_M']:.1f}M | {row['Final_Val_Loss']:.4f} | "
                    f"{row['Final_Perplexity']:.2f} | {row['Training_Time_Seconds']:.0f}s | "
                    f"{row['Description']} |\n")

        # 添加测试集结果
        if any('Test_' in col for col in df.columns):
            f.write("\n## Test Set Performance\n\n")
            test_cols = [col for col in df.columns if 'Test_' in col]
            header = "| Model | " + " | ".join(
                [col.replace('Test_', '').replace('_Perplexity', '') for col in test_cols]) + " |\n"
            f.write(header)
            f.write("|-------|" + "|".join(["----------" for _ in test_cols]) + "|\n")

            for _, row in df.iterrows():
                test_values = " | ".join([f"{row[col]:.2f}" if not pd.isna(row[col]) else "N/A" for col in test_cols])
                f.write(f"| {row['Model']} | {test_values} |\n")

        f.write("\n## Key Findings\n\n")
        if len(df) > 0:
            best_model = df.iloc[0]
            f.write(f"- **Best Model**: {best_model['Model']} (Perplexity: {best_model['Final_Perplexity']:.2f})\n")
            f.write(f"- **Most Efficient**: {df.loc[df['Parameters_M'].idxmin()]['Model']} "
                    f"({df['Parameters_M'].min():.1f}M parameters)\n")
            f.write(f"- **Fastest Training**: {df.loc[df['Training_Time_Seconds'].idxmin()]['Model']} "
                    f"({df['Training_Time_Seconds'].min():.0f} seconds)\n")

        if 'No_Positional_Encoding' in df.columns:
            pos_encoding_models = df[df['No_Positional_Encoding'] == True]
            if len(pos_encoding_models) > 0:
                baseline_models = df[df['No_Positional_Encoding'] == False]
                if len(baseline_models) > 0:
                    pos_encoding_impact = (pos_encoding_models.iloc[0]['Final_Perplexity'] -
                                           baseline_models['Final_Perplexity'].min()) / \
                                          baseline_models['Final_Perplexity'].min() * 100
                    f.write(
                        f"- **Position Encoding Impact**: Removing positional encoding increases perplexity by {pos_encoding_impact:.1f}%\n")
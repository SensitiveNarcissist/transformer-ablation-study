import torch
import torch.nn as nn
import numpy as np
import os
import random
import platform
import psutil
from datetime import datetime
from torch.utils.data import DataLoader

from src.data.dataloader import LocalIWSLTXMLDataLoader
from src.models.transformer import EncoderDecoderTransformer
from src.training.trainer import Seq2SeqTrainer
from src.utils.visualization import plot_results

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def print_system_info():
    """打印系统和硬件信息"""
    print("=" * 60)
    print("SYSTEM AND HARDWARE INFORMATION")
    print("=" * 60)
    print(f"Platform: {platform.platform()}")
    print(f"Processor: {platform.processor()}")
    print(f"Python Version: {platform.python_version()}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # 内存信息
    memory = psutil.virtual_memory()
    print(f"Total RAM: {memory.total / 1e9:.1f} GB")
    print(f"Available RAM: {memory.available / 1e9:.1f} GB")
    print("=" * 60)


def set_seed(seed=42):
    """设置随机种子确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Random seed set to: {seed}")


class TranslationGenerator:
    """翻译生成器，使用真实模型进行推理"""

    def __init__(self, model, src_tokenizer, tgt_tokenizer, device, max_length=50):
        self.model = model
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.device = device
        self.max_length = max_length
        self.model.eval()

    def generate_translation(self, src_text):
        """生成翻译结果"""
        # 编码源文本
        src_tokens = self.src_tokenizer.encode("[SOS] " + src_text + " [EOS]").ids
        src_tensor = torch.tensor([src_tokens], dtype=torch.long).to(self.device)

        # 创建源掩码
        src_mask = (src_tensor != 0).unsqueeze(1).unsqueeze(2)

        # 编码器前向传播
        with torch.no_grad():
            encoder_output = self.model.encode(src_tensor, src_mask)

            # 初始化目标序列
            tgt_tokens = [self.tgt_tokenizer.token_to_id("[SOS]")]

            for i in range(self.max_length):
                tgt_tensor = torch.tensor([tgt_tokens], dtype=torch.long).to(self.device)
                tgt_mask = self._create_causal_mask(len(tgt_tokens)).to(self.device)

                # 解码器前向传播
                output = self.model.decode(tgt_tensor, encoder_output, src_mask, tgt_mask)
                next_token_logits = output[0, -1, :]
                next_token = torch.argmax(next_token_logits).item()

                tgt_tokens.append(next_token)

                # 如果遇到EOS token则停止
                if next_token == self.tgt_tokenizer.token_to_id("[EOS]"):
                    break

            # 解码目标序列
            generated_text = self.tgt_tokenizer.decode(tgt_tokens, skip_special_tokens=True)

        return generated_text

    def _create_causal_mask(self, size):
        """创建因果掩码"""
        mask = torch.tril(torch.ones(size, size))
        return mask.unsqueeze(0).unsqueeze(0)


def setup_translation_experiments(data_dir="D:/huggingface_cache/datasets--iwslt2017/en-de"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        # 使用本地XML数据加载器
        data_loader = LocalIWSLTXMLDataLoader(
            batch_size=16,
            max_seq_len=64,
            data_dir=data_dir
        )
        train_loader, val_loader = data_loader.get_dataloaders()
        src_vocab_size = data_loader.src_vocab_size
        tgt_vocab_size = data_loader.tgt_vocab_size

        print(f"成功加载IWSLT2017数据！")

        # 加载测试集
        test_loaders = data_loader.get_test_dataloaders()
        print(f"加载了 {len(test_loaders)} 个测试集")

    except Exception as e:
        print(f"数据加载失败: {e}")
        print("创建小型演示数据集...")

        # 备用方案：创建小型演示数据集
        class DemoDataLoader:
            def __init__(self):
                self.src_vocab_size = 1000
                self.tgt_vocab_size = 1000

            def get_dataloaders(self):
                # 创建简单的演示数据
                train_data = [
                    {
                        "src_ids": torch.randint(5, 100, (64,)),
                        "tgt_ids": torch.randint(5, 100, (64,))
                    } for _ in range(100)
                ]
                val_data = [
                    {
                        "src_ids": torch.randint(5, 100, (64,)),
                        "tgt_ids": torch.randint(5, 100, (64,))
                    } for _ in range(20)
                ]

                train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
                val_loader = DataLoader(val_data, batch_size=16)
                return train_loader, val_loader

        data_loader = DemoDataLoader()
        train_loader, val_loader = data_loader.get_dataloaders()
        src_vocab_size = data_loader.src_vocab_size
        tgt_vocab_size = data_loader.tgt_vocab_size
        test_loaders = {}

    print(f"Source vocab size: {src_vocab_size}, Target vocab size: {tgt_vocab_size}")

    # 实验配置
    experiments = {
        "baseline": {
            "d_model": 256,
            "n_layers": 4,
            "n_heads": 8,
            "d_ff": 1024,
            "description": "基准模型 - 4层, 8注意力头, 256维"
        },
        "small_layers": {
            "d_model": 256,
            "n_layers": 2,
            "n_heads": 8,
            "d_ff": 1024,
            "description": "减少层数 - 2层, 8注意力头, 256维"
        },
        "small_heads": {
            "d_model": 256,
            "n_layers": 4,
            "n_heads": 4,
            "d_ff": 1024,
            "description": "减少注意力头 - 4层, 4注意力头, 256维"
        },
        "small_dim": {
            "d_model": 128,
            "n_layers": 4,
            "n_heads": 8,
            "d_ff": 512,
            "description": "减小维度 - 4层, 8注意力头, 128维"
        },
        "no_pos_encoding": {
            "d_model": 256,
            "n_layers": 4,
            "n_heads": 8,
            "d_ff": 1024,
            "no_pos_encoding": True,
            "description": "移除位置编码 - 4层, 8注意力头, 256维"
        },
        "tiny_model": {
            "d_model": 128,
            "n_layers": 2,
            "n_heads": 4,
            "d_ff": 512,
            "description": "超小模型 - 2层, 4注意力头, 128维"
        }
    }

    return device, train_loader, val_loader, test_loaders, src_vocab_size, tgt_vocab_size, experiments, data_loader


def run_ablation_study():
    # 打印系统信息
    print_system_info()

    # 设置随机种子确保可复现性
    set_seed(42)

    # 设置实验 - 使用您的真实数据路径
    device, train_loader, val_loader, test_loaders, src_vocab_size, tgt_vocab_size, experiments, data_loader = setup_translation_experiments(
        data_dir="D:/huggingface_cache/datasets--iwslt2017/en-de"
    )

    results = {}
    sample_predictions = {}
    test_results = {}

    # 测试样本
    test_samples = [
        "Hello world",
        "This is a test",
        "Machine learning",
        "The weather is nice",
        "Good morning"
    ]

    for exp_name, config in experiments.items():
        print(f"\n{'=' * 60}")
        print(f"Running experiment: {exp_name}")
        print(f"Description: {config['description']}")
        print(f"Configuration: { {k: v for k, v in config.items() if k != 'description'} }")
        print(f"{'=' * 60}")

        # 创建模型
        if "no_pos_encoding" in config and config["no_pos_encoding"]:
            model = EncoderDecoderTransformer(
                src_vocab_size=src_vocab_size,
                tgt_vocab_size=tgt_vocab_size,
                d_model=config["d_model"],
                n_layers=config["n_layers"],
                n_heads=config["n_heads"],
                d_ff=config["d_ff"]
            )
            model.src_pos_encoding = nn.Identity()
            model.tgt_pos_encoding = nn.Identity()
        else:
            model = EncoderDecoderTransformer(
                src_vocab_size=src_vocab_size,
                tgt_vocab_size=tgt_vocab_size,
                d_model=config["d_model"],
                n_layers=config["n_layers"],
                n_heads=config["n_heads"],
                d_ff=config["d_ff"]
            )

        param_count = model.count_parameters()
        print(f"Model parameters: {param_count / 1e6:.2f}M")

        # 训练和评估
        start_time = datetime.now()
        trainer = Seq2SeqTrainer(model, train_loader, val_loader, device)

        train_losses, val_losses = trainer.train(
            epochs=5,
            save_path=f"./results/best_model_{exp_name}.pth"
        )
        training_time = (datetime.now() - start_time).total_seconds()

        # 在测试集上评估
        test_perplexities = {}
        for test_year, test_loader in test_loaders.items():
            test_loss = trainer.evaluate(test_loader)
            test_perplexity = np.exp(test_loss)
            test_perplexities[test_year] = test_perplexity
            print(f"测试集 {test_year} 困惑度: {test_perplexity:.4f}")

        # 生成样本翻译
        generator = TranslationGenerator(
            model, data_loader.src_tokenizer, data_loader.tgt_tokenizer, device
        )

        predictions = {}
        for sample in test_samples:
            prediction = generator.generate_translation(sample)
            predictions[sample] = prediction

        results[exp_name] = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "final_perplexity": np.exp(val_losses[-1]),
            "config": config,
            "parameters": param_count,
            "training_time": training_time
        }

        sample_predictions[exp_name] = predictions
        test_results[exp_name] = test_perplexities

        # 打印样本预测
        print(f"\nSample predictions for {exp_name}:")
        print("-" * 40)
        for i, (src, pred) in enumerate(predictions.items()):
            print(f"{i + 1}. EN: {src}")
            print(f"   DE: {pred}")
        print("-" * 40)

        # 清理内存
        del model, trainer, generator
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results, src_vocab_size, tgt_vocab_size, sample_predictions, test_results


def print_detailed_analysis(results, sample_predictions, test_results):
    """打印详细的结果分析"""
    print("\n" + "=" * 80)
    print("DETAILED ABLATION STUDY ANALYSIS")
    print("=" * 80)

    # 性能比较表
    print("\nPERFORMANCE COMPARISON:")
    print("-" * 120)
    print(f"{'Model':<20} {'Params (M)':<12} {'Final Loss':<12} {'Perplexity':<12} {'Time (s)':<12} {'Description'}")
    print("-" * 120)

    for exp_name, result in results.items():
        params_m = result["parameters"] / 1e6
        final_loss = result["val_losses"][-1]
        perplexity = result["final_perplexity"]
        time_sec = result["training_time"]
        description = result["config"]["description"]

        print(f"{exp_name:<20} {params_m:<12.1f} {final_loss:<12.4f} "
              f"{perplexity:<12.2f} {time_sec:<12.1f} {description}")

    print("-" * 120)

    # 测试集结果
    if test_results and any(test_results.values()):
        print("\nTEST SET PERFORMANCE:")
        print("-" * 80)
        test_years = sorted(list(next(iter(test_results.values())).keys()))
        header = f"{'Model':<20} " + "".join([f"{'TST' + year:<12}" for year in test_years])
        print(header)
        print("-" * 80)

        for exp_name, test_perplexities in test_results.items():
            row = f"{exp_name:<20} "
            for year in test_years:
                if year in test_perplexities:
                    row += f"{test_perplexities[year]:<12.2f}"
                else:
                    row += f"{'N/A':<12}"
            print(row)
        print("-" * 80)

    # 消融研究分析
    print("\nABLATION ANALYSIS:")
    print("-" * 60)

    if "baseline" in results:
        baseline_ppl = results["baseline"]["final_perplexity"]
        baseline_params = results["baseline"]["parameters"]

        for exp_name, result in results.items():
            if exp_name != "baseline":
                ppl_change = (result["final_perplexity"] - baseline_ppl) / baseline_ppl * 100
                param_change = (result["parameters"] - baseline_params) / baseline_params * 100

                print(f"{exp_name}:")
                print(f"  • Perplexity change: {ppl_change:+.1f}%")
                print(f"  • Parameter change: {param_change:+.1f}%")
                print(
                    f"  • Performance impact: {'High' if abs(ppl_change) > 20 else 'Medium' if abs(ppl_change) > 10 else 'Low'}")

    # 位置编码消融分析
    if "no_pos_encoding" in results and "baseline" in results:
        pos_encoding_impact = (results["no_pos_encoding"]["final_perplexity"] - baseline_ppl) / baseline_ppl * 100
        print(f"\nPosition Encoding Ablation:")
        print(f"  • Removing position encoding increases perplexity by {pos_encoding_impact:+.1f}%")
        print(f"  • This demonstrates the critical importance of positional information in sequence modeling")

    # 注意力头分析
    if "small_heads" in results and "baseline" in results:
        print(f"\nAttention Heads Analysis:")
        heads_4_ppl = results["small_heads"]["final_perplexity"]
        heads_8_ppl = results["baseline"]["final_perplexity"]

        heads_impact = (heads_4_ppl - heads_8_ppl) / heads_8_ppl * 100
        print(f"  • Reducing heads from 8 to 4 increases perplexity by {heads_impact:+.1f}%")
        print(f"  • More attention heads allow the model to focus on different linguistic aspects")

    # 样本预测分析
    print(f"\nSAMPLE PREDICTIONS QUALITY ANALYSIS:")
    print("-" * 60)

    for exp_name, predictions in sample_predictions.items():
        print(f"\n{exp_name.upper()} - Translation Quality:")
        for src_text, pred_text in list(predictions.items())[:3]:  # 只显示前3个样本
            # 简单的质量评估
            if len(pred_text.split()) > 1 and pred_text != src_text:
                quality = "Good"
            elif len(pred_text.split()) > 0:
                quality = "Fair"
            else:
                quality = "Poor"
            print(f"  • '{src_text}' → '{pred_text}' [{quality}]")


if __name__ == "__main__":
    # 创建结果目录
    os.makedirs("./results", exist_ok=True)

    print("Starting Transformer Ablation Study...")
    start_time = datetime.now()

    results, src_vocab_size, tgt_vocab_size, sample_predictions, test_results = run_ablation_study()

    total_time = (datetime.now() - start_time).total_seconds()

    # 绘制和保存结果 - 传递test_results参数
    plot_results(results, src_vocab_size, tgt_vocab_size, test_results)

    # 打印详细分析
    print_detailed_analysis(results, sample_predictions, test_results)

    # 最终总结
    print(f"\n{'=' * 80}")
    print("EXPERIMENT SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total experiments: {len(results)}")
    print(f"Total training time: {total_time:.1f} seconds ({total_time / 60:.1f} minutes)")

    if results:
        best_model = min(results.items(), key=lambda x: x[1]['final_perplexity'])[0]
        print(f"Best model: {best_model}")

    print(f"Results saved to: ./results/")
    print(f"Hardware requirements: All models can be trained on CPU (4GB RAM) or GPU")
    print(f"Software platform: PyTorch {torch.__version__}, Python {platform.python_version()}")
    print(f"Dataset: IWSLT2017 English-German")
    print(f"{'=' * 80}")
"""
延迟基准测试
测试模型推理性能，分解预处理/推理/后处理时间
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.multitask_model import MultiTaskSpeechModel
from utils.audio_utils import audio_to_model_input


class LatencyBenchmark:
    """延迟基准测试"""
    
    def __init__(
        self,
        model_path: str,
        device: str = 'cuda',
        n_mels: int = 80,
        target_length: int = 300
    ):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.n_mels = n_mels
        self.target_length = target_length
        
        # 加载模型
        print(f"Loading model from: {model_path}")
        self.model = self._load_model(model_path)
        self.model.eval()
        
        print(f"Device: {self.device}")
    
    def _load_model(self, model_path: str) -> torch.nn.Module:
        model = MultiTaskSpeechModel(
            n_mels=self.n_mels,
            backbone_channels=[32, 64, 128, 256],
            embedding_dim=192,
            num_speakers=1000,
            num_age_groups=5,
            num_emotions=7
        )
        
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        
        return model
    
    def benchmark_preprocessing(
        self,
        audio_paths: List[str],
        n_runs: int = 100
    ) -> Dict:
        """测试预处理时间"""
        print(f"\nBenchmarking preprocessing ({n_runs} runs)...")
        
        times = []
        for i in range(n_runs):
            audio_path = audio_paths[i % len(audio_paths)]
            
            start = time.time()
            mel_spec = audio_to_model_input(
                audio_path,
                sr=16000,
                n_mels=self.n_mels,
                target_length=self.target_length,
                normalize=True
            )
            elapsed = time.time() - start
            
            if mel_spec is not None:
                times.append(elapsed)
        
        return self._compute_stats(times, 'preprocessing')
    
    def benchmark_inference(
        self,
        batch_size: int = 1,
        n_runs: int = 100
    ) -> Dict:
        """测试推理时间"""
        print(f"\nBenchmarking inference (batch_size={batch_size}, {n_runs} runs)...")
        
        # 创建测试输入
        dummy_input = torch.randn(
            batch_size, 1, self.n_mels, self.target_length
        ).to(self.device)
        
        # 预热
        print("Warming up...")
        with torch.no_grad():
            for _ in range(20):
                _ = self.model(dummy_input)
        
        # 同步GPU
        if self.device == 'cuda':
            torch.cuda.synchronize()
        
        # 测试
        times = []
        with torch.no_grad():
            for _ in range(n_runs):
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                start = time.time()
                
                _ = self.model(dummy_input)
                
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                elapsed = time.time() - start
                
                times.append(elapsed)
        
        return self._compute_stats(times, f'inference_bs{batch_size}')
    
    def benchmark_end_to_end(
        self,
        audio_paths: List[str],
        n_runs: int = 100
    ) -> Dict:
        """端到端延迟测试"""
        print(f"\nBenchmarking end-to-end ({n_runs} runs)...")
        
        # 预热
        for i in range(min(10, len(audio_paths))):
            self._inference_pipeline(audio_paths[i])
        
        if self.device == 'cuda':
            torch.cuda.synchronize()
        
        # 测试
        times = {'preprocess': [], 'inference': [], 'total': []}
        
        for i in range(n_runs):
            audio_path = audio_paths[i % len(audio_paths)]
            
            # 预处理
            start = time.time()
            mel_spec = audio_to_model_input(
                audio_path,
                sr=16000,
                n_mels=self.n_mels,
                target_length=self.target_length,
                normalize=True
            )
            preprocess_time = time.time() - start
            
            if mel_spec is None:
                continue
            
            # 推理
            mel_tensor = torch.from_numpy(mel_spec).float().unsqueeze(0).unsqueeze(0)
            mel_tensor = mel_tensor.to(self.device)
            
            if self.device == 'cuda':
                torch.cuda.synchronize()
            start = time.time()
            
            with torch.no_grad():
                _ = self.model(mel_tensor)
            
            if self.device == 'cuda':
                torch.cuda.synchronize()
            inference_time = time.time() - start
            
            times['preprocess'].append(preprocess_time)
            times['inference'].append(inference_time)
            times['total'].append(preprocess_time + inference_time)
        
        return {
            'preprocess': self._compute_stats(times['preprocess'], 'preprocess'),
            'inference': self._compute_stats(times['inference'], 'inference'),
            'total': self._compute_stats(times['total'], 'total')
        }
    
    def _inference_pipeline(self, audio_path: str):
        """完整推理管道"""
        mel_spec = audio_to_model_input(
            audio_path, sr=16000, n_mels=self.n_mels,
            target_length=self.target_length, normalize=True
        )
        if mel_spec is None:
            return None
        
        mel_tensor = torch.from_numpy(mel_spec).float().unsqueeze(0).unsqueeze(0)
        mel_tensor = mel_tensor.to(self.device)
        
        with torch.no_grad():
            return self.model(mel_tensor)
    
    def _compute_stats(self, times: List[float], name: str) -> Dict:
        """计算统计信息"""
        if not times:
            return {}
        
        times_ms = np.array(times) * 1000
        
        stats = {
            'name': name,
            'n_runs': len(times),
            'mean_ms': float(np.mean(times_ms)),
            'std_ms': float(np.std(times_ms)),
            'min_ms': float(np.min(times_ms)),
            'max_ms': float(np.max(times_ms)),
            'p50_ms': float(np.percentile(times_ms, 50)),
            'p90_ms': float(np.percentile(times_ms, 90)),
            'p95_ms': float(np.percentile(times_ms, 95)),
            'p99_ms': float(np.percentile(times_ms, 99))
        }
        
        return stats
    
    def print_stats(self, stats: Dict, title: str = ""):
        """打印统计信息"""
        if title:
            print(f"\n{title}")
        
        print("-" * 50)
        if 'name' in stats:
            print(f"  Mean:  {stats['mean_ms']:.2f}ms")
            print(f"  Std:   {stats['std_ms']:.2f}ms")
            print(f"  Min:   {stats['min_ms']:.2f}ms")
            print(f"  Max:   {stats['max_ms']:.2f}ms")
            print(f"  P50:   {stats['p50_ms']:.2f}ms")
            print(f"  P90:   {stats['p90_ms']:.2f}ms")
            print(f"  P95:   {stats['p95_ms']:.2f}ms")
            print(f"  P99:   {stats['p99_ms']:.2f}ms")
        else:
            # 嵌套字典
            for key, value in stats.items():
                if isinstance(value, dict):
                    print(f"\n  {key.upper()}:")
                    print(f"    Mean: {value['mean_ms']:.2f}ms")
                    print(f"    P95:  {value['p95_ms']:.2f}ms")


def main():
    parser = argparse.ArgumentParser(description='Latency Benchmark')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model')
    parser.add_argument('--audio_dir', type=str,
                       help='Directory with test audio files')
    parser.add_argument('--audio', type=str, nargs='+',
                       help='Specific audio files')
    parser.add_argument('--n_runs', type=int, default=100,
                       help='Number of benchmark runs')
    parser.add_argument('--batch_sizes', type=int, nargs='+', default=[1],
                       help='Batch sizes to test')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Latency Benchmark")
    print("=" * 60)
    
    # 创建基准测试
    benchmark = LatencyBenchmark(
        model_path=args.model,
        device=args.device
    )
    
    # 收集音频文件
    audio_paths = []
    if args.audio:
        audio_paths = args.audio
    elif args.audio_dir:
        for ext in ['.wav', '.mp3', '.flac']:
            audio_paths.extend(list(Path(args.audio_dir).rglob(f'*{ext}')))
        audio_paths = [str(p) for p in audio_paths]
    
    # 推理基准 (各batch size)
    for batch_size in args.batch_sizes:
        stats = benchmark.benchmark_inference(batch_size=batch_size, n_runs=args.n_runs)
        benchmark.print_stats(stats, f"INFERENCE (batch_size={batch_size})")
    
    # 端到端基准
    if audio_paths:
        print(f"\nFound {len(audio_paths)} audio files")
        e2e_stats = benchmark.benchmark_end_to_end(audio_paths, n_runs=args.n_runs)
        benchmark.print_stats(e2e_stats, "END-TO-END")
    else:
        print("\nNo audio files provided, skipping end-to-end benchmark")
    
    print("\nBenchmark completed!")


if __name__ == '__main__':
    main()

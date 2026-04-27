#!/usr/bin/env python3
"""
MT9655端侧Demo
支持: TFLite推理、内存优化、多线程
"""

import os
import sys
import argparse
import time
import threading
from pathlib import Path
from typing import Dict, Optional, List
import numpy as np

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.audio_utils import audio_to_model_input, load_audio


# 标签定义
EMOTION_LABELS = ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']
GENDER_LABELS = ['female', 'male']
AGE_GROUPS = ['child/teen', 'young adult', 'middle age', 'senior', 'elderly']


class MT9655Analyzer:
    """
    MT9655端侧分析器
    使用TFLite进行推理
    """
    
    def __init__(
        self,
        model_path: str,
        n_mels: int = 80,
        target_length: int = 300,
        num_threads: int = 4,
        use_xnnpack: bool = True
    ):
        self.n_mels = n_mels
        self.target_length = target_length
        
        # 加载TFLite模型
        self.interpreter = self._load_model(
            model_path,
            num_threads=num_threads,
            use_xnnpack=use_xnnpack
        )
        
        # 获取输入输出详情
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        print(f"Model loaded: {model_path}")
        print(f"Input shape: {self.input_details[0]['shape']}")
        print(f"Output shapes: {[d['shape'] for d in self.output_details]}")
    
    def _load_model(self, model_path: str, num_threads: int, use_xnnpack: bool):
        """加载TFLite模型"""
        try:
            import tensorflow as tf
        except ImportError:
            try:
                import tflite_runtime.interpreter as tflite
                tf = None
            except ImportError:
                print("Error: Neither tensorflow nor tflite-runtime installed")
                raise
        
        # 创建解释器
        if tf:
            # 使用TensorFlow
            interpreter = tf.lite.Interpreter(
                model_path=model_path,
                num_threads=num_threads,
                experimental_use_xnnpack=use_xnnpack
            )
        else:
            # 使用TFLite Runtime
            interpreter = tflite.Interpreter(
                model_path=model_path,
                num_threads=num_threads
            )
        
        interpreter.allocate_tensors()
        return interpreter
    
    def preprocess(self, audio_path: str) -> Optional[np.ndarray]:
        """预处理音频"""
        mel_spec = audio_to_model_input(
            audio_path,
            sr=16000,
            n_mels=self.n_mels,
            target_length=self.target_length,
            normalize=True
        )
        
        if mel_spec is None:
            return None
        
        # 添加batch和channel维度
        # (n_mels, time) -> (1, 1, n_mels, time)
        mel_spec = mel_spec[np.newaxis, np.newaxis, :, :]
        
        # 转换为INT8 (如果模型需要)
        input_dtype = self.input_details[0]['dtype']
        if input_dtype == np.int8:
            # 量化到INT8
            input_scale = self.input_details[0]['quantization'][0]
            input_zero_point = self.input_details[0]['quantization'][1]
            mel_spec = (mel_spec / input_scale + input_zero_point).astype(np.int8)
        elif input_dtype == np.float32:
            mel_spec = mel_spec.astype(np.float32)
        
        return mel_spec
    
    def inference(self, input_data: np.ndarray) -> Dict[str, np.ndarray]:
        """执行推理"""
        # 设置输入
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        
        # 运行推理
        self.interpreter.invoke()
        
        # 获取输出
        outputs = {}
        for i, detail in enumerate(self.output_details):
            output = self.interpreter.get_tensor(detail['index'])
            outputs[f'output_{i}'] = output
        
        return outputs
    
    def analyze(self, audio_path: str) -> Dict:
        """分析音频"""
        # 预处理
        start_time = time.time()
        input_data = self.preprocess(audio_path)
        
        if input_data is None:
            return {'error': 'Failed to load audio'}
        
        preprocess_time = time.time() - start_time
        
        # 推理
        start_time = time.time()
        outputs = self.inference(input_data)
        inference_time = time.time() - start_time
        
        # 解析结果
        results = self._parse_outputs(outputs)
        results['preprocess_time'] = preprocess_time
        results['inference_time'] = inference_time
        results['total_time'] = preprocess_time + inference_time
        
        return results
    
    def _parse_outputs(self, outputs: Dict[str, np.ndarray]) -> Dict:
        """解析模型输出"""
        results = {}
        
        # 假设输出顺序: emotion, gender, age_cls, age_reg, speaker_emb
        # 根据实际模型调整
        
        output_list = list(outputs.values())
        
        # 情绪 (softmax)
        if len(output_list) > 0:
            emotion_logits = output_list[0][0]
            emotion_probs = self._softmax(emotion_logits)
            emotion_idx = np.argmax(emotion_probs)
            
            results['emotion'] = {
                'label': EMOTION_LABELS[emotion_idx],
                'index': int(emotion_idx),
                'confidence': float(emotion_probs[emotion_idx]),
                'probabilities': {
                    EMOTION_LABELS[i]: float(emotion_probs[i])
                    for i in range(len(EMOTION_LABELS))
                }
            }
        
        # 性别 (softmax)
        if len(output_list) > 1:
            gender_logits = output_list[1][0]
            gender_probs = self._softmax(gender_logits)
            gender_idx = np.argmax(gender_probs)
            
            results['gender'] = {
                'label': GENDER_LABELS[gender_idx],
                'index': int(gender_idx),
                'confidence': float(gender_probs[gender_idx])
            }
        
        # 年龄回归
        if len(output_list) > 2:
            age_value = float(output_list[2][0])
            age_group_idx = min(int(age_value / 20), 4)
            
            results['age'] = {
                'estimated_years': age_value,
                'group': AGE_GROUPS[age_group_idx],
                'group_index': age_group_idx
            }
        
        # 说话人嵌入
        if len(output_list) > 3:
            embedding = output_list[3][0]
            results['speaker'] = {
                'embedding_dim': len(embedding),
                'embedding_norm': float(np.linalg.norm(embedding))
            }
        
        return results
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax计算"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    def benchmark(self, audio_path: str, n_runs: int = 100) -> Dict:
        """性能基准测试"""
        print(f"\nRunning benchmark ({n_runs} iterations)...")
        
        # 预热
        input_data = self.preprocess(audio_path)
        for _ in range(10):
            self.inference(input_data)
        
        # 测试
        times = []
        for _ in range(n_runs):
            start = time.time()
            self.inference(input_data)
            times.append(time.time() - start)
        
        times = np.array(times)
        
        return {
            'mean_ms': float(np.mean(times) * 1000),
            'std_ms': float(np.std(times) * 1000),
            'min_ms': float(np.min(times) * 1000),
            'max_ms': float(np.max(times) * 1000),
            'p50_ms': float(np.percentile(times, 50) * 1000),
            'p95_ms': float(np.percentile(times, 95) * 1000),
            'p99_ms': float(np.percentile(times, 99) * 1000),
        }
    
    def print_results(self, results: Dict):
        """打印结果"""
        print("\n" + "=" * 60)
        print("MT9655 Speech Analysis Results")
        print("=" * 60)
        
        if 'error' in results:
            print(f"Error: {results['error']}")
            return
        
        # 情绪
        if 'emotion' in results:
            emotion = results['emotion']
            print(f"\n🎭 Emotion: {emotion['label'].upper()}")
            print(f"   Confidence: {emotion['confidence']:.2%}")
        
        # 性别
        if 'gender' in results:
            gender = results['gender']
            print(f"\n👤 Gender: {gender['label'].upper()}")
            print(f"   Confidence: {gender['confidence']:.2%}")
        
        # 年龄
        if 'age' in results:
            age = results['age']
            print(f"\n📅 Age: ~{age['estimated_years']:.0f} years")
            print(f"   Group: {age['group']}")
        
        # 性能
        print(f"\n⚡ Performance:")
        print(f"   Preprocess: {results['preprocess_time']*1000:.1f}ms")
        print(f"   Inference:  {results['inference_time']*1000:.1f}ms")
        print(f"   Total:      {results['total_time']*1000:.1f}ms")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='MT9655 Demo for Speech Analysis')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to TFLite model')
    parser.add_argument('--audio', type=str, required=True,
                       help='Audio file path')
    parser.add_argument('--threads', type=int, default=4,
                       help='Number of threads')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run benchmark')
    parser.add_argument('--n_runs', type=int, default=100,
                       help='Number of benchmark runs')
    
    args = parser.parse_args()
    
    # 创建分析器
    analyzer = MT9655Analyzer(
        model_path=args.model,
        num_threads=args.threads
    )
    
    # 分析音频
    results = analyzer.analyze(args.audio)
    analyzer.print_results(results)
    
    # 基准测试
    if args.benchmark:
        bench_results = analyzer.benchmark(args.audio, n_runs=args.n_runs)
        print("\n" + "=" * 60)
        print("Benchmark Results")
        print("=" * 60)
        print(f"Mean:  {bench_results['mean_ms']:.2f}ms")
        print(f"Std:   {bench_results['std_ms']:.2f}ms")
        print(f"Min:   {bench_results['min_ms']:.2f}ms")
        print(f"Max:   {bench_results['max_ms']:.2f}ms")
        print(f"P50:   {bench_results['p50_ms']:.2f}ms")
        print(f"P95:   {bench_results['p95_ms']:.2f}ms")
        print(f"P99:   {bench_results['p99_ms']:.2f}ms")
        print("=" * 60)


if __name__ == '__main__':
    main()

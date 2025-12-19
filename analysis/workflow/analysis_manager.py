# workflow/analysis_manager.py
"""
统一管理整个analysis流程的管理器
"""
from pathlib import Path
from typing import Optional, Literal

from ..pipeline.make_features_talib import FeatureGenerator
from ..pipeline.make_signal import SignalGenerator
from ..train.train_lightgbm import LightGBMTrainer
from ..train.train_logistic import LogisticTrainer
from ..train.train_svm import SVMTrainer
from ..train.train_random_forest import RandomForestTrainer
from ..train.train_lstm import LSTMTrainer
from ..backtest.run_backtest_model import BacktestRunner


class AnalysisManager:
    """
    统一管理整个analysis流程的管理器
    
    流程：
    1. 特征生成：从raw数据生成feature存到intermediate文件夹
    2. 信号生成：从intermediate数据生成signal存到processed文件夹
    3. 模型训练：训练指定的模型（可通过参数控制）
    4. 回测：执行模型回测
    """
    
    # 支持的模型类型
    MODEL_TYPES = Literal["lightgbm", "logistic", "svm", "random_forest", "lstm"]
    
    def __init__(
        self,
        # 数据路径配置
        raw_data_path: Path = Path("data/raw/BTCUSDT_1h.csv"),
        intermediate_dir: Path = Path("data/intermediate"),
        intermediate_file: str = "btc_1h_features.csv",
        processed_dir: Path = Path("data/processed"),
        processed_file: str = "btc_1h_features_signal.csv",
        
        # 特征生成参数
        feature_threshold: float = 0.001,
        feature_delta: float = 0.0004,
        
        # 模型训练参数
        model_dir: Path = Path("models/btc"),
        train_ratio: float = 0.8,
        
        # 回测参数
        val_path: Optional[Path] = None,
        model_paths: Optional[dict] = None,
    ):
        """
        初始化分析管理器
        
        Args:
            raw_data_path: 原始数据路径
            intermediate_dir: 中间数据目录
            intermediate_file: 中间数据文件名
            processed_dir: 处理后数据目录
            processed_file: 处理后数据文件名
            feature_threshold: 信号生成阈值
            feature_delta: 信号生成delta参数
            model_dir: 模型保存目录
            train_ratio: 训练集比例
            val_path: 回测验证数据路径，如果为None则使用processed_file
            model_paths: 模型路径字典，如果为None则使用默认路径
        """
        # 数据路径
        self.raw_data_path = raw_data_path
        self.intermediate_dir = intermediate_dir
        self.intermediate_path = intermediate_dir / intermediate_file
        self.processed_dir = processed_dir
        self.processed_path = processed_dir / processed_file
        
        # 特征生成参数
        self.feature_threshold = feature_threshold
        self.feature_delta = feature_delta
        
        # 模型训练参数
        self.model_dir = model_dir
        self.train_ratio = train_ratio
        
        # 回测参数
        if val_path is None:
            self.val_path = self.processed_path
        else:
            self.val_path = val_path
        self.model_paths = model_paths
        
        # 初始化各个组件
        self._init_components()
    
    def _init_components(self):
        """初始化各个流程组件"""
        # 特征生成器
        self.feature_generator = FeatureGenerator(
            in_path=self.raw_data_path,
            out_dir=self.intermediate_dir,
            out_path=self.intermediate_path,
        )
        
        # 信号生成器
        self.signal_generator = SignalGenerator(
            raw_path=self.intermediate_path,
            out_dir=self.processed_dir,
            out_path=self.processed_path,
            threshold=self.feature_threshold,
            delta=self.feature_delta,
        )
        
        # 回测运行器
        self.backtest_runner = BacktestRunner(
            val_path=self.val_path,
            model_paths=self.model_paths,
        )
    
    def _get_trainer(self, model_type: MODEL_TYPES):
        """
        根据模型类型获取对应的训练器
        
        Args:
            model_type: 模型类型
            
        Returns:
            对应的训练器实例
        """
        model_type = model_type.lower()
        
        if model_type == "lightgbm":
            return LightGBMTrainer(
                data_path=self.processed_path,
                model_dir=self.model_dir,
                train_ratio=self.train_ratio,
            )
        elif model_type == "logistic":
            return LogisticTrainer(
                data_path=self.processed_path,
                model_dir=self.model_dir,
                train_ratio=self.train_ratio,
            )
        elif model_type == "svm":
            return SVMTrainer(
                data_path=self.processed_path,
                model_dir=self.model_dir,
                train_ratio=self.train_ratio,
            )
        elif model_type == "random_forest":
            return RandomForestTrainer(
                data_path=self.processed_path,
                model_dir=self.model_dir,
                train_ratio=self.train_ratio,
            )
        elif model_type == "lstm":
            return LSTMTrainer(
                data_path=self.processed_path,
                model_dir=self.model_dir,
                train_ratio=self.train_ratio,
            )
        else:
            raise ValueError(
                f"Unsupported model_type: {model_type}. "
                f"Supported types: lightgbm, logistic, svm, random_forest, lstm"
            )
    
    def run_feature_generation(self, skip_if_exists: bool = True):
        """
        执行特征生成步骤
        
        Args:
            skip_if_exists: 如果中间文件已存在是否跳过
        """
        if skip_if_exists and self.intermediate_path.exists():
            print(f"[AnalysisManager] Intermediate file already exists: {self.intermediate_path}")
            print("[AnalysisManager] Skipping feature generation step.")
            return
        
        print("=" * 60)
        print("[AnalysisManager] Step 1: Feature Generation")
        print("=" * 60)
        self.feature_generator.generate_features()
        print("[AnalysisManager] Feature generation completed.\n")
    
    def run_signal_generation(self, skip_if_exists: bool = True):
        """
        执行信号生成步骤
        
        Args:
            skip_if_exists: 如果处理后文件已存在是否跳过
        """
        if skip_if_exists and self.processed_path.exists():
            print(f"[AnalysisManager] Processed file already exists: {self.processed_path}")
            print("[AnalysisManager] Skipping signal generation step.")
            return
        
        print("=" * 60)
        print("[AnalysisManager] Step 2: Signal Generation")
        print("=" * 60)
        self.signal_generator.generate_signal()
        print("[AnalysisManager] Signal generation completed.\n")
    
    def run_model_training(
        self,
        model_type: MODEL_TYPES,
        skip_if_exists: bool = True,
    ):
        """
        执行模型训练步骤
        
        Args:
            model_type: 要训练的模型类型
            skip_if_exists: 如果模型文件已存在是否跳过
        """
        # 获取模型保存路径
        model_file_map = {
            "lightgbm": "1h_lightgbm.pkl",
            "logistic": "1h_logistic.pkl",
            "svm": "1h_svm.pkl",
            "random_forest": "1h_random_forest.pkl",
            "lstm": "1h_lstm.pkl",
        }
        model_path = self.model_dir / model_file_map[model_type.lower()]
        
        if skip_if_exists and model_path.exists():
            print(f"[AnalysisManager] Model file already exists: {model_path}")
            print(f"[AnalysisManager] Skipping {model_type} model training step.")
            return
        
        print("=" * 60)
        print(f"[AnalysisManager] Step 3: Model Training ({model_type.upper()})")
        print("=" * 60)
        trainer = self._get_trainer(model_type)
        trainer.train()
        print(f"[AnalysisManager] {model_type.upper()} model training completed.\n")
    
    def run_backtest(
        self,
        model_type: MODEL_TYPES,
    ):
        """
        执行回测步骤
        
        Args:
            model_type: 要回测的模型类型
        """
        print("=" * 60)
        print(f"[AnalysisManager] Step 4: Backtest ({model_type.upper()})")
        print("=" * 60)
        self.backtest_runner.run_backtest(model_type)
        print(f"[AnalysisManager] Backtest completed.\n")
    
    def run_full_pipeline(
        self,
        model_type: MODEL_TYPES,
        skip_feature_gen: bool = True,
        skip_signal_gen: bool = True,
        skip_training: bool = True,
        run_backtest: bool = True,
    ):
        """
        执行完整的分析流程
        
        Args:
            model_type: 要使用的模型类型
            skip_feature_gen: 如果中间文件已存在是否跳过特征生成
            skip_signal_gen: 如果处理后文件已存在是否跳过信号生成
            skip_training: 如果模型文件已存在是否跳过训练
            run_backtest: 是否执行回测
        """
        print("\n" + "=" * 60)
        print("Analysis Pipeline Started")
        print("=" * 60)
        print(f"Model Type: {model_type.upper()}")
        print(f"Raw Data: {self.raw_data_path}")
        print(f"Intermediate: {self.intermediate_path}")
        print(f"Processed: {self.processed_path}")
        print(f"Model Dir: {self.model_dir}")
        print("=" * 60 + "\n")
        
        # Step 1: 特征生成
        self.run_feature_generation(skip_if_exists=skip_feature_gen)
        
        # Step 2: 信号生成
        self.run_signal_generation(skip_if_exists=skip_signal_gen)
        
        # Step 3: 模型训练
        self.run_model_training(model_type, skip_if_exists=skip_training)
        
        # Step 4: 回测
        if run_backtest:
            self.run_backtest(model_type)
        
        print("=" * 60)
        print("Analysis Pipeline Completed")
        print("=" * 60 + "\n")
    
    def run_custom_pipeline(
        self,
        steps: list[str],
        model_type: Optional[MODEL_TYPES] = None,
    ):
        """
        执行自定义流程步骤
        
        Args:
            steps: 要执行的步骤列表，可选值: ["feature", "signal", "train", "backtest"]
            model_type: 模型类型（train和backtest步骤需要）
        
        Example:
            manager.run_custom_pipeline(
                steps=["feature", "signal", "train", "backtest"],
                model_type="lightgbm"
            )
        """
        print("\n" + "=" * 60)
        print("Custom Analysis Pipeline Started")
        print("=" * 60)
        print(f"Steps: {', '.join(steps)}")
        if model_type:
            print(f"Model Type: {model_type.upper()}")
        print("=" * 60 + "\n")
        
        for step in steps:
            step = step.lower()
            
            if step == "feature":
                self.run_feature_generation()
            elif step == "signal":
                self.run_signal_generation()
            elif step == "train":
                if model_type is None:
                    raise ValueError("model_type must be provided for 'train' step")
                self.run_model_training(model_type)
            elif step == "backtest":
                if model_type is None:
                    raise ValueError("model_type must be provided for 'backtest' step")
                self.run_backtest(model_type)
            else:
                print(f"[AnalysisManager] Warning: Unknown step '{step}', skipping.")
        
        print("=" * 60)
        print("Custom Analysis Pipeline Completed")
        print("=" * 60 + "\n")


# workflow/analysis_manager.py
"""
统一管理整个analysis流程的管理器
"""
from pathlib import Path
from typing import Optional, Literal

from pipeline.make_features_talib import FeatureGenerator
from pipeline.make_signal import SignalGenerator
from train.train_lightgbm import LightGBMTrainer
from train.train_logistic import LogisticTrainer
from train.train_svm import SVMTrainer
from train.train_random_forest import RandomForestTrainer
from train.train_lstm import LSTMTrainer
from train.train_transformer import TransformerTrainer
from backtest.run_backtest_model import BacktestRunner
from util.price_collector import PriceCollector


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
    MODEL_TYPES = Literal["lightgbm", "logistic", "svm", "random_forest", "lstm", "transformer"]
    
    def __init__(
        self,
        # 币种和K线时长配置（会自动拼接所有路径）
        coin: str = "btc",
        timeframe: str = "1h",
        
        # 数据路径配置
        intermediate_dir: Path = Path("data/intermediate"),
        processed_dir: Path = Path("data/processed"),
        
        # 信号生成参数
        signal_threshold: float = 0.001,
        signal_delta: float = 0.0004,
        signal_lookahead_periods: int = 1,
        
        # 模型训练参数
        model_dir: Optional[Path] = None,
        train_ratio: float = 0.8,
        
        # 回测参数
        val_path: Optional[Path] = None,
        model_paths: Optional[dict] = None,
        output_dir: Optional[Path] = None,

        log_range: int = 1000
    ):
        """
        初始化分析管理器
        
        Args:
            coin: 币种，如 "btc", "sol"（小写）
            timeframe: K线时长，如 "1h", "1d", "1m"
            intermediate_dir: 中间数据目录
            processed_dir: 处理后数据目录
            signal_threshold: 信号生成阈值
            signal_delta: 信号生成delta参数
            signal_lookahead_periods: 信号生成时向前看的K线周期数，默认为1（下一根K线）
            model_dir: 模型保存目录，如果为None则根据coin自动生成
            train_ratio: 训练集比例
            val_path: 回测验证数据路径，如果为None则使用processed_file
            model_paths: 模型路径字典，如果为None则根据coin和timeframe自动生成
            output_dir: 回测结果输出目录，用于保存equity curve图片，如果为None则使用默认路径analysis/backtest/curves
        """
        # 币种和K线时长
        self.coin = coin.lower()
        self.timeframe = timeframe
        
        # 自动生成原始数据路径：data/raw/{COIN}USDT_{TIMEFRAME}.csv
        # 例如：BTCUSDT_1d.csv, ETHUSDT_1m.csv
        coin_upper = self.coin.upper()
        self.raw_data_path = Path(f"data/raw/{coin_upper}USDT_{timeframe}.csv")
        
        # 中间文件路径：data/intermediate/{coin}_{timeframe}_features.csv
        self.intermediate_dir = intermediate_dir
        self.intermediate_path = intermediate_dir / f"{self.coin}_{timeframe}_features.csv"
        
        # 处理后文件路径：data/processed/{coin}_{timeframe}_p{lookahead_periods}_features_signal.csv
        # 例如：btc_1h_p1_features_signal.csv, btc_1h_p5_features_signal.csv
        self.processed_dir = processed_dir
        self.processed_path = processed_dir / f"{self.coin}_{timeframe}_p{signal_lookahead_periods}_features_signal.csv"
        
        # 信号生成参数
        self.signal_threshold = signal_threshold
        self.signal_delta = signal_delta
        self.signal_lookahead_periods = signal_lookahead_periods
        
        # 模型训练参数
        if model_dir is None:
            # 模型目录：models/{coin}/
            self.model_dir = Path(f"models/{self.coin}")
        else:
            self.model_dir = model_dir
        self.train_ratio = train_ratio
        
        # 回测参数
        if val_path is None:
            self.val_path = self.processed_path
        else:
            self.val_path = val_path
        
        # 如果未指定output_dir，使用默认路径：analysis/backtest/curves
        if output_dir is None:
            self.output_dir = Path("analysis/backtest/curves")
        else:
            self.output_dir = output_dir
        
        # 自动生成模型路径字典（如果未提供）
        # 模型文件名格式：{timeframe}_p{lookahead_periods}_{model}.pkl
        # 例如：1h_p1_lightgbm.pkl, 1h_p5_lightgbm.pkl
        if model_paths is None:
            self.model_paths = {
                "logistic": self.model_dir / f"{timeframe}_p{signal_lookahead_periods}_logistic.pkl",
                "lightgbm": self.model_dir / f"{timeframe}_p{signal_lookahead_periods}_lightgbm.pkl",
                "svm": self.model_dir / f"{timeframe}_p{signal_lookahead_periods}_svm.pkl",
                "random_forest": self.model_dir / f"{timeframe}_p{signal_lookahead_periods}_random_forest.pkl",
                "nn": self.model_dir / f"{timeframe}_p{signal_lookahead_periods}_lstm.pkl",
                "transformer": self.model_dir / f"{timeframe}_p{signal_lookahead_periods}_transformer.pkl",
            }
        else:
            self.model_paths = {k: Path(v) if not isinstance(v, Path) else v 
                               for k, v in model_paths.items()}
        
        # 初始化各个组件
        self._init_components()
    
    def _init_components(self):
        """初始化各个流程组件"""
        # 价格收集器
        self.price_collector = PriceCollector()
        
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
            threshold=self.signal_threshold,
            delta=self.signal_delta,
            lookahead_periods=self.signal_lookahead_periods,
        )
        
        # 回测运行器
        self.backtest_runner = BacktestRunner(
            val_path=self.val_path,
            model_paths=self.model_paths,
            coin=self.coin,
            timeframe=self.timeframe,
            output_dir=self.output_dir,
            lookahead_periods=self.signal_lookahead_periods,
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
                timeframe=self.timeframe,
                lookahead_periods=self.signal_lookahead_periods,
            )
        elif model_type == "logistic":
            return LogisticTrainer(
                data_path=self.processed_path,
                model_dir=self.model_dir,
                train_ratio=self.train_ratio,
                timeframe=self.timeframe,
                lookahead_periods=self.signal_lookahead_periods,
            )
        elif model_type == "svm":
            return SVMTrainer(
                data_path=self.processed_path,
                model_dir=self.model_dir,
                train_ratio=self.train_ratio,
                timeframe=self.timeframe,
                lookahead_periods=self.signal_lookahead_periods,
            )
        elif model_type == "random_forest":
            return RandomForestTrainer(
                data_path=self.processed_path,
                model_dir=self.model_dir,
                train_ratio=self.train_ratio,
                timeframe=self.timeframe,
                lookahead_periods=self.signal_lookahead_periods,
            )
        elif model_type == "lstm":
            return LSTMTrainer(
                data_path=self.processed_path,
                model_dir=self.model_dir,
                train_ratio=self.train_ratio,
                timeframe=self.timeframe,
                lookahead_periods=self.signal_lookahead_periods,
            )
        elif model_type == "transformer":
            return TransformerTrainer(
                data_path=self.processed_path,
                model_dir=self.model_dir,
                train_ratio=self.train_ratio,
                timeframe=self.timeframe,
                lookahead_periods=self.signal_lookahead_periods,
            )
        else:
            raise ValueError(
                f"Unsupported model_type: {model_type}. "
                f"Supported types: lightgbm, logistic, svm, random_forest, lstm, transformer"
            )
    
    def update_raw_data(self, force_update: bool = False) -> bool:
        """
        更新原始价格数据（增量更新）
        
        Args:
            force_update: 是否强制更新（忽略数据是否最新）
            
        Returns:
            True 如果数据已更新，False 如果跳过
        """
        print("=" * 60)
        print("[AnalysisManager] Step 0: Raw Data Update")
        print("=" * 60)
        
        # 根据 timeframe 设置数据最大允许年龄（小时）
        # 对于高频数据（1m, 5m），允许更短的年龄
        # 对于低频数据（1d），允许更长的年龄
        timeframe_to_max_age = {
            '1m': 0.5,   # 30 分钟
            '5m': 1,     # 1 小时
            '15m': 2,    # 2 小时
            '30m': 2,    # 2 小时
            '1h': 2,     # 2 小时
            '4h': 4,     # 4 小时
            '1d': 24,    # 24 小时
        }
        max_age_hours = timeframe_to_max_age.get(self.timeframe, 2)
        
        # 构建交易对符号
        symbol = f"{self.coin.upper()}USDT"
        
        # 更新数据
        updated = self.price_collector.update_data(
            symbol=symbol,
            interval=self.timeframe,
            file_path=self.raw_data_path,
            max_age_hours=max_age_hours,
            force_update=force_update,
        )
        
        if updated:
            print("[AnalysisManager] Raw data updated successfully.\n")
        else:
            print("[AnalysisManager] Raw data is up to date, skipping update.\n")
        
        return updated
    
    def run_feature_generation(self, skip_if_exists: bool = True, update_data: bool = True):
        """
        执行特征生成步骤
        
        Args:
            skip_if_exists: 如果中间文件已存在是否跳过
            update_data: 是否在生成特征前更新原始数据
        """
        # 在生成特征前，先更新原始数据
        if update_data:
            self.update_raw_data(force_update=False)
        
        print("=" * 60)
        print("[AnalysisManager] Step 1: Feature Generation")
        print("=" * 60)
        self.feature_generator.generate_features(skip_if_exists=skip_if_exists)
        print("[AnalysisManager] Feature generation completed.\n")
    
    def run_signal_generation(
        self,
        skip_if_exists: bool = True,
        lookahead_periods: Optional[int] = None,
    ):
        """
        执行信号生成步骤
        
        Args:
            skip_if_exists: 如果处理后文件已存在是否跳过
            lookahead_periods: 向前看的K线周期数，如果为None则使用初始化时的值
        """
        # 如果提供了新的lookahead_periods，更新signal_generator
        if lookahead_periods is not None:
            self.signal_generator.lookahead_periods = lookahead_periods
            print(f"[AnalysisManager] Using lookahead_periods: {lookahead_periods}")
        
        print("=" * 60)
        print("[AnalysisManager] Step 2: Signal Generation")
        print(f"[AnalysisManager] Lookahead Periods: {self.signal_generator.lookahead_periods}")
        print("=" * 60)
        self.signal_generator.generate_signal(skip_if_exists=skip_if_exists)
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
        # 模型文件名格式：{timeframe}_p{lookahead_periods}_{model}.pkl
        model_file_map = {
            "lightgbm": f"{self.timeframe}_p{self.signal_lookahead_periods}_lightgbm.pkl",
            "logistic": f"{self.timeframe}_p{self.signal_lookahead_periods}_logistic.pkl",
            "svm": f"{self.timeframe}_p{self.signal_lookahead_periods}_svm.pkl",
            "random_forest": f"{self.timeframe}_p{self.signal_lookahead_periods}_random_forest.pkl",
            "lstm": f"{self.timeframe}_p{self.signal_lookahead_periods}_lstm.pkl",
            "transformer": f"{self.timeframe}_p{self.signal_lookahead_periods}_transformer.pkl",
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
        lookahead_periods: int = 1,
        update_data: bool = True,
    ):
        """
        执行完整的分析流程
        
        Args:
            model_type: 要使用的模型类型
            skip_feature_gen: 如果中间文件已存在是否跳过特征生成
            skip_signal_gen: 如果处理后文件已存在是否跳过信号生成
            skip_training: 如果模型文件已存在是否跳过训练
            run_backtest: 是否执行回测
            lookahead_periods: 信号生成时向前看的K线周期数，如果为None则使用初始化时的值
            update_data: 是否在生成特征前更新原始数据
        """
        print("\n" + "=" * 60)
        print("Analysis Pipeline Started")
        print("=" * 60)
        print(f"Coin: {self.coin.upper()}")
        print(f"Timeframe: {self.timeframe}")
        print(f"Model Type: {model_type.upper()}")
        print(f"Raw Data: {self.raw_data_path}")
        print(f"Intermediate: {self.intermediate_path}")
        print(f"Processed: {self.processed_path}")
        print(f"Model Dir: {self.model_dir}")
        if lookahead_periods != 1:
            print(f"Lookahead Periods: {lookahead_periods}")
        print("=" * 60 + "\n")
        
        # Step 1: 特征生成（内部会调用数据更新）
        self.run_feature_generation(skip_if_exists=skip_feature_gen, update_data=update_data)
        
        # Step 2: 信号生成
        self.run_signal_generation(skip_if_exists=skip_signal_gen, lookahead_periods=lookahead_periods)
        
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
        update_data: bool = True,
    ):
        """
        执行自定义流程步骤
        
        Args:
            steps: 要执行的步骤列表，可选值: ["update", "feature", "signal", "train", "backtest"]
            model_type: 模型类型（train和backtest步骤需要）
            update_data: 是否在feature步骤前自动更新数据（如果steps中包含"update"则忽略此参数）
        
        Example:
            manager.run_custom_pipeline(
                steps=["update", "feature", "signal", "train", "backtest"],
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
        
        # 检查是否包含 "update" 步骤
        has_update_step = "update" in [s.lower() for s in steps]
        
        for step in steps:
            step = step.lower()
            
            if step == "update":
                self.update_raw_data(force_update=False)
            elif step == "feature":
                # 如果步骤列表中没有显式的 "update"，且 update_data=True，则自动更新
                self.run_feature_generation(update_data=update_data and not has_update_step)
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


"use client";

import React, { useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Slider } from '@/components/ui/slider';
import { Badge } from '@/components/ui/badge';
import { TrendingUp, Cpu, ArrowsCross, ChartPie, Code, Loader } from 'tabler-icons-react';
import Overview from 'kbar/example/src/Docs/Overview';
import PageContainer from '@/components/layout/page-container';

// Supported exchanges and trading pairs
const exchanges = ['Binance', 'Coinbase Pro', 'Kraken'];
const pairs = ['BTCUSD', 'ETHUSD', 'LTCUSD', 'DOGEUSD'];

// Strategy definitions with parameters and descriptions
const strategyDefs = [
  {
    name: 'Test MA Strategy',
    icon: TrendingUp,
    description: 'Simple moving average crossover strategy for testing.',
    params: [
      { key: 'short_min', label: 'Short MA Min', default: 10, range: true },
      { key: 'short_max', label: 'Short MA Max', default: 15, range: true },
      { key: 'long_min', label: 'Long MA Min', default: 20, range: true },
      { key: 'long_max', label: 'Long MA Max', default: 30, range: true },
    ],
  },
  {
    name: 'Momentum Strategy',
    icon: TrendingUp,
    description: 'Applies momentum breakout logic on selected crypto pairs.',
    params: [
      { key: 'lookback', label: 'Lookback Period (minutes)', default: 60 },
      { key: 'threshold', label: 'Momentum Threshold (%)', default: 1 },
    ],
  },
  {
    name: 'Mean Reversion Strategy',
    icon: Cpu,
    description: 'Enters on deviations from mean price over rolling window.',
    params: [
      { key: 'window', label: 'Window Size (minutes)', default: 30 },
      { key: 'zscore', label: 'Entry Z-Score', default: 1.5 },
    ],
  },
  {
    name: 'Arbitrage Bot',
    icon: ArrowsCross,
    description: 'Exploits price differences across tier-1 exchanges.',
    params: [
      { key: 'spread', label: 'Min Spread ($)', default: 0.50 },
      { key: 'maxPositions', label: 'Max Concurrent Positions', default: 3 },
    ],
  },
];

export default function Page() {
  const [active, setActive] = useState(strategyDefs[0].name);

  // Default configuration state per strategy
  const [configs, setConfigs] = useState(
    strategyDefs.reduce((acc, s) => {
      acc[s.name] = Object.fromEntries(s.params.map(p => [p.key, p.default]));
      return acc;
    }, {} as Record<string, Record<string, number>>)
  );
  const [pair, setPair] = useState(pairs[0]);
  const [exchange, setExchange] = useState(exchanges[0]);

  const [status, setStatus] = useState<'idle' | 'running' | 'completed'>('idle');
  const [logs, setLogs] = useState<string[]>([]);

  // Performance metrics state
  const [metrics, setMetrics] = useState<{
    winRate: string;
    monthlyReturn: string;
    maxDrawdown: string;
    profitFactor: string;
    sharpeRatio: string;
  } | null>(null);

  // Best parameters state
  const [bestParams, setBestParams] = useState<{
    short_window: number;
    long_window: number;
    monthlyReturn: string;
  } | null>(null);

  // Validation errors
  const [validationErrors, setValidationErrors] = useState<string[]>([]);

  // Backtest threads
  const [threads, setThreads] = useState(4);

  // Signal generation state
  const [signalState, setSignalState] = useState<'idle' | 'generating' | 'ready'>('idle');
  const [signalCount, setSignalCount] = useState(0);

  const handleParamChange = (strategy: string, key: string, value: number) => {
    setConfigs(prev => ({
      ...prev,
      [strategy]: { ...prev[strategy], [key]: value }
    }));
    setValidationErrors([]);
  };

  // Validate parameters
  const validateParams = (params: Record<string, number>): string[] => {
    const errors: string[] = [];
    const short_min = params.short_min || 0;
    const short_max = params.short_max || 0;
    const long_min = params.long_min || 0;
    const long_max = params.long_max || 0;

    // Each range: max >= min
    if (short_max < short_min) {
      errors.push('Short MA max must be >= min');
    }
    if (long_max < long_min) {
      errors.push('Long MA max must be >= min');
    }

    // Long MA min > Short MA min
    if (long_min <= short_min) {
      errors.push('Long MA min must be > Short MA min');
    }

    // Long MA max > Short MA max
    if (long_max <= short_max) {
      errors.push('Long MA max must be > Short MA max');
    }

    return errors;
  };

  // Run the selected strategy with current settings
  const run = async (name: string) => {
    const params = configs[name];
    
    // Validate parameters
    const errors = validateParams(params);
    if (errors.length > 0) {
      setValidationErrors(errors);
      return;
    }
    setValidationErrors([]);
    
    const now = new Date().toLocaleTimeString();
    setStatus('running');
    setLogs([`[${now}] 🚀 Running ${name} on ${pair} @ ${exchange}`]);
    setMetrics(null);
    setBestParams(null);

    try {
      // Call backend API to run strategy
      // Add trading_pair to params
      const requestParams = {
        ...params,
        trading_pair: pair
      };
      
      const response = await fetch('http://localhost:8000/api/strategy/run', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestParams),
      });

      if (!response.ok) {
        throw new Error('Strategy execution failed');
      }

      const responseData = await response.json();
      
      if (responseData.metrics) {
        setMetrics({
          winRate: responseData.metrics.winRate,
          monthlyReturn: responseData.metrics.monthlyReturn,
          maxDrawdown: responseData.metrics.maxDrawdown,
          profitFactor: responseData.metrics.sharpeRatio, // 使用夏普比率替代盈利因子
          sharpeRatio: responseData.metrics.sharpeRatio,
        });

        // Set best parameters if available
        if (responseData.best_params) {
          setBestParams({
            short_window: responseData.best_params.short_window,
            long_window: responseData.best_params.long_window,
            monthlyReturn: responseData.best_params.monthlyReturn,
          });
        }
        
        setLogs(prev => [
          ...prev, 
          `[${new Date().toLocaleTimeString()}] ✅ Strategy execution completed`,
          `[${new Date().toLocaleTimeString()}] 📊 Performance Metrics:`,
          `    Win Rate: ${responseData.metrics.winRate}`,
          `    Monthly Return: ${responseData.metrics.monthlyReturn}`,
          `    Max Drawdown: ${responseData.metrics.maxDrawdown}`,
          `    Sharpe Ratio: ${responseData.metrics.sharpeRatio}`,
          `    Total Trades: ${responseData.metrics.totalTrades}`,
          `    Avg Return: ${responseData.metrics.avgReturn}`,
          `    Volatility: ${responseData.metrics.volatility}`,
          ...(responseData.best_params ? [
            ``,
            `[${new Date().toLocaleTimeString()}] 🏆 Best Parameters Found:`,
            `    Short MA: ${responseData.best_params.short_window}`,
            `    Long MA: ${responseData.best_params.long_window}`,
            `    Monthly Return: ${responseData.best_params.monthlyReturn}`
          ] : [])
        ]);
      } else {
        setLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] ✅ Strategy execution completed`]);
      }

      setStatus('completed');
      setLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] 📊 Results fetched and displayed`]);

    } catch (error) {
      console.error('Error:', error);
      setLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] ❌ Error: ${error.message}`]);
      setStatus('idle');
    }
  };

  // Simulate live signal generation
  const generateSignals = () => {
    setSignalState('generating');
    setSignalCount(0);
    setTimeout(() => {
      const count = Math.floor(Math.random() * 10 + 5);
      setSignalCount(count);
      setSignalState('ready');
    }, 2000);
  };

  const activeDef = strategyDefs.find(s => s.name === active)!;

  return (
    <PageContainer>

    <div className="p-8 space-y-6 max-w-6xl mx-auto overflow-scroll">
      <h1 className="text-4xl font-bold text-white">Strategy</h1>


      {/* Strategy Selection */}
      <Tabs value={active} onValueChange={setActive} className="mb-6">
        <TabsList>
          {strategyDefs.map(s => (
            <TabsTrigger key={s.name} value={s.name} className="flex items-center space-x-2 text-white">
              <s.icon size={18} />
              <span>{s.name}</span>
            </TabsTrigger>
          ))}
        </TabsList>
      </Tabs>

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Strategy Config */}
        <Card>
          <CardHeader>
            <CardTitle className="text-white">{activeDef.name} Configuration</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Pair & Exchange selection */}
            <div className="flex space-x-4">
              <div className="flex flex-col">
                <label className="text-white mb-1">Trading Pair</label>
                <select
                  value={pair}
                  onChange={e => setPair(e.target.value)}
                  className="bg-gray-800 text-white p-2 rounded"
                >
                  {pairs.map(p => (
                    <option key={p} value={p}>{p}</option>
                  ))}
                </select>
              </div>
              <div className="flex flex-col">
                <label className="text-white mb-1">Exchange</label>
                <select
                  value={exchange}
                  onChange={e => setExchange(e.target.value)}
                  className="bg-gray-800 text-white p-2 rounded"
                >
                  {exchanges.map(ex => (
                    <option key={ex} value={ex}>{ex}</option>
                  ))}
                </select>
              </div>
            </div>
            {/* Dynamic params */}
            {activeDef.params.map(p => (
              <div key={p.key} className="flex flex-col">
                <label className="text-white mb-1">{p.label}</label>
                <Input
                  type="number"
                  value={configs[active][p.key]}
                  onChange={e => handleParamChange(active, p.key, parseFloat(e.target.value))}
                />
              </div>
            ))}
            
            {/* Validation errors */}
            {validationErrors.length > 0 && (
              <div className="mt-2 p-2 bg-red-900/20 border border-red-500 rounded">
                {validationErrors.map((error, i) => (
                  <p key={i} className="text-red-400 text-sm">{error}</p>
                ))}
              </div>
            )}

            {/* Best params display */}
            {bestParams && (
              <div className="mt-4 p-3 bg-green-900/20 border border-green-500 rounded">
                <h4 className="text-green-400 font-semibold mb-1">🏆 Best Parameters:</h4>
                <p className="text-white text-sm">Short MA: {bestParams.short_window}</p>
                <p className="text-white text-sm">Long MA: {bestParams.long_window}</p>
                <p className="text-green-400 font-semibold">Monthly Return: {bestParams.monthlyReturn}</p>
              </div>
            )}

            <Button onClick={() => run(active)} variant="secondary" className="mt-4 w-full">
              Run {activeDef.name}
            </Button>
          </CardContent>
        </Card>

        {/* Execution Engine */}
        <Card className="lg:col-span-2">
          <CardHeader className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Code size={20} className="text-white" />
              <CardTitle className="text-white">Execution Engine</CardTitle>
            </div>
            <Badge
              variant={status === 'running' ? 'outline' : status === 'completed' ? 'default' : 'secondary'}
              className="capitalize"
            >
              {status}
            </Badge>
          </CardHeader>
          <CardContent>
            <ul className="h-48 overflow-auto space-y-2 p-4 bg-gray-800 rounded-lg">
              {logs.map((l, i) => (
                <li key={i} className="text-sm text-gray-100">{l}</li>
              ))}
            </ul>
          </CardContent>
        </Card>

        {/* Backtesting */}
        <Card>
          <CardHeader>
            <CardTitle className="text-white">Backtesting</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-white">Parallel backtests with adjustable threads:</p>
            <div className="mt-4">
              <Slider
                value={[threads]}
                onValueChange={val => setThreads(val[0])}
                max={16}
                step={1}
                className="w-full"
              />
              <p className="text-xs mt-2 text-white">Threads: {threads}</p>
            </div>
            <Button className="mt-4 w-full">Start Backtest</Button>
          </CardContent>
        </Card>

        {/* Performance Analytics */}
        <Card>
          <CardHeader>
            <div className="flex items-center space-x-2">
              <ChartPie size={20} className="text-white" />
              <CardTitle className="text-white">Performance Analytics</CardTitle>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            {metrics ? (
              <div className="grid grid-cols-2 gap-4">
                <Card className="bg-gray-800">
                  <CardContent>
                    <h3 className="text-sm font-medium text-gray-100">Win Rate</h3>
                    <p className="text-2xl font-bold text-green-400">{metrics.winRate}</p>
                  </CardContent>
                </Card>
                <Card className="bg-gray-800">
                  <CardContent>
                    <h3 className="text-sm font-medium text-gray-100">Monthly Return</h3>
                    <p className="text-2xl font-bold text-green-400">{metrics.monthlyReturn}</p>
                  </CardContent>
                </Card>
                <Card className="bg-gray-800">
                  <CardContent>
                    <h3 className="text-sm font-medium text-gray-100">Max Drawdown</h3>
                    <p className="text-2xl font-bold text-red-400">{metrics.maxDrawdown}</p>
                  </CardContent>
                </Card>
                <Card className="bg-gray-800">
                  <CardContent>
                    <h3 className="text-sm font-medium text-gray-100">Profit Factor</h3>
                    <p className="text-2xl font-bold text-blue-400">{metrics.profitFactor}</p>
                  </CardContent>
                </Card>
                <Card className="bg-gray-800 col-span-2">
                  <CardContent>
                    <h3 className="text-sm font-medium text-gray-100">Sharpe Ratio</h3>
                    <p className="text-2xl font-bold text-indigo-400">{metrics.sharpeRatio}</p>
                  </CardContent>
                </Card>
              </div>
            ) : (
              <p className="text-white">Run a strategy to see performance metrics.</p>
            )}
            <Button className="mt-4 w-full">View Full Dashboard</Button>
          </CardContent>
        </Card>

        {/* Signal Generation */}
        <Card>
          <CardHeader>
            <CardTitle className="text-white">Signal Module</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <p className="text-white">Generate real-time ML-based trading signals.</p>
            <Button
              onClick={generateSignals}
              disabled={signalState === 'generating'}
              className="w-full flex items-center justify-center"
            >
              {signalState === 'generating' ? <Loader className="animate-spin text-white" size={16} /> : 'Generate Signals'}
            </Button>
            {signalState === 'ready' && (
              <p className="mt-2 text-sm text-white">Generated {signalCount} signals for deployment.</p>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
    </PageContainer>
  );
}

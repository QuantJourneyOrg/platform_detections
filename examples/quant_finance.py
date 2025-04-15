"""
Platform Detection Framework - Quantitative Finance Example

This example demonstrates how to use the platform detection framework
to optimize quantitative finance calculations.
"""

import numpy as np
import pandas as pd
import time
from . import optimize, get_detector
from platform_detection.backends import use_backend
from platform_detection.orchestrator import ComputeBackend


def generate_random_returns(n_assets=100, n_days=1000, seed=42):
    """
    Generate random daily returns for a portfolio of assets
    
    Args:
        n_assets: Number of assets in the portfolio
        n_days: Number of trading days
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with asset returns
    """
    np.random.seed(seed)
    
    # Generate random returns with some correlation structure
    cov_matrix = np.random.randn(n_assets, n_assets)
    cov_matrix = cov_matrix.T @ cov_matrix
    cov_matrix = cov_matrix / np.max(cov_matrix) * 0.01  # Scale to reasonable values
    
    # Generate correlated returns
    returns = np.random.multivariate_normal(
        mean=np.random.rand(n_assets) * 0.001,  # Random expected returns
        cov=cov_matrix,
        size=n_days
    )
    
    # Convert to DataFrame
    asset_names = [f"Asset_{i}" for i in range(n_assets)]
    dates = pd.date_range(end=pd.Timestamp.today(), periods=n_days, freq='B')
    return pd.DataFrame(returns, index=dates, columns=asset_names)


# Non-optimized functions
def calculate_correlation_matrix_standard(returns):
    """Calculate correlation matrix without optimization"""
    return np.corrcoef(returns.values.T)


def calculate_portfolio_risk_standard(returns, weights):
    """Calculate portfolio risk without optimization"""
    cov_matrix = returns.cov()
    return np.sqrt(weights.T @ cov_matrix @ weights)


def calculate_moving_averages_standard(prices, windows=[20, 50, 200]):
    """Calculate moving averages without optimization"""
    result = {}
    for window in windows:
        result[f'MA{window}'] = prices.rolling(window=window).mean()
    return pd.concat(result.values(), axis=1, keys=result.keys())


# Optimized functions using decorators
@optimize(operation_type="matrix")
def calculate_correlation_matrix_optimized(returns):
    """Calculate correlation matrix with automatic optimization"""
    return np.corrcoef(returns.values.T)


@optimize(operation_type="stat", data_size_estimator=lambda returns, weights: returns.size)
def calculate_portfolio_risk_optimized(returns, weights):
    """Calculate portfolio risk with automatic optimization"""
    cov_matrix = returns.cov()
    return np.sqrt(weights.T @ cov_matrix @ weights)


@optimize(operation_type="finance")
def calculate_moving_averages_optimized(prices, windows=[20, 50, 200]):
    """Calculate moving averages with automatic optimization"""
    result = {}
    for window in windows:
        result[f'MA{window}'] = prices.rolling(window=window).mean()
    return pd.concat(result.values(), axis=1, keys=result.keys())


def simulate_portfolio_optimization(returns, n_portfolios=1000):
    """
    Simulate a portfolio optimization by generating random weights
    
    Args:
        returns: DataFrame with asset returns
        n_portfolios: Number of random portfolios to generate
        
    Returns:
        DataFrame with portfolio weights, returns, and volatility
    """
    n_assets = returns.shape[1]
    results = []
    
    # Mean returns and covariance matrix
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    
    for i in range(n_portfolios):
        # Generate random weights
        weights = np.random.random(n_assets)
        weights = weights / np.sum(weights)
        
        # Calculate portfolio return and volatility
        portfolio_return = np.sum(mean_returns * weights) * 252  # Annualized
        portfolio_volatility = np.sqrt(weights.T @ cov_matrix @ weights) * np.sqrt(252)  # Annualized
        
        # Calculate Sharpe ratio (assuming risk-free rate of 0.01)
        sharpe_ratio = (portfolio_return - 0.01) / portfolio_volatility
        
        results.append({
            'return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio,
            'weights': weights
        })
    
    return pd.DataFrame(results)


def benchmark_optimizations():
    """Benchmark standard vs. optimized implementations"""
    print("Generating random returns data...")
    returns = generate_random_returns(n_assets=100, n_days=1000)
    
    # Create random weights
    weights = np.random.random(returns.shape[1])
    weights = weights / np.sum(weights)
    
    print("\nRunning benchmarks...\n")
    
    # Benchmark correlation matrix
    print("Correlation Matrix Calculation:")
    start = time.time()
    corr_std = calculate_correlation_matrix_standard(returns)
    std_time = time.time() - start
    print(f"  Standard:  {std_time:.4f}s")
    
    start = time.time()
    corr_opt = calculate_correlation_matrix_optimized(returns)
    opt_time = time.time() - start
    print(f"  Optimized: {opt_time:.4f}s")
    print(f"  Speedup:   {std_time/opt_time:.2f}x")
    
    # Benchmark portfolio risk
    print("\nPortfolio Risk Calculation:")
    start = time.time()
    risk_std = calculate_portfolio_risk_standard(returns, weights)
    std_time = time.time() - start
    print(f"  Standard:  {std_time:.4f}s")
    
    start = time.time()
    risk_opt = calculate_portfolio_risk_optimized(returns, weights)
    opt_time = time.time() - start
    print(f"  Optimized: {opt_time:.4f}s")
    print(f"  Speedup:   {std_time/opt_time:.2f}x")
    
    # Benchmark moving averages
    print("\nMoving Averages Calculation:")
    prices = (1 + returns).cumprod()  # Convert returns to prices
    
    start = time.time()
    ma_std = calculate_moving_averages_standard(prices.iloc[:, 0])  # Use first asset
    std_time = time.time() - start
    print(f"  Standard:  {std_time:.4f}s")
    
    start = time.time()
    ma_opt = calculate_moving_averages_optimized(prices.iloc[:, 0])  # Use first asset
    opt_time = time.time() - start
    print(f"  Optimized: {opt_time:.4f}s")
    print(f"  Speedup:   {std_time/opt_time:.2f}x")
    
    # Benchmark portfolio optimization with manual backend selection
    print("\nPortfolio Optimization:")
    
    start = time.time()
    portfolios_std = simulate_portfolio_optimization(returns, n_portfolios=1000)
    std_time = time.time() - start
    print(f"  Standard:  {std_time:.4f}s")
    
    # Get detector instance
    detector = get_detector()
    
    # Use optimal backend for matrix operations
    optimal_backend = detector.get_backend_for_operation("matrix", returns.size)
    
    start = time.time()
    with use_backend(optimal_backend):
        portfolios_opt = simulate_portfolio_optimization(returns, n_portfolios=1000)
    opt_time = time.time() - start
    print(f"  Optimized ({optimal_backend.value}): {opt_time:.4f}s")
    print(f"  Speedup:   {std_time/opt_time:.2f}x")


def demonstrate_portfolio_allocation():
    """Demonstrate a practical portfolio allocation use case"""
    # Generate returns for major asset classes
    asset_classes = ["US_Equity", "Intl_Equity", "US_Bonds", "Intl_Bonds", 
                    "Real_Estate", "Commodities", "Cash"]
    
    # Create more realistic returns with different characteristics
    np.random.seed(42)
    n_days = 1000
    
    # Custom means and standard deviations for each asset class
    means = np.array([0.00035, 0.0003, 0.0001, 0.00008, 0.00025, 0.0002, 0.00005])
    stds = np.array([0.01, 0.012, 0.003, 0.004, 0.015, 0.02, 0.001])
    
    # Correlation matrix (simplified)
    correlations = np.array([
        [1.0, 0.7, -0.1, -0.05, 0.6, 0.2, 0.0],   # US_Equity
        [0.7, 1.0, -0.05, 0.0, 0.5, 0.3, 0.0],    # Intl_Equity
        [-0.1, -0.05, 1.0, 0.8, 0.1, -0.1, 0.2],  # US_Bonds
        [-0.05, 0.0, 0.8, 1.0, 0.1, -0.05, 0.2],  # Intl_Bonds
        [0.6, 0.5, 0.1, 0.1, 1.0, 0.3, 0.0],      # Real_Estate
        [0.2, 0.3, -0.1, -0.05, 0.3, 1.0, 0.0],   # Commodities
        [0.0, 0.0, 0.2, 0.2, 0.0, 0.0, 1.0]       # Cash
    ])
    
    # Convert correlation matrix to covariance matrix
    cov_matrix = np.outer(stds, stds) * correlations
    
    # Generate correlated returns
    returns = np.random.multivariate_normal(means, cov_matrix, n_days)
    returns_df = pd.DataFrame(returns, columns=asset_classes)
    returns_df.index = pd.date_range(end=pd.Timestamp.today(), periods=n_days, freq='B')
    
    print("Asset Class Analysis")
    print("===================")
    
    # Get detector and optimal backend
    detector = get_detector()
    optimal_backend = detector.get_optimal_backend()
    print(f"Using {optimal_backend.value} backend for calculations\n")
    
    # Calculate portfolio characteristics
    with use_backend(optimal_backend):
        # Calculate annualized returns and volatility
        annual_returns = returns_df.mean() * 252
        annual_volatility = returns_df.std() * np.sqrt(252)
        
        # Calculate correlation matrix
        correlation_matrix = np.corrcoef(returns_df.T)
        
        # Generate efficient frontier
        print("Generating efficient frontier...")
        portfolios = simulate_portfolio_optimization(returns_df, n_portfolios=5000)
        
        # Find optimal portfolios
        min_vol_idx = portfolios['volatility'].idxmin()
        min_vol_portfolio = portfolios.loc[min_vol_idx]
        
        max_sharpe_idx = portfolios['sharpe_ratio'].idxmax()
        max_sharpe_portfolio = portfolios.loc[max_sharpe_idx]
    
    # Print results
    print("\nAnnualized Returns and Volatility:")
    for i, asset in enumerate(asset_classes):
        print(f"  {asset.ljust(12)}: Return {annual_returns[i]*100:.2f}%, Volatility {annual_volatility[i]*100:.2f}%")
    
    print("\nOptimal Portfolios:")
    print("  Minimum Volatility Portfolio:")
    print(f"    Return: {min_vol_portfolio['return']*100:.2f}%")
    print(f"    Volatility: {min_vol_portfolio['volatility']*100:.2f}%")
    print(f"    Sharpe Ratio: {min_vol_portfolio['sharpe_ratio']:.3f}")
    print("    Weights:")
    for i, asset in enumerate(asset_classes):
        print(f"      {asset.ljust(12)}: {min_vol_portfolio['weights'][i]*100:.2f}%")
    
    print("\n  Maximum Sharpe Ratio Portfolio:")
    print(f"    Return: {max_sharpe_portfolio['return']*100:.2f}%")
    print(f"    Volatility: {max_sharpe_portfolio['volatility']*100:.2f}%")
    print(f"    Sharpe Ratio: {max_sharpe_portfolio['sharpe_ratio']:.3f}")
    print("    Weights:")
    for i, asset in enumerate(asset_classes):
        print(f"      {asset.ljust(12)}: {max_sharpe_portfolio['weights'][i]*100:.2f}%")


if __name__ == "__main__":
    # Print available backends
    detector = get_detector()
    print(f"Optimal backend: {detector.get_optimal_backend().value}")
    print(f"Matrix operations backend: {detector.get_backend_for_operation('matrix').value}")
    print(f"Statistical operations backend: {detector.get_backend_for_operation('stat').value}")
    print(f"Finance operations backend: {detector.get_backend_for_operation('finance').value}")
    print()
    
    # Run benchmarks
    benchmark_optimizations()
    
    print("\n" + "="*60 + "\n")
    
    # Run portfolio demonstration
    demonstrate_portfolio_allocation()
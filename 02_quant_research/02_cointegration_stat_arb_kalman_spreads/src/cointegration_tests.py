"""
Cointegration testing module.

Implements Engle-Granger and Johansen cointegration tests for identifying
statistically significant long-run relationships between non-stationary series.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LinearRegression
import itertools
import logging

from utils import setup_logging
from stationarity_tests import StationarityTester


logger = setup_logging()


class CointegrationAnalyzer:
    """
    Performs cointegration analysis using Engle-Granger and Johansen methods.
    """
    
    def __init__(self, significance_level: float = 0.05):
        """
        Initialize cointegration analyzer.
        
        Args:
            significance_level: Significance level for hypothesis tests
        """
        self.significance_level = significance_level
        self.stationarity_tester = StationarityTester(significance_level)
    
    def engle_granger_test(self,
                          y: pd.Series,
                          x: pd.Series,
                          maxlag: Optional[int] = None) -> Dict:
        """
        Perform Engle-Granger two-step cointegration test.
        
        Step 1: Estimate cointegrating regression Y = α + β*X + ε
        Step 2: Test residuals for stationarity using ADF
        
        Args:
            y: Dependent variable (price series)
            x: Independent variable (price series)
            maxlag: Maximum lag for ADF test
            
        Returns:
            Dictionary with test results including hedge ratio and residuals
        """
        logger.info("Performing Engle-Granger cointegration test")
        
        # Align series
        df = pd.DataFrame({'y': y, 'x': x}).dropna()
        
        if len(df) < 30:
            raise ValueError("Insufficient data for cointegration test")
        
        # Step 1: Estimate cointegrating regression
        model = LinearRegression()
        X = df['x'].values.reshape(-1, 1)
        Y = df['y'].values
        
        model.fit(X, Y)
        
        alpha = model.intercept_
        beta = model.coef_[0]
        
        # Compute residuals (spread)
        residuals = Y - (alpha + beta * X.flatten())
        residuals_series = pd.Series(residuals, index=df.index)
        
        # Step 2: Test residuals for stationarity
        adf_result = adfuller(residuals, maxlag=maxlag, regression='c')
        
        adf_statistic = adf_result[0]
        p_value = adf_result[1]
        
        # Engle-Granger critical values are more stringent than standard ADF
        # Using MacKinnon (2010) critical values for cointegration
        eg_critical_values = {
            '1%': -3.90,
            '5%': -3.34,
            '10%': -3.04
        }
        
        # Determine cointegration
        is_cointegrated = adf_statistic < eg_critical_values['5%']
        
        # Compute R-squared
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((Y - np.mean(Y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        results = {
            'method': 'Engle-Granger',
            'alpha': alpha,
            'beta': beta,
            'hedge_ratio': beta,
            'r_squared': r_squared,
            'adf_statistic': adf_statistic,
            'p_value': p_value,
            'critical_values': eg_critical_values,
            'is_cointegrated': is_cointegrated,
            'residuals': residuals_series,
            'n_obs': len(df)
        }
        
        logger.info(
            f"Engle-Granger: β={beta:.4f}, ADF={adf_statistic:.4f}, "
            f"cointegrated={is_cointegrated}"
        )
        
        return results
    
    def johansen_test(self,
                     data: pd.DataFrame,
                     det_order: int = 0,
                     k_ar_diff: int = 1) -> Dict:
        """
        Perform Johansen cointegration test.
        
        Tests for cointegration in a multivariate system using maximum likelihood.
        Provides both trace and maximum eigenvalue statistics.
        
        Args:
            data: DataFrame with price series (2+ columns)
            det_order: Deterministic trend order
                      -1: no deterministic terms
                       0: constant term
                       1: constant + linear trend
            k_ar_diff: Number of lagged differences in VAR
            
        Returns:
            Dictionary with test results including cointegration rank and vectors
        """
        logger.info(f"Performing Johansen cointegration test on {data.columns.tolist()}")
        
        # Remove NaN values
        data_clean = data.dropna()
        
        if len(data_clean) < 30:
            raise ValueError("Insufficient data for Johansen test")
        
        # Perform Johansen test
        result = coint_johansen(data_clean, det_order=det_order, k_ar_diff=k_ar_diff)
        
        n_vars = data_clean.shape[1]
        
        # Extract test statistics
        trace_stats = result.lr1  # Trace statistics
        max_eigen_stats = result.lr2  # Maximum eigenvalue statistics
        
        # Critical values (90%, 95%, 99%)
        trace_crit = result.cvt
        max_eigen_crit = result.cvm
        
        # Determine cointegration rank using trace statistic at 5% level
        coint_rank = 0
        for i in range(n_vars):
            if trace_stats[i] > trace_crit[i, 1]:  # 95% critical value
                coint_rank = i + 1
        
        # Extract eigenvectors (cointegrating vectors)
        eigenvectors = result.evec
        
        # Normalize eigenvectors (first element = 1)
        normalized_eigenvectors = eigenvectors / eigenvectors[0, :]
        
        # Extract eigenvalues
        eigenvalues = result.eig
        
        results = {
            'method': 'Johansen',
            'n_vars': n_vars,
            'n_obs': len(data_clean),
            'det_order': det_order,
            'k_ar_diff': k_ar_diff,
            'trace_stats': trace_stats,
            'trace_crit_95': trace_crit[:, 1],
            'max_eigen_stats': max_eigen_stats,
            'max_eigen_crit_95': max_eigen_crit[:, 1],
            'coint_rank': coint_rank,
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors,
            'normalized_eigenvectors': normalized_eigenvectors,
            'is_cointegrated': coint_rank > 0,
            'columns': data_clean.columns.tolist()
        }
        
        logger.info(
            f"Johansen: rank={coint_rank}, trace_stat={trace_stats[0]:.2f}, "
            f"cointegrated={coint_rank > 0}"
        )
        
        return results
    
    def extract_hedge_ratios(self, johansen_result: Dict) -> pd.DataFrame:
        """
        Extract hedge ratios from Johansen test results.
        
        Args:
            johansen_result: Results from johansen_test()
            
        Returns:
            DataFrame with hedge ratios for each cointegrating relationship
        """
        eigenvectors = johansen_result['normalized_eigenvectors']
        columns = johansen_result['columns']
        coint_rank = johansen_result['coint_rank']
        
        # Create DataFrame with hedge ratios
        hedge_ratios = pd.DataFrame(
            eigenvectors[:, :coint_rank],
            index=columns,
            columns=[f'Relationship_{i+1}' for i in range(coint_rank)]
        )
        
        return hedge_ratios
    
    def compute_spread(self,
                      data: pd.DataFrame,
                      hedge_ratios: pd.Series) -> pd.Series:
        """
        Compute spread using hedge ratios.
        
        For a cointegrating relationship: S = Σ(β_i * P_i)
        
        Args:
            data: DataFrame with price series
            hedge_ratios: Series with hedge ratios (indexed by ticker)
            
        Returns:
            Spread series
        """
        # Align data and hedge ratios
        common_cols = data.columns.intersection(hedge_ratios.index)
        
        if len(common_cols) == 0:
            raise ValueError("No common columns between data and hedge ratios")
        
        # Compute spread
        spread = (data[common_cols] * hedge_ratios[common_cols]).sum(axis=1)
        
        return spread
    
    def scan_pairs(self,
                  prices: pd.DataFrame,
                  method: str = 'johansen',
                  min_correlation: float = 0.7) -> List[Dict]:
        """
        Scan all possible pairs for cointegration.
        
        Args:
            prices: DataFrame with price series
            method: 'engle_granger' or 'johansen'
            min_correlation: Minimum correlation threshold for pre-filtering
            
        Returns:
            List of dictionaries with test results for each pair
        """
        logger.info(f"Scanning {len(prices.columns)} tickers for cointegrated pairs")
        
        tickers = prices.columns.tolist()
        results = []
        
        # Pre-filter by correlation
        corr_matrix = prices.corr()
        
        # Generate all pairs
        pairs = list(itertools.combinations(tickers, 2))
        logger.info(f"Testing {len(pairs)} pairs")
        
        for ticker1, ticker2 in pairs:
            # Check correlation threshold
            correlation = corr_matrix.loc[ticker1, ticker2]
            
            if abs(correlation) < min_correlation:
                continue
            
            try:
                if method == 'engle_granger':
                    # Test both directions
                    result1 = self.engle_granger_test(
                        prices[ticker1],
                        prices[ticker2]
                    )
                    result1['pair'] = (ticker1, ticker2)
                    result1['direction'] = f"{ticker1} ~ {ticker2}"
                    result1['correlation'] = correlation
                    
                    result2 = self.engle_granger_test(
                        prices[ticker2],
                        prices[ticker1]
                    )
                    result2['pair'] = (ticker2, ticker1)
                    result2['direction'] = f"{ticker2} ~ {ticker1}"
                    result2['correlation'] = correlation
                    
                    results.extend([result1, result2])
                
                elif method == 'johansen':
                    pair_data = prices[[ticker1, ticker2]]
                    result = self.johansen_test(pair_data)
                    result['pair'] = (ticker1, ticker2)
                    result['correlation'] = correlation
                    
                    results.append(result)
                
                else:
                    raise ValueError(f"Unknown method: {method}")
            
            except Exception as e:
                logger.warning(f"Error testing pair ({ticker1}, {ticker2}): {e}")
                continue
        
        logger.info(f"Found {len(results)} test results")
        
        return results
    
    def rank_candidates(self,
                       results: List[Dict],
                       method: str = 'johansen',
                       top_n: Optional[int] = None) -> pd.DataFrame:
        """
        Rank cointegration candidates by statistical strength.
        
        Args:
            results: List of test results from scan_pairs()
            method: Test method used
            top_n: Return only top N candidates
            
        Returns:
            DataFrame with ranked candidates
        """
        if method == 'engle_granger':
            candidates = []
            for r in results:
                if r['is_cointegrated']:
                    candidates.append({
                        'pair': r['pair'],
                        'direction': r['direction'],
                        'hedge_ratio': r['hedge_ratio'],
                        'adf_statistic': r['adf_statistic'],
                        'r_squared': r['r_squared'],
                        'correlation': r['correlation'],
                        'n_obs': r['n_obs']
                    })
            
            df = pd.DataFrame(candidates)
            if len(df) > 0:
                # Rank by ADF statistic (more negative = stronger cointegration)
                df = df.sort_values('adf_statistic')
        
        elif method == 'johansen':
            candidates = []
            for r in results:
                if r['is_cointegrated']:
                    candidates.append({
                        'pair': r['pair'],
                        'coint_rank': r['coint_rank'],
                        'trace_stat': r['trace_stats'][0],
                        'trace_crit_95': r['trace_crit_95'][0],
                        'max_eigen_stat': r['max_eigen_stats'][0],
                        'correlation': r['correlation'],
                        'n_obs': r['n_obs']
                    })
            
            df = pd.DataFrame(candidates)
            if len(df) > 0:
                # Rank by trace statistic (higher = stronger cointegration)
                df = df.sort_values('trace_stat', ascending=False)
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        if top_n is not None and len(df) > 0:
            df = df.head(top_n)
        
        logger.info(f"Ranked {len(df)} cointegrated pairs")
        
        return df

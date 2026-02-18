"""
Residual diagnostics for Hawkes processes using time-rescaling theorem.
"""
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


class HawkesDiagnostics:
    """Diagnostic tools for validating Hawkes process fit."""
    
    def __init__(self, baseline, kernel):
        """
        Parameters
        ----------
        baseline : ndarray
            Baseline intensities
        kernel : MultiDimensionalKernel
            Fitted kernel
        """
        self.baseline = baseline
        self.kernel = kernel
        self.num_types = len(baseline)
    
    def compute_compensators(self, events):
        """
        Compute compensator (integrated intensity) for each event.
        
        By time-rescaling theorem, if model is correct, the transformed times
        should follow a unit-rate Poisson process (exponential(1) inter-arrival times).
        
        Parameters
        ----------
        events : list of tuples
            List of (time, event_type)
        
        Returns
        -------
        compensators : dict
            Compensators for each event type
        """
        n = self.num_types
        
        # Separate events by type
        event_times = [[] for _ in range(n)]
        for t, event_type in events:
            event_times[event_type].append(t)
        
        compensators = {}
        
        for i in range(n):
            if len(event_times[i]) == 0:
                compensators[i] = np.array([])
                continue
            
            times_i = np.array(event_times[i])
            comp_i = np.zeros(len(times_i))
            
            for idx, t in enumerate(times_i):
                # Compute Λᵢ(t) = ∫₀ᵗ λᵢ(s) ds
                comp = self._compute_compensator_at_time(t, event_times, i)
                comp_i[idx] = comp
            
            compensators[i] = comp_i
        
        return compensators
    
    def _compute_compensator_at_time(self, t, event_times, target_type):
        """Compute compensator Λᵢ(t) = ∫₀ᵗ λᵢ(s) ds."""
        # Baseline contribution
        comp = self.baseline[target_type] * t
        
        # Excitation contributions
        for j in range(self.num_types):
            if len(event_times[j]) > 0:
                past_times = np.array(event_times[j])
                past_times = past_times[past_times < t]
                
                if len(past_times) > 0:
                    # ∫₀ᵗ Σₖ φ(s - tₖ) ds = Σₖ ∫ₜₖᵗ φ(s - tₖ) ds
                    lags = t - past_times
                    integrals = self.kernel.integral(target_type, j, lags)
                    comp += np.sum(integrals)
        
        return comp
    
    def compute_residuals(self, events):
        """
        Compute inter-arrival times in compensator time (residuals).
        
        If model is correct, these should be exponential(1) distributed.
        
        Parameters
        ----------
        events : list of tuples
            List of (time, event_type)
        
        Returns
        -------
        residuals : dict
            Residuals for each event type
        """
        compensators = self.compute_compensators(events)
        residuals = {}
        
        for i in range(self.num_types):
            if len(compensators[i]) > 1:
                # Inter-arrival times in compensator time
                tau = np.diff(compensators[i])
                residuals[i] = tau
            else:
                residuals[i] = np.array([])
        
        return residuals
    
    def ks_test(self, events):
        """
        Kolmogorov-Smirnov test for exponential(1) distribution.
        
        Parameters
        ----------
        events : list of tuples
            List of (time, event_type)
        
        Returns
        -------
        results : dict
            KS test results for each event type
        """
        residuals = self.compute_residuals(events)
        results = {}
        
        for i in range(self.num_types):
            if len(residuals[i]) > 0:
                ks_stat, p_value = stats.kstest(residuals[i], 'expon')
                results[i] = {
                    'statistic': ks_stat,
                    'p_value': p_value,
                    'n_samples': len(residuals[i])
                }
            else:
                results[i] = None
        
        return results
    
    def qq_plot(self, events, event_type, ax=None, show=True):
        """
        Q-Q plot comparing residuals to exponential(1) distribution.
        
        Parameters
        ----------
        events : list of tuples
            List of (time, event_type)
        event_type : int
            Event type to plot
        ax : matplotlib axis, optional
            Axis to plot on
        show : bool
            Whether to display plot
        """
        residuals = self.compute_residuals(events)
        
        if len(residuals[event_type]) == 0:
            print(f"No residuals for event type {event_type}")
            return
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        
        # Theoretical quantiles (exponential(1))
        n = len(residuals[event_type])
        theoretical = stats.expon.ppf(np.linspace(0.01, 0.99, n))
        
        # Sample quantiles
        sample = np.sort(residuals[event_type])
        
        ax.scatter(theoretical, sample, alpha=0.6, s=20)
        ax.plot([0, theoretical.max()], [0, theoretical.max()], 'r--', lw=2, label='y=x')
        ax.set_xlabel('Theoretical Quantiles (Exp(1))')
        ax.set_ylabel('Sample Quantiles')
        ax.set_title(f'Q-Q Plot: Event Type {event_type}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if show:
            plt.tight_layout()
            plt.show()
    
    def plot_all_diagnostics(self, events, save_path=None):
        """
        Generate comprehensive diagnostic plots for all event types.
        
        Parameters
        ----------
        events : list of tuples
            List of (time, event_type)
        save_path : str, optional
            Path to save figure
        """
        residuals = self.compute_residuals(events)
        ks_results = self.ks_test(events)
        
        n_types = self.num_types
        fig, axes = plt.subplots(n_types, 2, figsize=(12, 3 * n_types))
        
        if n_types == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_types):
            if len(residuals[i]) == 0:
                axes[i, 0].text(0.5, 0.5, 'No data', ha='center', va='center')
                axes[i, 1].text(0.5, 0.5, 'No data', ha='center', va='center')
                continue
            
            # Histogram
            axes[i, 0].hist(residuals[i], bins=30, density=True, alpha=0.7, edgecolor='black')
            x = np.linspace(0, residuals[i].max(), 100)
            axes[i, 0].plot(x, stats.expon.pdf(x), 'r-', lw=2, label='Exp(1)')
            axes[i, 0].set_xlabel('Residual')
            axes[i, 0].set_ylabel('Density')
            axes[i, 0].set_title(f'Type {i}: Histogram (KS p={ks_results[i]["p_value"]:.3f})')
            axes[i, 0].legend()
            axes[i, 0].grid(True, alpha=0.3)
            
            # Q-Q plot
            self.qq_plot(events, i, ax=axes[i, 1], show=False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()

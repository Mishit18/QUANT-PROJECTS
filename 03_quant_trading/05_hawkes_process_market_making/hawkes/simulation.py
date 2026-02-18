"""
Hawkes process simulation using Ogata's thinning algorithm.
"""
import numpy as np
from hawkes.kernels import MultiDimensionalKernel


class HawkesSimulator:
    """Simulate multidimensional Hawkes process."""
    
    def __init__(self, baseline, kernel, random_state=None):
        """
        Parameters
        ----------
        baseline : ndarray, shape (n,)
            Baseline intensities μᵢ
        kernel : MultiDimensionalKernel
            Excitation kernel matrix
        random_state : int or RandomState, optional
            Random seed or state
        """
        self.baseline = np.asarray(baseline)
        self.kernel = kernel
        self.num_types = len(baseline)
        
        if self.num_types != kernel.num_types:
            raise ValueError("Baseline dimension must match kernel dimension")
        
        self.rng = np.random.RandomState(random_state)
        
        # Check stationarity
        is_stationary, spec_rad = kernel.check_stationarity()
        if not is_stationary:
            raise ValueError(f"Process is not stationary (spectral radius = {spec_rad:.3f} >= 1)")
    
    def simulate(self, T, max_events=None):
        """
        Simulate Hawkes process up to time T using Ogata's thinning.
        
        Parameters
        ----------
        T : float
            Time horizon
        max_events : int, optional
            Maximum number of events (safety limit)
        
        Returns
        -------
        events : list of tuples
            List of (time, event_type) tuples
        """
        events = []
        t = 0.0
        
        # Track intensity contributions for efficiency
        intensity_memory = [[] for _ in range(self.num_types)]
        
        event_count = 0
        max_events = max_events or int(1e7)
        
        while t < T and event_count < max_events:
            # Compute current intensities
            intensities = self._compute_intensities(t, intensity_memory)
            lambda_total = np.sum(intensities)
            
            if lambda_total <= 0:
                break
            
            # Sample next event time
            dt = self.rng.exponential(1.0 / lambda_total)
            t_proposed = t + dt
            
            if t_proposed > T:
                break
            
            # Thinning: accept or reject
            intensities_proposed = self._compute_intensities(t_proposed, intensity_memory)
            lambda_proposed = np.sum(intensities_proposed)
            
            u = self.rng.uniform()
            if u <= lambda_proposed / lambda_total:
                # Accept event
                t = t_proposed
                
                # Sample event type
                probs = intensities_proposed / lambda_proposed
                event_type = self.rng.choice(self.num_types, p=probs)
                
                events.append((t, event_type))
                intensity_memory[event_type].append(t)
                event_count += 1
            else:
                # Reject, continue
                t = t_proposed
        
        return events
    
    def _compute_intensities(self, t, intensity_memory):
        """Compute intensity vector at time t."""
        intensities = self.baseline.copy()
        
        for i in range(self.num_types):
            for j in range(self.num_types):
                if len(intensity_memory[j]) > 0:
                    past_times = np.array(intensity_memory[j])
                    lags = t - past_times
                    lags = lags[lags > 0]
                    if len(lags) > 0:
                        contribution = np.sum(self.kernel(i, j, lags))
                        intensities[i] += contribution
        
        return np.maximum(intensities, 0.0)
    
    def compute_intensity_at_events(self, events):
        """
        Compute intensity just before each event.
        
        Parameters
        ----------
        events : list of tuples
            List of (time, event_type)
        
        Returns
        -------
        intensities : ndarray, shape (n_events, n_types)
            Intensity vector before each event
        """
        n_events = len(events)
        intensities = np.zeros((n_events, self.num_types))
        
        for idx, (t, _) in enumerate(events):
            intensity_memory = [[] for _ in range(self.num_types)]
            
            # Add all events before current time
            for t_past, type_past in events[:idx]:
                intensity_memory[type_past].append(t_past)
            
            intensities[idx] = self._compute_intensities(t, intensity_memory)
        
        return intensities

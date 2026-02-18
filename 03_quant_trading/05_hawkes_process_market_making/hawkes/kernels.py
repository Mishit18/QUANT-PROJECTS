"""
Kernel functions for Hawkes processes.
"""
import numpy as np


class ExponentialKernel:
    """Exponential kernel: φ(t) = α * β * exp(-β * t) for t > 0."""
    
    def __init__(self, alpha, beta):
        """
        Parameters
        ----------
        alpha : float
            Excitation amplitude
        beta : float
            Decay rate (must be positive)
        """
        self.alpha = alpha
        self.beta = beta
        
        if beta <= 0:
            raise ValueError("Decay rate beta must be positive")
    
    def __call__(self, t):
        """Evaluate kernel at time t."""
        if np.isscalar(t):
            return self.alpha * self.beta * np.exp(-self.beta * t) if t > 0 else 0.0
        else:
            t = np.asarray(t)
            result = np.zeros_like(t, dtype=float)
            mask = t > 0
            result[mask] = self.alpha * self.beta * np.exp(-self.beta * t[mask])
            return result
    
    def integral(self, t):
        """Compute integral from 0 to t: ∫₀ᵗ φ(s) ds."""
        if np.isscalar(t):
            return self.alpha * (1 - np.exp(-self.beta * t)) if t > 0 else 0.0
        else:
            t = np.asarray(t)
            result = np.zeros_like(t, dtype=float)
            mask = t > 0
            result[mask] = self.alpha * (1 - np.exp(-self.beta * t[mask]))
            return result


class MultiDimensionalKernel:
    """Multidimensional kernel matrix for Hawkes processes."""
    
    def __init__(self, alpha_matrix, beta_matrix):
        """
        Parameters
        ----------
        alpha_matrix : ndarray, shape (n, n)
            Excitation amplitudes
        beta_matrix : ndarray, shape (n, n)
            Decay rates
        """
        self.alpha = np.asarray(alpha_matrix)
        self.beta = np.asarray(beta_matrix)
        
        if self.alpha.shape != self.beta.shape:
            raise ValueError("Alpha and beta matrices must have same shape")
        
        if len(self.alpha.shape) != 2 or self.alpha.shape[0] != self.alpha.shape[1]:
            raise ValueError("Kernel matrices must be square")
        
        self.num_types = self.alpha.shape[0]
        
        if np.any(self.beta <= 0):
            raise ValueError("All decay rates must be positive")
    
    def __call__(self, i, j, t):
        """
        Evaluate kernel φᵢⱼ(t).
        
        Parameters
        ----------
        i : int
            Target event type
        j : int
            Source event type
        t : float or ndarray
            Time lag(s)
        """
        kernel = ExponentialKernel(self.alpha[i, j], self.beta[i, j])
        return kernel(t)
    
    def integral(self, i, j, t):
        """Compute ∫₀ᵗ φᵢⱼ(s) ds."""
        kernel = ExponentialKernel(self.alpha[i, j], self.beta[i, j])
        return kernel.integral(t)
    
    def check_stationarity(self):
        """
        Check stationarity condition: spectral radius of branching matrix < 1.
        
        Returns
        -------
        is_stationary : bool
        spectral_radius : float
        """
        branching_matrix = self.alpha.copy()
        eigenvalues = np.linalg.eigvals(branching_matrix)
        spectral_radius = np.max(np.abs(eigenvalues))
        return spectral_radius < 1.0, spectral_radius

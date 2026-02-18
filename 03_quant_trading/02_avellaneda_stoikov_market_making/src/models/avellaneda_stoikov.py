"""
Avellaneda-Stoikov Market Making Model

Implements the closed-form solution to the HJB equation for optimal market making
under exponential utility with inventory risk.

References:
    Avellaneda, M., & Stoikov, S. (2008). High-frequency trading in a limit order book.
    Quantitative Finance, 8(3), 217-224.
"""

import numpy as np
from typing import Tuple


class AvellanedaStoikov:
    """
    Avellaneda-Stoikov optimal market making model.
    
    Solves for optimal bid/ask quotes that maximize expected exponential utility
    of terminal wealth under inventory risk.
    
    Mathematical Framework:
    ----------------------
    State: (t, S_t, q_t, x_t)
        t: time
        S_t: mid-price (dS_t = σ dW_t)
        q_t: inventory
        x_t: cash
    
    Control: (δ^bid, δ^ask) - spreads from mid-price
    
    Objective: max E[-exp(-γ(x_T + q_T S_T))]
    
    HJB Equation:
        ∂v/∂t + (σ²/2)∂²v/∂S² + 
        max_{δ^bid, δ^ask} [
            λ^bid(δ^bid)(v(t,S,q+1,x+S-δ^bid) - v) +
            λ^ask(δ^ask)(v(t,S,q-1,x+S+δ^ask) - v)
        ] = 0
    
    Solution (Closed Form):
    ----------------------
    Reservation price:
        r(t,S,q) = S - q·γ·σ²·(T-t)
    
    Optimal spreads:
        δ^bid* = (1/γ)log(1 + γ/κ) + (q + 1/2)·γ·σ²·(T-t)
        δ^ask* = (1/γ)log(1 + γ/κ) - (q - 1/2)·γ·σ²·(T-t)
    
    Optimal quotes:
        P^bid = r - δ^bid*
        P^ask = r + δ^ask*
    """
    
    def __init__(
        self,
        risk_aversion: float,
        volatility: float,
        terminal_time: float,
        intensity_decay: float
    ):
        """
        Initialize Avellaneda-Stoikov model.
        
        Args:
            risk_aversion: γ - risk aversion parameter (higher = more conservative)
            volatility: σ - mid-price volatility
            terminal_time: T - time horizon
            intensity_decay: κ - decay rate of order arrival intensity
        """
        self.gamma = risk_aversion
        self.sigma = volatility
        self.T = terminal_time
        self.kappa = intensity_decay
        
        # Precompute constant spread component
        self.base_spread = (1.0 / self.gamma) * np.log(1.0 + self.gamma / self.kappa)
    
    def reservation_price(self, mid_price: float, inventory: int, time: float) -> float:
        """
        Calculate reservation price (indifference price).
        
        The reservation price is the price at which the market maker is indifferent
        between holding q or q+1 shares. It shifts linearly with inventory to
        account for directional risk.
        
        Formula:
            r(t,S,q) = S - q·γ·σ²·(T-t)
        
        Args:
            mid_price: Current mid-price S_t
            inventory: Current inventory q_t
            time: Current time t
        
        Returns:
            Reservation price r(t,S,q)
        """
        time_to_maturity = self.T - time
        inventory_adjustment = inventory * self.gamma * (self.sigma ** 2) * time_to_maturity
        return mid_price - inventory_adjustment
    
    def optimal_spreads(self, inventory: int, time: float) -> Tuple[float, float]:
        """
        Calculate optimal bid and ask spreads from reservation price.
        
        Spreads are asymmetric based on inventory:
        - Long inventory (q > 0): wider ask, tighter bid (incentivize selling)
        - Short inventory (q < 0): tighter ask, wider bid (incentivize buying)
        
        Formulas:
            δ^bid* = (1/γ)log(1 + γ/κ) + (q + 1/2)·γ·σ²·(T-t)
            δ^ask* = (1/γ)log(1 + γ/κ) - (q - 1/2)·γ·σ²·(T-t)
        
        Args:
            inventory: Current inventory q_t
            time: Current time t
        
        Returns:
            (bid_spread, ask_spread) - spreads from reservation price
        """
        time_to_maturity = self.T - time
        inventory_term = self.gamma * (self.sigma ** 2) * time_to_maturity
        
        # Asymmetric adjustment based on inventory
        bid_spread = self.base_spread + (inventory + 0.5) * inventory_term
        ask_spread = self.base_spread - (inventory - 0.5) * inventory_term
        
        return bid_spread, ask_spread
    
    def optimal_quotes(
        self,
        mid_price: float,
        inventory: int,
        time: float
    ) -> Tuple[float, float]:
        """
        Calculate optimal bid and ask quotes.
        
        Combines reservation price and optimal spreads:
            P^bid = r - δ^bid*
            P^ask = r + δ^ask*
        
        Args:
            mid_price: Current mid-price S_t
            inventory: Current inventory q_t
            time: Current time t
        
        Returns:
            (bid_price, ask_price) - optimal quotes
        """
        r = self.reservation_price(mid_price, inventory, time)
        delta_bid, delta_ask = self.optimal_spreads(inventory, time)
        
        bid_price = r - delta_bid
        ask_price = r + delta_ask
        
        return bid_price, ask_price
    
    def quote_spreads_from_mid(
        self,
        mid_price: float,
        inventory: int,
        time: float
    ) -> Tuple[float, float]:
        """
        Calculate bid/ask spreads from mid-price (not reservation price).
        
        Useful for analysis and comparison with market data.
        
        Args:
            mid_price: Current mid-price S_t
            inventory: Current inventory q_t
            time: Current time t
        
        Returns:
            (bid_spread_from_mid, ask_spread_from_mid)
        """
        bid_price, ask_price = self.optimal_quotes(mid_price, inventory, time)
        
        bid_spread = mid_price - bid_price
        ask_spread = ask_price - mid_price
        
        return bid_spread, ask_spread
    
    def inventory_penalty(self, inventory: int, time: float) -> float:
        """
        Calculate inventory risk penalty.
        
        Measures the cost of holding inventory due to price risk.
        
        Formula:
            penalty = (1/2)·q²·γ·σ²·(T-t)
        
        Args:
            inventory: Current inventory q_t
            time: Current time t
        
        Returns:
            Inventory risk penalty
        """
        time_to_maturity = self.T - time
        return 0.5 * (inventory ** 2) * self.gamma * (self.sigma ** 2) * time_to_maturity
    
    def value_function(
        self,
        cash: float,
        inventory: int,
        mid_price: float,
        time: float
    ) -> float:
        """
        Approximate value function (certainty equivalent).
        
        Under exponential utility, the value function has the form:
            v(t,S,q,x) ≈ -exp(-γ(x + qS - inventory_penalty))
        
        We return the certainty equivalent wealth:
            CE = x + qS - inventory_penalty
        
        Args:
            cash: Current cash x_t
            inventory: Current inventory q_t
            mid_price: Current mid-price S_t
            time: Current time t
        
        Returns:
            Certainty equivalent wealth
        """
        marked_to_market = cash + inventory * mid_price
        penalty = self.inventory_penalty(inventory, time)
        return marked_to_market - penalty

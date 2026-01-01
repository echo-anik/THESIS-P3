"""
Grey Wolf Optimizer for Feature Selection

Implements metaheuristic optimization for selecting optimal sensor subset.

Key Benefits:
- Reduces 127 features to most relevant subset
- Improves model interpretability
- Reduces computational cost
- Maintains detection accuracy

Reference:
Mirjalili, S., Mirjalili, S. M., & Lewis, A. (2014). 
Grey Wolf Optimizer. Advances in Engineering Software, 69, 46-61.
"""

import numpy as np
from typing import Tuple, Callable, List
import random


class GreyWolfOptimizer:
    """
    Grey Wolf Optimizer (GWO) for feature selection
    
    Mimics the hunting behavior of grey wolves:
    - Alpha (α): Best solution
    - Beta (β): Second best
    - Delta (δ): Third best
    - Omega (ω): Rest of the pack
    """
    
    def __init__(
        self,
        n_features: int = 127,
        n_wolves: int = 20,
        max_iter: int = 100,
        min_features: int = 10,
        max_features: int = 50
    ):
        """
        Args:
            n_features: Total number of features
            n_wolves: Population size
            max_iter: Maximum iterations
            min_features: Minimum features to select
            max_features: Maximum features to select
        """
        self.n_features = n_features
        self.n_wolves = n_wolves
        self.max_iter = max_iter
        self.min_features = min_features
        self.max_features = max_features
        
        # Population (binary vectors)
        self.population = None
        
        # Leaders
        self.alpha = None  # Best
        self.beta = None   # Second best
        self.delta = None  # Third best
        
        self.alpha_score = float('-inf')
        self.beta_score = float('-inf')
        self.delta_score = float('-inf')
        
        # History
        self.convergence = []
        
    def _init_population(self) -> np.ndarray:
        """Initialize random binary population"""
        population = np.zeros((self.n_wolves, self.n_features))
        
        for i in range(self.n_wolves):
            # Random number of features to select
            n_select = random.randint(self.min_features, self.max_features)
            # Random feature indices
            selected = random.sample(range(self.n_features), n_select)
            population[i, selected] = 1
            
        return population
    
    def _to_binary(self, position: np.ndarray) -> np.ndarray:
        """Convert continuous position to binary using sigmoid"""
        sigmoid = 1 / (1 + np.exp(-position))
        binary = (sigmoid > np.random.rand(len(position))).astype(int)
        
        # Ensure feature count is within bounds
        n_selected = np.sum(binary)
        if n_selected < self.min_features:
            # Add random features
            zero_idx = np.where(binary == 0)[0]
            add_idx = np.random.choice(zero_idx, self.min_features - n_selected, replace=False)
            binary[add_idx] = 1
        elif n_selected > self.max_features:
            # Remove random features
            one_idx = np.where(binary == 1)[0]
            remove_idx = np.random.choice(one_idx, n_selected - self.max_features, replace=False)
            binary[remove_idx] = 0
            
        return binary
    
    def optimize(
        self,
        fitness_func: Callable[[np.ndarray], float],
        verbose: bool = True
    ) -> Tuple[np.ndarray, float]:
        """
        Run GWO optimization
        
        Args:
            fitness_func: Function that takes binary feature mask and returns fitness score
            verbose: Print progress
        
        Returns:
            best_features: Binary mask of selected features
            best_score: Best fitness score achieved
        """
        if verbose:
            print(f"\nGrey Wolf Optimizer")
            print(f"  Features: {self.n_features}")
            print(f"  Wolves: {self.n_wolves}")
            print(f"  Max iterations: {self.max_iter}")
            print()
        
        # Initialize population
        self.population = self._init_population()
        
        # Evaluate initial population
        for i in range(self.n_wolves):
            fitness = fitness_func(self.population[i])
            self._update_leaders(self.population[i], fitness)
        
        self.convergence.append(self.alpha_score)
        
        # Main loop
        for iteration in range(self.max_iter):
            # Linear decrease of a from 2 to 0
            a = 2 - iteration * (2 / self.max_iter)
            
            for i in range(self.n_wolves):
                # Update position based on alpha, beta, delta
                new_position = self._update_position(i, a)
                
                # Convert to binary
                new_binary = self._to_binary(new_position)
                
                # Evaluate fitness
                fitness = fitness_func(new_binary)
                
                # Update if better
                if fitness > fitness_func(self.population[i]):
                    self.population[i] = new_binary
                
                # Update leaders
                self._update_leaders(new_binary, fitness)
            
            self.convergence.append(self.alpha_score)
            
            if verbose and (iteration + 1) % 10 == 0:
                n_features = np.sum(self.alpha)
                print(f"  Iter {iteration+1:3d}: Best score = {self.alpha_score:.4f}, Features = {n_features}")
        
        if verbose:
            print(f"\n✓ Optimization complete")
            print(f"  Best score: {self.alpha_score:.4f}")
            print(f"  Selected features: {np.sum(self.alpha)} / {self.n_features}")
        
        return self.alpha, self.alpha_score
    
    def _update_position(self, wolf_idx: int, a: float) -> np.ndarray:
        """Update wolf position based on alpha, beta, delta"""
        # Current position (continuous representation)
        X = self.population[wolf_idx].astype(float)
        
        # Calculate coefficients
        r1, r2 = np.random.rand(2)
        A1 = 2 * a * r1 - a
        C1 = 2 * r2
        
        r1, r2 = np.random.rand(2)
        A2 = 2 * a * r1 - a
        C2 = 2 * r2
        
        r1, r2 = np.random.rand(2)
        A3 = 2 * a * r1 - a
        C3 = 2 * r2
        
        # Distance to leaders
        D_alpha = np.abs(C1 * self.alpha - X)
        D_beta = np.abs(C2 * self.beta - X)
        D_delta = np.abs(C3 * self.delta - X)
        
        # New positions
        X1 = self.alpha - A1 * D_alpha
        X2 = self.beta - A2 * D_beta
        X3 = self.delta - A3 * D_delta
        
        # Average position
        new_position = (X1 + X2 + X3) / 3
        
        return new_position
    
    def _update_leaders(self, wolf: np.ndarray, fitness: float):
        """Update alpha, beta, delta leaders"""
        if fitness > self.alpha_score:
            self.delta = self.beta.copy() if self.beta is not None else wolf.copy()
            self.delta_score = self.beta_score
            
            self.beta = self.alpha.copy() if self.alpha is not None else wolf.copy()
            self.beta_score = self.alpha_score
            
            self.alpha = wolf.copy()
            self.alpha_score = fitness
            
        elif fitness > self.beta_score:
            self.delta = self.beta.copy() if self.beta is not None else wolf.copy()
            self.delta_score = self.beta_score
            
            self.beta = wolf.copy()
            self.beta_score = fitness
            
        elif fitness > self.delta_score:
            self.delta = wolf.copy()
            self.delta_score = fitness
    
    def get_selected_features(self) -> List[int]:
        """Get indices of selected features"""
        if self.alpha is None:
            return []
        return list(np.where(self.alpha == 1)[0])


class FeatureSelector:
    """
    Feature selector using GWO for SCADA anomaly detection
    """
    
    def __init__(self, n_features: int = 127):
        self.n_features = n_features
        self.gwo = GreyWolfOptimizer(n_features=n_features)
        self.selected_features = None
        
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        model_factory: Callable,
        n_folds: int = 3
    ) -> List[int]:
        """
        Select features using cross-validation fitness
        
        Args:
            X_train: Training data
            y_train: Training labels
            model_factory: Function that creates model given feature mask
            n_folds: Number of CV folds
        
        Returns:
            List of selected feature indices
        """
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import f1_score
        
        def fitness_func(feature_mask: np.ndarray) -> float:
            """Evaluate feature subset using CV F1 score"""
            selected_idx = np.where(feature_mask == 1)[0]
            
            if len(selected_idx) == 0:
                return 0.0
            
            X_selected = X_train[:, selected_idx]
            
            # Cross-validation
            cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            scores = []
            
            for train_idx, val_idx in cv.split(X_selected, y_train):
                model = model_factory(len(selected_idx))
                model.fit(X_selected[train_idx], y_train[train_idx])
                y_pred = model.predict(X_selected[val_idx])
                scores.append(f1_score(y_train[val_idx], y_pred))
            
            return np.mean(scores)
        
        # Run optimization
        best_mask, best_score = self.gwo.optimize(fitness_func)
        self.selected_features = self.gwo.get_selected_features()
        
        return self.selected_features
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply feature selection to data"""
        if self.selected_features is None:
            raise ValueError("Must call fit() before transform()")
        return X[:, self.selected_features]
    
    def fit_transform(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        model_factory: Callable
    ) -> np.ndarray:
        """Fit and transform in one step"""
        self.fit(X_train, y_train, model_factory)
        return self.transform(X_train)


def demo_gwo():
    """Demo GWO feature selection"""
    print("="*60)
    print("GREY WOLF OPTIMIZER DEMO")
    print("="*60)
    
    # Create synthetic fitness function
    # Features 10, 20, 30, 40, 50 are "important"
    important_features = [10, 20, 30, 40, 50]
    
    def fitness_func(feature_mask):
        # Reward selecting important features
        selected = np.where(feature_mask == 1)[0]
        important_selected = len(set(selected) & set(important_features))
        # Penalize too many features
        penalty = 0.01 * len(selected)
        return important_selected / len(important_features) - penalty
    
    # Run GWO
    gwo = GreyWolfOptimizer(
        n_features=127,
        n_wolves=15,
        max_iter=50,
        min_features=5,
        max_features=20
    )
    
    best_mask, best_score = gwo.optimize(fitness_func)
    selected = gwo.get_selected_features()
    
    print(f"\nSelected features: {selected}")
    print(f"Important features found: {set(selected) & set(important_features)}")
    print(f"Convergence history length: {len(gwo.convergence)}")


if __name__ == "__main__":
    demo_gwo()

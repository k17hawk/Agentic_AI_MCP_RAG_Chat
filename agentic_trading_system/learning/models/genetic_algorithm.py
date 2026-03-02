"""
Genetic Algorithm - Evolutionary optimization for trading parameters
"""
from typing import Dict, List, Optional, Any, Tuple, Callable
import numpy as np
import random
from copy import deepcopy
from datetime import datetime
import json
import os
from utils.logger import logger as  logging

class GeneticAlgorithm:
    """
    Genetic Algorithm - Evolutionary optimization for trading parameters
    
    Features:
    - Population-based optimization
    - Crossover and mutation
    - Fitness evaluation
    - Elitism
    - Parallel evaluation support
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # GA parameters
        self.population_size = config.get("population_size", 100)
        self.generations = config.get("generations", 50)
        self.mutation_rate = config.get("mutation_rate", 0.1)
        self.crossover_rate = config.get("crossover_rate", 0.7)
        self.elite_size = config.get("elite_size", 5)
        self.tournament_size = config.get("tournament_size", 3)
        
        # Parameter bounds
        self.param_bounds = config.get("param_bounds", {})
        
        # Storage
        self.data_dir = config.get("data_dir", "data/genetic")
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Population
        self.population = []
        self.fitness_history = []
        self.best_individual = None
        self.best_fitness = -float('inf')
        
        logging.info(f"✅ GeneticAlgorithm initialized")
    
    def create_individual(self, param_ranges: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
        """
        Create a random individual with parameters within ranges
        """
        individual = {}
        for param_name, (min_val, max_val) in param_ranges.items():
            if isinstance(min_val, int) and isinstance(max_val, int):
                individual[param_name] = random.randint(min_val, max_val)
            else:
                individual[param_name] = random.uniform(min_val, max_val)
        return individual
    
    def initialize_population(self, param_ranges: Dict[str, Tuple[float, float]]):
        """
        Initialize random population
        """
        self.population = []
        for _ in range(self.population_size):
            individual = self.create_individual(param_ranges)
            self.population.append(individual)
        
        logging.info(f"✅ Initialized population of size {self.population_size}")
    
    def evaluate_fitness(self, individual: Dict[str, float], 
                        fitness_func: Callable) -> float:
        """
        Evaluate fitness of an individual
        """
        try:
            fitness = fitness_func(individual)
            return fitness
        except Exception as e:
            logging.error(f"Fitness evaluation error: {e}")
            return -float('inf')
    
    def evaluate_population(self, fitness_func: Callable):
        """
        Evaluate fitness of entire population
        """
        fitness_scores = []
        for individual in self.population:
            fitness = self.evaluate_fitness(individual, fitness_func)
            fitness_scores.append(fitness)
        
        # Track best
        for i, fitness in enumerate(fitness_scores):
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_individual = deepcopy(self.population[i])
        
        return fitness_scores
    
    def select_parent(self, fitness_scores: List[float]) -> Dict[str, float]:
        """
        Select parent using tournament selection
        """
        # Tournament selection
        tournament_indices = random.sample(range(len(self.population)), self.tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        
        # Select best from tournament
        best_idx = tournament_indices[np.argmax(tournament_fitness)]
        return deepcopy(self.population[best_idx])
    
    def crossover(self, parent1: Dict[str, float], parent2: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Perform crossover between two parents
        """
        if random.random() > self.crossover_rate:
            return deepcopy(parent1), deepcopy(parent2)
        
        child1 = {}
        child2 = {}
        
        # Single point crossover
        crossover_point = random.randint(0, len(parent1) - 1)
        param_names = list(parent1.keys())
        
        for i, param in enumerate(param_names):
            if i < crossover_point:
                child1[param] = parent1[param]
                child2[param] = parent2[param]
            else:
                child1[param] = parent2[param]
                child2[param] = parent1[param]
        
        return child1, child2
    
    def mutate(self, individual: Dict[str, float], 
              param_ranges: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
        """
        Mutate an individual
        """
        mutated = deepcopy(individual)
        
        for param_name, value in individual.items():
            if random.random() < self.mutation_rate:
                min_val, max_val = param_ranges[param_name]
                
                # Gaussian mutation
                if isinstance(min_val, int) and isinstance(max_val, int):
                    # Integer mutation
                    std = max(1, (max_val - min_val) * 0.1)
                    new_value = int(value + random.gauss(0, std))
                    mutated[param_name] = max(min_val, min(max_val, new_value))
                else:
                    # Float mutation
                    std = (max_val - min_val) * 0.1
                    new_value = value + random.gauss(0, std)
                    mutated[param_name] = max(min_val, min(max_val, new_value))
        
        return mutated
    
    def evolve(self, fitness_func: Callable, 
              param_ranges: Dict[str, Tuple[float, float]],
              generations: int = None) -> Dict[str, Any]:
        """
        Run genetic algorithm evolution
        """
        if generations is None:
            generations = self.generations
        
        # Initialize population if empty
        if not self.population:
            self.initialize_population(param_ranges)
        
        history = []
        
        for generation in range(generations):
            # Evaluate fitness
            fitness_scores = self.evaluate_population(fitness_func)
            
            # Track best
            gen_best_fitness = max(fitness_scores)
            gen_best_idx = np.argmax(fitness_scores)
            gen_best = self.population[gen_best_idx]
            
            history.append({
                "generation": generation,
                "best_fitness": gen_best_fitness,
                "avg_fitness": np.mean(fitness_scores),
                "best_individual": gen_best
            })
            
            logging.info(f"Generation {generation}: Best={gen_best_fitness:.4f}, Avg={np.mean(fitness_scores):.4f}")
            
            # Create new population
            new_population = []
            
            # Elitism - keep best individuals
            elite_indices = np.argsort(fitness_scores)[-self.elite_size:]
            for idx in elite_indices:
                new_population.append(deepcopy(self.population[idx]))
            
            # Create rest through selection, crossover, mutation
            while len(new_population) < self.population_size:
                # Select parents
                parent1 = self.select_parent(fitness_scores)
                parent2 = self.select_parent(fitness_scores)
                
                # Crossover
                child1, child2 = self.crossover(parent1, parent2)
                
                # Mutate
                child1 = self.mutate(child1, param_ranges)
                child2 = self.mutate(child2, param_ranges)
                
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)
            
            self.population = new_population
        
        # Final evaluation
        fitness_scores = self.evaluate_population(fitness_func)
        
        self.fitness_history = history
        
        return {
            "best_individual": self.best_individual,
            "best_fitness": self.best_fitness,
            "history": history,
            "population_size": self.population_size,
            "generations_run": generations
        }
    
    def optimize_parameters(self, param_ranges: Dict[str, Tuple[float, float]],
                           fitness_func: Callable,
                           n_runs: int = 3) -> Dict[str, Any]:
        """
        Run multiple optimizations and aggregate results
        """
        all_runs = []
        
        for run in range(n_runs):
            logging.info(f"Starting optimization run {run + 1}/{n_runs}")
            
            # Reset for each run
            self.population = []
            self.best_individual = None
            self.best_fitness = -float('inf')
            
            result = self.evolve(fitness_func, param_ranges)
            all_runs.append(result)
        
        # Aggregate results
        best_overall = max(all_runs, key=lambda x: x["best_fitness"])
        
        # Calculate parameter importance
        param_importance = self._calculate_param_importance(all_runs, param_ranges)
        
        return {
            "best_overall": best_overall["best_individual"],
            "best_fitness": best_overall["best_fitness"],
            "all_runs": all_runs,
            "param_importance": param_importance,
            "num_runs": n_runs
        }
    
    def _calculate_param_importance(self, runs: List[Dict], 
                                   param_ranges: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
        """
        Calculate parameter importance based on variation in best individuals
        """
        param_values = {param: [] for param in param_ranges.keys()}
        
        for run in runs:
            if run["best_individual"]:
                for param, value in run["best_individual"].items():
                    if param in param_values:
                        param_values[param].append(value)
        
        importance = {}
        for param, values in param_values.items():
            if len(values) > 1:
                # Use coefficient of variation as importance measure
                mean = np.mean(values)
                std = np.std(values)
                if mean != 0:
                    importance[param] = std / mean
                else:
                    importance[param] = std
            else:
                importance[param] = 0
        
        return importance
    
    def save_population(self, filename: str = None):
        """
        Save current population to disk
        """
        if filename is None:
            filename = f"population_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = os.path.join(self.data_dir, filename)
        
        data = {
            "population": self.population,
            "best_individual": self.best_individual,
            "best_fitness": self.best_fitness,
            "fitness_history": self.fitness_history,
            "config": self.config,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            logging.info(f"💾 Saved population to {filepath}")
        except Exception as e:
            logging.error(f"Error saving population: {e}")
    
    def load_population(self, filename: str):
        """
        Load population from disk
        """
        filepath = os.path.join(self.data_dir, filename)
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.population = data["population"]
            self.best_individual = data["best_individual"]
            self.best_fitness = data["best_fitness"]
            self.fitness_history = data["fitness_history"]
            
            logging.info(f"📂 Loaded population from {filepath}")
            
        except Exception as e:
            logging.error(f"Error loading population: {e}")
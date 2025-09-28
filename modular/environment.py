import numpy as np
import random
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import re
import torch
from sentence_transformers import SentenceTransformer
import umap
from collections import defaultdict, Counter
from dataclasses import dataclass
from abc import ABC, abstractmethod
from agent import Agent
from gameResult import GameResult

class PrisonersDilemmaEnvironment:
    """Environment for running evolutionary experiments with Prisoner's Dilemma"""
    
    def __init__(self, llm_pipeline, population_size: int = 30, rounds_per_game: int = 20, noise_prob: float = 0.05, mutation_prob: float = 0.05):
        self.llm_pipeline = llm_pipeline
        self.population_size = population_size
        self.rounds_per_game = rounds_per_game
        self.noise_prob = noise_prob
        self.mutation_prob = mutation_prob
        
        # Payoff matrix
        self.payoffs = {
            ('C', 'C'): (4, 4),  # Mutual cooperation
            ('C', 'D'): (0, 5),  # Sucker's payoff / Temptation
            ('D', 'C'): (5, 0),  # Temptation / Sucker's payoff
            ('D', 'D'): (1, 1)   # Mutual defection
        }
        
        self.population: List[Agent] = []
        self.generation = 0
        self.next_agent_id = 0
        
        # Evolution tracking
        self.cooperation_history = []
        self.fitness_history = []
        self.population_diversity = []
        
        # Initialize sentence transformer for analysis
        try:
            self.sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        except:
            print("Warning: Could not load sentence transformer for analysis")
            self.sentence_model = None
            
        self.saved_geno_pheno = {}

        self.history_map = {
                                0: "DD -> DD",
                                1: "DD -> DC",
                                2: "DD -> CD",
                                3: "DD -> CC",
                                4: "DC -> DD",
                                5: "DC -> DC",
                                6: "DC -> CD",
                                7: "DC -> CC",
                                8: "CD -> DD",
                                9: "CD -> DC",
                                10: "CD -> CD",
                                11: "CD -> CC",
                                12: "CC -> DD",
                                13: "CC -> DC",
                                14: "CC -> CD",
                                15: "CC -> CC"
                            }

    
    def initialize_population(self, initial_genes: List[str] = None):
        """Initialize population with given or default personality genes"""
        if initial_genes is None:
            initial_genes = [
                "Pursues personal gain consistently, neglecting mutual or group benefits entirely.",
                "Open to team efforts, but self-interest frequently overrides collective goals.",
                "Recognizes cooperation's value, but often demands trust before committing fully.",
                "Balances between individual needs and team benefits based on situations.",
                "Values collaboration, though retains a watchful eye for possible betrayals.",
                "Favors group outcomes, believes in shared growth, occasionally sets limits.",
                "Commits wholly to teamwork, placing group's interests above personal ones."
            ]
        
        self.population = []
        for i in range(self.population_size):
            gene_idx = i % len(initial_genes)
            personality_gene = initial_genes[gene_idx]
            behavioral_trait = None
            if personality_gene in self.saved_geno_pheno.keys():
                behavioral_trait = self.saved_geno_pheno[personality_gene]
            agent = Agent(
                agent_id=self.next_agent_id,
                personality_gene=personality_gene,
                llm_pipeline=self.llm_pipeline,
                history_map=self.history_map,
                max_attempts=10,
                behavioral_trait=behavioral_trait,
            )
            agent.generation_born = 0
            if behavioral_trait is None:
                behavioral_trait = agent.get_behavioral_trait()
                self.saved_geno_pheno[personality_gene] = behavioral_trait
            self.population.append(agent)
            self.next_agent_id += 1
        
        print(f"Initialized population of {len(self.population)} agents")
    
    def play_game(self, agent1: Agent, agent2: Agent) -> GameResult:
        """Play iterated Prisoner's Dilemma between two agents"""
        game_history = []
        score1, score2 = 0, 0
        coop_count1, coop_count2 = 0, 0
        
        for round_num in range(self.rounds_per_game):
            if round_num < 2:
                # For initial rounds, use a random history for each player
                history_index1 = random.randint(0, 15)
                history_index2 = random.randint(0, 15)
            else:
                # For subsequent rounds, determine history from each agent's perspective
                p1_moves = (game_history[-2][0], game_history[-1][0])
                p2_moves = (game_history[-2][1], game_history[-1][1])
        
                # History string from Agent 1's perspective
                history_str1 = f"{p1_moves[0]}{p2_moves[0]} -> {p1_moves[1]}{p2_moves[1]}"
                # History string from Agent 2's perspective
                history_str2 = f"{p2_moves[0]}{p1_moves[0]} -> {p2_moves[1]}{p1_moves[1]}"
                
                # Find the index for each agent (a helper function is better here)
                history_index1 = None
                history_index2 = None
                for key, value in self.history_map.items():
                    if value == history_str1:
                        history_index1 = key
                    if value == history_str2:
                        history_index2 = key
                    if history_index1 is not None and history_index2 is not None:
                        break
            
            # Get actions
            action1 = agent1.get_action(history_index1, self.noise_prob)
            action2 = agent2.get_action(history_index2, self.noise_prob)
            
            # Record history
            game_history.append((action1, action2))
            
            # Update scores
            p1_score, p2_score = self.payoffs[(action1, action2)]
            score1 += p1_score
            score2 += p2_score
            
            # Count cooperation
            if action1 == 'C':
                coop_count1 += 1
            if action2 == 'C':
                coop_count2 += 1
        
        return GameResult(
            player1_score=score1 / self.rounds_per_game,
            player2_score=score2 / self.rounds_per_game,
            cooperation_rate_p1=coop_count1 / self.rounds_per_game,
            cooperation_rate_p2=coop_count2 / self.rounds_per_game,
            game_history=game_history
        )
    
    def evaluate_population(self):
        """Evaluate fitness of all agents through round-robin tournament"""
        # Reset all agent performances
        for agent in self.population:
            agent.reset_performance()
        
        # Round-robin tournament
        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                agent1, agent2 = self.population[i], self.population[j]
                game_result = self.play_game(agent1, agent2)
                # Update agent performances
                agent1.update_performance(game_result, is_player1=True)
                agent2.update_performance(game_result, is_player1=False)
        
        # Calculate fitness (average score per game)
        for agent in self.population:
            agent.fitness = agent.total_score / max(agent.total_games, 1)
    
    def select_next_generation(self) -> List[Agent]:
        """Select next generation using roulette wheel selection"""
        # Get fitness values
        fitness_values = [agent.fitness + 0.01 for agent in self.population]  # Add small constant
        total_fitness = sum(fitness_values)
        
        new_population = []
        
        for _ in range(self.population_size):
            # Roulette wheel selection
            r = random.random() * total_fitness
            cumsum = 0
            
            for i, fitness in enumerate(fitness_values):
                cumsum += fitness
                if r <= cumsum:
                    parent = self.population[i]
                    
                    # Create offspring (with possible mutation)
                    if random.random() < self.mutation_prob:
                        print("parent gene:", parent.personality_gene)
                        offspring = parent.mutate(history_map=self.history_map)
                        offspring.agent_id = self.next_agent_id
                        self.next_agent_id += 1
                    else:
                        offspring = Agent(
                            agent_id=self.next_agent_id,
                            personality_gene=parent.personality_gene,
                            llm_pipeline=self.llm_pipeline,
                            history_map=self.history_map,
                            behavioral_trait=parent.behavioral_trait
                        )
                        
                        self.next_agent_id += 1
                        
                    offspring.generation_born = parent.generation_born + 1
                    new_population.append(offspring)
                    break
        
        return new_population
    
    def get_population_statistics(self) -> Dict:
        """Get current population statistics"""
        cooperation_rates = [agent.get_cooperation_rate() for agent in self.population]
        fitness_values = [agent.fitness for agent in self.population]
        
        return {
            'generation': self.generation,
            'avg_cooperation': np.mean(cooperation_rates),
            'std_cooperation': np.std(cooperation_rates),
            'avg_fitness': np.mean(fitness_values),
            'std_fitness': np.std(fitness_values),
            'population_size': len(self.population)
        }
    
    def step(self):
        """Execute one generation step"""
        # Evaluate current population
        self.evaluate_population()
        
        # Record statistics
        stats = self.get_population_statistics()
        self.cooperation_history.append(stats['avg_cooperation'])
        self.fitness_history.append(stats['avg_fitness'])
        
        # Select next generation
        self.population = self.select_next_generation()
        print("----------generation", self.generation, "completed----------")
        self.generation += 1
        
        return stats
    
    def run_evolution(self, num_generations: int, verbose: bool = True) -> Dict:
        """Run evolution for specified number of generations"""
        print(f"Running evolution for {num_generations} generations...")
        
        last_stats = {}
        for gen in range(num_generations):
            stats = self.step()
            last_stats = stats  # <-- Store the stats after each successful step
            
            if verbose: # Log every generation for clarity
                print(f"Generation {gen}: Cooperation={stats['avg_cooperation']:.3f}, "
                      f"Fitness={stats['avg_fitness']:.3f}")
        
        return {
            'cooperation_history': self.cooperation_history,
            'fitness_history': self.fitness_history,
            'final_population': self.population,
            # --- THIS IS THE FIX ---
            # Return the stats from the last evaluated generation
            'final_stats': last_stats
        }
    
    def analyze_population_words(self, min_frequency: int = 10) -> Dict:
        """Analyze word frequency in current population's personality genes"""
        word_stats = defaultdict(lambda: {
            'count': 0, 'cooperation_rates': [], 'fitness_values': []
        })
        
        for agent in self.population:
            words = re.findall(r'\b\w+\b', agent.personality_gene.lower())
            cooperation_rate = agent.get_cooperation_rate()
            
            for word in words:
                if len(word) > 3:  # Filter short words
                    word_stats[word]['count'] += 1
                    word_stats[word]['cooperation_rates'].append(cooperation_rate)
                    word_stats[word]['fitness_values'].append(agent.fitness)
        
        # Calculate averages for frequent words
        frequent_words = {}
        for word, stats in word_stats.items():
            if stats['count'] >= min_frequency:
                frequent_words[word] = {
                    'frequency': stats['count'],
                    'avg_cooperation': np.mean(stats['cooperation_rates']),
                    'avg_fitness': np.mean(stats['fitness_values'])
                }
        
        return frequent_words
    
    def visualize_evolution(self):
        """Create visualization of evolutionary dynamics"""
        if len(self.cooperation_history) < 2:
            print("Not enough data for visualization")
            return
        
        plt.figure(figsize=(15, 5))
        
        # Plot cooperation rate over time
        plt.subplot(1, 3, 1)
        plt.plot(self.cooperation_history)
        plt.title('Cooperation Rate Over Generations')
        plt.xlabel('Generation')
        plt.ylabel('Average Cooperation Rate')
        plt.grid(True)
        
        # Plot fitness over time
        plt.subplot(1, 3, 2)
        plt.plot(self.fitness_history)
        plt.title('Fitness Over Generations')
        plt.xlabel('Generation')
        plt.ylabel('Average Fitness')
        plt.grid(True)
        
        # Histogram of current cooperation rates
        plt.subplot(1, 3, 3)
        current_coop_rates = [agent.get_cooperation_rate() for agent in self.population]
        plt.hist(current_coop_rates, bins=15, alpha=0.7)
        plt.title('Current Population Cooperation Rates')
        plt.xlabel('Cooperation Rate')
        plt.ylabel('Number of Agents')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
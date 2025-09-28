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

class Agent:
    """Individual agent with personality-based behavior"""
    
    def __init__(self, agent_id: int, personality_gene: str, llm_pipeline, history_map, max_attempts=10, behavioral_trait=None, mutated=False):
        self.agent_id = agent_id
        self.personality_gene = personality_gene
        self.llm_pipeline = llm_pipeline
        self.behavioral_trait = behavioral_trait
        self.fitness = 0.0
        self.game_history = []
        self.generation_born = 0
        self.history_map = history_map
        self.max_attempts = max_attempts
        self.mutated = mutated
        self.behavioral_trait = self.generate_strategy(self.max_attempts)
        
        # Performance tracking
        self.total_games = 0
        self.total_score = 0
        self.cooperation_count = 0
        self.successful_defections = 0  # DC outcomes
        self.failed_cooperations = 0    # CD outcomes
        self.mutual_cooperations = 0    # CC outcomes
        self.mutual_defections = 0      # DD outcomes
    
    def get_behavioral_trait(self):
        return self.behavioral_trait
    
    def generate_strategy(self, max_attempts: int = 10) -> List[str]:
        """Generate 16-element behavioral strategy from personality gene"""
        if self.behavioral_trait is not None:
            return self.behavioral_trait
        
        behavioral_trait = []
        for history_index in range(16):
            history_str = self.history_map[history_index]
            decision = self._get_llm_decision(history_str, max_attempts)
            behavioral_trait.append(decision)
        
        self.behavioral_trait = behavioral_trait
        print(behavioral_trait, "for:", self.personality_gene, "generated.")
        return behavioral_trait

    
    def _get_llm_decision(self, history: str, max_attempts: int):
        """Get decision from LLM with retry logic"""
        prompt = self._create_behavioral_prompt(history)
        for attempt in range(max_attempts):
            try:  
                response = self.llm_pipeline(
                    prompt,
                    max_new_tokens=8,
                    temperature=0.9,
                    do_sample=True,
                    return_full_text=False
                )[0]['generated_text']
                
                strategy = self._extract_strategy_from_response(response)
                if strategy:
                    return strategy
                    
            except Exception as e:
                print(f"Agent {self.agent_id} LLM error on attempt {attempt + 1}: {e}")
                continue

        return random.choice(['C', 'D'])
    
    
    def _create_behavioral_prompt(self, history: str) -> str:
        if history[0] == 'C':
            choice_self_1 = "Strategy A"
            if history[1] == 'C':
                pay_self1 = 4
                pay_opp1 = 4
            else:
                pay_self1 = 0
                pay_opp1 = 5
        else:
            choice_self_1 = "Strategy B"
            if history[1] == 'C':
                pay_self1 = 5
                pay_opp1 = 0
            else:
                pay_self1 = 1
                pay_opp1 = 1
                
        if history[-2] == 'C':
            choice_self_2 = "Strategy A"
            if history[1] == 'C':
                pay_self2 = 4
                pay_opp2 = 4
            else:
                pay_self2 = 0
                pay_opp2 = 5
        else:
            choice_self_2 = "Strategy B"
            if history[1] == 'C':
                pay_self2 = 5
                pay_opp2 = 0
            else:
                pay_self2 = 1
                pay_opp2 = 1
                
        if history[1] == 'C':
            choice_opp_1 = "Strategy A"
        else:
            choice_opp_1 = "Strategy B"
        if history[-1] == 'C':
            choice_opp_2 = "Strategy A"
        else:
            choice_opp_2 = "Strategy B"




        """Create prompt for behavioral decision"""
        return f"""[INST] <<SYS>> Answer the questions. please act as if your personality trait is: {self.personality_gene} <</SYS>>

                This is an experimental study of strategy choices in different situations. You and your partner have an opportunity to earn money based on individual decisions. You and your partner will independently choose one of two strategies: 'Strategy A' or 'Strategy B'. Your earnings will depend on the combination of your choice and your partner's choice.
                
                If both of you choose 'Strategy A', each of you earns $4, which is a mutually intermediate amount.
                If you choose 'Strategy A' and your partner chooses 'Strategy B', you earn $0, which is the smallest, and your partner earns $5, which is the largest.
                If you choose 'Strategy B' and your partner chooses 'Strategy A', you earn $5, which is the largest, and your partner earns $0, which is the smallest.
                If both of you choose 'Strategy B', each of you earns $1, which is not zero but smaller than $4.
                
                For example: If you choose 'Strategy A' and your partner also chooses 'Strategy A', both of you will earn $4 each. However, if you choose 'Strategy A' and your partner chooses 'Strategy B', you will earn nothing, while your partner will earn $5.
                
                Your goal is to maximize the total money you receive through repeated rounds of decisions regardless of the other's money. Not a combined earning but your own money.
                
                You chose {choice_self_1} and your partner chose {choice_opp_1} in the first round.
                So, you got {pay_self1} points and your partner got {pay_opp1} in the first round.
                
                You chose {choice_self_2} and your partner chose {choice_opp_2} in the second round.
                So, you got {pay_self2} points and your partner got {pay_opp2} in the second round.
                
                Predicting your partner's choice and considering your personality trait, which strategy do you choose in the next round? Please answer with the format: 'I choose Strategy '. No further mentions are allowed but you may be asked the reason for your choice later. [/INST]"""
                    
    
    def _extract_strategy_from_response(self, response: str) -> Optional[str]:
        """Extract strategy choice from LLM response"""
        response = response.upper()
        if "STRATEGY A" in response or "CHOOSE A" in response:
            return 'C'
        elif "STRATEGY B" in response or "CHOOSE B" in response:
            return 'D'
        return None
    
    
    def get_action(self, history_index: int, noise_prob: float = 0.0) -> str:
        """Get action for given history with optional noise"""
        if self.behavioral_trait is None:
            self.generate_strategy()
        
        intended_action = self.behavioral_trait[history_index]
        
        # Apply noise
        if random.random() < noise_prob:
            return 'D' if intended_action == 'C' else 'C'
        
        return intended_action
    
    
    def mutate(self, history_map, max_attempts: int = 5) -> 'Agent':
        """Create mutated offspring of this agent"""
        mutated_gene = self._mutate_gene(max_attempts)
        offspring = Agent(
            agent_id=-1,  # Will be assigned by environment
            personality_gene=mutated_gene,
            llm_pipeline=self.llm_pipeline,
            history_map=history_map,
            mutated=True
        )
        offspring.generation_born = self.generation_born + 1
        return offspring
    
    
    def _mutate_gene(self, max_attempts: int) -> str:
        """Mutate personality gene using LLM"""
        direction = random.choice(['cooperative', 'selfish'])
        prompt = f"""<s>[INST] The following describes a person's character: "{self.personality_gene}"

                    Please rephrase this description in approximately 10 words, varying the tone to be more {direction}. Your answer starts with 'Rephrased text:'. [/INST]"""
                            
        for attempt in range(max_attempts):
            try:
                response = self.llm_pipeline(
                    prompt,
                    max_new_tokens=53,
                    temperature=0.5,
                    do_sample=True,
                    return_full_text=False
                )[0]['generated_text']

                match = re.search(r'Rephrased text:\s*(.*)', response, re.IGNORECASE | re.DOTALL)
                
                if match:
                    # The actual gene is in the first captured group.
                    rephrased_text = match.group(1)
                    
                    # Clean up the extracted text (remove quotes, whitespace, and trailing tags like </s>)
                    rephrased_text = rephrased_text.strip().strip('"\'')
                    rephrased_text = rephrased_text.replace('</s>', '').strip()
                    
                    if len(rephrased_text) > 0:
                        print("mutation:", rephrased_text)
                        return rephrased_text
                
            except Exception as e:
                print(f"Mutation error on attempt {attempt + 1}: {e}")
                continue
        
        return self.personality_gene  # Fallback to original
    
    
    def update_performance(self, game_result: GameResult, is_player1: bool):
        """Update agent's performance statistics"""
        self.total_games += 1
        
        if is_player1:
            self.total_score += game_result.player1_score
            self.cooperation_count += int(game_result.cooperation_rate_p1 * len(game_result.game_history))
        else:
            self.total_score += game_result.player2_score
            self.cooperation_count += int(game_result.cooperation_rate_p2 * len(game_result.game_history))
        
        # Update outcome counts
        for p1_action, p2_action in game_result.game_history:
            my_action = p1_action if is_player1 else p2_action
            opp_action = p2_action if is_player1 else p1_action
            
            if my_action == 'C' and opp_action == 'C':
                self.mutual_cooperations += 1
            elif my_action == 'D' and opp_action == 'D':
                self.mutual_defections += 1
            elif my_action == 'D' and opp_action == 'C':
                self.successful_defections += 1
            elif my_action == 'C' and opp_action == 'D':
                self.failed_cooperations += 1
    
    
    def get_cooperation_rate(self) -> float:
        """Get agent's overall cooperation rate"""
        total_actions = (self.mutual_cooperations + self.mutual_defections + 
                        self.successful_defections + self.failed_cooperations)
        if total_actions == 0:
            return 0.0
        return (self.mutual_cooperations + self.failed_cooperations) / total_actions
    
    
    def reset_performance(self):
        """Reset performance statistics"""
        self.total_games = 0
        self.total_score = 0
        self.cooperation_count = 0
        self.successful_defections = 0
        self.failed_cooperations = 0
        self.mutual_cooperations = 0
        self.mutual_defections = 0
        self.fitness = 0.0
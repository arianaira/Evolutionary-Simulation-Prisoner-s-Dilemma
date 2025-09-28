from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional

@dataclass
class GameResult:
    """Container for game results"""
    player1_score: float
    player2_score: float
    cooperation_rate_p1: float
    cooperation_rate_p2: float
    game_history: List[Tuple[str, str]]
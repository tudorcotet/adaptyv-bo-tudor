import random
import itertools
from typing import List, Dict, Optional
from config.optimization import GeneratorConfig
from generator.base import BaseGenerator

class CombinatorialGenerator(BaseGenerator):
    """
    A generator that creates new sequences by combinatorial mutation of initial sequences.

    Attributes:
        alphabet (List[str]): List of possible amino acids.
        sequences (List[str]): List of initial sequences.
        acquired_sequences (Set[str]): Set of sequences that have been generated.
        indices_to_mutate (Optional[List[int]]): Indices of positions to mutate in the sequences.
        all_candidates (List[str]): List of all possible candidate sequences.
    """

    def __init__(self, config: GeneratorConfig, initial_sequences: List[str], indices_to_mutate: Optional[List[int]] = None):
        super().__init__(config)
        self.alphabet = list(config.alphabet)
        self.sequences = initial_sequences
        self.acquired_sequences = set(initial_sequences)
        self.indices_to_mutate = indices_to_mutate
        self.all_candidates = self.generate_all()

    def generate(self, n_candidates: int) -> List[str]:
        """
        Generate a specified number of new candidate sequences.

        Args:
            n_candidates (int): Number of candidates to generate.

        Returns:
            List[str]: List of generated candidate sequences.
        """
        available_candidates = [c for c in self.all_candidates if c not in self.acquired_sequences]
        selected = random.sample(available_candidates, min(n_candidates, len(available_candidates)))
        self.acquired_sequences.update(selected)
        return selected

    def generate_all(self) -> List[str]:
        """
        Generate all possible mutant sequences based on the initial sequences.

        Returns:
            List[str]: List of all possible mutant sequences.
        """
        all_mutants = set()
        for seq in self.sequences:
            mutants = self._generate_mutants(seq)
            all_mutants.update(mutants)
        return list(all_mutants - self.acquired_sequences)

    def _generate_mutants(self, sequence: str) -> List[str]:
        """
        Generate all possible mutants for a given sequence.

        Args:
            sequence (str): The sequence to mutate.

        Returns:
            List[str]: List of all possible mutant sequences.
        """
        if self.indices_to_mutate is None:
            self.indices_to_mutate = list(range(len(sequence)))

        mutant_positions = []
        for idx in self.indices_to_mutate:
            mutant_positions.append([aa for aa in self.alphabet if aa != sequence[idx]])

        mutants = []
        for mutation in itertools.product(*mutant_positions):
            mutant = list(sequence)
            for idx, aa in zip(self.indices_to_mutate, mutation):
                mutant[idx] = aa
            mutants.append(''.join(mutant))

        return mutants

    def update_sequences(self, new_sequences: List[str]):
        """
        Update the list of sequences and acquired sequences with new sequences.

        Args:
            new_sequences (List[str]): List of new sequences to add.
        """
        self.sequences.extend(new_sequences)
        self.acquired_sequences.update(new_sequences)

class BenchmarkGenerator(BaseGenerator):
    """
    A generator that selects sequences from a predefined benchmark dataset.

    Attributes:
        benchmark_sequences (List[str]): List of all sequences in the benchmark dataset.
        acquired_sequences (Set[str]): Set of sequences that have been generated.
        all_candidates (List[str]): List of all possible candidate sequences.
    """

    def __init__(self, config: GeneratorConfig, benchmark_data: Dict[str, float], initial_sequences: List[str]):
        super().__init__(config)
        self.benchmark_sequences = list(benchmark_data.keys())
        self.acquired_sequences = set(initial_sequences)
        self.all_candidates = self.generate_all()

    def generate(self, n_candidates: int) -> List[str]:
        """
        Generate a specified number of new candidate sequences from the benchmark dataset.

        Args:
            n_candidates (int): Number of candidates to generate.

        Returns:
            List[str]: List of generated candidate sequences.
        """
        available_candidates = [c for c in self.all_candidates if c not in self.acquired_sequences]
        selected = random.sample(available_candidates, min(n_candidates, len(available_candidates)))
        return selected

    def generate_all(self) -> List[str]:
        """
        Generate all possible candidate sequences from the benchmark dataset.

        Returns:
            List[str]: List of all possible candidate sequences.
        """
        available_candidates = [seq for seq in self.benchmark_sequences if seq not in self.acquired_sequences]
        if not available_candidates:
            self.logger.warning("All benchmark sequences have been acquired. Returning all sequences.")
            return self.benchmark_sequences
        return available_candidates

    def update_sequences(self, new_sequences: List[str]):
        """
        Update the set of acquired sequences with new sequences.

        Args:
            new_sequences (List[str]): List of new sequences to add.
        """
        self.acquired_sequences.update(new_sequences)

class MutationGenerator(BaseGenerator):
    """
    A generator that creates new sequences by random mutation of existing sequences.

    Attributes:
        alphabet (List[str]): List of possible amino acids.
        sequences (List[str]): List of sequences to mutate.
        acquired_sequences (Set[str]): Set of sequences that have been generated.
        candidate_pool (List[str]): Pool of candidate sequences.
    """

    def __init__(self, config: GeneratorConfig, initial_sequences: List[str]):
        super().__init__(config)
        self.alphabet = list(config.alphabet)
        self.sequences = initial_sequences
        self.acquired_sequences = set(initial_sequences)
        self.candidate_pool = []

    def generate(self, n_candidates: int) -> List[str]:
        """
        Generate a specified number of new candidate sequences by mutation.

        Args:
            n_candidates (int): Number of candidates to generate.

        Returns:
            List[str]: List of generated candidate sequences.
        """
        new_candidates = self._generate_mutants(n_candidates)
        self.candidate_pool.extend(new_candidates)

        available_candidates = [c for c in self.candidate_pool if c not in self.acquired_sequences]
        selected = random.sample(available_candidates, min(n_candidates, len(available_candidates)))

        self.acquired_sequences.update(selected)
        self.candidate_pool = [c for c in self.candidate_pool if c not in selected]

        return selected

    def generate_all(self) -> List[str]:
        """
        Generate all possible candidate sequences from the current candidate pool.

        Returns:
            List[str]: List of all possible candidate sequences.
        """
        return [c for c in self.candidate_pool if c not in self.acquired_sequences]

    def _generate_mutants(self, n_candidates: int) -> List[str]:
        """
        Generate a specified number of mutant sequences.

        Args:
            n_candidates (int): Number of mutants to generate.

        Returns:
            List[str]: List of generated mutant sequences.
        """
        mutants = []
        while len(mutants) < n_candidates:
            seq = random.choice(self.sequences)
            mutant = list(seq)
            pos = random.randint(0, len(seq) - 1)
            new_aa = random.choice([aa for aa in self.alphabet if aa != mutant[pos]])
            mutant[pos] = new_aa
            mutant_str = ''.join(mutant)
            if mutant_str not in self.acquired_sequences:
                mutants.append(mutant_str)
        return mutants

    def update_sequences(self, new_sequences: List[str]):
        """
        Update the list of sequences and acquired sequences with new sequences.

        Args:
            new_sequences (List[str]): List of new sequences to add.
        """
        self.sequences.extend(new_sequences)
        self.acquired_sequences.update(new_sequences)

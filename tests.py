"""
Test of the markov chain functions
"""
import numpy as np
import markov_chain_polymer_sampling as mcps

# Test stable state 1
transition_matrix_1 = np.array([[0.5, 0.5], [0.5, 0.5]])
stable_state = mcps.stable_state_probability(transition_matrix_1)
stable_state = [round(s, 3) for s in stable_state]
assert stable_state == [0.5, 0.5]

# Test stable state 2
transition_matrix_2 = np.array([[0.9, 0.1], [0.6, 0.4]])
stable_state = mcps.stable_state_probability(transition_matrix_2)
stable_state = [round(s, 3) for s in stable_state]
assert stable_state == [0.857, 0.143]

# Test stable state 3
transition_matrix_3 = np.array([[0.8, 0.1, 0.1], [0.3, 0.4, 0.3], [0.4, 0.4, 0.2]])
stable_state = mcps.stable_state_probability(transition_matrix_3)
stable_state = [round(s, 3) for s in stable_state]
assert stable_state == [0.632, 0.211, 0.158]

# Test lengths distribution 1
STATES = ["A", "B", "C"]
trajectory_1 = "ABCAABBCCAAABBBCCC"
length_dist = mcps.length_distribution(STATES, trajectory_1)
assert length_dist == {"A": {1: 1, 2: 1, 3: 1}, "B": {1: 1, 2: 1, 3: 1}, "C": {1: 1, 2: 1, 3: 1}}

# Test lengths distribution 2
trajectory_2 = "ABAABBAAABBB"
length_dist = mcps.length_distribution(STATES, trajectory_2)
assert length_dist == {"A": {1: 1, 2: 1, 3: 1}, "B": {1: 1, 2: 1, 3: 1}, "C": {}}

# Test lengths distribution 3
trajectory_2 = "ABAABBAAABBBABB"
length_dist = mcps.length_distribution(STATES, trajectory_2)
assert length_dist == {"A": {1: 2, 2: 1, 3: 1}, "B": {1: 1, 2: 2, 3: 1}, "C": {}}


# Test relative frequencies 1
length_dist = {"A": {1: 1, 2: 1, 3: 1}, "B": {1: 1, 2: 1, 3: 1}, "C": {1: 1, 2: 1, 3: 1}}
rel_freq = mcps.relative_frequencies(STATES, length_dist, 18)
assert rel_freq == {
    "A": {1: 1 / 18, 2: 2 / 18, 3: 3 / 18},
    "B": {1: 1 / 18, 2: 2 / 18, 3: 3 / 18},
    "C": {1: 1 / 18, 2: 2 / 18, 3: 3 / 18},
}

# Test relative frequencies 2
length_dist = {"A": {1: 2, 2: 1, 3: 1}, "B": {1: 1, 2: 2, 3: 1}, "C": {}}
rel_freq = mcps.relative_frequencies(STATES, length_dist, 15)
assert rel_freq == {
    "A": {1: 2 / 15, 2: 2 / 15, 3: 3 / 15},
    "B": {1: 1 / 15, 2: 4 / 15, 3: 3 / 15},
    "C": {},
}

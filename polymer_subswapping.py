"""
Generates a sequence consisting of alternating 'BXBX...', with 'X' randomly populated by
either 'A' or 'B' based on the given probabilities.

The trivial case populates 'X' with 'A' or 'B' (1/0) creates alternating sequence of
'ABAB', and so on, meaning that both appear in chains of length 1 with a probability of 0.5.
"""

import numpy as np
from utils import length_distribution, relative_frequencies, plot_distribution_and_estimation


def swap_simulation(steps, states, p_A, p_B) -> str:
    # Initialize the trajectory with the initial state
    pattern = ["B", "X"] * (steps // 2)
    if steps % 2 == 1:
        pattern.append("B")
    pattern = np.array(pattern)
    random_values = np.random.choice(states, size=steps // 2, p=[p_A, p_B])
    pattern[pattern == "X"] = random_values

    return "".join(pattern)


if __name__ == "__main__":
    # Parameters
    STEPS = int(10e6)  # Length of polymer.

    p_A = 0.64  # Probability of swapping X to
    p_B = 1 - p_A

    FIT_FUNCTION_TO_DATA = False

    # Function to fit
    # Gamma probability density function
    FIT_FUNC = lambda x, m, s: gamma.pdf(x, m, s)

    # Simulation and analysis
    STATES = ["A", "B"]
    trajectory = swap_simulation(STEPS, STATES, p_A, p_B)

    lengths = length_distribution(STATES, trajectory)
    rel_freq = relative_frequencies(STATES, lengths, total_len=len(trajectory))

    # Print absolute counts and relative frequencies
    # print("ABSOLUTE COUNT")
    # print("State | Length | Count")
    # for state in STATES:
    #    for k, v in sorted(lengths[state].items()):
    #        print(f"{state}     | {k:5} | {v}")

    print("\nRELATIVE FREQUENCIES")
    print("State | Length | Frequency")
    for state in STATES:
        for k, v in sorted(rel_freq[state].items()):
            print(f"{state}     |  {k:5} | {v:.5f}")

    if FIT_FUNCTION_TO_DATA:
        plot_distribution_and_estimation(rel_freq, FIT_FUNC)
    else:
        plot_distribution_and_estimation(rel_freq, False)

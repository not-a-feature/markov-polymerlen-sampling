import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.optimize import curve_fit
from scipy.stats import gamma, lognorm


def markov_chain_simulation(states, steps, transition_matrix, initial_state) -> str:
    # Initialize the trajectory with the initial state
    trajectory = np.empty(steps, dtype=int)

    # Create a map from states to their indices
    state_index_map = {v: idx for idx, v in enumerate(states)}

    # Start state
    current_state = state_index_map[initial_state]

    state_range = np.array(range(len(transition_matrix)))

    for i in range(0, steps):
        choice = np.random.choice(state_range, p=transition_matrix[current_state])
        trajectory[i] = choice
        current_state = choice

    return "".join(states[state_idx] for state_idx in trajectory)


def length_distribution(states, trajectory):
    # Initialize the lengths dictionary
    lengths = {k: defaultdict(int) for k in states}

    # Initialize the current state and length
    current_state = trajectory[0]
    current_length = 1

    # Iterate over the trajectory to compute the lengths
    for state in trajectory[1:]:
        if state == current_state:
            current_length += 1
        else:
            lengths[current_state][current_length] += 1
            current_length = 1
            current_state = state

    # Update the last length
    lengths[current_state][current_length] += 1

    return lengths


def relative_frequencies(states, lengths_dict, total_len):
    # Initialize the relative frequencies dictionary
    rel_freqs = {k: {} for k in states}

    # Compute the relative frequencies
    for state in states:
        rel_freqs[state] = {k: k * v / total_len for k, v in lengths_dict[state].items()}

    return rel_freqs


def stable_state_probability(transition_matrix):
    """
    Calculate the stable-state probability of a Markov chain given its transition matrix.

    :return: A 1D NumPy array representing the stable-state probabilities.
    """
    n_states = transition_matrix.shape[0]

    # Transpose the matrix and subtract the identity matrix.
    A = transition_matrix.T - np.eye(n_states)

    # Set the last row to all ones to ensure the probabilities sum to 1.
    A[-1, :] = 1

    # Set the last entry of the target vector to 1.
    b = np.zeros(n_states)
    b[-1] = 1

    # Solve the linear system Ax = b.
    stable_state = np.linalg.solve(A, b)

    return stable_state


def assert_freq(distributions, stable_state_proba, delta):
    s = sum(sum(v for v in state_dist.values()) for state_dist in distributions.values())
    assert abs(1 - s) < delta

    for idx, state in enumerate(STATES):
        stable_state = sum(distributions[state].values())
        print(f"True stable state ({state}): {stable_state_proba[idx]}")
        print(f"Simulated stable state ({state}): {stable_state:.6f}")
        assert abs(stable_state_proba[idx] - stable_state) < delta


def plot_distribution_and_estimation(distributions):
    # Create a subplots grid
    fig, axs = plt.subplots(len(distributions), figsize=(10, 5 * len(distributions)))

    max_x_data = max(max(d.keys()) for d in distributions.values()) + 1
    x_estimation = np.arange(0, max_x_data)
    # Iterate over the distributions
    for idx, (char, lengths) in enumerate(distributions.items()):
        x_data, y_data = zip(*sorted(lengths.items()))

        # Scatter plot of the true distribution
        axs[idx].scatter(x_data, y_data, label=f"True distribution ({char})")

        if FIT_FUNCTION_TO_DATA:
            # Curve fitting
            popt, _ = curve_fit(FIT_FUNC, x_data, y_data, maxfev=10000)

            print(f"Fit parameters ({char}): {popt}")
            # Plot the estimated distribution
            y_estimation = FIT_FUNC(x_estimation, *popt)
            axs[idx].plot(
                x_estimation, y_estimation, label=f"Estimated distribution ({char})", color="r"
            )
            # Print gamma distribution parameters on the plot
            axs[idx].text(0.3, 0.9, f"Fit parameters: {popt}", transform=axs[idx].transAxes)

        # Set labels, legend, and title
        axs[idx].set_xlabel("Length")
        axs[idx].set_ylabel("Frequency")
        axs[idx].set_xlim(0, max_x_data)
        axs[idx].legend()
        axs[idx].set_title(f"Distribution and Estimation for {char}")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Parameters
    STEPS = int(10e5)  # Minimum length of Markov Chain.
    DELTA = 0.01  # Maximum allowed delta between integral of state probability and 1.

    STATES = ["A", "B"]
    TRANSITION_MATRIX = np.array(
        [
            [0.32, 0.68],
            [0.32, 0.68],
        ]
    )

    FIT_FUNCTION_TO_DATA = False

    # Function to fit
    # Gamma probability density function
    FIT_FUNC = lambda x, m, s: gamma.pdf(x, m, s)

    # Example with 3 states
    """"
    STATES = ["A", "B", "C"]
    TRANSITION_MATRIX = np.matrix(
        [
            [0.7, 0.2, 0.1],
            [0.6, 0.3, 0.1],
            [0.1, 0.1, 0.8],
        ]
    )
    """

    # Simulation and analysis
    stable_state_proba = stable_state_probability(TRANSITION_MATRIX)
    initial_state = STATES[np.argmax(stable_state_proba)]

    trajectory = markov_chain_simulation(STATES, STEPS, TRANSITION_MATRIX, initial_state)

    lengths = length_distribution(STATES, trajectory)
    rel_freq = relative_frequencies(STATES, lengths, total_len=len(trajectory))

    assert_freq(rel_freq, stable_state_proba, DELTA)

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

    plot_distribution_and_estimation(rel_freq)

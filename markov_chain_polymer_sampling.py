import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.optimize import curve_fit
from scipy.stats import gamma, poisson


def markov_chain_simulation(n_steps: int, transition_matrix: list, initial_state="A") -> str:
    # Create a map from states to their indices
    state_index_map = {v: k for k, v in enumerate(STATES)}

    # Convert the transition matrix to a NumPy array
    transition_matrix = np.array(transition_matrix)

    # Initialize the trajectory with the initial state
    trajectory = np.empty(n_steps, dtype=object)
    trajectory[0] = initial_state

    # Generate the random choices for all steps at once
    random_choices = np.random.choice(
        len(STATES), size=n_steps - 1, p=transition_matrix[state_index_map[initial_state]]
    )

    # Simulate the Markov chain
    for i in range(1, n_steps):
        trajectory[i] = STATES[random_choices[i - 1]]

    return "".join(trajectory)


def length_distribution(trajectory):
    # Initialize the lengths dictionary
    lengths = {k: defaultdict(int) for k in STATES}

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


def relative_frequencies(lengths_dict, total_len):
    # Initialize the relative frequencies dictionary
    rel_freqs = {k: {} for k in STATES}

    # Compute the relative frequencies
    for state in STATES:
        rel_freqs[state] = {k: k * v / total_len for k, v in lengths_dict[state].items()}

    return rel_freqs


def fit_func(x, p, s):
    # Return function to fit

    # Gamma probability density function
    return gamma.pdf(x, p, s)


def plot_distribution_and_estimation(distributions):
    # Create a subplots grid
    fig, axs = plt.subplots(len(distributions), figsize=(10, 5 * len(distributions)))

    # Iterate over the distributions
    for idx, (char, lengths) in enumerate(distributions.items()):
        x_data, y_data = zip(*sorted(lengths.items()))

        # Scatter plot of the true distribution
        axs[idx].scatter(x_data, y_data, label=f"True distribution ({char})")

        # Curve fitting
        popt, _ = curve_fit(fit_func, x_data, y_data)

        print(f"Fit parameters ({char}): {popt}")
        # Plot the estimated distribution
        x_estimation = np.arange(min(x_data), max(x_data) + 1)
        y_estimation = fit_func(x_estimation, *popt)
        axs[idx].plot(
            x_estimation, y_estimation, label=f"Estimated distribution ({char})", color="r"
        )
        # Print gamma distribution parameters on the plot
        axs[idx].text(0.3, 0.9, f"Fit parameters: {popt}", transform=axs[idx].transAxes)

        # Set labels, legend, and title
        axs[idx].set_xlabel("Length")
        axs[idx].set_ylabel("Frequency")
        axs[idx].legend()
        axs[idx].set_title(f"Distribution and Estimation for {char}")

    plt.tight_layout()
    plt.show()


# Parameters
n_steps = int(10e5)

STATES = ["A", "B"]
transition_matrix = np.matrix(
    [
        [0.9, 0.1],
        [0.6, 0.4],
    ]
)

"""
# Example with 3 states

STATES = ["A", "B", "C"]
transition_matrix = np.matrix(
    [
        [0.8, 0.1, 0.1],
        [0.6, 0.3, 0.1],
        [0.1, 0.1, 0.8],
    ]
)
"""

# Simulation and analysis
trajectory = markov_chain_simulation(n_steps, transition_matrix)
lengths = length_distribution(trajectory)
rel_freq = relative_frequencies(lengths, total_len=len(trajectory))

# Print absolute counts and relative frequencies
print("ABSOLUTE COUNT")
print("State | Length | Count")
for state in STATES:
    for k, v in sorted(lengths[state].items()):
        print(f"{state}     | {k:5} | {v}")

print("\nRELATIVE FREQUENCIES")
print("State | Length | Frequency")
for state in STATES:
    for k, v in sorted(rel_freq[state].items()):
        print(f"{state}     | {k:5} | {v:.5f}")

plot_distribution_and_estimation(rel_freq)

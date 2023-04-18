"""Utility function to calculate the distribution and frequencies."""
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.optimize import curve_fit
from scipy.stats import gamma, lognorm


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


def plot_distribution_and_estimation(distributions, fit_func=False):
    # Create a subplots grid
    fig, axs = plt.subplots(len(distributions), figsize=(10, 5 * len(distributions)))

    max_x_data = max(max(d.keys()) for d in distributions.values()) + 1
    x_estimation = np.arange(0, max_x_data)
    # Iterate over the distributions
    for idx, (char, lengths) in enumerate(distributions.items()):
        x_data, y_data = zip(*sorted(lengths.items()))

        # Scatter plot of the true distribution
        axs[idx].scatter(x_data, y_data, label=f"True distribution ({char})")

        if fit_func:
            # Curve fitting
            popt, _ = curve_fit(FIT_FUNC, x_data, y_data, maxfev=10000)

            print(f"Fit parameters ({char}): {popt}")
            # Plot the estimated distribution
            y_estimation = fit_func(x_estimation, *popt)
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

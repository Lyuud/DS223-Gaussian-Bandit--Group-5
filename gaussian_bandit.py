"""Gaussian Thompson sampling with known observation variance."""
import numpy as np

def run_gaussian_bandit(true_means, sigma, rounds, seed):
    means = np.asarray(true_means, dtype=float)
    if means.shape != (3,) or not np.all(np.isfinite(means)):
        raise ValueError('Provide three finite means')
    if not np.isfinite(sigma) or sigma <= 0 or int(rounds) != rounds or rounds <= 0:
        raise ValueError('Sigma and integer rounds must be positive')
    rng = np.random.default_rng(seed)
    posterior_means = np.zeros(3)
    posterior_variances = np.full(3, 25.0)
    counts = np.zeros(3, dtype=int)
    rewards, chosen, trajectories = [], [], []
    for _ in range(int(rounds)):
        samples = rng.normal(posterior_means, np.sqrt(posterior_variances))
        arm = int(np.argmax(samples))
        reward = float(rng.normal(means[arm], sigma))
        precision = 1.0 / posterior_variances[arm]
        noise_precision = 1.0 / (sigma ** 2)
        new_variance = 1.0 / (precision + noise_precision)
        posterior_means[arm] = new_variance * (precision * posterior_means[arm] + noise_precision * reward)
        posterior_variances[arm] = new_variance
        counts[arm] += 1
        rewards.append(reward)
        chosen.append(arm + 1)
        trajectories.append(posterior_means.tolist())
    return dict(true_means=means.tolist(), posterior_means=posterior_means.tolist(),
                posterior_variances=posterior_variances.tolist(), counts=counts.tolist(),
                total_reward=float(sum(rewards)), recommendation=int(np.argmax(posterior_means))+1,
                rewards=rewards, chosen=chosen, trajectories=trajectories,
                seed=seed, sigma=sigma, rounds=int(rounds))

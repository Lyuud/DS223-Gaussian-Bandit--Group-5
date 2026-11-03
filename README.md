# Gaussian Thompson Sampling

**Course:** DS223 — Marketing Analytics  
**Project:** Gaussian Bandit Assignment  
**Group:** 5

### Team Members
- Lyudmila
- Elen
- Nane
- Hamlet

## 1. Project Overview

This project extends the original Bayesian A/B Testing Streamlit application by adding **Gaussian Thompson Sampling**.

The original application uses Beta-Bernoulli Thompson Sampling for binary rewards (click or no click).

Our extension uses Gaussian Thompson Sampling for continuous rewards, such as revenue or customer engagement.

The goal is to identify the best-performing alternative while balancing exploration and exploitation.

## 2. Original Application

The original application uses Beta-Bernoulli Thompson Sampling.

- Each alternative has a Beta distribution representing its unknown click probability.
- The algorithm samples from these distributions and selects alternatives based on the sampled values.
- A click gives a reward of 1, while no click gives a reward of 0.
- After feedback, the Beta posterior is updated by increasing alpha for a click or beta for a non-click.
- Streamlit session state stores the carousel data and connects the Create Carousels, Interaction, and Analytics pages.

## 3. Gaussian Thompson Sampling

We added Gaussian Thompson Sampling to work with continuous rewards.

The algorithm uses three alternatives. Each alternative has an unknown true average reward.

Initially, all alternatives have the same prior distribution:

**Normal(0, 25)** — mean 0 and variance 25.

For each round:

1. Sample a possible mean from each alternative's current posterior distribution.
2. Select the alternative with the highest sampled value.
3. Generate a continuous reward using the selected alternative's true mean and known noise level.
4. Update only the selected alternative's posterior using the observed reward.

The posterior is updated using the Normal-Normal Bayesian update, where the new precision equals the previous precision plus the observation precision, 1/σ². The posterior mean is updated as a precision-weighted average of the previous estimate and the observed reward.

The true means are used only by the simulator to generate rewards. The algorithm does not know them and must learn from observations.

## 4. Implementation

We preserved the original application and made three main changes:

- **`gaussian_bandit.py`** — Added the Gaussian Thompson Sampling algorithm, reward generation, posterior updates, and result calculations.
- **`pages/Gaussian_Bandit.py`** — Added a Streamlit page with inputs, results, tables, and charts.
- **`main.py`** — Added the Gaussian Bandit page to the application's navigation.

The application allows users to enter three true means, known noise standard deviation (σ), number of rounds, and random seed.

It displays the posterior estimated means, selection counts, total observed reward, and final recommended alternative.

## 5. Experiments and Results

We conducted three experiments using true means **[1, 2, 3]**, the same Normal(0, 25) prior, and random seed **42**.

| Experiment | Rounds | σ | Estimated Means (A1, A2, A3) | Selection Counts (A1, A2, A3) | Recommendation |
|---|---:|---:|---|---|---|
| 1 — Short run | 100 | 1 | 1.570, 1.729, 2.995 | 4, 2, 94 | Alternative 3 |
| 2 — Longer run | 1,000 | 1 | 1.306, 2.142, 2.979 | 5, 13, 982 | Alternative 3 |
| 3 — Higher noise | 1,000 | 3 | 1.154, 1.484, 2.954 | 21, 23, 956 | Alternative 3 |

### Observations

**Experiment 1:** Even with 100 rounds, the algorithm identified Alternative 3 as the best option.

**Experiment 2:** With 1,000 rounds, the algorithm selected Alternative 3 more frequently, showing stronger exploitation of the best alternative.

**Experiment 3:** Increasing the noise made learning more difficult. The algorithm explored Alternatives 1 and 2 more frequently but still recommended Alternative 3.

In all three experiments, the selection counts summed to the number of rounds. Repeating the experiments with the same settings and seed produced identical results.

## 6. How to Run the Application

**Requirements:** Python 3.12 and uv.

Clone the repository:

```bash
git clone https://github.com/Lyuud/DS223-Gaussian-Bandit--Group-5.git
cd DS223-Gaussian-Bandit--Group-5
```

Install dependencies:

```bash
uv sync --locked
```

Run the application:

```bash
uv run streamlit run main.py
```

Open the local Streamlit URL displayed in the terminal.

## 7. Project Links

**GitHub Repository:**  
https://github.com/Lyuud/DS223-Gaussian-Bandit--Group-5

**Streamlit Application:**  
[Add your deployed Streamlit URL here]

**Original Application:**  
https://carousel-mvp.streamlit.app/

**Original Source Repository:**  
https://github.com/hovhannisyan91/DS223_Bayesian_AB_Testing

## 8. Conclusion

We successfully extended the original Bayesian A/B Testing application with Gaussian Thompson Sampling.

Our experiments demonstrated that the algorithm can learn which alternative provides the highest average reward without knowing the true means in advance.

More observations helped the algorithm focus on the best alternative, while higher noise increased uncertainty and exploration.

This method can be useful for marketing decisions involving continuous rewards.

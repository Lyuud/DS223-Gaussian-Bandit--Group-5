# Gaussian Thompson Sampling — DS223 Marketing Analytics

A Streamlit frontend extending the Bayesian carousel application with **Gaussian Thompson Sampling** for continuous rewards. The original Create Carousels, Interaction, and Analytics pages remain available.

## Attribution

Adapted from the course starter repository: https://github.com/hovhannisyan91/DS223_Group_Project (frontend code also supplied from https://github.com/hovhannisyan91/DS223_Bayesian_AB_Testing). The original project and its authors retain credit for the starter application.

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --locked
uv run streamlit run main.py
```

Alternatively, using Python 3.12: `python -m pip install -e .` followed by `streamlit run main.py`.

## Original model: Beta–Bernoulli Thompson Sampling

The original app creates a carousel with multiple alternatives. On each interaction, it draws a possible click-through rate for each alternative from its current Beta(alpha, beta) posterior and uses these draws to choose which alternatives to display. The user submits click/no-click feedback (Bernoulli rewards 1 or 0). A click increases alpha by one; a no-click increases beta by one for the corresponding displayed alternative. This balances exploration of uncertain alternatives and exploitation of promising alternatives. `st.session_state` holds the carousel dictionary, active carousel identifier, alternatives, and interaction history, allowing the creation, interaction, and analytics pages to share the same state within a user session. The analytics page displays posterior means and performance metrics. State is session-local, not a persistent backend database.

## Gaussian adaptation

For each of three alternatives, the unknown reward mean has prior `mu_i ~ Normal(0, 25)` (variance 25, standard deviation 5). The simulator generates a real-valued reward `R | A=i ~ Normal(true_mu_i, sigma^2)` using the **known** shared noise sigma. Rewards may be negative and are never converted to clicks.

At each round:
1. Draw one possible mean from each arm's current Normal posterior.
2. Select the arm with the highest sampled mean (not the highest true mean).
3. Generate a continuous Normal reward for that arm using its true mean and known sigma.
4. Update **only the selected arm** using the observed reward.

For an existing posterior `Normal(m, v)` and reward `r`, with known observation variance `sigma^2`, the conjugate update is:

```text
v_new = 1 / (1/v + 1/sigma^2)
m_new = v_new * (m/v + r/sigma^2)
```

The final recommendation is the alternative with the largest **posterior mean**. The true means are used only for simulation and evaluation, never for action selection. The seed makes the simulation reproducible.

## Experiments

All runs use **seed 42**, true means `[1, 2, 3]`, and prior `Normal(0, 25)`.

| Experiment | Known sigma | Rounds | Posterior estimated means (A1, A2, A3) | Selection counts (A1, A2, A3) | Total reward | Recommendation |
|---|---:|---:|---|---|---:|---|
| Short run | 1 | 100 | 1.5702, 1.7287, 2.9950 | 4, 2, 94 | 291.518 | Alternative 3 |
| Longer run | 1 | 1000 | 1.3062, 2.1421, 2.9792 | 5, 13, 982 | 2960.201 | Alternative 3 |
| Higher noise | 3 | 1000 | 1.1537, 1.4838, 2.9542 | 21, 23, 956 | 2884.604 | Alternative 3 |

### Discussion

In all runs, Alternative 3 is correctly recommended, because its true mean of 3 is the largest. With 1000 rounds instead of 100 at sigma=1, the model concentrates even more strongly on Alternative 3 (982 selections versus 94). Its estimated mean approaches the true mean, while estimates for rarely selected arms can remain noisy because those arms receive little data. Under higher noise, individual observations carry less information (observation precision decreases from 1 to 1/9), so posterior uncertainty shrinks more slowly. The model consequently explores the weaker alternatives more often (44 combined selections rather than 18 in the lower-noise long run). An initially surprising result is that the higher-noise run gives Alternative 1 a closer mean estimate than the lower-noise run: it was sampled more often. This is one random seeded run, not a universal guarantee.

**Validation:** identical inputs and seed reproduced identical results; selection counts were 100, 1000, and 1000 respectively, matching the number of rounds. Negative rewards remain permitted.

## Project links

- Original frontend: https://github.com/hovhannisyan91/DS223_Bayesian_AB_Testing/tree/main/app/front
- Original deployed app: https://carousel-mvp.streamlit.app/
- **Your GitHub repository:** ADD YOUR NEW REPOSITORY URL HERE
- **Your Streamlit deployment:** ADD YOUR NEW DEPLOYED URL HERE

## Deployment

1. Create a new public GitHub repository and upload **the contents of this folder into the repository root**, including `main.py`, `pages/`, `bandit_utils.py`, `gaussian_bandit.py`, `pyproject.toml`, and `uv.lock`.
2. Open https://share.streamlit.io/ and create a new app using your repository, branch `main`, and entrypoint `main.py`.
3. Configure Python 3.12 in Streamlit Community Cloud advanced settings, deploy, and test all four pages.
4. Replace the two placeholders in the project links above with your actual URLs.

## Presentation outline (3–5 minutes)

1. **Motivation:** The original application optimizes binary clicks using Beta–Bernoulli Thompson sampling; our new page handles continuous-valued rewards.
2. **Algorithm:** Each arm starts with Normal(0,25). Sample from the three posteriors, pick the largest sample, observe one Normal reward, and update only that arm using known sigma.
3. **Demo:** Show the original pages, then open Gaussian Bandit. Set true means 1,2,3, sigma 1, rounds 100, seed 42; click Run and explain the result table and charts.
4. **Experiments:** Compare 100 and 1000 rounds, then increase sigma from 1 to 3. Discuss convergence, exploration, and noise.
5. **Conclusion:** Alternative 3 wins all three experiments; the method balances exploration and exploitation using posterior uncertainty.

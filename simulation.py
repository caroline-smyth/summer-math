import numpy as np

def simulate_branching_gbm(
    n0: int,
    T: float,
    dt: float,
    mu: float,
    sigma: float,
    beta: float,
    x0: float,
    X_init: np.ndarray = None,
    seed: int = None,
) -> np.ndarray:
    """
    simulate n0 independent GBMs with drift mu, vol sigma, and 
    Poisson(β) resets to x0, over [0,T] with time‐step dt.
    returns the vector X of length n0 at time T.
    """
    if seed is not None:
        np.random.seed(seed)
    steps = int(np.ceil(T / dt))
    # initialize
    if X_init is None:
        X = np.full(n0, x0, dtype=float)
    else:
        X = X_init.astype(float).copy()
    for _ in range(steps):
        # simulate Brownian increment
        dW = np.random.randn(n0) * np.sqrt(dt)
        # Euler–Maruyama update
        X += mu * X * dt + sigma * X * dW
        # Poisson‐reset with probability beta*dt
        jumps = np.random.rand(n0) < beta * dt
        X[jumps] = x0
    return X

def empirical_cdf(X: np.ndarray):
    """
    given array X of length n, returns sorted X and F = [1/n, 2/n, …, 1].
    so F[i] = P_n(X <= X_sorted[i]).
    """
    n = len(X)
    X_sorted = np.sort(X)
    F = np.arange(1, n + 1) / n
    return X_sorted, F

def capital_distribution(X: np.ndarray):
    """
    given array X of length n, sorts descending, then
      G_vals[j] = (sum_{k=0..j} exp(X_desc[k])) / (sum_{k=0..n-1} exp(X_desc[k])),
    at q_vals = [(1/n), (2/n), …, 1].
    """
    n = len(X)
    X_desc = np.sort(X)[::-1]
    weights = np.exp(X_desc)
    cumw = np.cumsum(weights)
    G = cumw / cumw[-1]
    q = np.arange(1, n + 1) / n
    return q, G

def compute_ranks_and_positions(X: np.ndarray):
    """
    given X (array of length n of log-sizes or sizes), return:
      • ranks: array of ints in {1,…,n}, where ranks[i]=1 for the largest X[i], …, ranks[i]=n for the smallest.
      • positions: array of floats in (0,1], where positions[i] = 1 − (ranks[i]-1)/n,
        matching the Banner–Ghomrasni form b(1−(i−1)/n), σ(1−(i−1)/n) :contentReference[oaicite:0]{index=0}.
    """
    n = X.shape[0]
    # sort indices in descending order
    order_desc = np.argsort(X)[::-1]
    # invert that permutation to get rank of each original index
    ranks = np.empty(n, dtype=int)
    ranks[order_desc] = np.arange(1, n+1)
    # compute banner-style position = 1 − (rank−1)/n
    positions = 1.0 - (ranks - 1) / n
    return ranks, positions

def simulate_ranked_branching(
    n0: int,
    T: float,
    dt: float,
    b_func,
    sigma_func,
    beta: float,
    x0: float,
    seed: int = None,
):
    """
    Simulate n0 log-sizes X_i under rank-based drift/volatility and Poisson resets.
     - b_func(p): drift when a particle's banner-position = p in (0,1]
     - sigma_func(p): vol  when position = p
     - beta: reset-rate; x0: reset value
    """
    steps = int(np.ceil(T / dt))
    rng   = np.random.default_rng(seed)
    X     = np.full(n0, np.log(x0), dtype=float)

    for _ in range(steps):
        # 1) compute current ranks & banner-positions
        ranks, pos = compute_ranks_and_positions(X)
        #    pos[i] = 1 - (ranks[i]-1)/n0 ∈ (0,1]

        # 2) get drift & vol for each particle
        b_vals     = b_func(pos)        # shape (n0,)
        sigma_vals = sigma_func(pos)    # shape (n0,)

        # 3) brownian increment
        dW = rng.standard_normal(n0) * np.sqrt(dt)

        # 4) Euler–Maruyama update
        X += b_vals * dt + sigma_vals * dW

        # 5) random Poisson resets to x0
        jumps = rng.random(n0) < beta * dt
        X[jumps] = np.log(x0)

    return X


if __name__ == "__main__":
    # parameters
    n0    = 200      # number of particles
    T     = 1.0      # total time
    dt    = 1e-3     # time‐step
    mu    = 0.05     # drift
    sigma = 0.2      # volatility
    beta  = 0.1      # reset intensity
    x0    = 1.0      # reset value
    # run simulation
    X_T = simulate_branching_gbm(n0, T, dt, mu, sigma, beta, x0, seed=42)
    # compute F_t^n and G_t^n at t=T
    X_sorted, F_vals = empirical_cdf(X_T)
    q_vals,   G_vals = capital_distribution(X_T)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.step(X_sorted, F_vals, where="post")
    plt.xlabel("x")
    plt.ylabel(r"$F_T^{n_0}(x)$")
    plt.title("Empirical CDF at time T")
    plt.grid(True)

    plt.figure()
    plt.step(q_vals, G_vals, where="post")
    plt.xlabel("q")
    plt.ylabel(r"$G_T^{n_0}(q)$")
    plt.title("Capital‐distribution $G_T^{n_0}(q)$")
    plt.grid(True)
    plt.show()

    b_cont   = lambda p: 0.05 * (1 - p)        # drift smaller if high-ranked
    sigma_cont = lambda p: 0.2 * np.sqrt(p)    # vol increasing with rank

    X_final = simulate_ranked_branching(
        n0=200, T=1.0, dt=1e-3,
        b_func=b_cont,
        sigma_func=sigma_cont,
        beta=0.1,
        x0=1.0,
        seed=42,
    )

    # then you can compute F_T and G_T as before:
    Xs, F = empirical_cdf(X_final)
    q, G  = capital_distribution(X_final)

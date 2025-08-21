import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import LogExpectedImprovement as ExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.models.transforms import Normalize, Standardize
import math

import numpy as np
from tmp_test import PBFSimulation
from tqdm import tqdm

device = torch.device("cpu")

theta_bounds = torch.tensor([
    [500, 15000],  # gravity_y
    [1.3, 2.5],  # h_factor (вместо h)
    [0.3, 8.0],  # mass_coeff
    [0.01, 1.0],  # nu_xsph
], dtype=torch.double)

# c: контекст (корень из числа частиц, W, H)
c_fixed = torch.tensor([16, 100, 15, 10], dtype=torch.double)

d_theta = theta_bounds.shape[0]


# ---------- helpers: метрики ----------
def _pairwise_nn_dist(pos: np.ndarray) -> np.ndarray:
    """d_i = расстояние до ближайшего соседа для каждой частицы. O(N^2) — ок для N~200..1000."""
    N = pos.shape[0]
    diff = pos[None, :, :] - pos[:, None, :]
    D2 = np.sum(diff * diff, axis=-1)  # (N,N)
    np.fill_diagonal(D2, np.inf)
    dmin = np.sqrt(np.min(D2, axis=1))
    return dmin.astype(np.float32)


def _no_overlap_penalty(pos: np.ndarray, d0: float) -> float:
    """Штраф за пары с расстоянием < d0 (мягкий квадрат)."""
    N = pos.shape[0]
    diff = pos[None, :, :] - pos[:, None, :]
    D2 = np.sum(diff * diff, axis=-1)
    mask = np.triu(np.ones((N, N), dtype=bool), 1)  # i<j
    Dij = np.sqrt(D2[mask])
    overlap = np.clip(d0 - Dij, 0.0, None)
    return float((overlap ** 2).mean()) if overlap.size else 0.0


def _height_stats(pos: np.ndarray, ymin_clip=None, ymax_clip=None, trim=0.1):
    """Усечённые среднее/стд по высоте (устойчиво к брызгам)."""
    y = pos[:, 1]
    if ymin_clip is not None:
        y = y[y >= ymin_clip]
    if ymax_clip is not None:
        y = y[y <= ymax_clip]
    if y.size == 0:
        return 0.0, 0.0
    y_sorted = np.sort(y)
    k = int(trim * y_sorted.size)
    y_trim = y_sorted[k:-k] if k * 2 < y_sorted.size else y_sorted
    return float(y_trim.mean()), float(y_trim.std())


def _kinetic_energy(vel: np.ndarray) -> float:
    """Средняя удельная кинетическая энергия (без массы)."""
    return float(0.5 * np.mean(np.sum(vel * vel, axis=1)))


# ----- симуляция + метрики -> скалярная цель -----
def run_simulation_and_loss(theta: torch.Tensor, c: torch.Tensor) -> float:
    """
    theta: [gravity_y, h_target, mass_coeff, nu_xsph, rest_density]
    c:     [sqrtN, iterations, W, H]  (W,H трактуем как целые)
    Возвращает скалярную цель L (меньше — лучше).
    """

    gravity_y = float(theta[0].item())
    h_factor = float(theta[1].item())
    mass_coeff = float(theta[2].item())
    nu_xsph = float(theta[3].item())

    sqrtN = int(round(float(c[0].item())))
    iterations = int(round(float(c[1].item())))
    W = float(int(round(float(c[2].item()))))
    H = float(int(round(float(c[3].item()))))
    N = int(sqrtN * sqrtN)

    h_dx_ratio = max(h_factor, 1e-3)

    sim = PBFSimulation(
        bounds=(0.0, W, 0.0, H),
        N=N,
        gravity=(0.0, -abs(gravity_y)),
        rest_density=1000,
        h_dx_ratio=h_dx_ratio,
        iterations=iterations,
        nu_xsph=nu_xsph,
        mass_coeff=mass_coeff,
        scorr_k=0.001,
        scorr_n=4,
        steps=600,
        seed=42
    )
    hist = sim.run(record_every=10, dt=0.008)
    last = hist[-1]
    pos = last["pos"]
    vel = last["vel"]

    dx = np.sqrt((W * H) / float(N))

    k = 1.10
    s_target = k * dx
    s_lower = 0.95 * s_target
    s_upper = 1.05 * s_target
    margin = 0.8 * dx
    r_n = 1.2 * dx

    d_nn = _pairwise_nn_dist(pos)
    d_bar = float(np.mean(d_nn))
    d_cv = float(np.std(d_nn) / max(d_bar, 1e-6))

    too_close = np.clip(s_lower - d_nn, 0.0, None)
    too_far = np.clip(d_nn - s_upper, 0.0, None)
    J_small = float(np.mean(too_close ** 2) / (dx * dx))
    J_large = float(np.mean(too_far ** 2) / (dx * dx))

    diff = pos[None, :, :] - pos[:, None, :]
    D2 = np.sum(diff * diff, axis=-1);
    np.fill_diagonal(D2, np.inf)
    k_cnt = np.sum(D2 < r_n * r_n, axis=1)
    J_iso = float(np.mean(np.clip(4 - k_cnt, 0, None) ** 2))  # хотим >=4 соседей

    left = np.mean(pos[:, 0] < margin)
    right = np.mean(pos[:, 0] > W - margin)
    bottom = np.mean(pos[:, 1] < margin)
    top = np.mean(pos[:, 1] > H - margin)
    J_wallband = (left + right + bottom + top) / 4.0

    d0 = 0.9 * dx
    J_over = _no_overlap_penalty(pos, d0=d0) / (dx * dx)

    g = gravity_y
    nu = nu_xsph
    J_param = 0.0
    J_param += max(0.0, (1500.0 - g) / 1500.0) ** 2
    J_param += max(0.0, (nu - 0.6) / 0.6) ** 2

    y_mean, y_std = _height_stats(pos, ymin_clip=0.0, ymax_clip=H, trim=0.1)
    J_flat = y_std / max(y_mean, 1e-6)  # ровность

    J_KE = _kinetic_energy(vel)

    y_med = float(np.median(pos[:, 1]))
    J_fill = float(np.clip(0.2 * H - y_med, 0.0, None) ** 2) / (H * H)

    L = (
            14.0 * J_small +
            14.0 * J_large +
            10.0 * J_over +
            5.0 * J_iso +
            4.0 * d_cv +
            2.0 * J_flat +
            2.0 * J_KE +
            4.0 * J_wallband +
            2.0 * J_fill
    )

    L += 2.0 * J_param
    return float(L)


# ---------- утилиты нормировки ----------
def to_unit(theta_phys: torch.Tensor) -> torch.Tensor:
    """Перевод из физических единиц в [0,1]^d по theta_bounds с клэмпом."""
    lb, ub = theta_bounds[:, 0], theta_bounds[:, 1]
    x = (theta_phys - lb) / (ub - lb + 1e-12)
    return torch.clamp(x, 0.0, 1.0)


def to_phys(theta_unit: torch.Tensor) -> torch.Tensor:
    """Обратное преобразование: [0,1]^d -> физические единицы."""
    lb, ub = theta_bounds[:, 0], theta_bounds[:, 1]  # (d,)
    return lb + theta_unit * (ub - lb)


# ---------- надёжный вызов симулятора ----------
def safe_eval(theta_phys: torch.Tensor,
              c: torch.Tensor,
              big_penalty: float = 1e9) -> float:
    """Вызывает твою симуляцию, ловит NaN/исключения и возвращает большой штраф."""
    try:
        val = run_simulation_and_loss(theta_phys, c)  # твоя функция
        if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
            return float(big_penalty)
        if isinstance(val, (np.ndarray, torch.Tensor)) and (np.isnan(val).any() or np.isinf(val).any()):
            return float(big_penalty)
        return float(val)
    except Exception as e:
        return float(big_penalty)


user_seed_thetas_phys = [
    [1000.0, 2.0, 1.0, 0.4],
]
extra_random_inits = 7

# ---------- формируем начальный датасет ----------
X_theta_unit_list = []
X_theta_phys_list = []
Y_list = []

for row in user_seed_thetas_phys:
    theta_phys = torch.tensor(row, dtype=torch.double, device=device)
    theta_unit = to_unit(theta_phys)
    L = safe_eval(theta_phys, c_fixed)
    X_theta_unit_list.append(theta_unit)
    X_theta_phys_list.append(theta_phys)
    Y_list.append([L])

if extra_random_inits > 0:
    rand_unit = torch.rand(extra_random_inits, d_theta, dtype=torch.double, device=device)
    rand_phys = to_phys(rand_unit)
    for i in range(extra_random_inits):
        L = safe_eval(rand_phys[i], c_fixed)
        X_theta_unit_list.append(rand_unit[i])
        X_theta_phys_list.append(rand_phys[i])
        Y_list.append([L])

X_theta = torch.stack(X_theta_unit_list, dim=0)  # (n0, d)
X_theta_phys = torch.stack(X_theta_phys_list, dim=0)  # (n0, d)
Y = torch.tensor(Y_list, dtype=torch.double, device=device)  # (n0, 1)

# ----- цикл BO -----
N_ITER = 20
Y_obj = -Y.clone()
y_noise = 1e-6 * torch.ones_like(Y)

for t in tqdm(range(N_ITER), desc="BO iterations"):
    model = SingleTaskGP(
        train_X=X_theta,
        train_Y=Y_obj,
        train_Yvar=y_noise,
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    EI = ExpectedImprovement(model=model, best_f=Y_obj.max().item())  # max по -L

    bounds01 = torch.stack([
        torch.zeros(d_theta, dtype=torch.double, device=device),
        torch.ones(d_theta, dtype=torch.double, device=device),
    ])

    cand_norm, _ = optimize_acqf(
        EI, bounds=bounds01, q=1, num_restarts=10, raw_samples=128,
    )
    theta_next_norm = cand_norm.squeeze(0)
    theta_next = to_phys(theta_next_norm)

    L_next = run_simulation_and_loss(theta_next, c_fixed)

    X_theta = torch.vstack([X_theta, theta_next_norm.unsqueeze(0)])
    X_theta_phys = torch.vstack([X_theta_phys, theta_next.unsqueeze(0)])
    Y = torch.vstack([Y, torch.tensor([[L_next]], dtype=torch.double)])
    Y_obj = torch.vstack([Y_obj, -torch.tensor([[L_next]], dtype=torch.double)])
    y_noise = torch.vstack([y_noise, torch.tensor([[1e-6]], dtype=torch.double)])

    print(f"[iter {t + 1}/{N_ITER}] loss={L_next:.4f}, best={Y.min().item():.4f}")

# Лучшие θ:
best_idx = torch.argmin(Y)
theta_best = X_theta_phys[best_idx]
print("Best θ found:", theta_best.tolist(), "with loss", Y[best_idx].item())

import numpy as np

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:
    def njit(*args, **kwargs):
        def wrap(f):
            return f
        return wrap
    NUMBA_AVAILABLE = False


#      KERNEL HELPERS
@njit(cache=True)
def poly6_w(h: np.float32, poly6_coef: np.float32, r2: np.float32) -> np.float32:
    """W_poly6 по r^2 (3D-коэффициент, как в классике PBF)."""
    if r2 >= h*h:
        return np.float32(0.0)
    t = h*h - r2
    return poly6_coef * t * t * t  # t^3

@njit(cache=True)
def spiky_grad(h: np.float32, spiky_grad_coef: np.float32, rx: np.float32, ry: np.float32) -> (np.float32, np.float32):
    """∇W_spiky (3D-коэффициент), возвращаем 2D-вектор (gx, gy)."""
    r2 = rx*rx + ry*ry
    if r2 == 0.0 or r2 >= h*h:
        return np.float32(0.0), np.float32(0.0)
    r = np.sqrt(r2)
    mag = spiky_grad_coef * (h - r)*(h - r) / (r + 1e-12)  # -45/(π h^6) * (h-r)^2 * r̂
    return mag * rx, mag * ry


#   UNIFORM GRID HELPERS
@njit(cache=True)
def build_grid(pos: np.ndarray, xmin: np.float32, ymin: np.float32,
               inv_h: np.float32, nx: int, ny: int):
    """
    Строим сетку по позициям:
    - cell_id для каждой частицы
    - индексы, отсортированные по cell_id
    - массивы start/end для каждой ячейки
    """
    N = pos.shape[0]
    cell_id = np.empty(N, np.int32)

    for i in range(N):
        cx = int(np.floor((pos[i, 0] - xmin) * inv_h))
        cy = int(np.floor((pos[i, 1] - ymin) * inv_h))
        if cx < 0: cx = 0
        if cy < 0: cy = 0
        if cx >= nx: cx = nx - 1
        if cy >= ny: cy = ny - 1
        cell_id[i] = cy * nx + cx

    order = np.argsort(cell_id)
    cell_sorted = cell_id[order]

    n_cells = nx * ny
    start = np.full(n_cells, -1, np.int32)
    end   = np.full(n_cells, -1, np.int32)
    if N > 0:
        cur_id = cell_sorted[0]
        start[cur_id] = 0
        for k in range(1, N):
            cid = cell_sorted[k]
            if cid != cur_id:
                end[cur_id] = k
                if start[cid] == -1:
                    start[cid] = k
                cur_id = cid
        end[cur_id] = N

    return order, cell_sorted, start, end


@njit(cache=True)
def neighbor_ranges(xi: np.ndarray,
                    xmin: np.float32, ymin: np.float32,
                    inv_h: np.float32, nx: int, ny: int,
                    start: np.ndarray, end: np.ndarray):
    """
    Для частицы в xi возвращаем список (до 9) диапазонов [s:e) отсортированных индексов,
    соответствующих соседним ячейкам.
    """
    cx = int(np.floor((xi[0] - xmin) * inv_h))
    cy = int(np.floor((xi[1] - ymin) * inv_h))
    if cx < 0: cx = 0
    if cy < 0: cy = 0
    if cx >= nx: cx = nx - 1
    if cy >= ny: cy = ny - 1

    ranges = []
    for dx in (-1, 0, 1):
        ncx = cx + dx
        if ncx < 0 or ncx >= nx:
            continue
        for dy in (-1, 0, 1):
            ncy = cy + dy
            if ncy < 0 or ncy >= ny:
                continue
            cid = ncy * nx + ncx
            s = start[cid]
            if s == -1:
                continue
            e = end[cid]
            if e > s:
                ranges.append((s, e))
    return ranges


#       PBF KERNELS
@njit(cache=True)
def pbf_compute_density(N: int, pos: np.ndarray,
                        mass: np.float32, h: np.float32, poly6_coef: np.float32,
                        xmin: np.float32, ymin: np.float32, inv_h: np.float32, nx: int, ny: int,
                        order: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """ρ_i = m * sum_j W_poly6(|x_i - x_j|)."""
    rho = np.zeros(N, np.float32)
    for i in range(N):
        xi = pos[i]
        s = np.float32(0.0)
        ranges = neighbor_ranges(xi, xmin, ymin, inv_h, nx, ny, start, end)
        for (a, b) in ranges:
            for k in range(a, b):
                j = order[k]
                dx = xi[0] - pos[j, 0]
                dy = xi[1] - pos[j, 1]
                r2 = dx*dx + dy*dy
                w = poly6_w(h, poly6_coef, r2)
                s += w
        rho[i] = mass * s
    return rho


@njit(cache=True)
def pbf_compute_lambdas(N: int, pos: np.ndarray, rho: np.ndarray,
                        rest_rho: np.float32, mass: np.float32,
                        h: np.float32, spiky_grad_coef: np.float32,
                        xmin: np.float32, ymin: np.float32, inv_h: np.float32, nx: int, ny: int,
                        order: np.ndarray, start: np.ndarray, end: np.ndarray,
                        eps: np.float32 = np.float32(1e-6)) -> np.ndarray:
    """λ_i = -C_i / (∑|∇C_i|^2 + ε), C_i = ρ_i/ρ0 - 1."""
    lambdas = np.zeros(N, np.float32)
    inv_rho0 = np.float32(1.0) / rest_rho
    m_over_rho0 = mass * inv_rho0

    for i in range(N):
        xi = pos[i]
        # C_i
        Ci = rho[i] * inv_rho0 - np.float32(1.0)

        gx_i = np.float32(0.0)
        gy_i = np.float32(0.0)
        sum_sq = np.float32(0.0)

        ranges = neighbor_ranges(xi, xmin, ymin, inv_h, nx, ny, start, end)
        for (a, b) in ranges:
            for k in range(a, b):
                j = order[k]
                if j == i:
                    continue
                rx = xi[0] - pos[j, 0]
                ry = xi[1] - pos[j, 1]
                gx, gy = spiky_grad(h, spiky_grad_coef, rx, ry)
                # grad wrt i: m/ρ0 * ∇W
                gx_ij = m_over_rho0 * gx
                gy_ij = m_over_rho0 * gy
                gx_i += gx_ij
                gy_i += gy_ij
                sum_sq += gx_ij*gx_ij + gy_ij*gy_ij

        sum_sq += gx_i*gx_i + gy_i*gy_i
        lambdas[i] = -Ci / (sum_sq + eps)

    return lambdas

def s_corr(self, w, dist):
    if dist >= 0.5 * self.h:
        return 0.0
    wd = self.W_poly6(self.delta_q) + 1e-12
    return self.tensile_k * (w / wd) ** self.tensile_n


@njit(cache=True)
def pbf_compute_delta_p(N: int, pos: np.ndarray, lambdas: np.ndarray,
                        rest_rho: np.float32, mass: np.float32,
                        h: np.float32, spiky_grad_coef: np.float32,
                        poly6_coef: np.float32, delta_q: np.float32,
                        scorr_k: np.float32, scorr_n: np.int32,
                        xmin: np.float32, ymin: np.float32, inv_h: np.float32, nx: int, ny: int,
                        order: np.ndarray, start: np.ndarray, end: np.ndarray, r_cut: np.float32) -> np.ndarray:
    """Δp_i = (m/ρ0) ∑_j (λ_i+λ_j + s_corr) ∇W(x_i - x_j)."""
    dp = np.zeros_like(pos, np.float32)
    m_over_rho0 = mass / (rest_rho + 1e-12)
    w_d = poly6_w(h, poly6_coef, delta_q*delta_q) + 1e-12
    for i in range(N):
        xi = pos[i]
        lam_i = lambdas[i]
        dpx = np.float32(0.0)
        dpy = np.float32(0.0)

        ranges = neighbor_ranges(xi, xmin, ymin, inv_h, nx, ny, start, end)
        for (a, b) in ranges:
            for k in range(a, b):
                j = order[k]
                if j == i:
                    continue
                rx = xi[0] - pos[j, 0]
                ry = xi[1] - pos[j, 1]

                r2 = rx*rx + ry*ry
                w = poly6_w(h, poly6_coef, r2)
                scorr = -scorr_k * ((w / w_d) ** scorr_n)

                gx, gy = spiky_grad(h, spiky_grad_coef, rx, ry)
                coeff = (lam_i + lambdas[j] + scorr)
                dpx += coeff * gx
                dpy += coeff * gy

        dp[i, 0] = m_over_rho0 * dpx
        dp[i, 1] = m_over_rho0 * dpy

    return dp


@njit(cache=True)
def pbf_xsph(N: int, pos: np.ndarray, vel: np.ndarray,
             h: np.float32, poly6_coef: np.float32,
             xmin: np.float32, ymin: np.float32, inv_h: np.float32, nx: int, ny: int,
             order: np.ndarray, start: np.ndarray, end: np.ndarray,
             xsph_c: np.float32) -> np.ndarray:
    """XSPH: v_i += c * ∑_j W(r)(v_j - v_i). (упрощённый, без деления на ρ)"""
    out = np.zeros_like(vel, np.float32)
    for i in range(N):
        xi = pos[i]
        vix = vel[i, 0]; viy = vel[i, 1]
        sumx = np.float32(0.0)
        sumy = np.float32(0.0)

        ranges = neighbor_ranges(xi, xmin, ymin, inv_h, nx, ny, start, end)
        for (a, b) in ranges:
            for k in range(a, b):
                j = order[k]
                if j == i:
                    continue
                rx = xi[0] - pos[j, 0]
                ry = xi[1] - pos[j, 1]
                r2 = rx*rx + ry*ry
                w = poly6_w(h, poly6_coef, r2)
                sumx += w * (vel[j, 0] - vix)
                sumy += w * (vel[j, 1] - viy)

        out[i, 0] = xsph_c * sumx
        out[i, 1] = xsph_c * sumy
    return out


#      MAIN CLASS
class PBFSimulation:
    """
    Position-Based Fluids (2D), ускоренная версия:
      - uniform grid соседства
      - все тяжёлые шаги в @njit функциях
    """

    def __init__(self,
                 bounds=(0, 15, 0, 10),
                 N=1000,
                 gravity=(0.0, 0.0),
                 rest_density=1.0,
                 h_dx_ratio=2.0,
                 iterations=4,
                 nu_xsph=0.05,
                 mass_coeff=1,
                 scorr_k=0.001,
                 scorr_n=4,
                 steps=1000,
                 seed=42):

        self.bounds = bounds
        self.N = int(N)
        self.gravity = np.array(gravity, dtype=np.float32)
        self.rest_density = np.float32(rest_density)
        self.h_dx_ratio = np.float32(h_dx_ratio)
        self.iterations = int(iterations)
        self.nu_xsph = np.float32(nu_xsph)
        self.scorr_k = np.float32(scorr_k)
        self.scorr_n = int(scorr_n)
        self.steps = int(steps)
        self.seed = seed

        self.compress_factor = 1

        if seed is not None:
            np.random.seed(seed)

        xmin, xmax, ymin, ymax = bounds
        W = np.float32(xmax - xmin)
        H = np.float32(ymax - ymin)

        self.dx = np.float32(np.sqrt((W * H) / np.float32(N))) * 0.5
        self.h = self.h_dx_ratio * self.dx
        self.mass = self.rest_density * (self.dx ** 2) * mass_coeff
        self.poly6_coef = np.float32(315.0) / (np.float32(64.0) * np.pi * self.h**9)
        self.spiky_grad_coef = np.float32(-45.0) / (np.pi * self.h**6)

        nx = int(np.sqrt(N))
        ny = int(np.sqrt(N))
        xs = np.linspace(xmin + 0.25*W, xmax - 0.25*W, nx, dtype=np.float32)
        ys = np.linspace(ymin + 0.25*H, ymax - 0.25*H, ny, dtype=np.float32)
        X, Y = np.meshgrid(xs, ys)
        grid = np.column_stack([X.ravel(), Y.ravel()])
        self.positions = grid[:N].astype(np.float32).copy()
        self.velocities = np.zeros_like(self.positions, dtype=np.float32)

        self.xmin = np.float32(xmin)
        self.ymin = np.float32(ymin)
        self.nx_cells = int(np.ceil(W / self.h))
        self.ny_cells = int(np.ceil(H / self.h))
        self.inv_h = np.float32(1.0) / self.h

        order, cell_sorted, start, end = build_grid(
            self.positions,
            np.float32(self.bounds[0]), np.float32(self.bounds[2]),
            self.inv_h, self.nx_cells, self.ny_cells
        )

        rho_init = pbf_compute_density(
            self.N, self.positions, self.mass, self.h, self.poly6_coef,
            self.xmin, self.ymin, self.inv_h, self.nx_cells, self.ny_cells,
            order, start, end
        )


    def step(self, dt: float = 0.008):
        dt = np.float32(dt)
        N = self.N

        self.velocities += dt * self.gravity
        p_pred = self.positions + dt * self.velocities

        for _ in range(self.iterations):
            order, cell_sorted, start, end = build_grid(
                p_pred, self.xmin, self.ymin, self.inv_h, self.nx_cells, self.ny_cells
            )
            rho = pbf_compute_density(
                N, p_pred, self.mass, self.h, self.poly6_coef,
                self.xmin, self.ymin, self.inv_h, self.nx_cells, self.ny_cells,
                order, start, end
            )
            lambdas = pbf_compute_lambdas(
                N, p_pred, rho, self.rest_density, self.mass,
                self.h, self.spiky_grad_coef,
                self.xmin, self.ymin, self.inv_h, self.nx_cells, self.ny_cells,
                order, start, end, np.float32(1e-6)
            )
            dp = pbf_compute_delta_p(
                N, p_pred, lambdas, self.rest_density, self.mass,
                self.h, self.spiky_grad_coef,
                self.poly6_coef, self.h * np.float32(0.3),
                self.scorr_k, self.scorr_n,
                self.xmin, self.ymin, self.inv_h, self.nx_cells, self.ny_cells,
                order, start, end, np.float32(0.3)
            )

            p_pred += dp

        self.velocities = (p_pred - self.positions) / dt

        order, cell_sorted, start, end = build_grid(
            p_pred, self.xmin, self.ymin, self.inv_h, self.nx_cells, self.ny_cells
        )
        xsph = pbf_xsph(
            N, p_pred, self.velocities,
            self.h, self.poly6_coef,
            self.xmin, self.ymin, self.inv_h, self.nx_cells, self.ny_cells,
            order, start, end,
            self.nu_xsph
        )
        self.velocities += xsph

        self.positions = p_pred
        self.handle_boundaries()
        # отладка
        # print("mass: ", self.mass, "rest_densities: ", self.rest_density, "rho: ", rho.mean(), 'p_pred: ', p_pred[0], p_pred[100], 'xsph: ', xsph[0], xsph[100], sep=' ')
        # print(rho.mean())

    def handle_boundaries(self, restitution: float = 0.5):
        xmin, xmax, ymin, ymax = self.bounds
        eps = np.float32(1e-6)
        p = self.positions
        v = self.velocities
        for i in range(self.N):
            x, y = p[i, 0], p[i, 1]
            vx, vy = v[i, 0], v[i, 1]

            if x < xmin:
                p[i, 0] = xmin + eps
                if vx < 0: v[i, 0] = -vx * np.float32(restitution)
            elif x > xmax:
                p[i, 0] = xmax - eps
                if vx > 0: v[i, 0] = -vx * np.float32(restitution)

            if y < ymin:
                p[i, 1] = ymin + eps
                if vy < 0: v[i, 1] = -vy * np.float32(restitution)
            elif y > ymax:
                p[i, 1] = ymax - eps
                if vy > 0: v[i, 1] = -vy * np.float32(restitution)

    def run(self, record_every=50, dt=0.008):
        hist = []
        record_every = max(1, record_every)
        for s in range(self.steps):
            self.step(dt=dt)
            if (s % record_every) == 0 or (s + 1) == self.steps:
                hist.append({
                    "step": int(s),
                    "pos": self.positions.copy(),
                    "vel": self.velocities.copy()
                })
        return hist


if __name__ == "__main__":
    sim = PBFSimulation(
        bounds=(0, 15, 0, 10),
        N=1156,
        gravity=(0.0, 0.0),
        rest_density=1.0,
        h_dx_ratio=2.0,
        iterations=5,
        nu_xsph=0.05,
        scorr_k=0.001,
        scorr_n=4,
        steps=600,
        seed=42
    )
    hist = sim.run(record_every=60, dt=0.008)
    print("frames:", len(hist), "| numba:", NUMBA_AVAILABLE)

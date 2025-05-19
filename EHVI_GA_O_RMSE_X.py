import streamlit as st
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning, DominatedPartitioning
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.optim import optimize_acqf
from scipy.spatial import ConvexHull

st.title("Slurry 조성 최적화: qEHVI 기반 3목적 Bayesian Optimization")

url = "https://raw.githubusercontent.com/Yerimdw/2504_slurry/refs/heads/main/LHS_slurry_data_st.csv"
df = pd.read_csv(url)
x_cols = ["Graphite", "Carbon\nblack", "CMC", "SBR", "Solvent"]
y_cols = ["yield stress", "viscosity"]

X_raw = df[x_cols].values
Y_raw = df[y_cols].values
graphite_idx = x_cols.index("Graphite")
graphite_wt_values = X_raw[:, graphite_idx].reshape(-1, 1)
Y_raw_extended = np.hstack([Y_raw, graphite_wt_values])

x_scaler = MinMaxScaler()
x_scaler.fit(X_raw)
X_scaled = x_scaler.transform(X_raw)

train_x = torch.tensor(X_scaled, dtype=torch.double)
train_y = torch.tensor(Y_raw_extended, dtype=torch.double)

train_y_hv = train_y.clone()
train_y_hv[:, 1] = -train_y_hv[:, 1]

ref_point = [0.0, -10.0, 20.0]
partitioning = NondominatedPartitioning(ref_point=torch.tensor(ref_point, dtype=torch.double), Y=train_y_hv)

model = SingleTaskGP(train_x, train_y_hv)
mll = ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_mll(mll)

# ---------------- GA 초기 후보 생성 ------------------
# EI용 모델: yield stress만 학습
train_y_single = torch.tensor(Y_raw[:, 0], dtype=torch.float64).unsqueeze(-1)
ei_model = SingleTaskGP(train_x, train_y_single)
ei_mll = ExactMarginalLogLikelihood(ei_model.likelihood, ei_model)
fit_gpytorch_mll(ei_mll)

# EI 함수 정의
def expected_improvement(x_tensor, model, best_f):
    ei = ExpectedImprovement(model=model, best_f=best_f)
    return ei(x_tensor.unsqueeze(0)).item()

# GA 적합도 함수 정의
def fitness(x_tensor, model, best_f, scaler, bounds_tensor, graphite_idx, viscosity_idx):
    ei_val = expected_improvement(x_tensor, model, best_f)
    x_denorm = scaler.inverse_transform(x_tensor.unsqueeze(0).numpy()).squeeze()

    # Graphite와 viscosity 인덱스 알려줌
    graphite_idx = 0  # X에서 1번째 열
    viscosity_idx = 3  # Y에서 4번째 열

    graphite = x_denorm[graphite_idx]
    viscosity = x_denorm[viscosity_idx]
    graphite_norm = (graphite - 20.0) / (40.0 - 20.0)
    if viscosity < 0.1 or viscosity > 1.5:
        viscosity_penalty = -1.0
    else:
        viscosity_target = 0.5
        viscosity_penalty = -abs(viscosity - viscosity_target)
    return ei_val + viscosity_penalty + 0.5 * graphite_norm

# GA 실행 함수
def run_GA_for_initial_candidates(model, bounds_tensor, best_f, scaler, graphite_idx, viscosity_idx,
                                  pop_size=20, generations=50):
    dim = bounds_tensor.shape[1]
    pop = torch.rand(pop_size, dim, dtype=torch.float64)
    for _ in range(generations):
        fitness_vals = torch.tensor([
            fitness(x, model, best_f, scaler, bounds_tensor, graphite_idx, viscosity_idx) for x in pop
        ], dtype=torch.float64)
        topk = torch.topk(fitness_vals, k=pop_size // 2)
        parents = pop[topk.indices]
        children = []
        for i in range(0, len(parents), 2):
            p1, p2 = parents[i], parents[(i + 1) % len(parents)]
            alpha = torch.rand(1).item()
            child = alpha * p1 + (1 - alpha) * p2
            child += 0.05 * torch.randn(dim, dtype=torch.float64)
            child = torch.clamp(child, 0.0, 1.0)
            children.append(child)
        pop = torch.vstack((parents, torch.stack(children)))
    fitness_final = torch.tensor([
        fitness(x, model, best_f, scaler, bounds_tensor, graphite_idx, viscosity_idx) for x in pop
    ], dtype=torch.float64)
    best_indices = torch.topk(fitness_final, k=10).indices
    return pop[best_indices]

normalized_bounds = torch.tensor([[0.0] * train_x.shape[1], [1.0] * train_x.shape[1]], dtype=torch.float64)
best_y = train_y_single[:, 0].max().item()
initial_conditions = run_GA_for_initial_candidates(
    ei_model, normalized_bounds, best_y, x_scaler, graphite_idx, 1,
    pop_size=20, generations=50
).to(dtype=torch.double)

# GA → BO 초기 조성 출력
st.subheader("GA → BO로 넘긴 조성 후보 (정규화된 100 wt%)")

# 역정규화 및 100 wt%로 환산
init_candidates_np = x_scaler.inverse_transform(initial_conditions.cpu().numpy())
init_candidates_normalized = init_candidates_np / np.sum(init_candidates_np, axis=1, keepdims=True) * 100

# 표로 표시
init_df = pd.DataFrame(init_candidates_normalized, columns=x_cols)
init_df["Total"] = init_df.sum(axis=1).round(2)
init_df.index = np.arange(1, len(init_df) + 1)
st.dataframe(init_df.round(2))


# ---------------- EHVI 최적화 ------------------
acq_func = qExpectedHypervolumeImprovement(
    model=model,
    ref_point=ref_point,
    partitioning=partitioning
)
candidate_scaled, _ = optimize_acqf(
    acq_func,
    bounds=normalized_bounds,
    q=1,
    num_restarts=len(initial_conditions),
    raw_samples=128,
    options={"batch_initial_conditions": initial_conditions}
)
candidate_wt = x_scaler.inverse_transform(candidate_scaled.detach().cpu().numpy())[0]
candidate_wt = candidate_wt / np.sum(candidate_wt) * 100

if candidate_wt[graphite_idx] < 30.0:
    st.warning(f"Graphite wt%: {candidate_wt[graphite_idx]:.2f} wt% < 30.0 wt% (제약 무효)")
else:
    st.success(f"Graphite wt%: {candidate_wt[graphite_idx]:.2f} wt%")

st.subheader("최적 조성 (qEHVI 추천)")
for col in x_cols:
    idx = x_cols.index(col)
    st.write(f"{col}: **{candidate_wt[idx]:.2f} wt%**")
st.write(f"**총합**: {np.sum(candidate_wt):.2f} wt%")

X_predict = x_scaler.transform(candidate_wt.reshape(1, -1))
X_tensor = torch.tensor(X_predict, dtype=torch.double)
posterior = model.posterior(X_tensor)
pred_mean = posterior.mean.detach().cpu().numpy()[0]
yield_pred = pred_mean[0]
visc_pred = -pred_mean[1]
graphite_pred = pred_mean[2]

st.write(f"**예측 Yield Stress**: {yield_pred:.2f} Pa")
st.write(f"**예측 Viscosity**: {visc_pred:.3f} Pa.s")
st.write(f"**예측 Graphite wt%**: {graphite_pred:.2f} wt%")

pareto_mask = is_non_dominated(train_y_hv)
train_y_vis_plot = train_y_hv.clone()
train_y_vis_plot[:, 1] = -train_y_vis_plot[:, 1]
pareto_points = train_y_vis_plot[pareto_mask].numpy()

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(train_y_vis_plot[:, 1], train_y_vis_plot[:, 0], train_y_vis_plot[:, 2],
           color='gray', alpha=0.7, label='Data', s=30, depthshade=True)
ax.scatter(pareto_points[:, 1], pareto_points[:, 0], pareto_points[:, 2],
           color='red', edgecolors='black', s=90, marker='o', depthshade=True, label='Pareto Front')
ax.scatter(visc_pred, yield_pred, graphite_pred,
           color='blue', edgecolors='black', s=200, marker='^', label='Candidate')

if len(pareto_points) >= 4:
    try:
        hull = ConvexHull(pareto_points)
        for simplex in hull.simplices:
            tri = pareto_points[simplex]
            ax.plot_trisurf(tri[:, 1], tri[:, 0], tri[:, 2],
                            color='pink', alpha=0.4, edgecolor='gray', linewidth=1.2)
    except Exception as e:
        st.warning(f"Convex Hull 실패: {e}")

ax.set_xlabel("Viscosity [Pa.s] (↓)", fontsize=12, labelpad=10)
ax.set_ylabel("Yield Stress [Pa] (↑)", fontsize=12, labelpad=10)
ax.set_zlabel("Graphite wt% (↑)", fontsize=12, labelpad=15)
ax.set_zlim(20, 40)
ax.zaxis.set_ticks(np.arange(20, 45, 5))
ax.view_init(elev=25, azim=135)
ax.legend()
ax.grid(True)
plt.tight_layout()
st.pyplot(fig)

hv_log_path = "hv_tracking_3obj.csv"
hv_list = []
ref_point_fixed = torch.tensor([0.0, -15.0, 20.0], dtype=torch.double)

for i in range(1, len(train_y_hv) + 1):
    current_Y = train_y_hv[:i].clone()
    try:
        bd = DominatedPartitioning(ref_point=ref_point_fixed, Y=current_Y.clone().detach())
        hv = bd.compute_hypervolume().item()
    except Exception as e:
        hv = float('nan')
        st.warning(f"{i}번째 계산 중 에러: {e}")
    hv_list.append({"iteration": i, "hv": hv})

hv_df = pd.DataFrame(hv_list)
hv_df.to_csv(hv_log_path, index=False)

fig_hv, ax_hv = plt.subplots(figsize=(8, 4))
ax_hv.plot(hv_df["iteration"], hv_df["hv"], marker='o')
ax_hv.set_xlabel("Iteration")
ax_hv.set_ylabel("Hypervolume")
ax_hv.set_title("3D Hypervolume Progress Over Iterations")
ax_hv.set_xticks(np.arange(1, hv_df["iteration"].max() + 1, 3))
ax_hv.grid(True)
st.pyplot(fig_hv)
# GPT가 GA랑 BO 합쳐준거 : GA 맞는지 확인

# C:/YR/PycharmProjects/2505_data_driven_final/250519_GA_BO.py
# Step 3 : Genetic algorithm을 시작점으로 하는 Bayesian optimization

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

st.set_page_config(page_title="Bayasian optimization", layout="wide")
st.title("Slurry 조성 최적화: qEHVI 기반 3목적 Bayesian Optimization")

CSV_PATH = "Slurry_data_LHS.csv"
df = pd.read_csv(CSV_PATH)

x_cols = ["Graphite", "Carbon black", "CMC", "SBR", "Solvent"]  # 대괄호 두번써야 여러 열 한꺼번에 가져옴
y_cols = ["yield stress", "n", "K", "viscosity"]

# JK
# url = "https://raw.githubusercontent.com/Yerimdw/2504_slurry/refs/heads/main/LHS_slurry_data_st.csv"
# df = pd.read_csv(url)
# x_cols = ["Graphite", "Carbon\nblack", "CMC", "SBR", "Solvent"]
# y_cols = ["yield stress", "viscosity"]

# # Data lodaing - GitHub
# url = "https://raw.githubusercontent.com/Yerimdw/2504_slurry/refs/heads/main/LHS_slurry_data_st.csv"
# df = pd.read_csv(url)
# x_cols = df[["Graphite", "Carbon_black", "CMC", "SBR", "Solvent"]]   # 대괄호 두번써야 여러 열 한꺼번에 가져옴
# y_cols = df[["Yield_stress", "n", "K", "Viscosity"]]

X_raw = df[x_cols].values
Y_raw = df[y_cols].values
graphite_idx = x_cols.index("Graphite")
graphite_wt_values = X_raw[:, graphite_idx].reshape(-1, 1)
Y_raw_2 = Y_raw[:, [0, 3]]  # Yield Stress, Viscosity 만 사용
Y_raw_extended = np.hstack([Y_raw_2, graphite_wt_values]) # graphite도 최적화에 포함

x_scaler = MinMaxScaler()
x_scaler.fit(X_raw)
X_scaled = x_scaler.transform(X_raw)

train_x = torch.tensor(X_scaled, dtype=torch.double)
train_y = torch.tensor(Y_raw_extended, dtype=torch.double)

# 점도는 최소화 해야 하니까 부호 반전
train_y_hv = train_y.clone() # hypervolume  # 나중에 train_y_hv 사용할 수도 있으니까 복사본으로 함
train_y_hv[:, 1] = -train_y_hv[:, 1]


# ---------------- GA ------------------
# GP 모델 학습
train_y_single = torch.tensor(Y_raw[:, 0], dtype=torch.float64).unsqueeze(-1)
ga_model = SingleTaskGP(train_x, train_y_single) # SingleTaskGP : yield stress만 학습
ga_mll = ExactMarginalLogLikelihood(ga_model.likelihood, ga_model)
fit_gpytorch_mll(ga_mll)

# EI 함수 정의
def expected_improvement(x_tensor, ga_model, best_f):
    ei = ExpectedImprovement(model=ga_model, best_f=best_f)
    return ei(x_tensor.unsqueeze(0)).item()

# fitness = yield strss의 EI + viscoisty의 조건(penalty) + graphite의 양
def fitness(x_tensor, ga_model, best_f, scaler, graphite_idx, viscosity_idx):
    # x_tensor: 조성 한 개체, best_f: 최대 y.s 값, graphite_idx: x_tensor에서 graphtie의 위치
    ei_val = expected_improvement(x_tensor, ga_model, best_f)

    # graphtie_idx라는 GA가 생성한 개체의 0~1사이 값을 wt% 단위로 역정규화
    x_denorm = scaler.inverse_transform(x_tensor.unsqueeze(0).numpy()).squeeze()
    graphite = x_denorm[graphite_idx]
    viscosity = x_denorm[viscosity_idx]

    # graphite 양을 20 = 0.0, 40 = 1.0이 되도록 변환 (정규화)
    graphite_norm = (graphite - 20.0) / (40.0 - 20.0)

    # viscosity penalty 부여 (0.5 Pa.s에 가까울수록 좋음)
    if viscosity < 0.1 or viscosity > 1.5:
        viscosity_penalty = -1.0
    else:
        viscosity_target = 0.5
        viscosity_penalty = -abs(viscosity - viscosity_target)
    return ei_val + viscosity_penalty + 0.5 * graphite_norm

# GA를 통한 초기 후보 추천
def run_GA(model, bounds_tensor, best_f, scaler, graphite_idx, viscosity_idx,
                                  pop_size=20, generations=30):
    # bounds_tensor: 변수의 범위, pop_size: 한 세대 개체 수
    dim = bounds_tensor.shape[1] # 입력 변수의 수(train_x 열의 개수)
    pop = torch.rand(pop_size, dim, dtype=torch.float64)    # 개체 생성 (pop_size x dim 만큼)
    for _ in range(generations):
        # fitness 기반 평가
        fitness_vals = torch.tensor([
            fitness(x, ga_model, best_f, scaler, graphite_idx, viscosity_idx) for x in pop
        ], dtype=torch.float64) # tensor 형식으로 변환
        # 상위 절반을 부모로 선택
        topk = torch.topk(fitness_vals, k=pop_size // 2)  # torch.topk: 상위 k값과 인덱스 반환하는 함수
        parents = pop[topk.indices]

        # 자식 생성
        children = []
        for i in range(0, len(parents), 2):
            p1, p2 = parents[i], parents[(i + 1) % len(parents)]
            alpha = torch.rand(1).item()
            child = alpha * p1 + (1 - alpha) * p2
            child += 0.05 * torch.randn(dim, dtype=torch.float64)
            child = torch.clamp(child, 0.0, 1.0)  # 텐서 값 범위를 제한하는 함수
            children.append(child)
        # 다음 세대 업데이트
        pop = torch.vstack((parents, torch.stack(children)))

    # 최종 평가 후 상위 10개 선택
    fitness_final = torch.tensor([
        fitness(x, ga_model, best_f, scaler, graphite_idx, viscosity_idx) for x in pop
    ], dtype=torch.float64)
    best_indices = torch.topk(fitness_final, k=10).indices
    return pop[best_indices]

# GA 실행
normalized_bounds = torch.tensor([[0.0] * train_x.shape[1], [1.0] * train_x.shape[1]], dtype=torch.float64)
# GA가 0~1 사이의 개체를 만들도록 범위 정함
best_y = train_y_single[:, 0].max().item()  # yield stress 최대값 찾기

initial_conditions = run_GA(
    ga_model, normalized_bounds, best_y, x_scaler, graphite_idx, 3,
    pop_size=20, generations=50
).to(dtype=torch.double)

# GA → BO 초기 조성 출력
st.subheader("GA를 통한 BO의 초기 시작지점이 될 후보 추천 (wt%)")

# 원래 wt% 단위로 역정규화
init_candidates_np = x_scaler.inverse_transform(initial_conditions.cpu().numpy())
# 총합이 100 wt%가 되도록 환산
init_candidates_normalized = init_candidates_np / np.sum(init_candidates_np, axis=1, keepdims=True) * 100

# 표로 표시
init_df = pd.DataFrame(init_candidates_normalized, columns=x_cols)
init_df["Total"] = init_df.sum(axis=1).round(2)
init_df.index = np.arange(1, len(init_df) + 1)
st.dataframe(init_df.round(2))


# ---------------- EHVI 최적화 ------------------
ref_point = [0.0, -10.0, 20.0] # viscosity JK 15에서 YR 10으로 변경
partitioning = NondominatedPartitioning(ref_point=torch.tensor(ref_point, dtype=torch.double), Y=train_y_hv)
# non-dominated와 dominated(모든 성능 더 안좋은 영역) 영역으로 분할
# qEHVI가 non-dominated 영역 확장하도록

# BO의 GPR model
bo_model = SingleTaskGP(train_x, train_y_hv)
mll = ExactMarginalLogLikelihood(bo_model.likelihood, bo_model)
fit_gpytorch_mll(mll)

acq_func = qExpectedHypervolumeImprovement( # 현재 Pareto 경계 밖에서 HV을 최대화할 후보 찾음
    model=bo_model,
    ref_point=ref_point,
    partitioning=partitioning
)
candidate_scaled, _ = optimize_acqf(
    acq_func,
    bounds=normalized_bounds,
    q=1, # 조성 1개 추천
    num_restarts=len(initial_conditions),
    raw_samples=128,    # 탐색 시작 후보 개수 (임의로 정한 중간정도의 값)
    options={"batch_initial_conditions": initial_conditions}
)
# 실제 wt%로 역정규화
candidate_wt = x_scaler.inverse_transform(candidate_scaled.detach().cpu().numpy())[0]
# 총합이 100 wt%가 되도록
candidate_wt = candidate_wt / np.sum(candidate_wt) * 100

# if candidate_wt[graphite_idx] < 30.0:
#     st.warning(f"Graphite wt%: {candidate_wt[graphite_idx]:.2f} wt% < 30.0 wt% (제약 무효)")
# else:
#     st.success(f"Graphite wt%: {candidate_wt[graphite_idx]:.2f} wt%")

# 조성 Table 출력
st.subheader("qEHVI를 통한 최적화 조성 추천")
table_composition = pd.DataFrame({
    "Graphite": [f"{candidate_wt[x_cols.index('Graphite')]:.2f} wt%"],
    "Carbon black": [f"{candidate_wt[x_cols.index('Carbon black')]:.2f} wt%"],
    "CMC": [f"{candidate_wt[x_cols.index('CMC')]:.2f} wt%"],
    "SBR": [f"{candidate_wt[x_cols.index('SBR')]:.2f} wt%"],
    "Solvent": [f"{candidate_wt[x_cols.index('Solvent')]:.2f} wt%"]
})
st.dataframe(table_composition, hide_index=True)

X_predict = x_scaler.transform(candidate_wt.reshape(1, -1)) # 정규화
X_tensor = torch.tensor(X_predict, dtype=torch.double) # numpy 배열에서 PyTorch tensor로 변경
posterior = bo_model.posterior(X_tensor) # model에 input 넣어 예측값 구함
pred_mean = posterior.mean.detach().cpu().numpy()[0]
# 예측값
yield_pred = pred_mean[0]
visc_pred = -pred_mean[1] # 최소화를 위해 부호 반전했으니까 다시 - 붙임
graphite_pred = pred_mean[2]

# 예측값 Table 출력
st.subheader("추천 조성의 유변학적 물성")
table_predicted = pd.DataFrame({
    "Yield stress": [f"{yield_pred:.2f} Pa"],
    "Viscosity": [f"{visc_pred:.3f} Pa.s"],
    "Graphite wt% (Pred)": [f"{graphite_pred:.2f} wt%"]
})
st.dataframe(table_predicted, hide_index=True)

# Pareto front 출력
st.subheader("Pareto front graph")
pareto_mask = is_non_dominated(train_y_hv) # is_non_dominated 함수: 비지배 해 인덱스 추출
train_y_vis_plot = train_y_hv.clone()
train_y_vis_plot[:, 1] = -train_y_vis_plot[:, 1] # viscosity 부호 복원
pareto_points = train_y_vis_plot[pareto_mask].numpy()

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(train_y_vis_plot[:, 1], train_y_vis_plot[:, 0], train_y_vis_plot[:, 2],
           color='gray', alpha=0.7, label='Experimental Data', s=40, depthshade=False)
ax.scatter(pareto_points[:, 1], pareto_points[:, 0], pareto_points[:, 2],
           color='red', s=40, marker='o', edgecolors='black', depthshade=False, label='Pareto Front')
ax.scatter(visc_pred, yield_pred, graphite_pred,
           color='blue', edgecolors='black', s=40, marker='^', label='Candidate')

if len(pareto_points) >= 4: # 3개 이하는 convex hull 생성 불가
    try:
        hull = ConvexHull(pareto_points)
        for simplex in hull.simplices:
            tri = pareto_points[simplex]
            ax.plot_trisurf(tri[:, 1], tri[:, 0], tri[:, 2],
                            color='pink', alpha=0.4, edgecolor='gray', linewidth=1.2)
    except Exception as e:
        st.warning(f"Convex Hull 실패: {e}")

st.subheader("최적화 반복에 따른 Hypervolume 증가량")
ax.set_xlabel("Viscosity [Pa.s] (↓)", fontsize=12, labelpad=10)
ax.set_ylabel("Yield Stress [Pa] (↑)", fontsize=12, labelpad=10)
ax.set_zlabel("Graphite wt% (↑)", fontsize=12, labelpad=15)
ax.set_zlim(20, 40) # z축 범위를 20~40으로 제한 (Graphite)
ax.zaxis.set_ticks(np.arange(20, 45, 5))
ax.view_init(elev=25, azim=135) # 최적의 시야각 설정
ax.legend()
ax.grid(True)
fig.subplots_adjust(left=0, right=5, top=1, bottom=0)
st.pyplot(fig)

# st.subheader("5-Fold Cross Validation (RMSE)")
# kf = KFold(n_splits=5, shuffle=True, random_state=42)
# rmse_dict = {"yield_stress": [], "viscosity": [], "graphite_wt%": []}
# X_tensor = torch.tensor(X_scaled, dtype=torch.double)
# Y_tensor = torch.tensor(Y_raw_extended, dtype=torch.double)
#
# for train_idx, test_idx in kf.split(X_tensor):
#     X_train, Y_train = X_tensor[train_idx], Y_tensor[train_idx]
#     X_test, Y_test = X_tensor[test_idx], Y_tensor[test_idx]
#     Y_train_mod = Y_train.clone()
#     Y_train_mod[:, 1] = -Y_train_mod[:, 1]
#     Y_train_mod[:, 1] = -Y_train_mod[:, 1]
#     model_cv = SingleTaskGP(X_train, Y_train_mod)
#     mll_cv = ExactMarginalLogLikelihood(model_cv.likelihood, model_cv)
#     fit_gpytorch_mll(mll_cv)
#     preds = model_cv.posterior(X_test).mean.detach().numpy()
#     preds[:, 1] = -preds[:, 1]
#     for i, target in enumerate(["yield_stress", "viscosity", "graphite_wt%"]):
#         rmse = np.sqrt(mean_squared_error(Y_test[:, i], preds[:, i]))
#         rmse_dict[target].append(rmse)
#
# rmse_df = pd.DataFrame({
#     "Target": list(rmse_dict.keys()),
#     "RMSE Mean": [np.mean(v) for v in rmse_dict.values()],
#     "RMSE Std": [np.std(v) for v in rmse_dict.values()]
# })
# rmse_df.index = np.arange(1, len(rmse_df) + 1)
# st.dataframe(rmse_df)

hv_log_path = "hv_tracking_3obj.csv"
hv_list = []
ref_point_fixed = torch.tensor([0.0, -10.0, 20.0], dtype=torch.double)

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

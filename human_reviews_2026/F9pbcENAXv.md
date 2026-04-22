# Multi-Fidelity Physics-Informed Neural Networks (PINN) with Boundary-Aware Losses for Ice-Bed Topography Prediction

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 2, 4

## Abstract
Predicting ice dynamics and sea-level rise requires an understanding of subglacial bedrock topography; however, inversion remains a challenging task in data-sparse regions where surface observations are limited. Some conventional machine learning methods face challenges in predicting subglacial topography due to heavy reliance on purely data correlations and cannot guarantee physical consistency, especially in data-sparse regions. Physics-Informed Neural Networks (PINNs) address this limitation by embedding partial differential equation (PDE) constraints into deep learning, enabling more physically consistent predictions. However, most existing PINN formulations depend on a single fidelity of physics, and soft boundary penalties can still compromise performance. We propose a multi-fidelity PINN framework for ice-bed topography prediction that advances beyond these limitations in two ways. First, we introduce multi-fidelity residual coupling, jointly enforcing the shallow-ice approximation (SIA) and reduced-Stokes equations within a single network. This coupling improves accuracy while maintaining physics consistency, achieving strong predictive performance (e.g., Test MSE = 0.028, and $R^2$ = 0.97). Second, we design a boundary-aware weak-form loss that supports traction/flux (Neumann) and optional Dirichlet constraints, allowing flexible enforcement of margin physics. Experiments show that hard Dirichlet enforcement over-constrains the model and reduces accuracy, while soft or selective enforcement preserves predictive quality. To our knowledge, this is the first Physics-Informed Neural Network (PINN) framework for predicting ice-bed topography that unifies multi-fidelity partial differential equation (PDE) residuals with configurable boundary-aware losses, providing a practical and extensible approach to physically plausible predictions in data-sparse regimes.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper studies the problem of estimating the bedrock topography beneath glaciers using limited radar data. The task is important because the shape of the subglacial bed affects ice flow and therefore projections of sea-level rise. Standard data-driven regression methods, such as random forests or neural networks, can predict bed elevation from surface features (e.g., surface velocity, elevation, mass balance), but they do not respect the underlying physics and typically perform poorly in areas where no direct data are available. 

To address this, the paper uses PINNs, where a neural network is trained not only to fit data but also to satisfy physical equations. The authors extend a standard PINN by enforcing two PDE constraints simultaneously. One is the Shallow-Ice Approximation (SIA), a simplified diffusion-like model for ice flow:
$$
\nabla \cdot (M \nabla \hat{b}(x)) = 0,
$$
and the other is a reduced form of the Stokes momentum equation,
$$
-\nu \Delta \hat{b}(x) - f = 0,
$$
where $\hat{b}(x)$ is the predicted bed elevation, $\nu$ is viscosity, and $f$ is external forcing. These two PDEs correspond to different fidelities of physical modeling: SIA is faster but less accurate, while the reduced-Stokes formulation captures more physics at higher computational cost. The paper combines them in one loss function as
$$
L_{\text{phys}} = w_{\text{SIA}}\|r_{\text{SIA}}\|^2 + w_{\text{Stokes}}\|r_{\text{Stokes}}\|^2,
$$
with either fixed or uncertainty-based weights. In addition, the paper introduces a “boundary-aware” component that treats the glacier margins through a weak-form loss. At boundaries, the model enforces either a Neumann (flux) condition,
$$
r_{\text{Neu}} = \nabla \hat{b} \cdot n - g_N,
$$
or an optional Dirichlet (fixed value) constraint,
$$
r_{\text{Dir}} = \hat{b} - u_D,
$$
depending on the availability of data. The total loss includes data fitting and these physics terms:
$$
L = L_{\text{data}} + L_{\text{phys}} + \lambda_{\text{Neu}}\|r_{\text{Neu}}\|^2 + \lambda_{\text{Dir}}\|r_{\text{Dir}}\|^2.
$$
The method is tested on radar measurements from Greenland’s Upernavik glacier (around 600k samples) using an 80/20 train-test split.

### Strengths
The paper clearly defines a physically meaningful target problem and integrates two existing physical formulations (SIA and reduced-Stokes) in a single learning framework. The mathematical setup is well-documented. The overall loss function is transparent and can be reproduced from the description. The authors also analyze different training configurations, including the effect of adaptive weighting between PDE terms and boundary constraints. Reporting of both conventional regression metrics (MAE, RMSE, $R^2$) and PDE residual norms $\|r_{\text{SIA}}\|^2$, $\|r_{\text{Stokes}}\|^2$ is useful for understanding whether the model satisfies physics as intended. The inclusion of an explicit weak-form treatment for Neumann and Dirichlet boundaries makes the approach flexible, and the observation that large Dirichlet penalties can harm learning is empirically well-supported by the ablation results.

### Weaknesses
While the integration of two PDE fidelities is technically correct, the contribution over standard PINNs is incremental rather than conceptual. The combination of residuals $r_{\text{SIA}}$ and $r_{\text{Stokes}}$ is achieved through a weighted sum; the paper does not show that this coupling leads to fundamentally new behavior beyond regular multi-task loss balancing. The choice of weights ($w_{\text{SIA}}=0.25$, $w_{\text{Stokes}}=0.75$) or the uncertainty-based alternative is only briefly justified, and the sensitivity of results to these parameters is not explored. The reduced-Stokes and SIA equations are simplified to the extent that important aspects of ice flow (e.g., vertical shear, thermomechanical coupling) are not represented, so the resulting “physical consistency” is limited to these approximations. The experimental setup focuses on one glacier system; it is unclear whether the method generalizes to other regions or to cases with different boundary geometries. The boundary-aware formulation depends on knowing or interpolating Dirichlet targets $u_D$ from radar data, which assumes such data exist at margins; in many glaciers, this is not true. The claim that hard Dirichlet enforcement “over-constrains” the solution is qualitative and might depend on the scaling of $\lambda_{\text{Dir}}$ rather than an inherent property of the method. Many details such as the computational aspects, like runtime or convergence stability of the PINN relative to standard PDE solvers, are also not discussed.

### Questions
1. The paper includes both the Shallow-Ice Approximation (SIA) and a reduced-Stokes model as residual penalties in the loss function. Could the authors provide more insight into how these interact during training? In particular, is there empirical evidence that the inclusion of both leads to improved learning dynamics (e.g., faster convergence, better generalization) compared to enforcing either alone? A comparison of gradient norms or loss curvature for each residual term would be informative. If both terms are correlated or redundant in some regions, how is this handled?

2. The experiments use fixed weights (e.g., $w_{\text{SIA}} = 0.25$, $w_{\text{Stokes}} = 0.75$) and an uncertainty-based alternative (log-variance weighting). Could the authors clarify how sensitive the final performance is to these weights? Was a grid search or hyperparameter sweep conducted? If uncertainty weighting is used, how stable are the learned variances across runs? A plot of weight evolution or an ablation comparing adaptive vs. fixed weighting would help interpret the benefit.

3. The boundary-aware loss relies partly on interpolated Dirichlet values at the glacier margins. In practice, such boundary values may not be available or may have high uncertainty. How robust is the method to incorrect or missing Dirichlet constraints? Could the authors describe what happens when only Neumann flux terms are used, and no $u_D$ is provided? Also, how were the boundary normals $n$ estimated in the Neumann residual term?

4. The method is evaluated only on the Upernavik glacier system. Can the authors comment on the model’s ability to generalize to different glaciers, such as those with different bed roughness, mass balance profiles, or margin geometries? Was any transfer learning or leave-one-glacier-out evaluation attempted?

5. The paper does not discuss the runtime, convergence stability, or computational challenges of training the proposed PINN with physics constraints. How do training times compare to standard data-driven baselines or traditional PDE solvers (e.g., finite element models)? 

6. The result that “hard Dirichlet enforcement degrades accuracy” is stated somewhat strongly. Could the authors clarify whether this is sensitive to the weight $\lambda_{\text{Dir}}$ or the fraction of boundary points receiving Dirichlet labels? Would softer enforcement or using learned penalties improve outcomes?

7. The paper mentions that the code is available, but no link is provided in the anonymized version. For reproducibility and further analysis, will all scripts, data preprocessing steps, and hyperparameter configurations be released upon acceptance? If synthetic or simplified datasets were used during development, sharing those might also benefit future benchmarking.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes a physics-informed neural network for subglacial bed topography that couples two physics fidelities (shallow-ice approximation and a reduced-stokes surrogate) inside one PINN loss and adds a boundary-aware weak-form term that mixes Neumann traction and optional Dirichlet constraints at glacier margins. On a Greenland radar track dataset, the method reports test mse with $r^2$ and claims better physics residuals than single-fidelity or purely data-driven baselines.

### Strengths
1. The paper aims to address a relevant geoscience problem where labeled data are sparse and physical constraints matter, and motivates the need for physics-guided inversion rather than black-box regression.
2. The multi-fidelity idea is sensible: use SIA for cheap broad constraints and a higher-order residual for added fidelity, with learned or fixed weights to balance them.

### Weaknesses
1. The “reduced-stokes” residual is specified as $r_{Stokes} = −ν\Delta \hat b − f$ with $ν=1$ and $f=0$, which collapses to a Poisson-like smoothness on the bed field rather than a demonstrably derived momentum balance tied to ice rheology or sliding. This risks being a hand-crafted regularizer rather than a true higher-fidelity physics term.
2. The physical role of the predicted variable is unclear: the network maps surface features to bed elevation $\hat b$, but the residuals are written directly on $\hat b$ without showing how $\hat b$ couples to velocity, thickness, or stresses in SIA/Stokes. 
3. Neumann boundary condition uses $g_N = 0$ by default, which is a strong assumption..
4. The dataset split appears random 80/20 over track points; this can cause spatial leakage because nearby points on a flight line are strongly correlated. The text says the team ensured points were “not too similar,” but it lacks a rigorous spatial holdout protocol and distance thresholds.
5. Metric reporting mixes “training units” and “physical units,” leading to confusing cross-model comparisons (e.g., random forest shows r²=0.987 yet huge mae/rmse due to unit scaling). 
6. The baseline shows a “weighted physics objective” of 0 while listing nonzero residuals; SIA and Stokes residuals are identical to two decimals in multiple rows. Can you explain this?
7. The “uncertainty weighting” via Kendall log-variance is used for loss balancing, but no calibration, uncertainty evaluation, or learned weight trajectories are presented, so the “uncertainty” interpretation is not strong.
8. Interpolated boundary labels may leak target information.

### Questions
Please address the weaknesses above.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposed a multi-fidelity, boundary-aware Physics-Informed Neural Network (PINN) framework for ice-bed topography prediction. The proposed framework couples the shallow-ice approximation (SIA) and reduced-Stokes equations and integrates weak-form boundary conditions. Experimental results show that the proposed framework outperforms both the non-physics and physics-only baselines. The authors also provided the results of the variants to support the importance of each component in the proposed framework.

### Strengths
1. The proposed multi-fidelity framework is novel to the application of ice-bed topography prediction.
2. The experiments were conducted using the real-world dataset, which demonstrates the effectiveness of the method.
3. The proposed framework with all the components achieved superior performance compared to the baselines.

### Weaknesses
1. The experiments are all conducted on a single dataset. I’m not familiar with this task, but I think if more datasets are involved, the experiments would be more solid and comprehensive.
2. The quantitative results of the baselines are provided in Table 2. However, the reported errors of these methods (including the single-fidelity PINN) are several orders of magnitudes larger than those of the proposed method and its variants. It would be better to include some qualitative results of these methods to make the results more convincing. 
3. For the single-fidelity PINN baseline, only the one that enforces SIA is covered. The baseline that enforces reduced-Stokes equations should also be included.

### Questions
1. In Table 1, could the authors clarify what training units and physics units are?
2. In Equation (3), could the authors further explain why $g_N=0$? Similarly, it is unclear why $\lambda_{Dir}=0$ in the loss fuction.
3. In Section 6.2, what is the difference between Main Multi-fidelity (uncertainty weighting) and Boundary-aware (Dirichlet optional)? What are the boundary conditions in Main Multi-fidelity? From my understanding, the Boundary-aware includes all the components. However, it underperforms Main Multi-fidelity in Table 1.
4. How did the weights in the loss function selected? The description should be included if cross validation is employed.
5. In Figure 3 and Figure 5, it is a little bit hard for readers to find the differences between the results. It would be better to highlight the regions for comparison.
6. [Minor] There is no need to include parameters from codes such as “USER_UNCERTAINTY=True” or “HARD DIRICHLET=True” as they may be confusing for readers due to lack of context.
7. [Minor] Section 3 can be merged with Section 6.1.
8. [Minor] The caption of Figure 1 is hard to understand.

### Soundness
2

### Presentation
2

### Contribution
3

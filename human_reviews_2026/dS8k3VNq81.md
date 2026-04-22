# Oh SnapMMD! Forecasting stochastic dynamics beyond the Schrödinger bridge’s end

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Scientists often want to make predictions beyond the observed time horizon of “snapshot” data following latent stochastic dynamics. For example, in time course single-cell mRNA profiling, scientists have access to cellular transcriptional state measurements (snapshots) from different biological replicates at different time points, but they cannot access the trajectory of any one cell because  measurement destroys the cell. Researchers want to forecast (e.g.) differentiation outcomes from early state measurements of stem cells. Recent Schrödinger-bridge (SB) methods are natural for interpolating between snapshots. But past SB papers have not addressed forecasting. Some natural immediate extensions of existing methods would (1) reduce to following pre-set reference dynamics (chosen before seeing data) or (2) require the user to choose a fixed, state-independent volatility since they minimize a Kullback–Leibler divergence. Either case can lead to poor forecasting quality. In the present work, we propose a new framework, SnapMMD, that learns dynamics by directly fitting the joint distribution of both state measurements and observation time with a maximum mean discrepancy (MMD) loss. Unlike past work, our method allows us to infer unknown and state-dependent volatilities from the observed data. We show in a variety of real and synthetic experiments that our method delivers accurate forecasts. Moreover, our approach allows us to learn in the presence of incomplete state measurements and yields an $R^2$-style statistic that diagnoses fit. We also find that our method's performance at interpolation (and general velocity-field reconstruction) is at least as good as (and often better than) state-of-the-art in almost all of our experiments.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper considers the problem of forecasting stochastic dynamics from snapshot data, such as single-cell RNA-seq or ecological systems, where only population-level states at discrete times are observed. The authors propose SnapMMD, a framework that learns both drift and state-dependent diffusion of an SDE by directly matching the joint distribution of states and times using a Maximum Mean Discrepancy (MMD) loss. Unlike Schrödinger-Bridge (SB) methods that minimize a KL divergence to a fixed reference process, SnapMMD dispenses with the reference dynamics, allowing the g_\theta(x,t) to be learned from data. It also introduces an RKHS-based R^2 statistic for model fit and can handle partially observed states.

### Strengths
- Clear motivation for moving beyond interpolation toward genuine forecasting.

### Weaknesses
1. Confounded gains from structural priors vs. objective/volatility learning. Only SBIRR is configured to incorporate the same drift-structure prior as SnapMMD, whereas several interpolation/generative baselines cannot accommodate comparable inductive bias. Consequently, observed improvements may partly reflect unequal access to the mechanistic structure rather than the MMD objective or the learned, state/time-dependent diffusion.
2.  Volatility advantage may be overstated without tighter controls. Baselines are run with fixed, known diffusion (e.g., $\sigma=0.1$), while SnapMMD learns state/time-dependent volatility; the paper does not provide matched ablations using fixed $\sigma$ values calibrated to the learned diffusion’s empirical scale (e.g., mean over states/times). This leaves open whether forecast gains stem from better volatility calibration rather than the genuine benefits of state dependence.
3. Partial-observability claims rely mainly on small-scale simulations. The “mRNA+protein” advantage is demonstrated on synthetic data; real high-dimensional experiments (e.g., PBMC) do not include additional unobserved variables, so practical utility in complex biological settings remains under-substantiated.
4. Identifiability is not guaranteed and is acknowledged by the authors. Even full time series of marginals may not uniquely determine drift and diffusion; without strong priors, learned $(b_\theta,g_\theta)$ can be non-unique up to observational equivalence, limiting mechanistic interpretability.
5. Metric and kernel dependence may bias training/evaluation. The method and primary metric rely on RBF-MMD with median-heuristic bandwidth; the paper notes cases (Gulf of Mexico) where MMD prefers more diffuse point clouds despite inferior geometric fidelity, and reports discrepancies between MMD and EMD. This raises concerns about sensitivity to kernel choice and bandwidth in both optimization and evaluation.
6. Simulation budget and sample-imbalance handling lack principled guidance. Training minimizes a weighted sum of per-time MMDs using U-statistic estimates from M simulated paths per time, but the paper provides no principled selection or diagnostics for M or for mitigating time-wise sample-size imbalance beyond the fixed weighting scheme.
7. Baseline coverage and parity are incomplete. Empirical comparisons omit additional strong or structure-compatible baselines (e.g., neural-ODE variants using the same mechanistic drift, or other recent trajectory inference methods, e.g., JKOnet (Terpin et al., NeurIPS 2024)), which limits claims of superiority across design spaces and problem regimes.
8. Compute scalability remains a practical concern. The approach is not simulation-free; training cost scales with sample size and dimensionality. The paper acknowledges this but does not quantify scaling behavior in large real-world settings.

### Questions
1. Disentangling sources of gain. Could you discuss ablations that (i) remove the mechanistic drift prior (use fully-neural b_\theta), (ii) keep the prior but fix diffusion, and (iii) keep the prior and learn diffusion, so that improvements can be attributed separately to structural priors, the MMD objective, and state/time-dependent diffusion? Please report both forecasting and interpolation metrics.  
2. Volatility calibration controls. What are the empirical summaries (e.g., mean/variance over (x,t)) of the learned $g_\theta(x,t)$? If baselines use a fixed $\sigma$, how do results change when $\sigma$ is set to the learned diffusion’s mean (or a time-only or piecewise-constant surrogate)? Please include visual/quantitative comparisons of trajectory spread vs. fixed-\sigma surrogates.
3. Identifiability scope. Under what parametric restrictions or priors can parts of $(b_\theta,g_\theta)$ be identified from marginals? Could you formalize identifiable functionals even if b,g are not fully unique?  
4. Metric/kernel sensitivity. How sensitive are training and model selection to kernel choice and bandwidth (median heuristic) in the MMD?
5. Simulation budget M and imbalance. What principles guide the choice of the number of simulated paths M per time point ?
6. Coverage of recent SOTA. Could you include additional trajectory-inference baselines (e.g., JKOnet)?
7. Forecast horizon robustness. How does forecasting error grow with horizon length beyond the last snapshot? 
8. Could you quantify runtime/memory vs. dimension and dataset size?

I believe this paper considers an important problem—forecasting from population snapshots. However, several aspects require more careful justification. I therefore lean toward a borderline rejection at this stage, but I would be open to reconsidering if the authors address the above issues with convincing rebuttals.

References
1. Terpin, Antonio, et al. "Learning diffusion at lightspeed."NeurIPS 2024.

The reviewer wrote this review. The LLM was used only to improve both grammar and clarity.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper studies forecasting stochastic dynamics from population snapshots (unpaired observations at discrete times). The main idea is to learn SDE parameters by matching the *joint* state–time distribution between model and data via an RKHS MMD. With a time–separable kernel, the joint MMD decomposes into a weighted sum of per–time MMDs, yielding a least–squares style fit across times. The method, SnapMMD, jointly learns drift and *state–dependent* volatility and can embed mechanistic structure through parametric SDE families. An RKHS $R^2$ diagnostic is introduced using a barycenter baseline. Experiments on synthetic and real systems report strong forecasting and competitive interpolation.

### Strengths
* **(S1) Volatility learning.** Ability to estimate state–dependent diffusion is a practical advantage in scientific systems with heterogeneous noise.

* **(S2) Structural priors.** Parametric SDE families let users encode mechanistic knowledge; ablations show gains when structure is informative.

* **(S3) Diagnostics.** The MMD–based $R^2$ is simple and useful for model selection.

### Weaknesses
* **(W1) Simulation cost and trajectory of the field.** The method is not simulation–free and requires path simulation and adjoint backprop through an SDE solver. The paper acknowledges this as a primary limitation. Meanwhile, the literature in 2024–2025 has rapidly advanced simulation–free alternatives for snapshot matching and forecasting; relative to that trend, this work doubles down on a simulation–based line with high compute burden.

* **(W2) Data efficiency and compounding error.** Training subsamples unpaired points from each time marginal. Across times, this can compound sampling noise and induce temporal mismatch; performance may degrade unless $N$ is very large and the distribution heterogeneity is not increasing over time.

* **(W3) Model misspecification sensitivity.** Performance depends on the chosen SDE family; fully neural SDEs underperform, indicating reliance on good inductive bias.

* **(W4) Horizon robustness.** Forecasting beyond short horizons is not stress–tested; failure modes with increasing extrapolation time are unclear.

* **(W5) Kernel and weighting sensitivity.** Results and $R^2$ depend on $K_y$ and bandwidth; the time–kernel choice induces squared snapshot–frequency weights, which may over–emphasize heavy–sampled times.

* **(W6) Identifiability under partial observation.** Without an explicit measurement model, recovering latent coordinates and diffusion can be weakly identified.

### Questions
1. **Compute and scaling.** Please report wall–clock and asymptotic scaling in state dimension $d$, number of times $|\mathcal{T}|$, and samples $N$. Any variance–reduction or multi–trajectory reuse to lower gradient noise?

2. **Simulation–free baselines.** How does SnapMMD compare against recent simulation–free methods on the same datasets and metrics, especially in compute and sample efficiency?

3. **Subsampling effects.** Can you quantify the bias/variance impact of unpaired subsampling across times, and whether stratified or control–variate schemes reduce temporal mismatch?

4. **Long–horizon stress test.** Please evaluate error as the forecast horizon increases and characterize failure modes.

5. **Kernel and weighting robustness.** Sensitivity of results and $R^2$ to $K_y$, bandwidth, and to replacing $\delta(t-t')$ with a normalization that yields linear (rather than squared) time weights.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a method for trajectory inference of *both* the drift vector field *and* the state- and time-dependent diffusion coefficients for an Ito SDE which approximates the temporal evolution of marginal distributions observed via independent 'snapshot' samples.   While Schrodinger Bridge methods are customarily applied to these problems, the authors propose a conceptually different approach by (i) parameterizing the drift & diffusion coefficients of the stochastic dynamics, (ii) calculating the MMD loss between generated and observed samples, and (iii)  backpropagating through the parameterized stochastic dynamics using the stochastic adjoint method.    This approach allows for more flexible incorporation of prior knowledge of the dynamics via the chosen dynamics parameterization, and appears to be more amenable to forecasting beyond the temporal horizon of observed samples.

### Strengths
Recent methods for trajectory inference have perhaps over-indexed to Schrodinger Bridge (SB) methods given their mathematical connections with popular diffusion models.   While ingredients of the proposed method are not new in themselves, overall, the authors provide more refined approach to the particular problem at hand.  

Backpropagation through (stochastic) dynamics (Li et. al 2020, "Scalable Gradients for SDEs") may not be the method of choice for large-scale generative modeling, but allows the authors to directly incorporate prior knowledge using *informed variational parameterizations* in trajectory inference.
- Previous methods may introduce 'prior knowledge' via the reference drift of the SB problem or a state-cost in optimal control problems (Generalized SB, Neklyudov et. al 2023, "Wasserstein Lagrangian Flows"), but these approaches are arguably less direct and natural than the proposed approach (as the authors argue).

The authors demonstrate empirical benefits from learning the diffusion coefficient, which is indeed underexplored in the literature and is naturally accommodated in the proposed approach.
- Authors may be interested in (concurrent) Guan et. al arxiv 05.2025 "Gradient-flow SDEs have unique transient population dynamics" discussing identifiability of diffusion coefficient inference.

The authors frame the exposition around forecasting or extrapolation beyond the observed samples, which is clearly an important problem not naturally handled by existing methods.   Results in Fig 1-3 show improved performance in this respect, although I have questions below.


I appreciate the extensive experimental results in the Appendix, and look forward to the authors having extra space to incorporate some of these in a camera-ready version.

### Weaknesses
The exposition of the paper is not sufficiently clear in several crucial ways.

*Parametric Components / `Variational Family of SDEs'*

> L241: "We pick the candidate conditional fθ (·|t) based on domain knowledge by specifying parametric components of the SDE in every experiment, reflecting the partial mechanistic understanding available in scientific settings."

I understand this point to reflect a `variational family of SDEs', where the drift and diffusion have learnable coefficients.   These details may be difficult to specify due to different parameterizations across settings, but this is a major point in the exposition and understanding of the method.   This needs to be clarified with concrete equations or an example in the main text.
- Are *all* experiments within known variational families of SDEs?
- In which cases do we expect the parameters to be time-dependent? (see Questions below)


*Missing dimensions*
> L79-80:  Our framework offers the added benefit of enabling robust interpolation and forecasting even with incomplete state measurements, as in the protein expression example above.

> L250-253: "More precisely, since our loss (Eq. (4)) is defined over the observed state variables (together with time), the model is trained to match the marginal distribution of the observed variables along with time, without making any additional assumptions or imputations for the missing dimensions.  (*note: duplicate / repeated clauses*)

This point was not at all clear to me until inferring the parameterization point above and inspecting App F.8.     From the current exposition, the reader might wonder (i) why the unobserved dimensions are necessary at all? or (ii) how do they receive supervision from the data?
- Having established the informed variational family of SDEs in the main text, this point can be explained by the fact that the dynamics of observed dimensions are coupled with (parameters, dynamics) of unobserved dimensions within the given family (?)




*Schrödinger Bridge Example and Focus*

I found the reference to App B1 in L138-141 to be distracting, and did not find the eventual argument in App. B1 to be convincing.
- It is clear that the (multimarginal) SB problem is underdetermined beyond the final marginal.  
- As the authors point out, extrapolation via the reference is only one possible resolution.  
- The authors use extrapolation of learned vector fields (with $t > t^{\text{observed}}_{\text{max}}$?) for some baselines, which is  most natural but can be easily argued to lack any concrete mechanism for matching true extrapolating dynamics
- with a learned reference within a known family (SBIRR, see below), the extrapolation via the reference seems reasonable

That said, I was left wondering **what are the actual mechanisms by which the proposed method achieves improved extrapolation?** (see below).

### Questions
*Mechanisms for Extrapolation*
> L75-76:  "Our framework allows data to guide extrapolation beyond observed time"
> L68-69:  "[proposed method] shifts modeling focus from interpolation to accurate... forecasting"
 
At first glance, I was quite confused by these statements given that 
- The method is never trained at extrapolating times
- The $\delta_0(\mathbf{y})$ conditional in Eq 2 is not specified but also never appears to be used (?)


Instead, it seems *much more likely* that the forecasting extrapolation benefits are due to accurate parameter estimation within an appropriate choice of variational family.  

- If the parameters within the given variational family of SDEs were time-independent, then we would expect to learn these parameters well with a reasonable amount of data.   These would naturally extrapolate by simply running the parameterized dynamics from $X_T$
- If the parameters were time-dependent but only slowly changing over time, we might still expect to get reasonable performance by simulating using parameter estimates at time $T$ forward in time.
- by contrast, the solution to SB problems is fundamentally time-dependent in most interesting cases, which should threaten extrapolation (especially combined with time-conditioned NN vector fields).

I'm tentatively positive on the paper, but with the remaining concerns above as the emphasis and performance on forecasting did not fully compute for me.

*SB Exposition and Baselines* (more speculative)

Given my previous points, I was also distracted by the use of Schrödinger Bridge in the title given that this is not an SB method.   
- I think exposition comparing with the most similar SB-method (SBIRR?) would be useful, where the known family of SDEs is used to update the reference path measure and we optimize over the first argument with a more flexible family.  
    - the authors could emphasize why this cannot capture learned volatility  
- Would a not-quite-SB baseline be to simply swap out MMD for the MLE KL ($\mathbb{E}$ over data, $\log$ prob over parameterized `reference' SDE family)?   
    - This essentially swaps out MMD for MLE KL, and would seemingly accommodate learning the volatility using a likelihood approximation (as in SBIRR) and backprop through time/dynamics (as in this work).  May require working in discrete time.
- Again, in this case, extrapolation using learned dynamics within an appropriate family seems quite reasonable.

If any of this makes sense, it might help to navigate the design space of algorithms and motivate the choices made.   Again, not crucial, this is just my personal random walk from SB => SBIRR => this method.


**Minor comments / questions** (no need to respond):


- Clarify domain of $g_0$ (vector/diagonal matrix?  more general?)  Why are we using 0-subscripts?

- Is it ok to use $\hat{f}(y|t)$ notation in Eq. 2?   It's used in $f_\theta(y|t)$ anyway, and L164 is hard to read.

- The conditional/joint notation appears to be inconsistent in App. B1, L727-747.   It seems like you mean $q_{t_T, t_{T+1}} = \pi_T p_{t_{T+1}|t_T}$ instead of  $q_{t_T, t_{T+1}} = \pi_T p_{t_T, t_{T+1}}$ (i.e. $q_{t_{i},t_{i+1}}$ for joint distribution on LHS, whereas $p_{t_i, t_{i+1}}$ is used for conditionals in stated notation and RHS).

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work introduces SnapMMD, a framework for reconstructing underlying dynamics from snapshot data by directly fitting the joint distribution of state and time using an RKHS MMD loss with a factorized kernel. This kernel choice enables the decomposition of this MMD into a least-squares sum over time steps; hence, training can be effectively interpreted as a weighted sum of time-marginal MMDs between simulated and empirical snapshots. Experimentation on synthetic and real datasets exhibit strong one-step-ahead forecasting and competitive or better interpolation.

### Strengths
- Problem & motivation: This work tackles forecasting from snapshot data, which is a central and integral problem in a variety of disciplines.  The authors clearly explain their motivation and why current SB-based methods, although strong for interpolation, often struggle in extrapolation.
- Clarity: The paper is well structured, readable, and easy to follow.
- Novelty: The theoretical analysis in the paper is sufficiently thorough. This work treats snapshots as a joint (state, time) distribution and optimizes a kernel MMD loss, resulting in an original and novel formulation.
- Experimental evaluation: The authors present an extensive experimental evaluation of their algorithm with consistently strong results in both extrapolation and interpolation across tasks.

### Weaknesses
- Model-class dependence. The approach assumes a (semi-)parametric family for the dynamics; both performance and identifiability hinge on this choice. Clearer guidance or guarantees on when drift/volatility are recoverable—especially under misspecification—would strengthen the claims.
- Sensitivity to reference dynamics. Following the previous point, the fully non-parametric setting (Sec. 4.6) exhibits a marked performance drop, suggesting that the method’s success is tied to a strong reference dynamics class. An ablation on reference capacity/inductive bias (and its interaction with kernel choices) would clarify robustness.

### Questions
- Multi-step forecasting. It would be interesting to see how error accumulates over multiple future steps (e.g., 2, 3, etc)? Do the authors have any intuition if the method is still stable if attempting to forecast multiple steps, and what future adjustments could improve longer horizon stability?

- Baselines. In addition to the current baseline methods, is it possible to compare your methodology to algorithms specialized in time series performing forecasting and imputation, such as [1], [2]? Additionally, it would be interesting if you could show similar strong one-step extrapolation capabilities in time series datasets. For this, you could use the synthetic ones used in [2].

- Reference dynamics sensitivity. It was shown that the fully non-parametric variant (Sec. 4.6) underperforms. Is it possible for your method to leverage external neural representations? For example, is it possible to use a pretrained model from models that solve the multi-marginal SB or the mmFlow Matching problem (e.g., [3, 4, 5]) as the reference? Does this mitigate the dependence on a parametric reference class? This could be particularly useful in applications where the reference dynamics are not known.

- Compute & stability. What are the compute costs and stability characteristics for long rollouts (e.g., step size, simulator variance, number of paths per time) across datasets?

- Kernel sensitivity. Could you perform an ablation study to demonstrate how sensitive the results are to kernel family and bandwidth?

--- 
References

[1] FM-TS: flow matching for time series generation

[2] Provably Convergent Schr¨ odinger Bridge with Applications to Probabilistic Time Series Imputation

[3] Deep Momentum Multi-Marginal Schrödinger Bridge

[4] Multi-Marginal Stochastic Flow Matching for High-Dimensional Snapshot Data at Irregular Time Points

[5] Modeling complex system dynamics with flow matching across time and conditions.

### Soundness
3

### Presentation
3

### Contribution
2

000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053 We introduce **Adaptive World Models for Data-Efficient Learning (AWML)**, a framework that combines structured latent world models, certified counterfactual augmentation, and calibrated uncertainty filtering to improve sample efficiency in low-data regimes. AWML firstly learns modular latent dynamics under domain priors. Secondly, it generates counterfactuals through modular recombination. Finally, it accepts synthetics only when an uncertainty estimator satisfies a calibrated acceptance condition.

Theory. - **Modular amplification (Thm. 3.5):** estimation error scales as

O

![0_image_0.png](0_image_0.png)

 with additive bias 2D from per-module TV deviations.

- **Uncertainty filtering (Thm. 3.6):** a pointwise calibration bound |q(τ ) − p(τ )| ≤ L U(τ )
yields the deployment-level control

$$\mathrm{TV}(P_{\mathrm{aug}},P)\leq{\frac{B}{N+1}}$$
N + B
L u + ε,
Together these results give a unified excess-risk guarantee (Cor. 3.9) that makes explicit the bias–variance trade-off governed by estimation quality, the acceptance threshold u, and the accepted mass B. Algorithm. AWML pairs neural-operator backbones with modular causal blocks and safeguards (ensemble calibration, denominator clamping, diagnostic audit flags). Results. Synthetic AR(1) studies show consistent RMSE reductions (Ridge: 0.227 → 0.219; MLP: 0.253 → 0.233). On the Uganda LSMS 2019 dataset, AWML yields substantial AUC gains in low-label regimes (n = 25: 0.8797 → 0.9402) under conservative TV diagnostics. Conclusion. AWML provides provable conditions for safe augmentation together with practical diagnostics that indicate when augmentation should stop or be audited.

## 1 Introduction

Modern ML often assumes access to large labelled datasets. Many important domains do not offer such scale, including low–resource languages, small clinical cohorts, and sparse Earth and climate observations. In these settings, models face high sample complexity and persistent spurious patterns, and they degrade quickly under shift (Guo et al., 2017). We need methods that use structure to gain statistical efficiency and remain reliable when data are limited. We introduce *Adaptive World Models for Data-Efficient Learning* (AWML). AWML brings together four ideas.

Anonymous authors Paper under double-blind review

## Abstract

which directly bounds excess risk.

1

# Adaptive World Models For Data-Efficient Learning

054 055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107

## Contributions.

1. We present AWML as a unified framework with structured latent models, modular counterfactual generation, calibrated filtering, and adaptive transfer across environments.

2. We derive finite-sample bounds. These include generalization improvements from structured priors, a modular amplification bound with variance of order N
−1/2 eff , and a certified acceptance bound with bias governed by Q(*U > u*) + u.

3. We provide a practical algorithm that combines neural-operator or physics-aware layers, modular latent blocks, a counterfactual generator, and a calibrated acceptance filter.

4. We validate the framework in synthetic and real low-label settings. Synthetic studies match the predicted N
−1/2 eff scaling. A real low-label study demonstrates the trade-off predicted by our theoretical bounds.

Positioning. Meta-learning and self-supervised approaches reduce labelled data requirements (Finn et al., 2017; Chen et al., 2020), yet they often rely on transfer heuristics and unstructured augmentation. AWML reduces hypothesis complexity through structured priors, increases effective sample size through modular recombination, and certifies augmentation using calibrated accept–reject rules.

## 1.1 Related Work

Data-efficient learning. Many approaches aim to improve performance when labeled data are limited. Meta-learning methods help models adapt from a few examples (Finn et al., 2017). Largescale self-supervised models learn features that transfer well and reduce label demands (Devlin et al., 2019; Chen et al., 2020). These strategies often rely on weak structural assumptions and do not provide guarantees under distribution shift. AWML complements them by adding structure in the latent space and by using certified filtering during augmentation. Latent dynamics and world models. Latent world models learn compact representations of system dynamics and support prediction and planning (Ha & Schmidhuber, 2018). Neural differential equation models provide continuous-time formulations for similar tasks (Chen et al., 2018). AWML builds on this line of work by making the latent space modular so that parts of the dynamics can be recombined to form counterfactual rollouts. Neural operators. Neural operators learn mappings between function spaces and offer strong performance for scientific and physics-informed learning (?Kovachki et al., 2023; Raissi et al., 2019). These models help encode structural priors such as invariants or operator form. AWML uses such priors inside the latent transition model to reduce hypothesis complexity.

1. It learns a latent world model with structured priors such as modularity, invariants, or operator-form structure. These ideas build on world models and neural ODEs for latent dynamics (Ha & Schmidhuber, 2018; Chen et al., 2018).

2. It generates counterfactual samples by recombining latent modules in a way motivated by structural causal models (Pearl, 2009; Imbens & Rubin, 2015; Kusner et al., 2017).

3. It filters synthetic data using calibrated uncertainty, supported by results on neural calibration (Guo et al., 2017) and conformal prediction (Romano et al., 2019).

4. It separates priors into transferable and mutable parts to support adaptive transfer across environments.

AWML relates to several established lines of work. Neural operators offer efficient, physics-aware mappings between function spaces (Li et al., 2020; Kovachki et al., 2023; Raissi et al., 2019), and AWML uses these ideas inside its structured latent model. Causal representation work highlights the value of modular latent factors (Scholkopf et al., 2021). Meta-learning and self-supervised ¨ pretraining reduce label needs (Finn et al., 2017; Devlin et al., 2019; Chen et al., 2020), but they often rely on weak priors or heuristic transfer. AWML complements these approaches by adding structure, controlled augmentation, and certified acceptance. Causal and counterfactual reasoning. Structural causal models provide a formal language for interventions and counterfactuals (Pearl, 2009; Imbens & Rubin, 2015). Counterfactual generation has been explored in fairness and representation learning (Kusner et al., 2017; Scholkopf et al., 2021). ¨ AWML adopts this perspective by treating modular interventions in the latent state as counterfactual edits that follow the learned causal structure. Modularity and disentanglement. Modular and disentangled representations support transfer and improve generalization (Locatello et al., 2019). Weakly supervised signals can also encourage modular structure (Locatello et al., 2020). AWML uses modularity to isolate parts of the latent dynamics so that they can be recombined across trajectories. Uncertainty and calibration. Reliable uncertainty estimates are important when using synthetic data. Temperature scaling improves calibration in deep networks (Guo et al., 2017). Conformal prediction provides finite-sample coverage guarantees (Romano et al., 2019). AWML combines these tools to filter synthetic trajectories based on calibrated uncertainty.

108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 Transfer and domain adaptation. Transfer learning studies how information moves across related distributions (Sugiyama et al., 2007). Transportability theory analyzes when causal knowledge can be reused across environments (Bareinboim & Pearl, 2012). AWML follows this principle by separating priors into parts that remain stable and parts that adapt to each environment.

## 2 Preliminaries And Problem Setup

We study a family of related environments E. Environments are related in the sense that they share state, action, and observation spaces and a common invariant structure, for example a conservation law or symmetry, but they differ in parameters.

Each environment E ∈ E generates latent states st ∈ S, actions at ∈ A, and observations ot ∈ O
for times t = 1*, . . . , T*. The dynamics and observations follow pE(st+1 | st, at), pE(ot | st), pE(s1).

Unless stated otherwise, actions come from a policy πE(at | o1:t).

The learner observes a small factual dataset DE = {(ot, at, ot+1)}
N
t=1, N ≪ benchmark scale, and must return a compact model that transfers across environments in E and remains reliable when augmented with synthetic data.

Goals. Our goals are threefold. First, learn an encoder ϕ: O → R
dthat maps observations to latent states zt. Second, fit latent transition and emission models pθ(zt+1 | zt, at) and pθ(ot | zt).

Third, generate synthetic examples whose inclusion improves downstream performance while keeping augmentation bias under control. We use standard learning-theory notation such as hypothesis class H, empirical risk Rbn, Rademacher complexity Rn(H), and covering numbers N (H, ε) (Mohri et al., 2018; Bartlett & Mendelson, 2002). Latent world-model backbone. We use a compact latent-variable sequence model trained with an ELBO or a similar variational objective. The joint model has the form

$$p_{\theta}(o_{1:T},z_{1:T}\mid a_{1:T})=\prod_{t=1}^{T}p_{\theta}(o_{t}\mid z_{t})\,p_{\theta}(z_{t}\mid z_{t-1},a_{t-1}).$$
$$(1)$$

pθ(ot | zt) pθ(zt | zt−1, at−1). (1)
Domain priors such as modularity, invariants, and operator structure reduce effective complexity and promote latents that can be recombined and transferred across environments (Ha & Schmidhuber, 2018; Hafner et al., 2020; Chen et al., 2018; Li et al., 2020; Kovachki et al., 2023; Raissi et al., 2019). Modularity and recombination. We assume a modular latent representation 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215

$$R(\widehat{h}_{\mathcal{P}})\leq\operatorname*{inf}_{h\in\mathcal{H}_{\mathcal{P}}}R(h)+2\,\Re_{n}(\mathcal{H}_{\mathcal{P}})+\sqrt{\frac{\log(1/\delta)}{2n}}.$$
2n. (4)
zt =z
(1)
t*, . . . , z*
(M)
t.

The transition is approximately factorized in a local sense

$\eqref{eq:walpha}$. 
4
$$p_{\theta}(z_{t+1}\mid z_{t},a_{t})\approx\prod_{m=1}^{m}p_{\theta}^{(m)}\bigl(z_{t+1}^{(m)}\mid z_{t}^{(\mathrm{pa}(m))},a_{t}\bigr),$$
M
.)  $\star\star\star$ . 
$$(2)$$
$$\mathbf{\partial}\cdot\varepsilon_{t},\qquad\varepsilon_{t}\sim{\mathcal{N}}(0,\Sigma).$$
t, at, (2)
where pa(m) is a small parent set of modules for module m. Modules learned from data can then be recombined across trajectories to form new rollouts. This recombination increases effective sample size but introduces additive augmentation bias through per-module errors. We control this bias using uncertainty-aware acceptance and by restricting module complexity (Locatello et al., 2019; 2020).

Counterfactuals. We use the term counterfactual in an operational sense inspired by structural causal models (Pearl, 2009; Imbens & Rubin, 2015; Kusner et al., 2017). A counterfactual example is a synthetic trajectory obtained by intervening on one or more modules in the learned latent model and then rolling out the dynamics under this intervention. Concretely, we replace the update rule for a chosen module while holding other modules and the policy fixed. We then check the validity of such trajectories using model uncertainty scores. Structured transition parameterization. To encode physics or other domain structure we parameterize transitions as zt+1 = hθ(zt, at) + εt, εt ∼ N (0, Σ). (3)
The map hθ uses modular neural blocks and, when appropriate, neural-operator components (Li et al., 2020; Kovachki et al., 2023; Raissi et al., 2019). Architectural constraints and penalty terms enforce invariants and define a structured hypothesis class HP ⊆ H with lower complexity and stronger finite-sample behavior. Uncertainty filtering and certified acceptance. Let T be the pool of candidate synthetic trajectories generated by modular recombination. Let U : T → R+ be an uncertainty score, for example ensemble variance, a conformal score, or predictive entropy (Guo et al., 2017; Romano et al., 2019). We accept a candidate trajectory τ only when U(τ ) ≤ u, where the threshold u is chosen by crossvalidation or a data-dependent rule. If U is calibrated so that it upper-bounds a divergence between synthetic and factual distributions on accepted samples, then discarding high-uncertainty trajectories yields an augmented dataset with tunable and provable bias. Assumptions and scope. Section 3 states the precise assumptions on mixing, per-module estimation error, and noise tails. Implementation choices and diagnostics appear in Appendix A. Key notation is as follows. E is the family of environments, N is the factual sample size, zt is the latent state, M is the number of modules, HP is the structured hypothesis class, T is the synthetic candidate set, and U is the uncertainty score.

## 3 Generalization, Modular Amplification, And Certified Augmentation

We now state the main theoretical results. First, structured priors reduce hypothesis complexity and improve generalization. Second, modular recombination increases effective sample size at a controlled bias. Third, thresholded uncertainty converts generator bias into a tunable deployment bound. We give proof sketches here and provide full proofs with constants in Appendix A.

Setup. Losses are bounded, so ℓ ∈ [0, 1]. Let P be the target data distribution. Let HP ⊆ H be the hypothesis subclass induced by structure, for example invariants or operator form. Let bhP be an empirical risk minimizer over HP .

Theorem 3.1 (Generalization under structured priors). Let n samples be drawn i.i.d. from P *and let* δ ∈ (0, 1)*. With probability at least* 1 − δ, This shows that structure helps whenever it shrinks the Rademacher complexity Rn(HP ).

216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 Sketch. We control the deviation R(h) − Rbn(h) uniformly over h ∈ HP . Symmetrization bounds this deviation in terms of the empirical Rademacher complexity. Standard concentration then yields the high probability term in equation 4; see Mohri et al. (2018); Bartlett & Mendelson (2002).

Then Sketch. Write the total variation distance as

$$\mathrm{TV}(p,q)=1-\int\operatorname*{min}\{p,q\}.$$

For product measures, the pointwise minimum factors as the product of per-module minima. Each module contributes at least a factor (1 − δm) by the assumption on its total variation. Multiplying these contributions and integrating yields the bound.

$$\mathrm{TV}(p,q)\leq1-\prod_{m=1}^{M}(1-\delta_{m}).$$

This bound tracks how small per-module discrepancies aggregate into a global generator bias.

Lemma 3.3 (Risk shift via total variation). Let P and Q be distributions and let f be measurable
with ∥f∥∞ ≤ 1*. Then*
$$\left|\mathbb{E}_{P}[f]-\mathbb{E}_{Q}[f]\right|\leq2\operatorname{TV}(P,Q).$$
In particular, for losses ℓ ∈ [0, 1], |RP (h) − RQ(h)| ≤ 2 TV(*P, Q*).

This shows that total variation directly bounds the worst-case change in risk when we move from P to Q. Sketch. Write the difference of expectations as

$(h)-R_{\mathrm{O}}(h)|\leq2\,\mathrm{TV}(P_{\mathrm{c}})$
$$\mathbb{E}_{P}[f]-\mathbb{E}_{Q}[f]=\int f\,\mathrm{d}(P-Q).$$
Bound the absolute value using ∥f∥∞ ≤ 1 and the definition of total variation distance as the supremum over bounded test functions; see Gibbs & Su (2002).

Lemma 3.4 (Uniform convergence under a generator). Let RbNeff *be the empirical risk from* Neff i.i.d. samples drawn from a generator Q. Let H *be a hypothesis class with covering numbers* N (H, ε) and let β ∈ (0, 1). There exists an absolute constant C *such that, with probability at* least 1 − β,

$$\operatorname*{sup}_{h\in\mathcal{H}}\left|R_{Q}(h)-\widehat{R}_{N_{\mathrm{eff}}}(h)\right|\leq C\sqrt{\frac{\log\mathcal{N}(\mathcal{H},\varepsilon)+\log(1/\beta)}{N_{\mathrm{eff}}}}+\varepsilon.$$

This is the standard covering number bound that captures the variance term due to finite Neff.

Sketch. Construct an ε-net for H under the L1(Q) metric. Apply Hoeffding's inequality to each function in the net and take a union bound. Extend the bound from the net to all of H using the triangle inequality; see Mohri et al. (2018). Bias bookkeeping for modular generators. We next control how per-module errors combine and how they shift risk.

Lemma 3.2 (Product total variation bound). Let p =QM
m=1 pm and q =QMm=1 qm. Suppose that for each module m *and each conditioning* x,

$$\operatorname*{sup}\mathrm{TV}(q_{m}(\cdot\mid x),p_{m}(\cdot\mid x))$$
x
TVqm(· | x), pm(· | x)≤ δm.
270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323

$$\begin{array}{l}{{s a c n\;m a,\;w i n\;p r o b a u n i y\;a i\;e u s i\;\Gamma=\beta,}}\\ {{\quad R_{P}(\widehat{h})-R_{P}(h^{\star})\leq C\sqrt{\frac{\log{\mathcal{N}}({\mathcal{H}},\varepsilon)+\log(1/\beta)}{N_{\mathrm{eff}}}}+2D+\varepsilon,}}\\ {{\quad h\in{\mathcal{H}}\;R_{P}(h).}}\end{array}$$
Neff+ 2D + ε, (5)
This bound makes the trade off explicit. More modular recombination reduces the variance term through larger Neff, while per-module errors increase the bias through D. Sketch. First, view the generator Q as an approximation to P obtained by composing the per-module estimators. Lemma 3.2 bounds TV(*P, Q*) by D. Lemma 3.3 then shows that the induced shift in risk is at most 2D. Finally, apply Lemma 3.4 to control the estimation error from training on Neff samples drawn from Q. Combining these pieces yields equation 5. From generators to deployment: certified acceptance We now show how thresholding by an uncertainty score reduces the bias from synthetic data. Assumption 3.6 (Pointwise calibration for acceptance). *There is a nonnegative discrepancy* d :
T → R+ such that for any measurable f with |f| ≤ 1,
EP [f] − EQ[f] ≤ EQ[d].

The acceptance score U satisfies U(τ ) ≥ d(τ ) *almost surely for* τ ∼ Q.

This assumption says that U upper bounds a per-sample discrepancy that controls the shift between P and Q.
Definition 3.7 (Thresholded generator). For u ≥ 0, define the accepted set Au = {τ ∈ T : U(τ ) ≤ u} *and the conditional generator* Qu(·) = Q(· | Au). Theorem 3.8 (Certified acceptance reduces bias). Let Assumption 3.6 hold. For any u ≥ 0 *and any* |f| ≤ 1,
$$\left|\mathbb{E}_{P}[f]-\mathbb{E}_{Q_{u}}[f]\right|\leq2\,Q(A_{u}^{\mathbb{C}})+2u.$$
For losses ℓ ∈ [0, 1] *this gives*
|RP (h) − RQu(h)| ≤ 2 Q(U > u) + 2u.
I am not sure if $\alpha$ is 10-11 $\alpha$. 

This bound replaces an opaque generator bias by a quantity that depends only on the acceptance threshold u and the tail Q(*U > u*).

Sketch. Write the expectation under Q as a mixture over Au and A∁uand then condition on Au. On Au, U ≥ d implies d(τ ) ≤ u, so the expected discrepancy is at most u. On A∁u, losses are in [0, 1], so the contribution is at most Q(A∁u
). Combine these two parts and use the same argument as in Lemma 3.3 to relate discrepancy and risk. When U comes from a conformal construction, Q(*U > u*) is controlled by a finite sample coverage guarantee (Romano et al., 2019).

Corollary 3.9 (Deployment bound for AWML). *Train ERM* bhu on Neff accepted samples from Qu.

For any β ∈ (0, 1)*, with probability at least* 1 − β,

$$R_{P}(\widehat{h}_{u})-R_{P}(h^{\star})\leq C\sqrt{\frac{\log\mathcal{N}(\mathcal{H},\varepsilon)+\log(1/\beta)}{N_{\rm eff}}}+2\,Q(U>u)+2u+\varepsilon,$$

where C *is absolute.* where h
⋆ = arg minh∈H RP (h).

Theorem 3.5 (Certified modular data amplification). Assume the modular factorization equation 2.

From N factual trajectories, estimate per-module conditionals pbm *that satisfy* sup x TVpbm(· | x), pm(· | x)≤ δm.

Define the aggregate generator bias

$$D:=1-\prod_{m=1}^{M}(1-\delta_{m}).$$
$$(S)$$

Let Q be the product generator formed from the pbm*, and draw* Neff i.i.d. samples from Q*. Let* bh be the empirical risk minimizer trained on these Neff samples. Then for any β ∈ (0, 1) there is an absolute constant C such that, with probability at least 1 − β, This combines the variance term from Lemma 3.4 with the bias term from Theorem 3.8.

324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377 Then for ℓ ∈ [0, 1] *and any hypothesis* h,

$$P_{\mathrm{aug}}=\alpha\,\hat{P}_{N}+(1-\alpha)\,\hat{Q}_{u,B},\qquad\alpha=\frac{N}{N+B}.$$

RP (h) − RPaug(h) ≤ 2(1 − α)Q(U > u) + u+ oN,B(1),
where oN,B(1) → 0 almost surely as N, B → ∞. Sketch. Decompose RPaug(h) into a factual part and an accepted part using α and (1 − α). Control the factual part by uniform convergence of PbN . Control the accepted part by Theorem 3.8 and weight it by (1 − α). Collect empirical fluctuations from both parts into oN,B(1).

$$\mathcal{E}_{\mathrm{target}}(\widehat{h}_{\mathrm{AVML}})\leq C_{1}\frac{dW^{2}}{n}+C_{2}\frac{dW^{2}}{N_{\mathrm{src}}}+C_{3}\sqrt{\frac{\log\mathcal{N}(\mathcal{H},\varepsilon)+\log(1/\beta)}{N_{\mathrm{eff}}}}+2(1-\alpha)\big{(}Q(U>u)+u\big{)}+\varepsilon_{\mathrm{app}}+\varepsilon.$$

The full theory shows that AWML pulls three levers together. Structure lowers the complexity of the hypothesis class. Modular recombination grows the sample size. Calibrated acceptance keeps the bias small. Exploration adds information efficiently when new data can be collected. These components interact in explicit ways in the final bound, which is why AWML remains data-efficient while giving a clear bias–variance–transfer trade-off.

Amplification is useful when the variance term of order 1/
√Neff is larger than the bias term. Permodule errors set the generator bias D. The acceptance rule with threshold u can then reduce this bias to a level governed by Q(*U > u*)+u. If modules are dependent, we apply the mixing correction in Appendix A.

Theorem 3.10 (Certified augmentation for empirical mixtures). Let Q *be a generator and* U an acceptance score that satisfy Theorem 3.8. Fix u ≥ 0 and draw B accepted samples from Qu*. Let* PbN be the empirical distribution of N factual samples and let Qbu,B *be the empirical distribution of* the accepted samples. Define

## Practical Interpretation

Interpretation Structured priors shrink the hypothesis class and improve generalization (Theorem 3.1). Modular recombination increases Neff but introduces a generator bias D (Theorem 3.5).

Certified acceptance replaces the fixed D by a tunable quantity Q(*U > u*) + u (Theorem 3.8). AWML chooses structure and the threshold u to trade data efficiency against guaranteed control of deployment bias. OPERATIONAL TAKEAWAY
The threshold u sets the bias scale through the term Q(*U > u*)+u. The mix weight (1−α) = B
N+B
sets how much influence synthetic data has. In practice we choose u by cross-validation or a small calibrator set and increase B only while the validation error decreases. Theorem 3.12 (Greedy exploration under submodular information). Let F(A) = I(Θ; OA) be the mutual information between latent parameters Θ and observations from a chosen set A*. Assume* F is nonnegative, monotone, and submodular. For any budget B, the greedy set GB *satisfies* I(Θ; OGB ) ≥1 −
1 e I(Θ; OA⋆B
),
where A⋆B is the best size-B *set. Greedy near-optimality follows from the classical result of* Nemhauser et al. and holds for many Gaussian and conditional-independence models (Nemhauser et al., 1978; Krause & Guestrin, 2008). Corollary 3.13 (Unified AWML bound: transfer and augmentation). Combine Theorem A.4 and Corollary 3.11. Let Neff = N + B and α = N/(N + B)*. With probability at least* 1 − β,

$$R_{P}(\widehat{h}_{\mathrm{{aug}}})-R_{P}(h^{*})\leq C{\sqrt{\frac{\log{\mathcal{N}}(\mathcal{H},\varepsilon)+\log(1/\beta)}{N+B}}}+2(1-\alpha)\left(Q(U>u)+u\right)+\varepsilon.$$

Corollary 3.11 (Excess risk under accepted augmentation). Let bhaug be an ERM over H *trained on* the N + B mixed samples. For any β ∈ (0, 1)*, with probability at least* 1 − β,

## 4 Experimental Validation 378

379 380 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431 We test the claims of Sections 2 and 3 in two settings. First, a controlled synthetic model isolates modular amplification and the role of Neff in Theorem 3.5. Second, a real low-label case study exercises certified acceptance and empirical mixtures as in Theorem 3.8 and Corollary 3.11. All runs use fixed random seeds and a held-out factual test set. We report mean and standard error over n = 8 seeds. Paired t tests and bootstrap confidence intervals are reported in Appendix A. Table 1: Constants and quantities that appear in the bounds. Values are estimated per run; estimation details are in Appendix A.

| Symbol     | Meaning                                 | Typical value              |
|------------|-----------------------------------------|----------------------------|
| D          | generator total variation bias (Sec. 3) | < 0.25                     |
| u          | acceptance threshold (Thm. 3.8)         | tuned by validation        |
| Q(U > u)   | rejected mass at threshold u            | < 0.10                     |
| N, B, Neff | factual, accepted synthetic, total      | see main tables and App. B |

## 4.1 Synthetic Modular Amplification

Setup We simulate M independent AR(1) modules z
(m)
t+1 = amz
(m)
t + ε
(m)
t, ε
(m)
t ∼ N (0, σ2m), m = 1*, . . . , M,*
which satisfies the factorization in equation 2. Factual training uses Ntrain = 80 trajectories of length T = 6 and evaluation uses Ntest = 400 trajectories. We fit each per-module conditional by ordinary least squares and estimate per-module total variation errors ˆδm by converting empirical KL
estimates to total variation via Pinsker on held-out samples of z
(pa(m))
t. Details of the estimators are in Appendix B.

We use two predictors. The first is ridge regression with regularization parameter α = 1.0. The second is a one hidden layer MLP with 64 ReLU units, trained with Adam at learning rate 10−3for 150 epochs. Synthetic pools are formed by recombining module states across trajectories. We vary the effective synthetic size Neff in the set {1, 5, 20, 100, 500, 2000}. Findings First, the variance term follows the predicted scaling. Test RMSE decreases as Neff increases. A log–log fit gives slopes close to −1/2 for both models, which matches the N
−1/2 eff rate in Lemma 3.4 and Theorem 3.5. The MLP shows larger absolute gains, which is consistent with a larger effective complexity term in the bound. Second, the augmentation bias remains small and is tracked by per-module errors. We compute the empirical risk difference between models trained only on factual data and models trained with recombined data. This difference scales with Pm ˆδm and stays below the additive term 2D predicted by the theory in the regimes we study. Third, there is a clear trade-off in the number of modules. Larger M gives more distinct recombinations and higher Neff. If independence is overstated, the aggregate bias D increases and the gains from amplification diminish. Ablation studies on M and recombination depth quantify this trade-off in Appendix B.

| Model (single seed)   | Factual RMSE   | Augmented RMSE   |
|-----------------------|----------------|------------------|
| Ridge                 | 0.227          | 0.219            |
| MLP                   | 0.253          | 0.233            |

Table 2: Illustrative seed to show scale. Full results with means, standard errors, and bootstrap confidence intervals across n = 8 seeds are reported in Appendix B.

432

![8_image_0.png](8_image_0.png) 433 434 435 436 437 438 439 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485

## 4.2 Real-World Evaluation: Certified Acceptance Under Low Labels

Setup We study a low-label deployment where accepted synthetic samples are mixed with N
factual samples to form Paug as in Theorem 3.10. We use the Uganda Living Standards Measurement Study 2019 household survey and derive a binary electrification label from energy expenditure fields and household covariates (Uganda Bureau of Statistics & The World Bank, 2019). Features include numeric variables such as energy spending and household size and categorical variables such as region and urban or rural status. For each trial we draw a stratified labeled set with n ∈ {25, 50, 100} and hold out a large factual test set. Sampling and preprocessing details are in Appendix B. We compare AWML to three baselines. The first baseline is factual only logistic regression and a small MLP. The second baseline uses a self-supervised autoencoder that learns a representation on unlabeled data before fitting the same heads. The third baseline is a pool based active learner that uses uncertainty sampling under the same label budget. All methods share the same features and label splits. For AWML we build an ensemble of twenty small MLPs that outputs a predictive mean and variance. When validation data are available we apply isotonic calibration to improve probabilistic predictions (Guo et al., 2017). Modular recombination generates synthetic candidates with pseudo-labels. Each candidate receives an uncertainty score U based on predictive variance. We choose a threshold u on a validation set by grid search over validation AUC. Accepted samples are added to the labeled set and a final logistic regression classifier is trained on factual plus accepted data. Per run we log baseline and final AUC, the accepted count B, total variation diagnostics, and stability flags for calibration. Definitions and full logs are in Appendix B. Findings Bias control matches the behavior predicted by Theorem 3.8. For each threshold u we estimate the risk gap between models trained on factual data and models trained on accepted synthetic data. Empirical gaps stay below the curve 2Q(*U > u*) + 2u in the regimes where calibration diagnostics are stable. This supports the interpretation of Q(*U > u*) and u as practical bias controls.

The end to end bound of Corollary 3.11 also lines up with validation curves. As Neff = N + B
grows, the variance term shrinks roughly like 1/
√Neff until the bias term 2(1 − α)(Q(*U > u*) + u)
becomes dominant. The simple proxy 486 487 488 489 490 491 492 493 494 495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539

$$\widehat{B}(u)=C\sqrt{\frac{\log{\mathcal{N}}({\mathcal{H}},\varepsilon)}{N+B(u)}}+2(1-\alpha(u))\big(Q(U>u)+u\big)$$

reaches its minimum near the same threshold that minimizes validation risk. This gives a practical tuning rule for u. AWML improves AUC in all low label regimes and outperforms the baselines. For example, at n = 25 labels the AUC of a factual only model improves from 0.8797 to 0.9402 after acceptance and retraining. Self supervised and active learning baselines narrow the gap but remain below the AWML variant under the same budget. Full numbers and confidence intervals are in Appendix B.

## 4.3 Uncertainty Filtering On Lsms Data

Figure 2 summarizes the uncertainty filtering behavior on the LSMS task. Panel A shows the ac-

![9_image_0.png](9_image_0.png) ceptance curve, namely the accepted fraction as a function of the variance threshold. Panel B shows a reliability diagram for a representative run and highlights cases where calibration drifts. Panel C compares predictive standard deviation for factual and synthetic examples. Panel D compares ROC curves for the baseline and final models. In the n = 25 regime, the AUC again moves from 0.8797 to 0.9402 in the illustrated run. Table 3 reports aggregate results across repeats. We list baseline and final AUC, the number of accepted samples B, a conservative total variation diagnostic, and simple stability indicators.

540 541 542 543 544 545 546 547 548 549 550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 Alistair L. Gibbs and Francis E. Su. On choosing and bounding probability metrics. *International* Statistical Review, 70(3):419–435, 2002.

Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. On calibration of modern neural networks. In *Proceedings of the 34th International Conference on Machine Learning*, 2017.

| nlabels   | Baseline AUC   | Final AUC   | Accepted   | TV bound   | L95   | Clamp frac.   |
|-----------|----------------|-------------|------------|------------|-------|---------------|
| 25        | 0.8797         | 0.9402      | 1110       | 0.01200    | 8.135 | 0.060         |
| 50        | 0.9148         | 0.9454      | 3500       | 0.08692    | 3.523 | 0.046         |
| 100       | 0.8966         | 0.9483      | 3156       | 0.24556    | 4.372 | 0.019         |

Table 3: Aggregate LSMS results averaged over repeats. The TV bound is a conservative diagnostic derived from the acceptance rule. Per run values and bootstrap confidence intervals appear in Appendix B. Interpretation On LSMS, accepting low variance and well calibrated synthetic examples gives consistent AUC gains in very low label regimes when calibration diagnostics are stable. Total variation diagnostics and instability flags highlight runs where the assumptions behind Theorem 3.8 may fail and suggest human review. This matches the intended use of AWML as a conservative augmentation layer rather than an unchecked generator.

## 4.4 Reproducibility And Artifacts

All experiments are deterministic given a random seed and are repeated for n = 8 seeds. For each run we store raw CSV files, calibration diagnostics, bootstrap resamples, and plotting scripts. The reproduction archive and a single command pipeline are provided with the submission; Appendix B lists all files and commands.

## 4.5 Concise Summary

The experiments support the two main mechanisms from Section 3. Modular recombination amplifies the effective sample size and reduces the estimation term in Theorem 3.5 when the aggregate bias D is small. Calibrated acceptance converts generator level guarantees into deployment level risk control as in Theorems 3.8 and 3.10. Together with the transfer bound in Theorem A.4, these results explain why AWML can use structure, augmentation, and uncertainty to gain data efficiency while remaining auditable.

## References

Elias Bareinboim and Judea Pearl. Transportability of causal effects: Completeness results. In Advances in Neural Information Processing Systems (NeurIPS), volume 25, pp. 247–255, 2012.

Peter L. Bartlett and Shahar Mendelson. Rademacher and gaussian complexities: Risk bounds and structural results. *Journal of Machine Learning Research*, 3:463–482, 2002.

Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, and David Duvenaud. Neural ordinary differential equations. In *Advances in Neural Information Processing Systems*, volume 31, 2018.

Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey E. Hinton. A simple framework for contrastive learning of visual representations. In Proceedings of the 37th International Conference on Machine Learning, 2020.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In *Proceedings of the 2019 Conference of* the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, 2019.

Chelsea Finn, Pieter Abbeel, and Sergey Levine. Model-agnostic meta-learning for fast adaptation of deep networks. In *Proceedings of the 34th International Conference on Machine Learning*, 2017.

David Ha and Jurgen Schmidhuber. World models. ¨ *arXiv preprint arXiv:1803.10122*, 2018. Danijar Hafner, Timothy Lillicrap, Jimmy Ba, and Mohammad Norouzi. Dream to control: Learning behaviors by latent imagination. In International Conference on Learning Representations (ICLR), 2020. arXiv:1912.01603.

Daniel Hsu, Sham M. Kakade, and Tong Zhang. Random design analysis of ridge regression. In Proceedings of the 25th Annual Conference on Learning Theory (COLT), volume 23 of *JMLR* Workshop and Conference Proceedings, pp. 9.1–9.24. JMLR, 2012.

Guido W. Imbens and Donald B. Rubin. Causal Inference for Statistics, Social, and Biomedical Sciences: An Introduction. Cambridge University Press, 2015.

Nikola B. Kovachki, Zongyi Li, Burigede Liu, Kamyar Azizzadenesheli, Kaushik Bhattacharya, Andrew M. Stuart, and Anima Anandkumar. Neural operator: Learning maps between function spaces. *Journal of Machine Learning Research*, 24(89):1–97, 2023.

Andreas Krause and Carlos Guestrin. Near-optimal sensor placements in gaussian processes: Theory, efficient algorithms and empirical studies. *Journal of Machine Learning Research*, 9:235–
284, 2008.

Matt J. Kusner, Joshua Loftus, Chris Russell, and Ricardo Silva. Counterfactual fairness. In Advances in Neural Information Processing Systems, volume 30, 2017.

Zongyi Li, Nikola Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Kaushik Bhattacharya, Andrew M. Stuart, and Anima Anandkumar. Fourier neural operator for parametric partial differential equations. In *Advances in Neural Information Processing Systems*, volume 33, pp. 9522–9533, 2020.

594 595 596 597 598 599 600 601 602 603 604 605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 Francesco Locatello, Ben Poole, Gunnar Ratsch, Bernhard Sch ¨ olkopf, Olivier Bachem, and Michael ¨
Tschannen. Weakly-supervised disentanglement without compromises. In *Proceedings of the* 37th International Conference on Machine Learning (ICML), volume 119 of Proceedings of Machine Learning Research, pp. 6348–6359, 2020.

Aditya Krishna Menon and Cheng Soon Ong. Linking losses for density ratio and class-probability estimation. In *Proceedings of the 33rd International Conference on Machine Learning (ICML)*,
volume 48 of *Proceedings of Machine Learning Research*, pp. 304–313. PMLR, 2016.

Mehryar Mohri, Afshin Rostamizadeh, and Ameet Talwalkar. *Foundations of Machine Learning*.

MIT Press, 2 edition, 2018.

George L. Nemhauser, Laurence A. Wolsey, and Marshall L. Fisher. An analysis of approximations for maximizing submodular set functions. *Mathematical Programming*, 14(1):265–294, 1978.

Judea Pearl. *Causality: Models, Reasoning, and Inference*. Cambridge University Press, 2 edition, 2009.

Maziar Raissi, Paris Perdikaris, and George E. Karniadakis. Physics-informed neural networks: A
deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378:686–707, 2019.

Yaniv Romano, Evan Patterson, and Emmanuel J. Candes. Conformalized quantile regression. In `
Advances in Neural Information Processing Systems, volume 32, 2019.

Bernhard Scholkopf, Francesco Locatello, Stefan Bauer, Nan Rosemary Ke, Nal Kalchbrenner, ¨
Anirudh Goyal, and Yoshua Bengio. Toward causal representation learning. *Proceedings of* the IEEE, 109(5):612–634, 2021.

Francesco Locatello, Stefan Bauer, Mario Lucic, Gunnar Ratsch, Sylvain Gelly, Bernhard ¨
Scholkopf, and Olivier Bachem. Challenging common assumptions in the unsupervised learning ¨ of disentangled representations. In Proceedings of the 36th International Conference on Machine Learning (ICML), volume 97 of *Proceedings of Machine Learning Research*, pp. 4114–4124, 2019.
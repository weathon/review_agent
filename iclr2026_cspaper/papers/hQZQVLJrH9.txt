000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053

# A Unified First-Order Framework For Activa- Tion Steering And Data Influence

Anonymous authors Paper under double-blind review

## Abstract

Activation steering adds a low-dimensional vector to an intermediate layer of a neural network to elicit or suppress behaviors, whereas *influence functions* trace the effect of infinitesimally re-weighting training examples on model outputs. We prove that, to first order, these techniques are *equivalent*: any steering vector can be represented as an influence weighting over training data and vice versa. This duality yields: (i) a constructive algorithm for mapping undesired behaviors back to causal training examples; (ii) an optimal-control perspective on steering that reveals its regularization properties; and (iii) generalization bounds for low-rank steering interventions. Our analysis adds theoretical clarity to two popular but previously disconnected strands of interpretability research.

## 1 Introduction

Large-scale neural networks—exemplified by transformer language models, diffusion–based image generators, and vision transformers—have become indispensable across science, industry, and culture. Their success, however, stands in tension with two practical desiderata. First, behavioral steering: practitioners often wish to suppress toxicity, reveal internal reasoning, or insert new factual knowledge without the prohibitive cost of retraining billions of parameters. Second, causal attribution: when a model exhibits bias or hallucination, we would like to trace that behavior to the specific training examples that gave rise to it. Current toolkits address these goals along two largely independent lines. Activation steering. This family of methods keeps the learned weights fixed and instead injects a low-dimensional vector into an intermediate layer during inference (Subramani et al., 2022). Activation-space steering has been used to detoxify harmful or biased language outputs (Turner et al., 2023; Wang & Shu, 2024), compress or elicit chain-of-thought reasoning (Azizi et al., 2025), flip or erase specific factual memories via knowledge neurons (Dai et al., 2022), and robustly edit whole fact distributions with SAKE's optimal-transport activation edits (Scialanga et al., 2025). See also (Zou et al., 2023). Because it modifies only activations, steering is fast, does not disturb the original checkpoint, and can be toggled on or off per query. Training-data influence. Influence-function techniques tackle attribution from the opposite end. By differentiating the empirical loss twice, they estimate how infinitesimally up-weighting a single training example would have altered today's prediction (Koh & Liang, 2017). The resulting influence scores underpin modern workflows for dataset debugging, bias auditing, and dataset distillation. See also (Pruthi et al., 2020; Barshan et al., 2020; Toneva et al., 2019; Feldman & Zhang, 2020). Although both lines of work pursue model *controllability*, their operational spaces are orthogonal:
activation steering assumes frozen weights, whereas influence analysis assumes fixed activations and perturbs the weights that produced them. Practitioners therefore face an unsatisfying dichotomy: experiment blindly with steering and, if it fails, resort to expensive parameter interventions—without guidance on *when* steering can succeed or how to connect a successful steering vector back to its causal data. We show that these two perspectives are, to first order, projections of the same underlying sensitivity tensor. Concretely, we construct an **Influence-Aligned Steering** (IAS) vector that, for any infinitesimal influence re-weighting, induces an identical logit shift—and we prove the converse mapping 1 054 055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107 from steering to influence. This equivalence is not merely conceptual: it yields explicit diagnostic and optimization tools that scale to billion-parameter models.

Scope and empirical justification. We focus on the *small-edit* regime used in practice. First-order analysis yields closed-form constructions (IAS), principal-angle diagnostics (ω), and predictable compute. Empirically, predicted and realized logit shifts are nearly collinear for small edits (cosine
→ 0.98; Fig. 1). For compact weight-space adaptation, see (Hu et al., 2022; Aghajanyan et al.,
2021).

1. Steer–influence equivalence. We establish a closed-form duality that maps every steering perturbation to a signed influence measure over the training set, and vice versa.

2. Alignment-based feasibility. A single scalar ω(x)—the cosine of the smallest principal angle between two Jacobian subspaces—fully characterizes when perfect equivalence is possible. If ω(x)is small, we prove a no-free-lunch lower bound showing that no activationspace edit can replicate the effect of data re-weighting.

3. Spectral Optimality. Given a norm budget, the steering direction that maximizes first-order logit change is the leading eigenvector of a Fisher–influence matrix; this spectral recipe replaces hand-crafted vectors.

4. Practical workflow. All quantities (Section 5) reduce to Jacobian–vector products and pseudoinverses, requiring only two backward passes per input. Practitioners can therefore (i)
prototype with steering, (ii) identify the responsible training examples, and (iii) decidewith ω—whether weight-level editing is necessary.

By unifying steering and influence under one first-order lens, IAS offers a single, efficient workflow for controllability and data provenance.

## 2 Background And Notation

A running toy example. See Appendix C for a compact linear-network illustration of IAS.

Model and layer of interest. Let fω : X ↑Rm be a network with parameters ω ↓ RP and logits fω(x). Fix a layer of width d with pre-activations h(x) ↓ Rd. We use the Jacobians

$$\mathbf{J}_{h\to y}(x):=\frac{\partial f_{\boldsymbol{\theta}}(x)}{\partial\mathbf{h}(x)}\in\mathbb{R}^{m\times d},\quad\mathbf{J}_{\boldsymbol{\theta}\to y}(x):=\frac{\partial f_{\boldsymbol{\theta}}(x)}{\partial\boldsymbol{\theta}}\in\mathbb{R}^{m\times P},\quad\mathbf{J}_{\boldsymbol{\theta}\to h}(x):=\frac{\partial\mathbf{h}(x)}{\partial\boldsymbol{\theta}}\in\mathbb{R}^{d\times P}.$$

Assumptions. (i) *Feasibility:* when stated, Im(Jω→y) ↔ Im(Jh→y) so IAS exists and is unique;
(ii) *Local smoothness:* a ϑ-Lipschitz neighborhood for Jacobians (Cor. 2); (iii) *Affine independence:*
for ϖ1-minimality of ϱs in Cor. 1.

Notation. Sh(x) := Im(Jh→y(x)) and Sω(x) := Im(Jω→y(x)) are subspaces of logit space; ω(x) := cos ↭min(Sω, Sh) ↓ [0, 1] is their smallest principal-angle cosine; Fh := Jh→yJ↓h→y is the activation-Fisher.

Influence functions. Let ϖ(z, ω) be the per-example loss and Hω := ↗2ω 1 |Z|
!z↔Z ϖ(z, ω) the empirical Hessian (or its damped Gauss–Newton surrogate), assumed positive-(semi)definite on a relevant subspace. Up-weighting a training point z by ς ↘ 1 induces !ωz = ≃ς H↗1 ω ↗ωϖ(z, ω),
and the first-order logit shift on test input x is

$$\Delta y^{\mathrm{IF}}(x)=\mathbf{J}_{\theta\to y}(x)\,\Delta\theta_{z}.$$
$$(\mathbf{l})$$

!yIF(x) = Jω→y(x) !ωz. (1)
Define the per-example first-order logit influence as I(z ↑ x) := Jω→y(x) !ωz (cf. Eq. equation 1). We use a damped inverse (Hω + φI)↗1 for stability (Appendix D.1). In all experiments, φ > 0 is treated as a Tikhonov regularizer; H may be replaced by a Gauss–Newton approximation without changing the first-order theory.

Activation steering. Adding ↼s ↓ Rd at the chosen layer yields the logit shift

$$\Delta y^{\mathrm{SV}}(x)=\mathbf{J}_{h\to y}(x)\,(\alpha\mathbf{s}).$$

!ySV(x) = Jh→y(x) (↼s). (2)
Equations equation 1–equation 2 share a linear form; the remainder of the paper characterizes when one can stand in for the other and how to construct the corresponding perturbation efficiently.

## 3 Adual View: Parameter–Activation Sensitivities

Why add one more lens? We have already seen that two linear maps govern first-order behavior:
the parameter–logit Jacobian Jω→y and the activation–logit Jacobian Jh→y. Theorems 5.1–6.2 will quantify their interaction, but first we show that the maps form a *primal–dual* pair in the convexanalysis sense.

108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 Two complementary projections. The primal view is an *orthogonal projection* of the desired logit displacement Jω→y!↽ onto Sh(x), then a lift back to activation space with minimum energy.

The dual view projects in the *Fisher norm* induced by activations; the dual multiplier εε is the Fisher-metric certificate of effort required to cover components outside Sh(x).

Rule of thumb. If ⇐εε⇐ is small, steering is cheap and faithful; if large, a weight-space update is likely necessary. Computing εε is as cheap as IAS itself (two JVPs), so the check can precede any search for directions. We start from the *inverse* problem: given a desired parameter-space displacement !ω (e.g., an influence update), find the shortest activation change that reproduces its logit effect. 3.1 THE PRIMAL PROGRAM: LEAST-EFFORT STEERING 3.2 THE DUAL PROGRAM

$$\operatorname*{min}_{\Delta\mathbf{h}\in\mathbb{R}^{d}}{\frac{1}{2}}\|\Delta\mathbf{h}\|_{2}^{2}\quad{\mathrm{s.t.}}\quad\mathbf{J}_{h\to y}\,\Delta\mathbf{h}=\mathbf{J}_{\theta\to y}\,\Delta\boldsymbol{\theta}.$$

$$(\mathbf{P})$$
⇐!h⇐22 s.t. Jh→y !h = Jω→y !ω. (P)
Feasibility. If Im"Jω→y
\# ↔ Im"Jh→y
\#, the constraint is feasible and the Euclidean minimumnorm solution exists and is unique.

Introduce ε ↓ Rm. Minimizing the Lagrangian over !h yields !hε = J↓h→yεε with

$$\lambda^{\star}=-\left(\mathbf{J}_{h\to y}\mathbf{J}_{h\to y}^{\top}\right)^{\dagger}\mathbf{J}_{\theta\to y}\,\Delta\theta,\qquad\Delta\mathbf{h}^{\star}=\mathbf{J}_{h\to y}^{\dagger}\,\mathbf{J}_{\theta\to y}\,\Delta\theta.$$

\#†Jω→y !ω, !hε = J†h→y Jω→y !ω. (2)
Thus the *Influence-Aligned Steering (IAS)* vector is the projection of the target logit movement onto the activation-reachable subspace, lifted back with the Moore–Penrose pseudoinverse.

## 4 Steering–Influence Duality At The Data Level

The primal–dual view explains the existence of an optimal steering vector for a *given* parameter perturbation. Related scalable data-attribution methods include (Pruthi et al., 2020; Barshan et al., 2020). We now climb one level up and ask for a direct correspondence between steering interventions and *training-data* re-weightings.

$\eqref{eq:walpha}$. 
Computational primitives (cost model). All results rely on: (i) two Jacobian–vector or vector–
Jacobian products per input, (ii) a rank-d pseudoinverse of Jh→y (never larger than the layer width),
and (iii) a small SVD to estimate principal angles for ω.

Geometry and diagnosis. Fh := Jh→yJ↓h→y is the Fisher information of the logits w.r.t. activations; εε is the Fisher-metric certificate of effort. A large ⇐εε⇐ signals that most of the desired displacement lies outside the activation subspace and that steering will require large energy (or fail), anticipating the alignment bounds below.

Lemma 4.1 (Chain-rule factorization). For any scalar metric mω(x) *and any layer* ϖ, Sketch. Differentiate mω along the composite map ω↑h(ϑ)↑mω.

Residual when spans do not match. If Im(Jh→y) does not contain Im(Jω→y), perfect matching is impossible. Writing Ph for the orthogonal projection onto Sh(x), the irreducible residual obeys
%%(I ≃ Ph) Jω→y!↽%%2 ⇒ &1 ≃ ω(x)2 %%Jω→y!↽%%2, (3)
the logit-space version of Theorem 5.1. In practice, we use equation 3 as a pre-check: small ω(x) ⇑
skip steering.

The result holds exactly if the set {I(z ↑x)}z↔Z spans Im(Jh→y); otherwise Eq. equation 4 holds up to a residual whose norm is bounded by "1 ≃ ω(x)2\#1/2⇐↼s⇐.

Intuition. Equation 4 says that a steer vector ↼s acts like redistributing |↼| units of mass across training examples, weighted by how well their gradients correlate with s. The minimal-ϖ1 measure that achieves this correlation is precisely ϱs. Practical payoff. Given an empirical steering vector, ϱs pinpoints the *fewest* training examples to relabel/remove/examine to reproduce the behavioral change (see Section 7).

## 4.2 A Geometric Picture Of Alignment

162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 Let Sω(x) := Im(Jω→y(x)) and Sh(x) := Im(Jh→y(x)) be subspaces in logit space. The primal program P orthogonally projects Jω→y!ω onto Sh(x) and lifts to the minimum-norm activation; the dual equation 2 performs the projection in the Fisher norm Fh := Jh→yJ↓h→y. Small principal angles imply close projections and modest ⇐εε⇐; near-orthogonality yields the no-free-lunch regime.

## 4.1 From Steering To Data: A Causal Corollary

Corollary 1 (Minimal data re-weighting induced by steering). *Assume that the influence vectors*
{I(z↑x)}z↔Z are affinely independent; otherwise the ϖ1*-minimal solution need not be unique. Let*
(s, ↼) be an activation-space intervention at layer ϖ *with* ⇐s⇐ = 1 and |↼| ↘ 1*. Among all signed* measures ⇀ *on the training set that reproduce the first-order logit shift,*

$$\Delta y^{\mathrm{SV}}(x)\;=\;\sum_{z\in{\mathcal{Z}}}\nu(z)\,{\mathcal{I}}(z\!\to\!x),$$

the measure ϱs *constructed in Eq. 4 is* ϖ1-minimal*, i.e.* ⇐ϱs⇐1 = minϱ
'⇐⇀⇐1 :
⇀ satisfies the equation( = |↼|.

Idea of the proof. Equation 4 already realizes the shift with ⇐ϱs⇐1 = |↼|. If another measure ⇀
achieved the same shift with smaller ϖ1 norm, one could scale ϱs down and still match the shift, contradicting the definition of ↼ as the steering magnitude.

Theorem 4.2 (Steering–Influence Equivalence). Let s ↓ Rd *be added with magnitude* ↼ ↘ 1 at layer ϖ. There exists a signed measure ϱs *over the training set such that*

$$f_{\mathbf{\theta}}^{\mathbf{a},\alpha}(x)-f_{\mathbf{\theta}}(x)=\sum_{z\in\mathcal{Z}}\rho_{\mathbf{s}}(z)\,\mathcal{I}(z\!\rightarrow\!x)\;+\;O(\alpha^{2}),\quad\|\rho_{\mathbf{s}}\|_{1}=|\alpha|.\tag{4}$$

Conversely, any signed weighting w ↓ R|Z| with ⇐w⇐1 = ς admits a steering vector sw *with* ⇐sw⇐ = O(ς) *that realizes the same first-order output shift.*

$$v\to y\,\mathbf{\hat{\mathbf{\theta}}}\,\mathbf{\hat{\mathbf{\theta}}}\,\mathbf{\hat{\mathbf{\theta}}}\,\mathbf{\hat{\mathbf{\theta}}}\,\mathbf{\hat{\mathbf{\theta}}}\,\mathbf{\hat{\mathbf{\theta}}}$$

↗ω mω(x) = J↓ω→h(ω) ↗h(ω)mω(x).

Implication. Given an empirical steering direction, the associated measure ϱs points straight to the *most causal* training documents. In practice, one inspects the top-weighted examples to debug bias or privacy leaks.

216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 Practical diagnostic. The norm of εε quantifies unreachable components: small ⇐εε⇐ implies faithful, low-energy steering; large values suggest weight-space editing. Computing εε costs two JVP/VJPs (same as IAS), enabling a quick steer-vs-retrain decision. Choosing the layer ϖ **in practice.** Across LMs we find (Fig. 2) that ω typically increases toward later blocks. A simple heuristic is therefore: probe ω at a few candidate layers on a small prompt batch and pick the smallest layer index with ω ⇓ 0.7.

This balances headroom (later layers) with locality (earlier layers).

## 5 Main Theoretical Guarantees

5.1 WHEN DOES STEERING PERFECTLY MATCH INFLUENCE? Theorem 5.1 (Alignment Bound). For any infinitesimal parameter perturbation !ω, the relative logit error of the minimum-norm IAS vector !hε *satisfies* Intuition and use. &
Overlap (large ω) enables exact matching; misalignment limits fidelity at rate 1 ≃ ω2. Computing ω (two small SVDs) quickly certifies feasibility.

5.3 STEERING MAXIMALLY UNDER AN ϖ2 BUDGET
Theorem 5.3 (Spectral Optimality). Fix a norm budget ⇐s⇐ ⇒ B*. Let*

$=\;\bullet\:h\to$  . 
$\mathbf{u}$
$$\Sigma:=\frac{1}{|Z|}\sum_{z\in Z}\mathbf{J}_{\theta\to h}^{\top}\mathbf{H}_{\theta}^{-1}\nabla_{\theta}\ell\big(z,\theta\big)\,\nabla_{\theta}\ell\big(z,\theta\big)^{\top}\mathbf{H}_{\theta}^{-1}\mathbf{J}_{\theta\to h}\,,$$
$$\frac{\|\mathbf{J}_{\theta\to y}\Delta\boldsymbol{\theta}-\mathbf{J}_{h\to y}\Delta\mathbf{h}^{\star}\|_{2}}{\|\mathbf{J}_{\theta\to y}\Delta\boldsymbol{\theta}\|_{2}}\ \leq\ \sqrt{1-\gamma^{2}(x)},$$
where ω(x) is the cosine of the smallest principal angle between the column spaces of Jh→y(x) and Jω→y(x) *(Bjorck & Golub, 1973).* ¨
5.2 THE UNIQUE STEERING VECTOR IF ALIGNMENT HOLDS
Theorem 5.2 (Minimum-Norm IAS). If Im(Jω→y) ↔ Im(Jh→y)*, the unique steering vector that* solves problem equation P is
!hε = J†h→y Jω→y !ω.

Note. This is the orthogonal projection/lift solution; two JVP/VJPs and a rank-⇒ d pseudoinverse suffice in practice.

Corollary 2 (Second-order radius). *If the map* ω ⇔↑ "Jω→y, Jω→h
\#is ϑ-Lipschitz in a neighborhood of ω, then the Taylor remainder obeys ⇐fω+!ω ≃ fω ≃ Jω→y!ω⇐2 ⇒ ϑ⇐!ω⇐22, and the matching IAS perturbation incurs the same O(↼2) *error.* Estimating the spectral direction (practical recipe). Power iteration with Hutchinson-style mini-batches suffices:
1. Initialize v0 ↖ N (0, Id).

2. For t = 0, 1*,...*: draw a mini-batch B; compute gz := J↓ω→h(H + φI)↗1↗ωϖ(z, ↽) for z ↓ B;
set vt+1 ↙ !z↔B gz(g↓z vt).

3. Stop when ⇐vt+1 ≃ vt⇐/⇐vt⇐ < ⇁; return vt.

Note. " averages influence correlations; its top eigenvector gives a principled steering direction estimated via one power-iteration over mini-batches.

$$\mathbf{J}_{\theta\to h}^{\top}(\mathbf{H}+\lambda I)^{-1}\nabla_{\theta}\ell(z,\theta){\mathrm{~for~}}z\in{\mathcal{B}};$$

The steering vector that maximizes the expected first-order logit change is the top eigenvector smax of "*, and the achievable change equals* B&φmax(") ⇐↗hfω(x)⇐.

Consequently, mis-alignment compounds multiplicatively.

## 6 Generalization Under Low-Rank Steering

$$\Re_{n}(\ell\circ\tilde{f})\;\leq\;\Re_{n}(\ell\circ f_{\theta})\;+\;\alpha L\,\sqrt{\frac{2k}{d n}},$$
where d is the width of layer ϖ and n the sample size.
Sketch. Combine Thm. 2 of Pinto et al. (2024) with the fact that IAS changes only a rank-k submatrix of the layer weight. The additional Rademacher term is bounded by ↼L&2*k/dn*.

$${\mathcal{L}}({\bar{f}})-{\mathcal{L}}(f_{\theta})\;\leq\;2\,\Re_{n}(\ell\circ{\bar{f}})+c{\sqrt{\frac{\log(1/\delta)}{n}}}\;\leq\;2\,\Re_{n}(\ell\circ f_{\theta})+2\alpha L{\sqrt{\frac{2k}{d n}}}+c{\sqrt{\frac{\log(1/\delta)}{n}}},$$

n , (4)
for a universal constant c. Thus, for fixed budget ↼ and modest rank k ↘ d, the excess risk term due to IAS vanishes as d and n grow.

Practical guidance. (i) Prefer low ranks k and smaller ↼ unless ω is close to 1. (ii) When ω < 0.5, skip steering and switch to weight-space editing; the bound equation 3 predicts poor fidelity. (iii) Treat damping φ as a regularizer that trades a small bias for numerical stability in H↗1
(Appendix D.1).

## 6.1 When Steering Is Provably Insufficient

Theorem 6.2 (No-Free-Lunch). Let ω(x) *denote the cosine of the smallest principal angle between* Im(Jω→y(x)) and Im(Jh→y(x)). If ω(x) ⇒ ϱ < 1, then for every activation perturbation !h and the corresponding (best-possible) parameter perturbation !ω *we have*

$$\frac{\left\|J_{h\to y}(x)\,\Delta\mathbf{h}\right\|_{2}}{\left\|J_{\theta\to y}(x)\,\Delta\theta\right\|_{2}}\;\leq\;\gamma(x)\;\leq\;\rho.$$

270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323

## 7 Experiments

Setup. Unless stated otherwise we use GPT-2 Medium and steer at layer ϖ=8. Steering vectors are built from 50 toxic vs. 50 neutral Jigsaw prompts; evaluation uses 500 TOXIGEN prompts. Toxicity is scored with DETOXIFY; perplexity is measured on a benign WikiText subset.

Lemma 5.4 (Layer-wise composability). Let ω1, ω2 be the alignment cosines for two consecutive layers. Applying IAS at layer 1 and *layer 2 yields a combined alignment cosine at least*

$$\gamma_{12}\;\geq\;\gamma_{1}\,\gamma_{2}\;=\;\sqrt{1-\left(1-\gamma_{1}^{2}\right)}\,\sqrt{1-\left(1-\gamma_{2}^{2}\right)}.$$
$$(4)$$

Theorem 6.1 (Rademacher-complexity blow-up under rank-k steering). Let fω be the base model and ˜f = fω + ↼UV ↓ the model obtained by adding a rank-k IAS correction at layer ϖ*, with*
⇐U⇐2 = ⇐V ⇐2 = 1. For any loss ϖ that is L-Lipschitz in its first argument, the empirical Rademacher complexity satisfies From complexity to risk. Let L+ be the empirical risk and L the population risk. With probability 1 ≃ δ, Intuition. Poor alignment means the desired logit displacement lives largely outside the steering subspace; even an infinite-norm activation change cannot push further than factor ϱ. Consequence for practice. If the quick diagnostic yields a small ω(x), engineers can skip steering entirely and proceed straight to parameter-space editing. IAS is the exact minimum-energy activation edit matching a target first-order logit displacement; its fidelity is controlled by ω(x). The spectral recipe provides a principled way to choose a strong direction under a budget, and low-rank IAS has a benign impact on generalization. When ω is small, the geometry itself forbids steering to fully replace influence.

## 7.1 Language-Model Detoxification Via Steering

324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377

## 7.2 First-Order Equivalence: Ias Matches Influence At First Order

We compare Contrastive Activation Addition (CAA) with our Influence-Aligned Steering (IAS),
using identical ϖ2 magnitude and layer. Table 1 reports mean toxicity (lower is better) and benign-
PPL.

| Baseline          | CAA    | IAS    |        |
|-------------------|--------|--------|--------|
| Toxicity (mean) → | 0.0195 | 0.0150 | 0.0164 |
| Perplexity →      | 14333  | 13291  | 13701  |

Table 1: Detoxification on 500 TOXIGEN prompts with benign-PPL on WikiText (GPT-2 Medium, ϖ=8). The feasibility diagnostic ω(x) increases with depth on GPT-2 Medium, with the median rising from

![6_image_1.png](6_image_1.png) 0.64 at layer 0 to 0.94 by layer 11 (Figure 2). This supports Theorem 5.1: late layers provide the best subspace overlap for steering to match influence.

## 7.3 Alignment Vs. Layer Depth

Figure 1: IAS → **influence (first order).** Predicted (first-order) vs. actual logit shifts for n=5000 pairs at ϖ=8; cosine 0.978, slope 1.50. Our theory predicts that the *first-order* logit shift from an influence update is matched by the

![6_image_0.png](6_image_0.png) minimum-norm IAS vector. Over n=5000 prompt–token pairs at ϖ=8, predicted vs. actual shifts are nearly collinear (cosine 0.978, slope 1.50), consistent with the expected linear regime.

## 7.4 Spectral Optimality Of Steering Directions (Imagenet)

378 379 380 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431 Armen Aghajanyan, Sonal Gupta, and Luke Zettlemoyer. Intrinsic dimensionality explains the effectiveness of language model fine-tuning. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pp. 7319–7328, 2021.

Seyedarmin Azizi, Erfan Baghaei Potraghloo, and Massoud Pedram. Activation steering for chainof-thought compression. *arXiv preprint arXiv:2507.04742*, 2025.

We test the vision analog of Theorem 5.3 on ResNet-50 by estimating the spectral direction that

![7_image_0.png](7_image_0.png) maximizes the horse logit (class 339). Figure 3 compares the spectral shift against random directions: the spectral radius lies far in the tail of the null distribution (p=0.00498, z=3.55).

## 8 Related Work

Activation steering originated in sentiment control for language models (Turner et al., 2023) and has since grown into a family of latent-direction methods. Influence functions were ported from classical statistics to deep nets by Koh & Liang (2017). Our work is the first to give a closedform map between the two ideas and to quantify when one subsumes the other. Concurrent work on parameter-space editing (ROME (Meng et al., 2022), MEMIT (Meng et al., 2023)) tackles a complementary regime: finite, non-infinitesimal changes to factual knowledge.

## 9 Conclusion

We have shown that steering vectors and influence functions—previously separate tools—live on the same geometric plane. Influence-Aligned Steering provides the mathematical bridge, complete with error guarantees, constructive formulas, and impossibility results. Beyond its theoretical appeal, IAS
promises an integrated workflow for debugging, auditing, and aligning large neural models: steer first, trace provenance, edit weights only when the geometry demands it. IAS is a first-order theory; very large steering magnitudes or influence perturbations beyond the quadratic regime may violate the linear approximation. Extending the analysis to second orderwhere Hessian–Jacobian interactions appear—is left for future work. Moreover, computing exact pseudoinverses is tractable for single layers but challenging for deep stacks; exploring Krylov or randomized SVD methods is an open engineering problem. AI assistance disclosure. We used large language models to polish grammar and improve the clarity of some sentences.

## References

Elnaz Barshan, Marc-Etienne Brunet, and Gintare Karolina Dziugaite. Relatif: Identifying explanatory training samples via relative influence. In *International Conference on Artificial Intelligence* and Statistics, pp. 1899–1909. PMLR, 2020.

S Basu, P Pope, and S Feizi. Influence functions in deep learning are fragile. In International Conference on Learning Representations (ICLR), 2021.

Ake Bj ˚ orck and Gene H Golub. Numerical methods for computing angles between linear subspaces. ¨
Mathematics of computation, 27(123):579–594, 1973.

Damai Dai, Li Dong, Yaru Hao, Zhifang Sui, Baobao Chang, and Furu Wei. Knowledge neurons in pretrained transformers. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 8493–8502, 2022.

Vitaly Feldman and Chiyuan Zhang. What neural networks memorize and why: Discovering the long tail via influence estimation. *Advances in Neural Information Processing Systems*, 33:2881– 2891, 2020.

Edward J Hu, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. In *International Conference on* Learning Representations, 2022.

Pang Wei Koh and Percy Liang. Understanding black-box predictions via influence functions. In International conference on machine learning, pp. 1885–1894. PMLR, 2017.

Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. Locating and editing factual associations in gpt. *Advances in neural information processing systems*, 35:17359–17372, 2022.

Kevin Meng, Arnab Sen Sharma, Alex J Andonian, Yonatan Belinkov, and David Bau. Mass-editing memory in a transformer. *The Eleventh International Conference on Learning Representations*, 2023.

Andrea Pinto, Akshay Rangamani, and Tomaso Poggio. On generalization bounds for neural networks with low rank layers. *arXiv preprint arXiv:2411.13733*, 2024.

432 433 434 435 436 437 438 439 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485 Garima Pruthi, Frederick Liu, Satyen Kale, and Mukund Sundararajan. Estimating training data influence by tracing gradient descent. *Advances in Neural Information Processing Systems*, 33: 19920–19930, 2020.

Marco Scialanga, Thibault Laugel, Vincent Grari, and Marcin Detyniecki. Sake: Steering activations for knowledge editing. *arXiv preprint arXiv:2503.01751*, 2025.

Nishant Subramani, Nivedita Suresh, and Matthew E Peters. Extracting latent steering vectors from pretrained language models. In Findings of the Association for Computational Linguistics: ACL
2022, pp. 566–581, 2022.

Mariya Toneva, Alessandro Sordoni, Remi Tachet des Combes, Adam Trischler, Yoshua Bengio, and Geoffrey J Gordon. An empirical study of example forgetting during deep neural network learning. In *International Conference on Learning Representations*, 2019.

Alexander Matt Turner, Lisa Thiergart, Gavin Leech, David Udell, Juan J Vazquez, Ulisse Mini, and Monte MacDiarmid. Steering language models with activation engineering. *arXiv preprint* arXiv:2308.10248, 2023.

Haoran Wang and Kai Shu. Trojan activation attack: Red-teaming large language models using steering vectors for safety-alignment. In Proceedings of the 33rd ACM International Conference on Information and Knowledge Management, pp. 2347–2357, 2024.

Andy Zou, Long Phan, Sarah Chen, James Campbell, Phillip Guo, Richard Ren, Alexander Pan, Xuwang Yin, Mantas Mazeika, Ann-Kathrin Dombrowski, et al. Representation engineering: A top-down approach to ai transparency. *arXiv preprint arXiv:2310.01405*, 2023.
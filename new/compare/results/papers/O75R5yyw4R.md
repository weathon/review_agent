000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053

# Effective Test-Time Scaling Of Discrete Diffusion Through Iterative Refinement

Anonymous authors Paper under double-blind review

## Abstract

Test-time scaling through reward-guided generation remains comparatively less explored for discrete diffusion models despite its potential as a promising alternative. In this work, we introduce Iterative Reward-Guided Refinement (**IterRef**), a novel test-time scaling method tailored to discrete diffusion that leverages rewardguided noising-denoising transitions to progressively refine misaligned intermediate states. We formalize this process within a Multiple-Try Metropolis (MTM) framework, proving convergence to the reward-aligned distribution. Unlike prior methods that assume the current state is already aligned with the reward distribution and only guide the subsequent transition, our approach explicitly refines each state *in situ*, progressively steering it toward the optimal intermediate distribution. Across both text and image domains, we evaluate IterRef on diverse discrete diffusion models and observe consistent improvements in reward-guided generation quality. In particular, IterRef achieves striking gains under low compute budgets, far surpassing prior state-of-the-art baselines. Code will be publicly released.

## 1 Introduction

Breakthroughs in foundation models, such as large language models and diffusion models, have been driven by massive web-scale datasets and have led to remarkable advances in language and image generation tasks (Brown et al., 2020; Rombach et al., 2022). However, as recent models continue to scale, concerns have been raised about the availability of sufficiently diverse training data, suggesting a potential training-time scaling barrier (Villalobos et al., 2024). In parallel, the field has also explored *test-time scaling*, which leverages additional compute at inference to improve performance. This paradigm has recently shown promising results in both autoregressive (Snell et al., 2024) and continuous diffusion models (Ma et al., 2025), suggesting a viable path to further unlock their performances. While the importance of test-time scaling is increasingly recognized across different modeling paradigms, its role in *discrete diffusion* remains underexplored. Unlike continuous diffusion, where Gaussian noise enables gradient-based guidance and natural error correction (Uehara et al., 2025), test-time scaling in discrete diffusion poses unique challenges: (1) due to token discretization, gradients from reward models cannot be directly used for inference guidance, limiting their utility in reward alignment; and (2) incorrectly generated tokens cannot be corrected in subsequent denoising steps, since tokens are fixed once generated. Consequently, these challenges underscore the need for effective test-time scaling strategies tailored to discrete diffusion models. In this paper, we propose **IterRef**, a novel test-time scaling method for discrete diffusion. Our approach leverages MCMC transitions to iteratively refine tokens, progressively aligning them with the reward during sampling. As illustrated in Figure 1, inspired by the predictor–corrector paradigm (Song et al., 2020), we design the transition as a noising–denoising process: added noise promotes exploration, while denoising restores consistency with the target. To instantiate this design, we adopt the classical Multiple-Try Metropolis (MTM) framework (Liu et al., 2000) and tailor both the transition kernel and the balancing function to the reward alignment objective of discrete diffusion. This adoption yields a principled mechanism for test-time scaling, allowing us to further provide a theoretical guarantee that iterative refinement sampling converges to the target distribution. Through extensive experiments, we evaluate IterRef across multiple discrete diffusion backbones: MDLM (Sahoo et al., 2024) and LLaDA-8B (Nie et al., 2025) for language generation, and 1 054 055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107

![1_image_0.png](1_image_0.png) 

Figure 1: **Overview of IterRef**. (a) Reward-guided denoising trajectories: Blue nodes are selected samples, gray nodes are rejected candidates. Unlike existing single-step guidance methods (IS
and SMC), IterRef discovers higher-reward samples by iteratively applying noising-denoising kernels. Noising process (dotted nodes) with random remasking incurs nearly zero cost, while offering broader regions to explore and correct tokens. (b) Scaling performance. IterRef scales significantly faster (up to 8×) than baselines with safety reward on LLaDA-8B (See § 4.5 for details). MaskGiT (Chang et al., 2022) for image generation, using diverse reward functions, such as CoLA, Toxicity, Sentiment, and Perplexity for language, and CLIPScore for image generation. Compared with existing reward-guided diffusion methods, IterRef consistently demonstrates the *most effective* scaling across compute budgets, achieving up to a 2x improvement on Toxicity reward with LLaDA- 8B under the equal compute (See Figure 2). Furthermore, our in-depth studies on iteration count and number of particles uncover task-specific refinement dynamics, where the optimal application of iterations varies significantly across different objectives, Overall, our main contributions can be summarized as follows:
- We propose IterRef, an effective test-time scaling method for discrete diffusion, that consistently outperforms prior reward guidance methods across modalities, model backbones, and guided generation tasks. Notably, IterRef remains highly effective even under low NFE settings.

- We identify which noise levels in the diffusion sampling process play the most crucial role in shaping the final generation, providing new insights into the dynamics of discrete diffusion.

- We show that iterative refinement sampling is not simply heuristic: IterRef leads to convergence to the target distribution, and we provide an explanation of its effectiveness under certain assumptions (See Proposition 1).

## 2 Preliminaries

Discrete Diffusion Models. Diffusion models with discrete state space were initially formulated by considering processes over binary random variables (Sohl-Dickstein et al., 2015). Building on this, a more general framework that employs categorical random variables for diffusion models was later introduced (Austin et al., 2021). The most recent and effective approach is the absorbing state formulation, which considers a transition matrix in which each token transitions to a masked token m. The process is formulated over timesteps t ∈ [0, T], where the intermediate state xt is represented as a sequence of length L, xt = (x 1 t, x2 t*, . . . , x*L
t) ∈ Xt, with each position value x it taking values from the vocabulary V.

The forward noising process is defined as a stochastic transition distribution q(xt|xt−1), in which each token is independently retained or replaced with a mask token m according to a time-dependent corruption probability.

Given the forward noising process q(xt|xt−1), the generative model defines a reverse process parameterized by θ as

$p_{\theta}(x_{t-1}|x_{t})$, $t=1$, $\cdot\cdot\cdot$, $T$.  
2 The full denoising trajectory is then expressed as The objective of training is to learn pθ so that the marginal samples x0 ∼ pθ(x0) approximate the data distribution. Reward-Guided Generation. The goal of reward-guided generation is to preserve the naturalness of the samples while maximizing the given reward, or more generally, to draw samples from a target distribution that reflects human preferences. Concretely, the objective of reward-guided sampling with a reward function r(·) is to draw samples from the target distribution:
p
∗(x0) = arg max p Ex0∼p(·)
-r(x0)− α DKL (p(x0)∥pθ(x0)) ∝ expr(x0)/α)pθ(x0),
108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 Several notable approaches have explored particle-based methods to steer the denoising process toward reward-aligned intermediate distributions (Li et al., 2024). The distribution at each step cannot be perfectly approximated, and the resulting errors accumulate along the trajectory (Wang et al., 2025). Existing representative methods rely on sequential sampling, which advances in a single pass to the next step and therefore lacks mechanisms to refine intermediate distributions toward the optimal target as the process unfolds (Johansen, 2009). To address this limitation, we introduce IterRef, a refinement strategy based on the Multiple-Try Metropolis framework, which iteratively improves the intermediate steps. Section 3.1 provides a theoretical analysis of our method, Section 3.2 presents the algorithmic formulation, and Section 3.3 discusses its practical implementation and computational cost.

where α controls the strength of the KL divergence regularization term. In order to sample the target distribution through the reverse denoising trajectory {xT , xT −1*, . . . , x*0} of the diffusion model, each step must be drawn from the conditional distribution p
∗(xt−1 | xt). The conditional distribution p
∗(xt−1 | xt) can be expressed in terms of a reward function that predicts the expected future reward. Formally, the intermediate reward function is defined as r(xt) = α log Ex0∼pθ(·|xt)
-exp1α r(x0) .

$${\bf{}}^{-}\;{\cal{F}}^{-}{\bf{u}}$$
$\square$
$\mathbf{L}=\mathbf{a}$
Using the intermediate reward function, the optimal transition kernel p
∗(xt−1 | xt) can be expressed as follows:
p
∗(xt−1|xt) ∝ pθ(xt−1|xt) exp(r(xt−1)/α). (1)
Existing approaches to approximate the optimal transition kernel using Sequential Monte Carlo (Singhal et al., 2025) or Importance Sampling (Li et al., 2024); further details are provided in Appendix D.1.

## 3 Iterative Reward-Guided Refinement Via Multiple-Try Metropolis 3.1 Multiple-Try Metropolis For Discrete Diffusion

Problem Setup. Specifically, our goal is to sample each intermediate state xt in the denoising process from the optimal distribution p
∗(xt). To make this precise, we recall that the optimal distribution can be formally characterized as follows: Remark 1 (Arising naturally from the proof of Theorem 1 in Uehara et al. (2024)). *The optimal* distribution p
∗(xt) *induced by the optimal transition kernel is given by*

p ∗(xt) = p(xt) exp(r(xt)/α) Px∈Xt p(x) exp(r(x)/α)
.
The detailed derivation is provided in Appendix D.1. Accordingly, we establish p
∗(xt) as the target distribution for our method, and our approach is designed to iteratively refine intermediate distributions toward this target.

$p_{\theta}(x_{0}\cdot T)=p(x_{T})\prod_{t=1}^{T}p_{\theta}(x_{t-1}|x_{t})$.  
T
t=1
162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 3: *Resample and Update*: Draw N − 1 i.i.d. sample x
′′(1)
t*, . . . , x*′′(N−1) from K(x
′
t, ·) and define x
′′(N) = xt, then accept x
′t with probability

$$\beta=\min\left(1,\frac{\sum_{i=1}^{N}p^{*}(x_{t}^{\prime})K(x_{t}^{\prime},x_{t})\lambda(x_{t}^{\prime},x_{t})}{\sum_{i=1}^{N}p^{*}(x_{t}^{\prime\prime(i)})K(x_{t}^{\prime\prime(i)},x_{t}^{\prime})\lambda(x_{t}^{\prime\prime(i)},x_{t}^{\prime})}\right)$$

Design Choice. To further enhance the exploration capability of the sampler, we design the transition kernel by leveraging a noising-denoising process. Previous studies on diffusion models have shown that the perturbation-correction mechanism (Song et al., 2020) can effectively reduce errors during iterative refinement. Motivated by this, we design our transition kernel through a noising–denoising process, which enhances exploration while preserving the conditions required for MTM's theoretical guarantees. Importantly, our formulation integrates reward guidance directly within the noising–denoising steps, ensuring that the refinement process is not only error-corrective but also explicitly steered toward higher-reward solutions. Formally, we define the transition kernel K, encompassing both the noising and denoising operations and balancing function λ, which makes the overall algorithm executable as

$$K(x_{t},x^{\prime}_{t})\;=\;\sum_{x_{t}\in{\cal X}_{\pi}}q(x_{s}|x_{t})p_{\theta}(x^{\prime}_{t}|x_{s}),\;\lambda(x_{t},x^{\prime}_{t})=\frac{1}{p(x_{t})K(x_{t},x^{\prime}_{t})\exp\left((r(x_{t})+r(x^{\prime}_{t}))/\alpha\right)}\tag{2}$$
where t < s. As a consequence, the importance weight wn and the acceptance rate β are as follows:
* [16] M. C.  
$\overline{\phantom{\rule{0.000pt}{0ex}}}$
wn = N
−1, β = min(1, exp((r(x
′
t) − r(xt)/α)). (3)
Intuitively, the importance weight wn corresponds to uniform sampling over the proposals, while the acceptance rate β ensures that the overall procedure converges toward reward-aligned sampling.

This configuration improves the overall efficiency of the algorithm, as further detailed in Section 3.3.

Intermediate rewards r(xt) can approximate by evaluating the reward function on the diffusion model's prediction of x0 (Li et al., 2024; Singhal et al., 2025).

By applying MTM with the given kernel and balancing function, we establish the following convergence guarantee, showing that intermediate distribution, even if unaligned at the outset, can asymptotically converge to the optimal distribution p
∗(xt):
Algorithm 1 Multiple-Try Metropolis 1: **Require:** Transition kernel K(xt, ·), current state x, number of trial, target distribution p
∗(·)
2: *Proposal and Selection*: Draw N i.i.d. trial x
′(1)
t*, . . . , x*
′(N)
tfrom the transition kernel K(xt, ·),
apply weighted sampling with importance weight wn

$$x_{t}^{\prime}\sim\mathrm{Multinomial}\left(\left\{\frac{p^{\star}(x_{t}^{\prime(n)})K(x_{t}^{\prime(n)},x_{t})\boldsymbol{\lambda}(\boldsymbol{x_{t}^{\prime}},\boldsymbol{x_{t}})}{\sum_{j=1}^{N}p^{\star}(x_{t}^{\prime(j)})K(x_{t}^{\prime(j)},x_{t})\boldsymbol{\lambda}(\boldsymbol{x_{t}^{\prime}},\boldsymbol{x_{t}})}\right\}_{n=1}^{N}\right)$$

The Multiple-Try Metropolis. The Multiple-Try Metropolis (Liu et al., 2000) is a Markov chain Monte Carlo method that can be efficiently parallelized. MTM conducts rejection sampling based on the transition kernel, thereby forming a Markov chain that asymptotically converges to the target distribution. At each iteration, a set of proposals is drawn from the transition kernel K, and one of them is selected according to its importance weights. Subsequently, backward proposals are generated to ensure detailed balance. In this framework, the transition kernel K defines how proposals are produced, the balancing function λ, a freely chosen non-negative symmetric function that adjusts the proposal weights to facilitate tractable sampling, and the acceptance ratio β determines whether the chain moves to the selected proposal. The complete sampling procedure of MTM is formalized in Algorithm 1, and a more detailed explanation of the Metropolis algorithm is provided in Appendix E.

$$({\mathfrak{I}})$$
))  $|J|$ . 
The derivations of the importance weight and the acceptance rate are provided in Appendix D.2.

216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 1: **Input:** Reward model r(·); denoisers {pθ(· | xt)}
1 t=T; transition kernel K(xt, ·); hyperparameters *α, N, k*; effective timestep set U ⊆ {*T, . . . ,* 1}
2: **Initialize:** masked sequence xT
3: for t = *T, . . . ,* 1 do 4: if t ∈ U **then** ▷ reward-guided refinement at timestep t 5: for i = 1*, . . . , k* do 6: Propose N candidates {x
′(n)
t }
N
n=1 ∼ K(xt, ·)
7: Compute weights wn and select x
′t by weighted sampling with wn (Eq. 3)
8: Propose N−1 auxiliary samples {x
′′(n)
t }
N−1 n=1 ∼ K(x
′t
, ·) and set x
′′(N)
t = xt 9: Accept x cand t with probability β; if accepted set xt ← x
′t(Eq. 3)
10: Sample one-step denoising to proceed: xt−1 ∼ pθ(· | xt)
11: **else**
12: Sample one-step denoising: xt−1 ∼ pθ(· | xt)
Proposition 1 (Convergence of MTM to the Optimal Distribution). Let xt be a sample drawn from a distribution that is not reward-aligned. Assume that q and pθ form a reversible Markov kernel. By applying MTM with the transition kernel K and balancing function λ defined above, the resulting Markov chain satisfies the detailed balance condition. Moreover, as the number of iterations k→∞,
the chain converges to the optimal distribution p
∗(xt).

Proof. The complete proof is available in Appendix D.4.

## 3.2 Algorithmic Procedure

Because MTM can be applied to intermediate states within the sampling process, it imposes no constraints on the transitions at each stage of denoising. Thus, at steps where IterRef is not applied, denoising can be performed using SMC or importance-sampling. Since the refinement step can in principle be applied at every timestep, one may flexibly define an effective timestep set U and restrict the application of MTM only to selected stages. This flexibility allows us to balance computational cost and refinement effectiveness, adapting to the needs of different tasks or resource budgets. Algorithm 2 elaborates the pseudocode of our method. We initialize the masked input at step T (Line 2). For timesteps in the effective set U, we perform a k-step MTM refinement loop (Lines 5–
9): at each refinement step, we draw N candidates from K(xt, ·) (Line 6), select a candidate x
′t by reward-weighted sampling using wn (Eq. 3, line 7) the selected state directly serves as the proposal for the acceptance test and then generate N −1 auxiliary proposals from K(x
′t, ·) and append the current state xt as the N-th backward element (Line 8). The proposal is accepted with probability β (Eq. 3); upon acceptance we set xt ← x
′
t(Line 9). After completing the k refinements, we proceed with a one-step denoising update xt−1 ∼ pθ(· | xt) (Line 10). For timesteps outside U, we simply apply the one-step denoising update (Line 12). The overall process iterates over timesteps T (Lines 3–12). This structure preserves detailed balance at each refinement step while exposing clear compute knobs via k and U.

## 3.3 Practical Implementation And Complexity Analysis

In practice, the primary computational bottleneck of IterRef arises from the need to generate both forward proposals and backward auxiliary proposals at each refinement step.

To mitigate this cost, we adopt the following strategies:
- **Balancing Function and Pool Reuse.** Through an appropriate choice of the balancing function in Equation 2, the acceptance rate can be evaluated without the need for resampled proposals x
′′
t,while still preserving the theoretical guarantees of the MTM framework. Consequently, the practical implementation eliminates the resampling step and reduces the per-iteration cost by nearly half.

Algorithm 2 IterRef with k-step MTM Refinement 270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323 In addition, when a proposal is rejected, we simply reuse the previously generated sampling pool. Since the candidates were already drawn i.i.d. from the same transition kernel, the pool remains a valid proposal set, and no additional sampling is required. This reuse further reduces the computational overhead otherwise incurred by repeatedly generating new candidates.

- **Selective Refinement via Effective Timesteps.** The refinement is applied only at a subset of timesteps, determined by the effective set U. This allows one to trade off between computational cost and refinement accuracy by controlling the density of refinement steps along the denoising trajectory. In Section 4.4, we present the performance analysis in different application time steps.

Compared to existing particle-based approaches such as SMC, IterRef provides a more flexible refinement mechanism that allows computational cost to be concentrated where it is most effective. Particle-based methods propagating multiple trajectories throughout the entire denoising process naturally incur substantial overhead, whereas IterRef can be applied selectively to an arbitrary subset of timesteps, enabling localized allocation of computational resources. The computational structure of IterRef can be summarized as follows. When IterRef is applied at timestep t, each proposal must be refined over the remaining (s − t) steps, resulting in N(s − t) diffusion-model calls, along with an additional N reward-model evaluations required for computing the acceptance ratio. The relative contribution of these components depends on the model scale. In large generative models such as LLaDA-8B, diffusion-model calls dominate the computational cost, while in smaller discrete diffusion models such as MDLM, the reward model and the generative model have comparable computational footprints. Consequently, aggregating these into a single NFE value may obscure meaningful differences, and it is preferable to report generative-model calls and reward-model calls separately. Appendix C.4 provides a wall-clock time analysis comparing IterRef with baseline methods.

## 4 Experiments 4.1 Experimental Setup

Models. For language generation, we use two diffusion language models, MDLM (Sahoo et al., 2024) and LLaDA-8B (Nie et al., 2025), as discrete diffusion backbones. For image generation, we adopt MaskGIT (Chang et al., 2022). More details for each model are presented in Appendix B.3. Tasks. For language generation setting, we use 3 seed, 15 controllable prompts from Han et al. (2022), each sampled 20 times, and calcuate the mean score. To guide the generation process, we utilize four reward functions: (1) *Toxicity Classifier* (Logacheva et al., 2022), which penalizes toxic or harmful content; (2) *Sentiment Classifier* (Barbieri et al., 2020), which encourages outputs with a desired polarity (e.g., positive); (3) *Perplexity* computed by GPT-2 (Radford et al., 2019), serving as a proxy for fluency; and (4) *Linguistic Acceptability* (Morris et al., 2020), which favors grammatically well-formed sentences. For image generation setting, we conduct 50k conditional generations over randomly selected classes from ImageNet (Deng et al., 2009), with reward provided by *CLIPScore* (Hessel et al., 2021). Further details on the tasks are provided in Appendix B.2.

Baselines. We compare **IterRef** with four inferece-time guidance baselines: **Best-of-N (BoN)**, the simplest method that generalizes across language and image domains; **Search-over-Path (SoP)**(Ma et al., 2025), a highly effective method in continuous diffusion; **SVDD**(Li et al., 2024), a widely adopted approach for guided generation; and **FK Steering** (Singhal et al., 2025), a recently proposed approach applicable across language and image domains. Implementation Details. To ensure fairness, we compare IterRef and each baseline under the same computational budget, with configurations aligned to the settings in Singhal et al. (2025). In measuring inference compute cost, we use *numbers of function evaluations (NFEs)* (More & Wild, ´ 2009), and treat the reward model and the generative model on equal footing. The denoising steps are fixed to 1000 for MDLM, 64 for LLaDA, and 50 for MaskGIT. The hyperparameters for baselines are favorably configured by following the original papers.

![6_image_0.png](6_image_0.png) 

## 4.2 Inference-Time Guidance For Diffusion Language Models

MDLM Results. Figure 2(a) shows the results with MDLM on four guided generation tasks under varying the inference cost. Overall, IterRef consistently outperforms other baselines across all settings, showing the best scaling effect. Interestingly, on Sentiment, CoLA, and Perplexity, IterRef achieves higher reward scores with only 2T NFEs than all baselines obtain with 32T NFEs, indicating the effectiveness of the iterative noising–denoising process in guiding discrete diffusion. On Toxicity, IterRef with only 4T NFEs matches the reward score of FK with 32T NFEs, resulting in nearly an 8× **faster** inference-time scaling.

LLaDA Results. Figure 2(b) shows the performance with LLaDA-8B. Similarly, IterRef consistently outperforms baselines across most compute costs on Toxicity, CoLA, and Perplexity. However, on CoLA, Best-of-N (BoN) achieves larger gains, which can be attributed to the fact that LLaDA already generates a linguistically well-formed text, making reward-guided corrections on unstable intermediate states less effective. Notably, with LLaDA, the performance gap of IterRef over baselines became more pronounced as NFEs increased, whereas with MDLM, larger gains appeared at lower NFEs. For instance, on Toxicity, with the MDLM backbone, the reward of IterRef at 32T NFEs was similar to that at 8T NFEs, thereby narrowing the gap with FK.

## 4.3 Inference-Time Guidance For Discrete Image Diffusion Model

We further validated our approach in a different modality by applying IterRef to the discrete image diffusion model MaskGIT, using CLIPScore as the reward model. As shown in Table 1, which reports results against baselines under varying cost budgets, the effectiveness of our method is again confirmed, highlighting its versatility across modalities. Beyond quantitative results, we also provide qualitative comparisons in Figure 3. These examples illustrate that IterRef consistently enhances visual fidelity and semantic alignment with textual prompts, compared to baseline sampling methods. Furthermore, to assess whether the observed 324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377 378 379 380 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431 Table 1: **Quantitative Results with MaskGIT.** We compare IterRef with baselines under varying computational costs, guided by CLIPScore. Iter- Ref performs the best across all settings.

| CLIPScore↑   | 1    | 2    | 4    | 8    | 16   |
|--------------|------|------|------|------|------|
| BoN          | 30.5 | 32.1 | 33.2 | 34.0 | 34.7 |
| FK           | 30.5 | 32.1 | 33.2 | 34.1 | 34.8 |
| SoP          | 30.5 | 30.7 | 32.1 | 33.5 | 34.4 |
| SVDD         | 30.5 | 31.7 | 32.5 | 33.2 | 33.8 |
| IterRef      | 30.5 | 33.7 | 34.4 | 35.2 | 35.8 |

![7_image_0.png](7_image_0.png)

![7_image_1.png](7_image_1.png)

Figure 4: Scailing effects of MDLM with N and k. The figure illustrates the trade-off between iteration count k and candidates N. Increasing k consistently yields greater performance gains than increasing N, demonstrating the efficacy of iteration.

Table 2: **Effect of timesteps applying IterRef.** 'Evenly' denotes applying IterRef evenly at every timestep under the same total cost. 0.1T corresponds to a later stage as denoising proceeds from T to 0.

Applied Steps 0.9T 0.7T 0.5T *0.3T* 0.1T Evenly

Toxic↑ 7.0 13.0 16.3 21.0 37.6 **65.0** Sentiment↑ 30.5 30.7 32.1 33.5 37.6 **97.0** CoLA↑ 23.3 33.3 48.6 66.3 **87.0** 83.0 Perplexity↓ 68.9 54.4 52.2 46.9 39.5 **18.4**

| k   | N   | Toxic.↑   | CoLA↑   | Senti.↑   |
|-----|-----|-----------|---------|-----------|
| 1   | 32  | 3.3       | 8.7     | 5.0       |
| 2   | 16  | 22.2      | 35.0    | 30.0      |
| 4   | 8   | 46.7      | 57.3    | 57.4      |
| 8   | 4   | 54.0      | 85.3    | 74.0      |
| 16  | 2   | 48.0      | 75.3    | 74.7      |
| 32  | 1   | 34.3      | 63.0    | 62.0      |

improvements persist under human-aligned evaluation criteria, we report ImageReward (Xu et al., 2023) scores in Appendix C.1.

## 4.4 Analysis

Scaling Effects. We examine the scaling effect with respect to the number of iterations kand the number of proposed candidates N at each iteration. The experiments are conducted on four tasks using MDLM under the same setting as the main experiment. As shown in Figure 4, increasing the number of iterations k and candidates N consistently leads to performance improvements. Further experimental details are provided in Appendix B.4. Effective Timestep Search. The effectiveness of diffusion inference-time guidance is known to be sensitive to the step at which it is applied. For example, in continuous diffusion, when applying classifier-free guidance (CFG;(Ho & Salimans, 2022)), much of the content is determined at the early steps (Choi et al., 2022; Li et al., 2023; Wang & Vastola, 2023). Thus, we study at which diffusion step IterRef can more effectively guide discrete diffusion. Specifically, we evaluate the performance of MDLM when applying IterRef at different steps 0.9*T, . . . ,* 0.1T, where 0.1T refers to a later stage as denoising proceeds from T to 0. We fix the total computational budget by allocating 4T NFEs at each selected step.

8

![8_image_0.png](8_image_0.png) 

![8_image_1.png](8_image_1.png)

As shown in Table 2 across all tasks, IterRef at the later denoising stages consistently shows better performance than those applied at earlier stages. Interestingly, while IterRef applied evenly throughout denoising achieves the best results on Toxic, Sentiment, and Perplexity, IterRef applied only at 0.1T outperforms the balanced one on CoLA. Note that, this effectiveness in the later stage differs from continuous diffusion, where most of the content is determined in early sampling stages.

432 433 434 435 436 437 438 439 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485 Number of Iterations k **vs Number of Particles** N. We study the effect of the number of iterations k and particles N on the performance of IterRef. As shown in Table 3, increasing iterations is more effective than simply generating more particles. This observation indicates that the reward from additional particles remains largely similar, while iterative refinement progressively shifts the distribution toward better alignment. The results highlight the importance of an iterative approach and further emphasize the effectiveness of our method in achieving reward alignment.

## 4.5 Case Study: Safety Alignment For Llada-8B

While large language models often exhibit an inherent ability to reduce toxic generations, such capability remains imperfect and insufficient for safety-critical applications. Even residual toxicity can propagate harmful content and undermine user trust, highlighting the necessity of complete and reliable detoxification. Accordingly, in this case study we demonstrate that our method effectively mitigates this issue, showcasing its potential for robust alignment in safety-alignment scenarios (Geva et al., 2022; Liu et al., 2023; Youssef et al., 2025). To evaluate the effectiveness of IterRef in safety-critical scenarios, we conduct experiments on detoxification with the LLaDA-8B. Specifically, we adopt toxic prompts curated from RealToxicityPrompts (Gehman et al., 2020). To rigorously assess detoxification performance, we select 15 prompts with the highest toxicity scores and generate sequences with 20 samples per prompt, resulting in a total of 300 generations for evaluation. The evaluation metric is the proportion of generated sentences that are classified as toxic. The experimental results presented in Figure 5(a) demonstrate that IterRef achieves superior performance, notably reducing toxicity to below 10% starting from 4× computational budget. The performance gap between our method and baseline approaches consistently remains near 10% from 4 budget onwards. Figure 5(b) illustrates representative examples where detoxification effectively operates. We observe a tendency to reduce toxicity by completing sentences as if they were quoted speech from someone else.

## 5 Related Works

Discrete Diffusion Models and Scaling Building on advances in continuous diffusion models, research on discrete diffusion (Campbell et al., 2022; Sahoo et al., 2024) has accelerated as the framework was extended to discrete state spaces (Sahoo et al., 2024; Nie et al., 2025).While inferencetime scaling has been extensively studied in autoregressive LLMs that boosting compute during gen486 487 488 489 490 491 492 493 494 495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539 eration often proves more efficient than training-time scaling (Snell et al., 2024) analogous strategies for discrete diffusion models are comparatively less explored. In continuous diffusion, the variability introduced by Gaussian noise strongly shapes generation (Ahn et al., 2024; Qi et al., 2024), motivating test-time scaling via searches over noise trajectories (Ma et al., 2025; Zhang et al., 2025; Mao et al., 2023). Inspired by this perspective, analogous test-time scaling for discrete diffusion is realized through particle-based search; for example, FK steering (Singhal et al., 2025) resamples particles using potential functions to bias trajectories toward desirable regions. Nevertheless, Discrete diffusion faces unique challenges: token discretization prevents direct gradient usage, and incorrectly generated tokens cannot be corrected in subsequent steps. Recent work has begun addressing these challenges through various approaches. Wang et al. (2025) using re-masking in masked models, where tokens are strategically re-masked and unmasked at intermediate timesteps to enable error correction and exploration of alternative token configurations that would otherwise be fixed once generated, effectively circumventing the irreversibility problem inherent to discrete diffusion. Reward-Guided Generation. Reward-guided generation aims to maximize the reward while preserving the naturalness of the samples. Several studies have explored this direction, including SMC- based guidance (Wu et al., 2023; Dou & Song, 2024), which combines generation with SMC (Doucet et al., 2001), and SVDD (Li et al., 2024), which employs importance sampling for guidance. PG- DLM (Dang et al., 2025) applies Particle Gibbs sampling, repeatedly resampling the entire trajectory multiple times. In another line of work, the reward-guided generation process has been reformulated as a search problem. DSearch (Li et al., 2025) reframes inference-time alignment as a search procedure, dynamically adjusting the beam width and tree expansion. DTS (Jain et al., 2025) improves the soft value of intermediate states through Monte Carlo Tree Search–based value backup, thereby optimizing path selection. All these methods share a common focus: selecting better samples along the denoising trajectory or exploring superior paths. In contrast, IterRef does not search over trajectories nor maintain multiple trajectories. Instead, it leverages the noising–denoising structure of discrete diffusion to iteratively refine the current state itself.

## 6 Conclusion

We introduced Iterative Reward-Guided Refinement (**IterRef**), a test-time scaling framework for discrete diffusion that performs reward-guided iterative refinement via Multiple-Try Metropolis.

The proposed method improves the distribution through iterative updates at intermediate stages, thereby overcoming the limitation of prior approaches that struggle with mid-trajectory correction, while also allowing cost to be concentrated by adaptively selecting application points according to task characteristics. We demonstrated that our method is theoretically well-founded and practically robust, with strong empirical results across a wide range of modalities and tasks.

## Ethics Statement

We follow the ICLR Code of Ethics. Our work intentionally includes experiments that increase the toxicity of generated text in order to stress-test discrete diffusion–based language models and analyze their robustness under adversarial or misaligned conditions. We acknowledge that such experiments may appear unusual from a safety standpoint. However, these evaluations are conducted exclusively for research purposes, and no toxic outputs are used for deployment, user-facing settings, or model training. The goal of these evaluations is not to enable harmful generation, but to identify failure modes, diagnose reward over-optimization, and better understand where controllable generation methods may break down. By disclosing our findings transparently while restricting access to harmful content, we aim to contribute to the development of safer and more robust generative models.

## Reproducibility Statement

We provide hyperparameter details and setup of all experiments in Section 4.1 and Appendix B.4.

## References

540 541 542 543 544 545 546 547 548 549 550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 Donghoon Ahn, Jiwon Kang, Sanghyun Lee, Jaewon Min, Minjae Kim, Wooseok Jang, Hyoungwon Cho, Sayak Paul, SeonHwa Kim, Eunju Cha, et al. A noise is worth diffusion guidance. arXiv preprint arXiv:2412.03895, 2024.

Jacob Austin, Daniel D Johnson, Jonathan Ho, Daniel Tarlow, and Rianne Van Den Berg. Structured denoising diffusion models in discrete state-spaces. Advances in neural information processing systems, 34:17981–17993, 2021.

Mor Geva, Avi Caciularu, Kevin Ro Wang, and Yoav Goldberg. Transformer feed-forward layers build predictions by promoting concepts in the vocabulary space. *arXiv preprint* arXiv:2203.14680, 2022.

Francesco Barbieri, Jose Camacho-Collados, Leonardo Neves, and Luis Espinosa-Anke. Tweeteval: Unified benchmark and comparative evaluation for tweet classification. arXiv preprint arXiv:2010.12421, 2020.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. *Advances in neural information processing systems*, 33:1877–1901, 2020.

Andrew Campbell, Joe Benton, Valentin De Bortoli, Thomas Rainforth, George Deligiannidis, and Arnaud Doucet. A continuous time framework for discrete denoising models. Advances in Neural Information Processing Systems, 35:28266–28279, 2022.

Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, and William T Freeman. Maskgit: Masked generative image transformer. In *Proceedings of the IEEE/CVF conference on computer vision and pattern* recognition, pp. 11315–11325, 2022.

Ciprian Chelba, Tomas Mikolov, Mike Schuster, Qi Ge, Thorsten Brants, Phillipp Koehn, and Tony Robinson. One billion word benchmark for measuring progress in statistical language modeling.

arXiv preprint arXiv:1312.3005, 2013.

Jooyoung Choi, Jungbeom Lee, Chaehun Shin, Sungwon Kim, Hyunwoo Kim, and Sungroh Yoon.

Perception prioritized training of diffusion models. In *Proceedings of the IEEE/CVF conference* on computer vision and pattern recognition, pp. 11472–11481, 2022.

Meihua Dang, Jiaqi Han, Minkai Xu, Kai Xu, Akash Srivastava, and Stefano Ermon. Inferencetime scaling of diffusion language models with particle gibbs sampling. *arXiv preprint* arXiv:2507.08390, 2025.

Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In *2009 IEEE conference on computer vision and pattern recognition*, pp. 248–255. Ieee, 2009.

Zehao Dou and Yang Song. Diffusion posterior sampling for linear inverse problem solving: A
filtering perspective. In *The Twelfth International Conference on Learning Representations*, 2024.

Arnaud Doucet, Nando De Freitas, and Neil Gordon. An introduction to sequential monte carlo methods. In *Sequential Monte Carlo methods in practice*, pp. 3–14. Springer, 2001.

Samuel Gehman, Suchin Gururangan, Maarten Sap, Yejin Choi, and Noah A Smith. Realtoxicityprompts: Evaluating neural toxic degeneration in language models. *arXiv preprint* arXiv:2009.11462, 2020.

Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 12873–12883, 2021.

Aaron Gokaslan and Vanya Cohen. Openwebtext corpus. http://Skylion007.github.io/
OpenWebTextCorpus, 2019.

594 595 596 597 598 599 600 601 602 603 604 605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 Adam Johansen. A tutorial on particle filtering and smoothing: Fifteen years later. 2009. Lijiang Li, Huixia Li, Xiawu Zheng, Jie Wu, Xuefeng Xiao, Rui Wang, Min Zheng, Xin Pan, Fei Chao, and Rongrong Ji. Autodiffusion: Training-free optimization of time steps and architectures for automated diffusion model acceleration. In *Proceedings of the IEEE/CVF International* Conference on Computer Vision, pp. 7105–7114, 2023.

Xiner Li, Yulai Zhao, Chenyu Wang, Gabriele Scalia, Gokcen Eraslan, Surag Nair, Tommaso Biancalani, Shuiwang Ji, Aviv Regev, Sergey Levine, et al. Derivative-free guidance in continuous and discrete diffusion models with soft value-based decoding. *arXiv preprint arXiv:2408.08252*, 2024.

Xiner Li, Masatoshi Uehara, Xingyu Su, Gabriele Scalia, Tommaso Biancalani, Aviv Regev, Sergey Levine, and Shuiwang Ji. Dynamic search for inference-time alignment in diffusion models. arXiv preprint arXiv:2503.02039, 2025.

Jun S Liu, Faming Liang, and Wing Hung Wong. The multiple-try method and local optimization in metropolis sampling. *Journal of the American Statistical Association*, 95(449):121–134, 2000.

Sheng Liu, Haotian Ye, Lei Xing, and James Zou. In-context vectors: Making in context learning more effective and controllable through latent space steering. *arXiv preprint arXiv:2311.06668*, 2023.

Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. Roberta: A robustly optimized bert pretraining approach. *arXiv preprint arXiv:1907.11692*, 2019.

Varvara Logacheva, Daryna Dementieva, Sergey Ustyantsev, Daniil Moskovskiy, David Dale, Irina Krotova, Nikita Semenov, and Alexander Panchenko. Paradetox: Detoxification with parallel data. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics
(Volume 1: Long Papers), pp. 6804–6818, 2022.

Nanye Ma, Shangyuan Tong, Haolin Jia, Hexiang Hu, Yu-Chuan Su, Mingda Zhang, Xuan Yang, Yandong Li, Tommi Jaakkola, Xuhui Jia, et al. Inference-time scaling for diffusion models beyond scaling denoising steps. *arXiv preprint arXiv:2501.09732*, 2025.

Jiafeng Mao, Xueting Wang, and Kiyoharu Aizawa. Guided image synthesis via initial image editing in diffusion model. In *Proceedings of the 31st ACM International Conference on Multimedia*, pp. 5321–5329, 2023.

Nicholas Metropolis, Arianna W Rosenbluth, Marshall N Rosenbluth, Augusta H Teller, and Edward Teller. Equation of state calculations by fast computing machines. *The journal of chemical* physics, 21(6):1087–1092, 1953.

Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. Clipscore: A
reference-free evaluation metric for image captioning. *arXiv preprint arXiv:2104.08718*, 2021.

Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022.

Vineet Jain, Kusha Sareen, Mohammad Pedramfar, and Siamak Ravanbakhsh. Diffusion tree sampling: Scalable inference-time alignment of diffusion models. *arXiv preprint arXiv:2506.20701*, 2025.

Xiaochuang Han, Sachin Kumar, and Yulia Tsvetkov. Ssd-lm: Semi-autoregressive simplexbased diffusion language model for text generation and modular control. *arXiv preprint* arXiv:2210.17432, 2022.

W Keith Hastings. Monte carlo sampling methods using markov chains and their applications. 1970.
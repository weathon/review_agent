## Summary
This paper introduces TD-JEPA, a zero-shot unsupervised RL method that learns policy-conditioned latent-predictive representations from offline, reward-free transitions using a temporal-difference objective. The key idea is to train state and task encoders together with a predictor that approximates successor features in latent space, yielding zero-shot policies for rewards expressible through the learned task representation; empirically, the method is competitive across 65 tasks and is especially strong in pixel-based settings.

## Strengths
- **A genuinely nontrivial unification of latent prediction and successor-feature zero-shot RL.** The paper does more than add an auxiliary JEPA-style loss: the latent-predictive TD objective is the core training signal used to learn encoders, predictors, and policies jointly. This is a specific conceptual contribution, clearly articulated in Sec. 3.3: “the predictor may be leveraged as an approximation of successor features … to extract policies … for all reward functions in the span of the learned features.”
- **Theoretical connection between TD latent prediction and successor-measure factorization is novel and substantive.** The paper proves that, in an idealized setting, the learned predictors recover projected successor features / successor-measure factorizations, and relates the practical TD objective to forward/backward TD losses (Thms. 1–4). Even with strong assumptions, this is more than generic intuition: it gives a precise bridge from non-contrastive latent prediction to zero-shot value estimation.
- **Empirical gains are strongest exactly where they matter most: pixel-based zero-shot RL.** In Table 1, TD-JEPA is consistently at or near the top on DMCRGB and OGBenchRGB, which are the hardest settings for this class of methods. The paper also uses probability-of-improvement plots (Fig. 2) to argue consistency across domains rather than relying only on suite averages.
- **The baseline protocol is unusually careful and materially strengthens the empirical case.** The paper does not simply run prior methods “as is”; it standardizes architectures, adds explicit state encoders where beneficial, reports the effect of doing so (Appendix D.1 / Table 2), and is transparent that several compared methods are novel zero-shot instantiations of representation learners rather than native zero-shot methods. This is a specific fairness strength, not a generic “many baselines” claim.
- **The representation analysis goes beyond leaderboard reporting.** The paper includes ablations on multi-step/policy-aware prediction, symmetric vs asymmetric encoders, adaptation with frozen vs trainable representations, architecture-depth sweeps, and visualization of learned successor-geometry / goal alignment. These analyses support the claimed mechanism rather than only reporting final returns.
- **Learned representations appear reusable beyond zero-shot inference.** Figures 4 and 6 show that TD-JEPA pretraining improves sample efficiency for downstream offline/online adaptation, and that frozen representations are often already sufficient for rapid improvement.

## Weaknesses

###: Fatal

None.

### Major:
- **The paper overstates the scope of its “any reward” claim relative to what the method formally guarantees.**  
  The core mechanism in Sec. 3.3 requires projecting a downstream reward onto the learned task features via linear regression,
  \[
  z_r=\arg\min_z \mathbb{E}_{(s,r)\sim D_{\text{rwd}}}(r-\psi(s)^\top z)^2,
  \]
  and the method then returns policy \(\pi_{z_r}\). This means the practical guarantee is for rewards well represented by the span of \(\psi\), not arbitrary rewards in an unconstrained sense. The paper does partially acknowledge this repeatedly (“for all rewards in the span of \(\psi\)”, “the associated policy \(\pi_{z_r}\) is then returned”), and Theorem 4 is explicit about linear regression onto \(\psi\). However, the abstract still says “This enables zero-shot optimization of any reward function at test time” and the introduction similarly says “for any downstream reward, entirely in latent space.” The theory later refines this claim: exact zero-shot optimality for truly arbitrary rewards requires perfect successor-measure approximation and optimal policies for all linear rewards in \(\psi\)-space, which is a much stronger condition than the headline phrasing suggests. This is not a fatal flaw, but the claim should be narrowed to match the actual method and guarantees.
- **The theoretical results are informative but rest on assumptions that substantially limit their direct applicability to the practical algorithm.**  
  The main theorems in Sec. 4 assume orthonormal / identity-covariance representations, uniform state distributions, and symmetric transition kernels (A1–A3), plus linear predictors in a tabular setting. The non-collapse result in Theorem 2 further assumes a continuous-time relaxation where predictors are optimized to stationarity before representation updates. The paper does not hide this: it explicitly states the setting is “simplified,” notes these assumptions are inherited from prior latent-prediction analyses, and Appendix C discusses relaxations. Still, the practically important point remains that the strongest guarantees do **not** apply to the actual deep, off-policy, asymmetric, discrete-time training setup used in experiments. Appendix C also makes clear that removing symmetry in the cleanest way would require a backward-sampling variant that is “not easy to be optimized off-policy.” So the theory is valuable as structure and intuition, but the gap between theorem and practical algorithm is real and should be emphasized more plainly.
- **Offline robustness is not established as broadly as the paper’s framing suggests.**  
  The method is presented as learning from “offline, reward-free transitions,” and the main algorithm indeed bootstraps with actions sampled from the learned policy at next states. In the main experiments this works well on ExoRL and OGBench, but Appendix D.8 shows a meaningful limitation: on low-quality, low-coverage data, performance degrades and BC/FQL-style regularization becomes important. The paper does discuss dataset regimes (high-coverage ExoRL vs low-coverage OGBench) and in OGBench already uses BC-style regularization (“we additionally apply BC regularization in OGBench…”), so this is not an unacknowledged bug. Still, the main-text framing could better distinguish “works on the benchmarked offline datasets” from “robust on arbitrary offline reward-free data.” As written, the latter implication is stronger than the evidence supports.

### Minor
- **The asymmetric variant’s practical cost is nontrivial.**  
  TD-JEPA trains two encoders and two predictors. Table 4 shows that the asymmetric method is materially slower than the symmetric variant, often by a factor around 2–3x in steps/sec depending on suite. Given that the symmetric variant in Table 3 is often fairly competitive, a clearer compute/performance tradeoff discussion in the main text would strengthen the practical case.
- **The method appears somewhat sensitive to regularization and benchmark regime.**  
  Appendix E/Table 6 shows fairly different orthonormal regularization ranges across methods and domains, especially in OGBench navigation/manipulation. This does not invalidate the results—the authors are commendably transparent about tuning—but it suggests the method is not yet especially plug-and-play.
- **The main paper could foreground limitations of reward inference more explicitly.**  
  The reward-projection step is central to deployment yet mostly described as a linear regression recipe. The paper would benefit from a clearer discussion of when this step is expected to be reliable or fragile—for example, when \(\psi\) under-represents downstream rewards or when inference data are limited/noisy. This is especially relevant because the abstract-level claim is broad.
- **Some main-text performance narration is slightly stronger than the tables warrant.**  
  Overall the empirical story is good, but in several suites performance gaps are modest and confidence intervals overlap. The paper does mitigate this with probability-of-improvement analysis and overlap-aware bolding, which is good practice; still, some verbal claims could be phrased more conservatively.

### Trivial
- **Simulation-only evaluation limits demonstrated significance, though not the paper’s validity.**  
  The paper motivates real-world applications and cites humanoid/robotics directions, but all evidence here is from simulation benchmarks. This is acceptable for the paper’s scope, but real-world relevance remains prospective rather than demonstrated.

## Nice-to-Haves
- Add a main-text experiment or analysis isolating how violations of the theory assumptions (non-uniform data, asymmetric dynamics, predictor under-optimization) affect empirical stability.
- Include a more direct analysis of reward-inference robustness under limited or noisy rewarded samples at test time.
- Provide a clearer compute/performance Pareto comparison between asymmetric TD-JEPA, symmetric TD-JEPA, and strong baselines, since the symmetric variant is often competitive.
- Add more explicit failure-case analysis on underperforming tasks (e.g., harder OGBench domains), especially connecting those failures to representation geometry or reward misspecification.
- A direct frozen-representation downstream RL comparison against strong task-specific offline RL baselines would further clarify how much of the gain comes from zero-shot structure versus simply better pretraining.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“Missing comparison to PPO/SAC/CQL/IQL or other specific external baselines.”**  
  Removed because this is partly scope creep and partly unverifiable as a required omission. The paper is explicitly about zero-shot unsupervised RL and compares against a broad set of zero-shot / representation-learning baselines. While extra downstream-RL comparisons could be nice-to-have, framing the absence of particular named external methods as a core weakness is too strong.
- **“Ablation figures lack error bars/confidence intervals.”**  
  Removed as factually incorrect. Figure 3 explicitly says “Error bars represent standard errors,” and the appendix includes uncertainty reporting in multiple ablations/tables.
- **“The paper hides tuning ranges / exact hyperparameter sweeps.”**  
  Removed because the appendix provides substantial tuning details, including architecture hyperparameters (Table 5), regularization ranges (Table 6), and method-specific implementation details.
- **“The orthonormality regularizer is unexplained / not discussed at all.”**  
  Removed in strong form. The paper discusses collapse avoidance repeatedly, includes the regularizer in Algorithm 1, mentions its importance in relation to prior work, and provides theory on non-collapse in the idealized setting. It is fair to ask for more sensitivity analysis, but not fair to claim it is missing discussion entirely.
- **Generic strength: “the paper is well-written / experiments are extensive.”**  
  Removed because these are generic. The retained strengths point to specific unusual merits instead.

## Novel Insights
A key synthesis across the reviews and the paper itself is that TD-JEPA is strongest not merely because it is “non-contrastive,” but because it aligns the representation-learning target with **policy-conditional long-horizon occupancy** rather than behavior-policy prediction or generic future-state prediction. The appendix visualizations and the comparison against BYOL/BYOL-\(\gamma\) support a more precise interpretation: the method’s advantage in pixels likely comes from learning latents whose geometry is shaped by **directed control-relevant future behavior**, not just visual similarity or undirected visitation. This makes the paper less about swapping contrastive for JEPA, and more about choosing the right predictive object for zero-shot control.

## Suggestions
- Narrow the headline claim from “any reward” to “any reward representable or well-approximated in the span of the learned task encoder,” and state the stronger arbitrary-reward result only under the exact conditions of Theorem 4.
- Bring the low-coverage offline-data limitation from Appendix D.8 into the main paper, and explicitly position BC/FQL-style regularization as recommended in low-support regimes.
- In Sec. 4, add a short paragraph explicitly separating what the theory proves about the idealized linearized dynamics from what is only empirically validated for the practical deep algorithm.
- Add a concise main-text discussion of the asymmetric-vs-symmetric compute tradeoff using Table 3 + Table 4.
- Expand the test-time reward-inference discussion to address noisy/limited rewarded samples and possible conditioning issues in \(\mathbb{E}[\psi\psi^\top]\).
- If space allows, include one focused failure-case analysis on an OGBench task where the method is not best, linking failure to either reward misspecification, coverage, or representation limitations.


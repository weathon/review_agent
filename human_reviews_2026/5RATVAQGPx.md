# Jackpot: Align Actor-Policy Distribution for scalable and stable RL for LLM

- Decision: Accept (Poster)
- Scores: 6, 4, 4, 4

## Abstract
Reinforcement learning (RL) has become an increasingly important paradigm for improving large language models (LLMs) on alignment, reasoning, and coding tasks, yet it remains extremely costly. The majority of training time is spent on rollouts. Allowing actor and policy distributions to differ could unlock substantial scalability and efficiency benefits, such as supporting large-batch or asynchronous training, and even enabling a lightweight rollout model. However, existing importance sampling–based corrections for distribution mismatch suffer from an inherent trade-off between stability and training performance. To tackle this problem, we propose Jackpot, which leverages Optimal Budget Rejection Sampling to directly reduce the gap between actor and policy distributions. For efficiency and stability in practical training, We introduce an efficient probability estimation strategy based on Top-$K$ logits with batch bias correction, and designs a stabilized Jackpot-PPO loss that jointly accounts for both the importance sampling ratio and the trust-region constraint in PPO. Empirically, our method achieves stable improvements in large-batch and asynchronous training, and in extreme off-policy training it substantially delays the onset of collapse and delivers competitive performance. Specifically, we achieve 20\% improvement on AMC benchmarks and ~8\% AIME benchmarks over the off-policy baseline under 128$\times$ actor-policy update ratio for Qwen3-4B-Base and 64$\times$ for Qwen3-8B-Base, while achieving greater stability and better performance than prior off-policy RL methods under extreme settings.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
JACKPOT targets, actor–policy mismatch, the PPO pain point in LLM RL. It fixes the problem at sampling time with Optimal-Budget Rejection Sampling. The naïve OBRS recipe is unusable at scale because it needs full-vocab normalizers, yields unbounded correction weights, and can kill throughput when alignment is too strong. The paper’s value is a practical pipeline: estimate OBRS on a top-k union of actor/policy tokens and calibrate the missing mass with the simple identity that the normalizer equals the expected acceptance rate; then train with a stabilized PPO objective that factorizes the correction and clips each ratio so gradients stay bounded and trust-region friendly. Empirically it demonstrates it shines at where vanilla PPO struggles, large rollout batches at fixed minibatch, and extreme off-policy collection, delivering steadier learning and better scores without changing the base model or reward loop.

### Strengths
1. Targets PPO’s real pain points. It directly attacks actor–policy mismatch in the regimes that break PPO such as huge rollout batches with the same small minibatch and extreme off-policy actors, and shows steadier learning and better scores there.
2 Not a naïve OBRS port. The paper surfaces why a straight plug-in fails (full-vocab normalizers, unbounded weights, acceptance-throughput collapse) and then fixes each with a practical trio: top-k union for feasibility, acceptance-rate calibration for unbiased scale, and a factorized, clipped correction that keeps updates stable.
3. Clean narrative. The writing crisply motivates the problem, diagnoses naïve failures, and walks the reader through the recipe and its effects; the algorithm is easy to implement from the description.

### Weaknesses
1. Loss clipping bias. The stabilized objective clips two likelihood ratios to keep gradients tame. That’s an intentional bias; it can underweight rare-but-informative tokens and shrink effective step size. The paper argues stability, but it does not quantify when the bias alters policy improvement or exploration.
2. Actor–policy visitation skew. OBRS keeps tokens the current policy already likes, which shifts the behavior distribution. Alignment lowers variance, but it can also prune rare, hard states that matter for learning. Without coverage/diversity diagnostics, it is unclear whether OBRS quietly narrows what the policy ever sees.
3. Sequence or block-level OBRS. Tokenwise rejection is simple, but it fragments credit on long sequences where decisions cohere over spans. Span- or block-level acceptance could improve temporal coherence at the same keep-rate (see questions below).

### Questions
1. Clipping bias: compare unclipped vs clipped vs two-sided clipped weights on identical runs; report policy-improvement proxies, gradient norms, collapse rate.
2. Visitation skew: show state-coverage heatmaps before/after OBRS, effective sample size per update, entropy of the kept policy, and advantage-weighted coverage of rare regions.
3. Top-k curve: plot performance, gradient norms, and normalizer error versus k; provide a simple k rule or a dynamic k policy tied to a target acceptance band.
4. Potential KL "double counting"?: With OBRS plus the clipped reference ratio acting like a trust region, an explicit KL(pθ‖pref) may over-constrain updates. add curves with and without the explicit KL term to the reference under OBRS; clarify whether the second clipped ratio already suffices or if a reduced KL is still helpful.
5. Curious to see block-level OBRS.

### Soundness
3

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
3

### Summary
The paper introduces a new reinforcement learning framework aimed at improving the stability and efficiency of RL-based fine-tuning for large language models. The central problem addressed is the distribution mismatch between the actor (rollout) model and the policy being optimized, which often arises in large-batch, asynchronous, or off-policy training scenarios. Existing importance sampling corrections for this mismatch tend to suffer from instability or poor performance. To tackle this, the authors propose JACKPOT, a method that leverages Optimal Budget Rejection Sampling (OBRS) to directly reduce the KL divergence between the actor and policy distributions.

Empirical results on large-scale language models (Qwen3-4B and Qwen3-8B) demonstrate 20% improvement on AMC and 8% improvement on AIME benchmarks under extreme off-policy settings, as well as significant gains in large-batch asynchronous RL training. The method substantially delays training collapse and outperforms prior off-policy RL methods such as Truncated Importance Sampling.

### Strengths
1. The paper targets a concrete and important practical bottleneck: distribution mismatch between rollout models and the policy used for updates in RL for LLMs (large-batch, asynchronous, quantized/distilled rollouts, etc.). The motivation is clearly stated and practically relevant.
2. The paper proposes using Optimal Budgeted Rejection Sampling to directly reduce actor–policy KL, which is a principled move. The paper provides theoretical guarantees: OBRS is the unique budget-optimal token-wise accept/reject rule minimizing KL for a given acceptance budget, and the post-rejection distribution monotonically approaches the target as the scale parameter varies. These formal results strengthen the core idea.

### Weaknesses
1. The OBRS acceptance probability for a sampled token requires evaluating the ratio $p_{\theta_{new}} (a) / p_{inf} (a)$. In practice, this implies scoring candidate tokens under the *current policy* during data collection or otherwise obtaining these probabilities. However, the paper does not fully quantify the extra compute this incurs in real serving setups. This is important because the whole motivation is improving throughput; any extra compute could reduce or eliminate the gains.
2. The stabilized weight uses two clipped ratios and a stop-gradient on the alignment factor, trading bias for variance. The paper argues this is necessary for stability, but there is limited theoretical quantification of the bias introduced by clipping and the stop-gradient, nor diagnostics showing how much bias affects final performance in less extreme settings. More ablation (e.g., varying c1,c2, removing stop-grad) would help understand the tradeoffs.
3. The final stabilized Jackpot objective (Section 4.4) is a pragmatic heuristic. The paper admits that it "trades a small amount of bias (from clipping and approximation) for a massive reduction in variance". This factorization and clipping move the method away from the clean theoretical guarantees of OBRS. The paper does not provide any theoretical analysis of how this specific, biased re-weighting scheme affects the optimization landscape, convergence, or the final policy.

### Questions
1. In Algorithm 1, the acceptance probability $\alpha$ requires access to $p_{\theta_{new}}$ at sampling time. How is this computed in practice when rollouts are generated by a separate inference actor? Are logits from $p_{\theta_{new}}$ queried synchronously (which could double inference cost), or are they approximated, cached, or delayed? Please clarify how this fits into a large distributed RLHF or asynchronous setup, and quantify the extra compute or communication cost.
2. The stabilized Jackpot loss uses two clipped ratios and a stop-gradient. Could you formalize or empirically show how these choices affect gradient bias and variance? Specifically, how do c1,c2 and the stop-gradient operation trade off between bias and stability? It would help to include plots of gradient variance before and after stabilization.
3. The method is validated primarily on mathematical benchmarks (AMC, AIME, GSM8K, MATH-500). Evaluating on alignment or preference-based tasks (e.g., Anthropic-HH) would help test generality and robustness beyond numeric reasoning.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper aims to reduce the high rollout cost from large trajectories in reinforcement learning based post-training of LLMs while improving stability and scalability. The key idea is to sample trajectories from a lightweight actor policy different from the main training policy to make RL more efficient. To handle the resulting mismatch, the paper introduces a rejection-sampling–based approach that directly reduces the distribution gap between actor and policy. It further proposes a Top-K approximation with bias correction and a stabilized PPO loss to maintain trust-region stability. Experimental ablations shows promising gain over baselines.

### Strengths
- The idea of using simple light-weight policy for sampling trajectories is interesting and effective especially for larger policies
- bounding the KL and using rejection sampling helps in mitigating the distribution shift
- efficient Top-K probability estimation keeps it usable for large vocabularies.
- empirical performance are promising especially on AMC and AIME benchmarks compared to off-policy baselines.

### Weaknesses
- Although interesting, the idea is extremely similar to several parallel streams including speculative decoding, on policy distillation, weak-strong etc and the key novelty of the approach is not clear
- The paper ensures the KL divergence between the actor and policy distribution, however how it ensures closeness to the reference policy based on which standard RLHF policies are trained? How do you ensure closeness to that? Can you show a KL plot with the reference?
- It requires sampling multiple trajectories and computing the ratio of the probabilities, how much time it incurs additionally during training?
- Also, what kind of light-weight models can help to mitigate this shift in an efficient way - ie does compression or smaller finetuned models or distilled models, how are different models behaviour 
- How sensitive is the approach with the threshold and how is the threshold determined? That will also affect the number of trajectories to generate? It will be helpful to provide the details.

### Questions
See Weakness.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses the computational cost and instability arising from actor-policy distribution mismatch in reinforcement learning for large language models. While allowing these distributions to differ could enable significant efficiency gains—such as large-batch training, asynchronous updates, or using smaller rollout models—existing importance sampling-based corrections face a fundamental trade-off between stability and performance. The authors propose Jackpot, which leverages Optimal Budget Rejection Sampling (OBRS) to directly reduce the distributional gap between actor and policy networks. The method introduces three key contributions: an OBRS-based masking mechanism that maintains closer alignment between distributions, an efficient probability estimation strategy using Top-K logits with batch-wise bias correction to handle memory constraints, and a stabilized PPO loss jointly accounting for importance sampling ratios and trust-region constraints. Evaluated on large-batch training (128 mini-batches per rollout) and extreme off-policy scenarios, Jackpot demonstrates 20% improvement on AMC benchmarks and 8% on AIME benchmarks over baseline methods while maintaining stable training dynamics under severe distributional mismatch conditions.

### Strengths
- This paper tackles an important and practically relevant problem in reinforcement learning for LLMs: the gradient estimation bias and training instability caused by distribution mismatch between the actor and policy networks. 

- The proposed OBRS-based approach seems to be a principled solution to directly reduce this distributional gap.

-  The empirical evaluation demonstrates promising results, particularly in maintaining training stability.

### Weaknesses
- The paper lacks a detailed discussion of the differences between importance sampling ratio-based corrections and the rejection sampling approach, despite both relying on importance sampling ratio calculations. This makes it unclear why the optimal rejection sampling method provides advantages over existing correction techniques like TIS.

- The paper builds upon Optimal Budgeted Rejection Sampling (OBRS) but does not provide sufficient introduction or justification for this choice. Without adequate background and motivation, it is difficult for readers to accept that OBRS is the appropriate approach for addressing the distribution mismatch problem.

- The empirical evaluation lacks comprehensive ablation studies to disentangle the contributions of individual components. It remains unclear whether the OBRS-based masking mechanism, the Jackpot re-weighting strategy, or both components are essential for achieving the reported improvements in stability and performance.

### Questions
- The LaTeX format of this paper does not appear to follow the standard ICLR submission template.

- Does the Phase 1 Data Collection with OBRS require modifications to the sampling code in vLLM or other inference frameworks?

- How should the hyperparameters $c_1$ and $c_2$ be chosen? Is there any guidance provided for their selection. Furthermore, is the method robust to different choices of these hyperparameters?

### Soundness
3

### Presentation
2

### Contribution
2

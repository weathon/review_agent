---
job_id: 407b08fe-fb7b-4591-892e-92db58dcdf70
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: NfO2Lt2WY7.pdf
paper: Are Complicated Loss Functions Necessary for Teaching LLMs to Reason?
main_score_norm: 0.6
desk_reject: false
---
# Desk Rejection Assessment:

## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper analyzes and simplifies GRPO-style RL objectives for LLM reasoning, introduces a REINFORCE-based variant, and evaluates on math/STEM benchmarks, which fits squarely within reinforcement learning, representation learning for language, and optimization for LLMs.

## Minimum Quality
Pass ✅.  
All required sections (Abstract, Introduction, Related Work, Methodology/Approach embedded in Background + Experiments, Results, Discussion/Conclusion) are present and in English. The methods are reasonably described, experiments are non-trivial and on standard benchmarks, and there are no obvious fatal methodological errors or test leakage issues.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
No hidden prompts, manipulative instructions, or suspicious formatting targeting automated reviewers are visible in the provided content.

---

# Expected Review Outcome:

## Summary

The paper investigates which components of Group Relative Policy Optimization (GRPO) are actually needed to improve reasoning in large language models. Through controlled ablations, it studies (i) using only positive advantages, (ii) removing PPO-style clipping and ratios, and (iii) REINFORCE trained on raw rewards, and it compares these to RAFT and vanilla SFT on several math and STEM benchmarks. Based on this, the authors propose REINFORCE with Group Relative Advantage (RGRA), which keeps GRPO’s group-relative advantage but discards PPO-style constraints, and empirically show that RGRA is at least as stable as GRPO and often slightly outperforms it on Qwen2.5 (0.5B, 1.5B) and Llama3.2-1B.

## Strengths

1. **Clear and focused empirical question with practical impact.**  
   The paper tackles a concrete and relevant question for the RL-for-LLMs community: which parts of the now-standard GRPO loss are actually necessary for improving reasoning, and can we get away with a simpler REINFORCE-style objective. This is very aligned with how many practitioners are currently overengineering training pipelines.

2. **Systematic ablation of GRPO components.**  
   Section 3.2 carefully defines three variants: GRPO with positive-only advantages, RGRA (Eq. (2)), and REINFORCE with direct rewards, plus RAFT and SFT as non-RL baselines. The design isolates (i) the role of negative advantages, (ii) the role of advantage estimation per se, and (iii) the necessity of PPO-style clipping and ratios. This is more disciplined than the usual “yet another GRPO variant” story.

3. **Training-dynamics analysis across models, with clear visual evidence.**  
   Figure 1 (subfigures (a)–(f)) is a strong part of the paper. For all three models, the average reward and average response length plots neatly show that GRPO and RGRA maintain high rewards and long responses, whereas GRPO-pos and RAFT quickly collapse to near-zero response length, and REINFORCE on raw rewards collapses sharply even for larger models. This directly supports the claims about the indispensability of negative feedback and advantage estimation, and the non-necessity of clipping.

4. **Broad evaluation across multiple benchmarks and languages.**  
   Tables 1–3 cover GSM8K, MATH, OlympiadBench, AMC23, CMATH, CN-Middle-School, MMLU-STEM, and Gaokao2024 for three model families and six training regimes. Even if absolute gains are modest in places, this is a substantially more complete evaluation than many RLHF/GRPO variants that only report a couple of math datasets.

   - For instance, **Table 1** shows that RGRA achieves the best average Math-English accuracy for both Qwen2.5-0.5B (26.5 vs GRPO’s 25.6) and Qwen2.5-1.5B (38.3 vs 37.3), while being essentially tied with GRPO on Llama3.2-1B.
   - **Table 2** similarly shows RGRA outperforming GRPO on Qwen2.5-0.5B (55.1 vs 51.4 average) and Qwen2.5-1.5B (69.3 vs 65.7).
   - **Table 3** indicates that for STEM benchmarks, RGRA yields the strongest averages on Qwen2.5 (34.3 and 50.7) while GRPO remains slightly better on Llama3.2-1B.

   This breadth lends credibility to their “clipping not needed” story across tasks and languages.

5. **Concrete qualitative evidence of reasoning traces.**  
   Figure 2 provides qualitative samples from a Countdown-style task, contrasting a model that outputs a bare equation (`<answer> 35 - 19 + 44 </answer>`) with GRPO/RGRA models that produce multi-step explanations and self-correction. While anecdotal, this nicely illustrates the claimed emergence of explicit reasoning and is aligned with observed response-length statistics in Figure 1.

6. **Simple and practically implementable objective.**  
   The RGRA gradient in **Equation (2)** is a straightforward REINFORCE-style update with group-based standardized advantages plus a KL gradient term. Removing the policy ratio and clipping makes implementation easier and avoids some PPO-specific tuning/instability issues, which is attractive to practitioners who want a robust baseline without full PPO machinery.

7. **Reproducibility and experimental transparency.**  
   The paper gives a decent amount of detail: LoRA rank, group size, reward design, max tokens, and hyperparameters summarized in **Table 4**. The code link (Sec. 6) further supports reproducibility.

## Weaknesses

1. **Theoretical justification is minimal and leaves RGRA’s properties underexplored.**  
   The core claim is conceptual (“PPO-style clipping is not necessary”), but the analysis is entirely empirical. There is no attempt to connect Eq. (2) to standard policy-gradient theory or to clarify its on-/off-policy nature given that sampling for GRPO uses $\pi_{\theta_{\text{old}}}$ while RGRA samples from $\pi_\theta$ (Eq. (2) uses $\{o_i\}\sim \pi_\theta$, but in practice training uses offline samples). This mismatch between the formal gradient in Eq. (2) and the actual collection procedure (described in Sec. 3.1) is not discussed and creates conceptual confusion about what objective is actually optimized and whether the estimator is biased. Given that a main message is “we can drop PPO machinery,” the lack of any stability or convergence discussion for this non-standard REINFORCE-with-group-baseline update is a noticeable gap.

2. **Definition of REINFORCE and “REINFORCE with direct rewards” is underspecified and somewhat inconsistent.**  
   In Section 3.2, the paper describes “REINFORCE with Direct Rewards” as “start from RGRA, remove the group-relative advantage estimation, and train directly on the raw reward signal,” but there is no explicit formula analogous to Eq. (2). It is unclear whether:
   - the update uses per-token reward $r_{i,t}$ or trajectory-level reward $r_i$,
   - any baseline is used (seems not, but then what exactly is the scalar multiplying $\nabla_\theta \log \pi_\theta$),
   - and whether the same KL term as in RGRA is present.
   
   Given that a major claim is that “removing advantage estimation destabilizes learning,” the exact loss matters a lot. Right now, this variant could be interpreted in multiple ways, which weakens the conclusions about *why* it collapses.

3. **Off-policy vs on-policy subtleties are not addressed for GRPO and RGRA.**  
   In Section 2.2, GRPO is introduced with trajectories sampled from $\pi_{\theta_{\text{old}}}$ and ratios $r_{i,t} = \frac{\pi_\theta}{\pi_{\theta_{\text{old}}}}$. In contrast, RGRA’s Eq. (2) has sampling from $\pi_\theta(O\mid q)$ and no ratios. In the experiments, however, the implementation appears to reuse the group of 8 completions per prompt (Sec. 3.1) in a PPO-style sampling scheme, i.e., effectively off-policy with respect to $\pi_\theta$. If RGRA is actually implemented with off-policy batches (as is usual in RLHF/GRPO pipelines), Eq. (2) is not the estimator used. If instead they truly resample online under the current policy for each update, this should be spelled out explicitly. This ambiguity weakens the technical soundness of the algorithm description and the interpretability of comparative results.

4. **Claims about the necessity of negative feedback are stronger than what the experiments actually support.**  
   The paper attributes the collapse of GRPO-pos and RAFT mainly to “ignoring negative feedback” (p. 8). However:
   - RAFT differs from GRPO not only by discarding negatives but also by converting RL to supervised fine-tuning on the top-ranked sample with cross-entropy loss; the dynamics in **Figure 1(a–f)** for RAFT may therefore be affected by distributional shift and label leakage effects, not just missing negative advantages.
   - GRPO-pos uses an advantage truncated at zero, but its effective learning signal also depends on KL regularization and the reward scale; the observed collapse in Figure 1a/b for Qwen2.5-0.5B is very rapid, suggesting possible interactions with the fixed KL coefficient and learning rate that are not explored.
   
   Without ablations that vary reward scaling or KL strength specifically for GRPO-pos and RAFT, it is a bit ambitious to claim that ignoring negative feedback is *the* root cause rather than one factor among several.

5. **Scale and data regime are very limited for drawing broad conclusions.**  
   All experiments fine-tune tiny models (0.5B, 1.5B, and 1.0B) on only 1,800 GSM8K training examples. This is several orders of magnitude smaller than the regimes where GRPO and GRPO-like methods are usually used (DeepSeek-R1, DeepSeekMath, etc.). The paper does acknowledge hardware constraints in the conclusion, but many of its primary messages are stated in general terms (“PPO-style clipping is unnecessary,” “simpler REINFORCE-based approaches can effectively enhance reasoning in LLMs”) without clearly restricting claims to small models and tiny datasets. It is quite plausible that clipping becomes much more important at larger scale, under higher reward variance, or when the KL coefficient cannot be as strong.

6. **Magnitude of RGRA gains over GRPO is small and sometimes inconsistent.**  
   While RGRA frequently wins on averages, the gains over GRPO in Tables 1–3 are mostly 1–3 points and sometimes negative:

   - **Table 1**: For Qwen2.5-1.5B, on AMC23 GRPO gets 20.0 vs RGRA’s 17.5; for Llama3.2-1B, GRPO slightly outperforms RGRA on average (20.1 vs 20.2 is essentially noise-level).
   - **Table 2**: For CMATH at 1.5B, GRPO yields 75.0 vs RGRA’s 72.3.
   - **Table 3**: For Llama3.2-1B in English STEM, GRPO 32.6 vs RGRA 33.5 is again close, but in Chinese STEM, RGRA actually underperforms RAFT (11.4 vs 14.0) and GRPO (17.2).

   There is no statistical significance analysis, no multiple seeds, and no error bars. Without variance estimates, it is hard to know whether the reported 1–2 point differences supporting “RGRA has the potential to achieve stronger performance” are robust or just run-to-run noise.

7. **Limited diversity of reward functions and tasks.**  
   The reward is always a simple 0.1 format reward plus 1.0 correctness reward for GSM8K-style supervised math (Sec. 3.1). There is no exploration of:
   - partial credit rewards,
   - process-based rewards (e.g., stepwise CoT scoring),
   - non-math tasks where reward is noisier (e.g., preference models, safety),
   which is where PPO-style clipping is usually argued to matter most. Thus the conclusion that “PPO-style constraints are not required to improve mathematical reasoning or performance” is empirically supported only in a narrow, relatively low-variance supervised-math setting.

8. **Mathematical exposition of GRPO and RGRA has some notational and conceptual issues.**  
   - In **Equation (1)**, the loss $J_{\text{GRPO}}$ contains a “$\min[r_{i,t}\hat{A}_{i,t}, \text{clip}(r_{i,t}, 1-\epsilon, 1+\epsilon)\hat{A}_{i,t}]$” term, mimicking PPO, but the expectation and $\hat{A}_{i,t}$ definition use only trajectory-level rewards $r_i$ (before subscript $t$ appears). The paper never clarifies whether $\hat{A}_{i,t}$ is token-dependent or simply $\hat{A}_i$ repeated across tokens; the latter is standard in DeepSeekMath, but then writing $\hat{A}_{i,t}$ is misleading.
   - In the GRPO-pos variant, the text redefines $\hat{A}_{i,t}$ by truncation, but the displayed formula uses the same symbol on both sides of the case statement, which is confusing:
     \[
     \hat{A}_{i,t}=\begin{cases}\hat{A}_{i,t}&\text{if }\hat{A}_{i,t}>0\\
     0&\text{otherwise}\end{cases}
     \]
     It would be better to denote the truncated advantage as $\hat{A}^+_{i,t}$ to avoid circularity and to make clear what is actually zeroed in the implementation.
   - For RGRA in **Equation (2)**, the KL term is written as $-\beta \nabla_\theta D_{KL}[\pi_\theta \|\pi_{\text{ref}}]$ with no further detail; given the sequence setting, there should at least be an indication of whether this is token-wise KL over next-token distributions averaged over prompts, or sequence-level KL. This matters because its magnitude and interaction with advantages determine effective step sizes.

   These issues do not invalidate the methods, but they make it harder for a reader to precisely reconstruct and reason about the training objectives.

9. **Emergent reasoning analysis is thin and qualitative.**  
   The “Emergence of Reasoning Behaviors” subsection uses one dataset (Countdown) and a single illustrative example in **Figure 2**. There is no systematic measurement of reasoning length distributions, self-correction frequency, or CoT explicitness, beyond the average response-length curves in Figure 1. Given that the paper heavily emphasizes “teaching LLMs to reason,” it would be valuable to quantify reasoning behaviors more directly (e.g., what fraction of answers include multi-step explanations vs direct answers for each method).

10. **Some important recent analyses of GRPO are missing from related work.**  
    While the paper cites several variants (CPPO, Prefix Grouper, S-GRPO, GTPO, etc.), it omits more theoretical treatments of GRPO and its group-relative REINFORCE nature (see “Potentially Missing Related Work”), which weakens the positioning of RGRA with respect to known interpretations of GRPO as a kind of REINFORCE-with-baseline already.

## Potentially Missing Related Work

The following works are, based on their titles and topics, directly relevant and should be discussed:

1. **Yao et al., “Group-Relative REINFORCE Is Secretly an Off-Policy Algorithm: Demystifying Some Myths About GRPO and Its Friends” (2026).**  
   This work apparently analyzes GRPO from a REINFORCE/off-policy perspective, which is highly relevant to the present paper’s claim that a REINFORCE-style algorithm with group-relative advantages (RGRA) suffices. It should be cited in Section 2.1 and discussed near Eq. (2), clarifying how RGRA relates theoretically to group-relative REINFORCE and what additional assumptions or approximations are being made.

2. **Noguer I Alonso, “The Mathematics of Group Relative Policy Optimization: A Multi-Agent Reinforcement Learning Approach” (2025).**  
   Provides a mathematical treatment of GRPO which could clarify the role of group-relative baselines and help ground the paper’s interpretation that advantage estimation is essential while clipping is not. It should be included in the “Advancements and Limitations in GRPO” subsection and potentially referenced when introducing $\hat{A}_{i,t}$ on p. 4.

3. **Wu et al., “Multi-Scale Group Relative Policy Optimization for Large Language Models” (2025).**  
   Proposes a multi-scale GRPO variant for LLMs; this is directly comparable empirically and should be included in Section 2.1 as an additional GRPO-family method that tackles stability/efficiency, to better contextualize RGRA as a simplification rather than a new family.

4. **Zhang et al., “Scaf-GRPO: Scaffolded Group Relative Policy Optimization for Enhancing LLM Reasoning” (2025).**  
   Specifically targets reasoning improvements via a scaffolded GRPO structure. This is highly relevant to Section 2.1 and Section 4 (“Emergence of Reasoning Behaviors”) and should be discussed as an alternative approach for eliciting reasoning traces, especially since this paper claims RGRA can “teach LLMs to reason” with a simpler loss.

5. **Le et al., “Token-Regulated Group Relative Policy Optimization for Stable Reinforcement Learning in Large Language Models” (2025).**  
   Focuses on stability improvements via token-level regulation. Since this paper’s main rationale for clipping removal is observed stability at small scale, TR-GRPO-like approaches should be acknowledged and compared in Section 2.1 and Section 4 as complementary ways of achieving stability, particularly for large models where clipping or token-level regulation might still be advantageous.

6. **Mundada et al., “WS-GRPO: Weakly-Supervised Group-Relative Policy Optimization” (2025).**  
   Explores GRPO with weaker supervision signals. The present paper mostly handles fully verifiable math rewards; WS-GRPO’s setting would be useful to mention when discussing limitations in Sec. 5 and the need to explore more complex reward regimes, as well as in the related work.

## Questions

1. **Clarification of RGRA sampling and on-/off-policy nature.**  
   In practice, are batches for RGRA sampled from $\pi_\theta$ at each update (on-policy) or from a frozen “old” policy that is periodically updated as in PPO/GRPO? If the latter, how is Eq. (2) justified as an estimator, and have you observed any off-policy instability or drift?

2. **Exact definition of the REINFORCE-with-direct-rewards objective.**  
   Could you provide the explicit gradient or loss used for the REINFORCE baseline, including:
   - whether the reward is per-token or per-trajectory,
   - whether any baseline (e.g., moving-average reward) is used,
   - and whether a KL term analogous to RGRA’s $-\beta D_{KL}$ is included?  
   This would help isolate whether the collapse is due to absence of advantage estimation, absence of KL, or both.

3. **Variance across seeds and significance of RGRA vs GRPO differences.**  
   Are the experiments in Tables 1–3 run with a single random seed per configuration? If you have any multi-seed results, how large is the standard deviation in accuracy across runs, and do RGRA’s average gains over GRPO remain after accounting for variance?

4. **Effect of reward scaling and KL coefficient on GRPO-pos and RAFT.**  
   Have you tried different reward scales or KL coefficients specifically for the GRPO-pos and RAFT settings? For example, if you reduce the correctness reward magnitude or strengthen the KL penalty, do you still observe the same form of collapse in Figure 1, or is training salvaged? This would strengthen the claim that the key issue is ignoring negative feedback rather than generic instability.

5. **Token-level vs trajectory-level advantage in GRPO/RGRA.**  
   Is $\hat{A}_{i,t}$ implemented as a per-token estimate (e.g., process-based reward) or simply the same trajectory-level standardized reward $\hat{A}_i$ replicated across all $t$? If it is trajectory-level, can you confirm whether using token-level signal (e.g., reward shaping from partial correctness) changes the relative behavior of GRPO and RGRA?

6. **Scalability expectations.**  
   Based on your experience, do you expect RGRA to behave similarly on larger models (e.g., >7B) and larger datasets, or do you anticipate regimes where clipping and policy ratios become crucial again? Any preliminary experiments or theoretical arguments here would be very informative.

## Flag For Ethics Review

- No ethics review needed.  

## Details Of Ethics Concerns

N/A.

## Soundness Rating

3: good.  
The methods are mostly correct and well-implemented, and the empirical results support the main qualitative conclusions (negative feedback and advantage estimation matter more than clipping) within the explored regime. However, the lack of a precise treatment of on-/off-policy sampling, underspecified REINFORCE baseline, and absence of variance analysis leave some uncertainty about the generality and rigor of the claims.

## Presentation Rating

3: good.  
The paper is generally well written, with clear organization, helpful figures (especially Figure 1), and comprehensive tables (Tables 1–3 and 4). Some notation and definitions around advantages and objectives are sloppy, and the theoretical framing of RGRA could be sharper, but these are fixable.

## Contribution Rating

3: good.  
The contribution is primarily empirical and conceptual rather than algorithmically deep, but it targets a highly relevant question for RLHF/GRPO practice and provides useful evidence that REINFORCE-style training with group-relative advantages can match or outperform GRPO in small-model math settings. The simplification and analysis are valuable to the community, though not transformative.

## Overall Rating

6: Marginally above the acceptance threshold. But would not mind if paper is rejected.  
The work offers a thoughtful and surprisingly informative dissection of GRPO, together with a simple alternative (RGRA) that performs at least as well in the presented experiments. The limitations in scale, theory, and variance analysis prevent a higher recommendation, but the study is solid enough and practically relevant enough that I lean toward acceptance.

## Reviewer Confidence

4: confident.  
I am familiar with RLHF/GRPO literature and have checked the math and experimental design in reasonable detail. Some uncertainty remains around implementation specifics and off-policy aspects, but not enough to alter the overall judgment.
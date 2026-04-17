---
job_id: 35d2f1d3-2d27-4a88-98ea-d2af77ac8fb8
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: CTEXdHB1BB.pdf
paper: Conditional Advantage Estimation for Reinforcement Learning in Large Reasoning Models
main_score_norm: 0.6
desk_reject: false
---
# Desk Rejection Assessment:
## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper is squarely about reinforcement learning with verifiable rewards for large language models, focusing on advantage estimation and training signals like entropy and length. This is fully aligned with ICLR topics (RL, optimization, representation learning for language).

## Minimum Quality
Pass ✅.  
The paper is complete (Abstract, Introduction, Related Work, Method, Experiments, Analysis, Conclusion), written in clear English, and presents a concrete algorithm with non‑trivial theory and extensive experiments. I do not see fatal methodological errors or evaluation practices that would force an automatic desk rejection.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I see no hidden instructions, prompt injection, or manipulation targeting automated reviewers. The only meta text concerns reproducibility and ethics in a normal academic style.

---

# Expected Review Outcome:

## Summary

The paper proposes **CANON** (Conditional advANtage estimatiON), an advantage estimator for reinforcement learning with verifiable rewards (RLVR) in large reasoning models. CANON splits sampled rollouts for each prompt into two groups according to a scalar metric (e.g., entropy, response length), and defines **inter‑group** and **intra‑group** advantages (Equations (3)–(5)), which are combined with a weight μ. Inter‑group comparison captures which direction of the metric correlates with higher reward, while intra‑group comparison promotes better samples within each trend. The authors prove that when groups are equal size, inter‑group advantage amplifies the signal induced by the grouping metric compared with DR.GRPO, and empirically show consistent gains in math and complex logic reasoning benchmarks, as well as improved performance–efficiency Pareto frontiers when grouping by length.

## Strengths

1. **Clear conceptual idea with a simple implementation hook into existing RLVR pipelines.**  
   The core idea of conditional regrouping and defining inter‑ vs intra‑group advantages (Section 4.2, Equations (3)–(5)) is easy to implement on top of GRPO‑style methods: group responses by a metric, compute group means, and replace the baseline. Figure 1 visualizes this flow clearly: from query and policy model to verifier rewards, metric computation, regrouping, and separate inter/intra comparisons. This makes the method practically useful for people already running DR.GRPO/GRPO pipelines.

2. **Theoretical analysis that distinguishes metric‑specific amplification from naive scaling.**  
   Theorem 1 and its detailed derivation (Appendix E, especially Equations (16)–(21)) show that, when the two groups have equal size, the inter‑group advantage increases the magnitude of the advantage signal relative to DR.GRPO, but only in a way controlled by the probability \(p\) of satisfying the condition. The reformulation of DR.GRPO as the uniform mixture \(\hat A^{\text{DR.GRPO}} = \tfrac12 \hat A^{\text{inter}} + \tfrac12 \hat A^{\text{intra}}\) in Equation (7) is an insightful connection that positions CANON as a principled generalization rather than an ad‑hoc tweak. Theorem 2 further formalizes that amplifying one condition \(c_1\) does not inadvertently amplify another independent condition \(c_2\).

3. **Strong and broad empirical evaluation with competitive baselines.**  
   Experiments span three models (Qwen2.5‑Math‑7B, Qwen2.5‑Math‑1.5B, Llama3.1‑8B), six math datasets (AIME24/25, OlympiadBench, AMC, MATH‑500, GSM8k) and three ZebraLogic complexity subsets. Baselines include major RLVR advantage estimators (ReMax, RLOO, GRPO, DR.GRPO, REINFORCE++), plus entropy‑based methods (Entropy Adv, Clip‑Cov) and several length‑penalty baselines.  
   * In **Table 1**, entropy‑based CANON‑Inter reaches 57.6 average math accuracy vs 55.7 for DR.GRPO, and length‑based CANON‑Inter achieves similar math accuracy (55.3) while cutting tokens from 1522 to 1008 (−33.8%).  
   * In **Table 2**, dynamic scheduling (CANON‑Dynamic) improves both math and logic performance across all three models relative to DR.GRPO, which supports the claim that combining inter and intra components can jointly benefit easy and complex tasks.  
   * **Table 3** and **Figure 4** convincingly show that CANON‑Eff (length‑based CANON with α < 1) dominates length‑penalty baselines in the performance–token trade‑off and explores the Pareto frontier more stably.

4. **Insightful analysis leveraging figures and ablations.**  
   Figure 2 is particularly informative:  
   * Panel (a) shows training reward, where CANON‑Inter tracks or slightly exceeds DR.GRPO.  
   * Panel (b) shows math test accuracy improving fastest under CANON‑Inter.  
   * Panel (c) shows complex logic performance, where CANON‑Intra catches up and eventually surpasses others, aligning with the narrative that intra‑group advantage is better for exploration in hard reasoning tasks.  
   * Panels (d) and (e) track response length and entropy respectively: CANON‑Inter steadily reduces entropy and keeps length moderate; CANON‑Intra leads to higher entropy and longer responses.  
   * Panel (f) shows “gain of rethinking” crossing zero for CANON‑Intra late in training, again matching the explanation that intra‑group advantage encourages reflection patterns.  
   Figure 3’s radar chart elegantly summarizes the math vs logic trade‑off across methods and models, with CANON‑Dynamic filling a larger area than DR.GRPO. Figure 5 shows how entropy monotonically tracks μ, giving an interpretable knob.

5. **Practical contribution for efficient reasoning and token budgets.**  
   Section 5.3 and Table 3, along with Figure 4, directly address a very practical concern: reducing chain‑of‑thought length without catastrophic performance drops. Length‑weighted CANON‑Eff with α = 0.96 achieves essentially the same overall accuracy (56.2 vs 56.6 for DR.GRPO) but uses ~26% fewer tokens. With α = 0.88, at matched performance to an aggressive Length Reward baseline, CANON cuts token cost by 45.5%, and at low budgets achieves 2.63× higher performance (Figure 4b). Figure 4c’s Pareto frontiers highlight that tuning α sweeps a smooth frontier, whereas classical length penalties show brittle behavior (e.g., Length Reward (+) collapsing when increasing coeff from 0.004 to 0.005).

6. **Empirical evidence that CANON is not just “scaled DR.GRPO.”**  
   Table 4 compares straightforward numerical amplification (doubling advantages, Entropy Adv) against CANON. While numerical scaling gives small math gains, logic performance suffers, and Entropy Adv severely hurts logic (18.5 vs 26.2). CANON‑Intra and CANON‑Inter exhibit distinct behavior: CANON‑Inter improves math without wrecking logic, while CANON‑Intra boosts logic performance. This, plus the theoretical arguments, supports the authors’ claim that metric‑conditioned regrouping provides qualitatively different signals than global scaling.

7. **Some attention to hyperparameter behavior and interpretability.**  
   Section D.4 and Table 10 systematically vary μ, showing monotonic changes in entropy and a smooth trade‑off between math and logic performance. Section D.5 and Table 11 similarly vary α, with performance gently decreasing as α drops while token cost decreases strongly. This is helpful evidence that the added knobs behave in a controlled and interpretable way, which matters given the community’s sensitivity to RL hyperparameter brittleness.

## Weaknesses

1. **Conceptual novelty is moderate; the method is structurally close to existing relative advantage schemes.**  
   At its core, CANON takes the group \(G_q\), sorts by a scalar metric, splits into two halves, and uses group means as baselines in two slightly different ways. From a policy‑gradient perspective, this is a particular choice of control variate that is not far from:  
   * DR.GRPO’s group mean baseline (Equation (2) without std), which the authors later show is exactly the average of their inter and intra advantages (Equation (7));  
   * contrastive or stratified advantage methods that compare samples within subgroups.  
   While formalizing “conditional regrouping” and separating inter vs intra signals is useful, it is not a radical departure conceptually. The paper could do a better job in Section 4 of contrasting CANON against, for example, grouping by metric and applying DR.GRPO separately per group (beyond the brief qualitative comment after Equation (4)), or alternative baselines like quantile‑based control variates.

2. **Mathematical exposition has several clarity issues despite being basically correct.**  
   * Equation (4) and the following explanation contain a confusing and partially malformed inequality:  
     > “\(1-\text{mean}(\{R_{o'}|o' \in G_q^+\} > 1-\text{mean}(\{R_{o'}|o' \in G_q^-\}\) when ...”.  
     Parentheses are mismatched, and the logic is hard to follow. The point (that intra‑group advantages favor correct samples in the lower‑reward group since their surplus over the group mean is larger) is important and should be expressed cleanly, e.g., by explicitly computing \(A^{\text{intra}}\) for \(R_o=1\) when \(\mathbb E[R | G_q^+] < \mathbb E[R | G_q^-]\).  
   * The statement of Theorem 1 (Equation (6)) is syntactically awkward: “>1, only when |C_q^+| = |C_q^-| if |C_q^+| is a constant” is confusing, and the proof is fairly heavy for what is essentially an algebraic comparison of two baselines with 0–1 rewards.  
   * In Theorem 2, the notation \(\hat A_{q,o,t}^{\text{inter based on } c_1}\) is slightly sloppy; explicit dependence on the grouping probability \(p_1\) would make the invariance more transparent.  
   None of these are fatal, but for a paper presenting itself as giving theoretical justification, these sections should be tightened substantially.

3. **Assumptions behind the theory are restrictive and not well connected to real training behavior.**  
   Theorem 1 assumes: (a) rewards are Bernoulli correctness indicators \(R_o\in\{0,1\}\), (b) equal group sizes \(|C_q^+|=|C_q^-|\), and (c) grouping determined by a single sorted metric with a fixed cut ratio λ. In practice:  
   * The group size equality is enforced by construction (sorting and splitting the 16 samples in half), but the paper does not analyze robustness if G is small or if one group has almost all correct samples. How does variance compare to DR.GRPO when the correct answers are concentrated in one side?  
   * The analysis completely ignores reward variance within each condition; yet in RLVR, rewards can be shaped (as in length penalties), not just 0/1.  
   * Theorem 2’s independence assumption between conditions \(c_1, c_2\) is strong when entropy, length, reflection count, etc., are often highly correlated in practice; the paper still uses the theorem to argue qualitatively that other factors are not amplified.  
   This gap between assumptions and actual RLVR training regimes weakens the strength of the “theoretical justification” claim.

4. **Hyperparameter and scheduling complexity is non‑trivial and somewhat ad hoc.**  
   While the authors provide analyses in D.4/D.5, the actual training setup uses a fair number of knobs:  
   * μ controlling inter vs intra mixing, often scheduled dynamically by accuracy or cosine schedules with restarts (Equation (10) and Section 5.2).  
   * α controlling length weighting in inter‑group advantages (Equation (9)).  
   * Metric choice (entropy vs length vs reflection count).  
   Table 2 shows that different models require **different scheduling strategies** (“Cosin‑First‑Inter‑Later‑Intra” vs “First‑Inter‑Later‑Intra”), and Figure 10 / Figure 9 illustrate non‑trivial μ schedules. This undercuts the claim that CANON avoids sensitive priors: the metric’s direction is not hard‑wired, but the schedule on μ and α is still effectively encoding preferred trends (e.g., early exploitation then exploration). Practitioners would need to perform non‑trivial meta‑tuning, and the paper does not provide clear guidelines beyond “try a cosine schedule” or “use accuracy as Λ”.

5. **Positioning relative to very recent entropy and advantage‑shaping work is incomplete.**  
   The Related Work cites some near‑contemporaneous entropy‑based methods (e.g., Cheng et al., 2025, Cui et al., 2025, Chen et al., 2025b), but omits several closely related pieces that are highly relevant to CANON’s framing: conditional advantage shaping based on entropy or confidence, and entropy‑guided exploitation of “difficult” prompts. Missing works include, among others (details in the next section):  
   * Jin et al. (2025) on revisiting entropy in RL for LRMs;  
   * Chen et al. (2025) on low‑entropy segment‑based advantage shaping;  
   * Wu et al. (2026) on step‑potential advantage using intermediate correctness/confidence;  
   * Le et al. (2026) on entropy‑guided advantage shaping targeted at zero‑variance prompts.  
   These works appear to address very similar questions (how to use entropy/correctness signals for better advantage estimation) and should be explicitly contrasted in Section 2 and in the experiments (at least conceptually if code is not available). Right now, CANON’s originality and advantages over these techniques are not convincingly delineated.

6. **Evaluation is strong but somewhat limited in diversity and stability reporting.**  
   * All tasks are math and logic benchmarks with verifiable rewards. This is by design, but many RLVR setups now consider code generation or other verifiable domains. It remains unclear whether CANON’s behavior (especially entropy‑based scheduling) generalizes beyond math, where proofs are highly structured and reward signals 0/1.  
   * Most results are single runs; there is no reporting of variance across seeds. Given that G=16, reward sparsity in AIME/Olympiad, and the complexity of μ/α schedules, it would be important to show at least 2–3 seeds for key setups in Table 1 and Table 3 to establish robustness.  
   * For Llama3.1‑8B, the dataset is changed to an easier 35k subset (Appendix C.5, Figure 8). This makes sense given its weakness, but then comparisons across models (Figure 3) must be interpreted cautiously. The paper does not attempt to calibrate difficulty across these datasets, so the cross‑model comparison is somewhat cosmetic.

7. **Some design choices are under‑motivated or driven by hindsight.**  
   * The training setup in Section 5.1 removes KL and entropy losses entirely, which interacts heavily with entropy‑based advantage estimation. There is little discussion of how CANON behaves when these regularizers are present, which is common in RLHF/RLVR pipelines.  
   * The context length expansion for Qwen2.5‑Math‑7B (Appendix C.3, Figure 7) is introduced after observing too much truncation; this is practical but also biases the setup towards methods that actively reduce length. It would be useful to know whether CANON‑Eff still dominates when the base model does not suffer from heavy clipping.  
   * The choice of G=16 samples per prompt is fixed; there is no analysis of how performance scales with group size, even though the grouping and variance of group means are central to the method.

8. **Some analysis results are descriptive but not deeply probed.**  
   Several interesting observations are made but not fully unpacked. For example:  
   * Figure 6 shows that pure CANON‑Intra hurts training reward but helps “gain of rethinking,” whereas CANON‑Inter maintains high reward but no positive gain, and CANON‑Dynamic balances both. It would be informative to quantify how these rethinking gains translate into actual solution diversity or failure modes reduced.  
   * Table 7 (per‑token reflection metric) shows CANON‑Inter shortening reflection while improving math, CANON‑Intra encouraging reflection and boosting logic; this is consistent but nearly identical to the entropy story. The paper does not try a truly orthogonal metric (e.g., verifier margin, step‑level correctness) to stress‑test the generality claim.

Overall, the work is technically sound and empirically strong, but the conceptual leap over DR.GRPO and concurrent entropy/advantage‑shaping literature is moderate, and the added complexity in μ/α scheduling plus somewhat narrow domain coverage keep it just below a “clear accept”.

## Potentially Missing Related Work

The following closely related works are not cited and should be discussed:

1. **Renren Jin, Pengzhi Gao, Yuqi Ren, “Revisiting Entropy in Reinforcement Learning for Large Reasoning Models”, 2025.**  
   Directly investigates entropy dynamics in RLVR for LRMs, which is highly relevant to the motivation in Sections 1 and 2 and to entropy‑based grouping in CANON. It should be discussed in the Related Work section and, if possible, compared conceptually to CANON‑Inter’s entropy‑reduction behavior seen in Figure 2e and Figure 5.

2. **Xinzhu Chen, Xuesheng Li, Zhongxiang Sun, “Beyond High‑Entropy Exploration: Correctness‑Aware Low‑Entropy Segment‑Based Advantage Shaping for Reasoning LLMs”, 2025.**  
   Proposes advantage shaping that selectively emphasizes low‑entropy segments conditioned on correctness, which is very close in spirit to using metric‑conditioned baselines. This belongs in Section 2 under “Entropy‑related baselines” and should be contrasted with CANON’s regrouping at the response level (Equations (3)–(5)).

3. **Fei Wu, Zhenrong Zhang, Qikai Chang, “Step Potential Advantage Estimation: Harnessing Intermediate Confidence and Correctness for Efficient Mathematical Reasoning”, 2026.**  
   Introduces a fine‑grained advantage estimator using intermediate confidence/correctness, also for math reasoning. This is directly relevant to CANON’s claim of better credit assignment and should be mentioned in Section 2 and the discussion around Table 1, ideally clarifying whether CANON is complementary (metric‑level) vs step‑level methods like step potentials.

4. **Thanh‑Long V. Le, Myeongho Jeon, Kim Vu, “No Prompt Left Behind: Exploiting Zero‑Variance Prompts in LLM Reinforcement Learning via Entropy‑Guided Advantage Shaping”, 2026.**  
   Uses entropy‑guided advantage shaping to extract learning signal from zero‑variance prompts. This speaks to the same challenge of making use of entropy signals without hand‑crafted priors. The relationship should be discussed in Section 2 and Section 6, especially regarding Theorem 2 and CANON’s claim of “selective amplification”.

5. **Xumeng Wen, Zihan Liu, Shun Zheng, “Reinforcement Learning with Verifiable Rewards Implicitly Incentivizes Correct Reasoning in Base LLMs”, 2026.**  
   Analyzes how RLVR itself shapes reasoning behaviors in base models, very relevant background for Section 1 and 2. It would help contextualize CANON as modifying the *form* of RLVR’s incentives rather than the reward function.

6. **Yan Sun, Jia Guo, Stanley Kok, “Interpretable Intrinsic Cues for Efficient Reinforcement Learning with Large Language Models”, 2025.**  
   Discusses leveraging intrinsic cues for efficient RL with LLMs, which is directly related to using length/entropy/reflection as metrics. It should be cited when motivating CANON’s use of training metrics as grouping conditions (Section 4.1) and in Section 5.3 on efficient reasoning.

7. **Haoran Zhang, Yafu Li, Zhi Wang, “Characterizing, Evaluating, and Optimizing Complex Reasoning”, 2025.**  
   Provides a framework for complex reasoning evaluation and optimization, relevant to the ZebraLogic experiments and the interpretation of “high‑complexity reasoning” in Section 5.1 and Figure 2c/2f. It should be referenced when defining the complexity subsets and discussing why gains on XLarge are particularly meaningful.

Incorporating these works will significantly strengthen the positioning and clarify what is distinctive about CANON relative to the growing entropy‑ and confidence‑based advantage shaping literature.

## Questions

1. **Robustness to reward shaping and non‑binary rewards.**  
   All theory and much of the intuition rely on 0/1 correctness rewards. In practice, length and other penalties are sometimes folded into rewards rather than advantages. How does CANON behave when the base reward is non‑binary, or when we use a dense reward model instead of verifiers? Can the authors provide ablations where the per‑sample reward includes, for example, a small length penalty but grouping is still based on entropy?

2. **Variance and sample efficiency compared to DR.GRPO.**  
   Inter‑ and intra‑group advantages replace the global mean baseline by groups of size \(|G_q^+|=|G_q^-|=8\). This presumably increases variance. Have the authors measured gradient variance or learning speed (e.g., accuracy vs number of training steps) relative to DR.GRPO? Any evidence that CANON converges faster / slower for a given compute budget would be helpful.

3. **Guidance for practitioners on choosing μ and its schedule.**  
   The paper currently uses different schedules per model, sometimes accuracy‑based, sometimes cosine‑based. Suppose a practitioner wants to use CANON on a new LRM in a different domain. What concrete recommendation can the authors give: start with μ=0.5, and then move towards inter or intra depending on observed entropy/accuracy curves? Are there diagnostics (e.g., reflection gain from Figure 2f) that indicate when to increase intra‑group weighting?

4. **Generalization beyond math/logic.**  
   Do the authors have any preliminary results or arguments on how CANON might behave on other RLVR‑ready tasks such as code generation with unit tests, factual QA with retrieval verification, or symbolic theorem proving? Even a small‑scale experiment on a code benchmark would significantly increase confidence that the method is not tuned exclusively to math reasoning.

5. **Comparison against concurrent entropy/confidence shaping methods.**  
   If code for some of the missing works (e.g., Jin et al., Chen et al., Wu et al.) is not available, can the authors at least implement simplified baselines that approximate their key ideas (e.g., entropy‑weighted advantages at token level, confidence‑based segment reweighting)? This would sharpen the empirical argument that conditional regrouping is a better inductive bias than straight advantage scaling.

6. **Ablation on smaller group sizes and unequal splits.**  
   How does CANON perform when G is 4 or 8 instead of 16, and when the split is not exactly 50/50 (e.g., high‑entropy quartile vs the rest)? The theory focuses on equal sizes, but empirically small G is much more realistic in resource‑constrained RLVR. An ablation could clarify how essential the equal‑size split is.

## Flag For Ethics Review

- No ethics review needed.  

## Details Of Ethics Concerns

N/A.

## Soundness Rating

3: good.  
The method is mathematically straightforward, the theorems are reasonable under stated assumptions, and experiments are extensive and well grounded. Some theoretical assumptions are restrictive and the exposition has rough edges, but there are no obvious fatal flaws.

## Presentation Rating

3: good.  
The main ideas, equations, and experimental setup are mostly clear. Figures (especially Figures 1–4) and tables (1–4) are informative. However, some mathematical explanations (e.g., around Equations (4), (6), and Theorem 1) are awkwardly written, and the Related Work omits several directly relevant recent papers.

## Contribution Rating

3: good.  
CANON is a solid incremental contribution: it generalizes DR.GRPO via metric‑conditioned regrouping, offers a nontrivial perspective on entropy/length signals, and shows meaningful empirical gains and efficiency improvements. The novelty is moderate, and the method introduces additional hyperparameters, but the overall contribution is clearly valuable to the RLVR / reasoning LLM community.

## Overall Rating

6: marginally above the acceptance threshold. But would not mind if paper is rejected.  
The paper is technically sound, empirically strong, and offers a useful and interpretable extension of GRPO‑style advantage estimation that demonstrably improves both accuracy and efficiency on important reasoning benchmarks. Its conceptual novelty over existing advantage‑shaping work is moderate and the hyperparameter/scheduling complexity is non‑trivial, and the positioning relative to closely related recent work must be improved. With these caveats addressed, it would make a good ICLR contribution.

## Reviewer Confidence

4: confident.  
I am familiar with RLHF/RLVR and recent GRPO‑style methods, and I carefully checked the main equations and experiments. Some extremely recent related work might still be missing, but I am reasonably confident in the assessment.
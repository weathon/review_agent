---
job_id: 69c8c83c-d96b-4594-adf0-ffc71bd24296
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: bhR00j6Mku.pdf
paper: On The Fragility of Benchmark Contamination Detection in Reasoning Models
main_score_norm: 0.6
desk_reject: false
---
# Desk Rejection Assessment:

## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper studies benchmark contamination detection and RL training dynamics for large reasoning models, touching representation learning, reinforcement learning, evaluation/benchmarks, and security/privacy-style membership inference, all squarely within ICLR’s scope.

## Minimum Quality
Pass ✅.  
The paper is complete (Abstract, Introduction, Related Work, Methodology/Analysis, Experiments, Results, Discussion/Conclusion, Limitations, etc.), technically nontrivial, and written in clear English. Experiments are extensive and fairly described, and I do not see fundamental methodological or theoretical errors that would warrant immediate desk rejection.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I do not see any hidden prompts, steganographic text, or instructions aimed at influencing automated reviewers within the paper body.

---

# Expected Review Outcome:

## Summary

The paper investigates how benchmark contamination in large reasoning models (LRMs) can be both introduced and subsequently concealed, focusing on two stages: (I) when a base model is turned into an LRM via SFT and RL, and (II) when an already strong LRM receives final-stage contamination via CoT SFT.  

For Stage I, extensive experiments across 10 contamination-detection methods show that initial SFT contamination is detectable, but subsequent GRPO-style RL sharply reduces AUROC for membership inference without removing the performance gains. The authors complement this with a theoretical analysis arguing that PPO-style importance sampling and clipping drive a contraction of the loss gap between members and non-members, empirically validated by ablations comparing RAFT, RAFT++, and GRPO.  

For Stage II, they contaminate several strong LRMs (DeepSeek-R1 distill variants, OpenThinker) via CoT SFT on benchmark members, observing large pass@1 inflation (Table 4) while almost all detection methods operate near random guess (Table 5). They argue that LRMs generalize contamination in a way that undermines log-probability–based memorization detectors.

---

## Strengths

1. **Clear, timely problem formulation and two-stage framing.**  
   The paper cleanly separates contamination scenarios into Stage I (pre-LRM: base model → LRM via SFT + RL) and Stage II (post-LRM: final CoT SFT on an already strong LRM), as visually summarized in **Figure 1**. This framing is actually operationally realistic and helps organize both the experiments and the narrative around where contamination can happen and how it manifests.

2. **Extensive and careful empirical study across many axes.**  
   The experiments are broad: 6 math/science benchmarks (AIME24/25, AMC23, OlympiadBench, GPQA-Diamond, Minerva Math), plus additional non-math datasets (MMLU-Pro non-STEM, LiveCodeBench) in Appendix E.11; multiple base models (Qwen2.5-3B/7B/14B, Llama-3.1-8B) and several advanced LRMs (DeepSeek-R1 distill LLaMA/Qwen 7B/14B, OpenThinker3-7B); 10 contamination detectors spanning generation-based, perturbation-based, reference-based, embedding-based, and reference-free approaches.  
   - **Table 1** systematically explores combinations of SFT and RL contamination/clean data for two base models, making the point that SFT contamination is the main source of pass@1 inflation (e.g., for Qwen2.5-7B, clean SFT gives 38.41% → 47.23% with SFT contamination, an extra +8.82 points avg, while RL contamination on top of that shifts results minimally).  
   - **Tables 2, 7, 12–17** provide AUROC results over many detection methods and RL objectives/steps, giving a convincing empirical picture of the concealment effect.

3. **Concrete evidence that RL training conceals contamination signals while preserving inflated performance.**  
   For Stage I, **Table 2** shows that, starting from an SFT-contaminated Qwen2.5-7B, GRPO reduces AUROC for essentially all detectors while pass@1 remains inflated. For example, LiRA’s average AUROC drops from 89.13% (before RL) to 80.14% (RL w/ clean) and 74.89% (RL w/ clean + mem), and reference-free detectors like Loss fall from 75.48% to 61.26% and 58.80%.  
   The authors also rule out the “model forgot contamination” explanation: **Table 1** shows that GRPO’d contaminated models still enjoy a ~7% average pass@1 gain over clean SFT, and **Table 23** shows that additional clean SFT alone does not conceal contamination (AUROCs stay high) but still increases pass@1. This triangulation is strong.

4. **Theoretical analysis that isolates PPO-style clipping/importance sampling as the main culprit.**  
   Section 3.2 defines the member vs non-member NLL gap \(G_k\) in **Equation (1)** and the per-prompt NLL drift \(\Delta_x\) in **Equation (2)**, then derives **Theorem 3.1 / Equation (5)**, decomposing \(\Delta_x\) into a mean term \(\mu(x)\) and a covariance term \(\beta(x)\) involving \(\sum_t \tilde{A}_t^w\). The subsequent instantiations for RAFT, RAFT++, and GRPO analytically argue that RAFT’s lack of clipping keeps the covariance term offsetting the mean gap (so no contraction), whereas adding clipping makes \(\mathrm{Cov}(\ell_k,\sum \rho_t m_t)\) negative and more negative for non-members, driving \(\Delta_N - \Delta_M < 0\).  
   This is then tightly backed by **Table 3**, which shows Loss AUROC for RAFT actually slightly improves versus SFT-only (77.51% vs 75.48%), while RAFT++ and GRPO with clipping drop sharply (57.58% and 61.26%, respectively). The ablation rows with clipping removed for RAFT++ and GRPO (“×” in the “Clipping” column) restore AUROCs near the SFT baseline, closely matching the theory. This theory–experiment alignment is a strong point.

5. **Nice use of distributional visualizations to explain detector failures.**  
   **Figures 3 and 7–9** plot log-probability distributions for members vs non-members under various training regimes. For example, **Figure 3** (Page 6) on GPQA/Olympiad shows that, after GRPO, the member and non-member log-prob distributions nearly overlap, whereas in the SFT-only contaminated model they are clearly separated. This directly explains why NLL-based detectors like Loss, Min-K, Max-K collapse toward AUROC ≈ 0.5 in **Figure 2**.  
   Similarly, **Figure 4** (Stage II, advanced LRMs) shows that extensive CoT SFT on members shifts both member and non-member log-prob distributions upward roughly equally; the AUROC printed in the panels goes from ≈0.49–0.50 pre-contamination to only ≈0.53–0.57 post-contamination. This visual evidence well supports the claim that LRMs generalize the contaminated signal rather than purely memorizing sequences.

6. **Stage II result is both surprising and important for practice.**  
   **Table 4** shows large pass@1 boosts from final-stage CoT contamination on strong LRMs (e.g., DeepSeek-R1-Distill-Llama-8B average 48.38 → 60.14; DeepSeek-R1-Distill-Qwen-14B average 59.83 → 69.24). Yet **Table 5** shows that almost all detectors, including LiRA, operate near random (average AUROC ~55–63%) across all models and benchmarks, with several cases essentially at 0.5. This is a very actionable finding for the community: current contamination detection ecosystems are basically blind to final-stage CoT contamination of LRMs, while leaderboards can be heavily inflated.

7. **Cross-checks that rule out trivial confounders.**  
   The authors repeatedly test alternative explanations:  
   - **Figure 6** shows that response lengths do not change dramatically under GRPO/RAFT, yet AUROC behaves very differently, supporting the claim that entropy/log-prob shifts, not length, drive concealment.  
   - **Appendix E.2 / Tables 8–11** compare using question tokens vs response tokens and thinking vs non-thinking tokens for detection; the chosen setup (responses) is empirically justified.  
   - **Embedding-based detection** in **Table 19** shows the same contraction trend, confirming that the effect is not limited to log-prob metrics.

8. **Broader-domain evidence (beyond math) suggests the phenomenon is not narrow.**  
   **Table 21** (Stage I) and **Table 22** (Stage II) show similar RL concealment and Stage II detection failures on non-STEM MMLU-Pro and coding (LiveCodeBench). This is important for generality: the paper is not just about competition math leaderboards.

---

## Weaknesses

1. **No constructive detection or mitigation method is proposed.**  
   While the paper convincingly reveals a serious vulnerability, it stops at diagnosis. Section 5 only lists high-level desiderata (release more checkpoints, move beyond log-prob–based detection). There is no attempt to sketch or empirically evaluate alternative detectors that might be more robust in LRM settings (e.g., trajectory-level or verifier-based statistics, distributional calibration across prompts, cross-model differentials, or training-dynamics–based tests). Given the strong empirical base, even a simple prototype detector exploiting, say, reward variance under RL or cross-entropy under multiple RL checkpoints would have materially increased the paper’s contribution beyond “everything is broken”. As it stands, the work is closer to a well-executed negative result plus analysis than to a full solution.

2. **Theoretical analysis in Section 3.2 rests on unverified and sometimes hand-wavy assumptions.**  
   The derivations around **Theorem 3.1 / Equation (5)** assume a tabular setting, small natural gradient steps, and crucially sign properties of covariance terms (\(\beta_M,\beta_N\)) that are not fully proven. For RAFT, the argument relies on “lower loss corresponds to higher \(p_k(s_t)\), thus \(\beta^{\mathrm{RAFT}}(x)>0\)” and that \(\beta_N > \beta_M\) because non-members “have more variance”; for RAFT++, the key step is that \(\mathrm{Cov}(\ell_k, \sum \rho_t m_t)\) is negative and “more prominent in non-members”. These are plausible but not rigorously established: there is no formal statement of required assumptions (e.g., monotonicity relations between loss and success probability, distributional properties of trajectories) or bounds quantifying when covariance dominates mean differences.  
   Moreover, the simplification for GRPO where \(A_k(x,y) = r(x,y) - p_k(x)\) without variance normalization is not the actual implementation used in experiments (Section B defines \(A_i\) with group-wise standardization). The paper asserts that the covariance argument “extends similarly” but does not show this under realistic GRPO advantages. This does not invalidate the main empirical claims, but it weakens the “root cause” story presented as theorem-driven rather than heuristic.

3. **Some generalizations about RL algorithms are overstated relative to evidence.**  
   The paper concludes (e.g., Abstract, end of Section 3.2) that “a broad class of RL methods” with PPO-style importance sampling and clipping will inherently conceal contamination. However, all empirical evidence is on a single reward schema (verifiable math correctness), specific GRPO/RAFT++ variants, and relatively short RL horizons (up to 156 steps in core results, with a few longer runs in Appendix E.7). Other RL designs that also use clipping, but with different advantage estimators, KL penalties, or exploration strategies, might behave differently.  
   Also, Section 3.1 infers that “we expect that extensive GRPO training would render all existing detection methods to near-random performance eventually”, based on monotonic AUROC decline in **Figure 2** and **Table 12**, but that is an extrapolation, not demonstrated. A more tempered wording emphasizing “we empirically observe monotonic decline over the ranges we studied” would be more accurate.

4. **Evaluation of detectors is somewhat rigid and limited to a specific instantiation.**  
   The detection pipeline fixes several design choices that might matter but are underexplored:  
   - Every method averages scores over 8 rollouts per question. It would have been informative to see how AUROC changes with 1 vs 8 vs 32 rollouts, especially for generation-based detectors like CDD and Verbatim which could be sensitive to sampling variance.  
   - For reference-based methods, only Bespoke-MiniCheck-7B is used as \(\pi_{\text{ref}}\) (Appendix D.2.3). Performance might differ with a reference closer or farther from the target model; this is alluded to but not evaluated.  
   - In generation-based Verbatim, the partial-prompt ratio is fixed at 80%; the authors note that smaller ratios make LRMs answer instead of continue, but do not empirically show that 80% is near-optimal or robust across datasets.  
   These are not fatal, but they limit how strongly we can conclude “all existing detectors” fail, versus “this reasonable but particular instantiation of them fails”.

5. **Stage II interpretation leans heavily on an untested “generalization vs memorization” narrative.**  
   The authors argue, based on **Figure 4** and **Figures 13–14**, that LRMs “internalize reasoning” such that both members and non-members see similar log-prob increases after member-only CoT SFT, so detectors fail because they assumed contamination implied lower loss only on training sequences. While plausible, there is no deeper investigation into *where* this generalization applies and where it does not.  
   For example, it would be informative to stratify non-members into “hard” vs “easy” questions, or measure cross-benchmark transfer (train contamination on benchmark A, test detection on benchmark B) to see whether the gains are truly global or just local around the training distribution. Without such analysis, the conclusion that “contamination is no longer mainly about memorization” is somewhat speculative and may depend on the particular benchmarks and LRMs studied.

6. **Limited exploration of alternative detection signals beyond log-prob and token-level statistics.**  
   Almost all detectors here ultimately reduce to log-prob or simple functionals of token probabilities (including Min-K%, Max-K%, Loss, LiRA, Ref). The paper briefly tests an embedding-based classifier (Table 19) using last-token hidden states, which also shows contraction, but this is still a shallow, pointwise representation. There is no exploration of more structured sequence-level signals (e.g., diversity over multiple generations, verifier reward distributions, or trajectory-level features like number of reasoning steps, failure modes, or error patterns).  
   Given that LRMs are inherently about long CoT trajectories, the strict focus on per-token probability signals might be under-fitting the space of potential detectors and thus somewhat biases the conclusion that “all existing methods” are fragile.

7. **Some expository gaps and notation inconsistencies in the math sections.**  
   While the main story is understandable, several pieces of notation in Section 3.2 could be clearer or more rigorously tied to the algorithms:  
   - The definition of \(w_t = \rho_t m_t\) and the role of \(m_t\) as a clipping mask is only loosely connected to the actual GRPO loss in Appendix B. In particular, GRPO’s loss in Appendix B is sequence-level with KL penalties, yet Section 3.2 analyzes per-token PPO-style updates without discussing how KL terms or grouped advantages interact with the covariance decomposition.  
   - There is a notational mismatch between \(\tilde{A}_t^w\) in Equation (4) and the \(\hat{A}_t^w\) used in Equation (5)/(6) and the proof, which could confuse readers trying to reproduce the derivation.  
   - When introducing \(B(s)\) and \(C(s)\), the paper uses \(\mathbb{E}_{a\sim\pi}[\rho(s,a)m(s,a)q_k(s,a)]\) but does not state explicitly what \(\pi\) is (current vs old policy) or how these are estimated in practice.  
   None of these are fatal, but they make the theoretical analysis feel more heuristic than formally watertight.

8. **Scope of benchmarks is still skewed toward math and reasoning-heavy domains.**  
   Although the appendix adds non-STEM MMLU-Pro and LiveCodeBench, the core analysis and majority of figures/tables center on math/science reasoning datasets. The Stage II “CoT contamination barely leaves evidence” claim may not hold for more natural language tasks (e.g., news classification, summarization) where CoT might be shorter or less central. The authors acknowledge this partially in Appendix E.11, but the main text could be more explicit about this domain bias and the limits of generalization.

9. **Minor but noticeable presentation issues.**  
   - Some figure captions are dense and repeat text from the main body (e.g., **Figures 7–9**), while others (like **Figure 2** and **Figure 5**) introduce new claims (monotonic AUROC decline) that are only loosely referenced in the main text.  
   - The use of “Olypaid” vs “Olympiad” in **Tables 1–3, 7, 18, 20, 23–24** is a small but recurring typo that should be cleaned up.  
   - A few references are inconsistently formatted (e.g., “Olypaidbench” in Table 6, mixing dash vs dot numbering in references).

---

## Potentially Missing Related Work

The “Directly Related Research Papers” provided are mostly foundational model/architecture papers (GPT-2/3, T5, Transformer, etc.) or vision architectures, which are not directly about benchmark contamination detection or RL-based concealment and are already broadly represented via other citations in the paper. I do not see clearly *directly* missing works beyond what is already cited.

N/A.

---

## Questions

1. **Robustness of the theoretical explanation:**  
   Can you more precisely state the assumptions under which Theorem 3.1 and the subsequent RAFT / RAFT++ / GRPO instantiations hold? In particular, what conditions on the distribution of losses \(\ell_k\), success probabilities \(p_k(s)\), and importance ratios \(\rho_t\) are required to guarantee \(\beta_N > \beta_M\) and negative \(\mathrm{Cov}(\ell_k,\sum \rho_t m_t)\) for non-members? A more formal statement (even if stylized) would significantly strengthen the theoretical part.

2. **Impact of reference model choice on reference-based detectors:**  
   How sensitive are LiRA and Ref to the choice of \(\pi_{\text{ref}}\)? Have you tried a closer/larger reference, e.g., the same architecture trained only on clean SFT, or a much smaller one? If you have internal experiments or can add them, it would clarify whether reference-based detectors fundamentally fail under GRPO/Stage II, or whether their performance is contingent on reference alignment.

3. **Stage II generalization vs memorization hypothesis:**  
   Can you provide more concrete evidence differentiating generalization from memorization in Stage II? For example, have you looked at:  
   - Cross-benchmark effects (contaminate benchmark A, test AUROC on benchmark B that is distributionally similar vs dissimilar)?  
   - Stratifying non-members by difficulty or similarity (e.g., via embedding distance) to members, to see if log-prob gains concentrate near members or are global?  
   Such analyses would make the “contamination is no longer mostly memorization” claim more convincing.

4. **Alternative detection signals beyond per-token log-prob:**  
   Given that RL training directly manipulates reward distributions, have you explored detectors using reward-based or verifier-based statistics, such as the distribution over rewards across multiple rollouts per question, or the variance in reasoning path lengths / formats? If not, could you comment on whether you see any promising directions here that might circumvent the contraction effect you describe?

5. **Longer RL training runs and later checkpoints:**  
   You observe monotonic AUROC decline up to 156 GRPO steps in **Figure 2** and **Tables 12–14**, and some additional runs up to 280 steps in Appendix E.7. Do you have any evidence from longer runs (e.g., similar to the thousands of steps used in some public RL-trained LRMs) that AUROC indeed saturates near 0.5 and does not rebound? Even a single curve from a longer run on one benchmark/model would help support the extrapolation.

6. **Interplay with benchmark design and mitigation strategies.**  
   How do you expect common mitigation strategies (rephrasing, paraphrased variants, adversarially generated questions, etc.) to interact with your findings? For example, if benchmarks are regularly rotated or augmented, does GRPO still contract member/non-member gaps as strongly, or does the detection problem fundamentally change?

Answers and, where feasible, additional experiments addressing these points could easily strengthen the paper and would likely increase my confidence in a stronger recommendation.

---

## Flag For Ethics Review

- No ethics review needed.

---

## Details Of Ethics Concerns

N/A. The work studies contamination and evaluation integrity, but does not introduce new privacy attacks or release sensitive data; it uses public datasets and models.

---

## Soundness Rating

3 good.  
The empirical methodology is strong and diversified, and the central empirical claims are well supported. The theoretical analysis is insightful but rests on some unformalized assumptions, so I would not call it “excellent” rigor, but it is still a meaningful component.

---

## Presentation Rating

3 good.  
The paper is generally well written, with clear figures and tables (e.g., **Figure 2** and **Tables 1–5** are informative), though the math exposition in Section 3.2 could be cleaner and some notational inconsistencies and typos should be fixed.

---

## Contribution Rating

3 good.  
The work identifies an important and under-explored vulnerability in LRM benchmarking, provides broad evidence across methods and datasets, and offers a plausible mechanistic explanation at the RL objective level. The main limitation is the absence of a concrete mitigation/detection proposal.

---

## Overall Rating

6 Marginally above the acceptance threshold. But would not mind if paper is rejected.  

The paper delivers a substantial, carefully executed empirical study and a reasonably convincing theoretical story about a problem that is highly relevant for the current LRM ecosystem: RL training with PPO-style objectives can conceal benchmark contamination; final-stage CoT SFT can strongly inflate reasoning benchmarks while leaving little detectable evidence. Despite the lack of a proposed remedy and some theoretical looseness, the evidence base and the practical importance of the findings justify a positive recommendation.

---

## Reviewer Confidence

4 confident.  
I am familiar with contamination/membership-inference work and RL training for LMs, and I carefully checked the equations and experimental design. Some aspects of the theory rely on behavioral assumptions that are hard to fully verify, but this does not undermine the main empirical conclusions.
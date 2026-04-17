# On Computation and Generalization of Group Relative Policy Optimization

- Decision: Reject
- Scores: 4, 4, 6, 2

## Abstract
Group Relative Policy Optimization (GRPO)~\citep{shao2024deepseekmath,guo2025deepseek} has rapidly become a critic-free default for aligning LLMs, yet its statistical and computational foundations remain unclear. We close this gap by providing the first unified theory of GRPO that simultaneously addresses generalization and optimization in the original, practitioner-used formulation and over multiple outer iterations. On the generalization side, we derive sequential (multi-iteration) PAC-Bayes–Bernstein bounds under Markov mixing that concentrate the \emph{empirical GRPO surrogate} around its population counterpart across all iterations; a Transformer path-norm corollary yields substantially tighter capacity terms than spectral norms. We further prove a TRPO-style return bridge showing that ascent in the population GRPO surrogate provably improves true return, with explicit, controllable bias from clipping and KL regularization. On the optimization side, we establish non-PL \emph{stationarity} guarantees for SGDM and AdamW (both $\tilde O(1/\sqrt{K})$) and provide complementary PL-based rates, with variance controlled by $t_{\mathrm{mix}}/(G\sqrt{K})$. Together with interactive information-theoretic lower bounds, our results deliver the first end-to-end, multi-iteration statistical and computational guarantees for GRPO with function approximation. Experiments corroborate the predicted trends and offer practical guidance on group size, clipping, and KL weight; code will be released.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
- This paper presents a comprehensive theoretical treatment of Group Relative Policy Optimization (GRPO), a critic-free alignment algorithm for LLMs.
- It quantifies (i) how much data is needed (sample complexity) and (ii) how the group size $G$ affects performance, via generalization bounds under Markov (non-IID) dependence—a strictly more challenging setting than standard IID theory.
- It proves that these generalization rates are minimax-optimal (up to constants/log factors), with explicit dependence on mixing time and sample size.
- On the optimization side, it establishes $O(1/\sqrt{K})$ convergence to first-order stationarity for SGDM and AdamW.
- Experiments across models and tasks corroborate the theory, reproducing the predicted trends as key parameters (e.g., $N$, $G$, mixing time) are varied.

Note: I used ChatGPT for minor language editing and phrasing assistance; all technical assessments are my own.

### Strengths
- The first unified theory of GRPO, combining statistical generalization upper/lower bounds with an optimization analysis.
- The experiments complement the theory by varying key hyperparameters (e.g., \(N\), \(G\)) and reproducing the predicted trends.
- For generalization, they handle non-IID sequential data modeled as Markov chains and obtain (nearly) minimax-optimal rates.
- On the optimization side, they analyze an unclipped surrogate objective to derive tractable convergence guarantees.

Note: I used ChatGPT for minor language editing and phrasing assistance; all technical assessments are my own.

### Weaknesses
- Their optimization analysis focuses on the unclipped surrogate loss $-J_{\mathrm{sur}}$, rather than the clipped population objective $-\tilde{J}_{\mathrm{GRPO}}$.
- They show that (i) minimizing the surrogate implies improving the true return (Theorem 3), and (ii) the gradient mismatch between the two objectives is bounded (Lemma 4).
- That said, the two landscapes may still differ: the upper-bound term on the right-hand side of Eq. (11) in Lemma 4 may be non-negligible in practice. (See the second question.)
- Moreover, optimizing the unclipped surrogate $-J_{\mathrm{sur}}$ might be inherently easier (no clipping-induced non-smoothness), potentially leading to fewer spurious local minima or saddle points than the original clipped objective.

Note: I used ChatGPT for minor language editing and phrasing assistance; all technical assessments are my own.

### Questions
- Can you formally justify that the optimization trajectories or the difficulty of optimization for minimizing $-J_{\mathrm{sur}}$ and $-\tilde{J}_{\mathrm{GRPO}}$ remain close? (See “Weaknesses” for details.)
- Empirically, does Eq. (11) in Lemma 4 make sense in the intended regime—i.e., is clipping rarely active (effectively $\epsilon$ large) or do importance ratios concentrate near 1 so that the RHS term is negligible?


Note: I used ChatGPT for minor language editing and phrasing assistance; all technical assessments are my own.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper claims studying  Group Relative Policy Optimization (GRPO).  It analyzes though  a mean only  normalized advantage variant i.e., group-centering by subtracting the group mean reward (often called Dr. GRPO ) while referring to the analysis as analyzing GRPO of Shao et al 2024.  

The paper provides (1) high-probability generalization bounds under token-time mixing of the policy-induced Markov chain, (2) a TRPO-style “return bridge” linking improvement of the clipped surrogate to improvement in true return, and (3) non-asymptotic convergence rates for SGDM/AdamW with explicit dependence on group size \(G\) and a mixing parameter \(t_{\text{mix}}\). 

Experiments based on Open-R1/TRL are used to illustrate qualitative trends.

### Strengths
End to end analysis: the paper unifies statistical generalization (with dependent data), optimization guarantees (SGDM/AdamW), and a policy improvement 

Markov dependence :the paper uses mixing-based concentration (block/regenerative arguments) to control per-token sums multiplied by a trajectory-level advantage, making the dependence on \(t_{\text{mix}}\) explicit.

Lower-bound:  Information-theoretic arguments indicate statistical rates are near-optimal up to constants/logs.

### Weaknesses
The paper is positioning as GRPO analysis as it is used by practitioners but it studies instead a mean only normalization, without studying the mean-variance (whitening normalized ) advantage as defined in  the original GRPO paper. 

Algorithm naming / mismatch: Theory is for mean-only group centering (Dr. GRPO) with trajectory level advantages. The paper repeatedly says “GRPO", and experiments rely on Open-R1/TRL but do not disclose whether they used Dr.GRPO (mean-only) or  (“regular”) GRPO (z-scored/whitened), nor whether advantages are trajectory  or token level. As written, it is not clear if  the empirical section validates the objective analyzed.

Analysis does not cover variance-normalized GRPO as is: Z-scoring introduces a random, data-dependent denominator (group std). Without an explicit floor (e.g., \(\varepsilon\)-stabilization), leave-one-out/shrinkage, or self-normalized mixing inequalities, key steps (concentration, policy-improvement constants, optimizer noise bounds) do not carry over.

Mixing assumption under-specified: Results implicitly require a uniform bound on token-time  mixing across the training path \(\{\pi_{\theta_k}\}\) (or an explicit $\sup_k t_{\text{mix}}(\pi_{\theta_k})$. With finite horizons / early EOS, the effective dependence penalty should be read as $\min (t_{\text{mix}}, T_{\max})$.

Missing references: In the related work section the paper misses citation that attempted to study GRPO for example [1] What is the Alignment Objective of GRPO?, [2] Reinforcement Learning with Verifiable Rewards: GRPO's Effective Loss, Dynamics, and Success Amplification

Experimental transparency: Missing the exact TRL/Open-R1 algorithm used .. did you use GRPO or Dr. GRPO? Reproducibility and theory–experiment alignment are weak.

### Questions
1. Which objective did you actually run? In TRL/Open-R1 nomenclature, was it **Dr. GRPO (mean-only)** or **regular GRPO (z-score/whitening)**? 

2. Can you align theory and experiments? Either (a) re-run with Dr. GRPO (mean-only) and reposition the paper to match the analysis, or (b) extend the theory to variance-normalized GRPO by introducing an  $\varepsilon$ stabilized standard deviation  and carrying the resulting constants through concentration, the TRPO bridge, and optimizer rates.

3. Clarify mixing and needed uniform bounds on iterations. 

4. Acknowledge prior art in analysis of GRPO.

5. Provide configs & an ablation: Share the exact TRL/Open-R1 config and add a small ablation comparing  Dr GRPO (mean-only) vs  z-scored GRPO (same seeds).

### Soundness
3

### Presentation
2

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
This paper presents the first comprehensive theoretical analysis of GRPO (Group Relative Policy Optimization) under Markov dependence. It establishes both generalization guarantees (via PAC-Bayes–Bernstein bounds and Transformer path-norm capacity) and optimization convergence rates (for SGDM and AdamW). The work further proves a TRPO-style monotonic improvement theorem and near-minimax optimal lower bounds. Experiments on Qwen and LLaMA models empirically confirm the theoretical trends.

### Strengths
- First end-to-end theory for GRPO: unifies generalization, optimization, and return guarantees.
- Experiments verify predicted dependencies on group size, mixing time, and variance.

### Weaknesses
- The work is mainly theoretical; practical improvements or new algorithms are limited.
- Some sections, especially in Appendices, could benefit from conceptual summaries before technical proofs.

### Questions
- Although you have validated the theory through experiments, what is the practical significance of this? Can these findings provide any insights that help us design more effective and efficient algorithms?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper’s “unified” theory leans on unobservable or unjustified quantities—mixing time, transformer path norms, data-dependent PAC-Bayes posteriors, and PL/AdamW assumptions—so the guarantees are elegant.

### Strengths
Many theoretical results are presented.

### Weaknesses
1. **Definition of (t_{\max}) in LLMs.**
   Please define (t_{\max}) and explain its role. What value did you use in your experiments, and why? It also appears that the evaluation does not probe the assumed Markov mixing behavior—please clarify whether and how this assumption is tested.

2. **Positioning relative to [1].**
   Clearly articulate your contributions compared to Mroueh et al. [1]. What is novel here (theory, algorithms, or empirical findings), and in which settings does your approach provide advantages?

3. **Limited discussion of theoretical results.**
   Expand the discussion of the theory: interpret the bounds, state the regimes where they are tight/loose, and spell out practical implications and limitations of the assumptions.

4. **Code availability.**
   Is the code implemented and available? If so, provide a link and minimal instructions;

5. **“ICLR bound” label in tables.**
   Define precisely what the “ICLR bound” refers to, cite its source, and ensure the tables and captions make this unambiguous.

6. **Paper structure and assumptions.**
   The current structure is hard to follow. Consider consolidating all assumptions in a dedicated section, and add a concise “Contributions” section to help readers track the main ideas and how they connect to the results.

7. **Verification of PAC-Bayes bounds.**
   Explain how the PAC-Bayes bounds are instantiated and evaluated in the experiments (e.g., choice of priors/posteriors, empirical estimators, confidence levels, and any calibration or surrogate approximations).

---

**Overall assessment.**
The paper would benefit from substantial revision and editing to address the points above.

[1] Mroueh, Youssef, et al. “Revisiting Group Relative Policy Optimization: Insights into On-Policy and Off-Policy Training.” arXiv:2505.22257 (2025).

### Questions
See weaknesses

### Soundness
2

### Presentation
1

### Contribution
2

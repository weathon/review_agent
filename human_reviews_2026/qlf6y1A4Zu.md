# Jailbreaking the Matrix: Nullspace Steering for Controlled Model Subversion

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 2, 6, 6

## Abstract
Large language models remain vulnerable to jailbreak attacks, inputs crafted to bypass safety mechanisms and elicit harmful responses, despite advances in alignment and instruction tuning. Existing attacks often rely on prompt rewrites, dense optimization, or ad hoc heuristics, and lack interpretability and robustness. We propose **Head-Masked Nullspace Steering (HMNS)**, a circuit-level intervention that (i) identifies attention heads most causally responsible for a model’s default behavior, (ii) suppresses their write paths via targeted column masking, and (iii) injects a perturbation constrained to the orthogonal complement of the muted subspace. This geometry-aware intervention preserves fluency while steering the model toward completions that differ from baseline routing. HMNS operates in a closed-loop detection–intervention cycle, re-identifying causal heads and reapplying interventions across multiple decoding attempts. Across multiple jailbreak benchmarks, strong safety defenses, and widely used language models, HMNS attains state-of-the-art attack success rates with fewer queries than prior methods. Ablations confirm that nullspace-constrained injection, residual norm scaling, and iterative re-identification are key to its effectiveness. To our knowledge, this is the first jailbreak method to leverage geometry-aware, interpretability-informed interventions, highlighting a new paradigm for controlled model steering and adversarial safety circumvention.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes Head-Masked Nullspace Steering (HMNS), a novel LLM jailbreak attack method. HMNS identifies attention heads that make the greatest causal contribution to a model’s output, dynamically masks their output projections, and injects a perturbation vector orthogonal to the masked subspace, thereby efficiently bypassing alignment constraints during inference. Experimental results show that HMNS achieves state-of-the-art attack success rates (ASR) on multiple jailbreak benchmarks (e.g., AdvBench and HarmBench) while exhibiting significantly higher query efficiency (ACQ) than baseline methods.

### Strengths
- The HMNS method proposed in this paper extends jailbreak attacks from the input level to the level of interfering with model mechanisms, opening up a new direction for understanding and controlling model behavior. It demonstrates methodological innovation and potential academic impact.
- Across multiple jailbreak benchmarks , HMNS achieves state-of-the-art jailbreak performance on models of various scales and architectures. Moreover, it requires an exceptionally low average number of queries, demonstrating outstanding attack efficiency.
- Compared with traditional black-box attacks, HMNS offers higher interpretability. It helps identify vulnerabilities in safety alignment mechanisms and provides a valuable analytical tool for future security research.

### Weaknesses
- The paper conducts attack-defense experiments against six existing defense methods， However, the evaluation primarily focuses on input-stage and internal model defense mechanisms, without covering output detection based defense approaches such as self-defense [1]. It is recommended that supplementary experiments be incorporated to address this gap.
- The HMNS method is a white-box attack approach and cannot be applied to closed-source commercial models accessed via APIs (such as GPT-4, Claude, and Gemini). This limitation significantly reduces the method’s practical threat level and broad applicability, as the most critical real-world security challenges often stem from closed-source models. The paper should explicitly highlight this limitation and discuss possible directions for future research.

Reference:

[1] Phute, Mansi, et al. "Llm self defense: By self examination, llms know they are being tricked." arXiv preprint arXiv:2308.07308_ (2023).

### Questions
The paper uses KL divergence as the attribution method to identify the top-K attention heads with the greatest causal influence on the model’s output behavior. However, the following aspects require further clarification:

- Why was KL divergence chosen over other attribution metrics (e.g., gradient sensitivity)? Does KL divergence offer unique advantages in quantifying head-specific influence?
- Can it be demonstrated that the identified heads are indeed the most responsible for the model’s ​safety-aligned refusal behavior​​? Is there direct evidence linking these heads to safety-related functionality?
- Are there more robust or efficient ways to determine the heads most critical to safety mechanisms? For example, using contrastive attribution between safe and unsafe examples, or leveraging known characteristics of safety circuits?
- A compelling question for future work is whether the core principles of HMNS (e.g., causal attribution and precise intervention) can be repurposed or adapted to enhance model safety. For instance, could similar techniques be used to reinforce the stability of these components, thereby improving the model's robustness against adversarial manipulations?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes a jailbreak attack called Head-Masked Nullspace Steering (HMNS). The approach involves a direct intervention within the model by: 


$\textbf{1)}$ finding most important attention heads for normal behavior, 

$\textbf{2)}$ excluding/muting their output by column masking, and

$\textbf{3)}$ injecting a perturbations to the orthogonal complement of the muted subspace. 


The results show effectiveness of the approach and provide an ablation study

### Strengths
* Conceptually, the combination of interpretability and adversarial control is interesting 
* The authors provide an in-depth ablation study showing the contribution of each component.

### Weaknesses
# Major

1. **Threat Model:**
This topic is exhaustively investigated, and while this paper tries to provide a new perspective,  it came at the cost of very weak threat model and assumptions. The paper does not provide a threat model but the underlying assumptions (implicitly) understood from the approach is that the adversary has not only passive White-box access, but **active** white box access. This is a weak threat model; the adversary could arguably do any possible manipulation of the model's behaviour under these assumptions, and maybe there is no need then for the attention masking and all the effort the paper is proposing. Since the threat model is unrealistically privileged, I think the contribution may be of limited relevance to model alignment and safety.
2. **Missing relevant related work:** The proposed approach is based on the role of attention in the safety/alignment mechanism. However, the paper misses discussion relevant related work that uses attention as attack vector for Jailbreak. For example, 
- AttnGCG: Enhancing Jailbreaking Attacks on LLMs with Attention Manipulation (Wang et al., 2024) — shows that by manipulating attention-scores via prompt design (no head-masking or residual edits) can improve jailbreak success across multiple LLMs.
- Attention Eclipse: Manipulating Attention to Bypass LLM Safety‑Alignment (Zaree et al., 2025) — shows that attention manipulation (shifting attention away from safety-guard tokens) increases success rate and is transferable to closed models.

In contrast with this paper, a these attention-based jailbreak techniques operate with only input-level control, or regular white-box assumptions.

3. **Unfair comparison** Even for the jailbreak attack the paper compares to, the comparison is unfair because of the difference in the threat model. The paper must demonstrate strong gains (or new capabilities) that cannot be achieved via input-only or simpler internal manipulations. Without that, the added complexity and privileged access reduce the strength and impact of the contribution.

# Minor

- Table 2 appears early, and is not referenced in the text; I assume this should belong to (and be commented in) Section 5.

### Questions
The authors should explicitly discuss the threat model and the assumptions involved. The experimental setup and the analysis should also be from the prism of these assumptions for a fair comparison and to highlight what are the takeaways and contributions that this work provides compared to what we know from all existing jailbreak attacks

### Soundness
2

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
1. The paper discusses jailbreaking attacks against Large Language Models (LLMs), including generating adversarial suffixes for aligned models and optimizing attack methods.

2. It involves various technical frameworks, such as adaptive dense-to-sparse optimization, personalized encryption, and multiple defense baselines.

3. Mathematical formula analysis is proposed, such as utilizing the submultiplicative property of operator norms to measure the energy distribution of masked attention heads.

### Strengths
1. Covers multiple attack and defense approaches, and conducts comparative experiments under a unified framework with detailed tabular data.

2. Proposes HMNS as a novel attack strategy, which achieves significantly higher success rates than existing methods while maintaining low side-effect metrics.

3. Not only demonstrates experimental results, but also explains the mechanisms through operator norm and energy analysis.

### Weaknesses
1. HMNS has not been verified on closed-source commercial models such as GPT-4 or Claude.

2. Although methods like safety decoding and prompt patching are mentioned, experiments specifically targeting countermeasures against these defenses are lacking.

3. The impact of the attack on computational cost and inference speed has not been quantified.

4. It remains unclear whether the effectiveness of the attack degradation has been evaluated in multi-turn dialogue and long-context scenarios.

### Questions
see Weakness Section

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Head-Masked Nullspace Steering (HMNS), a circuit-level jailbreak attack that combines three mechanisms: (1) KL-divergence-based identification of causally important attention heads, (2) suppression of their write paths via column masking, and (3) injection of perturbations constrained to the orthogonal complement (nullspace) of the masked subspace. The method operates in a closed-loop detection-intervention cycle across multiple decoding attempts. Evaluated on AdvBench, HarmBench, JBB-Behaviors, and StrongReject benchmarks using LLaMA-2-7B-Chat, Phi-3-Medium-14B, and LLaMA-3.1-70B, HMNS achieves state-of-the-art attack success rates (96-99%) with low average query counts (ACQ ≈ 2). The authors introduce compute-normalized metrics (FEP, IPC, FPS, LPS) to account for internal overhead and demonstrate robustness under six defenses. Extensive ablations validate each component's contribution.

### Strengths
1. Theorems 2-7 provide formal guarantees about orthogonality, irreproducibility, and numerical stability. The mathematical framework is significantly more rigorous than typical jailbreak papers.

2. Leveraging nullspace projections to ensure perturbations cannot be canceled by masked heads is elegant and principled. This is a meaningful advance over ad-hoc activation steering.

3. Section 4.3 and Appendix A3-A6 introduce FEP-based accounting and compute-matched baselines. This addresses a critical methodological gap—most jailbreak papers report only external query counts.

### Weaknesses
1. While Appendix A3-A6 thoroughly analyzes compute, the main paper (Section 4.3) is brief. Key facts buried in appendix:
IPC ≈ 32 requires "proxy pre-selection" (A6.3) not detailed in main text;
Without pre-selection, IPC = 1 + 10·Tattribution passes = 101 worst-case (line 1577).

2. Missing comparisons: representation engineering, linear probe steering, contrastive activation addition (CAA is mentioned only in appendix A5.2)

3. Table 2 compares only to detector-based defenses, not other steering methods

### Questions
1. How much do head rankings change across the closed loop? Could you show attribution correlation matrices between iterations? Does instability explain the need for re-identification?

2. For the 1-4% of cases where HMNS fails (e.g., 96-99% ASR means 1-4% failure), can you characterize these failures? Are they:
Specific prompt types (e.g., certain harmful categories)? 
Model-dependent (e.g., certain architectures more robust)?
Predictable from attribution patterns?

### Soundness
3

### Presentation
3

### Contribution
3

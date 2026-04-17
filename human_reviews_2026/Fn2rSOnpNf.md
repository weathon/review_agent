# SlotGCG: Exploiting the Positional Vulnerability in LLMs for Jailbreak Attacks

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 4

## Abstract
As large language models (LLMs) are widely deployed, identifying their vulnerability through jailbreak attacks becomes increasingly critical. Optimization-based attacks like Greedy Coordinate Gradient (GCG) have focused on inserting adversarial tokens to the end of prompts. However, GCG restricts adversarial tokens to a fixed insertion point (typically the prompt suffix), leaving the effect of inserting tokens at other positions unexplored. In this paper, we empirically investigate slots, i.e., candidate positions within a prompt where tokens can be inserted. We find that vulnerability to jailbreaking is highly related to the selection of the slots. Based on these findings, we introduce the Vulnerable Slot Score (VSS) to quantify the positional vulnerability to jailbreaking. We then propose SlotGCG, which evaluates all slots with VSS, selects the most vulnerable slots for insertion, and runs a targeted optimization attack at those slots. Our approach provides a position-search mechanism that is attack-agnostic and can be plugged into any optimization-based attack, adding only 200ms of preprocessing time. Experiments across multiple models demonstrate that SlotGCG significantly outperforms existing methods. Specifically, it achieves 14% higher Attack Success Rates (ASR) over GCG-based attacks, converges faster, and shows superior robustness against defense methods with 42% higher ASR than baseline approaches. Our implementation is available at https://github.com/youai058/SlotGCG.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes SlotGCG, which extends GCG jailbreak attacks by inserting adversarial tokens at multiple vulnerable positions throughout prompts rather than only at the suffix. The method uses a Vulnerable Slot Score based on attention patterns to identify optimal insertion positions. Experiments on 6 LLMs show average 14% ASR improvement, faster convergence, and 42% higher robustness against defenses.

### Strengths
1. well-motivated problem: The systematic exploration of positional vulnerability is underexplored. 

2. Comprehensive empirical validation: Testing across 6 models × 4 attack variants × 4 defenses with consistent improvements demonstrates robustness of the approach.

3. Practical efficiency: The method adds only 200ms preprocessing but achieves up to 10× faster convergence, making it immediately deployable as a drop-in enhancement to existing GCG-based methods.

### Weaknesses
1. SlotGCG shows no improvement or degradation on Mistral-7B and Vicuna-7B in Table 1, but the paper provides no analysis of why positional vulnerability varies across architectures. This limits understanding of when the method applies.

2. The observation that defenses can increase ASR  due to GPT-4 filtering during optimization suggests the evaluation methodology itself may be problematic, undermining confidence in the reported improvements.

3. Some hyperparameters lack justification, e.g., why temperature T=8? What happens with other layer selections or temperatures?

### Questions
1. Can you characterize what architectural or training differences cause SlotGCG to fail on Mistral/Vicuna?

2. What is the performance with (1) only lower layers, (2) only upper layers, (3) random layer selection?

3. AutoDAN also uses flexible token placement. How does SlotGCG compare in effectiveness and efficiency?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper argues that jailbreaking susceptibility depends strongly on where adversarial tokens are inserted. It introduces a Vulnerable Slot Score (VSS) to rank token positions by “positional vulnerability,” and proposes SlotGCG, which allocates/optimizes adversarial tokens at high-VSS slots rather than only at the suffix. Across Llama-2/3-8B, Mistral-7B, Vicuna-7B, and Qwen-2.5, SlotGCG reportedly improves attack success rate (ASR) over several GCG-family baselines, converges in fewer iterations, and remains more effective against several input-filtering defenses.

### Strengths
* The paper presents a clear and original problem framing, defining insertion slots, formalizing the Vulnerable Slot Score (VSS), and linking it to attention patterns to show that positional vulnerability is largely prompt dependent.
* The method is attack-agnostic and simple, with a clear step-by-step presentation. The analysis is insightful, using random multi-position insertion and attention heatmaps that convincingly support the positional hypothesis.
* The results show significant empirical gains, with large performance improvements and meaningful reductions in the number of optimization steps needed for success.
* The attack appears more robust than prior methods against several defenses.
* The writing is clear, well-structured, and easy to follow.

### Weaknesses
1. The paper lacks an analysis of transferability across models. It remains unclear whether positional vulnerabilities are model-specific or primarily prompt-dependent. Evaluating SlotGCG as a black-box attack would provide valuable insight into this question.
2. The evaluation is limited to AdvBench, while several newer jailbreak or safety datasets now exist [1-3]. Including additional benchmarks would strengthen the empirical claims and demonstrate broader robustness.
3. The method is only tested within the GCG family. Other optimization-based attacks exist, and it is unclear whether the proposed position-finding process generalizes to them. Since the abstract claims applicability to “any” attack, evidence from beyond GCG is needed.
4. The defense selection is weak. The Erase-and-Check (suffix) version is expected to perform poorly when there is no suffix, as it effectively just deletes the response. Evaluating only the SmoothLLM swap defense is also insufficient; since the attack produces more uniform attention maps, token swapping may be less effective. Other SmoothLLM variants (insert, patch) and stronger recent defenses [4] should be tested for a fair assessment. 
5. The reported preprocessing cost of “+200 ms” is not empirically demonstrated or discussed in detail. The paper should clarify how this value was obtained.
6. Although the calculation of the Vulnerable Slot Score (VSS) is novel, the general idea that attack performance depends on token position has been explored up to some level in prior work [5-7]. These earlier studies should be acknowledged and discussed.

***Minor remarks:***
1. Table 5 presents very strong and important results, so it would be better placed in the main text rather than the appendix to highlight its contribution more clearly.
2. Line 472 refers to “Table 3” for the VSS distribution, but this table is no related. This reference should be corrected.

[1] Zeng, Y., Shen, T., Ding, Y., Zheng, L., Sun, Y., & Chen, H. (2024). JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models

[2] Mazeika, M., Wei, A., Casper, S., Rafailov, R., Dragan, A. D., Finn, C., & Hadfield-Menell, D. (2024). HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal

[3] Xu, W., Wang, X., Zhang, Z., & Li, M. (2023). ToxicChat: Unveiling Hidden Challenges of Toxicity Detection in Real-World User-AI Conversation

[4] Yi, S., Liu, Y., Sun, Z., Cong, T., He, X., Song, J., Xu, K., & Li, Q. (2024). Jailbreak Attacks and Defenses Against Large Language Models: A Survey

[5] Wang, J., Li, H., Peng, H., Zeng, Z., Wang, Z., Du, H., & Yu, Z. (2025). Activation-Guided Local Editing for Jailbreaking Attacks.

[6] Mu, J., Ying, Z., Fan, Z., Jing, Z., Zhang, Y., Yu, Z., & Zhang, X. (2025). Mask-GCG: Are All Tokens in Adversarial Suffixes Necessary for Jailbreak Attacks?

[7] Rocamora, E., Dubey, A., Jauhri, A., Pandey, A., Letman, A., Mathur, A., & Vaughan, A. (2024). Revisiting Character-Level Adversarial Attacks for Large Language Models.

### Questions
1. Are token budgets (total adversarial tokens) matched across baselines in Table 1?
2. How sensitive are results to VSS temperature, and number of slots selected?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces SlotGCG, a positional variant of GCG that exploits positional vulnerability in LLMs. Instead of appending adversarial tokens as a suffix, SlotGCG identifies vulnerable token slots within the prompt using a lightweight Vulnerable Slot Score (VSS) derived from attention patterns, then inserts and optimizes attack tokens at those positions. The method is attack-agnostic and can be used as a plug-in front end to multiple GCG-style optimizers with minimal extra overhead. Experiments on several open-source models report higher ASR, faster convergence, and improved robustness under certain defenses, with success judged via automatic and human checks.

### Strengths
1. Novelty

Reframes jailbreak optimization from “suffix-only” to positional attacks by identifying vulnerable token slots via a lightweight attention-derived score (VSS) and inserting/optimizing adversarial tokens at those positions.


2. Method is general, plug-and-play, and more efficient


Attack-agnostic front end that can be attached to multiple GCG-based optimizers with minimal overhead.
Results show faster convergence/fewer steps and higher ASR than standard suffix-only pipelines under comparable budgets.

3. Good experimental coverage

Evaluates across several commonly used open-source instruction models (e.g., Llama, Mistral, Vicuna, Qwen).
Adapts to multiple GCG-based attack variants and compares under several defenses, demonstrating consistent gains.

### Weaknesses
1. Threat model and usability boundaries

The core VSS metric depends on attention weights (upper-half layers from the after-chat template to adversarial tokens), which are typically unavailable in black-box/closed models. The paper does not clarify applicability in strict black-box settings or provide surrogate attack choice.

2. Transferability is underexplored
  - Cross-model transfer: Do attack prompts found on one model transfer to other models without further optimization (zero-shot transfer)?
  - Seed sensitivity: How does optimization vary with different random seeds (initial tokens, sampling orders)? 
  - Context/system-prompt robustness: For the same target model, does changing the system prompt or different context affect ASR?

3. Recency of attack targets

Experiments focus on open-source instruction models (the newest being Qwen-2.5). There is no demonstration on newer/stronger/closed-source LLMs, limiting external validity.

4. Hyperparameter choices lack justification

The effects of temperature in VSS, the precise definition of “upper-half layers,” and the impact of different after-chat template tokens are not detailed and analyzed. It remains unclear how sensitive VSS and final ASR are to these design choices.

5. Confusion in section THE ROBUSTNESS OF SLOTGCG UNDER DEFENSE METHODS

Perplexity Filter yields 0 ASR for all attack variants, yet the paper claims “Erase-and-Check yields the largest reduction in ASR.” This seems to appear inconsistent.

The paper attributes some failures to GPT-4 misclassification due to biases in the GPT-based filtering mechanism, but overall ASR is still measured by the same GPT-based judge. This creates a tension: if the judge is unreliable for filtering, why is it reliable for final success labels?

6. Motivation and definition of VSS are hard to follow

Figure 4 is used to motivate “developing a metric,” but VSS has not yet been defined at that point, making the figure difficult to interpret on first read.

### Questions
1. White-box assumptions and transferability

- Is SlotGCG a pure white-box attack (requiring attention weights) during both scoring and optimization?

- If so, can the resulting adversarial prompts transfer to other models without further optimization (zero-shot cross-model transfer)? 

2. Effectiveness against deployed guardrails

Can you please evaluate SlotGCG against current guardrails (e.g., Llama Guard or similar safety classifiers/filters)?

3. Which VSS is shown in Figures 4 and 8?

In Figures 4 and 8, $\text{VSS}^{\text{final}}$ represents the VSS of which slot?

4. Random Multi-Position Insertion in Figure 5

What is the exact algorithm for **Random Multi-Position Insertion**? Why does random slot insertion without token optimization achieve faster convergence than GCG?

5. Ablation on only insertion and token allocation via VSS

Can you provide results for **VSS-based slot insertion only** (no token optimization), and compare them with **GCG-only token optimization** (no VSS-based slotting)? An ablation contrasting these two against the full SlotGCG would clarify each component’s contribution.

6. Effect of the **token budget (m)**

How does the **token budget \(m\)** affect SlotGCG’s **ASR**, convergence speed, and stability? Please include curves or tables showing performance as \(m\) varies.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces SlotGCG, a novel extension of gradient-based jailbreak optimization that explicitly models positional vulnerability in prompts. The key idea is to identify slots—token-level positions that are more susceptible to adversarial perturbation—using an attention-derived Vulnerable Slot Score (VSS). The method first probes each slot’s sensitivity, then assigns probabilistic insertion weights and integrates them into the GCG optimization loop. Experiments across multiple open-weight LLMs and defenses demonstrate that SlotGCG improves attack success rate, convergence efficiency, and robustness against defense mechanisms, while remaining lightweight and compatible with existing frameworks.

### Strengths
The paper connects positional token vulnerability with optimization-based jailbreaks, introducing VSS as a quantifiable measure of slot sensitivity. The slot-probing stage is lightweight and can be easily integrated into other attack pipelines, enhancing general applicability.

### Weaknesses
1. Tokenizer dependence – As slots are token-based, specify the tokenizer used and discuss whether different tokenizers could affect slot boundaries or results.

2. Optimality of Step 3 formula – It is unclear whether the slot-selection formula is optimal. Would selecting top-k slots and renormalizing yield different outcomes? Clarify if this is a tunable hyperparameter and analyze its effect on ASR and prompt coherence.

3. Defense and baselines – The defense side lacks diversity and novelty. The chosen baselines and target models are relatively standard and dated. Including stronger or more recent defense baselines (e.g., [1] [2]) would strengthen the experimental credibility.

4. Limited contribution – The method builds on optimization-based jailbreak attacks (e.g., GCG), yet its improvements appear easily neutralized by simple defense strategies. This raises the question of why such an optimization-based formulation is chosen in the first place. If the approach can be trivially mitigated, the paper should clarify what fundamental insight or practical benefit this “slot vulnerability” perspective contributes beyond existing optimization-based jailbreak methods.

5. Target model and PPL results – Sec. 5.3 does not specify the target model, and the statement that “PPL mitigation is moderate” seems inconsistent with near-zero results. Please clarify both.

6. Unclear notation – In Step 3 of Sec. 4, the variables fsi and S* are undefined. Add explicit notation or a brief symbol explanation for clarity.

7. Minor textual error – Line 213 should mention three prompts instead of four. Please verify and correct.

8. Slot normalization – In Sec. 3.1, slot indices are normalized by the longest prompt in the batch, which likely prevents values near 1.0. The motivation and comparison with per-prompt normalization should be clarified.

[1] Robust Prompt Optimization for Defending Language Models Against Jailbreaking Attacks

[2] SafeDecoding: Defending against Jailbreak Attacks via Safety-Aware Decoding

### Questions
see above

### Soundness
2

### Presentation
3

### Contribution
2

# Beyond Refusals: Fine-grained Safety Alignment for Reasoning LLMs

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 6, 4, 4

## Abstract
Large Reasoning models (LRMs) with reasoning capabilities have demonstrated remarkable performance on complex tasks, yet achieving robust safety alignment remains a significant challenge. Supervised fine-tuning (SFT) with safety data is a widely-used approach to improve the models' safety, however, we identify that current safety alignment methods with SFT often induce a phenomenon we term $\textbf{Shortcut Alignment}$. 
In this case, the model learns to recognize the patterns in harmful inputs and emit templated refusals (e.g., "I'm sorry...") while decoupling the final response from its internal chain-of-thought (CoT) reasoning. This superficiality leads to two critical problems: (i) refusals without reasoning carry no informative value, and (ii) models become overly cautious, leading to excessive false refusals on benign queries and thereby degrading their general helpfulness.
To understand this behavior, we formalize it through the lens of conditional mutual information (CMI), hypothesizing that when the information gain from CoT is low, such shortcuts become low-resistance solutions that reduce training loss with little cost. We empirically verify this hypothesis via probe experiments that estimate the gap between predictions with and without CoT on harmful versus benign data. 
Motivated by these insights, we propose Deep Instruct Fine-tuning  (DIFT), which uses $\textbf{CMI-Loss}$, explicitly penalizing shortcut predictions while preserving original instruct-tuning on benign examples. Through theoretical analysis and empirical evidence, we show that our method offers a better solution. It alleviates erroneous refusals while preserving safety. Our work bridges theory and practice, offering the first fine-grained alignment method that explicitly targets shortcut alignment in LRMs.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper addresses "Shortcut Alignment" in large reasoning models (LRMs), a phenomenon where models learn to issue templated refusals (e.g., "I'm sorry...") to harmful inputs without actually grounding the decision in their internal chain-of-thought (CoT) reasoning. This superficial alignment not only makes refusals uninformative but also causes models to become overly cautious, leading to excessive false refusals on benign, safe queries. To fix this, the authors propose Deep Instruct Fine-tuning (DIFT), a new method that uses a "CMI-Loss" function. This CMI-Loss specifically penalizes the direct input-to-refusal "shortcut" on harmful examples, forcing the model to rely on its CoT reasoning to make a safety decision. The results demonstrate that this method successfully alleviates erroneous refusals on benign inputs while preserving the model's safety and reasoning capabilities.

### Strengths
1. The paper introduces "Shortcut Alignment", an observation that current safety methods (SFT) can lead to models appearing safe by using superficial cues to generate templated refusals, while decoupling this refusal from their internal reasoning (CoT).
2. The paper introduces Deep Instruct Fine-tuning (DIFT), which uses a CMI-Loss to penalize the x→y (input-to-answer) shortcut.
3. The authors validate their method by showing it reduces over-refusal on benchmarks while maintaining safety on benchmarks and preserving general reasoning abilities. They further perform probe-based analyses that confirm the intended mechanism that refusals become more reliant on the CoT rather than generic cues.

### Weaknesses
1. There is no description of the evaluation method and its justification. How is the refusal accuracy and non-refusal rate calculated? How reliable is the evaluation?
2. The improvement in over-refusal is not significant. The p-value shows "most likely" that the proposed method improves over-refusal over the baseline, not indicating the scale of improvement.
3. The abstract mentions one of the problems for "Shortcut Alignment" is "refusals without reasoning carry no informative value". The rest of the paper does not mention this point, and the proposed method does not improve on this problem.
4. Why and whether "Shortcut Alignment" will lead to over-refusal needs more intuitive and quantitative justification. 
5. Presentation issues: For instance, in Figure 2, the reason for using normalized harmful dependence, and what it indicates, are not explained. Others see Questions.

### Questions
1. Line 192 can remove the "(A)" since there is no other paragraph.
2. Strata-Sword is mentioned in line 308 but not used in the experiment.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper identifies and defines Shortcut Alignment as a key failure mode in Large Reasoning Models (LRMs). The authors posit that models learn to bypass their internal Chain-of-Thought (CoT) processes, instead issuing templated refusals based solely on surface cues from the input. They argue this shortcut is the root cause of widespread over-refusal on benign queries, which degrades the model's general helpfulness. To address this, the paper proposes a new training method called DIFT, centered on a CMI-Loss.

### Strengths
The paper's  contribution is the identification and formalization of "Shortcut Alignment." This is a precise and insightful diagnosis of why modern safety-aligned LRMs suffer from over-refusal. Instead of relying on the generic "safety vs. helpfulness trade-off," the authors provide a mechanistic explanation: the model's refusal decision becomes decoupled from its internal reasoning.

### Weaknesses
1. The paper's "Shortcut Alignment" premise is undermined by its own evidence in Appendix G.3. The failure case (Case E1) does not show a \mathbit{c}\ \rightarrow\ \mathbit{y} decoupling; rather, the refusal (\mathbit{y}) is perfectly faithful to a flawed CoT (\mathbit{c}) that misinterprets the query. This suggests the root cause of over-refusal may be poor CoT quality, not a dependency on the CoT. The proposed CMI-Loss, which enforces \mathbit{c}\ \rightarrow\ \mathbit{y} dependency, may therefore be misdiagnosing the problem.

2. The CMI-Loss mechanism enforces the answer's (\mathbit{y}) dependency on the CoT (\mathbit{c}), which implicitly assumes the CoT is robust. However, the CoT itself is a known attack vector (e.g., H-cot [1]、mousetrap [2].). By training the model to trust its CoT more, the method risks amplifying the impact of CoT poisoning or hijacking attacks, forcing the model to faithfully execute a compromised reasoning chain. The paper lacks a robustness analysis against this critical, emergent threat.

- [1] Martin Kuo, Jianyi Zhang, Aolin Ding, Qinsi Wang, Louis DiValentin, Yujia Bao, Wei Wei, Hai Li, and Yiran Chen. H-cot: Hijacking the chain-of-thought safety reasoning mechanism to jailbreak large reasoning models, including openai o1/o3, deepseek-r1, and gemini 2.0 flash thinking. arXiv preprint arXiv:2502.12893, 2025.

- [2] Yang Yao, Xuan Tong, Ruofan Wang, Yixu Wang, Lujundong Li, Liang Liu, Yan Teng, and Yingchun Wang. A mousetrap: Fooling large reasoning models for jailbreak with chain of iterative chaos. arXiv preprint arXiv:2502.15806, 2025.

3. The central claim of "preserving safety" is not fully supported by the quantitative data in Table 1. In several instances, the 'ours' method scores lower on key safety benchmarks (e.g., WildJB, WildChat) than the baseline. For instance, DeepSeek-32B's WildJB score drops from 90.8 to 88.8, and Qwen3-14B's drops from 94.4 to 92.8. This consistent (though small) degradation suggests the method introduces a "safety tax" and is, in effect, trading safety for reduced over-refusal (NOR), which should be explicitly acknowledged.

### Questions
See above.

### Soundness
3

### Presentation
2

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
This paper addresses the problem of "Shortcut Alignment" in LRMs, where models learn to emit template refusals from surface cues while decoupling the final response from internal chain-of-thought reasoning. The authors propose Deep Instruct Fine-tuning (DIFT) with a Conditional Mutual Information Loss (CMI-Loss) that penalizes shortcut predictions on harmful examples while preserving standard supervised fine-tuning on benign data. Experiments on Qwen3 and DeepSeek-R1 models demonstrate improved safety and reduced over-refusal rates while maintaining reasoning capabilities.

### Strengths
- The paper formalizes shortcut alignment through conditional mutual information, providing clear mathematical justification for why models learn input-only shortcuts.

- The probe-based analysis with training-time and inference-time diagnostics provides valuable mechanistic insights into how the method shifts model reliance to matched CoT.

- The paper is well-written and is easy to follow.

### Weaknesses
I generally recognize the approach and motivation of this work, but I have several concerns that need to be addressed:

- **Insufficient Experimental Details and Unclear Evaluation Protocols.** The experimental evaluation lacks critical details necessary for reproducibility and proper interpretation. The paper does not clearly specify how many samples from each benchmark were used, nor does it describe the evaluation methodology in sufficient detail (keyword matching vs. LLM-as-judge). Most importantly, the StrongReject evaluation appears to test only weak attacks, as evidenced by near-perfect safety rates (99-100%) across all models in Table 1. StrongReject is designed to evaluate robustness against strong adaptive jailbreak attacks, yet the results show no room for improvement, making it impossible to differentiate method effectiveness. The paper needs to clarify which StrongReject attack variants were tested and why stronger attacks were not included, as this significantly undermines the claimed robustness benefits.

- **Missing Comparisons with State-of-the-Art Methods.** The paper lacks comparisons with recent state-of-the-art safety alignment methods specifically designed for reasoning models, such as Safechain, RealSafe-R1 and Oyster-I. Table 2 only compares against test-time interventions (SVA, SCANS), which serve different purposes than training-time alignment. Without these critical comparisons, it is impossible to assess whether the proposed method truly represents an advancement over current approaches. This is also a significant gap that needs to be addressed.

- **Unprofessional Presentation and Superficial Related Work.** The paper exhibits several presentation issues that detract from its professionalism. The experimental analysis frequently uses informal arrow notation. Additionally, Figure 1 suffers from illegibly small fonts. More critically, the related work section (Section 4) is cursory and lacks systematic organization. It briefly mentions evaluation benchmarks and test-time interventions without providing a clear taxonomy of existing approaches, detailed discussion of concurrent work, or critical analysis of why existing methods fail. Given the rapidly evolving nature of this research area, a comprehensive literature review is essential to properly position the contribution.

- **Questionable Assumptions About CoT Quality.** The entire method rests on a critical assumption that the chain-of-thought reasoning is reliable and that strengthening the coupling between CoT and the final answer will improve safety. However, what if the CoT itself contains unsafe reasoning, biased logic, or arrives at a safe conclusion through flawed reasoning paths? The paper does not adequately address scenarios where the CoT quality is problematic.

### Questions
The questions are listed in the weaknesses, and if authors can address my concerns, I will raise my rating.

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
4

### Summary
This paper addresses the problem of shortcut alignment in safety fine-tuning of large reasoning models, which tend to produce templated refusals depending on the prompts instead of grounded in reasoning. The authors formalize this phenomenon using conditional mutual information (CMI) between the final answer and the CoT, and propose a new fine-tuning objective, CMI-Loss, which penalizes harmful-only input–output shortcuts while preserving standard supervised fine-tuning on benign data. Experiments on several safety and over-refusal benchmarks across Qwen3 and DeepSeek series show that the approach reduces over-refusals on benign inputs without degrading safety or reasoning ability.

### Strengths
* The paper identifies the issue of "shortcut alignment", which is an interesting and important problem in safety alignment for reasoning LLMs.
* The theoretical framework is well-structured with CMI, offering a reasonable foundation for the simple approach.
* The experiments are extensive, covering multiple model families and popular benchmarks.
* The presentation and visualizations are clear and intuitive, effectively supporting the analysis.

### Weaknesses
* **The motivation is not clear.** The introduction points out that existing alignment methods for reasoning models face two main issues: over-refusal and templated refusal. However, in the analysis and method sections, the paper mainly focuses on the latter, while the experimental results primarily reflect improvements on the former. The paper does not clearly explain whether there is a causal relationship between these two problems. In addition, the claim that templated refusals are "decoupled" from the model’s actual reasoning process is not directly supported by evidence. As these ideas are the core motivation of the work, I find it a little difficult to follow at the beginning and this also impacts the value of the problem.
* **The evaluation on safety is limited.** The paper mainly uses standard harmful-question datasets such as StrongReject and JBB, but lacks testing on common jailbreak attacks. From the results on WildJailbreak, the proposed method does not consistently improve or maintain safety performance. This raises concerns about the robustness and reliability of the approach
* **The interpretation of the $\Delta Y$ dynamics in Figure 2 (left) is a little confusing (personally).** At the beginning of training, the gap between benign and harmful samples is large, but the reason for this difference is not explained. Does this mean that before training, there has been a divergence between the response pattern to benign and harmful prompts ? Moreover, the overall $\Delta Y$ values decrease as training progresses, which seems inconsistent with the claim that the model’s reasoning–refusal coupling becomes stronger. It would be helpful to show the training curve of the baseline model to verify that the observed trends are indeed due to the proposed CMI-loss rather than general training dynamics.
* **Formatting and citation issues.** The reference format seems changed from the default author-year form in the template. I am not sure if this breaks any guidelines for submission, so I will leave this to AC. But at least, the reading is incoherent due to the appearance of numbers. Besides, there is at least one broken appendix reference ("Appendix ??") near line 269.

### Questions
Seak weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

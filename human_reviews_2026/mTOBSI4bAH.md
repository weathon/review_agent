# Term2Note: Synthesising  Differentially Private Clinical Notes from Medical Terms

- Avg Score: 2.67
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 2

## Abstract
Training data is fundamental to the success of modern machine learning models, yet in high-stakes domains such as healthcare, the use of real-world training data is severely constrained by concerns over privacy leakage. A promising solution to this challenge is the use of differentially private (DP) synthetic data, which offers formal privacy guarantees while maintaining data utility. However, striking the right balance between privacy protection and utility remains challenging in clinical note synthesis, given its domain specificity and the complexity of long-form text generation.
In this paper, we present Term2Note, a methodology to synthesise full-length clinical notes under strong DP constraints. By structurally separating content and form, Term2Note generates section-wise note content conditioned on medical terms, with terms and notes privatised under separate DP constraints. A DP quality maximiser further enhances synthetic notes by selecting high-quality outputs.
Experimental results show that Term2Note produces synthetic notes with statistical properties closely aligned with real clinical notes, demonstrating strong fidelity. In addition, multi-label classification models trained on these synthetic notes perform comparably to those trained on real data, confirming their high utility. Compared to existing DP text generation baselines, Term2Note achieves substantial improvements in both fidelity and utility, while avoiding reliance on label distribution assumptions, suggesting its potential as a viable privacy-preserving alternative to using sensitive clinical notes.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper presents Term2Note, a framework for generating differentially private synthetic clinical notes from medical terms. It addresses the challenge of creating long-form medical text that ensures both privacy protection and data utility. Term2Note generates clinical notes section by section. Each section is based on the extracted medical terms and the previously generated sections, and each process, term generation and note generation, is protected under independent differential privacy (DP) constraints. The framework further enhances text quality through a DP quality maximiser, which selects the most coherent outputs based on perplexity. The authors claim that Term2Note achieves high fidelity to real clinical notes, competitive utility in downstream ICD code prediction tasks, and superior performance to existing DP text generation baselines, while maintaining formal privacy guarantees.

### Strengths
The paper proposes a new structured framework for generating long-form synthetic clinical text, a problem that has rarely been explored due to the complexity of medical contents and strict privacy constraints. The framework is systematically modularized, dividing the synthetic data generation process into distinct stages—term extraction, term generation, note generation, and quality maximization—which makes the overall structure organized and understandable. It also effectively leverages existing DP algorithms and their theoretical properties to ensure formal privacy guarantees throughout the workflow, maintaining a careful balance between privacy and utility. Overall, the work holds methodological significance for healthcare AI by suggesting a new approach to generating and sharing sensitive medical data, addressing one of the major challenges in privacy-preserving data acquisition.

### Weaknesses
1. Limited Stability of Utility Results

The paper claims that Term2Note consistently outperforms other baselines in terms of fidelity and utility. This claim is well supported for fidelity, as Term2Note shows superior structural, syntactic, and semantic similarity across all privacy settings. However, the result in utility do not fully support this statement. While Term2Note achieves the highest F1 score, some results in AUC and Precision values are slightly higher for AUG-PE and FastDP across privacy budgets (ε = 8, 5, and 2), according to Table 1. It suggests that the advantage in utility is not fully uniform across settings. Also, the inconsistency between statement and experiment results can weaken the credibility of the paper. Acknowledging these variations and discussing their possible causes, such as model randomness or task sensitivity, would make the analysis more balanced and credible.

2. Concerns Raised in Qualitative Analysis

Physicians’ feedback in Qualitative Analysis section suggests discrepancies with the paper’s assertion of high-quality text generation. The evaluators pointed out several issues in Term2Note outputs, including missing or misordered sections, internal inconsistencies, and vague or generic phrasing. One physician also noted medication misclassification, illogical or irrelevant narrative insertions, and factual errors. Although these issues may not be prevalent, their presence indicates that further refinement is needed to ensure clinical reliability and practical usability. The authors are encouraged to investigate these qualitative errors systematically and discuss possible mitigation strategies.

3. Lack of Clarity in Notation and Experimental Setting

The paper omits several explanations that are necessary for readers to fully understand the framework. In 2 Background & Related Work, the mathematical formulation of DP is insufficiently explained: variables such as 𝜒 and 𝑅 appear without definition, making the DP derivation difficult to follow. This paper replaces empirical validation of the crucial element of privacy protection with mathematical proofs alone, without sufficient experiments. Therefore, a highly rigorous mathematical statement is required.
Moreover, in 4 Experiment Setting, the baseline description is uneven: while the DP-SGD methodology underlying FastDP is explained earlier in the paper, AUG-PE is only briefly mentioned in one sentence without sufficient explanation of its mechanism or how it differs from Term2Note. Providing explicit definitions and more balanced baseline explanations would greatly improve readability.

4. Insufficient Justification for Model and Hyperparameter Choices

The paper lacks justification for model selection and hyperparameter configuration. Term2Note integrates multiple large models—GPT-2 Large, Llama-3.2-1B or Gemma-3-1B, Asclepius-Llama3-8B—but the rationale for choosing these specific architectures or parameter scales is not discussed. In addition, the paper sets the ε (epsilon) hyperparameters to 2, 5, and 8 for both term-level and note-level privacy budgets, yet provides no explanation for why these specific values were chosen or how they correspond to concrete levels of privacy protection. A short discussion of how these ε values translate to practical privacy guarantees would help readers better interpret the trade-off between privacy and utility.

### Questions
1. Scope of DP Definition: In 2 Background & Related Work, you define DP for an arbitrary randomized function. Immediately after, the text states “any function”. Is this extension intentional and mathematically valid? If so, please clarify the conditions under which the result holds, and cite a supporting theorem. Also, please provide the definition for “randomized function” or “randomized algorithm”.

2. TERMEXT Notation Consistency (Sec. 3.2): The final line in 3.2 Format and Term Identification writes TERMEXT([SEC1, ... , SECm]). Should this instead write [TERMEXT(SEC1), ... , TERMEXT(SECm)]? If the current form is correct, please explain how TERMEXT operates.

3. Index Variable Mismatch (Alg. 1 vs. Sec. 3.4): Algorithm 1 says 𝑗 indexes sections, while Sec. 3.4 uses 𝑖 for sections. Is one of these a typo? Please standardize the index variable across the paper or state the mapping explicitly.

4. Model Capacity for Note Generation: Physician feedback points to missing/misordered sections, internal inconsistencies, and medication errors. Consider evaluating a larger NOTEGEN (e.g., next-scale Llama/Gemma) under the same DP budgets. It could show whether quality deficits stem from model size rather than the framework.

5. End-to-End Example Through the Paper: To make the framework easier to follow, it would be helpful to include one representative example that runs through the entire Term2Note pipeline. Such an example would help readers intuitively understand how the original text is transformed into a synthetic note and how potential re-identification risks are mitigated.

6. Paragraph Break in Sec. 3.3 (Line 190): Up to “projected embeddings.” appears to explain Algorithm 1, Line 5; the text then shifts to Line 7. It would be better to split into two paragraphs or add a transition sentence to mark the change in step.

7. Clarification on ε Bound (Sec. 3.6): In Section 3.6, the paper states that the overall ε is upper-bounded by the larger of εₙ and εₜ. However, in Table 1, configurations are presented where εₜ = ∞ while εₙ decreases. If ε is bounded by the larger value, the overall privacy budget should remain infinite. How is this reconciled? Please provide an explanation.

8. Explanation and examples for MIMIC Datasets: The paper uses MIMIC-III as the public dataset and MIMIC-IV as the private dataset. Could the authors elaborate on the key differences between the two and justify why each is suited for its respective role? Were any preprocessing or normalization steps? Providing an example or brief explanation would help readers understand the rationale for this.

9. Clarification on Semantic Similarity Evaluation: The paper equates semantic similarity with distributional alignment measured in MAUVE. However, MAUVE evaluates global distributional fidelity rather than sentence- or concept-level semantic similarity. Could you clarify how distributional similarity reflects semantic similarity, or justify this interpretation?

10. Typos and Style Consistency: 

- Line 223: “preform” → “perform.”

- Line 485: Replace the period after “NLP” with a comma. 

- Table 1: Unify the spelling of FastDP (with or without hyphen) across the paper.

### Soundness
1

### Presentation
2

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
This paper focuses on how to acquire differentially private clinical notes for reliable, privacy-preserved research. The paper proposes Term2Note, a method that first produces a series of medical terms, then generates private clinical notes based on these terms. To achieve this, Term2Note includes a fine-tuned embedder and a TermGen model for term generation, whose objective is to reconstruct the ground-truth term lists. Next, Term2Note includes a NoteGen model responsible for generating clinical notes. Experiments demonstrate the fidelity (measured by how closely the generated notes resemble real data) and utility (measured by their usefulness for downstream tasks).

### Strengths
1. The design of term generation followed by text generation makes the model more transparent and explainable, which could be essential for clinical use.
2. Experimental analyses are conducted under various $\epsilon$ settings, which helps reveal the applicability of the proposed method under different privacy requirements.
3. Compared with automated metrics, human evaluation with licensed physicians makes the results more convincing.

### Weaknesses
1. Some ablation studies could demonstrate the role of each module more clearly. For example, according to Algorithm 1, Term2Note first relies on $\theta_t$ and $\theta_p$ to generate clinical terms, and the training objective is to reconstruct the ground-truth terms in $D_{public}$. As an ablation, what if the model directly uses the ground-truth terms from $D_{public}$ during inference? What would the performance gap be, and how would the term generation model perform? This becomes more important since $\theta_t$ is GPT-2-Large, which does not have very strong capability compared with current LLMs.
2. The experimental results do not show solid improvements. According to Table 1, the utility measures between Term2Note and the baselines are comparable, and FastDP often performs better across various $\epsilon$ settings. This seems to contradict the analysis that Term2Note achieves the best performance under varying privacy budgets (L421–423).
3. Some implementation details could be described more clearly. For example, term generation first projects the embeddings with $\theta_p$, then generates using the generative model $\theta_t$. How are the embeddings provided to $\theta_t$ (e.g., fed as inputs, used in cross-attention during generation, etc.)?
4. The main contribution of this paper seems to lie in the pipeline design (first generate medical terms, then generate clinical notes). The algorithms used, such as FastDP, are from previous works. Thus, the technical contribution could be limited.
5. Minor: Figure 2 includes negative N-gram frequency. Although this is due to smoothing, negative N-gram frequency seems do not have physical meanings, and maybe the paper should consider truncate them.

### Questions
Please see the weakness, especially for 1-4. The rebuttal could focus on better explanation of technical innovation and focus on some ablation studies to demonstrate the role of term generation and notes generation respectively. Also, It could be helpful to explain the experiment results and implementation clearer (Weakness 2,3) or highlights the part I misunderstood.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces a novel framework named Term2Note for synthetic clinical note under DP constraint. The method process long-form clinical note by separating it into two sub tasks: key medical term extraction and note format generation. After applying DP training methods, experiment outcomes imply that Term2Note outperforms baseline models on fidelity to real clinical data, downstream task utility and human expert preference, while providing formal privacy guarantees. The author conducted thorough experiments on variants including base model and privacy budget, which provides a solid foundation for the main contribution.

### Strengths
The framework fulfills existing gap between synthetic text generation and differential private on clinical notes.

The evaluation is thorough。

### Weaknesses
The paper adopts FastDP but provides little explanation for why this particular method is preferred over other differentially private mechanisms (e.g., DP-SGD variants, PATE, or Rényi-based approaches). The DP section reads more like a straightforward application of an existing framework rather than a carefully motivated design.

The composition analysis is overly simplified and lacks a solid connection to established DP accounting techniques. From a differential privacy standpoint, the reasoning behind the overall privacy budget allocation and composition strategy is not well articulated, making the DP guarantee less convincing.

The empirical results in Table 2 appear inconsistent with the expected DP–utility trade-off. Specifically, when epsilon goes infinity the results should approximate the non-private baseline, and as ϵ decreases, utility should generally drop. However, this trend is not observed, and in some cases, DP models even outperform the non-DP setting, which is counterintuitive and suggests potential issues in experimental implementation or evaluation. And the budget is relatively large, especially after naive composition,  makes the dp guarantee very weak.

The paper does not clearly justify the use of the term extraction–narrative generation framework, nor discuss the mechanisms underlying this design. A brief explanation of the intuition behind this two-stage formulation (why term extraction benefits note generation and how it differs from end-to-end models) would help clarify the methodological motivation and strengthen the paper’s contribution.

The choice of baselines is not well justified. From a utility perspective, the paper should compare against clinical note generation methods without DP, even if they lack formal privacy guarantees. Within the DP setting, using only a simple DP-SGD + LLM configuration is not an adequate baseline for evaluating the effectiveness of FastDP.

The diagram illustrating the proposed framework lacks clarity—some arrows and dependencies are ambiguous, and the role of DP within the overall pipeline is not clearly reflected. The figure does not effectively communicate how the DP mechanism is integrated into the model training or generation process.

Comparing text length distribution with KL divergence do not intuitively capture structural or semantic similarity between the synthetic clinical notes and the real data. While this metric may reflect differences in overall length patterns, it provides limited insight into whether the generated notes preserve linguistic coherence or structural arrangement.

### Questions
As above in weakness

### Soundness
3

### Presentation
2

### Contribution
3

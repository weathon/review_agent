# Tracing and Reversing Edits in LLMs

- Decision: Accept (Poster)
- Scores: 4, 4, 6, 6

## Abstract
Knowledge editing methods (KEs) are a cost-effective way to update the factual content of large language models (LLMs), but they pose a dual-use risk. While KEs are beneficial for updating outdated or incorrect information, they can be exploited maliciously to implant misinformation or bias. In order to defend against these types of malicious manipulation, we need robust techniques that can reliably detect, interpret, and mitigate malicious edits. To that end, we introduce the tasks of tracing and reversing edits. We propose a novel method to infer the edited object entity, solely based on the modified weights, without access to the editing prompt or any other semantically similar prompts, with up to 99\% accuracy. Further, we propose an effective and training-free method for reversing edits. Our method reverses up to 94\% of the edits, and helps regain the original model's output distribution without access to any information about the edit. This method can further be repurposed to distinguish between edited and unedited weights. Our findings highlight the feasibility of tracing and reversing edits based on the edited weights, opening a new research direction for safeguarding LLMs against adversarial manipulations.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes countermeasures against the potential misuse of model editing, namely **tracing and reversing edits**. The paper proposes using fixed inputs to guide the edited model in producing the target output. To approximate the update matrix, it employs rank-one approximations based on the highest singular values from a Singular Value Decomposition (SVD) of the edited matrix.

### Strengths
* The paper presents a clear motivation and addresses an important research problem.
* It proposes corresponding solutions for both **tracing** and **reversing** edits.
* Extensive experiments are conducted to validate the effectiveness of the proposed methods.

### Weaknesses
* The writing of the paper needs improvement.
* The proposed **tracing and reversing edits** methods are only validated on the ROME series of approaches, and their effectiveness on other model editing methods remains unknown, which limits the applicability of the proposed approach.
* The paper only considers **rank-one updates** for single pieces of knowledge, without addressing multi-knowledge or sequential knowledge updates.
* The results of **ANALYSIS OF RANK-ONE APPROXIMATIONS** and **REVERSAL** appear to be uncorrelated and may even contradict each other.

### Questions
* Lines 130–131: Why can’t the *unedited model* be the *original model*?
* The results in Figure 4 show that the **maximum cosine similarity** differs significantly between GPT-J and Llama, yet their **reversal accuracy** is similar. How can this be explained?
* In Figure 4, when *k = 1*, GPT2-XL achieves the highest **maximum cosine similarity** but the lowest **reversal accuracy**. How can this discrepancy be interpreted?
* Based on the results in Sections 6.2 and 6.3, there seems to be no clear correlation between **rank-one approximation** and **reversal edits**. How do the authors explain this? Figure 4 indicates that smaller *k* values make the rank-one approximation closer to the update matrix, while Table 2 shows that larger *k* values yield better **reversal and editing accuracy**, and Table 4 shows that larger *k* values make the reversed model closer to the original model. These findings appear counterintuitive, as one would expect that a closer rank-one approximation to the update matrix would lead to the reversed model being closer to the original model. How do the authors reconcile this?
* Can the results for **Qwen** in Figure 4 be shown?
* Do the conclusions in lines 309–313 imply that the assumption at the beginning of Section 6.2 does not hold?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces the tasks of tracing and reversing malicious rank-one knowledge edits, relying on the modified model weights without access to the editing prompt or original weights. For tracing, the authors propose a novel method that trains the unedited weights with a fixed random input to decode the edited MLP projection matrix and accurately infer the edited object. For reversing, they introduce an efficient, training-free method using bottom-rank approximations to neutralize the edit and recover the model's original output distribution. The methods achieved high accuracy across various LLMs and showed generalization, highlighting the feasibility of developing robust countermeasures against adversarial knowledge manipulation.

### Strengths
- The proposed methods for both tracing and reversing are designed to operate solely on the edited weights, without requiring access to the editing prompt, unedited weights, or any other information about the edit. This makes the countermeasures more practical for real-world defense against malicious editing.
- The tracing method achieved high accuracy in identifying the edited object and showed strong generalization to out-of-distribution data and different editing methods (ROME and r-ROME). Similarly, the reversal method recovers up to 93% of edits and significantly restores the original model's output distribution.
- The edit reversal technique, based on bottom-rank approximations, is training-free, making it highly efficient. This same technique can be repurposed to distinguish between edited and unedited weights by observing the number of unique predictions on unrelated text, offering a robust detection mechanism.

### Weaknesses
- The effectiveness of the reversal (and analysis of rank-one approximations) is shown to be model-dependent. For example, the optimal rank k for bottom-rank approximation varies significantly across models (e.g., k=11 for GPT2-XL vs. k=15 for llama3 to achieve highest reversal accuracy), and the similarity of the top rank-one approximation to the update matrix is much lower for LLAMA3 than for GPT models. This suggests a need for model-specific tuning of the reversal hyperparameter. 
- The core of the methods, especially the reversal technique, relies on the assumption that the malicious edit is a rank-one update (like ROME or r-ROME). The effectiveness for different types of model edits is not explored. If the proposed method only works on rank-one update methods, the usability is limited since there are many other types of model editing methods like memory-based and constrained-tuning-based methods.
- While the reversal accuracy is high (up to 93%). A qualitative analysis showed that even when reversed, the outputs, while semantically similar, are sometimes not identical to the original model's unedited outputs. Also, the decrease in editing accuracy is higher than the increase in reversal accuracy, indicating the method is better at removing the edit than fully recovering the original output distribution.

### Questions
- How does the required model-specific tuning of the optimal rank k for bottom-rank approximation impact the practical deployability of the edit reversal method?
- How can the reversal method be made less model-dependent?
- To what extent does the non-identical nature of the model's reversed outputs (despite being semantically similar) and the accuracy imbalance between edit removal and original output recovery limit the utility of the reversal method?

### Soundness
2

### Presentation
3

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
The paper proposes methods to detect and undo malicious or unintended knowledge edits. These edits modify a model’s internal parameters to change factual outputs, which can be useful for updating information but also pose security risks. The authors introduce two tasks: tracing edits and reversing edits. Their tracing method infers the edited object solely from altered weights, achieving high accuracy. Their reversal method uses bottom-rank approximations from singular value decomposition to remove edits without retraining. The study demonstrates strong generalization across 2 datasets and 4 models, showing that both tracing and reversal are feasible using only model weights.

### Strengths
- The paper introduces a training-free framework for detecting and reversing malicious edits directly from model parameters, a new defense direction for LLM safety.
- Experimental results show high accuracy and generalization across different models and datasets, suggesting good robustness.
- The methods are computationally efficient and require no access to original weights or editing prompts, enhancing practical applicability for security auditing.

### Weaknesses
- The study focuses only on rank-one edits, limiting applicability to other editing methods and scenarios like MEMIT, MEND, SERAC.
- The motivation
The evaluation scope is restricted to controlled datasets and synthetic edits, leaving real-world validation uncertain.
- The interpretability of why bottom-rank approximations work well for reversal is not fully explored, reducing theoretical clarity of the mechanism.
- The motivation for reversing edits is questionable, since model editing is primarily designed to update or add new knowledge rather than to be undone; thus, the practical need for edit reversal appears limited. The paper does not compare its reversal method against simply reapplying the inverse or original edit, which would be a more direct and intuitive baseline for restoring model behavior

### Questions
- Can you clarify whether the proposed tracing and reversal techniques would still work, or how they might adapt, when applied to higher-rank or alternative editing methods such as MEMIT, MEND, or SERAC?
- Have you considered testing their approach on real or naturally occurring edits (beyond controlled datasets) to assess its reliability in more practical or adversarial settings?
- Can you provide a clearer explanation or empirical evidence for why bottom-rank components capture “pre-edit” information and effectively reverse edits?
- What is the practical advantage of using the proposed reversal method over simply reapplying the inverse or original edit, especially given that model editing is typically meant to add or update knowledge rather than undo it?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies how to trace and reverse malicious edits in large language models that were modified using rank-one editing methods like ROME or r-ROME. It introduces two complementary defenses: (1) Tracing, which identifies the specific edited fact (object) directly from the edited weight matrix without needing the original weights or prompts, and (2) Reversing, which removes the malicious change by replacing the edited matrix with its bottom-rank singular value decomposition (SVD) approximation, effectively removing the edit signal concentrated in the top singular modes. Experiments across multiple LLMs show that tracing achieves near-perfect accuracy and reversal restores the model’s original behavior with high fidelity.

### Strengths
+ The proposed defense is practical and lightweight, requiring only access to the edited weights and no training data or edit prompts, making it suitable for real-world forensic use.
+ The reversal approach is simple yet effective, using an interpretable SVD-based method that efficiently removes the edit signal while maintaining model integrity.
+ The experimental validation is comprehensive and convincing, demonstrating strong performance across multiple models and datasets with clear quantitative results and ablation studies.

### Weaknesses
- The method’s generality is limited, as it is evaluated only on single-layer, rank-one edits and may not extend to more complex, multi-layer, or non-rank-one scenarios.
- The evaluation scope is narrow, focusing mainly on object recovery and KL divergence without exploring broader behavioral or capability effects after reversal.

### Questions
See the weaknesses.

### Soundness
2

### Presentation
3

### Contribution
3

# Align Once, Benefit Multilingually: Enforcing Multilingual Consistency for LLM Safety Alignment

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
The widespread deployment of large language models (LLMs) across linguistic communities necessitates reliable multilingual safety alignment. However, recent efforts to extend alignment to other languages often require substantial resources, either through large-scale, high-quality supervision in the target language or through pairwise alignment with high-resource languages, which limits scalability.
In this work, we propose a resource-efficient method for improving multilingual safety alignment. 
We introduce a plug-and-play Multi-Lingual Consistency (MLC) loss that can be integrated into existing monolingual alignment pipelines. 
By improving collinearity between multilingual representation vectors, our method encourages directional consistency at the multilingual semantic level in a single update. This allows simultaneous alignment across multiple languages using only multilingual prompt variants without requiring additional response-level supervision in low-resource languages. We validate the proposed method across different model architectures and alignment paradigms, and demonstrate its effectiveness in enhancing multilingual safety with limited impact on general model utility. Further evaluation across languages and tasks indicates improved cross-lingual generalization, suggesting the proposed approach as a practical solution for multilingual consistency alignment under limited supervision.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes a plug-and-play Multilingual Consistency (MLC) loss to make safety alignment transfer across languages without collecting response-level labels in every language. During alignment, the trainer feeds translated versions of the same prompt through the model, builds a small matrix of their internal representations and adds an auxiliary loss that pushes those representations to be rank-1 and that pushes their hidden representations toward dominant singular direction. The intuition is if the model interprets what safe looks like similarly in all languages at the hidden-state level, then a single-language alignment, usually English will lift safety everywhere. The loss is implemented via a temperature-softmax over the singular values of the representation matrix, encouraging the top singular value to dominate. The goal is to close the gap in safety where multilingual safety often lags far behind English.

### Strengths
- The rank-1 and softmax-over-singular-values objective is a clean way to apply make language variants point the same way and can be easily integrated into standard pipelines. It is differentiable, easy to bolt on, and doesn't interfere with the main alignment loss. 

- No need for response-level labels in low resource languages, prompt translations are sufficient.

- The method generalizes to unseen languages in MultiJail, supporting the claim that it regularizes a language-agnostic safety direction rather than overfitting to particular languages.

### Weaknesses
- The paper does not clearly surface which layer the representations are extracted from or how the extractor is designed. Also, the main text does not show a quick sensitivity table or plot to convey how these choices affect results.

- The connections between the spectral objective, its softmax relaxation over singular values and the final training loss are not well traced in the main part and that makes it hard for readers to follow the derivation without repeatedly jumping to the appendix.

- The utility evaluation is summarized briefly and the paper does not give practical guidance on how to tune the auxiliary-loss weight or temperature to manage trade-offs between safety gains and general multilingual capability.

### Questions
- Can you give reference to the MMMLU-lite dataset? What is that dataset? 

- How should practitioners tune the auxiliary-loss weight and temperature to balance safety gains against utility?

- What happens when translations are imperfect such as noisy or partially wrong? Can you weight pairs by MT quality or be robust to mismatches?

- Does the layer choice interact with backbone architecture?

### Soundness
3

### Presentation
2

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
The paper introduces a plug-and-play Multilingual Consistency loss to improve alignment across language representation and to transfer safety capabilities of LLMs from high-resource to low-resource languages. This auxiliary loss promotes consistency across languages by pushing the model to produce similar internal activations of queries written in different languages. The loss can also be integrated into various safety alignment paradigms, such as SFT and DPO. The results demonstrate improved safety performance across languages, notably closing the gap between high- and low-resource languages and largely preserves the general multilingual capability of the model.

### Strengths
1. The objective is intuitive, effective and does not rely on any anchor languages.
2. The auxiliary loss objective can be generalized and integrated to any post-training safety paradigms.
3. The approach improves substantially safety performance of low-resource languages, while retaining that of high-resource languages.

### Weaknesses
1. The approach is potentially sensitive to hyperparameters such as layer selection. The best layer where representation alignment is most effective also seems task sepcific.
2. Scaling behavior of the objective is not tested beyond 7B. Divergence across languages may be beneficial for even larger models, where the consistency objective may not be effective.

### Questions
1. Would the method be effective too for larger models, e.g., Qwen 14B / 32B?
2. Beyond safety applications, can the method be used to reduce low- and high-resource language gaps exist for other multilingual capabilities? 
3. Could the method be harmful when handling culturally sensitive tasks? What about the possibility of altering cultural-specific knowledge in the model, which might be encoded in language-specific representation?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper proposes a resource-efficient method to improve multilingual safety alignment in LLMs. The core contribution is a Multi-Lingual Consistency (MLC) loss, which enforces representational collinearity among semantically equivalent prompts across languages. The method aligns multilingual internal representations toward a shared semantic direction, improving safety consistency without requiring response-level supervision in low-resource languages.

Experiments on Qwen and Gemma models demonstrate substantial improvements in safety rates, especially for low-resource languages, while maintaining general capabilities. The approach is presented as plug-and-play, compatible with existing alignment paradigms such as DPO and SFT, and efficient in data usage.

### Strengths
- Clear motivation: Addresses a real and underexplored challenge: multilingual imbalance in LLM safety alignment.

- Conceptual simplicity: The MLC loss is an elegant addition that can integrate easily with existing pipelines.

- Empirical breadth: Includes multiple backbones (Qwen, Gemma), alignment paradigms (DPO, SFT, SimPO, ORPO), and both in- and out-of-distribution tests.

- Data efficiency: Claims strong multilingual gains with minimal additional data (∼1.8M tokens vs. 15M+ for comparable baselines).

- Consistency analyses: Representation-space visualizations (Gram matrices) and PAG metrics provide insightful evidence of improved cross-lingual alignment.

### Weaknesses
- Incremental contribution: The MLC loss is essentially a regularization of multilingual representations, conceptually simple and not a fundamentally new paradigm.

- Theoretical shallowness:  Despite heavy mathematical framing (singular value decomposition, spectral view), the theoretical section adds little genuine insight beyond enforcing collinearity.

-  Experimental bias: Evaluations rely on safety datasets constructed in English, potentially conflating multilingual improvement with translation artifacts rather than genuine alignment.

### Questions
Please see the weaknesses part.

### Soundness
2

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
This paper tackles the challenge of multilingual safety alignment in large language models (ensuring models refuse harmful prompts consistently across languages). The authors propose a Multilingual Consistency (MLC) loss that complements existing post-training methods such as SFT or DPO. The loss encourages shared multilingual representations by promoting collinearity across query embeddings in different languages. Specifically, for each multilingual query set, the hidden representation of the last token in each language is linearly projected, normalized, and stacked into a matrix. The method minimizes the distance of this matrix from its best rank-1 approximation, derived via its top singular value, effectively enforcing a shared semantic direction.

The approach does not require multilingual responses, only crosslingual prompts, making it resource-efficient. Experiments on two safety benchmarks (PKU-SafeRLHF and MultiJail) across ten languages show substantial improvements in multilingual safety consistency, especially for low-resource languages, while maintaining general capabilities on MMLU.

### Strengths
- **Important problem:** The paper addresses the problem of ensuring safe and consistent behavior across languages in LLMs.
- **Conceptually elegant and technically sound:** The proposed spectral regularization via rank-1 optimization is simple yet well motivated and theoretically grounded.
- **Strong empirical results:** Comprehensive evaluations across datasets, languages, and base alignment paradigms demonstrate consistent gains, especially for low-resource settings.
- **Practical and efficient:** The method is plug-and-play, adds minimal computational cost, and does not require multilingual response data.

### Weaknesses
- **Weak related work discussion:** The discussion of multilingual alignment baselines (e.g., MPO, SDRRL) is both incomplete and difficult to follow. The main paper only names them without explanation, forcing readers to consult the appendix, which is itself hard to follow. As a result, it is difficult to understand how these baselines differ conceptually or why they are appropriate points of comparison.
- **Limited baselines:** The paper lacks an upper-bound comparison, e.g., training with fully translated safety data across languages, to contextualize achievable performance ceilings.
- **Evaluation of general capabilities is narrow:** The use of MMLU alone (amultiple-choice benchmark) provides a limited view of cross-lingual reasoning and generation quality. More generative evaluations could clarify whether safety alignment affects multilingual fluency or reasoning.
- **Missing ablation studies:** The paper would benefit from ablations isolating the contribution of (i) the linear projection, (ii) the choice of singular-value-based regularization versus alternatives such as cosine similarity, and (iii) the temperature parameter τ.

### Questions
- **Linear projection:** Is the linear extractor $W$ trained jointly with the model? Please clarify in Section 3.2.
- **Baselines:** Briefly summarize how MPO and SDRRL operate in the main paper, this would make the comparison more self-contained.
- **Ablations:** Could you provide results using alternative similarity measures (e.g., cosine loss) or removing the linear projection to test sensitivity?
- **Capability evaluation:** MMLU is a multiple-choice benchmark and thus does not assess generation abilities or language control. I suggest adding a CoT-style evaluation where the model must generate reasoning in the target language. This would allow you to measure both accuracy and linguistic consistency (e.g., avoiding language mixing), providing a more complete picture of whether the proposed training preserves generative behavior across languages.

### Soundness
3

### Presentation
3

### Contribution
3

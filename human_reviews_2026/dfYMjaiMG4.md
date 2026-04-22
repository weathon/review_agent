# Identifying and Evaluating Inactive Heads in Pretrained LLMs

- Avg Score: 4.80
- Decision: Accept (Poster)
- Scores: 4, 2, 6, 4, 8

## Abstract
Attention is foundational to large language models (LLMs), enabling different heads to have diverse focus on relevant input tokens. However, learned behaviors like attention sinks, where the first token receives the most attention despite limited semantic importance, suggest some heads may be inactive, and point to a significant source of computational redundancy. To analyze this phenomenon, we evaluate 12 score functions that measure different ways a head can be inactive. Thresholding these scores allows us to analyze different sets of potentially inactive attention heads. We evaluate whether identified heads are inactive through model interventions, finding that more than 12% of attention heads are inactive on average, and can be ablated in specific contexts while maintaining MMLU accuracy to within 1% of the pretrained LLM. Across 3 model families, our score functions that measure the average norm of a head's output consistently identify inactive heads that would not have been found by score functions that rely solely on attention weights. We establish that relying on a score function that measures a first token attention sink would underestimate the prevalence of inactive heads, failing to identify more than 7\% of inactive heads on average. We also show how measuring score distributions can provide insights into attention behavior. For instance, we find evidence that finetuning causes little to no change in attention behavior, and that even within the same model family, large model scales present different attention behaviors.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a comprehensive perspective for detecting inactive heads in Transformers and identifies Average Head Output Norm (AHON) as the most effective indicator, thereby enhancing our understanding of inactive attention heads.

### Strengths
1.	The paper is well organized and easy to follow.
2.	The conclusion is clear and direct, effectively demonstrating that the Average Head Output Norm (AHON) serves as a reliable indicator for identifying inactive heads.

### Weaknesses
1.	While the paper arrives at a clear and well-supported conclusion, it lacks further discussion on how the proposed detection method could be effectively utilized in practice. This limitation raises questions about the extent of the paper’s contribution beyond.
2.	The experiments are conducted on a limited set of datasets, which may restrict the generalizability of the findings to broader domains or tasks.
3.	Although the proposed set of metrics is comprehensive, the paper lacks deeper analysis and interpretation of the observed results. The integration of these indicators might not yet be sufficiently thorough, and the underlying causal explanations for the findings remain somewhat unclear.

### Questions
1.	Could the authors further clarify the significance of dynamically detecting inactive heads?
Based on my understanding, this dynamic pruning is not directly applicable for inference efficiency, since the pruning is not static.
If the main goal of the paper is to enhance our understanding of inactive attention heads in Transformers, what is the broader significance of detecting them dynamically? What insights or implications can we gain from this beyond efficiency?
2. I am also curious whether the set of dynamically inactive heads changes significantly with different levels of context or question difficulty？
3. Could the authors comment on whether the presence of “inactive” heads should be viewed as a normal characteristic of the model rather than a deficiency to be mitigated? For instance, in simpler contexts, certain heads might naturally remain unused, whereas in more complex inputs, more heads could become activated. Do the authors expect such adaptive activation behavior, and would they consider it a potentially desirable property?

### Soundness
2

### Presentation
3

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
The paper aims to reform the existing attention-sink-based characterization of inactive heads by focusing on the distribution of attention weights in LLMs. It introduces a taxonomy of 13 score functions to identify inactive heads, based on either the attention weight distribution or the norm of an attention head’s value vector or output at different positions, under various normalization strategies. Through ablation-based studies across 14 models, it shows that up to 12% of heads can be ablated without accuracy loss when the proper score metric is chosen, and that output-norm-based metrics outperform traditional attention-weight heuristics in many cases. The study also finds that fine-tuning strategies and moderate scaling barely affect attention behavior, highlighting strong structural invariance across models within the same family.

### Strengths
1. This paper represents a meaningful step toward moving beyond the current landscape of inactive head identification and attention pruning, which is predominantly attention-sink-based and focuses heavily on surface-level attention weight characterization. The score metrics proposed in the paper—based on the norms of different aspects of attention head outputs across token positions—are indeed novel. They open the possibility for a new paradigm in this subfield that diverges from the traditional attention-sink-centered narrative and instead evaluates inactivity through more direct and mechanistically grounded measures of an attention head’s functional output.

2. The discussion of intercorrelations among inactive heads defined under different criteria across model sizes, families, and training stages (SFT/RLHF/DPO) is both novel and compelling. This comparative perspective is the paper’s key advantage over prior studies, which typically characterize head inactivity within each model in isolation. By analyzing how these relationships generalize across architectures and alignment methods, the paper contributes a broader, more systematic understanding of attention head behavior.

### Weaknesses
1. Despite its novelty, the proposed taxonomoy of inactivity criteria seem highly ad-hoc and disorganized, due to following reasons:

   **Problematic definition**: the author uses a rather unorthodox definition of the attention head output, i.e. as the convex combination of value vectors weighted by attention coefficients prior to multiplication with the output projection matrix. This would result in a $d_v$ (dimension of value embedding) dimensional vector, which diverges from the more widely-accepted and -used circuit-based definition in [1] which multiplies the $d_v$ dimensional vector futher with the corresponding $R^{d_v \times d}$ dimensional slice of the projection matrix.

   **Significant conceptual overlap between criteria**: As a direct consequence of the problematic definition, there is significant conceptual overlap and redundancy between the value-vector-based criteria and head-output-based criteria, since the norm head output at a token position $i$ could be trivially bounded by the maximum (and minimum) norm of value vectors from position 1 to $i$. Thus it is very likely that a head with low average value vector norm will have low average head output norm/last token head output norm and vice versa. Indeed the paper's observation of a 0.55 IOU value between average head output norm and average value vector norm demonstrates this redundancy.
  
   **No justification for the inclusion and exclusion standards**: The authors made no justifications for why they include/exlucde certain criteria in their taxonomy. For instance, since they included the last token head output norm, they could have easily included the last token value vector norm as well. But instead they only included the first token value vector norm, without explaining the rationale.

   **Disorder related to normalization strategies**: The authors included layer-normalized version of their proposed 6 criteria into the taxonomy, without explaining the reason for this. They further included a head-normalized version specifically for the last token head output norm, but not for the other metrics.

2. The paper spent little efforts in analyzing the properties of the identified inactive heads, which could include the heads distribution across layers, their activation/attention weights patterns for different kinds of inputs, etc. Moreover, the paper only discussed the accuracy consequences of the ablation of the identified heads, without any qualitative analysis of how the ablation affects model outputs, or any fine-grained quantitative analysis of how the ablation affects the model's full forward computation process (hidden states etc). These shortcomings limit the mechanistic depth of the paper's analysis.

3. The writing of certain parts of the paper appears perplexing. For instance, lines 400-403 appear completely irrelevant to the context. There are also grammatical issues here and there. For instance line 420: "Every model and score function combination produce (should be produces)..."

[1] Elhage, Nelson, et al. "A mathematical framework for transformer circuits."

### Questions
1, In the first row of Table 2, why is there a difference percentage between AWFT and your method (which is also AWFT)? The same applies to the case of Qwen2.5-1.5B and Qwen 2.5-14B as well.

2, Line 407-408: "Each of 13 score functions use a different set of 7 thresholds, which may not identify 30% of model heads (the maximum we consider)". I am unsure of what this means. Don't you ablate the same percentage of heads (whether it is 30% or not) identified using different criteria?

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
This paper investigates inactive attention heads in pre-trained LLMs by proposing a taxonomy of 13 score functions that measure attention weights, value vectors, and head outputs. The authors verify head inactivity through model interventions, zeroing out identified heads and measuring MMLU accuracy across 14 models from 3 families. Their main finding is that more than 12% of attention heads can be zeroed while maintaining accuracy. These results are significantly better than prior methods. The paper also analyzes score distributions to show that finetuning causes minimal changes to attention behavior and that model scaling has little effect until very large scales.

### Strengths
1. Clarity and readability
The writing style of this paper significantly enhances the quality of the work.

2. Depth
The paper discusses a wide-range of scores, to identify and evaluate inactive attention heads in a robust manner.

3. Novelty
The paper shares a new surprising finding about LLMs that challanges prior work.

3. Rigorous Experimental Design
The paper tests across 14 models from 3 families (Llama, OLMo, Qwen), it uses model interventions to verify inactive heads rather than just classification.

### Weaknesses
1. Single Benchmark Evaluation
The current evaluation is limited to MMLU, but inactive heads might become active in very specific context that are not triggered in MMLU. 

2. Lack of Practical Implications
The fact that 12% of heads are inactive is astounding, however, it is unclear what kind of actionable steps can be taken to improve modern LLMs. Does this insight imply a different architecture? Could we remove heads in deployment and achieve better performance?

### Questions
1. Do you have a speculation on Why the model does not leverage 12% of its heads?
2. Is it clear how to design better architecture to take advantage of that?
3. Can we prune these heads in production system to improve latency and cost?
4. Do you have any hypothesis on why these head emerge?
5. Can we change the training data or algorithm to incentivize the model to use all attention heads?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates the phenomenon of "inactive" attention heads in large language models (LLMs). The authors argue that prior work, which defined inactivity solely through attention patterns (like "attention sinks" on the first token), provides an incomplete picture. They propose that a head can be inactive in multiple ways, not just through its attention weights, but also through its value vectors or its final output vectors.

### Strengths
The paper moves beyond a single definition of inactivity to propose a multi-faceted taxonomy. The 13 score functions provide a much richer analysis.

Testing across 14 models from 3 different families provides robust, generalizable results that are not specific to a single architecture or training corpus.

The analysis of score distributions to study the effects of fine-tuning and model scaling provides valuable insights into model behavior.

### Weaknesses
**Limited Task Evaluation**: The primary evaluation is on MMLU, a multiple-choice knowledge-based benchmark. The generality of the findings to other tasks (e.g., code generation, long-context reasoning, creative writing) is not verified. A head inactive on MMLU might be active on a different task.

**Static, Single-Pass Analysis**: The analysis is performed on a single forward pass. The role of a head might be context-dependent or dynamically change across layers or tokens in a way that this static analysis misses. The paper acknowledges this by noting that inactivity is verified "in specific contexts."

**Threshold Reliance**: The method relies on choosing thresholds, which, while systematically swept, is still a heuristic. The "inactive" label is not absolute but relative to a chosen threshold.

### Questions
Did the authors observe any cases where a head identified as inactive on MMLU became critical for performance on a different type of task? How task-specific do you believe the phenomenon of head inactivity is?

The analysis provides a "snapshot" of head activity. Did the authors investigate if the inactivity of a head is static, or can a head dynamically switch between active and inactive states depending on the specific input token or the evolving context within a sequence?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper studies the phenomenon of "inactive heads" in pretrained LLMs. To comprehensively identify inactive heads, the authors propose 13 score functions related to attention heads for evaluating the importance of heads and assess their effectiveness in identifying inactive ones. Extensive experiments on MMLU and pretraining corpora show that the proposed "Average Head Output Norm" metric performs best in identifying inactive heads. It can remove as many attention heads as possible while maintaining the original performance, and it detects more inactive heads than the traditional method based on the first-token attention sink. The paper also uses these metrics to briefly analyze attention behaviors across models of different sizes and training settings, showing that finetuning hardly changes attention behavior, while larger models exhibit distinct attention patterns.

### Strengths
1. The proposed scoring functions comprehensively covers various aspects of the attention layers, and are extensively evaluated on various LLMs across different model families, showing strong generality.

2. The proposed AHON metric effectively identifies a more complete range of inactive heads and provides useful guidance for studying inactive heads.

3. The paper is clearly written, includes complete experimental results, and is easy to follow.

### Weaknesses
1. The main contribution of the paper is limited to identifying inactive heads in LLMs. Beyond that, the discussion on the mechanisms and behaviors of inactive heads is quite restricted. I would like to see why AHON can better identify inactive heads from an internal, model-based behavior perspective.

2. In the experiment section, the paper only uses the pretraining dataset and the 5-shot MMLU dataset. As MMLU is indeed commonly used to evaluate general model capability, using only one dataset is rather limited. Moreover, the 5-shot setting may interfere with the evaluation of inactive heads related to in-context learning.

### Questions
1. In Section 4.2, the paper states that different score functions identify different groups of inactive attention heads, with low overlap among them. Could the authors explain what kinds of attention heads are identified by each metric? Do these metrics capture different aspects of the model's behavior?

2. Head scores computed by functions in Table 1 require computation to obtain during the actual inference. I am curious whether the model's static parameters might already contain some clues about inactive heads (e.g., the value projection matrix of each head, or the corresponding output projection matrix).

3. In Section 4.4, Line 453, the authors suggest that larger models learn to specialize their heads in different ways. However, the score distribution change only appears from 7B to 14B, and there is no larger model to support this claim. I hope the authors can validate this hypothesis on larger models (e.g., 32B).

4. Do the findings in this paper have some pratical contributions, such as guiding model performance improvement? I know that this work focuses on interpretability, but it would be better if it also had potential for practical guidance.

5. Minor formatting and typo issues: several elements exceed the page boundary (e.g., Table 1, Figures 6–7, and the example text in Section A.6); the opening quotation mark in Line 148 is incorrect.

### Soundness
3

### Presentation
4

### Contribution
3

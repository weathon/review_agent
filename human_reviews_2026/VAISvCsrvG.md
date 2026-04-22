# Rethinking LLM-as-a-Judge: Representation-as-a-Judge with Small Language Models via Semantic Capacity Asymmetry

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 4

## Abstract
Large language models (LLMs) are widely used as reference-free evaluators via prompting, but this “LLM-as-a-Judge” paradigm is costly, opaque, and sensitive to prompt design. In this work, we investigate whether smaller models can serve as efficient evaluators by leveraging internal representations instead of surface generation. We uncover a consistent empirical pattern: small LMs, despite with weak generative ability, encode rich evaluative signals in their hidden states. This motivates us to propose the Semantic Capacity Asymmetry Hypothesis: evaluation requires significantly less semantic capacity than generation and can be grounded in intermediate representations, suggesting that evaluation does not necessarily need to rely on large-scale generative models but can instead leverage latent features from smaller ones. Our findings motivate a paradigm shift from LLM-as-a-Judge to Representation-as-a-Judge, a decoding-free evaluation strategy that probes internal model structure rather than relying on prompted output. We instantiate this paradigm through INSPECTOR, a probing-based framework that predicts aspect-level evaluation scores from small model representations. Experiments on reasoning benchmarks (GSM8K, MATH, GPQA) show that INSPECTOR substantially outperforms prompting-based small LMs and closely approximates full LLM judges, while offering a more efficient, reliable, and interpretable alternative for scalable evaluation. The code and data are available at: https://github.com/zhuochunli/Representation-as-a-judge

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper challenges the conventional LLM-as-a-Judge paradigm by showing that small language models (SLMs), despite weak generative ability, encode rich evaluative signals in their hidden representations. It proposes the Semantic Capacity Asymmetry Hypothesis, suggesting that evaluation requires less semantic capacity than generation. The authors introduce INSPECTOR, a probing-based framework that predicts aspect-level evaluation scores using internal representations of SLMs. Experiments on reasoning datasets (GSM8K, MATH, GPQA) demonstrate that INSPECTOR approaches the performance of large LLM judges while being more efficient and interpretable. This work advances a Representation-as-a-Judge paradigm that enables cost-effective, scalable, and transparent evaluation of model outputs.

### Strengths
1. The paper introduces the Representation-as-a-Judge framework which is a new way of performing evaluation without decoding, shifting the paradigm from generation-based judgment to representation-based probing.
2. The paper demonstrates consistent and large gains over prompt-based baselines across multiple reasoning datasets, showing that evaluative signals exist in intermediate representations.
3. The proposed approach significantly reduces inference cost, avoids dependence on proprietary LLMs (like GPT-4), and provides interpretable layer-level insights.
4. Uses binary probing classifiers to filter low-quality data effectively, improving downstream supervised fine-tuning.

### Weaknesses
1. Despite promoting efficiency, the pipeline still depends on large LLMs (like DeepSeek-V3) to provide "gold" evaluation scores. Gathering these scores is itself expensive.
2. Experiments focus only on mathematical reasoning (GSM8K, MATH, GPQA); it’s unclear whether the approach generalizes to open-domain, creative, dialogue tasks or safety evaluation.
3. While simple probes (logistic regression) are interpretable, they may underfit complex evaluative signals, limiting the upper bound of achievable fidelity. 
4. The paper lacks qualitative analysis and error analysis. Observing some samples where these models perform much better than baselines, or looking at error buckets would be nice.

### Questions
1. Could multi-task or cross-aspect probing improve performance compared to aspect-specific classifiers?
2. How stable are the layer rankings across different datasets — do the same layers consistently encode evaluative information?
3. Can this method be extended to explain why a response was scored poorly (e.g., highlighting reasoning errors)?
4. Could the representation-based judge itself be used as a reward model in reinforcement learning from human feedback (RLHF)?

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
3

### Summary
This paper proposes a new evaluation paradigm called Representation-as-a-Judge, which shifts from relying on large language model (LLM) generation to using internal representations from small LMs (SLMs) for evaluation. The authors introduce the Semantic Capacity Asymmetry Hypothesis, arguing that evaluation requires less semantic capacity than generation. They instantiate this idea in INSPECTOR, a probing-based framework that trains lightweight classifiers on SLM hidden states to predict quality scores (e.g., logicality, factuality) derived from a strong LLM judge. Experiments on GSM8K, MATH, and GPQA show that SLM probes outperform prompting baselines and closely approximate full LLM evaluations, especially in binary classification tasks. The approach also proves useful for filtering training data to improve downstream supervised fine-tuning.

### Strengths
1. The Representation-as-a-Judge framework is technically sound, which could be an alternative to the prevalent “LLM-as-a-Judge” approach.

2. The approach enables efficient evaluation using smaller, open-source models instead of proprietary LLMs.

3. The empirical results are strong, showing significant gains over some prompting and fine-tuning baselines.

### Weaknesses
1. A limitation is that the method still requires a powerful LLM in the loop to obtain initial evaluation scores (for training data). This paper assumes the LLM’s scores are gold-standard. It would strengthen the work to either validate against human ratings or discuss the implications of this dependency.

2. The probing classifiers achieve relatively low accuracy on fine-grained multiclass (1–5) predictions. This might limit the method’s use if one requires precise scoring, and also indicates some inherent issue that the approach doesn’t fully match the large model in terms of detailed gradations of quality. 

3. The INSPECTOR pipeline introduces additional complexity and tuning that a direct LLM judge does not require. One must decide how to pool representations, which layers to select, what classifier to use, etc. This paper doesn’t deeply explore the sensitivity to these choices.

### Questions
1. Have the authors evaluated or considered how the probe’s judgments (or the large LLM’s scores it learns from) align with human evaluations of the responses?

2. The experiments focus on mathematical reasoning problems. How well do the authors expect the Representation-as-a-Judge approach to transfer to other domains or tasks?

3. Did the authors experiment with regression instead of classification to predict aspect scores on a continuous scale?

### Soundness
3

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
4

### Summary
This paper proposes INSPECTOR, a probing-based framework that evaluates model outputs using internal representations instead of text generation. It introduces the Semantic Capacity Asymmetry Hypothesis, suggesting that evaluation requires less semantic capacity than generation. By leveraging small open-source LMs, INSPECTOR enables lightweight, interpretable, and scalable evaluation, achieving performance comparable to large proprietary models.

### Strengths
1. The paper reframes model evaluation as Representation-as-a-Judge, shifting from prompt-based evaluation to representation-based probing.
2. By leveraging internal representations from small open-source LMs instead of large proprietary models, INSPECTOR offers a lightweight and interpretable alternative that significantly reduces computational cost.
3. The framework achieves high predictive accuracy on reasoning benchmarks (e.g., GSM8K, MATH, GPQA), demonstrating that small models (1.7B) can approximate large-scale evaluators.
4. The proposed method can enhance LLM evaluation pipelines, reduce dependence on expensive closed models, and support data filtering for downstream tasks.

### Weaknesses
1. Experiments are primarily focused on reasoning benchmarks; it is unclear whether the method generalizes to other domains such as dialogue, summarization, or open-ended generation.
2. The performance quality of the judge (small LM) could be dependent on how the evaluation criteria are established.

### Questions
1. Is there a specific reason why the paper focuses only on decoder-only models instead of considering encoder-only architectures?
2. I believe the Semantic Capacity Asymmetry Hypothesis aligns with a well-established understanding in LLM research. It would be helpful to cite prior work that supports this perspective.

### Soundness
3

### Presentation
3

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
This paper observes that small LMs, although with their limited generative ability, can still provide reliable evaluations within their hidden representations. The authors then propose the Semantic Capacity Asymmetry Hypothesis, which posits that evaluation requires significantly less semantic capacity than generation and can rely on intermediate representations. The authors propose Representation-as-a-Judge, and implement this idea through INSPECTOR, a probing-based framework that predicts aspect-level evaluation scores directly from small model representations. Experimental results demonstrate that INSPECTOR substantially outperforms prompting-based small LMs and closely approximates large model judges.

### Strengths
**Novel idea for LLM-as-a-Judge.**
The concept of semantic capacity asymmetry is creative and appealing. It provides a fresh lens for understanding evaluation-relevant signals in internal representations, suggesting that weak evaluation performance in small LMs may arise from surface-level generation limitations rather than an inherent lack of semantic understanding.

**Effective probing design.**
The paper designs probing mechanisms to capture evaluation-related signals from model representations. These probes are empirically shown to be effective and help our understanding of how evaluation-relevant signals work in hidden layers.

**Practical significance.**
Building on this idea, the work demonstrates that smaller open-source models can serve as reliable evaluators, substantially reducing cost while maintaining evaluative fidelity, which is practically meaningful.

### Weaknesses
**1. Missing intuition behind the probing design.**
The paper introduces probing experiments to support the idea of semantic capacity asymmetry, but the intuition for the probing setup is underexplained. It is unclear what specific representational property the probes aim to capture or why this design shows semantic capacity.

**2.Limited dataset coverage.**
The evaluation focuses on three reasoning benchmarks (GSM8K, MATH, and GPQA), all within the mathematics and science domain. Other tasks types, such as open-ended generation, summarization, dialogue and code generation are not considered here. Incorporating a broader range of domains and task formats would strengthen the generality of the proposed idea.

**3. Clarity and presentation.** 
Several sections are dense and could be improved with additional figures or concrete examples. For instance, a figure to illustrate how linear probes are constructed.

### Questions
1. Can the authors provide a formal definition or metric for semantic capacity asymmetry beyond the conceptual description?

2. The work uses judgments from DeepSeek-V3 as gold labels for training and evaluation, but does not incorporate human annotations as ground truth. Have the authors considered comparing (a) human judgments, (b) large LLM-as-a-Judge outputs, and (c) the proposed Representation-as-a-Judge ouputs?

3. Some minor questions: 

   (1) In Eq (6), the symbol $\mathcal{C}$ is missing, and the meaning of $\mathcal{F}$ is unclear.

   (2) In the ablation study shown in Figure 4, is the classifier fixed as Logistic Regression in the left plot (where pooling methods vary), and is mean pooling fixed in the right plot (where classifier methods vary)?

### Soundness
3

### Presentation
3

### Contribution
3

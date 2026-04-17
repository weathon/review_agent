# Where Did It Go Wrong? Attributing Undesirable LLM Behaviors via Representation Gradient Tracing

- Decision: Accept (Poster)
- Scores: 2, 6, 4

## Abstract
Large Language Models (LLMs) have demonstrated remarkable capabilities, yet their deployment is frequently undermined by undesirable behaviors such as generating harmful content, factual inaccuracies, and societal biases. Diagnosing the root causes of these failures poses a critical challenge for AI safety. Existing attribution methods, particularly those based on parameter gradients, often fall short due to prohibitive noisy signals and computational complexity. In this work, we introduce a novel and efficient framework that diagnoses a range of undesirable LLM behaviors by analyzing representation and its gradients, which operates directly in the model's activation space to provide a semantically meaningful signal linking outputs to their training data. We systematically evaluate our method for tasks that include tracking harmful content, detecting backdoor poisoning, and identifying knowledge contamination. The results demonstrate that our approach not only excels at sample-level attribution but also enables fine-grained token-level analysis, precisely identifying the specific samples and phrases that causally influence model behavior. This work provides a powerful diagnostic tool to understand, audit, and ultimately mitigate the risks associated with LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces Representation Gradient Tracing (RepT), a data attribution framework designed to identify which training examples caused specific undesirable behaviors in large language models, such as harmful outputs, backdoor activation, or factual contamination. Rather than operating in parameter space, RepT works in representation space, combining (1) the hidden state of the final prompt token and (2) the gradient of the first output token to construct a compact “signature” vector for each sample. Attribution is then performed via cosine similarity between signatures. The approach also supports token-level attribution by examining gradient-aligned influence scores across individual tokens.

### Strengths
The paper addresses a meaningful and timely challenge in model auditing: tracing the origins of harmful, biased, or otherwise undesirable behaviors in large language models. The shift from parameter-space attribution to representation-space attribution is conceptually coherent and aligns with recent evidence that internal activations capture semantically organized information. The proposed method is also computationally efficient, requiring only one backward pass per sample and storing compact representation signatures, which makes it more practical than influence-function-based approaches that require either Hessian approximations or historical gradient storage. Finally, the token-level attribution capability is a useful feature.

### Weaknesses
The evaluation setup substantially limits the strength and generality of the conclusions. All experiments are performed on synthetically constructed failure cases, where a very small and clearly defined set of poisoned or altered examples is inserted into an otherwise clean dataset. In these scenarios, the test prompts that elicit undesirable behavior are structurally and semantically very close to the corrupted training samples that caused them. This makes the attribution problem relatively straightforward, because the causal samples have strong, direct representational signatures that the method can recover via similarity-based comparison. In realistic settings, however, harmful, biased, or misleading behaviors rarely originate from a single or easily identifiable training example. Instead, they tend to arise from diffuse and distributed correlations spread across many heterogeneous data sources, often contaminated by noise, paraphrasing, or domain variation. The paper does not evaluate RepT under such circumstances, so the reported performance does not provide evidence that the method can handle real-world model failures.

The consistently near-perfect attribution accuracy reported across all models and all tasks is another point of concern. When a method achieves performance close to 1.0 in every condition, it suggests that the benchmarks may be systematically aligned with the method’s core retrieval mechanism. In this case, because the corrupted training examples closely resemble the test prompts in representation space, RepT may simply be identifying semantic similarity rather than performing genuine causal attribution. The paper does not examine how RepT behaves when the harmful or incorrect examples are paraphrased, stylistically altered, contextually embedded, or obfuscated, i.e., when surface-level correspondence no longer provides a direct signal. Without such tests, it is unclear whether the method is robust to more subtle or realistic forms of data contamination.

The method’s reliance on selecting a particular “phase transition” layer introduces another source of uncertainty. Although the paper proposes a heuristic based on representation similarity across layers, this procedure is not theoretically justified and is only lightly evaluated. There is no analysis of how stable the layer selection is across different domains, prompts, task types, or probing datasets, nor whether attribution accuracy deteriorates when the probing distribution diverges from the failure domain. Given that representation geometry varies significantly across models and training regimes, the absence of a stability study leaves open the question of whether the technique generalizes beyond the specific setups demonstrated.

Finally, the comparisons to baselines are not fully convincing due to insufficient detail about implementation constraints. Methods such as TracIn, LESS, and LoGra are highly sensitive to gradient storage policies, checkpoint frequency, normalization strategies, and available memory budgets. The paper reports performance differences but does not establish that these baselines were configured with matched computational constraints or tuned to competitive settings. As a result, it is difficult to determine whether RepT’s empirical advantage reflects an inherent methodological improvement or simply differences in experimental favorability.

### Questions
How does the method perform when the source of an undesirable behavior is distributed across many training instances, rather than localized to a single or a few poisoned samples?

Can the authors evaluate the method on naturally occurring hallucinations or biased generations, where the “ground truth causal sample” is unknown and must be approximated through retrieval or human labeling?

### Soundness
1

### Presentation
3

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
The paper introduces a novel and efficient framework that diagnoses a range of undesirable LLM behaviors by analyzing representation and its gradients, which operates directly in the model’s activation space to provide a semantically meaningful signal linking outputs to their training data. Overall, I think the method is reasonable and the experimental results are promising.

### Strengths
- Reasonable method. The idea of using gradients and activations for tracing relevant training data sounds promising.
- Good results. The experimental results are good, demonstrating the advantages of the proposed method.
- Comprehensive ablation study. Ablation study shows the effectiveness of each component.

### Weaknesses
- Baseline selection. I think in some settings, the displayed baselines may be not appropriate. For example, for harmful data identification, a straight forward approach is to use safety classifiers. Additionally, for backdoor data detection, there are various baselines specific for backdoor detection. I think these baselines should be stronger than the current baselines.
- Scalability. The paper includes an interesting experiment part for knowledge contamination detection. While the experiment shows that the method works well for a small fine-tuning dataset, I think knowledge contamination is more important when applied on pretraining data. However, the current method seems hard to work on pretraining data, as it requires model inference on each training sample, whose cost is too high.

### Questions
Typo: line 237 effective->effectivenes

### Soundness
3

### Presentation
3

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
This paper introduces Representation Gradient Tracing (RepT), a novel framework that diagnoses undesirable LLM behaviors by analyzing representations and their gradients in the activation space rather than the parameter space. RepT constructs compact signature vectors by concatenating the last prompt token's hidden state with the first response token's gradient, enabling efficient sample-level and token-level attribution through cosine similarity. Experiments across harmful content identification, backdoor detection, and knowledge contamination show RepT achieves near-perfect precision (≈100% auPRC) while requiring 1000× less memory and significantly faster computation than gradient-based baselines.

### Strengths
1. I like the task and it is well motivated. I think attributing LLM behavior is quite important and this paper provides a practical and efficient method.
2. The shift from parameter space to representation space is sound and addresses the limitations of gradient-based methods (high dimensionality, noise, semantic gap). The design of signature vectors—combining H_last (contextual understanding) and g_H_first (predictive direction)—elegantly captures the causal link between inputs and outputs.

### Weaknesses
1. **Missing Important Intuitive Baselines**: The paper lacks comparison with simpler, more intuitive baseline methods. For instance, one could directly use a lightweight embedding model (e.g., the popular bge-m3) to compute semantic similarity between training samples and problematic test cases, which does not rely on gradient-based approaches at all. Since Appendix A provides insufficient implementation details for existing baselines, it remains unclear for me whether such embedding-based methods were considered or why they were excluded from evaluation.
2. **Insufficient Scalability Analysis**: The paper does not adequately address how RepT scales along two critical dimensions of LLM growth. First, as model size increases, the hidden layer cache requires proportionally more storage and computation—how does this affect feasibility for 70B+ parameter models? Second, and more critically, how does performance degrade with massive training corpora? The method's linear time/space complexity with respect to training set size (O(N)) makes it seemingly impractical for pretraining scenarios with 10T+ tokens. Moreover, attribution accuracy likely decreases as the training set grows larger and more diverse—does the method still precisely identify causal samples when N reaches millions or billions? These scaling properties are not empirically studied.
3. **Lack of Analysis for Complex Attribution Scenarios**: The evaluation assumes each bad case can be traced to a single training sample, but real-world failures are often more nuanced. First, many undesirable behaviors emerge from the **interaction of multiple training samples**—none problematic individually but harmful in combination. Can RepT detect such collective effects? Second, some failures stem from the **absence of appropriate training data** rather than the presence of bad data. In such cases, would the method force-match to irrelevant samples, producing misleading attributions? The paper does not discuss these failure modes or provide diagnostic guidance for practitioners.

### Questions
See above

### Soundness
2

### Presentation
2

### Contribution
2

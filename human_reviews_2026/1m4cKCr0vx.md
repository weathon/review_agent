# Scaling Laws for Parameter Pruning in LLMs

- Avg Score: 2.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 2, 4

## Abstract
Scaling up model parameters and training data consistently improves the performance of large language models (LLMs), but at the cost of rapidly growing memory and compute requirements, which makes deployment on resource-limited hardware infeasible. *Model pruning*, a widely used compression technique, reduces inference costs by removing redundant parameters. However, its impact on downstream performance remains unpredictable and is typically assessed only through costly empirical sweeps. To address this gap, we introduce *pruning laws* -- simple and interpretable scaling relations that connect a pruned LLM's post-pruning performance to its unpruned performance and pruning ratio. Across five LLMs (2.7B–13B parameters), three pruning strategies (unstructured, width, and depth), and eight diverse tasks, we show that pruning laws achieve strong predictive accuracy (average extrapolation error $<7$%), reliably quantify performance degradation, and identify critical pruning thresholds beyond which recovery is infeasible. Moreover, we demonstrate that the same laws transfer universally across architectures, pruning methods, and even unseen models in zero-shot and one-shot setups. These results provide both researchers and practitioners with a principled framework to select pruning strategies, estimate safe pruning ratios without exhaustive tuning, and deploy LLMs efficiently under real-world compute and latency constraints.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces pruning laws, which are analytical scaling relationships that model how a large language model’s (LLM’s) performance degrades as parameters are pruned. The authors propose a simple power-law formulation $L = L_0 P_0 (1-r)^{\alpha}$ linking post-pruning performance $L$ to base performance $L_0$ and pruning ratio $r$, with fitted parameters $\alpha$ and $P$.

### Strengths
1. The experiments are extensive, covering multiple architectures (OPT, LLaMA, Phi-3), pruning strategies, and task types. The universality and cross-method transfer analyses (zero-shot and one-shot setups) convincingly demonstrate robustness.
2. The proposed formulation enables practitioners to estimate safe pruning ratios without retraining or fine-tuning.
3. The derivation of the pruning law, its logarithmic linearization, and the OLS-based fitting procedure are clearly articulated.

### Weaknesses
1. The main concern with developing such a scaling law is that post-pruning evaluation is relatively inexpensive compared to pre-training. It is therefore unclear why a separate scaling law is necessary to model post-pruning performance, given that pruning results can typically be obtained within minutes. In addition, AutoML techniques can be employed to efficiently narrow the search space and identify optimal pruning strategies.
2. The study explicitly avoids recovery fine-tuning. While this isolates pruning effects, it limits applicability, since fine-tuning is standard practice for high-quality pruned models. A section quantifying how fine-tuning interacts with the law would be valuable.
3. The proposed pruning method demonstrates limited effectiveness when applied to LLaMA-3.2 and Qwen-3 architectures. In recent work, pruning techniques are often integrated directly into the training process to achieve better efficiency and stability.
4. There are several existing papers on scaling laws for inference-efficient models; it would be helpful if the authors discussed how their approach differs from these works: (1) https://arxiv.org/abs/2401.00448 (2) https://arxiv.org/abs/2501.18107

### Questions
See weaknesses.

### Soundness
2

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
4

### Summary
This paper introduces pruning laws, a framework to predict the performance of a pruned Large Language Model (LLM) based on its original performance and the pruning ratio. The authors propose a simple power-law that connects the post-pruning performance to the base model's performance and the pruning ratio. 

The author empirically demonstrates that the law can be applied to newer model architectures.

### Strengths
1. The paper is clearly written, and the experiment results are comprehensive.
2. The authors are commendable in that they reported negative results along with positive ones.

### Weaknesses
1. It's not clear whether the power law is the best curve to fit for this problem. Figure 3 shows many tasks that don't seem to fit well at all. The authors did not discuss other curve-fitting options.

2. Even if we ignore the point above and assume that the power law is a good fit for OPT and LLaMA, there is not enough evidence that the curve holds for modern massively overtrained LLMs. It's quite difficult to believe that these laws would hold for massively overtrained LLMs (e.g., the latest LLaMA/Qwen/Gemma, etc) without incorporating training data as a factor. The intuition is that the more overtrained the models, the less sparsity there is in the model, and the harder it is to prune the models without degrading the downstream tasks' performance significantly. They could behave completely differently compared to older model generations. One experiment on OOD model extrapolation is not nearly sufficient to show why these laws would work on modern models. 

3. Width-pruning and unstructured-pruning provide much less real-world benefit than depth-pruning (due to difficulty in converting to wall-clock latency gain). The utility of these scaling laws is diminished if they don't hold for depth-pruning.

4. The paper's interpretation of P_0 is problematic when it's greater than 1. Does that mean pruning a small /epsilon would improve performance over the baseline?

5. It's commendable that the authors show many negative results, but it would be of real scientific value if the authors could explain the failure modes and what the failures reveal about the limitations of the scaling laws.

### Questions
1. Pruned models often go through a recovery fine-tuning stage to recover performance on downstream tasks. How would that affect the scaling law? 

2. How is the latency measured? The paper lacks critical details in the measurement setup.

3. Why does the law fail on knowledge-intensive tasks such as MMLU?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes a simple, empirical power-law relations to predict a pruned LLM's performance (L) based on its original score (L0)
and the pruning ratio (r). The model is fitted using five LLM families (2.7B–13B), three pruning strategies, and eight tasks.

While the evaluation is empirically broad and the authors claim strong universality with average extrapolation error below 7%, the contribution is fundamentally a descriptive curve-fitting exercise. While practically useful, the analytical justification is minimal.

### Strengths
The paper addresses the important and under-studied problem of predicting pruning behavior. The evaluation is reasonably extensive, including multiple architectures, pruning methods, tasks, and metrics. The resultant model is simple with large application potential.

### Weaknesses
The law is essentially obtained by empirical curve-fitting exercise using power-law regression on known monotonic degradation trends, with little theoretical justification. Despite claims of strong universality, actual errors could be high, especially given the reasonable but still small size of the chosen downstream tasks tested. The robustness of the transferability test is limited, raising concerns of overfitting.

Excluding recovery fine-tuning may bias the performance results downward and distort real-world pruning behavior.

### Questions
+ How sensitive are the two fitted parameters to dataset choice, metric noise, and random seeds? 

+ Why does the "one-shot calibration" process sometimes worsen performance? Does this indicate potential overfitting? 

+ Would the same α coefficient remain applicable to models that have undergone recovery fine-tuning? 

+ How does the pruning law perform when extrapolated beyond 90% pruning—does it accurately predict model collapse or divergence?

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
3

### Summary
This paper introduces “Pruning Laws”: a concise, interpretable power-law that directly relates the downstream performance of a pruned LLM to its un-pruned performance and the pruning ratio r. While scaling parameters and data jointly boosts accuracy, the accompanying memory/compute explosion makes deployment on resource-constrainedhardware prohibitive. Model pruning is a popular remedy, yet its impact on downstream tasks has remained unpredictable and costly to assess. Over 5 models, 3 pruning granularities  and 8 tasks (reasoning, QA, language modeling) show that Pruning Laws
•	predict post-prune accuracy with <7 % mean extrapolation error and ≤8 % error when zero-shot transferred to unseen models (LLaMA-3.1, Phi-3) or algorithms (SlimGPT, SVD-LLM);
•	foretell the critical pruning threshold beyond which performance collapses, eliminating expensive grid-search;
•	quantify task sensitivity: reasoning is most robust, QA most fragile; depth pruning yields 5× speed-up but high variance, unstructured keeps accuracy yet almost no speed-up, width sits in between;
•	deliver actionable guidelines for choosing method and ratio under any task or budget, enabling zero-shot or single-point-calibrated deployment.
Pruning Laws thus provide a principled, universally applicable framework for compressing and deploying LLMs without full re-tuning.

### Strengths
1.	The experiments are large-scale, detailed, and sufficient; all reproduction materials are fully open-sourced.
2.	The paper attempts to uncover a law governing the trade-off between model pruning and performance, offering both theoretical and practical value.
3.	The proposed formula is concise, intuitive, and easy to use.

### Weaknesses
1.	The proposed formula is largely empirical and lacks solid theoretical justification.
2.	The experiments show that the parameters P₀ and α depend on the specific model, pruning method, and task; their determination in practice remains highly empirical.
3.	The law fails to fit knowledge-extraction tasks such as MMLU, and it is still unclear whether it generalizes to more complex scenarios like mathematics, code generation, or multimodal applications.

### Questions
1.	The explanation of the proposed formula is largely empirical. Could you further investigate, from a theoretical perspective and based on the architecture of large models, why the pruning law can be modeled as such a power-law relationship? Have you attempted to fit the data with other functional forms?
2.	P₀ and α are conditioned on the specific model, pruning method, and task. Could you elaborate on the exact procedure used to derive their values for a given task in the experiments?
3.	The pruning scaling law performs well on models ranging from 2.7B to 13B parameters. Does it still hold for much smaller models (e.g., below 1B) or much larger ones (e.g., above 100B)?
4.	In the paper, recovery fine-tuning is deliberately excluded to isolate the effect of pruning itself. However, in practical applications, recovery fine-tuning is a common step. If recovery fine-tuning is introduced, will the pruning scaling law still hold? Would it be necessary to modify or extend the law?

### Soundness
3

### Presentation
3

### Contribution
3

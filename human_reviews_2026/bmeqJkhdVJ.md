# TELL-TALE:  Task Efficient LLMs with Task Aware Layer Elimination

- Avg Score: 3.33
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 4

## Abstract
In this paper we introduce Tale, Task-Aware Layer Elimination, an inference-time algorithm that prunes entire transformer layers in an LLM by directly optimizing task-specific validation performance. We evaluate Tale on 9 tasks and 5 models, including LLaMA 3.1 8B, Qwen 2.5 7B, Qwen 2.5 0.5B, Mistral 7B, and Lucie 7B, under both zero-shot and few-shot settings. Unlike prior approaches, Tale requires no retraining and consistently improves accuracy while reducing computational cost across all benchmarks. Furthermore, applying Tale during finetuning leads to additional performance gains. Finally, Tale provides flexible user control over trade-offs between accuracy and efficiency.  Mutual information analysis shows that certain layers act as bottlenecks, degrading task-relevant representations. Tale's selective layer removal remedies this problem, producing smaller, faster, and more accurate models that are also faster to fine-tune while offering new insights into transformer interpretability.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes a task-specific greedy method (TALE) for structured pruning of LLMs by removing redundant layers based on the change in performance over the validation set of the task at hand. Specifically, the algorithm retains the pruned model with the highest validation accuracy. The method is tested on 9 datasets using 5 different open source LLMs. Moreover, TALE is combined and tested also in different combinations with fine-tuning and analyzed with Information-theoretic measurements.

### Strengths
- The experiments are carried out on real world, production-ready, models.
- The analysis using mutual information is somewhat interesting and it seems a promising direction to study the causes of why removing certain layers cause performance to improve.
- Also Figure 4 reveals an interesting trend for LLMs: contrary to what happens in vision models, layer importance varies depending on the downstream task.

### Weaknesses
Unfortunately, the paper in its current form has several key flaws that undermine the validity of its claims and contributions. In detail:

**W1.**

Pruning on LLMs is a very active studied field (see eg. [1,2] and their related works). However, apparently TALE is not compared with any baseline (the closest work to TALE is [1] which could be a good comparison). Consequently, any claim on performance for now comes without any context.

---------------

**W2.**

The novelty stems primarily from the observation that careful layer removal in some cases improves the model performance on the downstream task. This fact is well known (and often observed) in the pre-trained model pruning literature for both vision and language tasks (see eg. Figure 1 of [2], where perplexity has qualitatively the same trend, and Figures in [3]).

---------------

**W3.**

The pruning costs of TALE seem a bottleneck, given it potentially requires a non-trivial amount of inference on data. Moreover, it seems very task-specific and requires also label supervision which may not justify its costs.

---------------

**W4.**

The Information-theoretic analyses of Sec.6 are framed as an explanation as to why pruning certain layers, in some cases, improves performance. Unfortunately, these analyses are purely observational. Figure 3 just shows that some layers carry "more information of the task label" (in the sense of $I_{\text{probe}}(X^{(l)},Y)$) than others. I would advise the authors to tone down the claims in this section and give a small refresher on notation after Equation 1 (especially what is $X^{(l)}$ and $Y$ when applying it to LLMs).

Also, it is unclear how well $I_{\text{probe}}(X^{(l)},Y)$ approximates true Mutual Information.

---------------

**Minor Weaknesses.**

- The claim on Pareto-optimality (LL260-264) of the authors' solution is not backed up neither by theoretical nor by empirical evidence.
- Figure 4 is a bit difficult to parse at first glance as too much information is displayed at once (probably an histogram would convey the same information in a cleaner way).

---------------

**_References_**:

[1] Song, Jiwon, et al. "Sleb: Streamlining llms through redundancy verification and elimination of transformer blocks." ICML 2024.

[2] Frantar, Elias, and Dan Alistarh. "Sparsegpt: Massive language models can be accurately pruned in one-shot." ICML 2023.

[3] Chen, Tianlong, et al. "The lottery tickets hypothesis for supervised and self-supervised pre-training in computer vision models." CVPR 2021.

### Questions
Thanking in advance for their responses, I'd kindly ask the authors:
- to reproduce their experiments with some baseline from literature (see Weaknesses). As of now, it is very difficult to understand the contribution, both in terms of results and novelty.
- to check if the accuracy improvements are transferable across tasks (i.e. pruning with TALE and other methods on task A and testing on task B yields an improvement for both tasks).
- to comment on the pruning procedure costs and comparing it with other methods from the literature (in terms of overall time/compute/memory usage).
- to address W4 and Minor Weaknesses in this review.

I'm aware my questions may be too extensive. But, unfortunately, at this stage I'm finding this work quite limited and very difficult to judge, given no comparison is available.

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes a model compression technique by model pruning strategy during inference time. Layers are pruned iteratively one at a time based on validation accuracy and retain only the layers which doesn’t reduce the validation accuracy below a threshold. They provide empirical analysis to show that certain layers are redundant or even detrimental for some task-specific datasets. They use an approximation of mutual information to provide an explanation on why pruning can improve task-specific performance.

### Strengths
* The method does not require any retraining, making it beneficial for various use cases
* A systematic analysis with four settings has been done with extensive experiments 
* The algorithm is clearly described, and the analysis and inferences are well structured and easy to follow

### Weaknesses
* Computational cost during the pruning phase - it will become expensive for very large models to evaluate the impact of dropping every single layer in every iteration
* The MI approximation provides solid theoretical insights but the authors use trained linear probes as an approximation 
* The paper do not relate TALE’s to existing pruning techniques

### Questions
* Since TALE relies on validation performance, how the validation set size might affect the pruning decisions?
* The pruning phase explores all the layers iteratively which can be computationally intensive. Could the authors provide an estimate for it (may be also for all the four settings that you have considered)?
* It would be interesting to compare the layers dropped using TALE and the layers dropped using MI analysis. Is there a consensus between these two approaches?
* It would also be interesting to explore other pruning techniques and compare them with TALE. For example, does TALE tend to remove layers with smaller weight magnitudes? What about SynapticFlow? Additionally, why is ‘importance’ defined in terms of validation accuracy?
* For clarity, what is “speedup”? Does it mean the pruned model is 1.2 times faster than base model? 
* Could the authors clarify whether Figure 4 corresponds to the non-FT Llama3-8B model?
* Edit suggestions for better readability
1. In Figure 2: increasing the size of the green star would improve visibility; it will be interesting for the readers to know which layer was dropped at each iteration - could be added as text annotation on top of each dots/iter
2. Line 365: Citation format needs to be corrected

### Soundness
3

### Presentation
3

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
This work proposes TALE, a simple pruning method for LLMs that sequentially removes entire transformer layers in a greedy manner based on whether they affect significantly the model validation accuracy on a specific task. This method was developed based on the intuition that sometimes applying the output projection to intermediate layers leads to better performance. Therefore, given the additional residual path of transformer layers, removing entire layers leads to the representations from intermediate layers propagating directly to the output layer. Furthermore, the authors provide a simple information-theoretic analysis on why this phenomenon manifests, visualizing how, an approximate estimate of the, mutual information between representations and the downstream task label changes across layers. Finally, the authors evaluate quite extensively their method on several tasks and architectures.

### Strengths
I believe that TALE is an algorithm that can be interesting for practitioners, because of the following reasons

- TALE is very simple to implement; it just requires a task validation set and then multiple inferences to score the contribution of each layer. 
- TALE can be applied to arbitrary models, provided that they have a residual path (although the specific residual structure can limit a bit what can be removed)
- Finally, TALE seems to provide performance improvements on a per-task level and also seems to be compatible with finetuning.

### Weaknesses
- The task-specific nature of TALE can be limiting; one needs to first have a representative dataset for the downstream task in order to realize the efficiency gains. This is in contrast to more traditional pruning / quantization methods for efficiency that adapt the model on a general dataset and then apply it as is to new tasks. I believe this work could benefit by more comparisons with these set of methods, targeting similar efficiency gains. TALE can also be made "task-agnostic" by optimizing jointly on all tasks as the authors show on Appendix G but the performance is lacking, with the performance being mostly worse than the baseline and the speedup being minor. 

- TALE can be computationally expensive due to the greedy search performed, especially when a model is deep. While one could argue this only needs to happen once per task, LLMs now are easily used in multiple tasks via in-context learning / specific prompting, which might lead to excessive computational costs. 

- The authors do not  ablate the dependence of TALE on the validation set, which can be a critical hyperparameter. One can imagine that the validation set needs to be sufficiently large in order for the search to succeed and lead to better downstream task accuracy, so its size will be informative for practitioners.

- Finally, the mutual information analysis, while interesting, it is a bit crude as it relies on just linear probes. Furthermore, I am not sure how much additional information it adds, given the results on Appendix C.

### Questions
Besides what was mentioned in the weaknesses section, some more questions for the authors are

- Algorithm 1 allows for some performance degradation between steps according to the hyperparameter epsilon, what are typical values for it?
- The baseline performances shown at Table 5 are generally lower than what reported in the respective model cards for Llama 3.1 8B. Any idea on why this is the case? This is important as, for example, the performance on MMLU for Llama 3.1 8B Instruct (the same version that the authors used) is reported as 69.4% here https://huggingface.co/meta-llama/Llama-3.1-8B, which is already better than all of the accuracies reported on Table 5, hence it is unclear what the gains of TALE are in such settings.
- In the first paragraph of the discussion section, the authors mention that their later layer deletion finding challenges prior claims that early layers are redundant and more amenable to removal, without providing any reference for the prior claims. Where was this claim originally mentioned? If anything I would have expected prior claims to be the other way around, as the first layers are the ones that are closest to the raw information of the data.

Overall, at least at the moment, I am on the negative side for this work, primarily due to the missing ablations and comparisons I mentioned in the weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
2

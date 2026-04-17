# Elastic MoE: Unlocking the Inference-Time Scalability of Mixture-of-Experts

- Decision: Reject
- Scores: 4, 4, 4, 8

## Abstract
Mixture-of-Experts (MoE) models typically fix the number of activated experts $k$ at both training and inference. Intuitively, activating more experts at inference $k'$ (where $k'> k$) means engaging a larger set of model parameters for the computation and thus is expected to improve performance. However, contrary to this intuition, we find the scaling range to be so narrow that performance begins to degrade rapidly after only a slight increase in the number of experts. Further investigation reveals that this degradation stems from a lack of learned collaboration among experts. To address this, we introduce Elastic Mixture-of-Experts (EMoE), a novel training framework that enables MoE models to scale the number of activated experts at inference without incurring additional training overhead. By simultaneously training experts to collaborate in diverse combinations and encouraging the router for high-quality selections, EMoE ensures robust performance across computational budgets at inference. We conduct extensive experiments on various MoE settings. Our results show that EMoE significantly expands the effective performance-scaling range, extending it to as much as 2–3$\times$ the training-time $k$, while also pushing the model's peak performance to a higher level.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper postulates that elastic allocation of compute in MoE layers (e.g., activating more experts per token than in the training setup) can provide additional flexibility for inference. However, as the authors show, the naive approach, where we just increase the top-k in the router, does not result in model quality improvement. The authors hypothesize that the reason is lack of collaboration between experts - the model is not able to leverage new combinations of experts not seen during training. The proposed solution is to "simulate" training with larger top-k by sampling smaller groups of experts from larger, "ideal" groups, that will be seen during inference. The paper shows that when this method is in use, increasing router top-k to ks larger than seen during training is significantly more effective.

### Strengths
1. The paper is clearly written and easy to follow.
2. The motivation is clear. The paper tackles a natural problem, provides convincing reasoning on its nature, and proposes a solution.
3. The methodology is consistent and the results are positive.
4. Evaluation is based on three open-source models and two training scenarios (LoRA and MoE FFN).

### Weaknesses
1. Examination of the proposed approach is limited to training on the small instruction-tuning dataset.
2. The instruction fine-tuning is extremely lightweight (only 50K samples). Let's say we have an MoE model trained with topk=2. If we want to infer it with topk=4, we could just directly fine-tune on top-k=4, instead of using co-activation sampling, where we sample 2 experts from groups of size 4. The argument of the reduced training cost with stochastic co-sampling is not relevant since the entire fine-tuning phase has negligible cost compared to the model pretraining. This weakness is connected to weakness 1. mentioned above.
3. The introduction of hierarchical router loss feels a bit orthogonal to the main body of the paper, especially given results in the Section 5.3, showing that the main ingredient contributing to the benefit of EMoE is co-activation sampling.

### Questions
1. Did authors consider evaluating the proposed technique in the pretraining or continued pretraining setup?
2. In Table 1, there is a comparison of models trained with k experts when using k'>k during inference. Could the authors add to the comparison the model fine-tuned on the target k'? So for example, in the second group of Table 1, I would love to see a comparison between EMoE(k'=16) and OLMoE-1B-7B-0924 directly fine-tuned on topk=16. This model directly fine-tuned with topk=16 would technically have a larger cost, but as mentioned before, any cost of fine-tuning on the small dataset can be considered negligible, so ultimately the best approach is the one giving the best performance after tuning on this dataset. 
3. The authors may also comment or provide additional experiments on the points mentioned in the Weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper addresses the topic of improving the inference-time scalability of Mixture-of-Experts (MoE) models. The authors observe that standard MoE architectures, when trained with a fixed number of active experts (k), tend to perform poorly if a larger number (k′>k) is used at inference. They identify the cause as insufficient co-training of expert combinations and propose Elastic Mixture-of-Experts (EMoE) to mitigate this issue. EMoE combines (1) stochastic co-activation sampling, which encourages expert collaboration across diverse subsets without additional training cost, and (2) a hierarchical router loss that produces more stable expert rankings. The experiments show that EMoE models trained with small k exhibit improved performance when evaluated with higher k′, outperforming standard Top-k and dynamic routing baselines.

### Strengths
* The topic is very timely and important, as efficient scaling and inference-time flexibility in MoE models are of central interest to current LLM research.
* The paper is clearly written and presents the motivation and results in a structured way.
* The empirical evaluation is thorough and includes ablations isolating the effects of the proposed components.
* The approach introduces almost no additional training cost and could be easily adopted in existing MoE frameworks.

### Weaknesses
The main conceptual limitation is the lack of comparison with a model trained with a larger number of experts. Figure 2 in the paper itself shows that when a model is trained with higher k_train, it already generalizes better to larger k′ at inference. This raises an important baseline question: if we simply train the model for larger k, would that not achieve the same or better performance than EMoE, possibly without additional algorithmic complexity? 

Since EMoE aims to emulate large-k behavior while training with small k, the natural control experiment would be: train standard MoE with k_train = 6 or 8, evaluate it at comparable inference budgets, and compare both training cost and resulting performance. Without this, it is difficult to assess whether EMoE’s advantage stems from genuine elasticity or from the baseline’s artificially small k. In practice, the training-cost versus performance trade-off determines real-world usefulness, and that dimension is underexplored.

### Questions
1. How does EMoE compare to a model trained directly with a higher k_train, both in performance and in compute cost?
2. Would the claimed benefits persist if training compute were scaled accordingly (i.e., fair total-FLOP comparison)?

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
3

### Summary
The paper proposes Elastic Mixture-of-Experts (EMoE), a training strategy that enables post-hoc inference scalability in sparse MoE architectures. By introducing stochastic co-activation sampling and a hierarchical router loss, the authors aim to alleviate the mismatch between the training-time expert co-occurrence distribution and the inference-time routing patterns when increasing the number of activated experts. Experiments on several MoE backbones (LoRA-MoE, OLMoE, DeepSeek-V2-Lite) demonstrate smoother performance scaling and moderate improvements under larger inference budgets.

### Strengths
- The work systematically investigates the phenomenon that scaling up the number of activated experts at inference time causes performance degradation, which is empirically well presented.

- The proposed stochastic sampling mechanism and hierarchical router regularization are simple to implement and compatible with existing MoE frameworks.

- The method yields clear empirical gains with nearly no additional training cost, offering some practical relevance for deployment scenarios with dynamic compute budgets.

### Weaknesses
- The paper does not provide sufficient comparison against directly fine-tuning or training a model with a larger number of active experts 𝐾 under the same data and compute budget. Empirically, the reported improvements are modest (mostly 1–2 points) and statistically insignificant relative to directly increasing 𝐾. Since the training cost difference is marginal, simply fine-tuning with a larger  𝐾 remains a simpler and equally effective alternative, which undermines the necessity and originality of EMoE.

- The proposed stochastic co-activation sampling and hierarchical router loss offer only minor methodological variations over existing ideas such as DropMoE, expert dropout, or router regularization. 

- The elasticity of EMoE is restricted: the method performs reasonably only within roughly 2–3× of the training (K_{\text{train}}); performance deteriorates beyond this range. The method is also sensitive to the hyperparameter (K_{\text{ideal}}), requiring task-specific tuning. Moreover, EMoE does not address fundamental MoE bottlenecks such as communication overhead or expert load imbalance, raising doubts about its scalability and practicality for large-scale deployment.

### Questions
- How does EMoE fundamentally differ from directly fine-tuning or training a model with a larger number of active experts?

- Can the authors provide quantitative evidence that stochastic co-activation sampling improves expert collaboration, rather than simply introducing noise? For example, can you measure diversity or mutual information across experts before and after applying EMoE?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
Typically, MoE models use a fixed number of experts (k) during both training and inference. The authors identify that if one tries to activate more experts at inference (k′ > k), performance unexpectedly drops after a slight increase, rather than improving. They trace this to a lack of learned collaboration among experts beyond the combinations seen during training. The paper proposes (1) Stochastic co-activation sampling, which during training randomly samples diverse combinations of experts (from a larger candidate pool) to co-activate, thereby teaching experts to work together in various groupings. (2) Hierarchical router loss, a regularization term (based on KL divergence) that encourages the router’s output distribution to be sharp (non-uniform), producing a clear expert ranking for each token. EMoE is shown to extend the inference expert count to about 2–3× the training count effectively.

### Strengths
Well-jusified problem to develop a method that offers large-k flexibility at inference, while avoiding its training cost. A good motivation example.

The authors convincingly diagnose this as a result of experts never having learned to work together in those larger groupings.

During training, instead of always using the top-k experts, EMoE occasionally samples a small subset from a larger top-$k_{ideal}$ pool. It is a simple but innovative solution.

The paper is well-written, clearly explained.

The experiments are extensive (Moe using FFN and Lora, nine diverse tasks). EMoE-trained models exhibit monotonic improvement as the number of inference experts increases, eliminating the drop seen in standard MoEs.

### Weaknesses
The motivation of the hierarchical router loss in view of the entire paper that rather focuses on dealing with the problem of lack of collaboaration is a little weak.

One missing comparison is to a model trained with a higher k from the start. For instance, if we train a model with k_train = 4 (using 4 experts per token) and use 4 at inference, how does that compare to an EMoE model trained with k_train = 2 but using 4 at inference?

### Questions
How does the method behave beyong the scenario with more than 3x experts at inference?

Are there any disadvanatges of having a too sharp expert distribution? given the theme of the paper, could making the router too sharp hurt diversity?

### Soundness
3

### Presentation
4

### Contribution
3

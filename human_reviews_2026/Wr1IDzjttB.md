# Amortized Bayesian Meta-Learning for Low-Rank Adaptation of Large Language Models

- Decision: Reject
- Scores: 2, 2, 2, 2

## Abstract
Fine-tuning large language models (LLMs) with low-rank adaptation (LoRA) is a cost-effective way to incorporate information from a specific dataset. However, it is often unclear how well the fine-tuned LLM will generalize, i.e., how well it will perform on unseen datasets. Methods have been proposed to improve generalization by optimizing in-context prompts, or by using meta-learning to fine-tune LLMs. However, these methods are expensive in memory and computation, requiring either long-context prompts or saving copies of parameters and using second-order gradient updates. To address these challenges, we propose Amortized Bayesian Meta-Learning for LoRA (ABMLL). This method builds on amortized Bayesian meta-learning for smaller models, adapting this approach to LLMs while maintaining its computational efficiency. We reframe task-specific and global parameters in the context of LoRA and use a new hyperparameter to balance reconstruction accuracy and the fidelity of task-specific parameters to the global ones. ABMLL provides effective generalization and scales to large models such as Llama3-8B. Furthermore, as a result of using a Bayesian framework, ABMLL provides improved uncertainty quantification. We test ABMLL on CrossFit and Unified-QA datasets and find that it outperforms existing methods on these benchmarks in terms of both accuracy and expected calibration error.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper sets out to improve the speed with which LoRA fine-tuned LLMs can generalize to unseen datasets via an amortized Bayesian meta-learning approach, with the additional benefit of providing uncertainty estimates on the predictions made by these fine-tuned LLMs. The method introduces a hierarchical Bayesian prior over LoRA weights. Each adapted weight matrix's LoRA parameters are modelled as Gaussian random variables, whose means and variances are themselves produced by LoRA adapters, yielding distributions over weight updates that encode both fit and uncertainty. Training proceeds via a $\beta$-tempered meta-ELBO, which attempts to balance data fit against a KL regularizer that ties the per-task posteriors to a global prior. The authors conduct a set of experiments on common meta-learning datasets, and show modest improvements on a couple of baselines, while remaining computationally cheaper than full MAML-style meta learning algorithms.

### Strengths
The paper is clearly written and does a good job of setting out the related work, the problems faced therein with the expensive 2nd order gradient calculation, and describing their proposed method.

### Weaknesses
Unfortunately the paper has a few weaknesses.

The first is a high-level concern about the relevance of MAML-style approaches in the age of modern pre-trained foundation models. One of the strengths of LLMs and agent systems is that they can leverage their context windows to learn how to perform a one-off task by drawing in relevant research, examples and other information to learn in-context how to solve the task. This requires no gradient updates during deployment, is cheap and simple. For tasks which will be encountered repeatedly, where the user may wish to fine-tune the model to amortize the cost of learning to perform the task, then vanilla LoRA fine-tuning is already computationally cheap, and as shown by the authors themselves in their experiments, performs well. I recommend that instead of studying MAML-style approaches in isolation, these simpler baselines which are predominantly adopted by the community should be carefully run as baselines and proven to be inferior before proceeding.

The second concern is that the method, despite the name, does not appear to be truly amortized. In Ravi & Beatson 2019, the model learns a single inference network which maps the task's dataset to the posterior parameters in one forward pass (requiring no optimization at inference time, thus yielding constant adaptation cost wrt. the number of tasks). In contrast, the method proposed in this paper takes $K$ gradient steps on each task's LoRA parameters $\phi_{i}$ at both train and test time. The amortization is therefore only structural in that the same LoRA parameterization and global prior $\theta$ is shared across tasks, however the inference itself is still optimized per task. As a result, ABMLL appears to inherit the same per-task adaptation cost as MAML/Reptile, albeit avoiding the second-order gradients.

The proposed method does not appear to be "Bayesian" either. This arises due to a number of factors; the first is that the global "posterior" $q(\theta)$ is collapsed to a point estimate due to setting $KL[q(\theta) \Vert p(\theta)] = -\log p(\theta)$, thus reducing any uncertainty arising at the meta level (i.e. if a task is far from the distribution encountered during meta-training). Further, the authors correctly note that in Bayesian deep learning, particularly for networks with large parameter counts, the regularization provided by the KL term can overwhelm the data fit and reduce performance (akin to the issues described by [Trippe & Turner, 2018](https://arxiv.org/abs/1801.06230)). While moderately tempering the posterior is seen by some practitioners as a valid way to deal with this shortcoming, the authors appear to use a very aggressive tempering of $\beta = 10^{-7}$. If the training proceeds in bf16 or fp16 mixed precision, then the contribution from any term multiplied by this value risks being numerically negligible. This, combined with the low batch size of 2 resulting in high gradient noise means that the method (as trained) effectively performs maximum-likelihood fine-tuning in LoRA space, with a tiny Gaussian prior regularizer which risks being dominated by the size of the gradient updates. For the ablation with $\beta = 10^{-16}$, the KL term is effectively zeroed out in half-precision.

Finally the evaluations in the paper are unfortunately quite limited: using only a single model - it would have been good to examine a broader range of model families and sizes (for instance spanning Llama, Qwen, Olmo, Phi, Gemma and sizes ranging from 1.5B parameters to 70B parameters). The paper makes a claim about the efficiency of this method, however the authors omit a study of the FLOPs or memory consumption of the method in comparison to baselines. The empirical performance of the method is also unfortunately slightly underwhelming, closely resembling the regular "structured" LoRA, with greater computational and memory cost as a result of the four LoRA adapters for the means and variances of the task-specific and global parameters, respectively. This additional cost and complexity does not seem to be commensurate with the performance gains.

### Questions
- You claim improved calibration. Have you examined whether the learned variance parameters correspond to meaningful epistemic uncertainty (e.g. via correlation with task difficulty or out-of-domain samples)?
- On the comparisons to in-context learning, in which scenarios would you expect ABMLL to outperform well-designed in-context learning baselines?
- You mention that the prior reflects the "spread" of pretrained parameters. Could you describe exactly how this was estimated? Was it layer-wise empirical variance, or simply a Normal-Gamma prior with fixed hyperparameters?
- What numeric precision was used during training? For smaller $\beta$ values of $10^{-7}$ and below, how did you ensure that these coefficients were represented accurately and did not underflow to zero in mixed-precision computation? With $\beta$ so small, and the prior's effect minimal, how do you explain the improved calibration?
- The abstract and introduction emphasised that ABMLL is computationally and memory efficient; could you provide explicit measurements (such as time per epoch, peak GPU memory, etc) compared to LoRA, Reptile, and MAML under equal settings. I also note that ABMLL introduces four pairs of LoRA adapters, roughly quadrupling adapter parameters vs vanilla LoRA. How do you control for fairness when comparing to baselines?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper combine ABMLL with LoRA to mitigates the high computational cost of Bayesian meta learning method. The results show that the proposed method achieves better performance compared to baseline methods.

### Strengths
* The presentation and organization of the paper are clear and easy to follow.
* The proposed method is simple and effective.
* The paper effectively mitigates the high computational cost of Bayesian meta learning through introducing LoRA.

### Weaknesses
* The paper essentially combines Bayesian method and LoRA in a straightforward manner. Combining Bayesian methods with LoRA is a common approach in recent work; based on this, the paper lacks furthermore novelty.

* Computational cost is a crucial aspect to the proposed method, as reducing memory usage is the primary motivation for introducing LoRA. However, the corresponding analysis is missing.

* The baseline method is relatively outdated; the authors should consider comparing with more recent approaches.

### Questions
* Does the proposed method require multiple samples from the posterior distribution during inference? If so, what is the additional computational cost introduced?

* According to Figure 2, ABMLL appears to require significantly more epochs for ECE to converge compared to other methods. Is this behavior consistent across all training datasets? This potential weakness should be further analyzed and discussed in the paper.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes a scalable Bayesian meta-learning framework to improve the generalization and uncertainty estimation of LoRA-based fine-tuning for LLMs. Traditional meta-learning methods like MAML and Reptile are computationally expensive and memory-intensive, while LoRA fine-tuning lacks generalization across unseen domains. ABMLL adapts amortized Bayesian meta-learning by modeling global and task-specific LoRA parameters under a probabilistic framework, balancing data reconstruction with parameter regularization through a tunable β term. Experiments on CrossFit and Unified-QA benchmarks using LLAMA3-8B show that ABMLL consistently achieves higher accuracy and better calibration (lower ECE) than standard LoRA and Reptile methods, while maintaining scalability and efficiency. The approach bridges Bayesian inference with parameter-efficient LLM adaptation, offering principled uncertainty quantification and improved cross-task generalization.

### Strengths
+ **Reasonable integration of Bayesian meta-learning and LoRA.** The paper adapts amortized Bayesian meta-learning to large-scale LoRA fine-tuning, offering a principled probabilistic framework for task adaptation and uncertainty quantification in LLMs.
+ **Good writing.** The paper's writing is clear and easy to follow, making it accessible to the audience. 
+ **Improved generalization and uncertainty calibration.** Empirical results show consistent improvements in both accuracy and expected calibration error, demonstrating enhanced generalization to unseen tasks.

### Weaknesses
+ **Lack of Technical Novelty.** The paper essentially merges two fairly well-known ideas: (a) PEFT via LoRA and (b) meta-learning (or Bayesian meta-learning) to obtain better generalization. One could argue that while the Bayesian angle gives a spin, the core innovation is limited. The amortised Bayesian meta-learning applied to LoRA is interesting, but meta-learning for PEFT has already been explored [1]. If the improvement comes mainly from combining existing techniques (LoRA + amortised Bayesian meta-learning), is this more an engineering/combination paper rather than a fundamentally new algorithmic contribution?
+ **Insufficient experiments: datasets and applications.** The empirical evaluation is limited to a small number of tasks/datasets (namely, Winograd and cls). While these are valid, the scope is narrow.
+ **Performance issue.** In these settings, the calibration of method does not bring significant improvement (as shown in Table. 1).
+ **Missing Baselines.** The idea of combining PEFT and Meta-Learning has been explored by prior work [1,2,3,4,5] (I might have missed some too). Please discuss the relationship between them and the reason why they are not included for comparison. 

**References**
- [1] Gheini, Mozhdeh, Xuezhe Ma, and Jonathan May. "Know Where You're Going: Meta-Learning for Parameter-Efficient Fine-Tuning." arXiv preprint arXiv:2205.12453 (2022).
- [2] Hou, Zejiang, Julian Salazar, and George Polovets. "Meta-learning the difference: preparing large language models for efficient adaptation." Transactions of the Association for Computational Linguistics 10 (2022): 1249-1265.
- [3] Shao, Yihua, et al. "In-Context Meta LoRA Generation." arXiv preprint arXiv:2501.17635 (2025).
- [4] Cheng, Bo, et al. "MeTA-LoRA: Data-Efficient Multi-Task Fine-Tuning for Large Language Models." arXiv preprint arXiv:2510.11598 (2025).
- [5] Tian, Zichen, Yaoyao Liu, and Qianru Sun. "Meta-Learning Hyperparameters for Parameter Efficient Fine-Tuning." Proceedings of the Computer Vision and Pattern Recognition Conference. 2025.

### Questions
See above.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes Amortized Bayesian Meta-Learning for Low-Rank Adaptation of Large Language Models (ABMLL), an approach that combines amortized Bayesian meta-learning with LoRA to enable scalable adaptation of large-scale LLMs. The method treats the global LoRA parameters as Bayesian latent variables that generate task-specific LoRA adapters, allowing the model to learn a shared probabilistic prior across tasks while efficiently adapting to new ones.

### Strengths
-  The paper makes an effort to address generalization as an important problem in large language model (LLM) adaptation.

 - The presentation is clear and the motivation is well established, providing a coherent link between Bayesian meta-learning and LoRA-based fine-tuning.

 - The experiments are well-presented, evaluating accuracy and expected calibration error.

### Weaknesses
- W1. Limited novelty over prior work.
 In Algorithm 1, the main training and testing procedures of ABMLL closely mirror Algorithm 1 and Algorithm 2 in [1], with the only modification being the reparameterization via LoRA. Since [1] also targets generalization performance in meta-learning, and the paper does not clearly articulate the conceptual or methodological advances beyond [1], the novelty of the proposed approach appears extremely limited.

 - W2. Lack of empirical comparison with strong baselines.
 Although the paper positions ABMLL as an approach for improving generalization through optimizing in-context prompts or meta-learning, it fails to empirically validate these claims against concurrent methods. In particular, the authors discuss the limitations of related work but do not provide quantitative comparisons with representative baselines such as LiFT, especially regarding peak memory cost and computational efficiency. This omission weakens the paper’s contribution relative to parallel research efforts.

 - W3. Heuristic and unstable hyperparameter choices.
 The paper acknowledges that balancing the reconstruction error is crucial, yet the corresponding hyperparameters are chosen heuristically without theoretical justification. As shown in Table 2, the method is highly sensitive to these hyperparameters, undermining robustness across tasks. Moreover, the same set of hyperparameters is used for two terms in Eq. (2), further raising concerns about its practical choice.

 - W4. Insufficient experimental motivation and limited improvement.
 The experiments mainly compare ABMLL with standard LoRA and Structured LoRA, yielding marginal performance gains. Given that the paper emphasizes cross-task generalization, the experimental design lacks motivation and diversity. It would be more convincing to include baselines that explicitly target generalization, enabling a clearer evaluation of the proposed method’s effectiveness.

[1] Ravi S, Beatson A. Amortized bayesian meta-learning. International Conference on Learning Representations. 2019.

### Questions
Stated in Weakness

### Soundness
2

### Presentation
3

### Contribution
1

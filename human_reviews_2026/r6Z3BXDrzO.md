# On the Impossibility of Retrain Equivalence in Machine Unlearning

- Decision: Reject
- Scores: 2, 6, 6, 4

## Abstract
*Machine unlearning* seeks to selectively remove the "influence" of specific training data on a model’s outputs. The ideal goal is *Retrain Equivalence*--behavior identical to a model trained from scratch on only the retained data. This goal was formulated for models trained on i.i.d. data batches, but modern pipelines often involve multi-stage training, with each stage having a distinct data distribution and objective. Examples include LLM fine-tuning for alignment, reasoning ability, etc. Our study shows via theory and experiments that this shift to multi-stage training introduces a fundamental barrier for machine unlearning. The theory indicates that the outcome of local unlearning—methods that only use gradients computed on the forget set—is path-dependent. That is, a model's behavior during unlearning is influenced by the *order* of its training stages during learning, making it impossible for path-oblivious algorithms to universally achieve Retrain Equivalence. We empirically demonstrate the same phenomenon in LLM post-training across Llama and Qwen models (1B–14B) with gradient ascent, NPO, and SimNPO local unlearning algorithms. Models fine-tuned via different orderings of identical training stages diverge in behavior during unlearning, with the degradation in GSM8K accuracy after unlearning varying by over 20\% across paths. We also observe that some learning paths consistently produce models that unlearn slowly. During unlearning, whether the probability mass gets squeezed into paraphrasing or alternative concepts is also path-dependent. These results consistently show that Retrain Equivalence is an ill-posed target for local unlearning algorithms, so long as the target models are trained in stages. In situations where access to models' training histories is hard, the current work calls for rethinking the definition and desiderata of machine unlearning.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper considers the difficulty of unlearning with “local” algorithm, i.e., those that only use the model weights and the forget set. They conduct experiments on LLMs and note the ordering of the training data can make such unlearning algorithms significantly less effective , and provide some theory in the case of linear regression. They conclude that further work is needed in formulating this type of unlearning.

The authors seem unaware of past work which had already shown this formulation for unlearning is ill-defined; a dataset without the forget set can give the same model as training with the dataset (with some probability over the training orderings), e.g., Thudi., et al [1]. Note at the time “approximate unlearning” only really consisted of “local” unlearning algorithms. This led to further research into forgeability, and analysis of the closely related notion of per-instance privacy. It also has been one of the primary reasons for stronger algorithmic definitions of unlearning (which use the training algorithm and retain dataset) and disclosures when using weaker definitions, e.g., see the ToFU benchmark [2].

In the context of this literature, this paper makes further observations on the nature and prevalence of forgeability. However, this paper does contribute to the existence of this phenomenon, which was already proved in general for mini-batch SGD training algorithms [1] using an even stronger metric (i.e., the model weights are the same which implies predictions are the same). I recommend this blog post for a more intuitive understanding of forging https://ai.stanford.edu/~kzliu/blog/unlearning (which is described in terms of verifying unlearning), and the work of Kong et al., [3] which recaps the results of [1] a bit more cleanly.

In the questions I make some suggestion for how the authors might change their presentation given this; while I do not think the original claims are valid, the paper can claim to add to existing observations about this unlearning setting.

[1] Thudi, Anvith, et al. "On the necessity of auditable algorithmic definitions for machine unlearning." 31st USENIX security symposium (USENIX Security 22). 2022.

[2] Maini, Pratyush, et al. "Tofu: A task of fictitious unlearning for llms." arXiv preprint arXiv:2401.06121 (2024).

[3] Kong, Zhifeng, Amrita Roy Chowdhury, and Kamalika Chaudhuri. "Forgeability and membership inference attacks." Proceedings of the 15th ACM workshop on artificial intelligence and security. 2022.

### Strengths
1) The linear regression theorem adds to theory around quantifying notions of forgeability.

2) Experimental results quantifying the path dependence of various local unlearning algorithms seem strong and valuable to the literature on unlearning difficulty.

### Weaknesses
1) The theory, while interesting, is still focused on linear regression where efficient effective unlearning is not a problem, e.g., Guo et al., [4].

2) Moreover, bounds for the difficulty of unlearning when using SGD like algorithms (even when applied to deep neural networks) already exists, Sephavand et al., [5]; the argument deep learning is mysterious so we work with linear regression seems ill-made given the progress in the unlearning literature.

3) On the observation of path dependence (i.e., one starts with different initial models), old work (before even the unlearning community's observation) had already discussed the high variance in models between data orderings and how this can be exploited [6].


[4] Guo, Chuan, et al. "Certified data removal from machine learning models." arXiv preprint arXiv:1911.03030 (2019).

[5] Sepahvand, N.M., Thudi, A., Isik, B., Bhattacharyya, A., Papernot, N., Triantafillou, E., Roy, D.M., Dziugaite, G.K.. (2025). Leveraging Per-Instance Privacy for Machine Unlearning. Proceedings of the 42nd International Conference on Machine Learning

[6] Shumailov, Ilia, et al. "Manipulating sgd with data ordering attacks." Advances in Neural Information Processing Systems 34 (2021): 18021-18032.

### Questions
Below I give some suggestion for how the authors might re-present their work, which I think can help them be clear about what their contribution to the literature is. I think with such changes to the presentation this paper can be valuable to the literature on unlearning, and will be happy to reconsider the score. However, the paper misses the mark right now given it lacks context with existing results.

Suggestions: 

1) While the phenomenon that unlearning is dependent on training trajectory is known in various contexts (which can lead to it being ill-defined), it seems a concrete contribution of this paper is that it measures this dependence (often called forging) for LLM fine-tuning. I suggest the authors no longer claim to discover the former, and instead discuss past work on this phenomenon and more broadly unlearning difficulty. They can then just focus on their contribution to measuring this dependence in LLM fine-tuning, which is already mentioned throughout the paper (but often after claiming the former).


2) Given this, the description of the contribution of the theoretical result could be rephrased to emphasize that it quantifies how specific orderings diverge with continual “local” unlearning; the exponential divergence seems novel to the unlearning (difficulty) literature, and is already what is described in the theorem. One just needs to rephrase the summary of this result (in the context of past results) in the introduction and elsewhere in the paper.

3) Similarly the empirical section seems to be correct on its own, and just how these results are interpreted in the context of past results needs to be made clearer. For example, as far as I know the observation about the recency phenomenon seems novel, and to me is something worthy of a paper; one just needs to not claim to be the first to observe that machine unlearning depends on data ordering, and instead only make the more fine-grained claims of what dependencies were specifically observed.

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper aims to study the effect of data order in training pipeline that is later being unlearned could have significant impact on how unlearning might have effect. As such, they claim that ideal unlearning might be impossible as we don't have exact kwnoledge about the data order in training pipeline.

They further support this claim by an empirical setup where unlearning targets happen at different positions of training pipeline,  and observe that applying the same training procedure results in different outcomes, showcasing the senstitivy of unlearning algorithms on the place where unlearning targets are in training pipeline.

### Strengths
The idea of this paper is interesting. It analyzes how the exact position at which unlearning targets appear can affect unlearning performance, and thus why ideal unlearning may be impossible when the algorithm is not given this positional information.

The theoretical results look sound, and the empirical results support them.

Overall, the paper sheds light on a question that was not clearly studied before. Most prior work focused on cases where unlearning targets were introduced immediately before the unlearning step.

### Weaknesses
I think the main weakness is that the empirical results are somewhat limited. I would encourage the authors to run experiments on additional benchmarks such as RESTOR [1] and MUSE [2] to better assess the sensitivity of unlearning algorithms to how recently unlearning targets were introduced across different setups.

Also, I wouldn’t frame this as an impossibility of unlearning; rather, it is an impossibility of defining the ideal model when, in practice, we may not have access to it.

-------------


[1] Rezaei, Keivan, et al. "RESTOR: Knowledge Recovery in Machine Unlearning." arXiv preprint arXiv:2411.00204 (2024).

[2] Shi, Weijia, et al. "Muse: Machine unlearning six-way evaluation for language models." arXiv preprint arXiv:2407.06460 (2024).

### Questions
Given the results in this paper, what do the authors propose for evaluating machine unlearning in practical scenarios? In existing work, it is often assumed that the model was trained on the unlearning targets immediately prior to applying the unlearning algorithm, so the ideal model is simply the checkpoint before the introduction of those targets.

Do the authors suggest excluding all metrics that compare to this ideal model? Or do they offer ideas for redefining or approximating the ideal model so that such comparisons remain meaningful?

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
3

### Summary
This paper investigates how the effectiveness of machine unlearning aimed at achieving Retain Equivalence varies depending on the order in which the model learns different datasets. The study focuses on local unlearning methods that rely solely on gradients computed from the forget set. The authors theoretically prove, under a linear model assumption, that the outcome of unlearning is path-dependent, meaning it can differ significantly depending on the sequence of training data. Empirically, the authors simulate a sequential post-training process using LLaMA and Qwen models and similarly observe that unlearning behavior is strongly path-dependent.

### Strengths
* Unlike prior unlearning studies that mainly focus on designing algorithms to maximize benchmark performance, this paper highlights a more fundamental issue: the effectiveness of unlearning critically depends not only on the algorithm itself but also on how the forget data was originally learned by the model.
* The argument is further strengthened by a theoretical result showing that, under a linear model assumption, performing local unlearning on two models trained with different data orders cannot simultaneously satisfy the defined notion of Retain Equivalence.
* Extensive experiments with LLMs empirically confirm the path-dependent nature of unlearning and show interesting phenomena such as the recency effect, thereby connecting the theoretical findings to observable behaviors in realistic post-training pipelines.

### Weaknesses
* Definition 2.1 formalizes Retain Equivalence (RE) as the pointwise closeness of model predictions on a test set (any generic test set $X_{test}$). While this assumption enables a theoretical analysis, it arguably over-constrains the notion of equivalence. In practice, two models trained on the same data but with different random seeds can yield noticeably different predictions, especially on out-of-distribution inputs, yet should still be regarded as equivalent from a statistical or functional standpoint. Therefore, RE as defined in this paper captures only pointwise similarity rather than distributional or behavioral equivalence. Extending RE to a distribution-level notion, such as expected loss or output distribution similarity under the retain data distribution, would make the theoretical conclusions more general and more aligned with standard unlearning objectives.
* Section 4.3 appears to overstate the evidence for “path-dependent superficial forgetting”. The claim of a newly identified phenomenon is supported by only 40 prompts for each dataset (C, U, and R), i.e., 120 samples total, in a largely qualitative setup, which is insufficient to draw reliable or generalizable conclusions about superficial vs. deep forgetting. While the observation is interesting as an illustrative example, it does not yet constitute empirical evidence of a new phenomenon. 
* The paper defines local unlearning as using only the gradient of the forget set, thereby excluding many practical algorithms that employ lightweight retain-based regularization or contrastive objectives (e.g., Gradient Difference, DPO). This narrow definition simplifies the theoretical treatment but makes the conclusions less representative of real-world unlearning methods. As a result, the claimed impossibility result may not fully capture the feasibility of semi-local or weakly regularized unlearning approaches commonly adopted in practice.

### Questions
* Is there a specific reason for using different learning rates for both LLaMA2-13B and Qwen2.5-14B?
* Why did the authors choose to use LoRA? If the goal is to simulate the post-training process of LLMs, full fine-tuning might be a more appropriate setup. Also, during unlearning, are only the LoRA weights updated?
* In Figure 4, the results clearly show that unlearning is path-dependent, but how should we interpret this finding? In the setting where “deep forgetting” occurs, U is trained first. Does this imply that the effect arises because the recency effect is weaker in this configuration?

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
3

### Summary
This work considers unlearning within a multi-stage training framework. Specifically, this work propose a theoretical understanding that model's behavior during unlearning is influenced by the order of its training stages during learning. Therefore, it is impossible to universally achieve a retrained equivalent model with a path agnostic algorithm. Experimental results on different LLMs and unlearning heuristics verify the theory.

### Strengths
- The paper look at unlearning at a novel perspective: under multi-stage training.
- The paper propose a novel theoretical analysis for the impossibility result of universal relearning equivalence, which is important for understanding unlearning.

### Weaknesses
While I appreciate the theoretical insights provided by this work, I have a few concerns.
- How large is $t_0$? Looks like corollary 1 does not hold when $t* < t_0$. Then does that mean when using less amount of unlearning iterations, it is more likely that the model will achieve $\varepsilon$-RE?
- The theoretical results is based on 1. linear models, 2. gradient ascent. It's very different from the experiments which focuses on LLM, with different stages, and each stage will use some different method for learning. Grad ascent is also just a most naive unlearning heuristics for LLM.
- Given the theoretical analysis, I think it's more valuable to provide more simplified experiments to verify the theoretical understanding (e.g. classification tasks or even simpler linear models)
- I'm not convinced by the significance of the universality of retrain equivalence. Why do we need to find a universal $t$, such that under any path, the model unlearned at step $t$ has to achieve the same $\varepsilon$-RE? What's the issue with having different $t$ for different paths?

### Questions
- What is the motivation of using an approximate unlearning definition that measures similarity in output space, rather than the classic approximate unlearning definition which measures similarity in the model parameter distribution space?

### Soundness
3

### Presentation
3

### Contribution
2

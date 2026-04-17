# Learning from Observational Outcomes: Toward Causally-Aligned Language Model Fine-Tuning

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 4

## Abstract
Large language models are being widely used across industries to generate text that contributes directly to key performance metrics, such as medication adherence in patient messaging and conversion rates in content generation. Pretrained models, however, often fall short when it comes to aligning with human preferences or optimizing for business objectives. As a result, fine-tuning with good-quality labeled data is essential to guide models to generate content that achieves better results. Controlled experiments, like A/B tests, can provide such data, but they are often expensive and come with significant engineering, logistical, and ethical challenges. Meanwhile, companies have access to a vast amount of historical (observational) data that remains underutilized. In this work, we study the challenges and opportunities of fine-tuning LLMs using observational data. We show that while observational outcomes can provide valuable supervision, directly fine-tuning models on such data can lead them to learn spurious correlations. We present empirical evidence of this issue using various real-world datasets and propose DeconfoundLM, a method that explicitly removes the effect of known confounders from reward signals. In simulation experiments, DeconfoundLM more accurately recovers causal relationships and mitigates failure modes of methods that assume counterfactual invariance, achieving over 16% higher objective score than ODIN and other baselines, when entangled confounding is present.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper addresses the challenge of using large language models (LLMs) to optimise a specific task objective (reward) when only historical observational data are available for post-training. In such settings, distribution shifts between the observational data and the deployment environment—often caused by confounding variables (e.g. seasonality)—can bias the learned reward signal. As a result, directly fine-tuning the LLM on this observational data risks reinforcing spurious correlations rather than genuine causal relationships. To mitigate this, the authors propose a method to estimate a “deconfounded” reward function using an instrumental variable (IV) approach. This procedure aims to remove the influence of confounding factors and recover the causal effect of model actions on the true reward. The LLM is then (I believe) trained using this deconfounded reward estimate, rather than the reward observed in the historical dataset. Through empirical evaluation on a semi-synthetic dataset, the authors demonstrate that this IV-based correction leads to improved downstream performance and more robust generalisation to the true deployment environment.

### Strengths
- Using observational data for the purpose of LLM training and fine-tuning is an important problem with many real world applications.
- The authors provide many datasets and examples where training on observational data could lead to performance benefits.

### Weaknesses
1. **Lack of details:** My biggest issue with this paper is that the proposed method and experiments (particularly in Section 4) are not described in details sufficient to fully understand the contributions of this work, evaluate their validity and ensure the reproducibility of results. Most importantly:
    - What causal assumptions are imposed on the set of confounders for DefoncoundLM to work? (See also the question below)
    - What does DeconfoundLM really entail? How exactly is the contribution rom the observed outcomes “removed” (l. 320)?
    - After it has been removed, why happens next? Is the “confounder-adjusted reward” used as a reward function for RL?
    - If RL is used for post-training, what kind of optimisation algorithm is used? PPO? GRPO?
    - Does the method involve training a linear probe on the LLM embeddings? If yes, what is the architecture of this probe? If no, how is the reward predicted in Table 2?
    - For the baselines described in l. 383, how are the baselines exactly constructed and what loss was used for training?
I also provide additional questions in the ‘Question’ section below. I consider providing exhaustive answers to these questions a necessary step towards improving the paper, without which it is very difficult for me to evaluate the quality of this work.
2. **Limited novelty:** Results in Section 3 seem to validate the standard principles of the causal modelling: if you do not account for confounding in your modelling framework, your results will learn spurious correlations that will not generalise to the randomised (rather than observational) setting. The results in this paper seem to validate that this is also the case when fine-tuning language models (or training regression/classification heads on their embeddings), showing that LLMs follow the training dynamic imposed by the non-causal loss function, as expected. While this validation is nice, it provides limited novel insight or contributions.
3. **Limited evaluations:** The proposed method is only evaluated on a single semi-synthetic dataset. More dataset, with different relationships between confounders and the outcome function, would be necessary to further see the level of improvement the method can provide, and further illuminate under which circumstances it provides most benefits.

### Questions
- In line 255 you say claim that ‘the results emphasise the need to scale regularisation appropriately with model capacity to maintain generalisation’. However, in your results in Figure 5b, isn’t it the case that $\lambda=10000$ provides best performance across all model sizes? If yes, where is the above conclusion coming from?
- In the paragraph opening section 4, I think there is some confusion around the notation. In particular, what is the relationship between the auxiliary features $\tilde{\mathbf{X}}_i$, the observed features $\tilde{\mathbf{F}}_i$ and the confounders $\mathbf{C}_i$? Are they overlapping or disjoint sets? A causal diagram would be very helpful in understanding the relationships between different variables. Further, to make this estimation problem well-defined, it would be great to state what exact assumptions are imposed on the set of confounders and the general observational data. For example, is the set of observed confounders sufficient to disentangle the effect of the textual “action” on the outcome Y (unconfoundedness)? Answering this question rigorously and in detail would also make it clearer under which circumstances we can expect most gains from using this method.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper tackles how to fine-tune language models on observational data (like clicks or engagement) without letting them overfit to spurious confounders such as timing or popularity. It proposes DeconfoundLM, a method that tries to isolate the causal effect of the model’s text on outcomes and train on that signal, and shows in controlled experiments that this can outperform standard RLHF/DPO-style approaches.

### Strengths
Importance of the problem: leveraging observational data is important, and taking into account confounders is even more crucial based on this context.

Empirical evidence for the problem tackled: the stackexchange and Upworthy experiments illustrate the potential of the LLM for learning spurious correlations from observational data. The Upworthy study also shows a nice insight: larger LMs overfit more strongly to confounded observational signals and require much heavier regularization.

### Weaknesses
Observed confounders: the method assumes access to known confounders and corrects for them, but the paper doesn’t clearly specify  how these confounders are encoded in practice with DeconfoundLM, or how robust the method is to missing/mismeasured confounders.

Violoation of the IV assumption: the method relies on an IV-style correction and assumes the IV only affects reward via popularity in the example in 4.1. However, the exclusion restriction is quite strong and deserved more discussion.

HPT tuning: regularization strength is chosen using experimental ground truth CTR, not purely observational data. This gives the observational models access to oracle feedback that wouldn’t exist in a purely observational setting.

Experimental setup: the method is validated using only one synthetic experiment, which is not sufficient. Furthermore, this synthetic setup is designed to satisfy the assumptions that the authors make (additive structure, exclusion,...) More experiments are required to illustrate the benefits of the method, for example training on real world observational data and evaluating on held-out A/B tests.

Lack of details: it is not clear how to plug the paper's method into RLHF/DPO pipelines. Should we deconfound before training the reward model, then do standard RLHF? Should we modify DPO-style objectives?

### Questions
See weaknesses section.

### Soundness
2

### Presentation
2

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
The paper studies the risk of values of fine-tuning LLMs with historical observational data. It proposes a novel fine-tuning method, called DeconfoundLM, the debias the influence of observed confounders from the reward signals.

The paper provided well-controlled experiments to demonstrate the pitfalls and potentials  in SFT and DPO setting.

### Strengths
- The experimental design is clear and interesting to demonstrate the influence of latent confounders on fine-tuning.
- The proposed DeconfoundLM combined rigorous methods from causal inference to debias the influence from confounders.

### Weaknesses
- The paper is quite compact, and some important contents are not presented in the main body. This makes it difficult to have a detailed review. Please refer to the question part.

### Questions
- How does the proposed DeconfoundLM actually work? Readers may expecting a set of detailed equations with concrete examples. In addition, it would be much better to provide an algorithm box.
- What is the background behind the MIND dataset? and how to compare and interpret the *W*, *C*, and *E* metrics (are they defined?) in Table 1 and 2.

### Soundness
2

### Presentation
2

### Contribution
2

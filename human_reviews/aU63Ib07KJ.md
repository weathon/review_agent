# A Little Help Goes a Long Way: Efficient LLM Training by Leveraging Small LMs

- Decision: Reject
- Scores: 8, 5, 6, 3

## Abstract
A primary challenge in large language model (LLM) development is their onerous pre-training cost. Typically, such pre-training involves optimizing a self-supervised objective (such as next-token prediction) over a large corpus. This paper explores a promising paradigm to improve LLM pre-training efficiency and quality by suitably leveraging a small language model (SLM). In particular, this paradigm relies on an SLM to both (1) provide soft labels as additional training supervision, and (2) select a small subset of valuable ("informative'' and "hard'') training examples. Put together, this enables an effective transfer of the SLM's predictive distribution to the LLM, while prioritizing specific regions of the training data distribution. Empirically, this leads to reduced LLM training time compared to standard training, while improving the overall quality. Theoretically, we develop a statistical framework to systematically study the utility of SLMs in enabling efficient training of high-quality LLMs. In particular, our framework characterizes how the SLM's seemingly low-quality supervision can enhance the training of a much more capable LLM. Furthermore, it also highlights the need for an adaptive utilization of such supervision, by striking a balance between the bias and variance introduced by the SLM-provided soft labels. We corroborate our theoretical framework by improving the pre-training of an LLM with 2.8B parameters by utilizing a smaller LM with 1.5B parameters on the Pile dataset.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper explores the use of small language model (SLM) to provide useful supervision in the early stage of large language model (LLM) training. Specifically, the training of LLM is divided into two stages. In the first stage, it employs KD from an SLM to provide soft labels as additional training supervision. In the second stage, it resorts to the standard MLE training. Though it looks counterintuitive of employing a smaller model to teach a larger one, this paper provides theoretical justifications for such a methodology. Empirical results on language modeling also validate the utility of this approach, which saves 28% training time compared to the standard training.

### Strengths
1. A significant contribution of this paper is its theoretical understanding of knowledge distillation. The bias-variance trade-off in generalization bounds is particularly interesting, which effectively elucidates why even small teacher models can be helpful. However, since I am not an expert in this field, I would need to refer to the opinions of other reviewers on this part.
2. The proposed method shows good experimental results, which outperforms standard training while saving 28% training time.

### Weaknesses
1. The small teacher model employed in the experiments is not small enough (1.5B teacher v.s. 2.8B student ). The authors assume that the small language model is readily available, but it is not always the case. For example, before the release of Llama-405B, the largest Llama model only has 70B parameters. It is unknown whether such a much smaller teacher model can help the training of student model. 
2. This paper lacks discussion and comparison with recent progress of efficient LLM training [1,2,3].

[1] Yao et al., Masked Structural Growth for 2x Faster Language Model Pre-training, https://arxiv.org/pdf/2305.02869  
[2] Li et al., FLM-101B: An Open LLM and How to Train It with $100K Budget, https://arxiv.org/pdf/2309.03852  
[3] Shao et al., Patch-Level Training for Large Language Models, https://arxiv.org/pdf/2407.12665

### Questions
None

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper studies reversed knowledge distillation for language modeling, where a smaller language model provides target probabilities to a larger model.
First, the authors derive two excess surrogate risk bounds under different assumptions: one that bounds this risk based mostly on the distribution of the loss on the training data distribution, and the other based on the expected robustness of the teacher model when increasing past context. They then show that the variance on this robustness can be regularized by the $\omega$ weight of the distillation loss.

They leverage their theoretical analysis of KD to (pre)train large language models in two phases, first using a distillation loss, and then using a cross-entropy loss. Overall, they show that their KD approach outperforms both a vanilla baseline and a one-phase reverse KD baseline in terms of final training performance. They also propose a variant, $SALT_{DS}$, where the second training phase is using data labeled as hard by a heuristic classifier based on the small LM perplexity. They evaluate their models for few-shot performance on classical benchmarks, fine-tune them, and provide a short analysis of performance against sample difficulty along training.

Overall, their approach provides a promising method to accelerate language model training using smaller model distillation as a loss regularization factor.

### Strengths
The authors conduct a very novel and interesting study on knowledge distillation for language modeling in a relatively broad setting. They make reasonable assumptions, and obtain informative bounds on the performance that can be achieved through distillation.
They conduct experiments on relatively large models (2.8B), which confirms the relevance of their approach in a setup that approaches industrial standards.
This paper could motivate further work on distilling small models into larger ones, as it provides a theoretical justification for such approaches, and an example of empirical confirmation of this idea.
The paper is also well-written and clear to follow.

### Weaknesses
The main weakness of this paper lies in the lack of articulation from the theoretical analysis to the experimental work. Several concerns can be raised on this subject:
- The theoretical part is purely based on causal language modeling, while the models are trained with the UL2 procedure, which includes prefix language modeling (which is a variation of causal language modeling), and most importantly the span corruption task, which is not causal. Although it should be possible to extend the theoretical work in that direction, the experiments seem not to accurately reflect the theoretical results.
- Conversely, the paper lacks empirical experiments that would help validate the introduced assumptions, and help quantify some of the factors of the bounds. For instance, empirical estimations based on existing language models could help refine the comments on the dependency of $C$ and $\{V_t\}$ on $T$ and the robustness of models. This could for instance motivate the choice of the size of the SLM during training.
- Experimentation at smaller scale could be done to provide intuition on the benefits obtained when $\omega$ varies, and help illustrate how the mentioned tradeoff can be found.
- More generally, although the authors define several values that should be affected by KD in the theoretical part (e.g. $\xi$), they do not explore the effect of KD on these metrics in the experimental part.

Additionally, I noted several unnatural design choices in the experimental part, which I happen to be more knowledgeable about:
- In Figure 2, which summarizes (and introduces) the benefits of the methods, the authors use top-1 accuracy as the main LM performance metric, while it is more common to show cross-entropy or perplexity in this case. They also report log-perplexity in Table 1, instead of perplexity, which seems like a more natural choice.
- Regarding the model architecture, the chosen vocabulary size seems particularly large (256000) compared to standards in the literature for (a priori) monolingual English datasets (e.g. GPTNeo: 50k; Llama2: 32k; Llama3, which is trained on multilingual data: 150k). The choice of the tokenizer in this case could be important, as it can deeply affect the nature of the modeled distribution $\mathcal{D}$, and thus of the tradeoff offered by $\omega$.
- Regarding the optimization, my opinion is that Adam (or AdamW) would have been a more neutral choice than Adafactor, although it has been used for several models as well. The authors also do not discuss the impact of learning rate when using the KD loss, which is understandable given the cost of running such a hyperparameter search at scale.

In my opinion, another weakness of the paper lies in its lack of transparency and precision on the computational overhead of the proposed method. The authors briefly discuss this overhead, arguing that "As a rule of thumb, a forward pass constitutes 1/4th cost of a training step". This claim lacks support in my opinion, as it depends on implementation and hardware among other things, and as previous works also mention 1/3rd as an estimate (cf. https://arxiv.org/pdf/2203.15556). Moreover, their study does not take the cost of pretraining the SLM into account for the overall training time, or the memory cost of having two models in memory simultaneously. Finally, given the 12% overhead mentioned by the authors and the plot from Figure 2, it is not clear that their approach performs better for pure in-domain language modeling performance when using a FLOPS-related metric on the x-axis. These points are not prohibitive for the method, but should be discussed in the paper.

### Questions
- I found it a bit confusing to distinguish between $V_t$ and $V_N$. Although they model similar aspects in both sections, it would probably be safer to make both more distinguishable.
- In Figure 2, do you display results for SALT or for SALT-DS? Why didn't you display both in this graph?
- Interestingly, the SALT models seem to not drastically outperform the vanilla baseline on in-domain language modeling evaluation (Figure 2, Table 1), but have a noticeable edge in few-shot evaluation and fine-tuning. It could be estimated that they are only 5-10% more data-efficient for pre-training, but are 30% more efficient for evaluation, and in other words that a KD model that reaches a given perplexity gives better evaluation results than a vanilla model at the same perplexity level. Do you have an explanation for this?
- In a way, the SALT approach could be purely justified from an intuitive standpoint : obtain a relatively strong language model quickly through reverse distillation up to the point where the student is slightly below teacher level, and then proceed with regular training. How would you justify precisely the usefulness of the theoretical part in the design of the experimental part?
- Similarly, the SALT-DS approach can also be framed as a (relatively) basic data-filtering technique. Do you think it would be relevant to invoke the curriculum learning literature in that case?

**Typos**
- There is a parenthesis issue with equation 50.

### Soundness
2

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
4

### Summary
This paper proposes an efficient two-stage pre-training method called SALT. In the early stage of LLM pre-training, SALT utilizes an SLM (Small Language Model) as the teacher model for knowledge distillation, while also using the SLM to select challenging yet learnable data to ensure effective knowledge transfer. In the second stage, the training switches to standard self-supervised learning, thereby significantly improving the quality and efficiency of LLM training without increasing pre-training costs. Experimental results demonstrate that the SALT method outperforms standard LLM training on several tasks, while also achieving notable training time savings.

### Strengths
1. The paper leverages a small language model (SLM) as the teacher model for the knowledge distillation process. Instead of using all data for distillation, it selectively transfers knowledge by choosing challenging yet learnable data points, reducing the risk of transferring erroneous knowledge and enhancing the effectiveness and specificity of the knowledge transfer.
2. The 2.8B parameter LLM trained with SALT achieves better performance than a 2.8B LLM trained with standard pre-training on various popular few-shot benchmarks, while requiring only about 70% of the training steps and saving approximately 28% in wall-clock time. SALT models also consistently show substantial performance gains in multiple domains after SFT.

### Weaknesses
1. In Section 4, this method selects samples with high information content and those that are learnable by the SLM through Equation 11, focusing on top-m score sequences. This introduces some uncertainty into the model’s learning process, especially when using an early checkpoint of the SLM, which may further amplify this uncertainty. This can lead to the model learning unreliable information in the early stages, thereby causing a negative impact. As shown in Table 1, SALT_{DS} does not yield performance improvement over the baseline in the early stage, and even exhibits negative effects, indicating that excessive knowledge distillation on uncertain samples may introduce noise rather than effective knowledge, thus affecting model performance.

2. In Table 1, at the early stage, both SALT and SALT_{DS} show limited improvement over the baseline, and even experience noticeable drops in some metrics.

### Questions
1.For Table 1, from the experimental results, it appears that in the Early phase, SALT and SALT_{DS} did not improve performance compared to the Baseline, and even showed a slight decrease in some metrics. However, in the Final phase results, SALT and SALT_{DS} outperform the Baseline. Where do you think this benefit comes from? Additionally, although SALT and SALT_{DS} show some improvement, they do not significantly outperform the Baseline. Have you conducted multiple runs to confirm the stability of these results?
2.Do you plan to test this approach on a larger model to further validate its feasibility?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper addresses the significant computational challenges in training large language models (LLMs) by proposing an approach that leverages smaller language models (SLMs) to improve both training efficiency and model quality. The authors introduce a two-stage training methodology where knowledge distillation from an SLM is used during the early phase of LLM training, followed by standard self-supervised training.

The work makes three main contributions: (1) A theoretical framework analyzing knowledge distillation in language modeling, providing novel risk bounds that explain how even a weaker teacher model can benefit a (stronger) student model; (2) The SALT methodology, which uses SLMs both for knowledge distillation and data selection; and (3) Empirical validation showing that SALT can reduce LLM training time while maintaining or improving performance across various benchmarks.

### Strengths
The authors validate their approach by training a 2.8B parameter model using a 1.5B parameter teacher model on the Pile dataset. They demonstrate improvements in both few-shot learning across multiple downstream tasks. The theoretical analysis provides insights into the bias-variance trade-off in knowledge distillation and justifies why selective knowledge transfer from smaller models can be beneficial. And more pros:

- The paper presents a novel and counter-intuitive approach of using smaller models to improve larger model training (though the "large" is required further clarification).
- The theoretical framework for analyzing knowledge distillation in language modeling is novel.
- The proposed data selection mechanism using SLMs is innovative and well-motivated.
- Addresses a crucial effectiveness improvement of the computational resource utilizations  (training efficiency).

The theoretical framework is in demand at the fast developed field of data cleansing and knowledge distillation for LLMs (mostly driven by empirical results)

### Weaknesses
- The size gap of the 1.5B teacher and 2.8B student remains questionable to define them as a pair of relatively "small" and "large" LMs. If the author do not have sufficient resource to scale up the large model, they need to prove that a smaller 
- The paper doesn't thoroughly explore the impact of different teacher-student size ratios. (Also in Question 1)
- The paper primarily focuses on relatively small models (1.5B and 2.8B parameters) compared to state-of-the-art LLMs. It's unclear if the benefits would scale to much larger models (eg. 7B).

- Limited analysis of computational overhead from the knowledge distillation phase: the authors only provide the resource difference between SALT and the baseline, but not providing the cost of the data selection part.

### Questions
1. How would SALT perform with larger/smaller teacher-student size ratios? For example, would the benefits persist when training a 70B parameter model with a 7B teacher? When the ratio decreases to 1, will the framework still work (eg., similar to bootstrapping).

2. Is there a minimum performance threshold the teacher must achieve for the method to be effective? Is it reasonable to set this as the baseline performance?

### Soundness
2

### Presentation
4

### Contribution
3

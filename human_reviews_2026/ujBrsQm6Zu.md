# NFT: Bridging Supervised Learning and Reinforcement Learning in Math Reasoning

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 4, 8, 2, 4

## Abstract
Reinforcement Learning (RL) has played a central role in the recent surge of LLMs' math abilities by enabling verification-driven training through binary verifier signals. In contrast, Supervised Learning (SL) is rarely considered for such verification-driven training, largely due to its heavy reliance on reference answers and inability to reflect on mistakes. In this work, we challenge the prevailing notion that self-improvement is exclusive to RL and propose Negative-aware Fine-Tuning (NFT) --- a supervised approach that enables LLMs to reflect on their failures and improve autonomously with no external teachers. In online training, instead of throwing away self-generated negative answers, NFT constructs an \textit{implicit} negative policy to model them. This implicit policy is parameterized with the same positive LLM we target to optimize on positive data, enabling direct policy optimization on all LLMs' generations. 
We conduct experiments on 7B and 32B models in math reasoning tasks. Results consistently show that through the additional leverage of negative feedback, NFT significantly improves over SL baselines like rejection fine-tuning, matching, or even surpassing leading RL algorithms like GRPO and DAPO. Furthermore, we demonstrate that NFT and GRPO are actually equivalent in strict-on-policy training, even though they have entirely different theoretical foundations. Our experiments and theoretical findings bridge the gap between SL and RL methods in binary-feedback learning systems.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes Negative-aware Fine-Tuning (NFT), a new supervised learning algorithm designed to bridge the conceptual and empirical gap between SL and RL in math reasoning.

Unlike traditional SL methods that discard incorrect generations, NFT explicitly models negative data by introducing an implicit negative policy parameterized by the same LLM. The paper shows that NFT can optimize both positive and negative samples via a single model, with negligible memory cost. Empirically, experiments on Qwen2.5-Math-7B and 32B show that NFT outperforms or matches leading RL baselines like GRPO and DAPO on math reasoning benchmarks (AIME24/25, AMC23, MATH500, OlympiadBench, Minerva).

### Strengths
1. The theoretical analysis is interesting to bridge the gap between GRPO and NFT.

2. The empirical observations are interesting. The analysis of entropy trends and scaling effects (negative feedback being more important for larger models) adds valuable interpretive depth.

3. The writing is good and easy to follow.

### Weaknesses
1. It is well known that incorporating negative examples can enhance model performance, both in offline and online settings. Therefore, the observed performance improvement from training with negative data is not particularly surprising.

2. I am not sure how the authors define SFT, as the proposed method seems more consistent with an RL training setup. I am not fully convinced that NFT should be categorized as a purely supervised fine-tuning (SFT) algorithm. It would be great if the authors can further elaborate on this.

3. From Figure 1, it appears that the main difference between NFT and GRPO lies in the use of binary labels, which essentially changes how the advantage of responses is computed. 

4. It seems that NFT does not outperform DAPO (even slightly worse). What is the key advantage of NFT compared to GRPO-like algorithms such as DAPO?  What can drive people to use NFT rather than DAPO?

### Questions
Please see the weakness.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper introduces Negative-aware Fine-Tuning (NFT), a supervised learning framework that allows large language models (LLMs) to learn from both correct and incorrect generations without RL. Unlike traditional supervised fine-tuning (or rejection finetuning), NFT constructs an implicit negative policy parameterized by the same model as the positive policy, enabling direct policy optimization on both positive and negative examples. The authors show theoretically that NFT is equivalent to RL under strict on-policy conditions, thereby bridging the conceptual and empirical gap between supervised and RLVR. Experiments on math reasoning tasks with Qwen-2.5 models (7B and 32B) demonstrate that NFT matches or surpasses leading RL methods like GRPO and DAPO while being simpler and more memory-efficient.

### Strengths
1. The paper is well written and clearly structured, making it easy to follow.

2. Theoretical contribution — the equivalence between NFT and GRPO is well derived and provides valuable conceptual clarity between reinforcement learning and supervised fine-tuning.

3. The method is simple, effective, and memory-efficient, avoiding the complexity of traditional RL pipelines.

4. Unlike DPO, NFT does not require explicit positive–negative pairs, which improves data efficiency and implementation flexibility.

5.  Empirical results demonstrate strong performance and good test-time scalability, with well-conducted analysis and ablation studies that support the claims.

### Weaknesses
1. The evaluation is limited to mathematical reasoning, with no experiments on logical or commonsense reasoning tasks, limiting evidence of generalization.

2. While Table 2 shows strong performance compared to other RL-based methods, the paper lacks quantitative analysis of efficiency (e.g., training FLOPs, wall-clock time, or GPU hours). Since NFT claims to be simpler and more efficient than RL-based training, this measurement is important to support the claim.

3. Additional discussion on failure cases would improve interpretability.

### Questions
1. Can NFT generalize beyond mathematical reasoning to less verifiable domains such as commonsense or code generation tasks?

2. Could the authors provide quantitative comparisons of training efficiency (e.g., FLOPs, GPU hours, or convergence speed) against RL-based baselines such as DAPO or GRPO?

3. Could the authors provide generation examples or failure cases?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces Negative-aware Fine-Tuning (NFT), a ``supervised'' method for large language model training with binary verifier feedback. Unlike rejection-based fine-tuning (RFT), which discards incorrect responses, NFT constructs an implicit negative policy that allows the model to learn from both correct and incorrect generations using the same parameterized model. The method defines a token-level objective combining positive and negative log-likelihood terms and incorporates clipping and straight-through gradients for stability. Conceptually, NFT connects supervised and reinforcement learning: under on-policy conditions, its gradients align with those of GRPO, while maintaining a maximum-likelihood foundation. Experiments on 7B and 32B models for math reasoning tasks show that NFT surpasses supervised baselines and achieves performance comparable to leading RL algorithms.

### Strengths
- Relevant and timely topic.
- Comprehensive evaluation across a broad range of reasoning datasets.

### Weaknesses
- The selling point of NFT is that it is an online SFT method. However, being ``online SFT'' does not immediately provide conceptual advantages. If one considers RL as a paradigm of learning under interaction with an environment, then online SFT is simply a special case of RL. The advantages therefore need to be shown practically, for example, through computational or empirical improvements, rather than assumed conceptually. The paper state:
    > Memory Efficiency. NFT is memory-efficient. In practice, we keep only a single model copy in memory. The old policy likelihood πold(a|q) can be pre-computed during data generation.

    However, this should also applies to GRPO (if KL-free) and DAPO. 

- Empirically, NFT does not show notable differences from DAPO, suggesting that the distinction is modest in practice.

### Questions
- What confuses me is the following: what would happen if we simply optimize ($\beta = \pi_\text{old}$ due to rendering issue)

    $$\mathcal{L}(\theta) = \mathbb{E}_{x \sim \mathcal{D}, y \sim \beta} \left[ r(x, y) \left[ -\log\frac{\pi^\theta(y \mid x)}{\beta(y\mid x)} \right]  + (1- r(x, y) ) \left[ \log\frac{\pi^\theta(y \mid x)}{\beta(y\mid x)} \right] \right] $$

    This contrastive objective simply reduces the likelihood of negative samples while increasing that of positive ones.

    It would be helpful if the paper could clarify why Eq. (9) provides a more principled alternative to this naive approach, and in particular why the re-parameterization introduced in Eq. (7) is essential.

    An empirical comparison against this straightforward baseline, including the same practical tricks described in Section 3.3, would also make the claimed advantages of NFT more convincing.


#### minor comments: writing can be made more crisp


- Eq. (9): The notation $(a, q, r)\sim \mathcal{D}$ reads as if the dataset is static. It would help to clarify that $\mathcal{D}$ actually consists of online samples generated by $\pi_\text{old}$, rather than a fixed dataset.

- Algorithm 1: My intuition is that the training process should involve \emph{updating} $\pi_{\text{old}}$, yet this does not appear to be reflected in the pseudocode. If my intuition is correct, clarifying whether (and how) $\pi_{\text{old}}$ is updated across iterations would improve readability.

- The phrase ``a pretrained LLM $\pi_{\text{old}}$'' suggests that $\pi_{\text{old}}$ remains \emph{frozen}, tied to the initial pretrained model. It would be clearer to specify whether $\pi_{\text{old}}$ is only the initialization or if it is also updated online during training.

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
4

### Summary
This paper explores whether supervised fine-tuning approaches can be competetive with reinforcement learning. In particular, the authors identify a means of using negative samples to fine-tune an "implicit negative policy". Their approach successfully leverages both positive and negative samples, which can potentially improve over conventional supervised fine-tuning approaches that just use positive samples. Their results show some improvements over baselines, most notably in AMC23.

### Strengths
- The authors posit a clear research question to explore in their paper: the current approach to fine-tuning may be competitive for self-improvement compared to current approaches to reinforcement learning.
- Motivated by this research question, the authors identify a clear weakness in current apporaches to fine-tuning: it is not possible to use the full dataset when fine-tuning only on positive samples.

### Weaknesses
- While many aspects of the paper are clear, some of the benfits of their negative-aware fine tuning method are not obvious. For example, they choose to use a specific implicit policy approach, but dont compare against an explicit method that conditions on the reward. Also, in their ablations, their practical modifications seem to have a much larger effect on performance than solely using the negative data.
- The motivating research question does not feature prominently in the experiments, besides comparisons to reinforcement learning. That is, while supervised finetuning approaches appear competetive with reinforcement learning for performance improvement, this paper does not clearly demonstrate that this gap is closed by self-improvement from mistakes in negative data.

### Questions
- line 067: throughout the paper, you refer to an "implicit" negative policy. It would help to clarify early what it means for the policy to be implicit, and how this actually affects training the LLM
- Equation 7: Is the key observation being made in equation 7 is essentially the marginalization of $r$?
- Section 3.2: What is the advantage of this implicit formulation relative to training on an LLM that explicitly conditions on the reward? That is, training the positive policy as pi(a|q, r=1) and the negativev policy as pi(a|q, r=0).
- Theorem 3.1: This statement is essentially stating that the critical points are the same, which makes sense. But it is not quite clear what the advantage of this this particular formulation.
- Proposition 4.1: The similarity seems to be driven by the free parameter governing the prompt weighting, $\omega(q)$. Moreover, the differences pointed out in the gradients seem to originate in the practical modifications, rather than the core algorithm. As such, I am not sure whether these takeaways are related to negative-aware fine tuning in particular.
- Table 1: Results are a little inconclusive, where in some cases NFT offers a notable improvement (AMC23), but in others losing to RFT or RL based approaches
- Section 5.3 and Figure 8: I like the fact that the authors investigate the benefits of negative data, but the analysis here seems a little adhoc. For example, couldn't higher entropy be achieved by entropy regularization, which would not necessarily be helpful given that it doesnt use any negative data? The higher entropy could be explained by there being more data fit, which is harder than just fitting positive samples.
- Section 3.3 (practical modifications) and Section 5.4: I also appreciate that the authors investigate the importance of the practical modifications to the base algorithm. It is puzzling that prompt weighting improves accuracy for NFT well beyond RFT Would this seem to suggest that the modifications are more impactful than the negative data?

### Minor Comments
- Some ancillary comments do not seem to make any connection to the contribution, such as: "Later studies (Liu et al., 2025b) suggest removing the std term from Eq. 4"
- Notation is sometimes abused and informal, such as line 150: $\mathcal{D} \sim \pi$
- It is never explicitly defined that "positive answers" are correct answers.
- line 098: minor point but the larger issue in taking a gradient of the RL objective is actually that the parameter $\theta$ is involved in the sampling distribution rather than just in the log probability (as in supervised learning)
- Section 3: the paper would benefit from being more clear about what policy is being under consideration. For example, you refer to $\pi$ and $\pi_old$

### Soundness
2

### Presentation
3

### Contribution
2

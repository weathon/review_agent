# Statistical Rejection Sampling Improves Preference Optimization

- Decision: Accept (poster)
- Scores: 6, 6, 5

## Abstract
Improving the alignment of language models with human preferences remains an active research challenge. Previous approaches have primarily utilized online Reinforcement Learning from Human Feedback (RLHF). Recently, offline methods such as Sequence Likelihood Calibration (SLiC) and Direct Preference Optimization (DPO) have emerged as attractive alternatives, offering improvements in stability and scalability while maintaining competitive performance. SLiC refines its loss function using sequence pairs sampled from a supervised fine-tuned (SFT) policy, while DPO directly optimizes language models based on preference data, foregoing the need for a separate reward model. However, the maximum likelihood estimator (MLE) of the target optimal policy requires labeled preference pairs sampled from that policy. The absence of a reward model in DPO constrains its ability to sample preference pairs from the optimal policy. Meanwhile, SLiC can only sample preference pairs from the SFT policy. To address these limitations, we introduce a novel approach called Statistical Rejection Sampling Optimization (RSO) designed to source preference data from the target optimal policy using rejection sampling, enabling a more accurate estimation of the optimal policy. We also propose a unified framework that enhances the loss functions used in both SLiC and DPO from a preference modeling standpoint. Through extensive experiments across diverse tasks, we demonstrate that RSO consistently outperforms both SLiC and DPO as evaluated by both Large Language Models (LLMs) and human raters.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper introduces an innovative method, Statistical Rejection Sampling Optimization (RSO), for RLHF. RSO employs rejection sampling to extract preference data from the optimal target policy, facilitating a more precise estimation of this policy. Additionally, this paper provides a unified framework for previous approaches, i.e., SLiC and DPO. Experimental outcomes indicate that RSO consistently surpasses existing methods, as evaluated by both large language models and human raters.

### Strengths
1. Presents a novel approach for RLHF using rejection sampling.
2. Unifies prior work, including SLiC and DPO, within a comprehensive review.
3. Provides comprehensive and robust experiments with SOTA, demonstrating clear enhancements of RSO.

### Weaknesses
1. The unifying link between DPO and SLiC appears to be deliberately designed, particularly noticeable in the normalization term, i.e., $\pi_{sft}(y|x)$, in Eq 10.
2. The theoretical conclusion that the policy induced from RSO in Eq 4 is optimal seems questionable. RSO continues to use the proxy reward model to label responses, regardless of whether they're from the pre-trained or learned policy. However, the reward model is learned from $D_p$ and not $D_p^*$ without further updates, which could introduce approximation errors and bias in subsequent labeling and learning steps. In the other word, the policy induced from Eq 4 is not optimal because of the non-optimal reward model.
3. Works with a similar focus on rejection sampling, such as ReST(http://arxiv.org/abs/2308.08998), ALMoST (https://arxiv.org/abs/2305.13735) RAFT (https://arxiv.org/abs/2304.06767), should be involved in comparison.

### Questions
Why do not directly sample responses from $\pi_{r_{\psi}}(y|x)$ (with a reward threshold like ReST) rather than sampling from $\pi_{sft}(y|x)$ with the statistical rejection sampling to approximate the samples from  $\pi_{r_{\psi}}(y|x)$?

### Soundness
3 good

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a novel approach called Statistical Rejection Sampling Optimization (RSO) to improve preference optimization in language models. The authors address the limitations of existing methods by introducing rejection sampling to source preference data from the optimal policy. They also propose a unified framework that enhances the loss functions used in both Sequence Likelihood Calibration (SLiC) and Direct Preference Optimization (DPO). Through extensive experiments, the authors demonstrate that RSO consistently outperforms SLiC and DPO in terms of both Large Language Models (LLMs) and human raters' evaluations.

### Strengths
The authors provide a well-structured review of relevant literature, highlighting the limitations of existing methods and positioning their work in the context of prior research.

### Weaknesses
1. The core idea of utilizing rejection sampling to filter the response feels somewhat trivial 

2. Why is there a significant improvement in rso-sample-rank compared to sft-sample-rank on Reddit TL;DR, while the improvement is not as noticeable on AnthropicHH? What could be the potential reasons behind this?

3. There might be flaws in the design of the experimental part: the reward for training and testing is the same. Although there are evaluations from GPT/human, generally speaking, these are not very convincing.  You can refer to [1, 2] for Synthetic Data Setup, Recalibration, gold reward, and proxy reward to have a fair comparison.

[1] Let’s Verify Step by Step

[2] Scaling Laws for Reward Model Overoptimization

### Questions
Please refer to the weakness section

### Soundness
2 fair

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a new method to improving language model alignment using Statistical Rejection Sampling Optimization (RSO). Their method allows for more accurate estimation of the optimal policy, leading to better alignment with human preferences. Empirically, the authors demonstrate the effectiveness of RSO on two tasks: Reddit TL;DR summarization and AnthropicHH dataset. They further show that RSO works well on cross-task generalization from Reddit TL;DR. Lastly, the authors evaluate RSO using three different approaches: Reward Model, AutoSxS, and Human Evaluation. The results show that RSO variants outperform DPO and SLiC variants on both tasks.

### Strengths
Strengths:

- This paper introduces a new approach for improving language models alignment by exploiting rejection sampling to allow for a more precise estimation of the optimal policy. To the best of my knowledge, this proposed method is novel and well-founded. Moreover, it also include exsisting methods (e.g., DPO) as special cases.

- The paper is mostly well-written and the illustration figures are helpful for the undertsanding of this work.

- The empirical evaluation is thorough and demonstrate the effectiveness of the proposed method. Most of the claims / arguments are well-supported.

### Weaknesses
Weakness:

- There are still rooms for improving the presentation. For example, it would be better if the authors can explain a bit more on $\rho_{\psi}(x,y,y_b)$ when first introducing it. The authors could provide some examples or details about what would be an instantiation of $\rho_{\psi}(x,y,y_b)$.

- Step 3 in section 3.2 might be overly time-consuming and computationally expensive. Especially the set $M = \min\\{m | m\pi_{sft}(y|x) \geq \pi_{r_{\psi}}(y|x) \text{ for all } y \notin \mathcal{Y}\\}$ is expensive to compute. Rejection sampling is also slow.

- The results in Figure 3 (b) is not statistically signifcant and the confidence intervals are overlapping.

### Questions
1. The paper focuses on improving language models by aligning them with human preferences. I wonder how might this approach be adapted to address issues of bias and fairness in language models? Could the authors provide some additional discussions on this?

2. How does the proposed method compare against other methods, such as DPO, in terms computational efficiency?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

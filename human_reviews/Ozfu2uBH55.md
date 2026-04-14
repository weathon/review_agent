# Preference Optimization with Multi-Sample Comparisons

- Decision: Reject
- Scores: 3, 6, 6, 5

## Abstract
Recent advancements in generative models, particularly large language models (LLMs) and diffusion models, have been driven by extensive pretraining on large datasets followed by post-training. However, current post-training methods such as reinforcement learning from human feedback (RLHF) and direct alignment from preference methods (DAP) primarily utilize single-sample comparisons. These approaches often fail to capture critical characteristics such as generative diversity and bias, which are more accurately assessed through multiple samples. To address these limitations, we introduce a novel approach that extends post-training to include multi-sample comparisons. To achieve this, we propose Multi-sample Direct Preference Optimization (mDPO) and Multi-sample Identity Preference Optimization (mIPO). These methods improve traditional DAP methods by focusing on group-wise characteristics. Empirically, we demonstrate that multi-sample comparison is more effective in optimizing collective characteristics~(e.g., diversity and bias) for generative models than single-sample comparison. Additionally, our findings suggest that multi-sample comparisons provide a more robust optimization framework, particularly for dataset with label noise.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The authors propose two new methods: Multi-sample Direct Preference Optimization and Multi-sample Identity Preference Optimization. 
These approaches assess group-wise characteristics by using multiple samples during comparison, which improves optimization for collective properties such as diversity and helps mitigate label noise.
The paper's empirical results show that these multi-sample methods enhance performance in optimizing diverse and unbiased outputs compared to single-sample approaches. This is particularly important for applications where the model's output variability and accuracy are crucial.

### Strengths
1.  By targeting issues like label noise and output variability, the paper’s methods are designed to improve model performance in practical settings where handling data imperfections and ensuring diversity are essential for reliable outcomes. This is especially relevant in real-world applications.

2.  The methods proposed in this paper show promise for enhancing generative model capabilities in downstream tasks, such as improving the diversity of generated genres and reducing bias in outputs. These improvements could be particularly useful in applications like content creation and ethical A.

3. The paper demonstrates through empirical results that the multi-sample approach improves over traditional methods when optimizing group-wise characteristics like diversity. This validates the utility of the proposed methods in a variety of scenarios.

### Weaknesses
1. The core idea seems to be more of a refined definition for preference pairs rather than a groundbreaking concept. 

2. The settings and definition of mDPO are quite confusing and informal.  There are other works on multi-sample comparisons that are more well-defined and may provide stronger theoretical foundations [1].

3. The paper’s comparison with state-of-the-art (SOTA) iterative algorithms [2] is rather limited, particularly on standard benchmarks such as MT-Bench or Alpaca-Eval. A broader evaluation would provide a clearer understanding of the methods' effectiveness against existing approaches

4. While the paper focuses on multi-sample comparisons, there are already other multi-sample methods that may be better established or more rigorously defined. This raises questions about the comparative novelty and impact of the proposed approach.

[1] Towards Efficient Exact Optimization of Language Model Alignment

[2] RLHF Workflow: From Reward Modeling to Online RLHF

### Questions
The definition of “group” is very confusing. How can you define the probability space of the “group”? Can you guarantee that the event space is a $\sigma$-group?

If I define the preference by your so-called “groups”, what’s the differences between mDPO and vanilla one?

Can the proposed algorithm provide stronger results in standard benchmarks?

### Soundness
2

### Presentation
2

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
The authors identify that the widely applied single-sample comparisons often fail to capture characteristics like diversity. Thus, they introduce a new approach to perform alignment with multi-sample comparisons. Based on this idea, two multi-sample variants (mDPO and mIPO) are proposed. Through experiments on various applications, the method is shown to improve the diversity, fairness and robustness to label noise.

### Strengths
1. **Clear motivation**: The authors identify an important problem in current alignment paradigms, and the running example of generating random numbers is intuitive and helps the reader understand the problem well.

2. **Thorough experiments**: The authors use several use cases to demonstrate the effectiveness of the proposed methods. The results are intuitive and convincing.

### Weaknesses
1. **Potentially higher dataset requirements**: It seems that in the datasets for multi-sample comparisons, one prompt should have k chosen-rejected pairs of responses. This makes many readily available datasets not applicable, and the datasets need to be adjusted for different k.

2. **Lack of discussion on k**: I think the selection of k deserves more explanation. Under what circumstances would a larger k benefit?

3. **Comparison on general tasks**: It is clear that under scenarios where diversity and bias are concerns, the proposed approach performs better than the single-sample variants. However, in a general alignment setting, when shall we consider multi-sample over single-sample? Will there be a case where single-sample is actually better? Especially considering that multi-sample incurs higher computational and data collection cost.

### Questions
1. Intuitively, I think with larger k, the performance would be improved due to the smaller variance. However, that does not always seem to be the case, as in Table 3. Could you provide some comments about this phenomenon and the selection of k in general?

2. Do you have a runtime comparison for the multi-sample and single-sample variants? Certainly, the multi-sample one is expected to be slower, but it would be nice to have it quantified.

### Soundness
4

### Presentation
4

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
This paper advances the field of Reinforcement Learning from Human Feedback (RLHF) by exploring more nuanced preferences beyond traditional binary singleton preferences. Specifically, it investigates scenarios where preferences involve comparisons between two distributions. For example, if we want an LLM to generate random numbers between 0 and 10, singleton numbers 3 and 5 are equally preferred but a distribution with only 3s is less preferred as compared to a distribution with multiple numbers from 0 to 10 which also includes 5. 

The primary contribution of this work lies in the adaptation of two widely used alignment techniques—DPO (Differentiable Preference Optimization) and IPO (Inverse Preference Optimization)—to accommodate binary feedback between distributions. Additionally, the paper provides a theoretical analysis for gradient estimation within this new DPO/IPO framework. The authors present various experiments demonstrating the advantages of using distribution comparisons over singleton comparisons.

### Strengths
1. The motivation for incorporating more nuanced human feedback is compelling.
2. The paper is well-written, with a clear and organized presentation.
3. Comprehensive experiments demonstrate the effectiveness of the proposed methodology across diverse applications, including random number generation, fair image generation, varied writing styles, and iterative LLM enhancement.

### Weaknesses
1. While the problem is well-motivated, the proposed approach appears somewhat straightforward and lacks substantial innovation.
2. The methodology relies on the assumption that humans can easily distinguish between different distributions of items. However, in practice, it may be more cognitively challenging for individuals to differentiate between two distributions, and gathering such data could be more costly.

### Questions
1. Are all the responses used in generating distributions for mDPO/mIPO also used in training DPO/IPO? If not then it is not clear if the improvement is coming from using the modified loss function versus simply having more data.
2. mDPO/mIPO comparison: do you have any intuition on when it would be more beneficial to use mDPO vs mIPO?

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper proposes multi-sample versions of DPO and IPO, to be used when one set of samples is preferred over another. The proposed mIPO is an unbiased estimator while mDPO is biased. Experiments show that mDPO and mIPO outperform their single-sample counterparts and SFT in inducing 1. more uniform random number generation, 2. more balanced race and gender generations, and 3. more diversity in fiction genres. Moreover, the paper shows that mDPO and mIPO are more robust than DPO and IPO in the presence of preference label noise.

### Strengths
1. The paper is well-motivated and considers a novel use case for preference learning.
2. The method is clearly presented, and the theoretical analysis is clear and informative.

### Weaknesses
1. The experimental setup could be more clearly described, particularly with respect to the data seen for each method (e.g., is the number of pairings fixed, or the number of outputs or per-sample pairwise comparisons?). For instance, how are the datasets constructed for the single-sample and multi-sample methods in the label noise experiments? What data does SFT see in the uniformity/debiasing/diversity experiments? All the samples from the preferred set, or something else? Without an understanding of these details, it is hard to ascertain what to take away from the experiments / if baselines are fair comparisons.
2. Using mDPO and mIPO sounds interesting and promising for aligning towards distributional objectives, but there already exist other methods that aim to achieve the same goals, e.g., [Khalifa et al. 2020](https://arxiv.org/abs/2012.11635) (or [Korbak et al. 2022](https://arxiv.org/abs/2112.00791) for conditional generation), [Wu et al. 2022](https://arxiv.org/pdf/2209.06970). The paper would be strengthened if it compared with such methods and more directly discussed distributional control.
3. DPO seems like a strange baseline to consider for the uniformity/debiasing/diversity experiments in the first place; instead, it would be helpful to consider other simple methods that use the same data as mDPO and mIPO, such as maximizing the likelihood of the preferred set while minimizing the likelihood of the dispreferred set, or that target improving diversity / balance, such as training with samples weighted by the inverse of their probability under the model. A larger discussion of such approaches in the related work would also be helpful.

### Questions
1. Can the authors clarify the experimental setups, namely with respect to the data seen by the different algorithms?
2. Could the authors compare the proposed methods to other methods for aligning towards distributional goals, e.g., when one might practically consider one over another?
3. Could the authors also compare with a baselines for debiasing or diversity?

I would be happy to raise my score if these concerns are adequately addressed.

### Soundness
3

### Presentation
3

### Contribution
3

# Generalized Inference Time Unlearning --- Effective for A Fraction of the Cost

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4

## Abstract
Large Language Models (LLMs) can memorize and regurgitate sensitive training data, creating significant privacy and safety risks. While existing unlearning aim to address these risks, current methods are often computationally prohibitive and/or significantly degrade model utility. We introduce a framework for Inference-Time Unlearning, a new paradigm that steers an LLM's output at inference time using small secondary models, without altering the base model's weights. Through extensive experiments with LLMs we demonstrate that our method is highly effective at removing targeted verbatim and semantic knowledge, is orders of magnitude more computationally efficient---through profiling of more than 1,200 models---than traditional approaches, and fully preserves the base model's general capabilities. We then explore efficacy in unlearning visual semantics in generative image models and find similar evidence of effectiveness. Collectively, the framework offers a practical, scalable, and low-cost solution for selective forgetting, enabling more responsible and adaptable model deployment. All code to reproduce this work is available at the following anonymous link: https://anonymous.4open.science/r/inference-time-unlearning-iclr2026/

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposed an inference-based unlearning method, reducing the overhead of fine-tuning the LLM. The method based on divergence decoding, which two distributions are built to guide the LLM token decoding. The experimental results shows its effectiveness.

### Strengths
1. The paper proposed time-based unlearning benchmark, enabling more diverse evaluation. 

2. The proposed method is applicable to different data types, including text and image.

### Weaknesses
1. As pointed by the limitation 2 in the paper, as this method does not really change the internal representation of a model, it is more like guardrails instead of unlearning. Moreover, could the authors provide a concrete scenario how the proposed method would be used?

2. The inference time cost analysis miss the important factor: run time latency, as the proposed method will have to recompute the distribution every token, could the authors provide the total runtime before and after applying the proposed unlearning approach? Unlike other approaches, like GA, it could be costly during the fine-tuning but it should work as is during the inference.

### Questions
1. I wonder what is the connection of the proposed method to the LLM watermarking, which is also tiling the distribution in a certain way.

2. For time-based unlearning benchmark, what is the difference if we simply categorize those to-be-forgotten timeline into forget set and remaining go to retain set?

3. What is the effect of the ratio of model size of p and q (not absolute size)?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces Divergence Decoding (DD), an inference-time technique for unlearning. The technique involves finetuning 2 smaller models: 1 on a "retain set" that includes knowledge that should not be forgotten, and 1 on the "forget" set contains information on the concept that should be forgotten. The logits of the base model are then adjusted by the difference in logits of the forget and retain models. The authors propose a linear method of DD, as well as a rank-based method. They also introduce a new time-unlearning benchmark

### Strengths
- The problem of Unlearning is important and timely 
- Inference-time unlearning techniques, such as the proposed approach, are needed as finetuning is costly and prone to harming generalizability. 
- The technique of using two proxy models to adjust the logts of the base model is clever
- The proposed approach is somewhat backed by theory, as the authors relate it to Product of Experts and importance sampling.

### Weaknesses
- The experimental results lack error bars / confidence intervals. 
- Unclear whether DD outperforms NP on verbatim knowledge of the forget set on MUSE (figure 1)
- The biggest weakness in my eyes is the time-unlearning benchmark. I may be missing something, but frankly I do not see how this fits in with the rest of the paper. No unlearning methods, DD or otherwise, are evaluated on the proposed Time Unlearning dataset in the main paper. I also don't understand how this dataset relates to unlearning, since it focuses on lookahead bias rather than data removal per se. The paper would be much stronger if it removed this dataset and instead performed a more comprehensive analysis on standard unlearning benchmarks.

### Questions
- How does the Time Unlearn dataset relate to Unlearning? Why are no Unlearning methods evaluated on it? 
- I don't understand Figure 3. Can you explain in more detail what each axis represents? Why is a flat line desirable? Is DD performing poorly compared to NPO and SimNPO?

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
This paper introduces a new paradigm, namely Inference-Time Unlearning (ITU), which aims to remove unwanted knowledge generation from large language models without modifying their weights. The approach trains two much smaller auxiliary models — a forget expert fine-tuned on the forget set, and a retain expert fine-tuned on the retain set. During inference, these experts adjust the logits of the base model using a method called Divergence Decoding (DD), which steers the output distribution away from tokens upweighted by the forget expert and toward those upweighted by the retain expert. Two variants are proposed: a linear logit adjustment and a rank-based token suppression. This inference-time mechanism achieves effects that are similar to “unlearning” of targeted content while being orders of magnitude more computationally efficient than gradient-based methods and preserving general utility across MUSE and TOFU benchmarks.

### Strengths
- **New paradigm on LLM output steering.** The proposed method aims to efficiently approximate the data distribution of a target large model ($\hat Q$) using two introduced much smaller fine-tuned experts ($p,q$) plus the original model ($P$), achieving substantial computational savings while maintaining controllability.


- **The formulation is concise.** The derivation connecting Divergence Decoding to Product of Experts and importance sampling is mathematically coherent and provides interpretability. 

- **Extensive empirical verification.** The author demonstrates the effectiveness of proposed methods on multiple benchmarks and visual distributions: MUSE, TOFU, image distribution and the introduced “Time-unlearning benchmark”

### Weaknesses
- **Dependence on auxiliary models.** The method assumes well-trained retain and forget experts but gives little detail on how to build them when data are limited or noisy, leaving room for bias or domain overlap artifacts.

- **Evaluation gaps in image generative domains.** The image unlearning experiment relies solely on FID scores, which are inadequate for assessing semantic forgetting. The work would be beneficial to employ automated analysis verifying that specific visual concepts are unlearned while others remain intact, or include visual examples for qualitative evaluation. 

- **Conceptual ambiguity.** The method steers outputs but does not alter the model’s internal representations or parameters. Hence, it does not “remove” knowledge but suppresses its expression on targeted contents. This is better categorized as inference-time filtering or redirection, not unlearning in the formal sense.

- **Missing relevant work discussion.** Training-free steering-based approach for controlled model generation is not a completely new paradigm. While this work focuses on logit space steering, prior work on activation space steering [1][2] needs to be discussed, and potentially, compared efficiency, as steering-based methods all claim benefits on computational efficiency and utility preservation.

[1] Steering Language Models with Activation Engineering

[2] Programming Refusal with Conditional Activation Steering

### Questions
- **Clarification on experimental setup.** Are there any specific retain/forget set curation involved?

- Could you explicitly explain how proposed method handle non-targeted content generation?

- See weaknesses for other questions/suggestions

### Soundness
3

### Presentation
3

### Contribution
3

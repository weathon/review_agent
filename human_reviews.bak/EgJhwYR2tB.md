# Speculative Knowledge Distillation: Bridging the Teacher-Student Gap Through Interleaved Sampling

- Decision: Accept (Poster)
- Scores: 6, 6, 6

## Abstract
Recent advances in knowledge distillation (KD) have enabled smaller student models to approach the performance of larger teacher models. However, popular methods such as supervised KD and on-policy KD, are adversely impacted by the knowledge gaps between teacher-student in practical scenarios. Supervised KD suffers from a distribution mismatch between training with a static dataset and inference over final student-generated outputs. Conversely, on-policy KD, which uses student-generated samples for training, can suffer from low-quality training examples with which teacher models are not familiar, resulting in inaccurate teacher feedback. To address these limitations, we introduce Speculative Knowledge Distillation (SKD), a novel approach that leverages cooperation between student and teacher models to generate high-quality training data on-the-fly while aligning with the student's inference-time distribution. In SKD, the student proposes tokens, and the teacher replaces poorly ranked ones based on its own distribution, transferring high-quality knowledge adaptively. We evaluate SKD on various text generation tasks, including translation, summarization, math, and instruction following, and show that SKD consistently outperforms existing KD methods across different domains, data sizes, and model initialization strategies.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The method seems very simple and, to some extent, sounds like an interpolation between two distillation strategies. Specifically, in Supervised KD the next-token distribution of student and teacher are trained to be similar on a fixed text, while in On-policy it is done on student’s generated text. The method thus proposes to evaluate the tokens of the student and see if it’s within TopK of the teacher’s, if not, the generated tokens of the student is discarded and the rest is followed with teacher’s tokens. But if the generated token satisfies the TopK check, we continue with student’s tokens.


The idea is clearly motivated in the paper and makes sense. The quantitative results also seem promising (across tasks the method is mostly outperforming prior work by a good margin). Especially the method is consistently outperfoming On Policy KD, which supports the proposed idea of switching to teacher tokens if the student tokens are bad.

### Strengths
**(a)** I think the paper is extremely well written. Especially for someone who might not be fully familiar with how Knowledge Distillation is applied for the LLMs, I really appreciate how clearly the authors explain prior work and how they relate to each other. With only the first pass, I was able to grasp the ideas across related work and the main contribution of the work.


**(b)** I think the simplicity of method is one of its advantages. Simply checking the student token's quality, based on whether it's within TopK of the teacher's seems simple and intuitive. (see also Question a).


**(c)** The quantitative results, especially compared to On Policy KD, seem promising (please see the disclaimer in Weaknesses section).

### Weaknesses
**(a)** I like the idea of evaluating the student’s suggestions by the teacher and selecting the Top-K based on the teacher’s probabilities. However, what concerns me is the computational cost of the method. Given that this is the main contribution of the work, I think it has to be more clearly studied. Specifically, what I really like to see is how often the line 5 in Algorithm 1 is triggered. In Lines 212-216 it is essentially claimed that this trigger rate is decreased over training time, I think it has to be evidenced with a figure (x axis being the training steps and y axis being the trigger rate). In a similar sense, I think it should be reported what is the final rate (i.e how likely it is for the student’s tokens to be in TopK of the teacher’s after the distillation).


**(b)** Moreover, I think the value for K should be studied. Essentially for high Ks the method should get somewhat similar to On Policy KD and for lower ones should be closer to Supervised KD? So there should be a sweet spot. Also I’m wondering if the value for K needs to be adjusted based on the training step? In other words, maybe higher or lower Ks would be better at the early or later stages of distillation? Some ablation experiments (with different numbers for K) could be helpful in addressing this.

------ Minor issues ------

**(c)** In Line 119, it is stated that “the temperature t `introduces` randomness”. I think this is not entirely correct as it’s the sampling that introduces randomness. The temperature adjusts the distribution so it `adjusts` the randomness (as it is also stated correctly in the following sentence).
 
**(d)** In Line 184, “Correct previous mistakes”,  it is not clear what `mistake` is referring to. Is a mistake a poorly generated Token? What does `correcting` refer to?

**(e)** Line 250: “Since our assumption `is` that”

**(f)** In the pseudo-code (Algorithm 1), some variables, such as alpha in line 3, are not defined.

**(g)** This is not really an issue, rather a suggestion. A tiny figure to explain concepts defined in Section 3.1 would be very helpful. Basically a 1-row figure divided into maybe 4-5 sub-columns, with every column showing how each of the prior work applies the loss at token-level. Maybe something like Figure 1 in https://arxiv.org/pdf/2104.09044 I think Figure (2) can be refactored a bit so that it also includes other prior work. 




------ Disclaimer ------

I’m not very familiar with KD approaches for LLM, and therefore it is possible that I haven't noticed a missing baseline. I would really appreciate it if other reviewers could also verify if the set baselines is complete.

### Questions
**(a)** It is not 100% clear to me why would one replace a `bad token` with a token sampled from teacher's distribution. Could the authors give some intuition of why not re-attempt the sampling from student's distribution (line 4 in Algorithm 1) to replace the last token but under the condition that it's within TopK of the teacher? In otherwords, maybe the `bad` token sampled from student's distribution can be replaced with a `better` token that is sampled from the same distribution, rather than one that comes from the teacher's. If it is within the computational budget, an experiment to demonstrate this could also help prove the point.


**(b)** If I understood correctly, your method does not add any computational overhead to the distillation process, as it is a simple probability check and a token replacement. If so, please assert this both in rebuttal and also in the paper. Otherwise, a clear discussion on computational overhead is missing.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors present Speculative Knowledge Distillation (SKD), a new KD method for effective knowledge transfer from large teacher models to smaller student models. SKD combines on-policy sampling with speculative decoding, allowing students to propose tokens while the teacher verifies and refines them, addressing issues in traditional KD methods like distribution mismatches and low-quality samples.

The authors evaluate SKD across tasks including translation, summarization, math, and instruction following, consistently outperforming baseline methods across different domains, data sizes, and initialization strategies. SKD shows robust performance in both task-specific and task-agnostic distillation, with strong generalization on complex math tasks and advantages in low-data scenarios.

Contributions:
1. Introducing SKD, which integrates on-policy sampling and speculative decoding for adaptive, high-quality knowledge transfer.
2. Demonstrating SKD’s superior performance across varied tasks, outperforming supervised and on-policy KD.
3. Validating SKD’s adaptability and efficiency, especially in diverse model initializations and low-data settings, without requiring additional supervised fine-tuning.

### Strengths
1. The paper introduces Speculative Knowledge Distillation (SKD) to address the distribution mismatch between training and inference in traditional knowledge distillation. By combining on-policy sampling with speculative decoding, SKD enables efficient knowledge transfer, allowing the student model to autonomously generate and refine its samples. 

2. The authors validate SKD across various task scenarios, including translation, summarization, arithmetic reasoning, and instruction following, covering both task-specific and task-agnostic applications. Through comparisons with multiple baseline methods (e.g., supervised KD, on-policy KD, and ImitKD), the experimental setup is comprehensive and broadly applicable, providing certain support for SKD’s performance.

3. SKD’s design builds on imitation learning theory, lending theoretical validity to the approach. Additionally, the paper includes detailed implementation steps and hyperparameter settings, providing actionable guidance for future research.

### Weaknesses
1. Impact of Early Low-Quality Samples Lacks In-Depth Analysis: The SKD framework uses interleaved sampling and speculative decoding to handle low-quality samples generated by the student model during its initial training stages. However, the authors do not analyze how these low-quality samples may impact complex tasks, such as mathematical reasoning. Early low-quality samples could affect the stability of teacher feedback, slow convergence, and compromise overall stability. The authors could consider adding an analysis of model convergence behavior across tasks of varying complexity in their experiments to gain a more comprehensive understanding of SKD’s convergence characteristics in complex scenarios.

2. Task-Specific Adaptability: SKD maintains overall sampling quality through dynamic adjustments of token-level sample quality and adaptive switching between teacher and student sampling. However, the authors do not explore how to customize these mechanisms for different task types or complexities.

3. Effect of Model Initialization Lacks Discussion: While the authors examine SKD’s performance under different initialization conditions, such as instruction tuning and supervised fine-tuning, they do not discuss how initial model quality affects SKD’s convergence efficiency and stability. The authors might consider investigating convergence behavior under various initialization qualities, particularly in low-quality scenarios, by dynamically adjusting the proportion of teacher samples to stabilize the training process.

### Questions
Question 1: The paper does not analyze how early low-quality samples generated by the student model may impact complex tasks, such as mathematical reasoning. How do the authors believe these early samples could affect the stability of teacher feedback and convergence rates?

Question 2: Although the authors' framework incorporates dynamic adjustments and adaptive switching between teacher and student sampling, how do they envision customizing these mechanisms for different task types or complexities?

Question 3: While the authors examine SKD’s performance under various initialization conditions, how does initial model quality affect convergence efficiency and stability?

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposed a new knowledge distillation method called Speculative KD (SKD), inspired by speculated decoding, that addresses the drawbacks from supervised KD and on-policy, while keeping the advantages of both worlds. Specifically, to circumvent the issue of supervised KD, SKD enforces the teacher model to evaluate the output of the student model. To alleviate the "cold-start" OOD problem of on-policy KD, SKD filters out intermediate tokens that the teacher is unlikely to generate, and re-samples from the teacher.

### Strengths
- The motivation of the paper is convincing. Supervised KD is known to suffer from distribution shift, while on-polich KD suffers from the OOD problem.
- The organization is well and the paper is carefully written. The problem setting is well formulated.
- All-around empirical results pretty much validates the effectiveness of the method. The authors conducted experiments on four tasks and one task agnostic distillation.

### Weaknesses
- Although it is relatively straightforward to agree with the authors that the on-policy KD suffer from the OOD issue, is there a way to show how severe the issue is? For example, would it be possible to quantitatively evaluate?
- It is a bit hard to understand the task-agnostic distillation setting. I see the training and test sets are both based on the math reasoning datasets. Where does the shifts come from?
- In table 2, it is obvious that the improvements are rather marginal. In Table 1, it seems the proposed method falls short behind another baseline. Would you explain?
- Significant tests for Table 1 & 2 could further strengthen the results.

### Questions
- Is there a way to show how severe the OOD issue of on-policy KD is? For example, would it be possible to quantitatively evaluate?
- Why does the setting proposed in Section 5.2 a "task-agnostic" one?

### Soundness
3

### Presentation
4

### Contribution
3

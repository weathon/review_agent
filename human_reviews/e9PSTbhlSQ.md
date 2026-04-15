# Learning within Sleeping: A Brain-Inspired Bayesian Continual Learning Framework

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 3, 5, 5

## Abstract
Bayesian-based methods have emerged as an effective approach in continual learning (CL) to solve catastrophic forgetting. One prominent example is Variational Continual Learning (VCL), which demonstrates remarkable performance in task-incremental learning (task-IL). However, class-incremental learning (class-IL) is still challenging for the VCL, and the reasons behind this limitation remain unclear. Relying on the sophisticated neural mechanisms, particularly the mechanism of memory consolidation during sleep, the human brain possesses inherent advantages for both task-IL and class-IL scenarios, which provides insight for a brain-inspired VCL. To identify the reasons for the inadequacy of VCL in class-IL, we first conduct a comprehensive theoretical analysis of VCL. On this basis, we propose a novel bayesian framework named as Learning within Sleeping (LwS) by leveraging the memory consolidation. By simulating the distribution integration and generalization observed during memory consolidation in sleep, LwS achieves the idea of prior knowledge guiding posterior knowledge learning as in VCL. In addition, with emulating the process of memory reactivation of the brain, LwS imposes a constraint on feature invariance to mitigate forgetting learned knowledge. Experimental results demonstrate that LwS outperforms both Bayesian and non-Bayesian methods in task-IL and class-IL scenarios, which further indicates the effectiveness of incorporating brain mechanisms on designing novel approaches for CL.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper introduces a method of continual learning inspired by a human brain mechanism, namely, memory consolidation during sleeping. The proposed method was compared with the original VCL and other existing methods in the literature over Split MNIST, Split CIFAR-10, and Split Tiny-ImageNet datasets. The proposed method showed superior performance, especially in the class incremental learning scenario.

### Strengths
The introduction and the relation works sections are well written and clearly describe the goal of the work.

The inspiration from a human brain mechanism is exciting, and the proposed learning within sleeping (LwS) showed remarkable performance improvements.

### Weaknesses
While the idea is interesting and meaningful, the proposed method seems to combine the existing methods in the literature rather than a new one. In learning while sleeping, the memory reactivation is similar to the replay-based methods, and the distribution generalization is similar to the regularization-based methods. 

From a neuroscience perspective to this reviewer’s knowledge, learning during sleep is more related to the stimulus given before sleeping (memory reactivation); thus, it has nothing to do with a new stimulation (no sample from a new task is given). In this regard, this reviewer believes that the proposed LwS doesn’t follow the neural mechanism. 

Many typos in the context and the equations make it challenging to read and understand the paper. Especially for the equations, please carefully check the super/subscripts.

### Questions
Check the Weaknesses above.

For the consistency, the notations in Eq. (1) should be corrected as follows: (left) $f_{\theta}(g_{W}(x))$ -> $f_{\theta, W}(x)$ or $g_{W}(h_{\theta}(x))$; (right) $f_{\theta}(g_{W}(x_{i}))$ -> $f_{\theta, W}(x_{i})$ or $g_{W}(h_{\theta}(x_{i}))$.

Since the notations are unclear, it isn't very clear for Hypothesis 1, Theorem 1, and the Proof. How are the notations of $(\theta_{t+1}^{*’}, W_{t+1}^{*’})$ and $(\theta_{t+1}^{*}, W_{t+1}^{*})$ different? Especially the last sentence in Proof needs checking carefully, “~, it follows that $H~~~ <= H~~~~$.” 

In Algorithm 1, $D_{0}$ is not defined.

The authors also need to provide the results of VCL and LwS on other datasets in task-IL and class-IL scenarios.

In Figure 2, the markers in the graphs and the legends do not match.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper investigates the reason why variational continual learning does not work well for the class-incremental learning and then proposes two techniques to improve the performance including replay and knowledge distillation. The resulting method indeed shows enhanced performance on the class-incremental learning scenario from the results demonstrated in the paper.

### Strengths
The mathematical formulation seems to be correct.
The effectiveness of replay and feature distillation are again demonstrated.

### Weaknesses
Motivation of this paper is not clear. The paper extends on variational continual learning to the class-incremental learning scenario but it is unclear what are the particular benefits of extending over VCL rather than other approaches. 

It is nice that the paper writes the problem and solution in a clear mathematical form but what the conclusions and solutions can be easily explained with natural language. The mathematical proof seems unnecessary, which can be moved to supplemental. 

The novelty of the proposed method is very limited. The resulting solution is replay (similar to all replay methods like GEM, DER, DER++, GSS, [1], [2], [3], etc.) and perform distillation (similar to LwF) on the feature extraction part. But the solution is widely recognized as useful and proved in many prior works. I cannot identify any other contributions. 

The comparison is not convincing. Results are not compared to more recent works such as DER and [1][2][3] so it is not convincing that the result is significant at the current time. 

[1] Rishabh Tiwari et al. GCR: Gradient Coreset based Replay Buffer Selection for Continual Learning. CVPR 2022
[2] Elahe Arani et al. Learning Fast, Learning Slow: A General Continual Learning Method based on Complementary Learning System. ICLR 2022
[3] Da-Wei Zhou et al. A Model or 603 Exemplars: Towards Memory-Efficient Class-Incremental Learning. ICLR 2023

### Questions
No questions.

### Soundness
2 fair

### Presentation
1 poor

### Contribution
1 poor

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposed a new Bayesian continual learning method inspired by human memory consolidation process during sleep. The idea is interesting and novel, but in implementation it seems the method still trying to find a balance between the old and new data/features on a unified model, which human brain may not necessarily working in this way to process new memory. Results indicate this method is effective in a range of benchmarking experiments but fails to outperform some existing algorithms.

### Strengths
1. The idea of memory consolidation is novel and it is interesting to dig this further. A strong and convincing point of the justification in the paper is that model transformation by new data is a primary reason that cause the forgetting issue. 

2. Split the model into FE and FC seems a good idea to make this complicated problem into simpler and easier ones.

3. The way of presentation is good, easy to follow.

### Weaknesses
1. It seems this work was somehow rushed, especially the results section. I found a number of obvious typos, mainly spelling error. It would be good to at least carefully proofread few times before submission.

2. The idea and the inspiring source is novel, but unfortunately when it comes to implementation, the proposed method is still trying to find a balance between old and new data/features. I doubt this might be also the reason that makes its performance fairly good but not outstanding amongst existing work.

### Questions
1. Fig 1, top right, the yellow connected points before integration, shouldn't it named 'new knowledge' rather than 'old knowledge'?

2. It seems a higher buffer number favours the proposed work, could the authors further justify this?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
In this paper, the authors propose a novel brain-inspired Learning within Sleeping (LwS) approach to solve continuous learning problems, which simulates memory reactivation and distribution generalization mechanisms. It can be considered as a combination of a rehearsal technique with parameter distribution regularization. The experiments conducted show the performance of LwS on various CL benchmark scenarios, including both task incremental and class incremental learning.

### Strengths
(1) The paper is clearly written and easily understood.

(2) The proposed solution originates from a theoretical analysis of the problem of catastrophic forgetting in VCL models.

(3) The experimental results demonstrate the performance of LwS compared to the state-of-the-art, especially in class-IL scenarios.

### Weaknesses
(1) Although somewhat original, the model presented seems to be a combination of known solutions.

(2) The authors claim to "conduct a comprehensive theoretical analysis of VCL in the class-IL scenario". I find this a strong statement, so one would expect a serious in-depth analysis. Meanwhile, the reasoning presented in Section 3 seems only to formalize (perhaps sometimes in an unnecessarily complicated way) some natural observations.

(3) Since LwS includes a data buffer, its fair experimental competitors are the other replay-based models. In this respect, the authors only present a comparison with iCaRL (2017) and A-GEM (2019).

### Questions
(1) Regarding Tab. 1, what is the role of a buffer in the VCL? (Note that increasing a buffer does not increase accuracy).

(2) I would suggest expanding the experimental setup to include newer rehearsal-based competitors.

(3) Will you share the source code for performed experiments?

(4) Minor comments:

p. 7, Eqs. (9) and (10): $h\theta_{t-1} \to h_{\theta_{t-1}}$,

p. 8, l. 1 from bottom: evaluate $\to$ evaluate,

p. 9, under Tab. 1: replya $\to$ reply,

p. 9, l. 3-2 from bottom: that, our $\to$ that our, indicates $\to$ indicate.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
3 good

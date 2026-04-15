# EdVAE: Mitigating Codebook Collapse with Evidential Discrete Variational Autoencoders

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 5, 3

## Abstract
Codebook collapse is a common problem in training deep generative models with discrete representation spaces like Vector Quantized Variational Autoencoders (VQ-VAEs). We observe that the same problem arises for the alternatively designed discrete variational autoencoders (dVAEs) whose encoder directly learns a distribution over the codebook embeddings to represent the data. We hypothesize that using the softmax function to obtain a probability distribution causes the codebook collapse by assigning overconfident probabilities to the best matching codebook elements. In this paper, we propose a novel way to incorporate evidential deep learning (EDL) instead of softmax to combat the codebook collapse problem of dVAE. We evidentially monitor the significance of attaining the probability distribution over the codebook embeddings, in contrast to softmax usage. Our experiments using various datasets show that our model, called EdVAE, mitigates codebook collapse while improving the reconstruction performance, and enhances the codebook usage compared to dVAE and VQ-VAE based models.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
Targeting at the codebook collapse problem in training deep generative models, this work incorporates evidential deep learning instead of softmax to mitigate the overconfidence issue in matching codebook elements. Experiments on various datasets show that the proposed EdVAE method mitigates codebook collapse and improves the reconstruction performance. Source codes are provided in the supplementary material.

### Strengths
+ The motivation for using evidential deep learning to mitigate the overconfidence brought by Softmax is clear and convincing.
+ The paper is well-written and detailed math derivations are provided in the supplementary material, which makes theoretical contributions.
+ Comprehensive experiments are conducted, which demonstrate the effectiveness of this method.
+ Source codes are provided, which makes this work easy to reproduce.

### Weaknesses
- The Related Work section lacks works on evidential deep learning.
- It seems that the coefficient \beta in Eq.(8) is set to different values for different methods on different datasets. Is the experiment performance very sensitive to this coefficient? A hyper-parameter analysis experiment on \beta may be useful for better evaluating the effectiveness of the method.
- In the EDL formulation, uncertainty is calculated by u=C/S, which indicates that the uncertainty is inversely proportional to the sum of evidence. However, in your method, there is no explicit use of uncertainty. What's the reason?
- (Minor) How to ensure the evidence learned from the codebook meaningful?

### Questions
See weaknesses.

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The paper presents a novel approach, Evidential Discrete Variational Autoencoders (EdVAE), designed to mitigate the codebook collapse issue observed in Variational Autoencoders (VAEs). To address the confirmation bias issue associated with discrete VAEs (dVAE), the authors replace the softmax function with evidential deep learning (EDL). Extensive experimental results are provided to substantiate the effectiveness of EdVAEs in comparison to dVAEs.

### Strengths
- The authors adeptly pinpoint and tackle the confirmation bias problem induced by the softmax probabilities in dVAEs.
- Comprehensive experimental results are showcased, underlining the efficacy of the proposed method against established benchmarks.
- The paper is articulated well, featuring clear explanations and a coherent structure.

### Weaknesses
- There is a discrepancy between the method outlined in the text and its code implementation. The paper describes a two-stage sampling process: first sampling a distribution over the codebook from a Dirichlet distribution, parameterized by the output logits, and then sampling a code from this distribution. However, the code implementation directly samples the code from a categorical distribution parameterized by the output logits. Given this, it is unclear why EdVAE, which optimizes a more complex expression for KL divergence, outperforms dVAE, which directly optimizes entropy, when higher entropy is the desired outcome.
- Beyond the computation of the KL divergence, there are two other distinctions between EdVAE and dVAE: 1. The value of \alpha^i
  is capped at a maximum of 20, and 2. EdVAE utilizes a fully connected (FC) layer to compute logits instead of calculating the distance to each code embedding. It would be beneficial to explore whether these modifications would enhance dVAE's performance as well.
- The experimental results could be more compelling. The paper primarily offers quantitative results, but the improvements in MSE and FID over other methods appear to be marginal.

### Questions
- Can the authors clarify the inconsistency between the described method and the code implementation?
- Could the authors elucidate why EdVAE is more effective, given that the goal is to achieve higher entropy, even though EdVAE optimizes a different metric?
- Could the authors conduct ablation studies on alpha clamping and logits computation?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
In this paper, the authors proposed an original extension of discrete VAE with an evidential formulation (EdVAE) to tackle the problem of Codebook collapse. To be Specific, the authors utilize a Dirichlet instead of distribution as a distribution instead of stochastic process over the Categorical distributions that model the codebook embedding assignment to each spatial position. Extensive experiments demonstrate that the proposed methord improves the current benchmark performance.

### Strengths
(1)	The paper is organized and clearly written.
(2)	In this paper, the author attempted to utilize Dirichlet distribution to solve the problem of Codebook Collapse caused by stochasticity, which seems to be intuitively reasonable.
(3)	Sufficient experimental results demonstrate the effectiveness of the proposed method.

### Weaknesses
(1)	The proposed method lacks innovation. The authors proposed to utilized Evidential Deep Learning (EDL) to tackle the problems of codebook collapse that are the combination of two proposed framework.
(2)	There are few recently proposed methods in the experimental results so that I do not know whether the proposed method achieves the superior performance nowadays.
(3)	The paper lacks ablation experiments, which cannot prove the effectiveness of the proposed module.
(4)	The Motivation is unclear. The authors proposed to tackle the Codebook collapse problem in this paper. However, the proposed method has little relevance to the motivation. I would appreciate it if the authors could further explain the rationale for the proposed approach.

### Questions
Please see the Weaknesses.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

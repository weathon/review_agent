# How Transformers Learn In-Context Recall Tasks? Optimality, Training Dynamics and Generalization

- Decision: Reject
- Scores: 2, 6, 4, 4

## Abstract
We study the approximation capabilities, convergence speeds and on-convergence behaviors of one-layer decoder-only transformers trained on in-context recall tasks -- which requires to recognize the \emph{positional} association between a pair of tokens from in-context examples.
Existing theoretical results only focus on the in-context recall behavior of transformers after being trained for \emph{one} gradient descent step. It remains unclear what is the on-convergence behavior of transformers being trained by gradient descent and how fast the convergence rate is. In addition, the generalization of transformers in one-step in-context recall has not been formally investigated. This work addresses these gaps. We first show that a class of transformers with either linear, ReLU or softmax attentions, is provably Bayes-optimal for an in-context recall task. When being trained with gradient descent, we show via a finite-sample analysis that the expected loss converges at linear rate to the Bayes risks. Moreover, we show that the trained transformers exhibit out-of-distribution (OOD) generalization, i.e., generalizing to samples outside of the population distribution. Our theoretical findings are further supported by extensive empirical validations, showing that \emph{without} proper parameterization, standard one-layer transformer models surprisingly \emph{fail} to generalize OOD after being trained by gradient descent.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors study a simple model of associative recall. They show how a simplified Transformer model may solve a recall task, but demonstrate empirically that a model trained from scratch does not generalize on this task unless parameterized in a specific way.

### Strengths
The authors study a timely and fascinating topic. Particularly as LLMs make their way into more and more aspects of our lives, understanding their basic capabilities is important.

### Weaknesses
It remains unclear to me how novel the authors' contribution is. I'm not intimately familiar with this literature, but it seems to me many of the basic results the authors present have already been discovered and analyzed extensively before? 

Specifically, the central claim seems to be that transformers solve an in-context recall task optimally. This seems to have been well studied and validated since Bietti et al (https://arxiv.org/abs/2306.00802). It seems the authors seem to claim novelty by asserting that their analysis studies multiple occurrences of multiple query tokens, but I'm unsure if Bietti et al's analysis is invalidated in this setting? See also Chan et al (https://arxiv.org/abs/2410.23042), which explicitly study the role of having multiple query tokens on associative recall. Also Reddy (https://arxiv.org/abs/2312.03002, https://arxiv.org/abs/2412.00104) performs a detailed theoretical analysis of this phenomenon. Many of these studies also consider softmax attention with noise. Also, the convergence result asserting a linear rate seems to be shown already in Huang et al (https://arxiv.org/abs/2409.17335), no? Is their result invalid in your setting?

Separately, the empirics demonstrating that a Transformer trained from scratch *fails* to generalize well on your task seems to be a severe weakness. LLMs are able to solve associative recall tasks presumably without your parametrizations. Wouldn't this suggest that the phenomena you characterize with your parametrizations are not reflective of actual models?

A small aside, there seem to be frequent accidentally-omitted-words and grammatical typos that hinder your manuscript's clarity. You draft may benefit from a careful read-through to catch these typos.

### Questions
See Weaknesses above.

### Soundness
3

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper provides a theoretical and empirical analysis of how one-layer transformers learn in-context recall tasks—synthetic settings requiring recognition of positional token associations. The authors show (i) that such transformers are Bayes-optimal for both noiseless and noisy recall tasks, (ii) that gradient descent achieves linear convergence to the Bayes risk, and (iii) that the learned representations can generalize out-of-distribution (OOD) to unseen tokens.

### Strengths
1. This paper extends prior studies that only analyzed the first training step or infinite-sample limits by providing finite-sample analyses, explicit reparameterizations, and empirical validations demonstrating when proper parameterization is crucial for OOD generalization

2. This paper theoretically characterizes the different behaviors of the feed-forward layer and the attention layer in in-context recall tasks, which is insightful.

### Weaknesses
1. Limited architecture depth: Results are confined to one-layer, single-head transformers, far from the multi-layer, residual, or multi-head dynamics that dominate real LLMs.

2. The experiments in this paper focus on synthetic tasks; it would be better if the authors could consider real-world language tasks.

### Questions
The authors claim that “without proper parameterization, models with larger expressive power surprisingly fail to generalize OOD after being trained by gradient descent.” However, real-world LLMs usually do not employ such specific parameterizations, then how do LLMs acquire their generalization ability?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper provides a comprehensive analysis of the approximation, optimization, and generalization problem of in-context reasoning tasks with Transformers. The studied Transformers include linear, relu, and softmax attention. The generalization also involves the out-of-domain case on unseen data. Some experiments are provided to support the reparameterization considered in this work.

### Strengths
1. The setting of in-context reasoning is an important and interesting problem to study. 

2. The analysis is comprehensive, which contains many aspects of the theory.

### Weaknesses
1. The analysis is simplified to consider only $\lambda$ as the trainable parameter. This is too restrictive. 

2. The writing can be improved. Why not put real-world examples right after Definition 2.1? 

3. The experiments only show the necessity of reparameterization. However, I think it only applies to synthetic experiments with two-layer models. It is clear whether reparameterization is important in real-world experiments.

### Questions
1. I am fine with experiments with synthetic data and settings. However, I feel experiments that can verify Theorems 5.4, 5.5, and Eqn. (4) are more interesting. I noticed the results of Figure 3. Why not put them in the main body?

2. Can your analysis and the results be extended to multi-head and/or multi-layer Transformers?

3. Why do your gradient updates need to be normalized?

4. How do you motivate the reparameterization with only $\lambda$ as the learnable parameter? Maybe you can cite some papers that support the formulation of $W$, e.g., some works [1, 2, 3, 4] show that attention scores are concentrated on tokens with the same feature as the query. 

[1] Huang et al., ICML 2024. In-context convergence of transformers.

[2] Li et al., ICML 2024. How Do Nonlinear Transformers Learn and Generalize in In-Context Learning?

[3] Li et al., ICLR 2025. Training nonlinear transformers for chain-of-thought inference: A theoretical generalization analysis.

[4] Huang et al., ICLR 2025. A Theoretical Analysis of Self-Supervised Learning for Vision Transformers.

### Soundness
3

### Presentation
3

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
The paper studies a stylized in-context recall task and trains a one-layer decoder transformer with linear/ReLU/softmax attention under a reparameterization. It proves Bayes-optimality of the construction (noiseless and noisy), linear-rate convergence under normalized GD, and OOD generalization to unseen output words; it also gives a finite-sample guarantee and a result showing attention vs. FFN role separation at convergence.

### Strengths
- Clear theoretical guarantees across three attention types. Linear/ReLU (Lemma 3.1, Thm. 3.2) and softmax (Lemma 4.1, Thm. 4.2) get explicit parameterizations with linear convergence proofs.

- OOD to unseen outputs is formalized and proved in both noiseless and noisy settings (Thm. 3.3, Thm. 5.5).

- Mechanistic interpretability hook: theorem showing attention predicts outputs while FFN handles noise after enough steps (Thm. 5.6).

### Weaknesses
1. The abstract states: Existing theoretical results only focus on the in-context reasoning
behavior of transformers after being trained for the one gradient descent step.
This is not correct. Several papers analyze full training dynamics over many GD steps and prove convergence (often linear/finite‐time), not merely “one step” (you cited the first one, and didn't cite the last two), e.g.:

[1] Huang, Cheng & Liang (2023). In-context convergence of transformers. and In-context learning with representations: Contextual generalization of trained transformers. 

[2] Yang, Huang, Liang & Chi (2024). In-context learning with representations: Contextual generalization of trained transformers.

[3] Shen, Zhou, Yang, Shen (2025). On the Training Convergence of Transformers for In-Context Classification of Gaussian Mixtures.
...

---

2. It seems to me that there are some quantitative mismatch that makes some theorem statements numerically wrong (too optimistic) in their dependence on $|Q|$ and $t$:/

- Theorem 3.2 (NGD dynamics): Main text states $\lambda_{q,t} = \eta t/|Q|$ and uses $L(\lambda_t)=O(Ne^{-\eta t})$. But the appendix’s NGD derivation (equal gradient components $\Roghtarrow ||\nabla L||_2 = \sqrt{|Q|}|\partial L/\partial \lambda_q|) gives $\lambda_{1,t}=\eta t/\sqrt{Q}$. Elsewhere you even plug $\lambda_{q,t}=\eta t$. So the printed $|Q|$ and $\eta t$ versions overstate how fast $\lambda$ grows and thus how fast the loss decays.

- In Theorem 3.3, The lower bound that uses $exp(\eta t)$ should be using $\exp(\eta t/|Q|)$. As printed, it predicts higher accuracy sooner.

- Theorem 4.2, a softmax loss bound is written as $O(Ne^{-t})$, but it should retain $\eta: O(Ne^{-\eta t})$.

3. The task seems a bit artificial to me: The vocabulary is partitioned into trigger tokens $Q$ and output tokens $O$, plus a single “generic noise” token $\tau$. Sentences are forced to contain at least one $(q,y)$ bigram; any $(q,x)$ bigram must have $x\in\{y,\tau\}; and the final position is fixed to the trigger $z_H=1$. This raises SNR but is artificial as a language model training distribution.

### Questions
1. Can you compare with the missing literature? What are the novelty/contribution compared to them? e.g., the attention machanism seems very similar to Huang et al ([1], you both consider analyzing one matrix W, and for the softmax version, it's the same as Huang et al and converges to $s I$ with $s\rightarrow \infty$).

### Soundness
1

### Presentation
2

### Contribution
2

# On the Training Convergence of Transformers for In-Context Classification

- Decision: Reject
- Scores: 5, 6, 5, 8

## Abstract
While transformers have demonstrated impressive capacities for in-context learning (ICL) in practice, theoretical understanding of the underlying mechanism enabling transformers to perform ICL is still in its infant stage. This work aims to theoretically study the training dynamics of transformers for in-context classification tasks. We demonstrate that, for in-context classification of Gaussian mixtures under certain assumptions, a single-layer transformer trained via gradient descent converges to a globally optimal model at a linear rate. We further quantify the impact of the training and testing prompt lengths on the ICL inference error of the trained transformer. We show that when the lengths of training and testing prompts are sufficiently large, the prediction of the trained transformer approaches the Bayes-optimal classifier. Experimental results corroborate the theoretical findings.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
Yet another ICL theory paper that does not study actual ICL.

This paper, like many previous works, studies the training dynamics of a simple (single layer, linear) transformer model trained using ICL objective. They work with Gaussian data to study the training convergence rates; and impact of prompt length on error. With these settings, they find the rather unamusing results of linear convergence rates, and Bayes optimal behavior with asymptotic prompt lengths. No connection is made to emergent ICL in real LLMs whatsover. The authors should look at this recent [ICML position paper](https://arxiv.org/pdf/2310.08540) to find the distinction between the two.

**Disclaimer**: I have not read the proofs in detail to verify that they are correct (hence this review is rated at confidence level 4). My review is based on the assumption that the proofs are correct.

### Strengths
- The relation between prompt length and error is somewhat interesting because as far as I know, previous works studying this meta-learning capabilities of the transformer model (misnamed as ICL) do not talk about this learnability constraint for open ended problems. However, other works have looked at something similar [[link]](https://arxiv.org/pdf/2303.07971).
- The other claim about being the first to study multi-class classification maybe true. Its significance is unclear.

### Weaknesses
- The setting is too unrealistic to say anything about real ICL. For example,
  - Training on hard-coded ICL prompts, when LLMs are trained on next-word prediction (ICL structure is generally not present in the pretraining corpus). This is a major setup difference which makes them incompatible.
  - Studying single layer transformers with no non-linear activation functions. This is a good intellectual curiosity but its relevance and usefulness in understanding ICL remains unclear (even classic deep learning theory struggles to present useful insights by studying 2-layer networks). In this paper itself, we see a deviation from expectations when a 3-layer GPT2 architecture with softmax is tested (section 5.2).
  - Gaussian data that presents fixed one-token length inputs and outputs. I don't have a problem with Gaussian data, but the framework should be flexible enough to even somewhat resemble real ICL (where the inputs and outputs both can be variable lengths).

### Questions
- Binary classification is a special case of the multi-class classification. Why write up both?
- ICL in LLMs is a type of domain adaptation setting, where knowledge about a particular task is scarce in the pre-training corpus. The model needs to be "kindled" with ICL demos to get it to perform better on this task. In contrast, the presented theory first trains the transformers on samples of these tasks, and then requires this data to be "properly distributed" over the space of variations. It is not surprising that the presented results hold in this setting. How do you think the setup needs to change to reflect that realistic ICL in LLMs setting?
- With respect to the following: 
> probably because our transformer models were only trained with a small prompt length of N = 100.

  3 layer GPT2 seems like a small model, why not test with higher N?

### Soundness
3

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
This work studied the training dynamics of a one-layer linear transformer trained via GD for in-context multi-classification tasks. They established  convergence guarantees for in-context trianing and also provided an in-context inference error bound, which scales with the length of both training and testing prompts.

### Strengths
1. This work is the first to examine transformers for in-context multi-class classification from a training dynamics perspective.
2. The end-to-end connection between in-context inference and training offers a novel  insight.
3. Experimental results are provided to support theoretical statements.

### Weaknesses
My major concern lies in the analytical novelty of this paper compared to prior work on in-context regression [1]. While this study focuses on the multi-class classification problem, its model and analytical approach appear to share many similarities with [1]. It remains unclear how technically straightforward it is to generalize the results of [1] to multi-classification. Additionally, this paper restricts itself to the linear attention setting, simplifying the analysis and making it somewhat less impactful than [2], which addresses binary classification with strict data assumptions but in the more realistic softmax attention setting. Therefore, a thorough discussion clarifying the technical distinctions and contributions of this work relative to these previous studies would be helpful.




[1] Trained Transformers Learn Linear Models In-Context. Zhang et al., 2023

[2] Training nonlinear transformers for efficient in-context learning: A theoretical learning and generalization analysis. Li et al., 2024

### Questions
1. For data distribution, why is it essential to preserve the inner product of vectors in the $\Lambda^{-1}$-weighted norm? Is this primarily a technical consideration? It would be helpful if the authors could provide further clarification on the role of data distribution in the analysis.

2. For the inference stage, while $\mu_0$ and $\mu_1$ are not subject to additional constraints, $\Lambda$ remains fixed, imposing a strong assumption on the underlying structure of the data distribution. Do the authors have insights on how these results might extend to scenarios with a varying $\Lambda$ during inference?

3. The derived inference bound scales with $N$ and $M$ similarly to in-context regression [1]. Could the authors clarify the distinctive aspects of the multi-classification setting in this context? (This also points to weakness.)

4. For the multi-classification setting, what is the order of the number of classes $c$? On line 431, the authors mention that $c$ is treated as a constant coefficient—would a larger order of $c$ impact the analysis?








[1] Trained Transformers Learn Linear Models In-Context. Zhang et al., 2023

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper study the learning dynamics of transformers for in-context classification of Gaussian mixtures and prove the training convergence of in-context multi-class classification. The author presents three key findings: 1) a proof that a single-layer transformer trained via gradient descent converges to a globally optimal solution at a linear rate; 2) an analysis of how the lengths of training and testing prompts influence the inference error in in-context learning; and 3) evidence that, with sufficiently long training and testing prompts, the predictions of the trained transformer approach those of the Bayes-optimal classifier. Some of these results are validated through experiments.

### Strengths
1. The structure of this paper is very clear.
2. Analyzing the training dynamics of in-context learning is crucial.
3. The findings regarding infinite lengths of training prompts (N) and testing prompts (M) are interesting

### Weaknesses
1) This paper claims to be the first to explore the learning dynamics of transformers for in-context classification of Gaussian mixtures and to prove the convergence of training in multi-class classification. However, I find the significance of this assertion unclear, as the paper lacks sufficient detail. Specifically: 1) many prior works have analyzed in-context learning assuming $x $ comes from Gaussian distributions; what additional insights do the results on Gaussian mixtures provide? 2) Why is extending results from binary to multi-class classification considered essential and non-trivial?

2) Additionally, I have concerns regarding the techniques used: 
- The introduction of $ \tilde{L} $ appears to be a key element in proving Theorem 3.1, but its intuition is unclear, and I'm uncertain how it addresses the challenges posed by the non-linear loss function. 
- The paper heavily relies on Taylor expansion in its proofs, and I question whether this expansion can accurately approximate the original function. More detail is needed on this aspect.

### Questions
1.The condition (2) in Assumption 3.1 seems unusual to me. Could the authors provide more clarification on this assumption?

2.Some papers[1,2,3] have highlighted emergent behaviors in the training dynamics of in-context learning. However, this paper asserts that the transformer will converge to its global minimizer at a linear rate, which appears to contradict those findings. Can the authors discuss this further?

[1] In-context learning and induction heads

[2] Breaking through the learning plateaus of in-context learning in Transformer

[3] Training dynamics of multi-head softmax attention for in-context learning: Emergence, convergence, and optimality.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The paper investigates the training convergence of transformers for in-context classification tasks. It demonstrates that a single-layer transformer trained by gradient descent converges to a globally optimal model at a linear rate for in-context classification of Gaussian mixtures. Experimental results confirm the theoretical findings, showing that the trained transformers perform well in binary and multi-class classification tasks.

### Strengths
1. Rigorous theory: The paper provides a detailed theoretical analysis of the convergence properties of transformers for in-context classification tasks, demonstrating that under certain conditions, a single-layer transformer trained by gradient descent achieves global optimality at a linear rate.

2. Experimental validation: The theoretical claims are corroborated by experimental results. The paper's experiments on binary and multi-class classification tasks with Gaussian mixtures verify the theoretical predictions, showing that the transformers' prediction accuracy improves as the training and testing prompt lengths increase.

### Weaknesses
The paper primarily focuses on single-layer transformers with simplified linear attention mechanisms. While it provides valuable insights into the convergence properties and error bounds for these models, the findings may not fully extend to more complex multi-layer transformers with softmax or relu attention mechanisms. And the training dynamics of multi-layer transformers could be different to single-layer -- just like multi-layer MLP can be different than linear models.

Nonetheless, I don't regard this as a strong weakness since the field is evolving and to my best knowledge, most studies are still on one-layer transformers.

### Questions
Does the same training dynamics results apply to tasks beyond linear regression/classification? Would it be different for other mathematical/statistical tasks, for example, time series, etc.

### Soundness
4

### Presentation
3

### Contribution
3

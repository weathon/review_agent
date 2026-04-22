# EMA Without the Lag: Bias-Corrected Iterate Averaging Schemes

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 6, 4, 6

## Abstract
Stochasticity in language model fine-tuning, often caused by the small batch sizes typically used in this regime, can destabilize training by introducing large oscillations in generation quality.  A popular approach to mitigating this instability is to take an Exponential moving average (EMA) of weights throughout training.  While EMA reduces stochasticity, thereby smoothing training, the introduction of bias from old iterates often creates a lag in optimization relative to vanilla training.  In this work, we propose the Bias-Corrected Exponential Moving Average (BEMA), a simple and practical augmentation of EMA that retains variance-reduction benefits while eliminating bias. BEMA is motivated by a simple theoretical model wherein we demonstrate provable acceleration of BEMA over both a standard EMA and vanilla training.  Through an extensive suite of experiments on Language Models, we show that BEMA leads to significantly improved convergence rates and final performance over both EMA and vanilla training in a variety of standard LM benchmarks, making BEMA a practical and theoretically motivated intervention for more stable and efficient fine-tuning.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes BEMA, a new approach to estimate the optimal solution $\mu\^\star$ using the historical trajectory. Theoretically, this proposed method models the optimization dynamic as an OU process and derive the MLE based on this model. The upper bound show that the proposed MLE and OUEMA will converge to the optimal parameter $\mu\^\star$. On the empirical side, this paper proposes a practical approach to obtain the derived estimator and validate its performance in multiple LLM benchmarks. The results show nearly consistent improvements compared to baseline approaches.

### Strengths
This paper is well-written and clearly presents its main contributions including a new design to replace EMA in the optimization algorithms.

On the theoretical side, the convergent results seem to be new, as existing aysmptotic results in MLE are mostly based on the sample size instead of the training time T. 

The experimental results are significant. The performance gains are visiable in all figures.

### Weaknesses
1. The theoretical framework is based on the noisy quadratic model. It is hard to say if this assumption is too strong as all models are quadratic near $\mu^*$ in a sufficiently small neighborhood by using the Tayler expansion. But it indeed restrict the scoop of this theory.

2. It seems that the BEMA requires a few more hyper-parameters, which may not be a desired property for a new approach. 

3.  The convergent results built in Section 3 mainly require $\hat{\mu}$ is unbiased; when the matrix A is replaced with an estimator, it seems that the new algorithm doesn't have any convergence guarantees. 

4. The experimental results are solid and cover every piece of this new approach. But the experiments are not very standard; it doesn't use any popular benchmark (e.g. nanoGPT speedrun) , instead it chooses their own benchmark.

### Questions
Based on the weaknesses, I would have a few questions:

1. As the matrix A may not be a diagonal matrix, it doesn't make too much sense for me to approximate it using simply a diagonal matrix. It is surprising that it "suffices to provide good performance in practice". Is this statement (on page 7, line 350) supported by some experiments or theoretical results? 

2. In Adam or AdamW, they are typically combined with an additional second-order momentum term. Are these second-order momentum also included in the BEMA implementation for the comparison experiments (Figure 4)? It can be better if it also includes the comparison with some common optimizer e.g. Adam. 

3. How is this proposed approach be scaled in larger models? This paper only considers a few small LLM (~1B), but in modern ML, people will use significantly larger model. It could be more convincing to include some modern architectures with 7B or even larger sizes.

### Soundness
2

### Presentation
3

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
This paper proposes BEMA (Bias-Corrected Exponential Moving Average), a theoretically grounded alternative to the widely used EMA for parameter smoothing in stochastic optimization. The authors identify that the exponential averaging of past parameters leads to systematic delay in convergence. Then they derive a bias-free correction based on the Ornstein–Uhlenbeck (OU) formulation of SGD dynamics. The proposed estimator achieves theoretically minimal MSE, matching the lower bound established in Proposition. They empirically verify their theory on small-to-medium language models (Qwen2.5-1.5B, Gemma3-1B, and Llama3.2-1B) across standard instruction-following benchmarks.

### Strengths
1. The paper provides a mathematically analysis of EMA under OU dynamics, formally decomposing the bias–variance tradeoff and offering a closed-form correction term. 
2. Results show that BEMA consistently improves stability and early-stage convergence compared to vanilla EMA.

### Weaknesses
1. A key limitation of this paper is the lack of comparison with modern adaptive optimizers such as Adam or AdamW. All experiments are conducted under the SGD + EMA setting, which is rarely used in contemporary large language model (LLM) fine-tuning. Since Adam itself maintains exponential moving averages of both the first and second moments of gradients, it already implicitly addresses part of the variance–bias tradeoff that the paper analyzes. Therefore, it remains unclear whether the proposed BEMA still provides additional benefits. 
2. The experiments are mainly conducted on relatively small or medium-sized models (Qwen2.5-1.5B, Gemma3-1B, and Llama3.2-1B). While the results are consistent with the theoretical analysis, it remains unclear whether the proposed BEMA retains its advantages when scaling to larger models (e.g., 7B–70B).

### Questions
Should the $t$ in the left-hand-side of the first equation between line 188-189 be $t+1$?

### Soundness
3

### Presentation
3

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
This paper introduces BEMA, a modification of EMA designed to retain its variance-reduction benefits while eliminating lag bias, starting from a stochastic-optimization viewpoint, modeling training trajectories as an OU process, deriving a MLE problem for the target parameter. Autors then propose a practical, drop-in implementation, experimental results demonstrate improved stability and convergence in LMs fine-tuning.

### Strengths
Strengths:
* The theoritical results are clearly presented and provides a strong foundation for motivating the proposed method.
* The proposed method is simple to implement and achieves better performance compared with mutiple baselines on accuracy and convergence speed.

### Weaknesses
Weaknesses & Questions:

* The motivation of the method is somewhat confusing. The paper begins by stating that closed-loop training requires stabilizers and then proposes BEMA. However, in the supervised fine-tuning setup that this paper focuses on, the training process is not a closed loop scenario. In SFT, the model is trained with a teacher-forcing mechanism, where during the forward pass the next token is predicted based on the ground-truth of the previous token. This differs from autoregressive generation or the setting used in the GVA paper.
* The proposed method introduces additional memory overhead equivalent to the size of the model itself, which could become significant when scaling to larger models.
* The importance of choosing $\theta_0$ as the anchor point remains unclear. It would be valuable to test the sensitivity of the method to anchor selection or provide a discussion on this design choice.
* A recent work [1] also uses $\theta_0$ as an anchor point to accelerate training, which appears conceptually related to this paper. Could you provide a brief discussion with the difference.
* Since the paper emphasizes practical implementation, it would be beneficial to include comparisons regarding resource efficiency, such as memory usage and GPU hours.
* All fine-tuning experiments are conducted on relatively small models (≤ 1.5B) and limited datasets. It remains uncertain whether the same improvements would hold for larger models (e.g., 7B or beyond) or in different training settings, such as instruction tuning.

[1] Harmony in divergence: Towards fast, accurate, and memory-efficient zeroth-order llm fine-tuning

### Questions
Some notation is conflict ($\eta$ used twice in line 142 and method part).

Please see weaknesses for questions.

### Soundness
4

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes Bias-Corrected Exponential Moving Average (BEMA), a new method for stabilizing the training of large language models (LLMs) during fine-tuning. BEMA is a simple and practical augmentation of EMA that eliminates this bias while retaining the variance-reduction benefits. BEMA is a simple extension, requiring only a two-line change to existing EMA implementations. Extensive experiments on LLMs demonstrate that BEMA significantly improves convergence rates and final performance . BEMA's performance is shown to be robust across various optimizer hyperparameters and models.

### Strengths
Below are some strengths I find in this paper:
- The authors identify an important issue (the "lag" problem) in traditional EMA when fine-tuning large language models (LMs) with small batch sizes, I think this is a relevant and practical concern in deep learning - resolving which can have a big impact on the quality of downstream applications resulting out of fine-tuning models.
- Stronger theoretical grounding (based on Ornstein-Uhlenbeck process in quadratic optimization) that shows provable acceleration. I think this provides more confidence to me on this method.
- Easy of use: Being a drop-in replacement makes it easy for practitioners to use thie method.
- The level of empirical study conducted to validate this method on real world LLMs models (Gemini, LLama etc) seem promising and assuring to me.

### Weaknesses
Despite the strengths, below are some comments I have on areas of imprvoements and some limitations I see:
- Despite the strong theoretical backing using the OU process, the main assumption of having a noisy quadratic model is bit too simplistic in my opinion. It is not clear to me whether the findings nicely carry over to the more complex landscape observed while training real-world Deep Neural Networks and LLMs.
- Not clear how much hyper-parameter tuning went into making this method work. If the authors could add a section or clarify this in much more detail that would be helpful.


Language, Grammar and Typos:
Line 51: The most empirically successful approach to stabilizization is iterate averaging, -> “stabilization”

Use of the term “stabilizer” in the paper is not clear to me. I feel it needs to be explained well, since this is not a standard term in my opinion.

### Questions
Some additional qns to the authors:
- I am curious to know what are some challenges authors foresee in adaptively estimating A?
- Authors mention: “observe that training with a fixed learning rate and then applying BEMA leads to the best performance throughout, providing preliminary evidence that applying post-hoc stabilization can obviate the need for learning rate decay in post-training.” -> Is this indeed true? How did the authors confirm this?

### Soundness
3

### Presentation
2

### Contribution
2

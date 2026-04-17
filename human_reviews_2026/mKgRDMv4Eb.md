# Identifying Support Knowledge Representation for Large Language Models

- Decision: Reject
- Scores: 2, 4, 6, 6

## Abstract
Large language models (LLMs) have demonstrated remarkable capabilities, yet their knowledge representation mechanism remains opaque. This paper interprets the LLM's attention layers as a notion of Support Knowledge (SK), a learned representation that LLMs use to ground their decisions, exhibiting properties of sparsity, causality, and atomicity. We provide a theoretical analysis of the mathematical connection between margin maximization and the identification of the Support Knowledge. Empirical analyses suggest a two-segment inference machanism of SK across diverse LLM families (e.g., Qwen3, Llama3.1, ChatGLM4), indicating that there exists layer-wise supporting information units. Interventions on the targeted SK-related neural units reveal their information preserving properties: attacking the SK-related neural units would reduce the model accuracy from 82.34% to 27.78%, while random attacks would only lead to a reduced accuracy of 80.43%.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces and systematically investigates the concept of Support Knowledge (SK), proposing that LLMs rely on a small, causally critical set of atomic tokens during reasoning analogous to support vectors in SVMs. To identify these tokens, the authors present a Gradient-Attention framework that combines two complementary  gradient  and attention to dynamically locate the input tokens the model truly depends on. 

The findings are intuitive and supported by thorough experiments. However, the work stops short of demonstrating practical gains from SK. Future efforts that integrate SK into model training or optimization could significantly enhance its practical relevance and impact.

### Strengths
1. Comprehensive analysis: The paper conducts extensive evaluations across multiple models (Qwen3, Llama3.1, ChatGLM4) and scales (8B–32B), showing that the proposed Support Knowledge (SK) phenomenon consistently exists across architectures.
    
    It further includes layer-wise analysis, cross-model comparisons, ablation studies, and baseline evaluations, which together make the conclusions convincing and well-grounded.
    
2. interpretability: The identified SK tokens often carry clear semantic meaning (e.g., “true”, “interstate”), intuitively reflecting the model’s reasoning logic. Moreover, the visualization of SK evolution across layers provides a clear and interpretable view of how reasoning dynamics emerge within the model.

### Weaknesses
1. Limited task coverage

All experiments are conducted on BoolQ, a binary QA task. It remains unclear whether the proposed findings generalize to other task types such as reasoning, summarization, or multi-turn dialogue.
The sparsity assumption of SK also remains untested in open-ended generation tasks (e.g., text or code generation), where token dependencies may differ substantially.

2. Lack of downstream validation (my core concern)

- Although the paper effectively identifies SK tokens, it does not apply them in fine-tuning, preference optimization, or reinforcement learning to assess their practical utility.
- As a result, SK currently functions as an interpretability tool rather than an optimization signal. Demonstrating its value in improving model performance or training efficiency would make the work more impactful.

### Questions
Did the authors use SK tokens for fine-tuning or reinforcement learning to verify their practical utility? The paper primarily focuses on interpretability and intervention experiments, rather than applying SK as a training signal. Although the idea of SK-aligned distillation is introduced, it has not been implemented or empirically validated. As it stands, SK remains an analytical concept rather than a demonstrated optimization method.

Related studies have explored similar directions, for instance, phi-4[1] identifies key tokens through multiple sampling for DPO, while [2] uses high-entropy tokens to stabilize RL training. It would be interesting to see whether the authors could leverage SK-identified tokens for comparable downstream applications, such as fine-tuning, preference optimization, or reinforcement learning.

[1] Abdin M, Aneja J, Behl H, et al. Phi-4 technical report[J]. arXiv preprint arXiv:2412.08905, 2024.

[2] Wang S, Yu L, Gao C, et al. Beyond the 80/20 rule: High-entropy minority tokens drive effective reinforcement learning for llm reasoning[J]. arXiv preprint arXiv:2506.01939, 2025.

### Soundness
2

### Presentation
3

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
The paper introduces Support Knowledge (SK), a sparse set of “causally critical” tokens identified by a combined gradient–attention score. The paper argues that Transformer inference unfolds in a two-segment pattern: early evidence gathering and later task framing. Concretely, the method computes per-token gradient norms and an attention-based max-margin term, normalizes them, mixes them with model-specific weights $\alpha$, $\beta$($\alpha+\beta=1$) , then selects a sparse set via an elbow rule; causal importance is probed by mean-replacement ablations of hidden states at a chosen layer. On BoolQ, targeting SK tokens yields substantially larger accuracy drops than random or simple baselines; similar qualitative patterns are reported for Llama3.1, ChatGLM4, and Qwen3.

### Strengths
- **Clear writing**: The writing is well-organized, moving from theoretical motivation to method and experiments with helpful figures and algorithms.

- **Simple, testable methodology**: Concrete SK definition, elbow-based selection, and mean-replacement interventions are spelled out with algorithms, aiding reproducibility.

- **Strong experiment result**: Reported drops are large under the chosen intervention (e.g., substantial degradation when ablating only a 17% SK subset), suggesting the score captures influential tokens.

- **Empirical patterning across models**: Reports of the two-segment SK distribution and layer-wise ablation sensitivity across multiple LLM families.

### Weaknesses
- Theory seems to be ad-hoc, The connection between LLM and SVM has been established before. Most of the theory part in the paper appears to be an ad-hoc. And the connection between attention gradient and SVM is too far-fetched.
- A lot of methods identifying important tokens exist[1][2][3]. To my knowledge, the key contribution of the paper seems to be identifying important tokens easily. However, the SK methods is not straightforward, requiring tuning $\alpha$ and $\beta$. ($\alpha+\beta=1.1$ for LlaMA in Table 2, contradiction of paper's setting). 

[1] Chefer, Hila, Shir Gur, and Lior Wolf. "Transformer interpretability beyond attention visualization." In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 782-791. 2021.
[2] Bai, Yu, Heyan Huang, Cesare Spinoso-Di Piano, Marc-Antoine Rondeau, Sanxing Chen, Yang Gao, and Jackie Chi Kit Cheung. "Identifying and Analyzing Performance-Critical Tokens in Large Language Models." arXiv e-prints (2024): arXiv-2401.
[3] Meng, Kevin, David Bau, Alex Andonian, and Yonatan Belinkov. "Locating and editing factual associations in gpt." Advances in neural information processing systems 35 (2022): 17359-17372.

### Questions
- Theory
  - The subscript $x_t \in T$ in equation (4) should be $t \in [1,T] \cap Z$. What is the relationship between equation (4) and the following content?
  - They state $||\nabla_{x_t} f(X)|| \propto \partial L / \partial d_t$, $d_t$ ​is the “distance to the decision boundary manifold,” and conclude tokens with large gradients are “like support vectors.” But the paper doesn’t define $d_t$ operationally for LLM. The boundary/manifold framing is imported from SVMs without a workable definition here.
  - Theorem 1 is very elegent, but what are definitions of $\alpha$, $b$ here? Where is the proof? How to map Transformer objectives/constraints to an SVM Lagrangian? What is the exact formulation of your primal and dual problem? 
- Experiment and Method
  - In Definition 1, '$a_t$ is the attention weight on token $t$'. However, attention is a matrix. Is this attention weight between predicted token and token $t$.
  - Most of your competitors in Table 1 are generally weak. Have you compare with stronger competitors? And your setting, mean-replacment invention seems to be unconventional. It appears that you create a personal benchmark and compare with weak baselines.

In general, the paper presents intriguing empirical patterns and a practical scoring rule. However, I personally does not get the meaning of the method, as a lot of methods identifying important tokens already exist. And the theory seems to be rather ad-hoc. I don't fully grasp it and welcome other reviewers' insights.

### Soundness
2

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
2

### Summary
The paper proposes a novel method for identifying tokens in the attention mechanism which are important to the model's output in a classification task. They show that "masking" these tokens results in drastic drop in performance, compared to masking a tokens at random. The proposed method shows interesting results in terms of LLM interpretability on the BoolQ dataset and seems applicable to other tasks, which should be of interest to the community.

### Strengths
- Interesting findings.
- Useful seemingly novel method for LLM interpretability.

### Weaknesses
While the findings are interesting, the authors only explored a classification task, namely the BoolQ dataset. While it would be nice to have results on more tasks, I believe that the findings are already sufficient. My problem is with how certain statements in the discussion section are phrased. For instance, they write "Our findings indicate that LLMs do not learn in a uniformly distributed manner, but instead converge
on a sparse, structured computational principle we term Support Knowledge (SK)." This is only true for the BoolQ dataset. The authors do however state right after that "While this study focuses on the binary QA format to cleanly isolate these core mechanisms, future work should explore their generalizability to other tasks." So I do not believe that the authors are intentionally over-claiming, but the way that certain sentences are phrased gives that impression. This is a relatively minor issue, but the paper would be better in my opinion if such claims were better phrased/contextualized.

### Questions
- You use the term "machanism" throughout the paper. Is this a typo or is this actually a word?
- When referring to the Appendix, you do not refer to the section in the Appendix.
- Line 29-30, I am not sure what this sentence is conveying, are you sure that the syntax is correct?
- Line 35-36 makes a claim about how the human mind works, but has no reference to back it up.
- Your references lack a space between the parenthesis and the word prior, e.g., line 74: "input(Sundararajan et al., 2017)." -> "input (Sundararajan et al., 2017)."
- Have you explored the potential computational benefits of identifying non SK tokens and replacing them with mean replacement? You do have to compute gradients however, so I am unsure if there is a potential benefit here or not.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This manuscript builds on the theoretical findings of this February 2024 paper: Tarzanagh et al. (2024) Transformers as Support Vector Machines, https://arxiv.org/abs/2308.16898. (Side note: There are two citations to the same paper in the References.)

The manuscript defines the "Support Knowledge" (SK) score for each token as the sum of (1) the gradient norm of the token's input embedding and (2) its attention-weighted max-margin score. Each component of the sum is weighted by a hyperparameter. 

The manuscript claims that SK is a byproduct of the gradient-based optimization similar to how SVMs identify support vectors. 

The manuscript contains experiments that show the importance of SK in binary QA classification. 

Side note: There is a typo in Line 294.

### Strengths
- The topic of relating the attention mechanism to SVMs is interesting.
- The paper is easy to read.

### Weaknesses
- No theoretical explanation is provided for the two-segment inference mechanism observed in the experiments.

- This statement is a bit misleading: "This correspondence arises because backpropagation through transformer layers effectively solves the dual problem of an analogous SVM optimization." (Line 187-188)

After training, the optimal transformer attention layer can be mathematically represented as the dual expansion of an analogous SVM optimization. However, backpropagation operates in the primal parameter space, and the dual solution is a property of the model at optimum, not the direct result of the training algorithm

- A short summary of Appendix B should be added after Equation 5 to give the reader an idea of the importance of the values of alpha and beta. 

- The explanation for why the values for alpha and beta in Equation 5 differ so much across various models (Lines 699-701) needs further investigation. 

- There are a couple of missing related works:
(a) Tan M. Nguyen et al. A Primal-Dual Framework for Transformers and Neural Networks (https://arxiv.org/abs/2406.13781; June 2024)
(b) Shahar Katz and Lior Wolf. Reversed Attention: On The Gradient Descent Of Attention Layers In GPT (https://arxiv.org/abs/2412.17019; December 2024)

### Questions
- Can you provide a theoretical explanation for the two-segment inference mechanism seen in the experiments? 

- Can you provide a more detailed explanation for why the values for alpha and beta in Equation 5 differ so much across various models?

### Soundness
3

### Presentation
3

### Contribution
3

# Mamba Can Learn Low-Dimensional Targets In-Context via Test-Time Feature Learning

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 6, 4, 6

## Abstract
Mamba, a recently proposed linear-time sequence model, has attracted significant attention for its computational efficiency and strong empirical performance. However, a rigorous theoretical understanding of its underlying mechanisms remains limited. In this work, we provide a theoretical analysis of Mamba's in-context learning (ICL) capability by focusing on tasks defined by low-dimensional nonlinear target functions. Specifically, we study in-context learning of a single-index model $y \approx g_*(\langle \boldsymbol{\beta}, \boldsymbol{x} \rangle)$, which depends on only a single relevant direction $\boldsymbol{\beta}$, referred to as *feature*. We prove that Mamba, pretrained by gradient-based methods, can achieve efficient ICL via *test-time feature learning*, extracting the relevant direction directly from context examples. Consequently, we establish a test-time sample complexity that improves upon linear Transformers—analyzed to behave like kernel methods—and is comparable to nonlinear Transformers, which have been shown to surpass the Correlational Statistical Query (CSQ) lower bound and achieve near information-theoretically optimal rate in previous works. Our analysis reveals the crucial role of the *nonlinear gating* mechanism in Mamba for feature extraction, highlighting it as the fundamental driver behind Mamba’s ability to achieve both computational efficiency and high performance.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper studies the in-context learning of Mamba model in a theoretical way. It provides theoretical sample complexity bound on the number of pretraining tasks to achieve a fixed ICL test error via a two-stage gradient pretraining scheme. The paper also proves Mamba can achieve test time feature learning and conducts experiment to verify this.

### Strengths
1. The paper theoretically studies the ICL of Mamba, which is an interesting topic.
2. The paper is the first to provide a sample complexity result on the ICL of Mamba.
3. The writing is clear which makes the paper easy to follow.

### Weaknesses
1. Since the authors listed proposition 4.1 as one of the core contributions, I think a formal version of it could benefit the paper, because currently it's hard to understand what $P_1$ and $P_2$ stands for without referring to the appendix (Lemma C.1).
2. The experiment doesn't collaborate with the theoretical results well. It only shows that Mamba **can** achieve in-context feature learning, which is not that surprising given empirical results on the ICL of Mamba models [1,2]. Perhaps some trends toward how the number of pretraining tasks affects the final ICL test error could be more relative to the point.
3. I'm currently not so sure about the siginificance of contribution compared with [3]. The two stage learning algorithm and the learning objective in this paper is the same as in [3], and the replacement of a transformer with a Mamba doesn't seem very fundamental.


[1] Grazzi, R., Siems, J., Schrodi, S., Brox, T., & Hutter, F. (2024). Is mamba capable of in-context learning?. arXiv preprint arXiv:2402.03170.

[2] Park, J., Park, J., Xiong, Z., Lee, N., Cho, J., Oymak, S., ... & Papailiopoulos, D. (2024). Can mamba learn how to learn? a comparative study on in-context learning tasks. arXiv preprint arXiv:2402.04248.

[3] Nishikawa, N., Song, Y., Oko, K., Wu, D., & Suzuki, T. Nonlinear transformers can perform inference-time feature learning. In Forty-second International Conference on Machine Learning.

### Questions
1. Could the authors ellaborate on the fundamental difference between your paper and [1]? Though I haven't read [1] carefully, but it seems to me that your work inherits the target function $f_*$, the learning objective loss $L$, the two-stage pretraining algorithm and the diagonal structure of $W_B^\top W_C$ (which in [1]'s case is $W_Q^\top W_K$, and the $1$ in the right bottom corner could be ommitted because Mamba doesn't have value matrix $W_V$). The main difference is the replacement of the transformer model with the Mamba model, which mimics a linear self attention (admittedly the nonlinear gating may add some difficulty). Overall, I'm concerned about the techinical contribution of this paper and would be willing to reconsider my rating if the authors could convince me of it.


[1] Nishikawa, N., Song, Y., Oko, K., Wu, D., & Suzuki, T. Nonlinear transformers can perform inference-time feature learning. In Forty-second International Conference on Machine Learning.

### Soundness
3

### Presentation
3

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
This paper studies the ICL ability of Mamba on single-index models by characterizing the optimization dynamics and the sample complexity in pretraining and testing. The theoretical results indicate that Mamba can implement test-time feature learning, which is enabled by the nonlinear gating. The authors also make a comparison between their results and previous works and other archtectures.

### Strengths
1. The theoretical analysis is quite strong and solid.

2. The studied problem on Mamba with single-index models is interesting and novel.

### Weaknesses
1. The illustration in Section 4 is confusing to me. 

(a) Overall, I think many of this section is not a "proof overview" but some important discussion that you can put in previous sections. In other words, in Section 4.1, you introduce too much about some theoretical conclusions or the differences from existing works rather than the logic chain to derive your main theorem, which should be the proof idea from my understanding. 

(b). Regarding the role of gating and input embedding, $c_\beta$ vanishes when $ie(g_*)>2$ if it is linear attention, but what about Mamba? Do you mean $c_\beta$ will not vanish in this case? When I first read it, I got confused because I thought you would directly answer this question, but you started to talk about the generative exponent. I think it is better to first answer this question or briefly summarize the role of gating at the beginning of this paragraph. 

(c). Proposition 4.1 is still not "informal" enough. Why not directly write $Mamba(\cdot)$ as a $g_*(\cdot)$ function in Eqn. 2? You mention this possibility in line 433 that $g_*(z)$ is a polynomial of $z^{ge(g_*)}$. By the way, I took some time to recall what $P_1$ and $P_2$ are in this proposition. I found they are defined in line 300. It is better to intuitively explain them around this proposition.

2. Not enough experiments are presented. Can you empirically show the advantages of Mamba in sample complexity compared with Transformers, and can you empirically relate it to the nonlinear gating part?

### Questions
1. Why do you need an MLP layer in your analysis? In line 365, you say the improvement of sample complexity comes from the MLP, then you need to clarify what makes Mamba better than Transformers in some aspect because MLP is not a key difference between Mamba and Transformers from my understanding. You need to distinguish what improvement comes from MLP and what comes from the key component of Mamba, i.e, the nonlinear gating. 

2. Why do you formulate $W_B^\top W_C$ with a simple veector $\gamma$ and why do you initialize $\gamma$ in a sepcial way in line 256?

3. It seems you reduce the analysis of a $d+1$-dimensional $w$ to a scalar $\rho$. What is the reason?

4. Why do you formulate $\phi(\theta)$ in a special way in line 154? Can you handle of case of $\phi(x)=x$?

### Soundness
3

### Presentation
2

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
This paper provides a theoretical analysis of the Mamba architecture’s in-context learning (ICL) capability, focusing on low-dimensional nonlinear target functions. The authors establish that Mamba, when pretrained via a gradient-based algorithm, can achieve efficient in-context feature learning at test time. They prove that Mamba extracts task-relevant features directly from contextual examples, yielding sample complexity comparable to nonlinear Transformers while improving upon linear kernel-based models.

### Strengths
1. Most existing theoretical works on understanding in-context learning (ICL) focus on Transformers, and this paper’s analysis of Mamba—along with its comparison to Transformers—offers valuable and insightful contributions.

2. Provide conceptual insight: The identification of Mamba’s nonlinear gating as the driver of test-time feature learning is conceptually meaningful for understanding gated recurrent architectures.

### Weaknesses
1. Algorithm 1 appears unnatural. Algorithm 1 optimize the Mamba layer and the MLP layer separately, which is not how neural networks are actually optimized in practice. This mismatch with realistic training procedures weakens the contribution of the paper.

2. The input embedding also appears unnatural and overly hand-crafted, raising doubts about whether real models could actually learn such embeddings in practice.

### Questions
1. Could the authors consider analyzing Mamba’s training convergence under a more practical optimization process? (Weakness 1)

2. Could the authors provide an analysis of generalization—for example, how the model behaves when the data distribution at test time differs from that during training?

### Soundness
3

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
2

### Summary
This paper analyzes Mamba's in-context learning on Gaussian single-index models and proves that pretrained Mamba achieves test-time feature learning by gating operation. This enables test-time sample complexity comparable to nonlinear Transformers, and better than linear-Transformers. The main theorem specifies pretrain and context lengths, number of tasks and width needed to achieve low test error with high probability. Experiments with r=8, d=16,32 show comparable performance of Transformer and Mamba, and that the results are insensitive to d. Both models outperform kernel ridge regression even when the kernel operates on the intrinsic features, supporting the main theory of test-time feature learning.

### Strengths
- The paper shows that the nonlinear gate changes the label statistic so the task becomes identifiable. This allows the dependence of the sample complexity on the true nonlinearity of the teacher function.
- The main theorem gives concrete conditions under which a one-layer Mamba with MLP head can learn features from the prompt at test time, and the required number of in-context examples scales with the intrinsic dimension of the task subspace. 
- The detailed analysis and comparison with not only the Transformer architecture but also kernel methods, explain the benefits of gating.

### Weaknesses
- Although not a major weakness, the distributional assumptions are strong: Gaussian single-index models with polynomial links and Hermite features.
- The experiments can be conducted over more range of intrinsic dimensions or sample sizes.

### Questions
- Which steps in the argument strictly require the Gaussian inputs and a polynomial link? Can this work be extended to non-Gaussian, or non-polynomial cases, with heavy-tailed noise?
- How do the required prompt lengths and task counts scale as the intrinsic dimension increases?

### Soundness
3

### Presentation
3

### Contribution
3

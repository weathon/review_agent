# Enhancing Zeroth-Order Fine-Tuning for LLMs via Gradient-Guided Subspace Selection

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 2, 4, 6, 6

## Abstract
As a promising memory-efficient technique, zeroth-order (ZO) optimization enables large language models (LLMs) to bypass costly backpropagation during fine-tuning by estimating gradients through function evaluations. However, to minimize approximate variance in high-dimensional parameter spaces, existing ZO methods focus on exploring the estimate of gradients within random subspaces, neglecting the benefits of searching for more accurate subspaces of LLMs on gradient estimates. Due to inaccurate gradient estimates obtained from random spaces, fine-tuning performance is inevitably degraded, thus compromising the performance of downstream tasks. To address the limitation of existing ZO methods, this paper proposes a novel ZO subspace fine-tuning method named *SVD-0*. Based on singular value decomposition (SVD), SVD-0 can effectively obtain more accurate subspace projection matrices, which can be used to improve the accuracy of gradient estimates. Experimental results on various complex language modeling tasks show that SVD-0 achieves better fine-tuning performance and faster convergence than state-of-the-art ZO methods.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes SVD-0, a zero-order (ZO) optimization method for LLM finetuning that aims to achieve more accurate gradient estimates while retaining the variance reduction benefits of subspace methods. Specifically, SVD-0 applies singular value decomposition (SVD) to the gradient estimates derived from an existing ZO optimizer, thereby constructing principled subspace projection matrices. The superiority of SVD-0 over various ZO methods is demonstrated by extensive experiments on a variety of language modeling tasks across model sizes and architectures.

### Strengths
* The paper is well-written, with a comprehensive literature review and clearly articulated motivations for the proposed algorithmic design.
* The proposed method exhibits strong empirical performance across diverse tasks, model sizes, and architectures, supported by extensive experimental evidence. Furthermore, the authors provide a detailed investigation of the effects of algorithmic hyperparameters.

### Weaknesses
1. The primary concern lies in the substantial overlap with SubZero (Yu et al., 2024), in both algorithmic design and theoretical analysis. The paper would be significantly strengthened by a clearer delineation of its novel contributions. 
    - The overall procedure of SVD-0  is highly similar to SubZero, except for the acquisition of matrices $U$ and $V$ in (2). However, several shared components, such as layer-wise perturbations and periodic updates, are also included in the claimed contributions, which may be an overstatement.
    - As is acknowledged in the paper, the theoretical derivation follows the same approach used in (Yu et al., 2024). 

2. There are issues in the statement and proof of Theorem 1. The theorem statement does not clearly distinguish the target optimization error $\epsilon$ from the perturbation scale $\varepsilon$. As a convention, the target accuracy should be an arbitrary tolerance, rather than the designated perturbation scale. Moreover, in the proof of Theorem 1 in Appendix C, a factor $1/T$ is missing before the summation at line 1126, which affects the final complexity expression. 
3. The empirical justification for the claimed advantages of the SVD-derived subspace could be more direct. To substantiate the claim of improved gradient accuracy, it would be beneficial to provide a direct comparison of the cosine similarity between the full gradient estimates and the true first-order gradients, rather than solely comparing their singular value vectors. Additionally, a quantitative analysis of the gradient estimate variance is needed to empirically substantiate the claim that the method achieves better variance reduction than random subspace approaches.
4. Since Adam is more common than SGD in practical fine tuning, it is recommended to apply the proposed method together with Adam in order to strengthen the practical contribution.
5.  There are several formatting issues. 
    - Tables 6, 7, 8, and 9 are placed too close to the main text.
    - There is an incorrect figure reference on line 125. It should be Figure 1 instead of Figure 3.
    - There is a typo in (3). The term $f(\theta)_t$ should be $\nabla f(\theta_t)$. 


Reference
Ziming Yu, Pan Zhou, Sike Wang, Jia Li, and Hua Huang. Subzero: Random subspace zeroth-order optimization for memory-efficient llm fine-tuning. arXiv preprint arXiv:2410.08989, 2024.

### Questions
1. Could the authors include dominant parameter-efficient fine-tuning baselines, such as LoRA and its advanced variants, to provide a clearer comparison against other widely-used memory-efficient adaptation methods? Additionally, have the authors explored integrating the proposed method with LoRA?
2. Could the authors provide training loss curves plotted against wall-clock time to compare the empirical convergence of SVD-0 against SubZero? While the paper establishes matching theoretical convergence rates in terms of iteration steps, SVD-0 incurs a higher per-step computational cost due to the SVD operation. This wall-clock time comparison is essential to determine which method converges faster in actual training time. 
3. How sensitive is the SVD-0 framework to the choice of the underlying ZO gradient estimator? Testing SVD-0 with other ZO estimators would clarify if this is a general subspace strategy or a heuristic tightly coupled to MeZO's specific estimation properties.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents SVD-0 as a memory-efficient method for fine-tuning large language models using zeroth-order (ZO) optimization. It improves gradient estimation by periodically applying singular value decomposition (SVD) to ZO gradients. The authors claim that this creates accurate layer-wise projection matrices to capture the most important optimization directions, thereby reducing the high variance commonly seen in earlier methods such as MeZO, SubZero, and LOZO. As a result, SVD-0 delivers outstanding performance on benchmarks like SuperGLUE tasks, applicable to models including OPT (1.3B/13B), RoBERTa-large, and Qwen-1.8B. It maintains low memory usage and provides proven convergence.

### Strengths
The paper is well-motivated. Furthermore, the narrative is remarkably clear and easy to follow, which makes the technical content highly accessible.

### Weaknesses
1) While the authors present a well-motivated goal of leveraging gradient information to guide parameter updates more effectively, their method relies entirely on historical zeroth-order gradients derived from MeZO. It is important to note that zeroth-order gradients diverge significantly from true gradients, and accurate gradient estimation under a single SPSA perturbation remains highly challenging[1]. As a result, although the intention is sound, the technical execution does not appear to constitute a substantial breakthrough.

2) the claimed contribution seems incremental when compared to SubZero[2]. The core distinction lies in the construction of the low-rank subspace: whereas SubZero employs random projection for singular value decomposition, SVD-0 utilizes historical zeroth-order gradients from MeZO. However, since these historical gradients essentially form a scalar-projected random matrix, performing SVD on them would only affect the singular values—not the orthogonal matrices. This implies that SVD-0 and SubZero are functionally equivalent as stochastic zeroth-order optimizers in subspace projection.

3) This work can be viewed as a minor variation of SubZero from a theoretical perspective, where the novel theoretical contributions remain unclear.

4) In ICLR submission，use \citet when the author is the subject of the sentence. Use \citep for all other citations, see line331-342.

[1] Malladi, Sadhika, et al. "Fine-tuning language models with just forward passes." Advances in Neural Information Processing Systems 36 (2023): 53038-53075.

[2] Yu, Ziming, et al. "Subzero: Random subspace zeroth-order optimization for memory-efficient llm fine-tuning." (2024).

### Questions
see the Weaknesses.

### Soundness
2

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
This paper introduces SVD-0, a zeroth-order (ZO) optimization method for improving large language model (LLM) fine-tuning by exploiting the low-rank structure of gradients. Motivated by the observation that ZO gradients share similar spectral properties with first-order (FO) gradients, the authors propose to extract a low-rank subspace via SVD on the approximated ZO gradients and restrict perturbations to this subspace. This approach can help reduce the approximation variance of ZO gradient estimation, thereby achieving more accurate gradient approximation and improved fine-tuning performance. Extensive experiments on a range of LLMs and benchmarks are provided to demonstrate the effectiveness of SVD-0 compared to prior ZO methods such as MeZO, SubZero, and LOZO.

### Strengths
The paper presents a well-motivated extension of GaLore-style low-rank subspace projection (Zhao et al., 2024) to the zeroth-order (ZO) optimization framework. By incorporating low-dimensional structure of gradients without backpropagation, it effectively improves the efficiency of ZO gradient estimation compared to prior methods that use random subspace directions. The experimental results are well designed.

### Weaknesses
While the preliminary study suggests that ZO gradients share similar spectral properties with exact FO gradients, the evidence given is somewhat limited.  Additional analysis would make the argument more convincing. The algorithm description is also unconvincing, particularly regarding the use of different perturbation matrices (see Questions below). The performance improvement over prior methods appears relatively limited.

### Questions
- In the prestudy of Section 3, could the authors provide more details about the experiments, such as the chosen rank and how the cosine similarity is computed (i.e., what the x-axis in Figure 1 represents and whether the cosine similarity is averaged across all singular vectors)? The result shows a cosine similarity around 0.45, but is this high enough to claim that the estimated ZO and FO gradients are similar? How does the similarity change for higher-order singular vectors and with more batches used for ZO estimation?

- In line 19 of Algorithm 3, the weights are updated with a new perturbation matrix, which is inconsistent with the SPSA in MeZO formulation where the same perturbation direction should be used for both gradient estimation and parameter update. Intuitively it doesn't really make sense to measure the difference of function values along one direction but apply the update in another direction. Could the authors clarify on this?

- The authors mention that the additional time required for SVD operations is negligible even though the time complexity of SVD is $O(n^3)$. Could the authors clarify on this?

- In Eq (3), should $U_t^\top f(\theta)_t V_t$ be $U_t^\top \nabla f(\theta_t)V_t$? Also, some symbols like $\theta$, $U$, and $V$ are written in non-bold fonts which is inconsistent with the rest of the paper.

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
3

### Summary
The paper proposes SVD-0, a zeroth-order (ZO) fine-tuning method that replaces random low-dimensional subspaces with gradient-guided ones computed via singular value decomposition (SVD) of ZO gradient estimates. Prior ZO subspace (low-rank) methods constrain perturbations to random low-rank subspaces, but their arbitrary projection matrices can misalign with the gradients’ intrinsic low-rank structure; meanwhile conventional ZO like MeZO suffers from high variance in billion-parameter spaces. SVD-0 periodically (every $F$ steps) computes layer-wise projection matrices $(U,V)$ from the estimated gradients, then perturbs parameters with low-rank updates $\tilde Z = U Z V^\top$ to improve gradient estimation accuracy while preserving ZO’s memory efficiency with minimal overhead.

### Strengths
1. The pre-study clearly demonstrates that zeroth-order (ZO) gradient estimates preserve the low-rank structure of the true gradients. This provides empirical motivation for constructing projection matrices from these estimates rather than from random perturbations.
2. The method design is well organized: (1) SVD-based acquisition of gradient-guided projection matrices $(U, V)$ from estimated gradients, and (2) generation of low-rank perturbations $\tilde{Z} = U Z V^T$ within these subspaces. Algorithm 1–3 explicitly describe this process.
3. Experimental results show SVD-0 outperforming MeZO, SubZero, and LOZO across multiple models and benchmarks. 
4. The method maintains the low-memory footprint of zeroth-order optimization. 
5. Provides theoretical analysis of convergence.

### Weaknesses
1. Standard deviations are missing for OPT experiments, even though they are provided for RoBERTa-large.
2. Line 125 references Figure 3, when I think it’s meant to refer to Figure 1.
3. The computational efficiency of SVD-0 can become a bottleneck for larger models due to the high time complexity of the SVD operation. While Table 1 should specify the model and experimental settings used for comparing the computational costs of different methods, it is also important to report these results on larger models (with 7B+ parameters).

### Questions
Please refer to the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
LLMs are large networks, and even first-order gradient descent can be too computationally and memory-intensive to finetune. Zeroth-order (ZO) skips backprop by probing the model with tiny nudges and seeing the loss change. This saves memory, but the “which way to move” signal is very noisy in huge models.

The paper's key idea is that, instead of searching in random directions, they perform an SVD to find a few strong directions, then explore mostly within that tiny set. Specifically, they 1) get a ZO gradient estimate using a standard method 2) run SVD on that estimate to get two small matrices $U, V$. They define a low-rank subspace. 3) Conduct updates with low-rank perturbations $U Z V^{\top}$ instead of full random noise. Then they recompute $U, V$ every F steps to balance compute and adaptability.

This works because ZO and true gradients share structure, so SVD on the ZO estimate reveals helpful directions. We can then search mostly where progress actually happens.

The paper shows that, in tests on SuperGLUE with OPT-13B and 1.3B, SVD-0 is consistently among the top ZO methods and often best overall versus MeZO and others.

SVD is $O\left(n^3\right)$ and it can be expensive. But they show that the overhead stays low, costing like 7% longer only. This is possible becauses they do 1) Layer-wise SVD, 2) Periodic refresh. and 3) No per-step extra work.

### Strengths
I was thinking that the observed ZO $\iff$ FO spectral alignment was interesting; it makes the SVD-guided subspace a reasonable lever. I believe their method's value is low integration cost and stable top-2 placement.

The cost of the proposed method is not significant, indicating it is practical.

### Weaknesses
The biggest weakness of this paper is the modest empirical gains it achieves relative to state-of-the-art baselines. Some margins are really narrow. The improvements are consistent but at best modest on core benchmarks.

One small caveat is that projection quality depends on ZO gradient precision, so that smaller models can yield noisier $U$ and $V$. But I believe zeroth-order gradients are mostly for tuning large networks.

### Questions
1. I wonder how the cost scales from 1.3B to 13B to 70B parameters.

2. I wonder how this method would work under quite noisy ZO gradients. 

3. How can you pick $r$ and $F$ automatically from live signals?

4. How often does the subspace become stale on domain shifts?

### Soundness
2

### Presentation
2

### Contribution
2

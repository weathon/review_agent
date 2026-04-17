# LoRA meets Riemannion: Muon Optimizer for Parametrization-independent Low-Rank Adapters

- Decision: Accept (Poster)
- Scores: 4, 4, 2, 6, 4

## Abstract
This work presents a novel, fully Riemannian framework for Low-Rank Adaptation (LoRA) that geometrically treats low-rank adapters by optimizing them directly on the fixed-rank manifold. This formulation eliminates the parametrization ambiguity present in standard Euclidean optimizers. Our framework integrates three key components to achieve this: (1) we derive **Riemannion**, a new Riemannian optimizer on the fixed-rank matrix manifold that generalizes the recently proposed Muon optimizer; (2) we develop a Riemannian gradient-informed LoRA initialization, and (3) we provide an efficient implementation without prominent overhead that uses automatic differentiation to compute arising geometric operations while adhering to best practices in numerical linear algebra. Comprehensive experimental results on both LLM and diffusion model architectures demonstrate that our approach yields consistent and noticeable improvements in convergence speed and final task performance over both standard LoRA and its state-of-the-art modifications.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes a fully Riemannian LoRA framework that optimizes the low-rank adapter directly on the fixed-rank manifold to remove factorization ambiguity. It proposes a Muon-style Riemannian optimizer (“Riemannion”) plus a locally optimal initialization. Experiments on commonsense reasoning and subject-driven diffusion report gains over LoRA variants.

### Strengths
1. Clear motivation for moving from factor-space updates to invariant steps on fixed rank manifold
2. Concrete algorithms: a step-by-step Riemannian procedure with Ortho/Project components and explicit per-iteration complexity
3. Experimental results suggest improved stability/accuracy vs. strong LoRA-style baselines (though see concerns below)

### Weaknesses
(Please reply to the Questions section directly, where I write the details of the weaknesses)
1. Insufficient positioning and no empirical comparison to LORO
2. Unclear and potentially confusing use of G in Eq. (4).
2. Missing memory-consumption analysis (vs. LoRA/PEFT)
4. Hyperparameter protocol is opaque near the main tables

### Questions
1. I think the idea of applying direct optimization on low rank manifold for LLM training is already considered in LORO [1]. Although LORO is for pre-training, it can be applied to fine-tuning very easily (with a fixed pre-trained model). I'm wondering how the proposed method perform conparing to this variant of LORO? Since LORO is the first to consider this proposed approach of low rank manifold optimization, I feel the discussion is not sufficient in the current manuscript to compare to it.
2. Clarity around G in Eq. (4): Readers may be confused because $G$ appears in the manifold parameterization but seems to disappear in later sections. It remains unclear which quantities are actually treated as trainable parameters in Algorithm 4 and which are only auxiliary. Please state explicitly what is stored, what is recomputed during retraction, and what constitutes the model’s trainable state.
3. No memory footprint comparison: The paper measures time overheads but omits GPU memory / parameter-state comparisons vs. LoRA, DoRA, etc. A table counting trainable params and optimizer states per layer (e.g., Adam’s m/v vs. Riemannion’s HB state) would strengthen claims of “no prominent overhead.” This is an important piece of information that is missing, since LoRA is designed to be a parameter or memory efficient optimizer.
4. For Table 1, is it exactly one set of hyperparameters used for Riemannion to achieve all these superior performance **across all tasks**? If so, I think this would be quite astonishing. If not, would it be a bit unfair if the authors didn't search hyperparameters for other methods?
5. A minor point: The geometry section is readable, but several practical choices (which Ortho operator was used in which layer types; retraction accuracy vs. rank; momentum transport details) are scattered. I suggest consolidating everything into an implementable algorithm, not just one step as in Algorithm 4.

References:

[1] Mo, Zhanfeng, Long-Kai Huang, and Sinno Jialin Pan. "Parameter and memory efficient pretraining via low-rank riemannian optimization." The Thirteenth International Conference on Learning Representations. 2025.

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
This paper proposed a novel Muon-style LoRA vairant through the lens of Riemannian optimization. Specifically, the authors proposed to apply the retraction-based Riemannian gradient update step (Eq. 8) to optimize LoRA factors, and augment the momentum term involved in the update step with the Muon-style orthorgonalization (Eq. 12). Empirically, the proposed method achieves decent performance on LoRA-style language model finetuning tasks (Table 1).

### Strengths
- The authors propose a computationally efficient Muon-style extension of Riemannian optimization for LoRA parameterized LLMs (Algorithm 1).
- The authors derive locally optimal initialization scheme to maximize the the LoRA factors can be optimized towards the fastest loss decrease direction on the low-rank manifold (Theorem 5.1).
- The authors develops single backward-pass gradient trick, to compute gradient-times-matrix products efficiently (Algorithm 3).

### Weaknesses
This paper is well-written. The proposed method is theoretically grounded, and empirically validated on several downstream tasks. I will raise my score upon my concerns being addressed.


--------------

**Concern 1. Feasible Region of (Eq 14)**

To my understanding, the authors aim to initialize the LoRA factors such that it can be optimized towards the fastest loss decrease direction on the low-rank manifold. In this case, does that means we should constraint the norm (or magnitude) of the LoRA factors in Eq. 14? Otherwise, scaling the LoRA initializaiton will lead to larger Riemannian gradients. 

--------------

**Concern 2. Ablation on LOI**

From my experience, the intialization of LoRA factors is usually chosen to ensure the adapted weight is identical to the loaded pretrained model, ensuring the model does not deviate from the well-trained local optima significantly. However, it seems that in LOI, a non-zero modification is made to the adapted weight. Does this affect the performance of the model? I recommend the authors to add discussion on this issue, and provide ablation studies to clarify the impace of LOI on the general performance of the proposed method.

--------------

**Concern 3. Effectiveness of the proposed methods on classic low-rank matrix optimization.**

To my understanding, the derivation of the proposed methods does not relies on specific assumptions on the optimization task. I recommend the authors to provide additional experiments on classic Riemannian optimization tasks to validate the effectiveness of the proposed methods in solving classic low-rank matrix optimization problems [1] that are more general and natural than the LLM finetuning tasks.

[1]. Bioli, I., Kressner, D., & Robol, L. (2025). Preconditioned low-rank Riemannian optimization for symmetric positive definite linear matrix equations. SIAM Journal on Scientific Computing, 47(2), A1091–A1116. https://doi.org/10.1137/24M1688540

### Questions
See **Weaknesses**.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper extends the Muon optimizer to fixed-rank matrix manifolds, proposes a Riemannion algorithm, and introduces an initialization strategy based on Riemannian gradient information. Finally, it demonstrates the performance of Riemannion on LLM fine-tuning and diffusion models.

### Strengths
This is the first work to apply the Muon optimizer to low-rank matrix manifolds, combining Muon with Riemannian manifolds.

### Weaknesses
1. Insufficient motivation in lines 63–69. 

$\cdot$ The authors present the main motivation as extending Muon, but do not sufficiently explain the necessity of this extension. To strengthen the argument, the core advantages of Muon should be outlined to clarify the specific benefits of applying it to fixed-rank manifolds. Furthermore, a comparison with existing manifold optimizers is essential to emphasize the strengths of the proposed method.

$\cdot$ The statement, `our design inherits Muon’s geometry-aligned normalization, yielding transformation invariance of the learned update', lacks theoretical or empirical support in the paper. It should be supported by providing proofs or references, or by experiments validating.

$\cdot$ To demonstrate the advantages of Riemannian gradient–informed initialization, they should compare different initialization strategies for the same algorithm through experiments to showcase the benefits of their proposed initialization.

2. In lines 160-161, the authors claim that `Note that acting on the two factors separately makes Muon non-reparameterization-invariant: its per-factor orthogonalization depends on arbitrary scalings or rotations, skewing the weight-space step and often letting one factor dominate.' Please provide the proof or detailed reasoning to support this claim.

3. In lines 168–171, the symbol G is already used earlier in the paper to represent the gradient. Reusing G in this section creates confusion. 

4. Please provide detailed derivations or references to support (5).

5. In line 188, the objective function is denoted as F, but is changed to L in Line 215. This inconsistency in notation should be resolved for clarity.

6. The complexity and scalability of Algorithm 2 need clarification. If a full SVD is performed on an m×n matrix at each step to determine 
$A_L$ and $B_R$, the computational cost is typically 𝑂(𝑚𝑛 min{𝑚, 𝑛}). The algorithm requires computing a full SVD at each step, which becomes extremely expensive when \(m\) or \(n\) is very large, making the algorithm inefficient in these cases.

7. In line 80,  `we show the connection of this initialization to LoRA-GA', but this paper does not explain the differences and connection between the proposed initialization and LoRA-GA.

8. The experiments only report results for r=4,8,16. It should include results for larger ranks (r=32,64, or even higher) to evaluate the stability, performance, and computational overhead of the proposed algorithm. 

9. The experimental results in Figure 4 regarding runtime provide limited insights. It would be more informative to include a comparison of the per-step computation time and memory usage of the proposed algorithm versus state-of-the-art algorithms.

10. The paper does not provide any theoretical guarantees for the convergence of the proposed algorithm. Including such guarantees, or at least a discussion on convergence properties, would strengthen the paper.

### Questions
as stated in weakness

### Soundness
2

### Presentation
1

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
This paper proposes a Riemannian optimization framework for Low-Rank Adaptation (LoRA) that directly optimizes low-rank adapters on the fixed-rank manifold instead of in standard Euclidean space. The approach introduces the Riemannion optimizer(a variant of Muon optimizer), along with a Riemannian gradient-informed initialization. Experiments on both large language models and diffusion architectures demonstrate improvements in convergence speed and performance compared to relevant methods.

### Strengths
The paper is well-written and provides solid mathematical formulation such as gradient project,retraction and vector transports via automatic differentiation.

This paper derives Riemannion, the first optimizer that generalizes Muon to manifold of fixed rank matrices, addressing a fundamenta issue in LoRA traiing that different factorizations A,B lead to different optimization trajectories.

Empirical experiments demonstrate the proposed method outperform baselines across LLM an diffuision benchmarks.

### Weaknesses
The main novelty(generalizing Muon) is somehow incremental.

LLM results focus on Llama-3 8B using rank 16 and commonsense benchmarks only. No ablations for ranks, downstream tasks (e.g. summarization,instruction tuning), ViT models.

The computational overhead need to be clarified.

### Questions
See weakness

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper identifies a key deficiency in standard Low-Rank Adaptation (LoRA) training: the lack of transformation invariance. The authors note that the update to a LoRA matrix $\Delta W$ is dependent on its specific factorization ($A$, $B$), which can lead to unstable training and sub-optimal results.

To solve this, the authors propose a novel, fully Riemannian framework. Instead of optimizing the ambiguous factors ($A$, $B$) in Euclidean space, this work proposes to optimize the low-rank matrix $\Delta W$ directly on the fixed-rank manifold $\mathcal{M}_r = \{X : rank(X) = r\}$.

### Strengths
1. Addresses a Fundamental Problem: The paper tackles a well-defined and important problem in parameter-efficient fine-tuning. The lack of transformation invariance is a genuine flaw in the standard LoRA optimization paradigm, and addressing it is a valuable contribution.
2. Conceptually Elegant Solution: By re-framing the optimization problem on the fixed-rank manifold, the framework eliminates the root cause of the invariance problem by construction, as the ambiguous factors ($A$, $B$) are no longer part of the optimization. 
3. Novel Optimizer: The generalization of the Muon optimizer to the fixed-rank manifold ("Riemannion") is a novel algorithmic contribution.
4. Thorough Handling of Efficiency: A primary concern for any manifold-based method is computational cost. The authors anticipate this and provide a convincing case for their method's efficiency. The design explicitly avoids forming full-size matrices, with a theoretical complexity of $\mathcal{O}((m+n)r^{2}+r^{3})$.

### Weaknesses
1. Insufficient Comparison to SOTA (LoRA-RITE): The paper's primary weakness is its engagement with its most direct baseline, LoRA-RITE (Yen et al., 2024).

2. Conceptual Comparison: While LoRA-RITE is cited for solving the invariance problem whilst conducting adaptive regularization, the paper misses a clear opportunity to discuss why its geometric approach should be theoretically superior to LoRA-RITE's preconditioning approach. The review a-priori is that a geometrically-native solution should be more stable or effective, but this is not argued explicitly. A wall-time comparison with LoRA-RITE would be great to have. 

3. Mismatch in Optimization Space: The paper defines its optimization space as the manifold $M_r = \{X : rank(X) = r\}$. However, the true set of all possible LoRA adapters is $M_{\le r} = \{X : rank(X) \le r\}$. The paper does not discuss the implications of this distinction. The set $M_{\le r}$ has "singularities" at all points where $rank < r$, and it is unclear how the optimizer behaves if the true optimal solution lies on one of these singularities. The $SVD_r$ retraction (Alg. 4, line 5) explicitly forces the update to stay on the $\mathcal{M}_r$ manifold, which could potentially prevent the model from finding a simpler, lower-rank solution.

4. The authors can mention an article on a related topic, specifically the "manifold Muon" optimizer described in the Modular Manifolds [1] article from Thinking Machines Lab. This article discusses a similar recipe (a Muon-like optimizer on a manifold) but applies it to the Stiefel manifold. A discussion of this related work is essential for correctly positioning the paper's novelty.

minor: typo in line 215

[1] Jeremy Bernstein, "Modular Manifolds", Thinking Machines Lab: Connectionism, Sep 2025.

### Questions
Could you elaborate on the conceptual advantages of your geometric framework over the adaptive matrix preconditioning approach of LoRA-RITE? Why is optimizing on the manifold better than forcing invariance in Euclidean space?

### Soundness
3

### Presentation
3

### Contribution
2

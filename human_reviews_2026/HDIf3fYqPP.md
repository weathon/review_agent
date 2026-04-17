# Null-Space Filtering for Data-Free Continual Model Merging: Preserving Stability, Promoting Plasticity

- Decision: Accept (Poster)
- Scores: 6, 8, 4, 4

## Abstract
Data-free continual model merging (DFCMM) aims to fuse independently fine-tuned models into a single backbone that evolves with incoming tasks without accessing task data.
This paper revisits two fundamental desiderata for DFCMM: stability, avoiding interference with earlier tasks, and plasticity, adapting faithfully to each new task. This poses a challenge that existing approaches fail to address: how to bridge data-level desiderata with parameter-space optimization to ensure stability and plasticity in the absence of task data.
To this end, we propose NUFILT(NUll-space FILTering) a data-free framework that directly links these desiderata into parameter-space optimization. Our key observation is that task vectors approximately align with representation subspaces, providing structural surrogates for enforcing stability and plasticity. 
Accordingly, we design a null-space projector that preserves prior responses by filtering overlapping components of new task vectors, ensuring stability.
We further introduce a lightweight LoRA adapter that injects complementary task-specific signals to enable plasticity.
The adapter is trained with a projection-based surrogate loss that preserves consistency with prior knowledge while introducing novel directions. 
This joint filtering–adaptation process enables the backbone to absorb new knowledge while retaining existing behaviors, with updates fused back in a layer-wise linear fashion without extra parameters or inference cost.
Theoretically, we establish approximate subspace alignment guarantees that justify null-space filtering. Empirically, NUFILT achieves state-of-the-art performance with minimal forgetting on both vision and NLP benchmarks, improving average accuracy by 4–7% over OPCM and WUDI-Merging, while narrowing the gap to fine-tuning and reducing computation overhead. The code is available at: https://github.com/zihuanqiu/NUFILT

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes Null-space Filtering to address the data-free continual model merging problem: sequentially merging multiple fine-tuned models into a single backbone to cover all task label spaces, without the original training data and only with access to the current task model and the merged backbone network. The authors formalize the problem as two data-level desiderata: transparency and fidelity, and propose a strategy to map these data-level objectives into parameter space: based on the observation that the main subspace of the task vector is approximately aligned with the representation subspace. The paper provides detailed vision and NLP experiments showing that NUFILT outperforms multiple recent baseline methods in ACC and BWT metrics.

### Strengths
1. The writing is quite easy to read and it was well-written
2. The idea of ​​constructing a null-space projection based on empirical findings is both intuitive and mathematically connected, providing a reasonable path for "replacing data supervision with parametric geometric structures in the absence of data."
3. The paper studies multiple base models
4. The gains from their proposed method of sampling are convincing and comprehensive

### Weaknesses
1. The use of Null Space has been applied in many reccent works, such as [1] [2] [3], so the contribution of the work is incremental. The author should further discuss the relation between this works.
2. The paper uses quantities such as singular vectors representing the data, the largest singular value of a matrix, and σ1(X) to derive bounds (e.g., Corollary 1). However, in the strict case of data obscurity, how these quantities are estimated/upper bounded in practice needs to be more clearly explained. The current bridge between theoretical assumptions and practical implementations is not transparent enough.
3. The paper claims to use a set of global hyperparameters, but does not provide guidance on hyperparameter adjustment for different backbones or longer sequences; in addition, it is unclear how the number of training iterations, rank selection, etc. scale to different model sizes.




[1] Alphaedit: Null-space constrained knowledge editing for language models

[2] Multi-task model merging via adaptive weight disentanglement

[3] Orthogonal Subspace Learning for Language Model Continual Learning

### Questions
1. What are the results of NUFILT when apply to larger scale models like qwen and llama ?
2. Wiil the initialization of LoRA affect the merging result。

If the author can solve the question and the weakness well, i will raise my score.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper proposes NUFILT for data-free continual model merging (DFCMM). Each new task’s update (task vector) is first projected onto the null space of a low-rank subspace spanned by the previously merged updates. Then a lightweight LoRA term is trained (using a projection-based surrogate objective) to recover task-specific signal. Finally, the projection and LoRA are fused back into the backbone so no extra parameters remain at inference. The paper motivates the projection with an “approximate alignment” claim between task-vector subspaces and data-representation subspaces, and reports gains over WA/TA, OPCM, WUDI-Merging and others on CLIP vision suites and GLUE with Flan-T5.

### Strengths
- The paper present a simple, practical recipe. The three-step pipeline is easy to follow. 
- The proposed method elegantly combines null-space projection with parameter-efficient LoRA adaptation, supported by solid mathematical reasoning and empirical evidence of subspace alignment. The approach is conceptually simple yet technically sound.
- Experiments are thorough and well-controlled, covering vision and language tasks, multiple architectures, and extensive baselines. The ablations and sensitivity analyses are carefully executed, providing convincing evidence of robustness and practical usability.

### Weaknesses
I overall like this paper, which is clearly written and technically rigorous. I have only few questions.

1. In Fig. 4(a), the accuracy drops when introducing a small projection rank 
(r_p = 1-4), even lower than using no projection (r_p = 0). Could the authors clarify why a small null-space rank initially hurts performance? Additionally, backward transfer first decreases and then improves as r_p increases, what causes this non-monotonic behavior?

2. The subspace affinity is defined as $A(V_d,\hat V)=\frac1{r_d}\|\hat V^\top V_d\|_F^2\in[0,1]$ , but Theorem 1 defines $A(V_d,\hat V)=1-\frac1{r_d}\|\hat V^\top V_d\|_F^2$ and claims $A\le\zeta^2$ (Eq. 6). I might have misunderstood. Please check this.

3. The method repeatedly requires SVDs per layer on $\tilde\tau_{\le t-1}$ and on $\tau_t$  but runtime in Table 4 only reports the adaptation loop, not the SVD cost. 

4. Although the robustness analysis is informative, the chosen global ranks might still require adjustment for different model scales. A short discussion on adaptive or automatic rank selection would strengthen the paper’s practical perspective.

### Questions
see weekness

### Soundness
4

### Presentation
4

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
This paper proposes NUFILT (NUllspace FILTering), a lightweight, data-free adaptation framework for Data-Free Continual Model Merging (DFCMM) aimed at improving transparency and fidelity. The authors argue that task vectors align with data representation subspaces and provide theoretical guarantees for approximate subspace alignment. Empirically, NUFILT reports 4–7% gains in average accuracy over prior work, with minimal backward transfer (i.e., limited forgetting of previously learned tasks), and requires only 50 optimisation steps.

### Strengths
(1) Leveraging null-space filtering for data-free continual model merging is an interesting novel idea.

(2) The adaptation strategy is lightweight and inexpensive, making it broadly applicable in data-restricted settings.

(3) Empirical results validate that NUFILT improves average accuracy while maintaining low forgetting.

### Weaknesses
(1) The paper presents transparency and fidelity as distinct desiderata, but they are closely related and longstanding goals in continual learning. As such, their framing does not feel like a novel contribution.

(2) Despite emphasizing transparency and fidelity, the paper does not quantitatively track or compare these properties for NUFILT vs. baselines.

(3) Evaluation is limited to relatively older architectures (CLIP and T5). Results on modern LLMs would materially strengthen the paper’s relevance.

(4) The work also lacks comparison against more recent merging techniques, such as ISO-Merging[1], KnOTS[2], TSV-M[3] which also use the SVD decomposition to align task vectors.

(5) Notation and clarity issues:

(a) Terms are introduced without definition (e.g., “representation covariance” at L214; use of $\sigma$ at L249).

(b) Overloaded symbols (e.g., T in Algorithm 1).

(c) Non-standard or unclear formulations (e.g., Eq. (3–4): the product of X with task vectors is not well specified).

(d) Undefined acronyms (e.g., ECDF).

Collectively, these hinder readability and reproducibility.

References:

[1] Marczak, D., Magistri, S., Cygert, S., Twardowski, B., Bagdanov, A. D., & van de Weijer, J. (2025). No task left behind: Isotropic model merging with common and task-specific subspaces. arXiv preprint arXiv:2502.04959.

[2] Stoica, G., Ramesh, P., Ecsedi, B., Choshen, L., & Hoffman, J. (2024). Model merging with svd to tie the knots. arXiv preprint arXiv:2410.19735.

[3] Gargiulo, A. A., Crisostomi, D., Bucarelli, M. S., Scardapane, S., Silvestri, F., & Rodola, E. (2025). Task singular vectors: Reducing task interference in model merging. In Proceedings of the Computer Vision and Pattern Recognition Conference (pp. 18695-18705).

### Questions
(1) Hyperparameter sensitivity: Fig. 4 suggests sensitivity to $r_p, r_l, r_v$. In a data-free setting, how were these tuned? Is there a data-free or proxy procedure (e.g., curvature/NTK proxies, spectral heuristics, or stability criteria) to select them?

(2) Rank assumption: L296 assumes $r_v \geq r_d$, yet the implementation fixes it to a very low value $r_v = 8$ which may not guarantee the above assumption is met. Does this have any implications?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents NUFILT, a data-free continual model merging approaches with good empirical performance and theoretical grounding.

### Strengths
1. The writing and formatting are good and clear.
2. The authors provide detailed and intuitional theoretical derivations to arrive at the final solution.
3. NUFILT achieves good empirical performance on both vision and NLP tasks, outperforming previous train-free and train-required merging methods on this task.

### Weaknesses
1. Potential unfair baseline comparison: The baseline train-free merging methods (e.g. TA, TIES) are implemented with theta_merged_t = theta_merged_t-1 + lambda tau_t. This implementation biases the model towards the early task vectors. For example, the optimal lambda for the first task is clearly 1.0, while later tasks have gradually reducing optimal lambda values. This probably explains the result of task arithmetic (Table 1) having lower performance than even weight averaging. I think the more fair comparison to these approaches should include a downscaling of previously merged task vectors (similar to the implementation of weight averaging), that is: theta_merged_t = (t-1)/t * theta_merged_t-1 + (1/t) * theta_0 + (lambda / t) * tau_t.

2. Comparison to OPCM. The first step of NUFILT resembles OPCM, i.e., they both try to project the current task vector to the orthogonal space of previous merged vector. Given the similarity on the intuition, a more detailed comparison should be added to explicitly demonstrate the differences and novelties, especially on how it addresses the criticism to OPCM that it struggles when task vectors are inherently entangled or non-orthogonal (section 3.2).

3. Introduced new terminologies: I do not agree it is necessary or worth to introduce the new terms transparency and fidelity, when they are just new names for the classical Stability-Plasticity trade-off in continual learning. They both measure the adaptiveness to new task and forgetting on previous tasks. Furthermore, the first contribution in the introduction that “formulating these two desiderata frames a new open challenge absent from prior work” seems an over-claim.

4. Computational cost to train-free methods: Table 4 only compares the computational cost to the training-required baselines. It is necessary to compare to the training-free baselines (which have way lower computational cost) as well to demonstrate the performance-computation trade-off.

5. Scalability to larger models: The paper demonstrates that it can scale to ViT and T5 models, but did not provide results with larger models (e.g., Llama), is it possible to scale to even larger language models given the training-required procedure?

### Questions
1. Using LoRA. One key benefit of LoRA is that it updates only a lightweight set of parameters. In this case, the LoRA parameters are going to be incorporated to the base parameters, thus this advantage is lost. Is it necessary to use LoRA then, or maybe it still brings some training regularization?
2. The empirical results demonstrate good accuracy but relatively poorer backward transfer (Table 1). Have you tried using a weighted sum for the two objectives in Equation 16 to enforce less forgetting?
3. From the description of OPCM [1], the correct citation of the vision benchmarks should be [2] and [3].

[1] Merging Models on the Fly Without Retraining: A Sequential Approach to Scalable Continual Model Merging

[2] Editing Models with Task Arithmetic

[3] Localizing Task Information for Improved Model Merging and Compression

### Soundness
3

### Presentation
2

### Contribution
2

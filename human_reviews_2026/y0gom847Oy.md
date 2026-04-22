# Gram-weighted Mahalanobis Fréchet Mean: A Hyperparameter-Tuning-Free Solution for Model Merging

- Avg Score: 5.33
- Decision: Reject
- Scores: 4, 6, 6

## Abstract
Model merging has emerged as a promising technique for integrating multiple fine-tuned models into a single unified model without additional training. This paradigm is particularly appealing in resource-constrained scenarios where access to data or retraining is limited. Existing techniques—such as Task Arithmetic, Ties-Merging, and AdaMerging—achieve competitive results but typically rely on extensive hyperparameter tuning, which can be prohibitively expensive for large-scale models. In this work, we propose a hyperparameter-robust merging method that reframes the problem as the estimation of a unified task vector that captures the principal directions of each task (i.e., dominant singular vectors). We formalize this process as the Gram-weighted Mahalanobis Fréchet mean (GMF-Mean), a convex optimization problem that admits a closed-form solution. Our theoretical analysis shows that GMF-Mean inherently adapts to both orthogonal (non-interfering) and conflicting (collinear but opposing) task interactions by automatically modulating the magnitudes of the principal directions. This property alleviates the need for costly hyperparameter tuning that is commonly required in Task Arithmetic-based methods. Empirical results on vision, language, and vision-language models show that GMF Mean achieves competitive performance compared to state-of-the-art baselines, while maintaining the advantages of being training-free, data-free, and hyperparameter-robust. These properties position GMF-Mean as a robust solution for real-world deployment.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper tackles the problem of training-free model merging. Task Arithmetic, which merges models by summing their task matrices, suffers from the limitation of treating all task singular vectors equally (isotropically). This occurs because the underlying optimization problem in Task Arithmetic seeks the task matrix barycenter according to the Euclidean distance across all task matrices. The authors propose to address a different problem by using a Mahalanobis distance as the metric for each task matrix, where the Mahalanobis metric is defined by the per-task Gram matrix. This formulation approximately corresponds to the Fréchet minimizer, but unlike the standard Fréchet minimizer, the metric is task-specific rather than global. Hence, the authors refer to their approach as the Gram-weighted Mahalanobis Fréchet Mean (GMF Mean). The solution to this problem allows scaling of task vectors according to their degree of alignment or conflict when constructing the merged model. The method manages the anisotropy of each task matrix by preserving the principal components when task singular vectors are not conflicting, while performing a weighted summation for shared or conflicting principal directions. The approach is hyperparameter-free, as the authors empirically show that the optimal scaling factor does not need to be tuned on a validation set, and the method achieves competitive performance with state-of-the-art approaches

### Strengths
- The quality of writing and the overall flow of the paper are good. From the introduction, it is immediately clear what problem the authors aim to address—namely, the weighting of singular vectors across task matrices and the issue of hyperparameter selection in model merging. The preliminaries are well integrated with the methodology, providing a self-contained explanation that helps the reader clearly understand the authors’ methodology  presented in Section 4. I appreciated the parallelism drawn between the barycenter problem in Task Arithmetic and the problem involving weighted Mahalanobis distances addressed by the authors in Section 4

-  The method is novel and I also found the closed-form solution for the optimization problem interesting. The algebraic proofs in the appendix are correct. The geometric examples in Section 5 help the reader to understand what the authors are trying to convey in terms of non conflicting and conflicting principal directions. 

- The method achieves comparable performance to the state-of-the-art.

### Weaknesses
I have several concerns regarding the theoretical assumptions underlying the proposed methodology and the experimental claim that the approach is hyperparameter-free—a point strongly emphasized in the title, introduction, and experimental section:

**Theoretical Assumptions on the problem**: 

$$\quad T^{\star} \in \arg\min_{T}
\sum_{k=1}^{K} 
\|| T - T_k \,||_{T_k T_k^{\top}}^{2}$$

where  $|| T - T_k ||_{T_k T_k^{\top}}$ denotes the generalized Mahalanobis distance weighted by the Gram matrix 
$T_k T_k^{\top}$.

I believe that this formulation is well defined if all the gram matrices share a common reference coordinate system; otherwise, the distance it is not well defined even can be solved numerically as the author did. In general, each task may induce its own local metric, which can differ in both scale and orientation.  In Figure 1, this global alignment appears to be implicitly assumed, but in practice – especially when dealing with multiple task matrices ( 8, 14, 20 for Vision and language models)-- It is unlikely that this assumption holds.  The theoretical assumptions on Gram Matrices and Task Vectors justifying this formulation, as well as an empirical spectral analysis comparing the  Gram matrices, are currently missing from the paper.



**Hyper-parameter Free Claim**

There is neither a theoretical guarantee (considering the solution of the problem above) nor sufficiently extensive empirical evidence demonstrating that the approach is truly hyperparameter-free:

- *Theoretical guarantees*. This point is related to the concern raised above. If the Gram matrices differ significantly in norm and are not expressed in a common reference system, the minimum-norm solution provided by the pseudoinverse may be meaningful only for similar tasks. Empirically analyzing the Gram matrices and highlighting their theoretical properties could help clarify under which assumptions this solution can indeed be considered hyperparameter-free.

-  *Empirical Evidence* The empirical evaluation of the proposed approach’s robustness to hyperparameter variations is limited. Only Figure 2—based on eight tasks using ViT-L/14—shows that the scaling factor remains close to 1 and stable. More extensive experiments are needed to thoroughly assess this robustness (see the question sections for details). Moreover, TSV, which is classified in Table 1 as requiring hyperparameter validation, could also be regarded as hyperparameter-free, since the original paper shows that its optimal scaling factor is close to 1. A direct comparison with TSV—the most closely related competitor—in terms of sensitivity to hyperparameter variations is missing from the empirical evaluation.

### Questions
- Under what theoretical assumptions is the problem well-posed in terms of the Task Matrices, Gram Matrices, and their orientation and scale? Addressing this could also help mathematically characterize the conditions under which the pseudoinverse solution can truly be considered hyperparameter-free. 

- Are the task-specific Gram matrices empirically aligned? If so, why? If not, the proposed approach might inherently favor tasks whose local metrics are better aligned with the global solution. Is this behavior related to task similarity? I encourage the authors to conduct a deeper analysis of both the task matrices and the corresponding Gram matrices to identify when the merged model may fail to represent certain task vectors, and under what conditions the proposed method may not perform as intended.

 
The robustness of the proposed method with respect to hyperparameter selection—a key claim of the paper—should be empirically verified through more extensive experiments:

- Is the method still robust when changing the architecture size (e.g., ViT-B/32 or ViT-B/16) or the number of tasks (e.g., 14 or 20)?

- How does the plot in Figure 2 appear for the Qwen and BLIP models? Including these comparisons would be valuable, especially since validation tuning becomes increasingly expensive as model size grows, as the authors mention in the introduction (e.g., for large language models). Is the method still robust when considering the NLP experiments using the fully fine-tuned T5-Large models from [1], across the seven NLP tasks described in [2]?

- Finally, adding a comparison with TSV in terms of sensitivity to hyperparameter variations is necessary, as it represents the most closely related competitor.

[1] Tam et al. Merging by Matching Models in Task Parameter Subspaces, TMLR 2024

[2] Prateek Yadav et al. Resolving interference when merging models, Neurips 2023. 

Imprecision (not significant for the score): 

Line 71: “We formalize this as the Gram-weighted Mahalanobis Frechet Mean (GMF Mean) ´ problem, which seeks to minimize the sum of Mahalanobis distances between the merged vector and each task vector, with weights determined by the Gram matrix”

There is not a single Gram matrix in this formulation—the method uses one Gram matrix per task. This sentence should be revised to accurately reflect that each task has its own Gram-defined Mahalanobis metric.

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
Existing model merging methods require hyperparameter tuning. The paper proposes a method, GMF-Mean, which eliminates this need lambda=1 is fine). They frame merging as estimating a unified task vector using a closed-form convex optimization approach. GMF-Mean automatically adapts to different task relationships and achieves competitive results , also including multimodal models. The method is elegantly simple, however the relation with some recent works needs further clarification. Also further experiments on more tasks should be added.

### Strengths
- the paper poses model merging as the optimization of the Gramm-weighted Mahalanobis Frechet Mean which has a closed form. Algorithm 1 is elegantly simple.
- the discussion in section 5 captures an interesting aspect of the proposed method, showing how the method operates differently on non-conflicting and conflicting directions. 
- results in general are good and the lambda=1 is an convenient advantage (I found the usage of the validation set always a bit dubious in model merging where training data is supposed to be absent).

### Weaknesses
- It would be nice if the authors report some numbers on hyperparameter cost of other methods, further motivating their approach.

- Results of TSV in original paper are higher than reported in Table 2 and higher than proposed method, can the authors comment on that.

- I would also like the authors to compare their method with 'No Task Left Behind: Isotropic Model Merging with Common and Task-Specific Subspaces' this method does something which is very similar to the proposed method (they also apply  whitening to the task vectors). The ISO method also obtains reasonable results with alpha=1 (no hyperparameter selection, see Fig 10).

- the paper should include the results on the 14 and 20 task settings.

### Questions
I would really like to better understand the contribution of the paper in the context of the other SVD based methods, which seem be doing something pretty similar (TSV, ISO). Does this paper lead to a better understanding of the performance gains observed in those works as well ? I would also like to see some additional experiments on many-model merging.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces the Gram-weighted Mahalanobis Fréchet Mean (GMF Mean), a hyperparameter-tuning-free method for model merging. The authors propose to aggregate task vectors from multiple fine-tuned models by minimizing a weighted sum of Mahalanobis distances, where the weights derive from the Gram matrices of the task vectors. The approach yields a closed-form, convex solution that is claimed to adaptively handle both non-conflicting and conflicting principal directions. Extensive experiments on vision, language, and vision-language tasks demonstrate that GMF Mean is consistently competitive with, and in some cases outperforms, existing state-of-the-art, especially among training- and hyperparameter-free methods.

### Strengths
1. The central contribution is a well-motivated re-framing of model merging as a Gram-weighted Mahalanobis Fréchet Mean problem. The method is derived with clarity and the closed-form solution is justified.
2. The proposed GMF Mean eliminates the need for costly validation-based hyperparameter tuning.
3. The method is thoroughly evaluated on a suite of state-of-the-art models and tasks spanning vision, language, and vision-language models, which is rare in this space. GMF Mean attains competitive or superior results on multiple benchmarks, outperforming all other training-free methods in several instances.

### Weaknesses
1. Section 2 tends to group a large set of advanced baselines under "training-free" but does not contextualize where each method stands in terms of requiring data, validation, or manual tuning.
2. While Section 5 and Appendix A give a theoretical argument for the adaptivity of GMF Mean to principal direction conflict, there is a lack of targeted empirical visualizations to support this.

### Questions
1. Is additional regularization needed in practical use for GMF Mean?
2. In scenarios where tasks arise from highly heterogeneous domains, or include adversarial/task-to-task conflicts, how does the GMF Mean behave?

### Soundness
3

### Presentation
3

### Contribution
3

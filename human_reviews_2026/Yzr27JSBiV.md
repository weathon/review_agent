# Knowledge distillation through geometry-aware representational alignment

- Decision: Reject
- Scores: 2, 4, 4, 6

## Abstract
Knowledge distillation is a common paradigm for transferring capabilities from larger models to smaller ones. While traditional distillation methods leverage a probabilistic divergence over the output of the teacher and student models, feature-based distillation methods often minimize variants of Euclidean norms between the hidden layer representations. The main goal is for the student to mimic the structure of the feature space of the teacher. In this work, we theoretically show that existing feature distillation methods, such as projection based mean squared loss or Centered Kernel Alignment (CKA), cannot capture the feature structure, even under zero loss. We then motivate the use of \textit{Procrustes distance} and the Frobenius norm of \textit{Feature Gram Matrix}, distances already common in the context of measuring representational alignment, as distillation losses. We show that feature distillation through our method showcases statistically significant improvement in distillation performance across language models families (BERT and OPT) in classification and instruction-following tasks by up to 2 percentage points, showcasing the potential of integrating feature geometry into existing distillation methods.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces two new forms of knowledge distillation that, informed by the linear representation hypothesis, minimise either the procrustes distance or the Frobenius norm of the feature gram matrix to improve distillation by improving representation alignment between students and teachers. The paper provides theoretical results, synthetic experiments and larger-scale experiments to show that this form of distillation shows statistically significant but marginal performance gains across a range of evaluations, namely: COLA, RTE, MRPC, SelfInst, U-NI and S-NI. Their core contribution is that minimising loss to preserve geometric properties of the teacher model's representation is a more effect way to perform knowledge distillation.

### Strengths
1. I appreciate the effort that has gone into exploring the geometry alignment that is enabled between students and teachers, I think that it is an interesting angle and the relation to the Linear Representation Hypothesis (LRH) is a good motivation for this work. 
2. The use of statistical significance is a positive as many works do not conduct this type of analysis. 
3. Comparison to lots of over methods is a benefit of the paper, however the lack of comparison to $D_{LinProj}$ is missed which feels like an important counterfactual to the LRH argument presented in this paper. 
4. It is appreciated two language models families (BERT and OPT) are evaluated in this paper showing the benefits of procrustes distance for both classification and instruction following tasks.

### Weaknesses
While the finding that minimising either the procrustes distance or the Gram matrix can improve performance is interesting, it is unclear whether the linear representation hypothesis is the main reason why this occurs. The paper has some issues with experimental consistency and data formatting, which make it difficult to track the findings. Also, the missing evaluation of hyperparameter sweeps over the $\alpha$ term in the loss reduces certainty over the findings presented in the paper. 


1. **Lack of hyperparameter sweeping on Alpha ($\alpha$)**: In the loss function introduced in equation (6) there is a balancing between knowledge distillation and the distance metric ( Procrustes ($D_{P}$), Gram Matrix ($D_{FG}$) or CKA ($D_{CKA}$)) used, it is unclear what the relationship is between the $\alpha$ hyperparameter and performance. I would like to see an experiment that varies this $\alpha$ hyperparameter keeping it constant across controls to show how $\alpha$ explicitly impacts performance of $D_{P}$, $D_{FG}$ or $D_{CKA}$, if alignment of orthogonality of the student and teacher is important, it should be expected that increasing the $\alpha$ value would put more preference to optimising towards this goal and improve performance. Also including results for $D_{LinProj}$ that show it is ineffective in boosting performance could improve the strength of the LRH argument. 

2. **Section 5.1 Synthetic Experiment**: In sub figure A, it can be observed that the number of ε- orthogonal vectors in the student model is consistently the highest for $D_{FG}$ over $D_{P}$, however, the results in Table 3 suggest that $D_{P}$ persistently outperforms. As a result, it seems that purely aligning orthogonality between the student and the teacher does not fully explain the benefits of using $D_{P}$. 

3. **Missing result in Table 1 for PKD**: There appears to be a missing result for the method PKD on the COLA evaluation - it is unclear why this result is missing, and I cannot find an explanation for its omission in the paper. 

4. **Lack of standard deviations in Tables 1 and 2**: There appears to be no standard deviations presented in two of the main tables in the paper. While Table 3 does have standard deviations recorded, it is unclear why this is omitted for Tables 1 and 2. 

5. **Lack of comparison with $D_{P}$**: In Table 2, it is unclear why we only compare to the procrustes distance ($D_{P}$) to  ($D_{CKA}$) when the Gram matrix ($D_{FG}$) is also proposed as an effective geometry preserving metric by this paper. 

6. **Minor spelling mistake**: See line 345 'Our finds corroborate' which should be  'Our findings corroborate'  and line 346 'extremely noise' which should be  'extremely noisy'.

### Questions
1. How does Alpha ($\alpha$) impact the knowledge transfer and corresponding performance? 

2. In Table 2, can you provide results for Procrustes + FT and CKA + FT? It is important to establish if minimising cross-entropy loss with these similarity metrics alone can improve performance; otherwise, it could be the case that these metrics merely add further regularisation to KD without directly improving performance through better or worse geometry alignment. 

3. Can you align your findings with the recent distillation scaling laws [1] to show how student capacity impacts the geometry alignment? 

4. Is there a specific relationship that emerges between which layers are aligned and how this impacts performance? Can ablations show that the layer presented represents a consistent trend for transfer? 

5. In the appendix (Line 1188) there is a statement *'CKA and learned linear projection are incapable of preserving the feature geometry'* - given this and that in Table 2 we observe that CKA can correspond with statistically significant improvements in accuracy (see CKA+KD+FT) does this not show that preserving feature geometry is not a necessary part of performance increase and that performance benefits may not be tied to the LRH generally as this work posits?  

References:

[1] Busbridge, D., Shidani, A., Weers, F., Ramapuram, J., Littwin, E. and Webb, R., 2025. Distillation scaling laws. arXiv preprint arXiv:2502.08606.

### Soundness
1

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
4

### Summary
This paper is about feature-based knowledge distillation for LLMs. It shows that the current methods fail to preserve the teacher's geometric structure and proposes a theoretically grounded, geometry-aware distillation based on Procrustes distance and the Frobenius norm of the feature gram matrix. These findings are supported experimentally and compared to the current baselines.

### Strengths
The main idea is presented and discussed clearly. The paper is well-organized, and the details of the experiments and findings are well presented.

### Weaknesses
The main weakness of this work is the limited scope, which does not include multi-teacher and or multi-modal distillation and VLMs. Also, the use of Procrustes for KD already exists in the literature. So the novelty of this work is not well discussed.

### Questions
- In both Theorems 1 and 2, it is assumed that $R_t$ and $R_s$ are centered, unit-norm matrices. Can authors discuss how these assumptions are realistic (can be explored experimentally) and what the possible impacts of not respecting these assumptions are on the experimental results?
- Looking at Table 2, how does the poor performance of "Procrustes + KD " compare with "CKA + KD " align with the (theoretical) finding of the paper? 
- Theoretically, can this method be applied to the multi-teacher KD? Can this be done by some experiment?

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
4

### Summary
This work tackles the conflict issues generated in graph manipulation steps of generated scene graphs for downstream tasks.

### Strengths
1. Fundamental issue raising for the use of distance metric on knowledge distillation
2. Theoretical works to find the problem of CKA, and conditions for optimal correlation.

### Weaknesses
### 1. Unclear Problem Definition 

This work proposes using the FG matrix or Procrustes distance instead of CKA. However, the problem setup presents that CKA performs worse than these metrics, not suggesting a more detailed problem of CKA. As a result, it is difficult to understand the actual problem being solved and benefits by replacing CKA with these alternatives. This should be more clearly elaborated.

### 2. Unclear Benefit of Using the Proposed Two Metrics
As a consequence of the first issue, it is not clearly shown which structures or representations can be better aligned in knowledge distillation compared to CKA, since the specific problem with CKA in aligning structures is not clearly defined. Furthermore, the validation section lacks qualitative analysis addressing this issue.


### 3. Missing Issue in Validation
A question comes up: "Does the number of orthogonal vectors indicate better feature structure alignment?"
While I agree with the intuition behind the results of Theorem 2, it remains unclear which structures are more accurately transferred, and whether the orthogonal vector size indicates effectively the degree of structure alignment. 

a) The results only show the impact of reduced expressiveness due to a low number of orthogonal vectors. We do not know whether this is a dominating cause, especially compared to the distance metric’s potential bias in aligning feature structures. It should be shown that the orthogonal vector size meaningfully impacts alignment beyond the quantitative performance evaluation. 

b) Even if orthogonal vector size is an important indicator of alignment quality, student feature representations may have fewer dimensions to represent the diversity of features (intrinsic dimension), which could require a lower number of required vectors. Learning unnecessarily large orthogonal vectors may therefore be unnecessary.

### Questions
### 1. Why is the performance of Procrustes + KD worse than CKA + KD in Table 2?
It requires more justification why using orthogonal vector size does not appear positively. 

### 2. Large-Scale Problems Are Not Evaluated in Table 1 and 2. Why?
It is expected that the proposed measure should work for larger-scale problems if it truly enables more accurate alignment of feature structures. However, Table 2 suggests that more accurate feature representation alignment does not necessarily benefit the preservation of some student model structures in general KD settings. This raises the question: do we really need a more precise distance measure for knowledge transfer, or should we focus on selectively transferring features to avoid?

### Minor comment on Demonstration
The figures on page 7 have no captions or figure indices.

### Soundness
2

### Presentation
2

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
The paper proposes a new approach for knowledge distillation from large teacher models to smaller student models by explicitly applying Procrustes distance and the Frobenius norm of Feature Gram Matrix as the objective. The paper theoretically demonstrates that existing knowledge distillation objectives, such as projection-based mean squared loss or Centered Kernel Alignment (CKA), cannot capture the feature structure even when optimized perfectly (zero loss). And the proposed Procrustes distance-based objective is theoretically and practically better. The authors provide comprehensive theoretical proof and conduct knowledge distillation experiments on various tasks to show that the proposed objective outperforms existing methods on distilling encoder-only and decoder-only language models.

### Strengths
1. Strong theoretical motivation showing why existing feature-based losses (e.g., CKA, projection MSE) fail to preserve true geometry.

2. Novel geometry-aware losses (Procrustes and Gram-matrix) that align representational structure up to orthogonal transformations.

3. Empirical validation across encoder (BERT) and decoder (OPT) models, covering both classification and instruction-following tasks.

### Weaknesses
1. Lack of experiments to show if the observations generally hold over multiple sizes of the student models. It would be great if the authors could add analysis, even only on one task, to show the performance on student models with different intermediate sizes.
2. There are also works on using mutual information maximization as a knowledge distillation objective. It would be better to discuss this branch of works either theoretically or practically.

For other concerns, please refer to the questions below.

### Questions
1. On line 90-93, it might be a bit confusing what a “mode” is referring to, especially for people who are not familiar with this field. Maybe add one more sentence here to explain it.

2. Do you have varied analysis on the choices of student dimension for the experiments in section 5.1 to check if the observations are consistent over teacher-student dimension differences?

3. In figure 2(b), there are spikes of CKA when optimizing over Procrustes, any explanation for this?

4. It seems that for decoder-only models, the performance difference between Procrustes and CKA (or FG) is smaller, do you have any intuitions why?

5. It would be great if the authors could do some analysis on the convergence speed of Procrustes compared to CKA (or FG) for section 5.2 and 5.3

6. What's the computation overhead of the proposed objective compared to CKA?

### Soundness
3

### Presentation
3

### Contribution
3

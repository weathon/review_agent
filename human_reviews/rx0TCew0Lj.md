# Beyond the convexity assumption: Realistic tabular data generation under quantifier-free real linear constraints

- Decision: Accept (Poster)
- Scores: 5, 6, 6, 8

## Abstract
Synthetic tabular data generation has traditionally been a challenging problem due to the high complexity of the underlying distributions that characterise this type of data. Despite recent advances in deep generative models (DGMs), existing methods often fail to produce realistic datapoints that are well-aligned with available background knowledge.
In this paper, we address this limitation by introducing Disjunctive Refinement Layer (DRL), a novel layer designed
to enforce the alignment of generated data with the background knowledge specified in user-defined constraints.
DRL is the first method able to automatically make deep learning models inherently compliant with constraints as expressive as quantifier-free linear formulas, which can define non-convex and even disconnected spaces. 
Our experimental analysis shows that DRL not only guarantees constraint satisfaction but also improves efficacy in downstream tasks. Notably, when applied to DGMs that frequently violate constraints, DRL eliminates violations entirely. Further, it improves performance metrics by up to 21.4\% in F1-score and 20.9\% in Area Under the ROC Curve, thus demonstrating its practical impact on data generation.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper proposes a novel Disjunctive Refinement Layer added to neural network synthetic data generators (to replace linear constraint layer) that naturally enforces complex (including non-convex and disconnected) QFLRA constraints on tabular data without rejection sampling, and also improves the synthetic data's utility.

### Strengths
+ It uses a disjunction of linear inequalities to decompose complex non-convex and disconnected constraints into a combination of simple constraints, whose satisfiability implies the entire inequality system's satisfiability.
+ It derives and resolves implicit constraints among previous variables that ensure the satisfiability of constraints on the latter variable, and hence allows satisfiability in the entire system.

### Weaknesses
- Mathematical statements need clarification:

    1. Line 127: If $w_k=0,\forall k$ and $b < 0$, which satisfies your constraint that $w_k,b\in\mathbb{R}$, would the claim that $\Omega(\Phi)$ is non-empty still be true? (Although similar case is addressed in Line 275-277, it does not validate Line 127.)
    2. Line 170-172: What if $\Psi$s in $\Pi$ are not simplified and somewhat standardized? 
            1) If for some $\Psi$, $l^\Psi_i >= r^\Psi_i$ (violating the constraint "whenever ..." in Line 159, if it could be violated), would the definitions of $l^\Pi_i$ and $r^\Pi_i$ still be valid?
            2) If for some $\Psi,\Psi'\in\Pi$, $[l^\Psi_i, r^\Psi_i]\cap$[l^{\Psi'}_i,r^{\Psi'}_i]\ne\emptyset$ (so that less than $|\Pi|+1$ disjoint intervals are created), would the definitions of $l^\Pi_i$ and $r^\Pi_i$ still be valid?
    3. Line 259-260: Ambiguous (although can be guessed from textual explanation), it's $(A\setminus B)\cup C$ or $A\setminus(B\cup C)$?
    4. Line 285-287: Similar problem as in Line 170-172

- Some implementation details not mentioned in paper and appendix:

    1. How the DRL is applied in all the experimented models is not explicitly stated, including Appendix.
    2. How constraints are determined (or if manually determined, what are they) is not mentioned in the main content nor appendix.

- The paper claims the additional sampling time is negligible, but I would like to doubt this on the following basis:

    1. 1000 samples could be a bit too small, taking the shared overhead of initializing the task and data transformation into account. 
    2. The sampling time is expected to be approximately in linear correlation to the number of samples, so that ~0.1s for 1000 samples mean ~1s for 10000 samples, ~10s for 100000 samples, etc. What matters more is not the absolute additional time per 1000 samples, but the relative additional time (e.g. how many times more needed). According to the reported generation time in Table 5, in 3/5 cases DRL needs more than twice the time than non-DRL. This might not be negligible.
    3. DRL is also added during training, given that it could need ~2x times the time to generate samples, it naturally leads to the question that does it need also ~2x times to train.

- Although the method is novel and sound, the utility of the proposed method is doubtful:

    1. On tabular data generation literature, a recently influential trend of the use of language models and large language models is not discussed, while they are better-performing than GANs, likely to have a very low CVR (may not be 0% in theory but could be very close to it in practice). If these recent models naturally achieves near-perfect result on constraints, then the proposed method may not be useful in practice. Thus, it needs to be discussed.
    2. The authors did not compare the proposed method versus direct rejection sampling. Rejection sampling also guarantees 100% satisfaction of all constraints, and according to the reported timing in Section 4.3 (Table 5), it is not necessarily worse. e.g. for URL dataset, non-WGAN models has a CVR of about 10%. This means ~0.15/0.9=0.17s < 0.27s is needed, so it's even faster. Moreover, rejection sampling preserves naturally generated samples from generators instead of editing them, and there may be a chance that its utility can be even better. 
    3. Putting the above two points together, there could be a model with <1% violation rate, so <+1% time for generation, that obtains better result than +DRL in the older models.


- A serious issue is that the page limit is 10 but section 7 and 8 in this paper is in the page 11, which can potentially lead to desk reject.

### Questions
As mentioned in Weaknesses.

### Soundness
2

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
2

### Summary
This paper proposes a post-processing method to ensure that already generated tabular data satisfies conditions defined as a union of feasible sets specified by linear inequalities. The constraints include complex forms, such as disconnected and non-convex sets, which are rarely addressed in traditional constraint problems. The proposed method, a variation of Fourier-Motzkin, progressively reduces constraints to derive a broader feasible region. If the generated data fails to meet predefined constraints, the original data point is adjusted to the one with the minimum Euclidean distance within the relaxed constraint region.

### Strengths
When the feasible region is a disconnected, non-convex set, it is challenging to directly apply projection methods for post-adjusting generated data. This paper is significant because it demonstrates the existence of complex constraints in tabular data generation and proposes a solution to address this issue.

### Weaknesses
The method proposed in this paper for post-processing already generated data may require further theoretical support. It isn't easy to assert that finding the data point with the shortest Euclidean distance within the constraint-satisfying set is the most reasonable approach. For example, the Mahalanobis distance might be a more effective metric. 

Furthermore, the paper does not compare information leakage and reproduction accuracy, essential performance metrics for generated data. It would be beneficial to demonstrate the trade-off between these two critical metrics in generative models. For example, it is helpful to report the Membership Inference Accuracy and Attribute Disclosure Risk of the considered models.

Another limitation of this model is that it requires ranking the variables included in the constraints to be satisfied for high-dimensional data. The proposed method seeks solutions within a feasible region larger than the region required to satisfy the constraints. Additionally, the constraint composed of the union of linear constraints covered in this paper could potentially be avoided during optimization using barrier methods. Although this approach differs from the method proposed in the paper, one could consider normalizing the model by using a penalty function to ensure the generated table model remains within the predefined feasible region. For instance, adding a log barrier function to the risk function can enforce the necessary constraints on the output of the VAE decoder. Alternatively, for computational stability, ReLU could be used instead of the log barrier function. This approach may serve as an alternative that fulfills both the objective function for model training and the given constraints, potentially bypassing the issue of defining metrics for post-processing generated data.

### Questions
1. In Lemma 3.1, what does " optimal " mean?

2. What is the meaning of "The CP resolution rule is sound"?

3. In Example 4, does the result depend on the ordering of variables?

4. Does the proposed method adjust the generated data to satisfy the given constraints in the problem, or does it use more relaxed constraints?

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
2

### Summary
This paper introduces a novel Disjunctive Refinement Layer (DRL) for Deep Generative Models (DGMs), aiming to generate synthetic tabular data that 1) the sample space of the model distribution is aligned with what is stated in the background knowledge. 2) The model distribution approximates the ground truth distribution. This paper extends constraint enforcement in data generation by leveraging Quantifier-Free Linear Real Arithmetic (QFLRA) formulas, enabling more complicated (e.g. non-convex constraints). The theorem proves the proposed DRL fully satisfies the constraints and is optimal in Euclidean distance.

### Strengths
1. The use of QFLRA constraints in DRL is a substantial advancement, enabling the generation of data that complies with complex, realistic conditions. This advancement is especially relevant for applications requiring high data fidelity, such as healthcare or finance.

2. Theory analysis proves the DRL will transform the generated data to fully satisfy the constraints while attaining optimal Euclidean distance to the original sample.

3. The paper includes extensive testing across five generative models and datasets, showing consistent improvements in constraint satisfaction and downstream model performance.

### Weaknesses
The main method section (Sec 3) is a little bit dense and hard to read. I think providing a pseudo code is helpful here.

### Questions
1. What is the computation complexity of DRL?

2. What are the constraints for different datasets? Are they manually chosen by authors or are included in the original datasets?

2. Why does satisfying the constraints help improve machine learning efficiency? Is that because we also transform the test dataset so that they satisfy the constraints?

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
2

### Summary
The authors of this paper study the problem of generating tabular data where complex constraints expressed by QFLRA should be satisfied. The authors propose a neuro-symbolic AI layer to tackle this problem and prove its effectiveness in respecting the constraints. Experimental results on several datasets clearly demonstrate the advantages of the proposed model over the baselines.

### Strengths
S1. The problem of satisfying QFLRA in building tabular data generative models is new.
S2. The proposed technique can be integrated into any deep model.
S3. The proposed method is sound as the authors prove its correctness.

### Weaknesses
W1. The authors mainly focus on proving that their proposed Neuro-symbolic AI layer can respect the complex constraints expressed by QFLRA. What about the impact of this new layer on the learned distribution? Is it possible that such a new layer may harm the learned distribution for satisfying the constraints? 

W2. It seems to me that the constraints of QFLRA may be very complex and even deciding if a QFLRA is satisfiable is a difficult problem. What would happen to the proposed model if the input QFLRA is not satisfiable? 

W3. In the experiments, the authors adopt some simple QFLRA constraints, where the feasible region has a couple of disconnected parts. For these relatively simple QFLRA, can we first identify each disconnected subregion, then use traditional models to learn the distribution over each subregion and exploit a mixture model to combine distributions over all subregions to have an overall generative model?

### Questions
Please refer to W1-W3.

### Soundness
3

### Presentation
3

### Contribution
3

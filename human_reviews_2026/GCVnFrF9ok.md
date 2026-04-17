# Minimal Repairs for Learning Over Incomplete Data

- Decision: Reject
- Scores: 2, 4, 6, 2

## Abstract
Missing data often exists in real-world datasets, requiring significant time and effort for data repair to learn accurate machine learning (ML) models. In this paper, we show that imputing all missing values is not always necessary to achieve an accurate ML model. We introduce concepts of minimal and almost minimal repair, which are subsets of missing data items in training data whose imputation delivers accurate and reasonably accurate models, respectively. Repairing these sets can significantly reduce the time, computational resources, and manual effort required for learning models. We show that finding these sets is NP-hard for SVM and linear regression and propose efficient approximation algorithms with provable error bounds. Our extensive experiments indicate that our proposed algorithms can substantially reduce the time and effort required to learn on incomplete datasets.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper defines Minimal Repair (MR) and Almost Minimal Repair (AMR) for learning with incomplete data. The authors give NP‑hardness proofs for MR/AMR with SVMs and linear regression, propose approximation algorithms (for SVM via “edge repairs” and support‑vector tests; for linear regression via an OMP‑style procedure), and analyze limited correctness properties (e.g., “no false positives” for the SVM routine; probabilistic inclusion conditions for the LR routine under incoherence and Gaussian assumptions). The paper acknowledges that guarantees and tractability are limited to convex settings and that MR/AMR can take as long as or longer than simple full imputation.

### Strengths
S1. Clear formalization of MR/AMR and useful reductions/NP‑hardness results for SVM and linear regression. 

S2. Concrete approximation algorithms with some correctness properties (e.g., SVM “no false positives”; LR OMP‑style procedure).

S3. Broad experimental suite across multiple datasets with both injected and naturally occurring missingness.

### Weaknesses
W1. “Minimal repair” is decoupled from ground truth; guarantees are model‑relative, not reality‑relative.
MR is defined to ensure existence of a certain model after imputing a subset; the paper explicitly states that this is orthogonal to the accuracy of the imputations and that a certain model may be accurate or inaccurate depending on the imputed values. Thus MR says nothing about recovering true data values or improving true predictive accuracy; it only targets invariance of the ERM across completions. This disconnect undermines the practical value of the theory.

W2. Error bounds are measured w.r.t. MR/ACM, not the true data or generalization performance.
All guarantees bound differences to the minimal (or almost minimal) set or to an approximately certain model—not to the ground‑truth dataset or population risk. For LR, the probabilistic inclusion guarantee relies on mutual incoherence and independent zero‑mean Gaussian assumptions over missing entries—conditions that are rarely realistic—and still only certifies proximity to the MR set, not to the true data or Bayes model. For AMR, the optimizer minimizes a worst‑case suboptimality proxy via sampled edge repairs and gradient/subgradient norms; again, this measures closeness to ACM, not to truth or test performance. Consequently, there is no reason to believe MR is better than AMR in practice, since both are benchmarked against MR/ACM rather than ground truth. 

W3. Missing—and arguably more relevant—baselines.
Despite framing the work as practical, the experiments do not compare to LLM‑based repair strategies or other modern learned imputers that can leverage semantics or context (even if guarantee‑free). The baselines listed in the tables are KNN, MICE, TCSDI (diffusion), MF, and ActiveClean; no LLM‑based repair is evaluated, and even some cited learned imputers (e.g., GAIN) are not included empirically. If MR/AMR are meant to reduce human/compute costs while maintaining model utility, comparisons against strong heuristics that practitioners actually use are essential. 

W4. Practical relevance is questionable given the paper’s own limitations.
The paper concedes (i) convexity is required for current guarantees/approximations, leaving non‑convex and deep models out of scope, and (ii) MR/AMR can match or exceed the time of full imputation when simple methods are used. In many realistic pipelines, simple imputers or domain‑specific heuristics are exactly what practitioners deploy; if MR/AMR provides no accuracy gains relative to ground truth and may cost as much or more time, its advantage is unclear. 

W5. Reliance on restrictive constructs and strong assumptions.
The SVM routine hinges on edge repairs (min/max bounds for every missing value) and support‑vector tests; this presupposes credible per‑feature bounds, which are often unavailable or uninformative for continuous attributes. For LR, the OMP‑style analysis assumes incoherence and Gaussian missing‑value distributions, which may not hold (especially under MNAR). These choices further limit applicability.

W6. Scope limited to SVM/linear regression; no path to deep models.
The framework’s theoretical underpinnings and efficiency claims currently do not extend to non‑convex learners; the paper notes this as an open challenge. For a venue like ICLR, where deep models are central, the lack of a credible path or partial results for non‑convex settings weakens the contribution.

### Questions
See the above weaknesses W1-W5.

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper utilizes an efficient approximation algorithm to solve the problem of reducing the time and effort needed to train accurate ML models on incomplete data by only imputing necessary missing values. Specifically, it first proposes the concepts of "minimal" and "almost minimal repair" (the minimal subsets of missing data needed for accurate models). Then show finding these sets is NP-hard for SVM/linear regression and propose efficient approximation algorithms to find them. The experimental results prove the superiority of the proposed method by showing a substantial reduction in time and effort required for learning.

### Strengths
1)The paper formally defines minimal repair for SVM and linear regression. They prove that finding these repairs is NP-hard, establishing the theoretical difficulty of the problem, and consequently propose efficient approximation algorithms with provable error bounds to make the solution practical.

2)This paper introduces the concept of almost minimal repair, a solution that is easier to find and guarantees a model loss close to that of a fully repaired dataset. They provide the necessary NP-hard proofs and corresponding approximation algorithms for this concept as well.

3)Empirical results demonstrate that the proposed algorithms efficiently approximate both minimal and almost minimal repairs. The proposed method allows users to substantially reduce the time and effort required for model-based imputation on large datasets without losing accuracy in the final downstream learning task.

### Weaknesses
1)It appears the experimental section of this paper only validates the efficacy of the proposed model using MCAR. Why were corresponding experiments not conducted for the previously mentioned MNAR mechanism, as well as the unmentioned MAR mechanism, to provide more sufficient evidence demonstrating the superiority of the proposed method? Furthermore, should the authors consider introducing mixed missing patterns (e.g., some attributes missing via MCAR, others via MNAR) within a single dataset to further validate the method's effectiveness?

2)The abstract mentions a reduction in computational resource consumption, but the experimental section only provides suggestions based on the percentage of imputed samples. It is recommended that the authors also provide corresponding data on the consumption of memory (RAM) and GPU memory (if used). This would more clearly and explicitly demonstrate the reduction in computational resource consumption achieved by the proposed method.


3)I personally believe there is room for further improvement in the comparison algorithms used in the paper. For instance, the algorithms could be categorized into three main groups: statistics-based methods, machine learning-based methods, and deep learning-based methods. For the first two categories, well-established, typical methods can be selected for comparison. For the third category, deep learning-based methods, it is suggested that the authors choose more recent comparison methods. These could include methods based on Variational Autoencoders (VAEs), Generative Adversarial Networks (GANs), and Flow Matching (in addition to the Diffusion-based method already mentioned in the paper, if applicable). I recommend that the author strive to enrich the variety of comparison methods and select the most up-to-date options available.

### Questions
See above.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses the high cost of imputing missing data in machine learning. The authors introduce the concepts of Minimal Repair and Almost Minimal Repair, defined as the smallest subsets of missing values that must be imputed to obtain an optimally accurate model or one within a specified error tolerance. They prove that identifying such repairs for support vector machines and linear regression is NP-hard, and propose efficient approximation algorithms with provable guarantees. Extensive experiments show that the proposed methods reduce imputation effort and computational cost while delivering models comparable in accuracy to those trained on fully repaired data.

### Strengths
1. The paper introduces the concept of minimal and almost-minimal repair for learning over incomplete data, a problem formulation that offers a new perspective on data preparation by questioning the necessity of full imputation.
2. The work establishes the theoretical complexity of the problem for SVM and linear regression and complements this analysis with the development of practical approximation algorithms accompanied by theoretical error bounds.
3. An extensive empirical evaluation on multiple real-world datasets demonstrates the potential of the proposed methods to reduce the scale of required imputations, suggesting practical utility for resource-constrained learning scenarios.

### Weaknesses
1. Section 3.1 notes that the proposed methods rely on the assumption of known domain bounds for each missing value. In practice, such bounds may not be easily determined, and the method for defining them warrants further discussion. The paper could be enhanced by providing specific guidance on how to establish these bounds and by further analyzing how overly wide or narrow bound specifications might impact the algorithm's performance and efficiency.
2. The paper does not seem to address the handling of common categorical variables, for which the concept of "bounds" is not clearly defined.
3. The experimental results indicate that AMR often ​requires more time and yields lower accuracy than​ MR. Therefore, it would be beneficial for the paper to further identify the particular use cases where the trade-off of AMR is most advantageous.
4. The AMR method lacks practical guidance for setting its error threshold e. A sensitivity analysis showing how e impacts both repair size and final model accuracy is needed to demonstrate its practical utility.
5. In Section 6, the paper points out that the methods still perform well empirically even when the theoretical assumptions (such as M-Lipschitz continuity for SVM) are not met. While the results are positive, discussing the potential reasons behind this robustness would help bridge the gap between theory and practice.

### Questions
1. The results show that the performance of MR and AMR is largely insensitive to the missingness ratio. Could you please explain the reason behind this?
2. Under what specific conditions would AMR offer a clear advantage in time efficiency over MR, given that it was often slower in the experiments?
3. What practical guidance can be offered for selecting a specific imputation method to use within the MR/AMR framework in a real application?
4. The paper notes the methods work well even when theoretical assumptions are violated. What are the potential reasons for this robustness that could bridge the gap between theory and practice?
5. The provided link to the code repository returns a "file not found" error. Could you please check it?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes the concept of minimal repair (MR) and almost minimal repair (AMR) for learning models over datasets with missing values. The core idea is to identify and impute a minimal subset of missing values that are sufficient to achieve an accurate model, thus reducing the time and computational resources typically required for full imputation. The authors provide theoretical foundations, proving that finding minimal repairs for SVM and linear regression is NP-hard, and propose efficient approximation algorithms with provable error bounds. Through experiments on real-world datasets, the authors demonstrate that their methods (MR and AMR) can reduce imputation time and manual effort without significantly compromising model accuracy.

### Strengths
S1. Missing data is a critical issue in real-world datasets, and addressing this problem with minimal imputation is valuable.

S2. The formal definitions of minimal and almost minimal repairs are clear and well-explained.

S3. The paper introduces approximation algorithms with provable error bounds, which is an important step in dealing with the NP-hard problem of finding minimal repairs.

### Weaknesses
W1. The core idea of minimal repair has been explored in prior work, especially in ActiveClean and Certain Model Learning, which already aim to reduce the cost of imputation without sacrificing model accuracy. The paper does not provide significant improvements or innovations to justify its claims as groundbreaking.

W2. The experiments are based on a narrow set of models (SVM and linear regression) and imputation methods (KNN, MICE). While these are standard, the results don’t convincingly show why MR/AMR is preferable in practice. The paper would have been stronger with a broader evaluation that includes more recent models or deep learning applications, where missing data issues are also common.

W3. Despite the claims of efficiency, the algorithms for finding MR and AMR can still be computationally expensive, especially in high-dimensional datasets. The claim that MR/AMR will always be faster than full imputation is questionable, particularly when simpler imputation methods like KNN are used. The paper does not sufficiently demonstrate the practical computational benefits for large-scale datasets.

W4. While the NP-hardness results are a nice theoretical contribution, they detract from the practical usability of the method. The algorithms for minimal repair are highly complex, and the paper does not present sufficient evidence to suggest they would be widely applicable in real-world data cleaning scenarios.

W5. The formalism and technical details make the paper hard to follow, especially for readers without a deep background in optimization and data imputation. The dense presentation of the experimental results in the tables, combined with jargon-heavy theoretical discussions, makes the paper less accessible and less impactful.

### Questions
Q1. Can the authors show any significant performance improvements when applying MR/AMR to more complex models (e.g., neural networks) or in real-world domains such as healthcare or finance?

Q2. How does the approach scale with high-dimensional datasets (e.g., millions of features or samples)? Can the authors provide additional scalability experiments?

Q3. The paper mentions that the algorithms may be computationally expensive. Could the authors offer further justification for why MR/AMR would be preferred over simpler methods in scenarios where full imputation methods are already cheap to compute?

Q4. What would happen to the performance if the imputation model is poor? Are there any safeguards or adjustments the authors propose to ensure model robustness?

### Soundness
2

### Presentation
3

### Contribution
2

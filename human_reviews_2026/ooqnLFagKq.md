# Coarse-to-Fine Learning of Dynamic Causal Structures

- Decision: Accept (Poster)
- Scores: 6, 4, 2, 6

## Abstract
Learning the dynamic causal structure is a difficult challenge in discovering causality from time series. Most existing studies rely on distributional or structural invariance to uncover the underlying causal dynamics, assuming stationary or partially stationary causality, which frequently conflicts with complex causal relationships in the real world. This boosts temporal causal discovery to encompass fully dynamic causality, where both instantaneous and lagged causal dependencies may change over time, bringing significant challenges to the efficiency and stability of causal discovery. To tackle these challenges, we introduce DyCausal, a dynamic causal structure learning framework that leverages convolutional networks to effectively model causal structures within coarse-grained time windows, and introduces linear interpolation to refine causal structures to each time step and recover time-varying causal graphs. In addition, we propose an acyclic constraint based on matrix norm scaling. It is more stable both theoretically and empirically, and constrains loops in dynamic causal structures with improved efficiency. Evaluations on both synthetic and real-world datasets prove that DyCausal significantly outperforms existing methods and identifies fully dynamic causal structures from coarse to fine.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a novel method for time-series causal discovery, specifically designed to handle dynamically changing causal relationships. This is a timely and important problem that has received relatively little attention in the literature. The proposed approach is well-motivated, and the paper is clearly written and supported by extensive empirical evaluations.

While the work is promising, several points regarding the experimental validation, comparison to related work, and clarity of the evaluation protocol require attention. Addressing these points would strengthen the paper and allow me to raise my rating further.

### Strengths
- **Clarity and Presentation:** The paper is well-written and mostly easy to follow. 
- **Novelty and Timeliness:** The paper addresses an important and under-explored problem in causal discovery: learning dynamic causal graphs from time-series data. This represents a valuable and timely contribution.
- **Empirical Evaluation:** The authors provide an extensive set of experiments on both synthetic and real-world datasets, which helps to demonstrate the method's effectiveness across various settings.

### Weaknesses
### **Major Weaknesses**

1. **Clarity on Real-World Dataset Dynamics and Evaluation:** The justification for using certain real-world datasets (e.g., `NetSim` or `CausalTime`) to evaluate a *dynamic* causal discovery method is unclear. To the best of my understanding, these datasets have a static ground truth graph (as they rely on some synthetic components).
    - The authors should clarify why a dynamically changing graph is expected or found in these cases. If there is no prior information on dynamics, this should be explicitly stated, as it impacts the interpretation of the results.
    - To further validate the method's ability to handle known dynamics, the authors are strongly encouraged to evaluate on benchmarks like **CausalRivers** [6]. This dataset's assumption of a fixed graph structure with dynamically changing edge weights perfectly matches the problem setting of this paper and would provide more convincing real-world evidence.

2. **Usage of $h_{norm}$:** While Eq. 7  suggests that the norm is used as a constraint, Algorithm 1 shows that it is simply added as a loss term. This suggests that it is not enforced that $h_{norm}$ is exactly 0, but rather it is only pushed close to 0 to minimize the loss. As the DAG structure is only enforced at $h_{norm} = 0$, how is it guaranteed that the solution is really a DAG? Please clarify this point.

3. **Ambiguity in Evaluation Protocol:** How are the performance metrics in Table 1 calculated for dynamic graphs? For instance, are the predicted graphs over time collapsed into a single summary graph for comparison with a static ground truth? How are baseline methods that produce only a single, static graph treated in this evaluation? This information is essential for interpreting the results correctly.

4. **Lack of Architectural Comparison and Positioning:** The paper would be strengthened by a more detailed comparison with related convolutional approaches such as NTS-NOTEARS, TCDF, and other recent works (e.g., [4, 5]). A discussion of the architectural differences and how they relate to the modeling of temporal dependencies would help readers better understand the unique contributions and advantages of the proposed method.

5. **Need for Granular Performance Analysis:** The paper would benefit from a more fine-grained analysis of the model's performance. Specifically, providing separate recovery scores for **instantaneous** and **lagged** causal links would offer deeper insights into the method's strengths and weaknesses.
7. **Unexplained Superiority on Static Graphs (Appendix A8):** A key result that requires further explanation is the proposed method's superior performance even on static graphs. This is counterintuitive and could suggest suboptimal hyperparameter tuning for the baseline methods. A more thorough investigation or discussion is needed to justify this finding and rule out experimental artifacts.



### **Minor Weaknesses**

- **Assumption on Graph Structure:** The method appears to assume a fixed graph structure with dynamically changing edge weights. While not a significant weakness, this should be made clear earlier in the paper, as the method does not address the problem of a changing graph *structure* (i.e., addition/removal of edges).
- **Smoothness Assumption:** The paper could benefit from a brief discussion on the limitations of the smoothness assumption for the function `F`. What happens in natural systems with phase transitions or abrupt changes (e.g., freezing at 0°C), where this assumption is violated? A small synthetic experiment on this (e.g., simply drawing coefficients from a uniform distribution) would be very interesting.
- **Literature Review:** The background section mentions two groups of CD methods, which may be an oversimplification. The review could be extended to better position the work by including other major families of methods, such as Granger Causality and constraint-based/noise-based approaches [1].
- **Related Work:** A recent paper at ICLR, on Meta Causal Graphs [2], appears closely related and should be discussed.
- **Terminology:**
    - **L49 "Faithful":** The use of the word "faithful" could be confused with the Causal Faithfulness assumption, a standard term in the causal inference literature. Rephrasing is recommended.
    - **"Significant":** The word "significant" is often used without statistical validation. Please ensure its use is appropriate or rephrase to be more descriptive (e.g., "substantial," "large").
    - **SEM vs. SCM:** The paper uses the term Structural Equation Model (SEM). To avoid confusion with the large body of work in social sciences also using this term, consider using Structural Causal Model (SCM) instead, as suggested by [3].
- **Clarity and Typos:**
    - **Table 1:** Typo `CUT+` should be `Cuts+`.
    - **Figures:** Many figures use labels like `W1`, `W2`, `W3` without a clear definition in the caption or in the legend. Adding a brief explanation in the figure captions would improve comprehension.
    - **Precision Metric:** The precision values in the tables appear to be between 0 and 100 but are reported without a decimal point (e.g., "95" instead of "0.95"). This should be clarified in the table captions.



¹  https://www.jair.org/index.php/jair/article/view/13428/26917

² https://openreview.net/forum?id=J9VogDTa1W

³ https://library.oapen.org/bitstream/id/056a11be-ce3a-44b9-8987-a6c68fce8d9b/11283.pdf

⁴ https://arxiv.org/abs/2408.08023

⁵ https://arxiv.org/html/2404.01466v1

⁶ https://openreview.net/forum?id=wmV4cIbgl6

### Questions
Answering these questions in the paper or rebuttal would greatly improve clarity:

1. **Robustness to Non-Cyclic Dynamics:** What is the expected behavior of the model if the underlying causal dynamics are not smooth or cyclic, but instead feature abrupt changes or non-periodic patterns?
2. **Acyclicity Constraint:** Could the acyclicity constraint be relaxed or removed? If so, what would be the expected impact on performance?
3. **Performance on Static Graphs:**  Can you give a rational why your method is typically even better than other methods in the static case? (A8). This suggests some Hyperparameter problems.
4. **Interpolation vs. Per-Step Estimation:** Do you have to perform linear interpolation, or is this something you simply do for computational efficiency? If this is the case, how do results change if the Causal Graph is estimated at every time step?
5. **Dynamics of Instantaneous Links:** There appears to be a contradiction between Eq. 2 (which suggests instantaneous links are static) and Section 5.1 (where they seem to be varied in experiments). Am I misunderstanding something?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes a coarse-to-fine causal discovery method to address fully dynamic causality. They leverage CNN to capture patterns within coarse-grained time windows, and apply linear interpolation to refine causal structures. The authors tested DyCausal's performance and compared it with previous methods on both synthetic and real-world datasets.

### Strengths
1. The authors addressed the limitations of previous methods by considering a fully dynamic causality.

2. They offered clear motivation for their DyCausal algorithm and provided a new form of acyclic constraint.

3. The authors tested DyCausal's performance and compared it with previous methods.

### Weaknesses
1. The proposed refined dynamic causal graph interpolation is based on the assumption that the causal relationship changes smoothly and follows a linear law, which is a very strong prior assumption.

2. The acyclicity constraint in the question appears to be a variant of the DAGMA constraint, and I am skeptical as to whether it is sufficient to be considered an innovative point.

3. Learning causal relationships requires theoretical guarantees of identifiability. The proposed CNN method employs an encoder and decoder to infer causal structures, treating it as an unsupervised sequence reconstruction task, which appears to lack theoretical support.

### Questions
1. There seems to be a lack of sensitivity analysis of penalty coefficient $\beta$, decay coefficient $\mu$, and decay rate $\gamma$.

2. The model uses a sliding window approach for causal discovery. The question is whether the causal graphs obtained at the intersection of two windows are consistent. For example, is there a large difference in the changes between $W_{t+S}$ and $W_{t+S+1}$? Are they smooth?

3. The experimental results on CausalTime do not appear to include some of the latest state-of-the-art methods, such as JRNGC.

4. Are there any experiments on a real-world dataset with large-scale nodes?

5. As far as I know, DYNOTEARS and NTS-NOTEARS are both static causal discovery methods. How do they obtain the results on the dynamic dataset?

[1] Zhou, W., Bai, S., Yu, S., Zhao, Q., & Chen, B. (2024). Jacobian regularizer-based neural granger causality. arXiv preprint arXiv:2405.08779.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes DyCausal, a framework for learning dynamic causal structures in time series that exhibit complex and time-varying dependencies commonly observed in real-world systems. The method addresses fully dynamic causality, where both instantaneous and lagged dependencies evolve over time. To tackle this challenge, DyCausal leverages convolutional networks to capture causal patterns within coarse-grained time windows, and applies linear interpolation to refine the causal structures at each time step, thereby recovering fine-grained and continuously evolving causal graphs. In addition, an acyclicity constraint based on matrix norm scaling is introduced to enhance computational efficiency while effectively preventing loops in the evolving causal structures. Experimental results demonstrate the effectiveness of the proposed approach.

### Strengths
1. The paper extends causal structure learning to a general setting of fully dynamic causality, where both instantaneous and lagged dependencies evolve over time. This broadens the applicability of causal discovery methods to more realistic time-varying systems.

2. The proposed approach appears technically sound.

### Weaknesses
1. The paper lacks significant technical innovation. Most components of the proposed framework appear to be incremental adaptations of existing techniques rather than conceptually new contributions. The use of convolutional networks for coarse-grained temporal modeling and the acyclicity constraint are engineering extensions that do not substantially advance the methodological frontier. Moreover, the theoretical results are mostly straightforward derivations of [1], without introducing new analytical insights.

2. The presentation and notation require improvement. For example, in Theorem 2 the symbol $=\in$ is incorrect; in Algorithm 1, the L₁-norm notation is inconsistent; and the usage of boldface for matrices and vectors is not uniform (e.g., the identity matrix I on line 214). In addition, the definition of Δ in Theorem 1 (line 239) is unclear, and line 7 of Algorithm 1 is difficult to interpret. These issues hinder clarity.

3. The experimental section contains several marking or annotation errors. For instance, on line 1098, the reported TPR: NTS-NO better than DyCausal. Similarly, see line 832.

[1] Bello K, Aragam B, Ravikumar P. Dagma: Learning dags via m-matrices and a log-determinant acyclicity characterization[J]. Advances in Neural Information Processing Systems, 2022, 35: 8226-8239.

### Questions
1. What are the limitations introduced by the coarse-grained approximation? 

2. Can the authors establish theoretical guarantees for convergence of proposed framework, similar to the Lemma 6 in [1]?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes structure learning for dynamic (i.e., time-varying) causal graphs in time series. This is proposed via a "coarse-to-fine" strategy; this proceeds by roughly identifying causal structures over some large sliding window, then linearly interpolating. An additional modification to the acyclicity penalty of DAGMA is introduced. Extensive experiments are undertaken, which show superior performance of the proposed method in real and synthetic datasets with time-varying causal mechanisms.

### Strengths
- The paper is well-written and easy to follow.
- The paper motivates and undertakes an important and under-discussed problem of time-varying causal structure learning.
- Relative to some prior work (e.g., DyCast), which only learns time-varying instantaneous graphs, DyCausal simultaneously learns time-varying lagged relationships. This seems to significantly improve performance. 
- The experiments clearly demonstrate strong performance relative to relatively recent and strong baselines.

### Weaknesses
I'm not totally convinced by the analysis of the revised log-determinant penalty. One reason why is that the benefits of DAGMA lie partially in nice optimization properties in a barrier method approach -- it's a priori possible but not obvious to me that the proposed version would perform better in the general structure learning setting. The ablation in C.10 is nice in theory, but the graph in Figure A9 seems to moreso suggest some catastrophic failure than poor optimization.

### Questions
1. How does h_log compare with more extensive hyperparameter tuning? Do the results of C.10 change? Furthermore, does the proposed version h_norm significantly change results in the general (static) structure learning setting?

2. How is initialization performed? Dagma initializes W to be the zero matrix, but this obviously results in an infinite denominator.

### Soundness
3

### Presentation
3

### Contribution
3

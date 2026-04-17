# HKAN: Hierarchical Kolmogorov-Arnold Networks for Efficient and Interpretable Feature Interaction Modeling

- Decision: Reject
- Scores: 6, 2, 4, 4

## Abstract
Learning complex feature interactions is central to modern machine learning, driving breakthrough performance across domains from structured data analytics to predictive modeling in recommender systems and beyond. 
However, despite notable progress, this field still faces three substantial challenges:
i) lack of adaptive topology discovery — models cannot automatically learn which features should interact and at what order; ii) the 'black-box' nature of deep neural
networks with poor explainability of the learned interaction patterns; iii) computational inefficiency due to parameter-heavy architectures with limited scalability.
To address these challenges, we propose a hierarchical sparse framework, namely Hierarchical Kolmogorov-Arnold Network (HKAN), for efficient and interpretable feature interaction modeling with three key aspects: 
i) factor-quality-guided evolutionary architecture
search (FG-EAS) to automatically discover data-centric optimal feature grouping
strategies;
ii) hierarchical sparse structure with superior parameter efficiency
iii) B-spline-based univariate function visualization and hierarchical factor structures with end-to-end interpretability
from local to global levels. 
To test the predictive and symbolic regression ability of HKAN, we conduct experiments across 10 tabular learning and 2 function fitting tasks. HKAN achieves state-of-the-art (SOTA) or highly competitive performance on the vast majority of datasets while utilizing significantly fewer parameters. Notably, on three of these datasets, it reaches state-of-the-art performance with less than 10\% of the parameters used by the baseline models. Moreover, HKAN can serve as a knowledge discovery tool with excellent explainability (e.g., explicit formulas of data patterns) compared to other black-box baselines, which represents a significant step toward building more trustworthy and accountable AI systems.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes Hierarchical Kolmogorov–Arnold Networks (HKAN), which integrate three key techniques: (1) a hierarchical sparse architecture, (2) dual-layer regularization, and (3) factor-quality-guided evolutionary search. Together, these components enhance parameter efficiency, improve interpretability, and enable automatic architectural adaptation across diverse tasks. The authors evaluate HKAN on a variety of tabular datasets, demonstrating comparable or superior performance to baseline models while using substantially fewer parameters and improved interpretability.

### Strengths
- The writing is clear, well-structured and easy to follow.
- Hierarchical sparse design with overlapping groups is intuitive and practically useful. These designs improve both parameter efficiency and interpretability.

### Weaknesses
- The paper does not fully address the computational challenge. Although HKAN substantially reduces the number of parameters, the evolutionary architecture search introduces considerable computational overhead, potentially increasing the overall training time. It would be helpful if the authors could provide a comparison of total training time—including both the architecture search and model training—against other baselines.
- The authors define three fundamental properties (independence, stability, and sparsity) for the dual-layer regularization and also employ them in the architecture search. An ablation study on these three components would be valuable to demonstrate their individual and combined effectiveness.
- The paper does not explain how the five $\lambda$ coefficients in the dual-layer regularization loss function are chosen. Since model performance may be sensitive to these hyperparameters, a justification or sensitivity analysis is recommended.
- For all experimental results, the standard deviations across multiple random seeds should be reported to assess the robustness and statistical significance of the findings.
- The positions of Sections 4.3 and 4.4 should be swapped. Placing the interpretability analysis (currently Section 4.4) before the performance evaluation (currently Section 4.3) would better highlight HKAN’s key contribution to improved interpretability.

### Questions
- The authors describe HKAN as a _unified framework_ in the abstract. However, this term is somewhat ambiguous. Based on my understanding, a “unified framework” typically implies that the method can encompass or generalize a range of existing neural architectures under a single formulation. But it seems that HKAN didn't achieve this. I recommend that the authors clarify in what sense HKAN is considered “unified”. 
- It appears that the improved interpretability mainly stems from the grouped feature structure combined with appropriate regularization, which helps preserve meaningful and disentangled feature relationships. If this understanding is correct, I suggest that the authors include an additional section providing a detailed analysis of _why_ interpretability is enhanced in HKAN. Moreover, this aspect should be emphasized more prominently in Section 4.4 (Function Fitting Analysis).
- I am interested in the interpretability performance of the original KAN on the UCI Heart Disease dataset. Could the authors provide the corresponding experimental results for comparison?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces HKAN, a novel architecture built upon Kolmogorov–Arnold Networks (KAN), along with a set of newly designed training losses—decoupled, sparse, and stability losses—to enhance representation learning. Additionally, a Factor-Quality Score is proposed to guide the architecture search process. The authors claim that HKAN effectively addresses the trilemma of achieving high predictive performance, automated architecture discovery, and end-to-end interpretability.

### Strengths
- Introduces HKAN, a novel KAN-based architecture, and designs customized training losses that markedly improve representation quality.
- Achieves state-of-the-art results across five benchmark datasets, demonstrating consistent superiority.

### Weaknesses
- **Literature review is insufficient.**The three “fundamental” challenges (manual pre-definition, black-box nature, computational inefficiency) are no longer open problems; recent works such as Fast Generic Interaction Detection for Model Interpretability and Compression already deliver efficient, interpretable solutions.  HKAN’s incremental contribution over these advances is not discussed.
- **Motivation and research gap are missing.**
The claimed trilemma (SOTA accuracy + automated architecture + end-to-end interpretability) is asserted rather than derived.  The paper never clarifies why existing KAN or post-hoc interpretation methods fail on tabular data, nor what specific architectural or algorithmic gap HKAN fills.
- **Notation is introduced carelessly.**
Core symbols such as Factor_k and KAN_k appear in equations and algorithms without formal definition, forcing the reader to guess their dimension, domain, and learnable status.
- **The test suite is simple.** No synthetic benchmarks are constructed to test interpretability claims.  Follow the protocol of Tsang et al. (2018): ten controlled synthetic functions with known ground-truth interactions should be included to verify whether HKAN actually recovers the presumed structure.

### Questions
1. What inductive biases or architectural ingredients in HKAN are responsible for its consistent top rank on small-sample tabular sets, and how do these differ from those of the compared baselines?
2. How do you get the learned expressions in eq(9)?
3. Is Table 4 the feature selection result for Heart-dataset? yet the numeric values diverge from the Heart row in Table 1. Clarify which experimental setup (metric, train/test split, seed) was used for each table and reconcile the mismatch.
4. Equation (2) can be recovered from Equation (1) by zeroing selected ψp,q. How does explicitly allowing overlapping groups Gi ∩ Gj ≠ ∅ advance the Kolmogorov–Arnold representation, given that the original theorem already permits the universal case Gi = Gj = [n]?
5. The stable-loss term penalizes variance and thus pushes univariate functions toward constants. Explain why this regularization does not collapse representational capacity and provide theoretical or ablative evidence that it improves generalization.

### Soundness
2

### Presentation
1

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
The paper introduces a KAN-based architecture called HKAN with feature selection at the intermediate layers.  A novel evolutionary search algorithm is designed for automatically learn the sparse feature subsets.  The resulting model has better interpretability due to the subset selection allowing for easier symbolic regression while achieving competitive performance on multiple tabular datasets.

### Strengths
- novel modification of KAN to achieve better pruned sparsity of the intermediate representations
- new evolutionary approach to be able to learn sparsity patterns beyond simple regularization approaches
- better interpretability of the final learned model
- good performance on real-world datasets

### Weaknesses
- The proposed trilemma does not seem to make much sense.  xDeepFM and  FT-Transformer do not have extensive manual predefinition of interactions.  I also think they are blackbox methods so it is hard for me to see what the authors point as the critical aspects of the trilemma.
- Insights into the hyperparameter choices seem to be completely missing (mainly the choice of weighting for the new FQS metric and the different operators of the evolutionary algorithm)
- The paper mostly discusses work on interactions from the factorization machine perspective, but additive models with sparse interactions [1] seem to be more relevant here.  In fact, equation (2) seems to represent exactly an additive model with sparse interactions where the $G_k$ are the interactions.
- The symbolic regression experiments are also unclear.  Symbolic regression should require another algorithm to extract the learned symbolic equations but I do not see a discussion of it.  Additionally, a symbolic regression algorithm can apply to any blackbox model, so this does not seem to directly show the interpretability of HKAN.


[1] Chun-Hao Chang, Rich Caruana, Anna Goldenberg. NODE-GAM: Neural Generalized Additive Model for Interpretable Deep Learning.

### Questions
Overall I am optimistic about the work and happy to raise my score if the weaknesses and questions can be addressed.

- How were the hyperparameters used for the HKAN algorithm chosen?
- What are some of the interpretable insights gained from the learn HKAN models?  Isn't it possible to plot all of the spline functions and report the learned interaction subsets?
- What are the times taken for training the HKAN architecture, especially what are the times taken for the different phases of the learning algorithm?
- What is the meaning of the trilemma presented in the introduction and how does HKAN overcome the challenges faced by existing methods?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes HKAN (Hierarchical Kolmogorov–Arnold Network), a framework for modeling complex feature interactions that is simultaneously accurate, parameter-efficient, and interpretable. It does this by (i) automatically discovering data-specific feature groups via a factor-quality–guided evolutionary search, (ii) processing each group with lightweight KANs to produce semantic “factors,” and (iii) combining them in a sparse hierarchical KAN so users can visualize the learned B-spline functions and factor structure.

### Strengths
1.	The paper tackles a clear and relevant gap—making Kolmogorov–Arnold networks usable for real, high-dimensional tabular data—by combining structure search with sparsity and interpretability controls.
2.	The proposed factor-quality–guided evolutionary search (FG-EAS) is a neat way to automate feature grouping.

### Weaknesses
1.	The core claim is “automated” discovery via FG-EAS, but the paper runs 50 candidates × 20–30 generations and trainsevery candidate for several epochs which is easily 1 000+ partial trainings. There’s no wall-clock, GPU-hour, or scaling analysis, so it’s hard to tell if HKAN is actually cheaper than just running a strong tabular model with Bayesian search. A clear ablation on budget vs. performance and a comparison to lightweight NAS would make the claim more credible.
2.	The nice visualizations and symbolic forms are shown for synthetic functions and for one real dataset (UCI Heart Disease), with a hand-picked story about factors 0 and 3 being retained. We don’t see whether, on big messy data like HomeCredit (696 features) or Delivery ETA (223 features), the extracted factors are still human-readable or just long spline soups. A paper that makes interpretability a central sales point should show 2–3 real, high-dimensional cases and report human-level summaries.
3.	Evolutionary operators are hand-crafted and not compared. Six mutation operators (add, remove, migrate, split, merge, delete) are introduced, but there is no study of which ones matter, how often they trigger, or whether a simpler “feature-migrate-only” EA would get within 1–2 AUC points.

### Questions
In Figure 2, we only see the efficiency of the parameters, but the actual computational complexity and latency are not reported. Could the authors include a discussion on these aspects?

### Soundness
3

### Presentation
2

### Contribution
3

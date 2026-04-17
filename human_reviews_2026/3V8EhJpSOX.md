# Causal-EPIG: A Prediction-Oriented Active Learning Framework for CATE Estimation

- Decision: Reject
- Scores: 4, 2, 6, 6

## Abstract
Estimating the Conditional Average Treatment Effect (CATE) is often constrained by the high cost of obtaining outcome measurements, making active learning essential. However, conventional active learning strategies suffer from a fundamental objective mismatch. They are designed to reduce uncertainty in model parameters or in observable factual outcomes, failing to directly target the unobservable causal quantities that are the true objects of interest. To address this misalignment, we introduce the principle of causal objective alignment, which posits that acquisition functions should target unobservable causal quantities, such as the potential outcomes and the CATE, rather than indirect proxies. We operationalize this principle through the Causal-EPIG framework, which adapts the information-theoretic criterion of Expected Predictive Information Gain (EPIG) to explicitly quantify the value of a query in terms of reducing uncertainty about unobservable causal quantities. From this unified framework, we derive two distinct strategies that embody a fundamental trade-off: a comprehensive approach that robustly models the full causal mechanisms via the joint potential outcomes, and a focused approach that directly targets the CATE estimand for maximum sample efficiency. Extensive experiments demonstrate that our strategies consistently outperform standard baselines, and crucially, reveal that the optimal strategy is context-dependent, contingent on the base estimator and data complexity. Our framework thus provides a principled guide for sample-efficient CATE estimation in practice.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a novel framework for active outcome acquisition for CATE estimation using observational data. In particular, the authors point out that simply applying active learning algorithms to CATE estimation can suffer from a fundamental objective mismatch compared to predictive tasks. The authors adopt a information-theoretic framework and propose two utility functions: one targets the underlying potential outcomes and the other one targets the CATE estimand. Moreover, the proposed framework can be applied to a wide range of bayesian CATE estimators, and the authors demonstrated improved performance over other baseline active learning algorithms through experiments.

### Strengths
- Results from both simulated and semi-synthetic experiments demonstrated strong performance over existing active learning algorithms. 

- Discussion on the potential outcome based vs. CATE estimate based utility is insightful.

### Weaknesses
- The technical innovation seems to be straight forward and directly built on top of EPIG. It would be a stronger paper if it included some theoretical guarantees (e.g. convergence, sample complexity, etc.)

- It seems like this framework would depend a lot on the posterior from the bayesian models. It would be nice to have some discussion and/or experiments that investigates posterior mis-specification.

### Questions
- It is a bit unclear how the proposed framework can also address the distribution shift problem, is it simply though taking the expectation over the target distribution?

- Are there assumptions under which the proposed strategies are optimal?

### Soundness
3

### Presentation
3

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
This paper introduces the principle of “Causal Goal Alignment” to establish Causal-EPIG, an active learning model that unifies theoretical principles with computational frameworks. It resolves the fundamental goal mismatch between traditional active learning (AL) and conditional average treatment effect (CATE) estimation. Grounded in information theory, this framework designs two sample acquisition strategies that demonstrate superior sample efficiency across multiple benchmark tasks. Experimental analysis further validates Causal-EPIG's superior efficiency compared to mainstream methods, alongside its adaptability and robustness across diverse estimators and data scenarios, demonstrating strong practical application potential.
However, the method retains certain limitations. Its theoretical framework relies on the strong assumption of no unobservable confounding, limiting applicability to broader observational data. Performance is highly sensitive to the quality of posterior calibration for the selected CATE estimator. Additionally, computational complexity increases with target set size, potentially encountering bottlenecks in large-scale, high-dimensional data environments. These aspects also provide valuable directions for future improvements. Overall, however, I still perceive the innovation and contribution of this paper as limited (my current understanding is that the main contribution lies in the modification of the utility function to a more causal form). Therefore, I tend to believe this paper falls slightly below the acceptance threshold for ICLR.

Translated with DeepL.com (free version)

### Strengths
see above

### Weaknesses
see above

### Questions
see above

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
2

### Summary
The paper proposes a active estimation framework for improving the sample efficiency of CATE estimation when outcome measurements are costly. It identifies a key misalignment in existing active learning methods, which typically optimize for uncertainty reduction in observable data or model parameters rather than the underlying causal estimands. To address this, the authors introduce the principle of causal objective alignment and develop an information-theoretic framework called Causal-EPIG, which quantifies the expected information gain about unobservable causal quantities. Two acquisition strategies are derived: one modeling the joint potential outcomes for comprehensive causal understanding, and another focusing directly on the CATE for greater efficiency. The framework is compatible with various Bayesian CATE estimators and demonstrates superior performance to existing baselines on synthetic and semi-synthetic datasets, showing consistent gains in sample efficiency.

### Strengths
- Proposes a clear and principled framework that addresses the objective misalignment in existing active learning approaches for CATE estimation, and formalizes it through a sound formulation based on mutual information.
- Demonstrates broad model compatibility, effectively integrating with several Bayesian CATE estimators, such as Gaussian Process models and Bayesian Causal Forests.  
- Offers supporting details on the implementation and comprehensive discussion on the intuition.
- Provides extensive experimental validation on synthetic and semi-synthetic benchmarks, showing consistent improvements in sample efficiency over existing baselines.

### Weaknesses
- Some of the definitions provided in the appendix should be incorporated into the main text to improve the overall coherence and readability of the paper. For instance, the formal definition of mutual information and the notation used in compared methods such as BALD could be clearly stated earlier.  
- The paper lacks theoretical analysis on the information gain. It would be valuable to establish how the efficiency of information gain relates to the accuracy of CATE estimation and whether there exist efficiency bounds that differentiate the proposed causal-oriented method from previous parameter-oriented approaches.  
- The work also lacks theoretical results on acquisition cost or sample complexity bounds, which are crucial for demonstrating the theoretical advantages of the proposed active learning framework.

### Questions
- Is the proposed framework adaptable to probabilistic CATE estimators beyond Bayesian models?  
- Could the authors provide theoretical justification showing that, for achieving a similar target (e.g., the same level of information gain), the proposed method requires fewer samples and thus serves as a more efficient active learning approach compared to existing methods?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces the Causal Objective Alignment (COA) principle, which mandates that acquisition functions directly target the CATE estimation. An information-theoretic approach adapting Expected Predictive Information Gain to quantify how much a query reduces uncertainty in CATE is also proposed, with two strategies covering joint potential outcomes and sole CATE. Extensive experiments on synthetic and semi-synthetic datasets show both strategies outperform baselines.

### Strengths
1. The Causal Objective Alignment principle fills a critical gap in traditional AL for CATE estimation by explicitly targeting unobservable causal quantities such as potential outcomes and CATE.
2. Experiments are designed to test the robustness of the proposed method across diverse scenarios, with exhaustive details in appendices to ensure reproducibility.
3. The paper is exceptionally clear, even for readers not familiar with the intersection of AL and causal inference.

### Weaknesses
1. The approach in this paper is consist of two strategies. But there is no guidelines for choosing between Causal-EPIG-$\mu$ and Causal-EPIG-$\tau$.
2. The framework relies heavily on Bayesian CATE estimators to compute posterior uncertainty. However, many practitioners use non-Bayesian estimators due to their computational efficiency and ease of implementation. These methods lack explicit posterior distributions, making Causal-EPIG inapplicable to them.
3. The paper uses a 50-sample warm-start to initialize the CATE estimator, but in many real-world scenarios, initial labeled data may vary in size. A sensitivity analysis on the warm-start sample size can be helpful.

### Questions
Could you please provide some theorectical analysis on the model effectiveness or efficiency?

### Soundness
4

### Presentation
4

### Contribution
3

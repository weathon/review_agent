# Benchmarking Stochastic Approximation Algorithms for Fairness-Constrained Training of Deep Neural Networks

- Decision: Accept (Poster)
- Scores: 2, 6, 2, 4

## Abstract
The ability to train Deep Neural Networks (DNNs) with constraints is instrumental in improving the fairness of modern machine-learning models. Many algorithms have been analysed in recent years, and yet there is no standard, widely accepted method for the constrained training of DNNs. In this paper, we provide a challenging benchmark of real-world large-scale fairness-constrained learning tasks, built on top of the US Census (Folktables,  Ding et al, 2021). We point out the theoretical challenges of such tasks and review the main approaches in stochastic approximation algorithms. Finally, we demonstrate the use of the benchmark by implementing and comparing three recently proposed, but as-of-yet unimplemented, algorithms both in terms of optimization performance, and fairness improvement. We release the code of the benchmark as a Python package at https://github.com/humancompatible/train.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces a new benchmark, based on the US census, for testing fairness-constrained learning tasks. The dataset is tested on various methods from the literature, which are implemented in a toolbox.

### Strengths
- A standard benchmark specific for testing the fairness of machine learning algorithms can surely be an interesting contribution, and the paper proposes to address this gap

### Weaknesses
- While the main claimed contribution of the paper is to provide a benchmark for fairness-constrained learning, I found the structure of the paper unclear, with most of the text reviewing existing literature and not many details on the introduced benchmark and dataset and on what the key novelties and results of the paper are.

- The proposed dataset is an instance of the ACS dataset, already published in [Frances Ding, Moritz Hardt, John Miller, and Ludwig Schmidt. Retiring adult: New datasets for fair machine learning. Advances in Neural Information Processing Systems, 34, 2021], and already used in the context of fairness applications. Consequently, the main contributions of the paper are the implementation and existing comparison of existing algorithms. In my opinion, this contribution, while potentially useful, is too incremental for publication in ICLR

### Questions
- What are the main contributions of the paper? Is the proposed dataset an instance of the ACS dataset? What does it make it particularly suitable for fairness testing compared to the original ACS dataset?

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
4

### Summary
This paper presents a benchmarking study and toolbox for evaluating stochastic approximation algorithms applied to fairness-constrained training of deep neural networks. The authors consider the general constrained empirical risk minimization (ERM) problem, where fairness is imposed through hard constraints (e.g., demographic parity, equal opportunity, equalized odds). They provide an open-source implementation integrating four major stochastic optimization algorithms (Stochastic Ghost (StGh), SSL-ALM, Augmented Lagrangian Method (ALM), and Stochastic Switching Subgradient (SSw)) within a unified framework built on PyTorch and Folktables datasets. The benchmark allows users to automatically construct constrained training formulations and apply fairness constraints across up to 5.7 billion protected subgroups from census-based datasets. Extensive experiments on the ACSIncome dataset compare the convergence speed, fairness violation, and test performance of each algorithm, as well as their robustness under different fairness metrics and constraint formulations. The work aims to standardize empirical evaluation practices in fairness-constrained deep learning and offer a reproducible experimental testbed for future research

### Strengths
1- Evaluates four distinct fairness-constrained optimization algorithms under identical experimental setups, providing valuable comparative insights.

2- Offers a transparent, well-engineered implementation with all datasets, hyperparameters, and metrics clearly documented.

3- Bridges fairness theory with realistic deep learning setups, enabling reproducible fairness experiments on real data.

4- Covers three key fairness notions (independence, separation, sufficiency) and links them to optimization constraints.

5- The benchmark includes stochastic ghost gradient methods, augmented Lagrangian, and subgradient-based solvers, giving a broad coverage of optimization paradigms.

### Weaknesses
1- Experiments are mainly conducted on a single dataset (ACSIncome), which restricts the scope of empirical validation. Inclusion of more varied domains (e.g., image or language tasks) would strengthen the claim of generality.

2- While results are reported, deeper analysis of when and why certain algorithms perform better (e.g., under which fairness metrics or subgroup imbalances) is missing.

3- Although billions of potential subgroup combinations are mentioned, the experiments do not convincingly demonstrate performance at that scale.

4- The paper does not clearly discuss how the proposed framework will be maintained or integrated with existing fairness toolkits, which may limit its long-term impact.

### Questions
1- How does the framework handle multiple simultaneous protected attributes, especially when fairness constraints interact (e.g., intersectional fairness)?

2- Are the results robust to different data distributions or dataset shifts (e.g., subsampled or noisy features)?

3- How computationally expensive are these fairness constraints for large-scale DNNs compared to regularization-based approaches?

4- Could the benchmark include group fairness metrics beyond accuracy, such as calibration or counterfactual fairness?

5- How do hyperparameter settings for fairness constraints (e.g., δ thresholds) influence the convergence behavior of the tested algorithms?

### Soundness
3

### Presentation
4

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
This paper provides a benchmark of several stochastic optimization algorithms for training deep neural networks under different fairness constraints. They consider datasets built on top of the US Census (Folktables) dataset. They consider the popular group fairness measures independence (statistical parity), sufficiency, and separation. Several experiments have been included in the paper that compare the performance of these existing optimization techniques in minimizing the measures of fairness over the datasets.

### Strengths
-- The paper has included a lot of experiments on different algorithmic (stochastic) variants of implementing fairness as a constraint during training. Methods considered include: (i) Stochastic ghost method; (ii) Stochastic smoothed and linearized AL method; (iii) Stochastic switching subgradient method, etc.

-- They consider the three popular fairness notions, and also multiple datasets. 

-- Presentation is generally good.

### Weaknesses
-- While the vast experiments are highly appreciated, I believe this paper is more suitable as a dataset/benchmark paper. The stochastic optimization algorithms already exist in the literature and have also been used for constraint optimization. The paper applies these constrained optimization variants for the specific constraint of group fairness and studies their performance. 

-- Indeed, the paper is quite comprehensive in their experimentation. But, still, the novelty would be limited for such a venue since it is more like a survey of applying different existing techniques to the fairness constraint and seeing the performance. The paper could also be better suited as a survey paper. Though there do exist several other survey papers on fairness in literature, and it would be important to highlight what is the technical gap in existing survey/benchmarking papers that this paper fills.

-- The measures of fairness are mainly the three popular ones. 

-- Some works compare tradeoffs between different group fairness measures. E.g. The possibility of fairness: Revisiting the impossibility theorem in practice. 
It would be good to compare with them.

### Questions
Q1. What would be the gap in existing survey papers or benchmarking papers on algorithmic fairness that this paper fills?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents a benchmark for evaluating stochastic approximation algorithms in fairness-constrained training of deep neural networks. Built on the US Census dataset, the benchmark enables large-scale experiments on fairness objectives formulated as constrained ERM problems. The authors review existing algorithms and implement three recent ones, namely Stochastic Ghost, Stochastic Smoothed and Linearized Augmented Lagrangian Method, and Stochastic Switching Subgradient, and compare them to SGD baselines. The study highlights the lack of unified toolkits and provides a first step toward standardized evaluation for fairness-constrained optimization prooblems.

### Strengths
1. This work provides a reproducible and extensible benchmark framework for fairness-constrained deep learning, filling a gap in the literature where no unified platform existed. 
2. The writing is very clear, and the notations are consistent. I appreciate Table 3, where the authors review a wide range of stochastic constrained optimization algorithms with a structured taxonomy and theoretical assumptions.
3. The work evaluates multiple fairness criteria, independence, separation, sufficiency, and Wasserstein distance, showing nuanced trade-offs among methods.

### Weaknesses
1. The paper primarily implements existing algorithms rather than introducing a new one. While benchmarking is valuable, this may limit perceived theoretical contribution.
2. Only one dataset with a binary protected attribute is used. The scalability and generalization to multiple attributes have not been tested.
3. The presentation of the experimental results can be improved. The current figures are difficult to read.
4. There is no discussion of hyperparameter search across algorithms. Why are the parameters set as in lines 366-375? The results could reflect suboptimal settings rather than intrinsic algorithmic differences.

### Questions
1. See some questions in Weaknesses.
2. Could the work extend to multi-group or intersectional attributes?
3. Are there plans to include computational efficiency or memory usage comparisons?

### Soundness
3

### Presentation
3

### Contribution
3

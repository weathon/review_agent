# Differentially Private Synthetic Data via APIs 4: Tabular Data

- Avg Score: 5.00
- Decision: Reject
- Scores: 2, 6, 4, 8

## Abstract
Tabular data is one of the most widely used formats in practice, yet much of it remains inaccessible due to privacy concerns. Synthetic data generation with formal privacy guarantees, i.e. differential privacy (DP), offers a promising solution to enable data sharing while protecting sensitive information. Despite extensive study, state-of-the-art methods often focus on minimizing low-order marginal query errors and overlook the challenges posed by high-order correlations. To address this gap, we adapt the Private Evolution (PE) framework, originally developed for DP-compliant image and text synthesis, to tabular data. We introduce Tab-PE -- an algorithm for generating synthetic tabular data under DP. Tab-PE refines a synthetic dataset by an evolutionary process that leverages APIs to generate variations of the data, privately evaluate them, and retain the highest-quality samples. While the original PE requires access to large foundation models, Tab-PE is computationally efficient with heuristic APIs specialized for tabular data. Through extensive experiments on real-world and simulation datasets, we demonstrate that Tab-PE substantially outperforms prior baselines on datasets exhibiting high-order correlations. Compared to the best baseline -- AIM, Tab-PE improves classification accuracy by up to 10\% while running 28$\times$ faster.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces Tab-PE, a differentially private data synthesis algorithm that adapts the Private Evolution (PE) framework to the tabular domain. The method iteratively perturbs and selects synthetic samples through a DP scoring mechanism based on nearest-neighbor histograms, aiming to capture high-order correlations between attributes without relying on query workloads or model-based approaches. The authors claim that Tab-PE achieves better downstream ML performance than prior baselines under comparable privacy budgets while being significantly faster and simpler to implement.

### Strengths
- Tab-PE is computationally lightweight, requires no model training, and scales efficiently to medium-sized tabular datasets, offering a practical alternative to query- or model-based DP synthesizers.

- Tab-PE achieves good downstream accuracy and privacy-utility trade-offs under moderate $\varepsilon$, outperforming several baselines.

### Weaknesses
- The paper overlooks major prior works that explicitly highlight or address high-order correlations in DP tabular synthesis, including [1], which identifies modeling higher-order correlations as a key open problem (at the end of Sec. 3.1), and Kamino [2], a constraint-aware DP synthesizer that directly tackles this challenge. **Failing to acknowledge or compare with these works reveals an incomplete understanding of the literature and weakens the positioning of the claimed contribution.**

- The authors never provide a concrete definition of “higher-order correlation”, in stark contrast to [1], which formalizes the concept through specific instances such as denial constraints and conditional functional dependencies. Here, the term is used loosely as a buzzword and assessed only through downstream ML performance, which is an unreliable proxy. It remains unclear whether the proposed Tab-PE method actually addresses the stated problem.

- Tab-PE performs substantially worse than AIM on low-order fidelity metrics (e.g., 1-way and 2-way TVD). If a method fails to preserve even low-order marginals, there is little reason to believe it can capture higher-order ones. 

- The sampling procedure in PE (top-K) is inherently biased, leading to tail loss and mode concentration. Tab-PE inherits this limitation, which likely explains its poor 1-TVD and 2-TVD results compared to state-of-the-art approaches such as AIM.

- Tab-PE uses a single $\lambda$ to trade off categorical and numerical scales in DP-NN. This hyperparameter must be dataset-dependent and the results could be very sensitive to this choice. This is a weakness inherent to the proposed algorithm.

- The core algorithm is a direct application of PE with simple tabular adaptations (Gaussian perturbation for numerical attributes, resampling for categorical ones), reusing the same scoring and update loop from prior PE papers. It has very limited technical novelty and does not bring new insights to the field of DP tabular data synthesis.

- The paper deviates from established evaluation practices in DP tabular synthesis. Rather than benchmarking on standard datasets  widely used in prior work (e.g., Adult, Bank, Census, Hospital), it selects its own suite tailored to its method and highlights improvements. This practice hinders cross-paper comparison and undermines the credibility of the empirical claims. **A rigorous study should first have validated Tab-PE on recognized benchmarks before proposing new ones**.


[1] Hu, Yuzheng, et al. "Sok: Privacy-preserving data synthesis." 2024 IEEE Symposium on Security and Privacy (SP). IEEE, 2024.

[2] Ge, Chang, et al. "Kamino: Constraint-aware differentially private data synthesis." VLDB 2021

### Questions
Can you answer the following question: Is it possible for a synthetic tabular dataset to preserve high-order correlations present in the original dataset yet exhibit poor low-order fidelity?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces Tab-PE, a method for generating differentially private synthetic tabular data by adapting the popular Private Evolution (PE) framework to tabular data. The paper argues that existing SOTA tabular DP-SDG methods, like marginal-based approaches, struggle to capture high-order correlations due to their restricted simplicity in measuring low-dimensional marginals. They argue that Tab-PE is an effective alternative as it does not need to explicitly instantiate many high dimensional marginal queries. They show empirically Tab-PE outperforms on datasets with complex dependencies while being much faster than current SOTA marginal-based methods like AIM.

### Strengths
- The paper clearly identifies and empirically validates a meaningful limitation in existing DP-SDG algorithms namely, their inability to scale to many columns and their inability to capture high-order correlations.
- The paper is generally well organized, includes detailed algorithm descriptions and ablations, and the authors provide an anonymized code repository.
- The adaptation of the PE framework to tabular data without using LLMs/foundation models is conceptually straightforward yet shown to be very effective against SOTA approaches .

### Weaknesses
- The paper should more clearly delineate how Tab-PE differs from the closest related work. Namely the approach of Swanberg et al. and the PrivGSD method (see questions below).
- The presentation surrounding the notion of "APIs" is unclear/misleading since, as far as I can tell, the work does not use APIs in the traditional PE sense e.g., leveraging foundation models.
- Some ablations (e.g., replacing Tab-PE’s variation API with PrivGSD’s crossover/mutation) are relegated mostly to the appendix but would be valuable to feature more prominently in the main text to clarify differences with the closest related work.
- The evaluation focuses on datasets with only a few columns. Given the central claim about scalability and modeling high-order correlations, it would be more valuable to include experiments on higher-dimensional data where this benefit is much clearer.

### Questions
1. The general presentation around "APIs" is somewhat confusing and misleading. Why retain the "API" terminology if no external API calls or foundation models are actually used as in traditional PE?
2. Related to the above, more could be done to delineate this work from Swanberg et al. which seems the closest related work from a tabular PE perspective. Why have you chosen not to empirically compare to this approach? 
3. It seems that PrivGSD is the closest related DP-SDG approach since it adapts the genetic algorithm approach to produce private tabular data which is closely related to Private Evolution. The specific differences between PrivGSD should be made much clearer in the main paper. Could you summarise the main differences?
4. The main argument for Tab-PE is that it retains high-order correlations and is much faster than competing methods like AIM. However, most of the benchmark datasets that are used have only a small number of columns. I would have preferred to have seen results that show Tab-PE on higher-dimensional datasets where the argument about capturing high-order correlation is usually much more necessary than on smaller datasets. Do you think Tab-PE can scale effectively to these scenarios?
5. Have you thought about lightweight hybrid approaches to extend Tab-PE with query-based methods (e.g., marginal information)? It seems to me a key weakness of Tab-PE is that it doesn’t capture 1-way or 2-way marginals as well as methods like AIM (which makes sense as AIM is explicitly trained to do so). However, it is fairly lightweight (from a DP perspective) to integrate marginal information e.g., using 1-way marginal information to initialize the Tab-PE approach (instead of random data).

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes a new tabular data synthesis algorithm that leverages the recent Private Evaluation (PE) technique. The method uses an API to generate candidate synthetic records and employs private histograms to iteratively select synthetic records that are more similar to the private data. Extensive experiments demonstrate the effectiveness of the proposed method across various scenarios, including low-order correlations, high-order correlations, and synthetic datasets.

### Strengths
1. The use of a simple XOR experiment to illustrate the weaknesses of existing marginal-based synthesis algorithms is interesting and well-motivated.

2. Employing a simple synthesis API (e.g., directly perturbing data through random selection or Gaussian noise) instead of relying on LLMs is an intriguing choice that still achieves strong performance.

3. The proposed method is simple, easy to implement, and computationally efficient.

### Weaknesses
1. Although the paper is overall interesting and well-written, I am concerned about the performance of Tab-PE. As shown in Table 1, the overall improvement over baseline methods is modest (at most around 10%), while the fidelity (even for simple one-way marginals) is significantly worse (up to 10 times worse on the Person Activity dataset). Given that low-order marginals are a crucial fidelity metric and have many applications (e.g., point/range queries) in tabular data, this represents a critical weakness. The paper should discuss this issue more thoroughly, especially regarding how Tab-PE should be used in practice.

2. While it is interesting to distinguish between low-correlation and high-correlation datasets, the performance differences of Tab-PE across these datasets are not particularly significant. In addition, Table 5 (fidelity evaluation on the Breast Cancer dataset) lacks bold or underlined highlights; this can be either a typo or a mistake in the results. Either way, Table 5 again shows that the advantage of Tab-PE over marginal-based methods lies mainly in ML prediction and embedding metrics, which is unsurprising and has already been validated by many prior studies.

3. Tab-PE relies on several pieces of so-called "public" information about the private tabular data, such as the domains of continuous and categorical values and class distributions. While the authors provide some justification for why these can be treated as public, directly using such information weakens the end-to-end privacy guarantee of Tab-PE and may not be practical when these statistics are unavailable. The authors should discuss these assumptions more carefully and provide a more rigorous privacy analysis.

### Questions
Q1. Given the relatively modest performance gains (and in fidelity, worse results than baseline methods), why do the authors claim that Tab-PE represents a new paradigm for tabular data synthesis? Under what conditions should Tab-PE be preferred over marginal-based methods?

Q2. Why does the performance of Tab-PE appear similar on both high-correlation datasets (Table 1) and low-correlation datasets (Table 5)? This similarity seems to contradict the motivation for distinguishing between these two types of datasets.

Q3. Why are certain statistics treated as public information, given that such assumptions could risk leaking private information?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduced Tab-PE, a proposed method extending the private evolution (PE) framework for text and images into tabular data. Unlike prior PE methods relying on foundation models, Tab-PE employs simple heuristic APIs to evolve candidate synthetic samples through controlled perturbations and DP nearest-neighbor scoring. The paper argues that this approach better models high-order correlations in tabular data where marginal-based methods struggle.

### Strengths
- The paper is well written and easy to follow. It highlights its novel contributions well and makes clear and distinct references to previous literature. Figures are clear and well motivated.
- The adaptation of the PE framework to tabular data is novel and interesting, and has not been studied in the literature before.
- The paper makes an important empirical point that current DP tabular synthesis methods implicitly optimize for low-order correlations and fail under high-order dependencies. The “XOR stress test” is a clear and well-designed diagnostic benchmark. 
- Tab-PE achieves consistent improvements across varied datasets and privacy levels, showing both utility gains and significant runtime savings. The scalability of the method makes it highly useful for real-world applications, such as financial or healthcare tabular data. Extensive experiments across synthetic (XOR, SCM) and real-world datasets demonstrate substantial performance gains (up to +10% accuracy) and improved computational efficiency (up to 28× faster than AIM) under comparable privacy budgets.
- The baselines are well chosen and ablations study is well-done with the selection strategy, polynomial decay schedule, and hyperparameter sensitivity studies and strengthen confidence in the design choices.

### Weaknesses
- While the authors reuse DP composition results from prior PE papers, the paper does not formalize why the DP nearest-neighbor histogram implicitly captures high-order correlations, as many other DP papers do. A more rigorous connection between the algorithmic process and statistical estimation of joint distributions would improve the conceptual depth. A theoretical contribution on how the nearest-neighbor scoring function relates to statistical query families or mutual information under DP noise would be helpful to make this paper useful beyond a purely engineering/empirical perspective.

### Questions
- Does Tab-PE’s advantage persist under stricter privacy (ε < 1)? Some prior DP synthesizers degrade sharply; how robust is your method in low-privacy regimes?
- For methods like AIM and RAP++, were hyperparameters re-tuned under identical privacy budgets, or were default settings used? Could differences in optimization explain part of the observed utility gap? 
- How do you choose the neighborhood radius or kernel bandwidth in the DP_NN_HISTOGRAM? Is it tuned on public data, or does it affect privacy accounting if selected adaptively?

### Soundness
4

### Presentation
4

### Contribution
3

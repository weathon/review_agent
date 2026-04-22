# Uncertainty-driven Embedding Convolution

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 8, 4, 4, 2

## Abstract
Text embeddings are essential components in modern NLP pipelines. Although numerous embedding models have been proposed, no single model consistently dominates across domains and tasks. This variability motivates the use of ensemble techniques to combine complementary strengths. However, most existing ensemble methods operate on deterministic embeddings and fail to account for model-specific uncertainty, limiting their robustness and reliability in downstream applications. To address these limitations, we propose Uncertainty-driven Embedding Convolution (UEC). UEC first transforms deterministic embeddings into probabilistic ones in a post-hoc manner. It then computes adaptive ensemble coefficients based on embedding uncertainty, derived from a principled surrogate-loss formulation. Additionally, UEC employs an uncertainty-aware similarity function that directly incorporates uncertainty into the similarity scoring, providing a theoretically grounded and efficient surrogate to distributional distances. Extensive experiments on diverse benchmarks demonstrate that UEC consistently improves both performance and robustness by leveraging principled uncertainty modeling.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper addresses the limitation of deterministic embedding ensembles, which fail to account for model-specific uncertainty, by proposing the Uncertainty-driven Embedding Convolution (UEC) method. UEC first converts pre-trained deterministic embeddings into probabilistic representations via a post-hoc Laplace approximation to quantify their uncertainty. It then derives Bayes-optimal ensemble weights that adaptively favor more confident models through a probabilistic contrastive loss. Based on this, the authors introduce an uncertainty-aware similarity function. Extensive experiments demonstrate that UEC consistently enhances performance and robustness across diverse NLP tasks by leveraging principled uncertainty modeling. The work provides a theoretically grounded and practical method for creating more reliable embeddings through ensembles.

### Strengths
1. The idea of moving from deterministic to probabilistic embeddings for ensemble embedding learning is interesting and well-motivated. It allows quantifying model-specific uncertainty, thus improving robustness. The "jaguar" example effectively illustrates the need for this method.

2. The paper provides strong theoretical foundations. Deriving the ensemble weights as a Bayes-optimal solution under a surrogate loss and connecting the similarity function to the 2-Wasserstein distance with a bounded error are significant contributions that add rigor and credibility to the proposed method.

3. Extensive experiments have demonstrated the effectiveness of UEC, covering multiple tasks (Retrieval, Classification, STS, Bitext Mining, Clustering, Reranking) across diverse benchmarks (MIRACL, MMTEB). The consistent outperformance of UEC over strong baselines, including model merging and other ensemble techniques, provides robust evidence for its effectiveness.

### Weaknesses
1. The paper is very dense, especially Section 3. The derivations, while valuable, could be summarized more intuitively in the main text, with full details reserved for the appendix. Besides, the transition from the loss function in Eq. (2) to the tractable approximation in Eq. (3) is somewhat abrupt.
2. There are certain hyperparameters, such as temperature and β. However, there is no analysis of UEC's sensitivity to the choices of these hyperparameters. 
3. While this ensemble method combines the strengths of specialists, i.e., different embedding models, LLMs perform like generalists. A comparison with LLM’s embeddings could justify the need for the ensemble method in practice.

### Questions
1. Would a general model, such as an LLM, be sufficient to handle the diverse tasks robustly? How would the proposed ensemble method perform compared to LLMs when applied to different tasks? 
2. How would the hyperparameters affect the performance of UEC?
3. In the poem classification task, UEC underperforms an individual model GTE-MB, when it ensembles three weaker models. Can the ensemble apply to stronger models, such as GTE-MB, and still yield performance gains?
4. How well could the proposed method estimate the mean and variance for the individual embeddings and the ensembled one, under different data sparsity settings? Are the estimations sensitive to noise?

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
3

### Summary
This paper proposes Uncertainty-driven Embedding Convolution (UEC), which ensembles pre-trained text-embedding models by converting their deterministic embeddings into Gaussian representations via a post-hoc Laplace approximation, weighting and combining them by convolution, and using a probabilistic similarity metric that approximates distributional distance. The resulting framework improves robustness and retrieval accuracy across multilingual and multi-domain tasks, while remaining computationally lightweight. Most of the methods are very solid with many inspiration taken from applied statists literature.

### Strengths
1. The method is conceptually clean and well grounded from existing literature.

2. The paper is clearly written and mostly mathematically correct, although with some times assumptions not specifically defined.

3. The empirical results are convincing, the authors covering retrieval, classification, and semantic textual similarity.

4. The approach is practical, post-hoc, scalable, and effective without model retraining.

### Weaknesses
1. The core components are standard statistical techniques: Laplace approximation inverse-variance weighting for Gaussian combination, and uncertainty-aware similarity based on probit approximations. Hence, I think the paper’s novelty lies mostly in assembling these ingredients into a single coherent pipeline for embedding ensembles, not in introducing fundamentally new theory.

2. The “Bayes-optimal” claim is slightly overstated given that the surrogate loss drops query-dependent terms (Eq. 3).

3. It would be valuable to analyse uncertainty calibration (ECE, reliability diagrams) to confirm that the modellled variances truly reflect epistemic uncertainty rather than functioning as learned scaling factors.

4. Why not included some other baselines (e.g., Bayesian model averaging or deep ensemble uncertainty fusion)?

### Questions
Please see my questions from weakness section.

### Soundness
3

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
2

### Summary
### Summary

The paper proposes **Uncertainty-Driven Embedding Convolution (UEC)**, a post-hoc probabilistic ensemble method for text embedding models. It combines uncertainty estimation, probabilistic ensembling, and calibrated similarity scoring to improve robustness in retrieval and representation tasks.

UEC works in three steps:
1. **Laplace Approximation (LA)** on the last layer produces a Gaussian posterior over embeddings, modeling epistemic uncertainty.
2. **Gaussian convolution** fuses multiple model embeddings into a combined mean and covariance with theoretically optimal coefficients.
3. **Probit similarity** computes uncertainty-aware similarity, provably approximating the Wasserstein-2 and Jeffreys divergences.

Experiments on MMTEB (retrieval, STS, reranking, clustering, bitext mining) show consistent improvements over deterministic ensembles with negligible computational overhead.

### Strengths
### Strengths

1. **Principled probabilistic foundation** — Correct derivation of Gaussian embeddings from Laplace posteriors.
2. **Theoretical rigor** — Ranking-preserving link between probit similarity, Wasserstein-2, and Jeffreys divergence.
3. **Practicality** — Post-hoc, model-agnostic, and nearly cost-free in computation.
4. **Empirical consistency** — Strong performance across MMTEB tasks and clear ablations.
5. **Trustworthy AI alignment** — Explicit uncertainty propagation enhances interpretability and robustness.
6. **Reproducibility** — Clear methodology, dataset references, and open hyperparameter reporting.

### Weaknesses
### Weaknesses

1. **Limited type of uncertainty.**  
   The method models only **epistemic uncertainty** (parameter uncertainty via Laplace Approximation on the last layer). It does not capture **aleatoric** or **predictive uncertainty**, nor propagate epistemic uncertainty through deeper layers.  

2. **Strong independence and diagonal assumptions.**  
   The Gaussian convolution assumes independent embeddings and diagonal covariances. These assumptions are rarely valid for transformer-based embeddings.  

3. **Calibration not analyzed.**  
   The paper assumes variances from LA are well calibrated, but provides no empirical evidence (e.g., ECE or reliability plots).  

4. **Benchmark scope limited.**  
   Only MMTEB is used. Out-of-distribution or multilingual setups would better demonstrate uncertainty benefits.

5. **Incremental conceptual contribution.**  
   The method is a sophisticated recombination of known tools (Laplace, Gaussian averaging, probit correction).  

6. **Fairness risk.**  
   Down-weighting uncertain embeddings could amplify demographic bias if uncertainty correlates with data imbalance.

### Questions
### Questions for Authors

1. Which **type(s) of uncertainty** does UEC explicitly model? From the derivations, it seems to capture **epistemic uncertainty** only (via parameter uncertainty in the last layer). Please confirm and discuss this limitation.
2. How sensitive are your Bayes-optimal coefficients to correlation among models? Would cross-covariances alter the results?
3. How robust is the Laplace fit when performed on out-of-domain data (e.g., fitting LA on MS MARCO, evaluating on STS)?
4. Could you provide calibration diagnostics (ECE, reliability curves) to evaluate whether \(\sigma_s^2\) reflects meaningful confidence?
5. Would deeper (full-network) Laplace approximations substantially alter uncertainty estimates?
6. Please include a sensitivity study on \(\beta\) and temperature parameters.
7. Can you empirically validate that the probit similarity preserves ranking order with respect to exact Wasserstein/Jeffreys distances?

### Soundness
3

### Presentation
3

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
This paper introduces Uncertainty-driven Embedding Convolution (UEC) which is a method to construct ensemble of embeddings in an uncertainty aware method. UEC consists of three key components. First, it converts pre-trained deterministic embeddings into probabilistic embeddings in a post-hoc fashion. Second, it computes adaptive ensemble weights based on estimated uncertainty, down-weighting less reliable embeddings. This weighting strategy is grounded in a Bayes-optimal solution under a surrogate loss. Third, UEC introduces an uncertainty-aware similarity function that incorporates both distance and variance into the similarity score, offering a theoretically grounded and efficient surrogate to distributional distances. Extensive experiments on diverse benchmarks demonstrate that UEC consistently improves both performance and robustness by leveraging principled uncertainty modeling

### Strengths
- This work provides a way to ensemble embeddings in an uncertainty aware manner.

### Weaknesses
- It seems like this work contains a lot of jargon without providing any insights on what the proposed method is and how does it fit into the existing literature.
- This work has major theoretical flaws.

### Questions
- Can the autors give some insight on how they are converting the deterministic embeddings into probabilisitic ones in section 3.1? It is very hard to follow currently.
- The solution of eqn 3 should put probability 1 on the lowest value. Why do the authors claim the result in eqn 4?

### Soundness
2

### Presentation
2

### Contribution
2

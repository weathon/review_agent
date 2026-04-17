# TESSAR: Geometry-Aware Active Regression via Dynamic Voronoi Tessellation

- Decision: Accept (Poster)
- Scores: 8, 8, 4

## Abstract
Active learning improves training efficiency by selectively querying the most informative samples for labeling. While it naturally fits classification tasks–where informative samples tend to lie near the decision boundary–its application to regression is less straightforward, as information is distributed across the entire dataset. Distance-based sampling is commonly used to promote diversity but tends to overemphasize peripheral regions while neglecting dense, informative interior regions. To address this, we propose a Voronoi-based active learning framework that leverages geometric structure for sample selection. Central to our method is the Voronoi-based Least Disagree Metric (VLDM), which estimates a sample’s proximity to Voronoi faces by measuring how often its cell assignment changes under perturbations of the labeled sites. We further incorporate a distance-based term to capture the periphery and a Voronoi-derived density score to reflect data representativity. The resulting algorithm, *TESSAR* (TESsellation-based Sampling for Active Regression), unifies interior coverage, peripheral exploration, and representativity into a single acquisition score. Experiments on various benchmarks demonstrate that TESSAR consistently achieves competitive or superior performance compared to prior state-of-the-art baselines.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper proposes TESSAR, a geometry-aware active learning framework for regression. The key idea is to model uncertainty through VLDM, which measures the geometric instability of samples under small perturbations of labeled points. By combining VLDM with distance and density terms, the method dynamically selects informative samples in a model-agnostic manner.

### Strengths
1. The paper proposes a novel geometric perspective, which is interesting. The use of Voronoi tessellation and the proposed VLDM provide an innovative and well-motivated formulation of uncertainty for regression tasks.
2. The method achieves notable performance gains across various datasets and baselines.

### Weaknesses
1. The paper should include a static Voronoi or less frequent update baseline to show the effect of dynamic tessellation. Also what is the selection stragegy of Gaussian perturbation parameter? 
2. It would be helpful to include an ablation replacing VLDM with simpler geometric proxies such as nearest-neighbor distance or local label variance. This would clarify whether the performance gains truly stem from the proposed VLDM formulation or can be achieved by simpler uncertainty measures.
2. Voronoi-based geometry can degrade in high-dimensional spaces due to distance concentration. Is the proposed approach still effective in high-dimensional regression tasks?

### Questions
See weakness.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces TESSAR, an active learning framework for regression tasks that leverages Voronoi tessellation to improve sample selection. The core innovation is the Voronoi-based Least Disagree Metric (VLDM), which identifies informative samples near Voronoi faces in the input space, addressing the limitations of traditional distance-based methods that often overlook dense interior regions. VLDM is combined with a distance score for peripheral exploration and a density-based representativity term, resulting in a unified acquisition function. The authors provide theoretical motivation linking Voronoi faces to high predictive variance, along with an efficient approximation for VLDM computation. Empirical evaluations on 14 tabular regression datasets show that TESSAR matches or surpasses state-of-the-art baselines such as LCMD and BADGE in terms of RMSE, although its runtime increases with larger datasets, reducing efficiency, a limitation the authors explicitly address.

### Strengths
The use of Voronoi tessellation as a geometric approximation for disagreement-based sampling in regression is a well-motivated idea. Unlike classification tasks with clear decision boundaries, regression lacks such structures, and this paper elegantly adapts the concept via VLDM. The theoretical analysis in Section 2.2, showing that points near Voronoi faces exhibit high variance under Lipschitz assumptions, provides solid grounding.

TESSAR integrates informativeness, diversity, and representativity into a single score, with efficient dynamic updates (Algorithm 2) to avoid recomputing VLDM naively. The empirical consistency of VLDM (Figure 3) and ablation studies (Figure 4) clearly show the complementary benefits of the components in TESSAR.

Evaluations on diverse datasets (e.g., Protein, Road, Stock) using performance profiles and penalty matrices (Figure 5, Table 1) highlight TESSAR's consistent superiority. It achieves the highest RA(0) of 41% in performance profiles, outperforming LCMD (29%). Runtime comparisons (Table 3) indicate it's competitive with baselines, with increases justified by better performance.

The paper is well-written, with clear pseudocode, evaluations and detailed appendices on datasets and metrics, and thoughtful discussion of limitations (e.g., computational cost in large pools, homoskedasticity assumption).

### Weaknesses
TESSAR's runtime scales with pool size and perturbations, making it slower on very large datasets (e.g., 547s on Road vs. ~150s for Coreset). The authors acknowledge this and suggest optimizations, but more scaling experiments (e.g., on million-scale data) could strengthen the case. Active Learning is designed for large data sets.

The related works section is comprehensive but could better highlight differences from clustering-based methods like LCMD.

### Questions
Can we pre-evaluate how much TESSAR will outperform random sampling on a given dataset?

 Following LCMD’s analysis (Holzmüller et al., 2023), which showed that the ratio of initial RMSE to MAE on a small training set strongly predicts the benefit of LCMD-TP over random selection, a similar diagnostic could be developed for TESSAR. For example,
 a pre-evaluation metric could forecast TESSAR’s sample efficiency gains, helping practitioners decide when to deploy it. 
		
How sensitive is TESSAR to the choice of feature extractor (e.g., vs. raw inputs or other  architectures)? The method relies on a feature mapping (e.g., neural network outputs), and varying this (e.g., with PCA) might affect Voronoi partitions and VLDM scores, warranting empirical sensitivity analysis.

In the theoretical analysis, the Lipschitz assumption is reasonable, but are there empirical cases where it fails, and how does TESSAR perform there?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper discusses TESSAR - an active learning method for regression that picks new points to label by using geometrical structure. TESSAR builds a Voronoi diagram around the currently labeled points and seeks unlabeled samples that lie near the boundaries of the diagram, where the model is least certain. It then balances this with two complementary signals: a score that encodes the density of points or how representative they are, and a score the encodes diversity by measuring distances from labeled data. The result is a single scoring rule that aims to be informative, diverse, and representative. Across many tabular datasets, this approach matches or beats strong baselines, with a practical update trick to keep computation reasonable.

### Strengths
The paper presents a clear geometrical idea for the solution of a well motivated problem in active learning, which is integrated into a unified comprehensive strategy.
Attention is given to computational complexity.

### Weaknesses
The technical sections are hard to follow, as several derivations are terse.
The experiments are limited to modest-size tabular datasets; given the inherent computational complexity of the method, more challenging datasets seem appropriate.

The authors appear unfamiliar with several active learning works that address a similar problem via coverage. Although those papers target classification at the low-budget regime, their focus - selecting spatially diverse and representative points from the underlying distribution - is closely related. Instead of employing a Voronoi diagram, the optimization is formulated in terms of set coverage; crucially, because the objective is submodular, the greedy solution enjoys efficient approximation guarantees. Published extensions in that line of work also incorporate uncertainty terms. It is therefore essential to compare against this coverage-based literature.

1) Yehuda, Ofer, et al. "Active learning through a covering lens." Advances in Neural Information Processing Systems 35 (2022): 22354-22367.
2) Bae, Wonho, Junhyug Noh, and Danica J. Sutherland. "Generalized coverage for more robust low-budget active learning." European Conference on Computer Vision. Cham: Springer Nature Switzerland, 2024.

### Questions
Please address the relationship to the coverage-based literature discussed above, and clarify the method’s scalability to larger datasets.

### Soundness
3

### Presentation
2

### Contribution
3

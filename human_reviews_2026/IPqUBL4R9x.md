# Distributional Machine Unlearning via Selective Data Removal

- Decision: Accept (Poster)
- Scores: 8, 2, 8, 6

## Abstract
Machine learning systems increasingly face requirements to remove entire domains of information—such as toxic language or biases—rather than individual user data. This task presents a dilemma: full removal of the unwanted domain data is computationally expensive, while random partial removal is statistically inefficient. We find that a domain's statistical influence is often concentrated in a small subset of its data samples, suggesting a path between ineffective partial removal and unnecessary complete removal. We formalize this as distributional unlearning: a framework to select a small subset that balances forgetting an unwanted distribution while preserving a desired one. Using Kullback-Leibler divergence constraints, we derive the exact removal-preservation Pareto frontier for Gaussian distributions and prove that models trained on the edited data achieve corresponding log-loss bounds. We propose a distance-based selection algorithm and show it is quadratically more sample-efficient than random removal in the challenging low-divergence regime. Experiments across synthetic, text, and image datasets (Jigsaw, CIFAR-10, SMS spam) show our method requires 15–82% less deletion than full removal for  strong unlearning effects, e.g., halving initial forget set accuracy. Ultimately, by showing a small forget set often suffices, our framework lays the foundations for more scalable and rigorous subpopulation unlearning.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper presents an intuitive method to select a subset of the forget data to remove/unlearn to balance model utility and unlearning performance.

### Strengths
The method is intuitive, novel, and well presented. The claims are theoretically and empirically backed up.

What I find especially relevant as a contribution is that this method can be used together with existing unlearning methods, improving them via a better composition of retain and forget sets.

### Weaknesses
The computational overhead for large datasets (e.g. web-scale data) is problematic. As this will require approximations to have a feasible retain set size, the performance of the method when the retain set is only available in an approximate manner (e.g. representative samples of an LLM training set) would be of interest as the results in table 2 already show a significant performance difference from Gaussian -> more realistic datasets (SMS, etc.).

### Questions
Q1 See question in weaknesses.

Q2 How should your method be used to pick the correct forget set (size of f samples) in practice when pairing it with an unlearning method (as in Table 5.)?

Q3 (Out of interest, feel free to ignore) Do you envision a version of your method that works in a setting where only the forget set is available?

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
3

### Summary
This paper introduces "distributional unlearning," a novel information-theoretic framework to address the problem of removing entire domains or subpopulations of data from a training set. The authors argue that full domain removal is computationally prohibitive, while random partial removal is statistically ineffective. Their key insight is that a domain's statistical influence is often concentrated in a small subset of its data. The proposed framework formalizes this by modeling the "forget" and "retain" data as two distinct probability distributions, p1 and p2. The goal is to find a data subset that maximally increases the KL divergence from p1 (removal) while minimally increasing the KL divergence from p2 (preservation). The paper provides strong theoretical results, deriving the removal-preservation Pareto frontier for exponential families and proving downstream log-loss guarantees for models trained on the resulting data. It proposes a simple and efficient distance-based algorithm—removing p1 samples that are furthest from the mean of p2—and proves it is quadratically more sample-efficient than random removal in low-divergence regimes. Experiments on synthetic, text, and image data validate the theory, showing that this selective approach can achieve strong unlearning effects with 15-82% fewer data deletions than full removal.

### Strengths
The paper's primary strength is its elegant formalization of domain unlearning as a distributional problem. By defining the objectives using KL divergence and characterizing the optimal trade-off via a Pareto frontier, the authors provide a rigorous, data-centric foundation for a problem that is often treated with ad-hoc heuristics. The connection established in Proposition 2 between the data-level KL objectives and the downstream model's expected log-loss is particularly powerful, as it provides a clear theoretical justification for why this data selection approach should work.

### Weaknesses
1. Novelty:
The core algorithm—removing points from one set based on their distance to the mean of another—is mechanically very simple. The novelty does not lie in a complex algorithmic contribution but rather in the application of this simple heuristic to the unlearning problem and, most importantly, the new theoretical framework that justifies it. While the framework is novel, the method itself could be seen as an application of standard outlier-detection or data-cleaning principles.
2. Experiments:
- Dependence on Feature Space: The effectiveness of the distance-based removal heuristics (e.g., MAHA-MU2, LR-COS) is entirely contingent on the quality of the embedding space. The paper uses standard features (e.g., from a pre-trained CNN), but provides no analysis of how the performance would change with a different, perhaps less-optimal, feature representation. If the feature space does not effectively separate the "statistically influential" samples, the distance metric may fail to identify them.
- Limited Baselines: The paper effectively demonstrates superiority over random removal and a simple coreset baseline. However, the coreset baseline (removing points closest to the p1 centroid) is shown to be ill-suited for this dual-distribution problem. A comparison with more sophisticated data selection methods, even if from different domains (e.g., active learning), could have provided a broader context for the performance gains.
- Theory-Practice Gap: The strong, closed-form sample complexity results are derived under the assumption of Gaussian distributions. While the authors acknowledge this and show that the qualitative insights generalize, the quantitative gap between the predicted 82% savings on synthetic data and the 15-50% savings on real-world data is substantial. This highlights that the complexity of real data distributions significantly tempers the theoretical gains.
- Practical Usage: the authors could consider more general/modern settings like LLM SFT.
3. Additional Things:
- The Oracle of p1 and p2: The entire framework operates on the strong assumption that the forget (p1) and retain (p2) sets are already known and perfectly separated. In practice, identifying every sample belonging to an unwanted domain (e.g., "toxic language") is an extremely challenging and often ambiguous task in itself. The paper solves the important second step of this process (which subset to remove) but does not address the first, which is a major practical caveat.
- Simplicity of "Statistical Influence": The method operationalizes "statistical influence" as distance from the retained data's mean. This is effective for unimodal, Gaussian-like distributions but may be less effective for multi-modal or complex distributions, where influence might be related to local density or boundary effects rather than just distance to a single central point.

### Questions
How sensitive are the results to the choice of the feature space? For the CIFAR-10 experiment, if you were to use features from a different model (e.g., a ViT instead of a CNN) or a model trained on a different dataset, would you expect the set of "most influential" cat images to remain largely the same?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper introduces the concept of distributional unlearning, a framework for studying the statistical properties of data removal in training datasets. The authors claim that may be convenitent to reduce not all the data of the forget set but a subset of them, the question they are trying to answer is what is the minimal subset of data points to remove (the forget set) in order to make the resulting data distribution far from the distribution of the forget set, while remaining close to the distribution of the retain set. 
To formalize this idea, the authors define the notion of $(\alpha, \varepsilon)$- distributional unlearning for a distribution $p$ with respect to two other distributions $p_1$ and $p_2$. A distribution $p$ satisfies $(\alpha, \varepsilon)$-distributional unlearning if it is at least $\alpha$-far from $p_1$ and at most $\varepsilon$-close to $p_2$, where "far" and "close" are measured in terms of the KL divergence between $p_1 or $p_2 and $p$. 
Most of the theoretical analysis in the paper focuses on Gaussian distributions, where the authors characterize the Pareto frontier of achievable $(\alpha, \varepsilon)$ values. They also prove a finite-sample guarantee for achieving $(\alpha, \varepsilon)$ distributional unlearning using two data deletion mechanisms: random removal, where a fixed number of samples are deleted uniformly at random; and relective removal, where data points are assigned scores proportional to their distance from the mean of the retain distribution, and high-scoring points are removed.
The authors compare the performance of these two algorithms on several datasets and also discuss the synergy of their approach with other unlearning methods.

### Strengths
- The paper is very well written, with clear exposition and excellent presentation. The theoretical claims are sound. 
- The introduction of distributional machine unlearning represents a novel and interesting contribution, offering a perspective on how data deletion can be studied from a statistical standpoint.
- The inclusion of coreset-based methods as an additional baseline is appropriate and well-motivated.

### Weaknesses
- The theoretical results are derived for data following two gaussian distributions (strong assumption) for two simple removal strategies. While these allow for closed-form analysis, they are not particularly practical for real-world unlearning. 
- The use of Kullback–Leibler (KL) divergence between data distributions as the central measure does not directly capture the notion of “forgetting” at the model level. KL divergence quantifies average differences between data distributions, not how much information a trained model retains about specific samples.
Therefore, KL-based analysis provides a bound on distributional robustness or generalization under deletion, rather than a guarantee of behavioral or algorithmic unlearning. In other words, a model may still encode information about deleted data even if the KL divergence between underlying distributions is small.
- It would be interesting to see 
- The paper is interesting from the thoeretical point of view and for the introduction of this problem. But is not clear how useful can be in practice since typically retraining from scratch is something one may want to avoid. 
- For the removal for instance of the class "cat" it is not shown how effective is this removal on samples that are 

Minor: 
- The setting considered in this paper is conceptually quite different from standard machine unlearning. The authors themselves acknowledge this distinction. Rather than proposing an algorithmic method to efficiently remove the influence of specific samples from a trained model, the work studies what happens statistically when data are deleted and the model is retrained from scratch.
Consequently, it is unclear whether the term unlearning is fully appropriate for this setting. The framework primarily measures how retraining after deletion affects model parameters and distributions, rather than addressing unlearning as a computational or algorithmic problem.
- The abstract highlights the derivation of the exact removal–preservation Pareto frontier for exponential families as a key contribution. However, this result appears only in the appendix and is not discussed or emphasized in the main text. Given its limited exposition and lack of integration into the core narrative, featuring it prominently in the abstract may be misleading or disproportionate relative to its role in the paper.

### Questions
- In practical machine learning settings, can the Kullback–Leibler (KL) divergence between two data distributions $p_1$ and $p_2$  remain small even when the samples removed from $p_1$  are rare but highly influential, such that the resulting model still retains significant information about those removed samples?
 I'm thinking for instance of removing a few minority-class examples can shift the decision boundary a lot or in linear regression about a few points can dominate parameter estimates, but contribute little to the overall KL.
- How is the synergy with other unlearning methods implemented in practice? For example, are the “selective samples” identified according to the procedure described in Section B.2.6 (so with the score only depending on their class and their embedding) and then used as the input (forget set) for the subsequent unlearning algorithm? (instead of performing retraining) 
- See also weaknesses

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper tackles the problem of machine unlearning, whcih traditionally focuses on removing entire domains or subpopulations of data as opposed to individual data points. The core of this paper is that a domain's statistical influence is often concentrated in small subsets of its samples. Thus, the authors propose distributional unlearning: a framework to select a small data subset by maximizing the statistical distance from the forget distribution while minimizing the distance to the retain distribution by using KL divergence. Experiments are conducted on synthetic, text, and image datasets.

### Strengths
- The paper does a good job of defining and motivating the proposed "distribution unlearning" framework. 
- The framework is built with a strong theoretical foundation:
  - Using KL divergence to define forget and retain objectives is simple yet effective.
  - The proposed selective removal algorithm is quite intuitive and is derived directly from the theoretical analysis. The authors also analyze theoretically how the selective removal algorithm is a strategy to maximizing the KL divergnce of the forget set while minimizing the KL divergence of the retain set. 
- Experimental evaluations, while not conducted on large-scale data, is well-structured and quite comprehensive.

### Weaknesses
- The framework is built on the assumption that the unwanted and retained data sets are already known. While the authors assert that this could be done via some upstream process (e.g., keyword filtering), there could be other scenarios where such techniques do not work. 
- I'm not entirely sure if the core motivation, that a domain's influence is concentrated in a small subset holds in the experimental results. For example, in CIFAR10 the removal is only observed after 50% deletion and in Jigsaw it is only observed after 80% deletion.

### Questions
I would appreciate if the authors addressed the few concerns raised in weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

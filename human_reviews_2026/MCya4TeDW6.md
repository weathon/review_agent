# Transformers Can Do Bayesian Clustering

- Decision: Reject
- Scores: 2, 2, 2, 4

## Abstract
Bayesian clustering accounts for uncertainty but is computationally demanding at scale. Furthermore, real-world datasets often contain missing values, and simple imputation ignores the associated uncertainty, resulting in suboptimal results. We present Cluster-PFN, a Transformer-based model that extends Prior-Data Fitted Networks (PFNs) to unsupervised Bayesian clustering. Trained entirely on synthetic datasets generated from a finite Gaussian Mixture Model (GMM) prior, Cluster-PFN learns to estimate the posterior distribution over both the number of clusters and the cluster assignments. Our method estimates the number of clusters more accurately than handcrafted model selection procedures such as AIC, BIC and Variational Inference (VI), and achieves clustering quality competitive with VI while being orders of magnitude faster. Cluster-PFN can be trained on complex priors that include missing data, outperforming imputation-based baselines on real-world genomic datasets, at high missingness. These results show that the Cluster-PFN can provide scalable and flexible Bayesian clustering.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors propose to use PFNs for Bayesian clustering. They introduce an augmented version of classical TabPFNs for the purpose of predicting the number of clusters and the cluster assignments in a given dataset. 
They evaluate their approach on the synthetic training data sampled from their prior, and a limited set of real-world datasets and benchmark against a variational inference (VI) approach, as well as "GMM" utilizing different information criteria. Several ablations, including one on relatively large datasets with 10 000 datapoints is conducted. 
Furthermore, a way to make the clustering robust to missing data is discussed and evaluated.

### Strengths
The paper is relatively well-written and follows a clear structure. While some important details on the baselines and datasets are missing, the most critical aspects of the experiments and the approach are made very clear. Them explicitly stating research questions helps the structure. 

To my best knowledge, using PFNs to predict cluster assignments **together with** the number of clusters has not been explored. The idea to make the model also robust to missing data is nice! 

In terms of experiments, the results they obtain look correct and reasonable and the large-dataset experiment (10,000 points) is interesting. 

The findings overall are somewhat interesting from a conceptual point of view or could form the foundation for very specific applications of PFNs to clustering where all other methods fail.

### Weaknesses
Unfortunately, there are several major weaknesses: 

1.) The paper completely ignores related work. Especially the paper "Reuter, Arik, et al. "Can Transformers Learn Full Bayesian Inference in Context?." Forty-second International Conference on Machine Learning." is highly relevant and seems to look into very related aspects of PFNs. In this paper, the authors also consider a PFN-approach to Bayesian clustering using GMMs and provide detailed experimental results. A thorough discussion of this paper is missing. Any other discussion of related work is also absent. 

2.) The paper massively overclaims the scalability of the approach in the abstract. Not only do the authors not even consider datasets with more than five features, but also don't include any results on reasonably-sized **real-world** datasets. It is also highly questionable that proposed PFN approach (which do in-context learning) has any conceptual advantage in terms of scalability compared to other methods. 

3.) Insufficient baselines: Just using one type of VI, that is not even properly explained or introduced, is clearly insufficient. The authors should explain and justify why they choose this particular type of VI and ensure that its hyperparamters are correctly set. Further VI baselines would also be needed for thorough experiments. Any sampling-based methods (MCMC) to perform inference are also missing. 

4.) Insufficient real-world experiments: Only one real-world dataset for the non-missing scenario is definitely not enough and makes the reader highly suspicious. 

5.) Insufficient number of tasks: It would be very interesting to see the Cluster PFN being trained on models other than GMMs to truly investigate Clustering as a problem and not just GMM clustering. 

5.) Details, including implementation details, on the datasets and all baselines are missing.

6.) Lacking Novelty: The approach is conceptually very similar to existing PFN approaches and investigates essentially the same task as Reuter et al.  

7.) The real-world applicability of this particular method is quite questionable. Why should anyone bother to fit a GMM with such an inefficient approach?

### Questions
Why do the authors believe that their method is scalable?

Why is the proposed PFN method conceptually a good approach for (Bayesian) clustering? 

What exactly is the type of VI that is used? 

Which hyperparameters are used for the VI method? 

Why don't the author's consider other VI methods? 

Why don't the authors consider sampling-based methods, in particular Hamiltonian Monte Carlo Samplers?

### Soundness
1

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces Cluster-PFN, a Transformer-based model for computing posterior cluster assignments.
The model is trained entirely on synthetic Gaussian mixture datasets to predict both the number of clusters and posterior cluster responsibilities in a single (or two-step) forward pass.

### Strengths
1) The idea of adapting PFNs to perform Bayesian clustering is interesting and original.

2) The code for reproducibility is available.

3) The authors also discuss the limitations of the proposed approach.

### Weaknesses
See *Questions*.

### Questions
1) The paper is very difficult to follow. Already from the abstract, key acronyms (AIC, BIC) appear before being defined. Throughout the text, the authors make strong but insufficiently supported claims (“The results are clear”, “Cluster-PFN approximates the true Bayesian posterior over the number of clusters”), yet the actual mechanism of the model remains opaque. After several readings, it is still unclear what Cluster-PFN concretely does and how it differs from the seminal work of Muller et al. In practice, the model appears to be trained via supervised meta-learning on synthetic GMMs and to only imitate posterior-like outputs, without estimating latent parameters or uncertainty over model parameters. **The work needs substantial rewriting**.

2) The presentation of results is confusing, with synthetic and real-data experiments mixed together and no clear boundary between what is meant to demonstrate “proof of concept” and what supports the claimed generalization ability. Most importantly, the Discussion section exposes a conceptual contradiction. The entire point of PFN-like and meta-learning models is to perform meta-training on synthetic tasks so that the learned model can generalize to a wide variety of real-world datasets. However, the authors themselves acknowledge that their approach must be trained “for a particular prior” and that “Cluster-PFN is not always competitive on real-world data, only offering clear benefits on the GLS1 dataset.” This statement undermines the main motivation of the work: if the model needs to be retrained for each prior and fails to generalize across domains, it is unclear what advantage Cluster-PFN offers over standard inference methods.

3) The model formulation is difficult to follow. It is not clear how the conditioning on the number of clusters $k$ is implemented in practice, and why two forward passes are required when $k=0$. This seems to contradict the usual “single forward-pass inference” property of PFNs. 
- Could the authors clarify how these two stages (estimating $k$ and computing cluster responsibilities) interact in the final inference pipeline?
- In addition, the model breaks label permutation invariance by assigning cluster labels through a fixed heuristic (“the cluster closest to the origin is label 0”). How much does this arbitrary rule influence the learned mapping?

### Soundness
1

### Presentation
1

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
The paper presents a modification of Prior-Data Fitted Networks (PFNs) to implicitly perform Bayesian clustering. The model is trained on synthetic datasets generated from Gaussian Mixture Models (GMMs) with specified priors, using the corresponding cluster assignments as supervision for a Transformer. To support inference of the number of clusters, the model is also provided with special tokens that guide the prediction of cluster counts. Empirical evaluation focuses primarily on synthetic data generated from GMMs with dimensionality up to five, with a limited number of real-world datasets included for additional validation.

### Strengths
- The paper reads well overall, making it easy for the reader to follow the core ideas.

- To the best of my knowledge, this is the first work to apply Prior-Data Fitted Networks (PFNs) to clustering, opening a promising new direction for amortized Bayesian inference in unsupervised learning.

### Weaknesses
- **Limited methodological novelty**: the approach primarily adapts existing PFNs by treating known cluster assignments as supervised labels, with only minor modifications to the training procedure. This incremental extension reduces the overall contribution, and in my opinion, could only be mitigated if highly significant results were provided, which is not the case.

- **Lack of motivation**: while the abstract emphasizes *"missingness"* as a central motivation, this aspect is not meaningfully developed in the main text. It is only briefly addressed in the experiments through a simple masking setup, making it feel peripheral and disconnected from the paper's core contributions.

- **Overly simplistic datasets**: unless supported by references to recent work, the use of five-dimensional synthetic data generated from Gaussian mixtures appears insufficient for evaluating clustering performance. This setting does not reflect the complexity of modern clustering tasks, which often involve high-dimensional, noisy, and structurally diverse data.

- **Weak baselines**: similarly, the baseline methods used (e.g., GMM and variational inference) are outdated and overly simplistic, limiting the relevance of the empirical comparisons to current challenges in clustering.

- **Weak empirical evaluation**: the evaluation relies heavily on qualitative visualizations using simple, easily separable datasets. This is insufficient for demonstrating the robustness or scalability of the approach. 

- **Unjustified computational cost**: while inference is reported to be ~50× faster, the training cost is cited as 60 GPU hours for clustering on five-dimensional Gaussian blobs, which seems unreasonable given the simplicity of the task, and reduces the practicality of the method.

- **Lack of experimental transparency**: key details are missing. I include some of them in the questions section.

### Minor Issues

- The methodological section (Section 3) is too brief and lacks details. Several important methodological details are deferred to Section 4, which is nominally focused on experiments. This separation impairs the paper's readability and logical flow.

- The use of the term *"Bayesian prior"* to describe data generated from a GMM does not align with the conventional meaning of priors in Bayesian inference—specifically, priors over model parameters or cluster assignments. This creates confusion.

- The Old Faithful dataset is not mentioned in the real-world dataset descriptions, yet it appears in the experimental results. This inconsistency should be addressed.

### Questions
- How is the maximum number of generated clusters $𝐾$ defined?

- How does the $\beta$ parameter produce cluster overlapping?

- What are the hyperparameters and implementation details for the baselines?

- The real-world datasets are poorly described, limiting reproducibility and interpretability.

### Soundness
1

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
The authors extend Prior‑Data Fitted Networks (PFNs) from supervised prediction to *Bayesian clustering*. Their **Cluster‑PFN** is a Transformer that, given a set of points (X) and an input (k), outputs (i) a distribution over the *number of clusters* (P(k \mid X)) (when fed (k=0)) and (ii) *responsibilities* (p(z_i=k\mid X,k)) for each point when a cluster count is provided. A special “collector” token (\rho) aggregates information to predict (P(k\mid X)) (Figure 2, p.3), and the model conditions on (k) via an embedding added to all tokens. 
The model is trained entirely on synthetic datasets drawn from a *finite GMM* with Normal–Inverse‑Wishart priors; versions include 2D and up to 5D inputs as well as *random missingness* up to 80%. Architecture uses a 4‑layer, 4‑head encoder with 256‑dim embeddings (Appendix C).

### Strengths
Demonstrates that a single Transformer forward pass—trained purely on synthetic prior‑samples—can approximate *both* (P(k\mid X)) and responsibilities, a compelling extension of PFNs into unsupervised Bayesian modeling. The special (\rho) token for (k) prediction (Fig. 2) and conditioning mechanism are simple but elegant.  Clear runtime wins vs VI, even when VI uses multiple inits (Table 2), and scaling tests up to 20k points show consistent advantages (times reported on p.8).

### Weaknesses
For AIC/BIC/silhouette, the search over (k) *excludes (k=1)* “since the silhouette score is undefined for a single cluster” (p.5). But AIC and BIC are perfectly well‑defined at (k=1). Excluding (k=1) likely *penalizes* AIC/BIC whenever the truth is one cluster, inflating Cluster‑PFN’s relative accuracy in Table 1. A fair protocol would allow (k\in{1,\ldots,K}) for AIC/BIC and handle silhouette separately. 

The paper argues Cluster‑PFN “approximates the true Bayesian posterior over the number of clusters” by analogy with supervised PFNs, but *responsibilities are learned independently* , not via a coherent joint posterior over ((\theta,z)). Moreover, when (k) is unknown, the method ultimately uses a *two‑pass MAP (k)* rather than integrating over (k) (the fully Bayesian option the model initially formulates), because of label‑ordering bias. This iweakens the Bayesian claim for responsibilities. 

 The model sometimes fails to obey a user‑specified (k), especially when the instruction is “wildly different” from the data’s structure; the authors quantify this and show accuracy improves when conditioning near the unconditioned prediction. For downstream pipelines that *require* exactly (k) clusters, this is a limitation. 

 The deterministic relabeling uses distance to the origin after zero‑one scaling. While consistent across training tasks, it encodes an *arbitrary geometry* (e.g., clusters near (\mathbf{0}) get low indices), is sensitive to min–max scaling quirks, and is not feature‑permutation invariant—an issue the authors also list as a limitation (Section 7). More principled label alignment (e.g., Hungarian matching to learned prototypes) or an equivariant architecture would be cleaner. 

The paper itself notes that on real datasets without severe missingness, Cluster‑PFN is “not always competitive,” with clear wins mainly on GLS1 and under high missingness (Section 6). This underscores *prior mismatch*: training solely on finite‑GMM priors may not capture real data complexity (also discussed on p.9).  Further,  experiments cap at 5D and produce 10 logits (Appendix C). This raises questions about behavior in high‑D tabular domains (common in genomics) and for larger (K). The paper mentions feature‑permutation invariance and scaling to higher (d) as future work (Section 7). 

Transformer’s (O(N^2)) attention remains; so  very large (N) regimes may still be challenging without sparse/linear attention variants. 

The paper reports NLL of responsibilities (with label‑permutation minimization) and shows a histogram (Fig. 5c–d), but does not assess calibration of (P(k\mid X)) or of per‑point responsibilities (e.g., reliability diagrams, ECE). This matters if outputs are to be trusted as Bayesian probabilities.

### Questions
Fix the (k=1) baseline issue  Re‑run AIC/BIC with (k\in{1,\ldots,K}); report per‑(k) accuracy and overall accuracy marginalizing (k\sim U(1,K)). 

Compare against EM/VI that marginalize missing features directly for GMMs, not only imputation‑based pipelines. 

Reliability diagrams and proper scoring (e.g., Brier) for (P(k\mid X)); temperature scaling if needed. Include calibration for responsibilities. 

 Train on broader priors (non‑Gaussian, skewed/cluster‑size imbalance, heavy‑tailed, anisotropic covariances) and evaluate on real data; quantify sensitivity to prior misspecification (Section 6 hints at this). 

 Demonstrate feature‑permutation‑equivariant variants (as suggested in Section 7) and report how accuracy and runtime change.

### Soundness
2

### Presentation
3

### Contribution
3

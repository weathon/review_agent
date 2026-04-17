# Sparse Feature Routing for Tabular Learning

- Decision: Reject
- Scores: 2, 2, 6, 2

## Abstract
The landscape of high-performance tabular learning is often framed as a choice between the efficiency of gradient-boosted trees and the performance of deep architectures, which increasingly rely on heavy, monolithic backbones to model feature interactions. We argue that this monolithic design overlooks a critical inductive bias: the inherent sparsity and modularity of tabular data. To address this, we introduce the Sparse Feature Routing Network (SFR Net), an architecture that decomposes computation into independent feature experts controlled by an entropy-regularized router, coupled with a low-rank module to capture non-additive dependencies. We evaluate SFR Net across 14 heterogeneous benchmarks, including standard datasets, high-dimensional multiclass tasks, and regression problems. Empirically, SFR Net demonstrates predictive performance competitive with, and often superior to, state-of-the-art deep tabular models and gradient-boosted ensembles. Beyond raw performance, SFR Net offers three distinct structural advantages: (1) efficiency, requiring up to $24\times$ fewer parameters and training $30\times$ faster than tabular Transformers; (2) intrinsic sparsity, dynamically activating only a small fraction of features per instance; and (3) faithful interpretability, where deletion tests confirm that the learned routing weights serve as reliable, causal instance-level attributions. These results position sparse feature routing as a lightweight, transparent, and high-performance alternative to dense tabular foundation models.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes Sparse Feature Routing (SFR Net): per-feature “experts” (tiny MLPs / embeddings) produce vectors that are softmax-gated by an “instance-wise sparse router” with an entropy penalty; a low-rank interaction head (factorization-style) models pairwise effects; a final MLP predicts the target. On four datasets (two classification: AD, JA; two regression: HE, CA) SFR Net reportedly outperforms several deep tabular baselines (MLP, DCNv2, AutoInt, FT-Transformer) and is competitive with XGBoost/CatBoost; ablations on Adult attribute gains to the feature-wise decomposition, routing, and the entropy regularizer.

### Strengths
•	Clear decomposition (feature-wise experts + mixer) with simple end-to-end training.  
	•	Readable method section and schematic; ablation table on Adult is helpful.  
	•	Attempts to provide per-instance attributions via router weights

### Weaknesses
The design closely parallels NAMs/GA2M (feature-wise functions) plus factorization-style interaction (FM/DCNv2) and feature selection ideas from TabNet / conditional computation. The router uses standard softmax + entropy (not sparsemax/top-k gating), so claimed “sparsity” is largely peaky-but-dense attention. The paper positions this as a novel inductive bias, but the components and motivation feel incremental.  

Results rely on four datasets; yet the abstract and discussion claim “across diverse benchmarks” and competitiveness with strong SSL methods. Modern, decisive baselines for tabular learning—TabM, TabPFN(v2), TabICL, CatBoost with careful HPO—are absent (only FT-Transformer/TabNet/etc. are included). With such limited scope, the paper cannot substantiate “transparent, efficient, and powerful foundation” claims.  

Using entropy-regularized softmax does not yield true zeros, and attention weights are not guaranteed faithful attributions under feature correlation. There’s no faithfulness check (remove top-k features and measure Δ performance), stability across seeds, or agreement with perturbation/SHAP. Claims of “native interpretability” therefore remain speculative.  

The paper asserts CPU-friendliness and linear scaling in feature count, but provides no runtime, FLOPs, or energy comparisons vs baselines (including tabular FMs/DCNv2/GBDTs). Ablations mention fewer epochs on Adult, but wall-clock comparisons are absent.  

Critical knobs—entropy weight λ, rank K, expert width/depth, and router temperature—lack robustness sweeps. The “sparsity” benefit over dense routing is tiny in Table 3; without broader ablations, it’s unclear that entropy-sparsification is consistently helpful.  

How ResNet backbones are adapted to tables (reshaping? tokenization?) isn’t specified; the comparison risks being apples-to-oranges and distracts from missing state-of-the-art tabular baselines.

### Questions
Replace softmax+entropy with sparsemax/entmax or top-k gating and compare—does true sparsity help?  
Provide faithfulness tests (deletion/keeping-k, knockoffs) and stability checks for router attributions.  
Add modern baselines (TabM, TabPFN/TabICL) and a larger benchmark (OpenMLCC18 or comparable), with paired tests and critical-difference diagrams.  
Report compute (GPU/CPU time, params, memory) vs DCNv2/FT-Transformer/GBDTs; include cost-normalized leaderboards.  
Run robustness to irrelevant/correlated/noisy features and missing-value handling; test scaling to high-dimensional tables.

### Soundness
2

### Presentation
2

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
The paper introduces a new tabular deep learning model, SFR-Net, which employs feature-wise independent expert networks and uses instance-dependent routing to combine feature-wise representations. The proposed approach is compared with three deep learning baselines across four datasets, and the authors claim superior performance.

### Strengths
* Interesting idea on MoE-style model for tabular data
* Important ablations in Section 4
* Overall, paper is clearly written

### Weaknesses
See questions for details. 

Summary of weaknesses:
* Limited number of datasets and baselines
* No details on experimental setup
* Missing comparison with NAM and numerical embeddings
* Poor empirical performance

### Questions
My main concern is that the results are not convincing. Full list of questions:

1. Modern tabular DL papers typically use benchmarks with dozens of datasets, whereas this paper uses only four but still claims a “comprehensive empirical study” (L19). [1, 2, 3]
2. What was the motivation for choosing the baselines? All selected models are relatively old, while many modern and stronger DL baselines exist. [1, 2, 3]
3. What is the training protocol? How HPO was performed? What values of hyperparameters are used?
4. The paper mentions NAM in the related work but does not compare against it. I believe this is an essential baseline since your method is closely related to NAM.
5. Embeddings for numerical features [4] are also very relevant since numerical embeddings essentially are feature-wise independent neural networks but they are concatenated in one big representation. While NAMs are more closely related to your approach, incorporating this method might improve your results.
6. An ablation study analyzing the benefits of higher-order interactions would be insightful.
7. Analysis on weights $\alpha$ would help reveal whether there is any sparsity.. Additionally, it is unclear why the method is referred to as “sparse.” The entropy loss on $\alpha$ does not necessarily imply that many weights will be sparse.
8. The authors claim effective training and inference (L74), but no supporting experiments or results are provided. On large datasets, a standard MLP is likely to be more efficient.
9. The ablations in Section 4.3 are valuable but are conducted on a single dataset, which may make the conclusions data-dependent.
10. Authors provide a performance of GBDTs "for reference" but modern DL architectures generally outperform  GBDTs.
11. Please, explain motivation for comparing with SSL methods while there is no comparison with strong tabular DL architectures?


[1]: Better by Default: Strong Pre-Tuned MLPs and Boosted Trees on Tabular Data. David Holzmüller, Léo Grinsztajn, Ingo Steinwart. 2024.  
[2]: TabM: Advancing tabular deep learning with parameter-efficient ensembling. Yury Gorishniy, Akim Kotelnikov, Artem Babenko . 2025.  
[3]: Accurate predictions on small data with a tabular foundation model. Hollmann et al. 2025.  
[4]: On Embeddings for Numerical Features in Tabular Deep Learning. Yury Gorishniy, Ivan Rubachev, Artem Babenko. 2022.

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes **SFR Net**, a decomposed architecture for tabular learning that routes each instance over **independent, per-feature experts** via a **sparse router**, then mixes selected features with a **low-rank interaction module**. 

Concretely, each feature $x_j$ is processed by its expert $E_j$ to produce $h_j \in \mathbb{R}^D$. A shared scoring MLP outputs scores $s_j$; the router produces instance-wise weights

$$
\alpha_j=\frac{\exp(s_j)}{\sum_{k=1}^{F}\exp(s_k)},\qquad
H(\alpha)=-\sum_{j=1}^{F}\alpha_j\log\alpha_j,
$$

and adds an entropy penalty $\lambda,H(\alpha)$ to encourage sparsity. First-order effects are aggregated as

$$
r^{(1)}=\sum_{j=1}^{F}\alpha_j,h_j,
$$
while higher-order interactions use shared low-rank projections $W_K,W_V\in\mathbb{R}^{D\times K}$ with $K\ll D$:
$$
k_j=h_j^\top W_K,\quad v_j=h_j^\top W_V,\quad
r^{(2)}=\sum_{j=1}^{F}\alpha_j,(k_j \odot v_j),\quad
r_{\mathrm{final}}=[,r^{(1)}; r^{(2)},].
$$

A final MLP maps $r_{\mathrm{final}}$ to predictions, trained with the task loss plus $\lambda,H(\alpha)$. The design aims to yield **native instance-level attributions** (the router’s $\alpha$) and computational efficiency. Experiments on several benchmarks indicate that SFR Net outperforms strong Transformer-based tabular baselines and is competitive with GBDTs; it also compares favorably to self-supervised pretraining approaches despite **no pretraining**.

### Strengths
* **Principled decomposition:** One expert per feature + instance-wise sparse routing provides a transparent, task-aligned inductive bias for tabular heterogeneity.
* **Low-rank interaction head:** Captures higher-order effects efficiently under a tunable rank budget $K$.
* **Native attributions:** Router weights $\alpha$ offer per-instance explanations without post-hoc methods.
* **Competitive results without SSL:** Outperforms strong neural baselines and remains competitive with GBDTs; compares well to SSL backbones **without** pretraining cost.
* **Ablations:** Sensible ablations indicate gains stem from decomposition, routing, and sparsity rather than brute-force capacity.

### Weaknesses
* **Dataset breadth:** The evaluation spans only a small number of datasets; lacks a larger, standardized suite (e.g., 10–20 public tabular benchmarks) with **average-rank** analyses and significance tests.
* **Sparsity mechanism:** Entropy regularization yields soft sparsity; comparisons to **hard top-$k$** or **entmax/sparsemax** would clarify sparsity–accuracy–efficiency trade-offs. Reporting the **average selected feature count** would help.
* **Complexity accounting:** No explicit wall-clock or memory comparisons vs. FT-Transformer/GBDTs/SSL; empirical scaling in $F,D,K$ is not quantified.
* **Interaction coverage:** The low-rank mixer may undercapture very high-order/non-linear dependencies unless $K$ grows; guidance on choosing $K$ is limited.
* **Robustness:** Systematic evaluations under missingness, extreme categorical cardinality, and distribution shift are not reported.

### Questions
1. **Router sparsity:** What is the **average number of selected features per instance** as a function of $\lambda$? Have you tried **hard top-$k$** or **entmax/sparsemax** routing, and how do accuracy/efficiency/attributions change?
2. **Complexity & scaling:** Please report wall-clock (train/infer) and peak memory vs. $F,D,K$, and provide Pareto curves (accuracy vs. time/memory) against FT-Transformer, GBDTs, and SSL baselines.
3. **Low-rank sensitivity:** How sensitive are results to $K$? On heavily interacting datasets, does increasing $K$ help, or would an additional cross-network/FM-style term improve performance?
4. **Attribution fidelity:** Do router weights correlate with SHAP/Integrated Gradients? Any randomization or sanity checks to validate attribution robustness?
5. **Robustness:** How does the router behave under **missing values**, extreme categorical cardinality, and covariate/label shift? Does sparsity sharpen or degrade under noise?
6. **Benchmark breadth:** Can you expand to a larger public suite (OpenML/UCI) and report **average ranks** and **statistical tests** to bolster generality?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes a novel neural architecture for supervised learning on tabular data, SFR Net, which consists of a per-feature component and a low-rank component. The paper compares the proposed architecture against some other supervised deep learning architectures, self-supervised tabular models, and standard tree-based models on four datasets, and finds that SFR Net outperforms the supervised models and ranks among the top unsupervised models.

### Strengths
- The paper addresses a classical problem of deep supervised learning on tabular data.
- The paper proposes a relatively simple architecture with an eye on interpretability and understandability of the architecture and model.

### Weaknesses
- Proposing a new architecture for a common tasks rests on the empirical evaluation of the model. Using four datasets is insufficient. Please use the TabArena benchmark, which consists of 51 datasets, or at least another recent benchmark suite like CC-18/CTR, TabZilla or Talent.

- The paper does not compare against any recent deep architectures such as TabM or RealMLP, and completely disregards current state-of-the-art foundational models such as TabPFN V2, TabICL, LimiX and TabDPT. Taking into account these models, many of the claims in the introduction are false, such as claiming "ill-suited, monolithic backbones". This seems not appropriate for any of these foundational models. In particular these use per-feature embeddings (all foundational tabular models except for TabPFN V1 do afaik).

- The appeal to interpretability of the models is interesting and a good motivation, but there is no experiments on interpretability, and it's unclear how the low-rank components could be interpreted.

### Questions
- How is the feature-wise experts different from a NAM?

### Soundness
2

### Presentation
3

### Contribution
2

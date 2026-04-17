---
job_id: 2e2b56d8-7659-473c-b3d9-c85949672f49
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: ecbEfXdqrV.pdf
paper: Why Is the Counterintuitive Phenomenon of Likelihood Rare in Tabular Anomaly Detection with Deep Generative Models?
main_score_norm: 0.4
desk_reject: false
---
# Desk Rejection Assessment:

## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The work is on likelihood-based anomaly detection with normalizing flows, theoretical analysis of likelihood paradoxes, and large-scale benchmarking in tabular data, which are squarely within ICLR’s scope (representation learning, generative models, OOD / anomaly detection).

## Minimum Quality
Pass ✅.  
The paper is in English and has all major sections (Abstract, Introduction, Related Work, formal “Definition of Counterintuitive Phenomenon”, Experiments, Results/Analysis, and Conclusion). It provides substantial experiments and nontrivial theoretical derivations. While I see issues in methodology and theory (detailed below), they are not at the level of a desk rejection.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I did not detect any hidden prompts or explicit attempts to manipulate automated reviewing systems in the provided content.

---

# Expected Review Outcome:

## Summary

The paper investigates why the well-known “counterintuitive likelihood” phenomenon of deep generative models in image OOD detection (anomalies receiving higher likelihood than in-distribution data) appears to be rare in tabular anomaly detection.  

First, the authors propose a domain-agnostic, performance-based definition of the “counterintuitive phenomenon” (Definition 3.3) using AUROC comparisons between a likelihood-only normalizing flow model (NF-SLT) and a set of baseline anomaly detectors. Then, using 47 tabular datasets and 10 CV/NLP embedding datasets from ADBench with 12 baselines, they show that NF-SLT rarely exhibits this phenomenon and in fact performs strongly. Finally, they provide a theoretical analysis linking dimensionality and entropy differences to the likelihood gap and AUROC (Theorem 5.4, Corollary 5.6) and an empirical study of feature correlation via intrinsic dimension estimates (Figure 1, Table 4) to argue why tabular data are less susceptible to the counterintuitive behavior than images.

## Strengths

1. **Extensive empirical study with strong flow performance**  
   - The core empirical result is clear: NF-SLT with NICE performs very well across a large set of tabular datasets. In **Table 1 (top)** on 47 tabular datasets, NF-SLT has the best average AUROC (0.8575), best AUPRC, the best average rank (3.43), the highest Top2 ratio (0.45), and the lowest fail ratio (0.02). This is a strong and practically relevant finding, especially given that several deep tabular AD models (ICL, MCM, NeuTraLAD) were designed specifically for this domain.  
   - The bottom part of **Table 1** shows that NF-SLT is also competitive on CV/NLP embedding datasets, outperforming other deep models on 9 of 10 datasets and only slightly worse on “imdb”. This supports the claim that likelihood test issues observed on raw images are mitigated on lower-dimensional embedding spaces.

2. **Clear experimental coverage and relatively careful benchmarking**  
   - Using *all* 47 tabular and 10 embedding datasets from ADBench is valuable: it reduces the risk of cherry-picked datasets and aligns with the selection-bias concerns raised by Shwartz-Ziv & Armon.  
   - The experimental section contains many sanity checks:  
     - Different flow architectures (NICE vs RealNVP) in **Table 13**,  
     - Hyperparameter sensitivity study in **Table 12**,  
     - Different test statistics (simple likelihood vs typicality test) in **Table 14**,  
     - Performance on different anomaly types (local, global, clustered, dependency) in **Table 8**,  
     - Categorical-heavy datasets (Table 9).  
   This makes the empirical story reasonably robust.

3. **Attempt to formalize “counterintuitive phenomenon” rather than relying on anecdotes**  
   - Definition 3.3 (Equations (2) and (3)) provides a quantitative criterion: a counterintuitive phenomenon occurs only if (i) a high fraction of comparison models outperform the generative model and (ii) the worst gap among outperforming methods is at least γ.  
   - This helps distinguish genuine “paradoxical” behavior from simple dataset difficulty or small performance gaps. The CIFAR-10 vs SVHN example is used as a motivating case that clearly meets this definition.

4. **Nontrivial theoretical analysis about dimension and likelihood gap**  
   - Equation (4) restates the entropic decomposition of the expected log-likelihood gap from Caterini & Loaiza-Ganem, and Theorem 5.4 then extends this to show that under independence and an entropy–KL condition, the lower bound of the likelihood gap becomes more negative linearly with dimension.  
   - Corollary 5.6 further links the upper bound of AUROC to dimension, under assumptions on the central moments of the log-likelihood difference. While somewhat restrictive, this is an interesting and technically involved attempt to connect high-dimensional geometry with practical detection performance.
   - The synthetic Gaussian experiments using NICE and RealNVP in **Figures 2 and 3** plus the log-likelihood, latent norm, and volume histograms in **Figures 4–11** (Appendix C) do a good job illustrating that as dimension increases, the AUROC of NF-SLT decays toward 0.5 and the log-likelihood distributions of normal and anomaly samples become almost indistinguishable. These figures provide concrete visual support for the high-dimensional analysis.

5. **Feature correlation / intrinsic dimension analysis offers an interesting angle**  
   - The “feature correlation perspective” in Section 5.2 is conceptually appealing: images have highly correlated homogeneous pixels, whereas tabular data tend to have more heterogeneous and less correlated features.  
   - The toy AR(1) Gaussian example with covariance matrix (Equation (5)) and the resulting intrinsic dimension trends in **Figure 1 (left and center)** convincingly demonstrate that increasing correlation (larger ρ) reduces estimated ID relative to ambient dimension.  
   - **Figure 1 (right)** and **Table 4 (top)** show that MNIST/CIFAR/SVHN have extremely small \(d\)-Ratios (~0.2–1%) while tabular datasets lie much closer to the diagonal (ID ≈ ambient dimension). This empirical gap nicely illustrates the proposed correlation difference between domains.

6. **Dimensionality experiments on images support the theory qualitatively**  
   - **Table 2** (ICA + RealNVP) and **Tables 5–6** (PCA/ICA + RealNVP) show that for many image OOD pairs with \(\mathbb{H}(P) > \mathbb{H}(Q)\), AUROC improves as dimension is reduced (e.g., CIFAR-10 vs SVHN improves from ~0.08–0.23 at 1024 components to ~0.31–0.47 at 30 components).  
   - **Table 3** (Glow on resized images) further illustrates that simple downscaling can drastically change AUROC, even flipping it above 0.5 in some settings. While not strictly under the independence assumptions of Theorem 5.4, these figures provide intuitive support for the claim that dimensionality magnifies the likelihood inversion issue.

7. **Reproducibility and thorough appendices**  
   - The paper provides detailed hyperparameter search spaces (**Table 10**), selected optimal configurations (**Table 11**), and per-dataset results (**Tables 16–21**), along with implementation sources and training details for all baselines.  
   - The hyperparameter sensitivity comparison in **Table 12** is particularly helpful: NF-SLT’s AUROC changes the least (0.0116) when moving from per-dataset tuning to a single global configuration, suggesting that its strong performance is not overly dependent on fine-tuning.

## Weaknesses

1. **Definition of the counterintuitive phenomenon is somewhat arbitrary and strongly tied to baselines**  
   - Definition 3.3 (Equations (2) and (3)) defines the phenomenon via thresholds β and γ on *relative AUROC to other models*. However, the paper never specifies concrete values of β and γ in the main text, nor shows sensitivity analyses for different threshold choices. Without this, the reader cannot assess how robust the claim “the phenomenon is rare in tabular data” is to the parameters of the definition.  
   - More fundamentally, basing the definition on whether “most comparison models outperform the generative model” (Assumptions 3.1 and 3.2) conflates the paradoxical likelihood behavior with the *relative competitiveness* of the generative model compared to whatever baselines are chosen and tuned. In particular:
     - If the baseline set changes (e.g., includes stronger tabular-specific methods or more flows) or is tuned differently, the same NF behavior could or could not be declared “counterintuitive”.  
     - The counterintuitive phenomenon in the image literature is inherently about the *ordering of likelihoods of P vs Q*, not about outperforming a particular suite of baselines. By design, Definition 3.3 cannot detect cases where likelihood inversion occurs but all methods do poorly (e.g., AUROC near 0.55 for all models).
   - This baseline-relative definition is a reasonable engineering proxy, but the paper sometimes uses it as if it were an intrinsic, domain-agnostic notion, which I find overstated.

2. **Theoretical analysis relies on restrictive independence and entropy conditions, with a loose connection to real tabular data**  
   - The central theoretical results (Theorem 5.4, Corollaries 5.5 and 5.6) assume that \(P = \prod_i p_i(x_i)\) and \(Q = \prod_i q_i(x_i)\), i.e., dimensions are independent, and further require conditions like \(\mathbb{H}(P) - \mathbb{H}(Q) > D_{KL}(Q\|P)\) and scaling of central moments as \(\mathcal{O}(d^k)\).  
   - These assumptions are quite strong. Real tabular datasets have nontrivial dependencies, and Section 5.2 itself is about feature correlation. The paper essentially analyzes the effect of dimension under *no correlation* (product measures), and then separately argues about correlation via intrinsic dimension, but the two are not mathematically integrated.  
   - Moreover, conditions like \(\mathbb{H}(P)-\mathbb{H}(Q) > D_{KL}(Q\|P)\) and the per-dimension version in Proposition 5.3 are not checkable on real data. The arguments about CIFAR-10 vs SVHN relying on these inequalities are therefore heuristic.  
   - In **Equation (17)** and the subsequent derivation of Theorem 5.4, the result is an approximate linear dependence on \(d\), but the paper jumps from this to fairly strong qualitative claims (“tabular data can be considered more advantageous … because they are less exposed to the problems that arise in high dimensions”) without quantifying how realistic these inequalities are in practice.

3. **Feature correlation / intrinsic dimension argument is insightful but empirically weakly supported and somewhat circular**  
   - Section 5.2 argues that high feature correlation reduces intrinsic dimension, and that images have far lower \(d\)-Ratios than tabular data. **Figure 1 (right)** and **Table 4 (top)** indeed show that the estimated ID of MNIST/CIFAR/SVHN is around 10–26 vs ambient dimension 784–3072, while tabular datasets are closer to the diagonal.  
   - However, the connection “lower \(d\)-Ratio ⇒ stronger correlation ⇒ more likelihood inversion” is not rigorously demonstrated. The AR(1) Gaussian toy experiment (Equation (5) and **Figure 1 left/center**) confirms that for that *one* parametric family, ID estimates decrease with ρ, but this does not establish a general one-to-one mapping between ID and correlation strength in arbitrary distributions.  
   - The paper itself notes that TwoNN and MLE ID estimators tend to underestimate the ID when it is large; this makes comparisons based on the exact ratios quantitatively unreliable. No confidence intervals or variance estimates for the ID estimators are provided.  
   - The conclusion that “tabular domain generally has low feature correlation” is therefore somewhat speculative, especially given that some real tabular datasets (e.g., genomics) are acknowledged in Appendix C.4 as highly correlated and high-dimensional.

4. **Limited empirical probing of dimensionality and correlation within tabular datasets**  
   - The dimensionality studies that most directly support the theory are conducted on image data (Tables 2, 3, 5, 6) and synthetic Gaussians (**Figures 2–3**). For actual tabular data, the only explicit dimensionality manipulation I see is **Table 7**, where PCA-based reduction on InternetAds improves NF-SLT’s AUROC modestly while hurting MCM’s.  
   - There is no systematic investigation such as: taking multiple tabular datasets, artificially increasing dimensionality (e.g., via adding noisy features) or creating synthetic tabular-like data with controllable correlations, then measuring how often the phenomenon occurs and how AUROC degrades.  
   - Thus, while the theory suggests that higher dimension and stronger correlation should induce more likelihood inversion, the evidence that “typical” tabular data are actually in the safe regime is mostly indirect. It hinges heavily on ADBench having relatively modest dimensions and moderate correlations, but this is not quantified beyond ID estimates.

5. **Experimental protocol raises questions about fairness and dependence on labeled test data**  
   - In Section 4 (“Evaluation”), the authors state:  
     > “For each dataset, after experimenting with all combinations in the hyperparameter searching space with 10 repeated experiments, the hyperparameter combination with the highest average AUROC for all datasets is selected as the representative hyperparameter combination…”  
     This implies that ground-truth anomaly labels on the *test* splits across all datasets are used to select a single global hyperparameter configuration per model. While this is not as egregious as per-dataset tuning on test data, it still violates strict unsupervised evaluation since test labels influence model selection.  
   - Appendix I reports a per-dataset optimal hyperparameter scenario (**Table 15**), which further uses AUROC labels to tune per dataset. The main narrative relies on the global configuration, but both settings mix training and evaluation roles of the test labels.  
   - Classical baselines like COPOD and ECOD are hyperparameter-free; PCA / LOF / IF / OCSVM have relatively small search spaces compared to deep models. The paper does not discuss how comparable the search efforts are between NF-SLT and tabular deep baselines (MCM, ICL, NeuTraLAD), especially given the large number of datasets. This could bias in favor of more heavily tuned methods.

6. **Mathematical clarity issues and minor inconsistencies**  
   - In Equations (2)–(3), the hyperparameters β and γ are introduced but never instantiated; Definition B.1 in Appendix B contains Equation (7) with a typographical issue: `\mathbbm{1}\sum_{i=1}^{k}... > \beta`, which appears to be missing normalization or parentheses (probably intended to be \(\frac{1}{k}\sum ... > \beta\)).  
   - In Theorem 5.4 and its proof (Appendix D), the notation occasionally switches between \(P\), \(P_\theta\), \(p\), and \(p_\theta\) somewhat sloppily, which makes it harder to track the exact assumptions of the convergence \(p_\theta \to p\).  
   - In Corollary 5.6, the assumption that the \(n\)-th central absolute moment of \(\log p_\theta(Y)-\log p_\theta(X)\) scales as \(\mathcal{O}(d^k)\) with \(k<n\) is justified only briefly at the end of Appendix D via a Gaussian example. For non-Gaussian flows or heavy-tailed anomalies, this scaling may fail, yet the discussion in the main text treats the corollary as broadly applicable.

7. **Missing or under-emphasized related work on flows for anomaly detection and tabular flows**  
   - Section 2.1 “Normalizing Flow” is largely generic and does not discuss prior works that specifically use normalizing flows for anomaly detection on tabular or time-series data, or that analyze flow failures in more depth. This weakens the positioning of the paper’s contributions relative to existing flow-based AD methods (see “Potentially Missing Related Work” below).  
   - The paper mentions Kamkari et al. (2024) and Caterini & Loaiza-Ganem (2022) but does not integrate flow-based AD frameworks like Ryzhikov et al. (2019) or self-supervised flow methods, which are directly relevant to the narrative that “NF-SLT works surprisingly well on tabular data.”

8. **Some empirical claims feel stronger than what is justified**  
   - From **Table 1**, NF-SLT is clearly a strong baseline; however, ICL and NeuTraLAD are relatively close (e.g., AUROC 0.8492 and 0.8391 with per-dataset tuning in **Table 15**) and in AUPRC ICL is competitive. Yet, the conclusion section conveys that flow-based likelihood tests “effectively detect tabular anomalies, outperforming traditional models without facing image domain challenges,” which might be interpreted as flows being decisively better.  
   - On some datasets like “yeast” (noted on Page 5), NF-SLT actually underperforms, and the explanation that the minimum performance gap is only 0.02 is somewhat hand-wavy: the phenomenon may still be present but weaker. A more nuanced discussion of where NF-SLT truly fails (even if not “counterintuitive” per Definition 3.3) would make the empirical narrative more balanced.

9. **Minor issues in exposition and references**  
   - Some references in the bibliography appear corrupted (e.g., Ahmadian et al. (2011) repeated “LKL” many times), which reduces clarity.  
   - Section 5.2 mixes discussion of homogeneity/heterogeneity, correlation, and inductive biases (CNN vs MLP) in a way that is conceptually rich but slightly scattered. Better structuring could help clarify which parts are empirically validated versus speculative.

Overall, the paper is valuable and thought-provoking, but several central claims (about definition, dimensionality, and correlation) rest on restrictive assumptions or partially supported empirical evidence.

## Potentially Missing Related Work

1. **Ryzhikov et al., “Normalizing Flows for Deep Anomaly Detection”, 2019**  
   - Relevance: Directly applies normalizing flows to anomaly detection, similar in spirit to NF-SLT. It discusses flow architectures and detection strategies and should be compared and cited in Section 2.1 and in the context of flow-based AD baselines.  
   - Suggested placement: Discuss in Related Work Section 2.1 and possibly as a baseline or at least a conceptual comparator when arguing about the practicality of NF-SLT.

2. **Zhang, J., Saleeby, K., Feldhausen, T., “Self-Supervised Anomaly Detection via Neural Autoregressive Flows with Active Learning”, 2021**  
   - Relevance: Proposes self-supervised AD using neural autoregressive flows. This highlights that flows can be used beyond simple likelihood tests and might provide alternative baselines or approaches for tabular data.  
   - Suggested placement: Section 2.2 (Counterintuitive Phenomenon) or near the discussion of deep AD models in Section 4, to contrast simple likelihood-based NF-SLT with more sophisticated flow-based AD methods.

3. **Baumgartner, D., de Souza da Silva, E., Urteaga, I., “Anomaly Detection in Time-Series via Inductive Biases in the Latent Space of Conditional Normalizing Flows”, 2026**  
   - Relevance: Explores anomaly detection using conditional flows with explicit inductive biases in the latent space. While focused on time-series, it is conceptually close to the paper’s discussion on inductive bias (MLP vs CNN) and might offer insights on how to design flows more tailored to tabular or structured data.  
   - Suggested placement: Section 2.1 with a short paragraph on flow architectures and inductive biases for AD, and perhaps referenced in the conclusion as part of future directions.

4. **Lee, J., Kim, M., Jeong, Y., “Differentially Private Normalizing Flows for Synthetic Tabular Data Generation”, 2023**  
   - Relevance: Uses normalizing flows specifically for tabular data, albeit for synthetic data generation under privacy constraints. It provides additional context on how flows behave on tabular domains and might reinforce the paper’s empirical finding that flows can model tabular distributions well.  
   - Suggested placement: Section 2.1 and/or the introduction as part of the justification for studying flows on tabular data.

5. **Kim, D., Phee, J. H., Yoon, H., “Challenging the Counterintuitive: Revisiting Simple Likelihood Tests with Normalizing Flows for Tabular Data Anomaly Detection”, 2024**  
   - Relevance: Apparently a prior work by the same authors on a closely related topic. It is important to clearly state how the current submission extends or supersedes that earlier work (e.g., new definition, more datasets, stronger theoretical analysis) to avoid confusion about incremental contribution.  
   - Suggested placement: Introduction and Related Work, explicitly delineating what is new here (e.g., Definition 3.3, dimensionality theory, ID/correlation analysis, use of complete ADBench).

6. **Behrmann, J., Vicol, P., Wang, K.-C., “Understanding and Mitigating Exploding Inverses in Invertible Neural Networks”, 2022**  
   - Relevance: Analyzes failure modes of invertible networks (normalizing flows) in high dimensions, which is highly relevant to this paper’s high-dimensional analysis and synthetic experiments in Appendix C.  
   - Suggested placement: Section 2.1 and Section 5.1 / Appendix C discussion around high-dimensional behavior and optimization/misestimation issues.

## Questions

1. **Choice and robustness of β and γ in Definition 3.3**  
   - What exact values of β and γ were used in your experiments to classify datasets as exhibiting the counterintuitive phenomenon?  
   - How sensitive are your conclusions (e.g., “almost all tabular datasets do not exhibit the phenomenon”) to these choices? A small study varying β and γ would help clarify whether the rarity is robust or an artifact of threshold selection.

2. **Use of labeled test data for hyperparameter selection**  
   - For the main results in **Table 1**, you state that a single global hyperparameter combination per model is selected based on the highest *average test AUROC across 47 datasets*. This still uses test labels for model selection.  
   - Can you clarify whether any validation splits (from training data) were used, and if not, why this evaluation protocol is acceptable for an unsupervised AD setting? Could you provide at least a subset of results where hyperparameters are tuned using only normal training data (e.g., via self-supervised or heuristic criteria) to verify that NF-SLT remains competitive?

3. **Empirical probing of dimensionality in tabular data**  
   - Beyond **Table 7** (InternetAds), do you have results for more tabular datasets where you systematically reduce dimension (e.g., via PCA or random projections) and track AUROC of NF-SLT and baselines?  
   - This would strengthen the claim that high dimensionality is a key driver of potential failures, specifically in the tabular domain.

4. **Quantitative connection between ID (d-Ratio) and NF-SLT performance**  
   - In **Table 4 (bottom)** you show that among datasets where NF-SLT is not top-ranked, a large fraction has low \(d\)-Ratios. Could you provide a scatter plot of NF-SLT’s AUROC vs d-Ratio across all tabular datasets, possibly with a simple correlation coefficient?  
   - This might reveal whether performance degrades smoothly as d-Ratio decreases, which would more directly support your narrative than the thresholded counts.

5. **Architectural choices for NF-SLT and alternatives**  
   - NF-SLT is implemented with NICE, and **Table 13** shows RealNVP slightly worse. Did you experiment with flows that incorporate stronger inductive biases for tabular data (e.g., masked autoregressive flows, coupling networks with feature grouping)?  
   - Given that CNN-based Glow models behave differently on images (Tables 2–3), a short discussion on why precisely NICE + MLP works particularly well for tabular data (beyond general MLP inductive bias) would be valuable.

6. **Clarification on synthetic Gaussian experiments and ReLU vs tanh flows**  
   - In Appendix C you mention that the problematic AUROC→0.5 behavior in very high dimensions disappears when using hyperbolic tangent activations instead of ReLU-like functions. Could you expand on this observation: was this consistent across many settings, and do you have any intuitive geometric explanation?  
   - This seems like a potentially important insight about flow architecture that is only briefly mentioned as future work.

## Flag For Ethics Review

- No ethics review needed.

## Details Of Ethics Concerns

N/A.

## Soundness Rating

2: fair.  
The empirical evaluation is extensive and reasonably well executed, but the theoretical analysis relies on strong independence and entropy assumptions; the definition of the phenomenon is baseline-dependent; and the evaluation protocol uses labeled test data for hyperparameter selection, which weakens the methodological rigor.

## Presentation Rating

3: good.  
The paper is generally well written, with detailed figures and tables (e.g., **Figure 1**, **Table 1**). However, some sections (especially the theoretical derivations and the feature-correlation argument) could be tightened, and a few equations / references contain minor inconsistencies.

## Contribution Rating

2: fair.  
The main empirical finding that simple NF-based likelihood works well on tabular data and rarely displays the image-domain paradox is interesting and practically useful. The theoretical and ID-based analyses provide additional insight but are somewhat speculative and not fully tied to real-data behavior.

## Overall Rating

4: Marginally below the acceptance threshold. But would not mind if paper is accepted.  

The paper tackles a timely and important question with a large-scale empirical study and nontrivial theoretical work. It convincingly shows that normalizing-flow likelihood tests are surprisingly strong for tabular anomaly detection on ADBench, and the dimensionality/correlation perspective is thought-provoking. However, the key conceptual pieces (definition of counterintuitive phenomenon, high-dimensional theory, correlation/ID link) rest on restrictive or unverified assumptions, and the empirical protocol leans on test labels for hyperparameter selection. These issues prevent me from recommending clear acceptance, although I see enough value that I would not object strongly if the paper were accepted after addressing some of these concerns in rebuttal.

## Reviewer Confidence

4: confident.  
I am familiar with normalizing flows, OOD/anomaly detection, and high-dimensional probability, and I carefully examined the main equations and experimental tables, though I did not line-by-line verify every proof in the appendices.
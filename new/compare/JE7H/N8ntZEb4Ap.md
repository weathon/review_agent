---
job_id: 4f1e0ea6-2b42-42fe-beba-51e847564007
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: N8ntZEb4Ap.pdf
paper: AUTONFS: Automatic Neural Feature Selection
main_score_norm: 0.4
desk_reject: false
---
# Desk Rejection Assessment:
## Paper Length  
Pass ✅.

## Topic Compatibility  
Pass ✅. The paper addresses differentiable feature selection for tabular data using neural networks, squarely within representation learning and general ML.

## Minimum Quality  
Pass ✅. The paper is in English and contains all required scientific sections (Abstract, Introduction, Related Work, Method, Experiments, Results/Analysis, Conclusion). The method is technically coherent, experiments are non‑trivial with multiple baselines and datasets, and no obvious fatal errors or data leakage are apparent from the text.

## Prompt Injection and Hidden Manipulation Detection  
Pass ✅. I do not see attempts to manipulate automated reviewing (no hidden prompts, meta‑instructions to reviewers, or suspicious content).

---

# Expected Review Outcome:

## Summary

The paper proposes AutoNFS, a neural feature selection framework that couples a masking network using Gumbel‑Sigmoid sampling with a task network trained jointly on a supervised objective. A sparsity penalty on the (continuous) mask encourages the model to automatically determine a small subset of features that suffices for a downstream classification or regression task. The authors evaluate AutoNFS on an OpenML‑based benchmark with synthetic feature corruptions and on 24 metagenomic datasets, and also report empirical time‑complexity scaling, showing competitive or better predictive performance while using fewer features than a set of classical and neural baselines.

## Strengths

1. **Clear and simple architecture, well visualized**  
   The core idea, shown in **Figure 1**, is conceptually clean: a small masking network maps a global embedding to feature logits, which go through Gumbel‑Sigmoid to form a mask that gates the input before an MLP task network. The description in Sections 3.2–3.4 and Algorithm 1 is easy to follow and implement.

2. **End‑to‑end differentiable feature selection with an explicit cardinality penalty**  
   The loss  
   \[
   \mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda \frac{1}{D}\sum_{j=1}^D m_j
   \]
   in Section 3.3 is straightforward and yields a direct handle on sparsity. The use of temperature annealing for Gumbel‑Sigmoid (Section 3.4) connects nicely to exploration / exploitation and is practically reasonable.

3. **Solid empirical performance on a standardized benchmark**  
   On the Cherepanova et al. benchmark, AutoNFS consistently achieves the best average rank across three corruption scenarios, as shown in **Figure 2**. The detailed scores in **Tables 3–5** show that AutoNFS is often best or tied‑best in accuracy / negative MSE while using substantially fewer features (see RHS of **Table 1**). This suggests the approach is at least competitive with classical FS and some neural regularization baselines.

4. **Compelling analysis of selected features and sparsity–accuracy tradeoff**  
   The mis‑selection analysis in **Figure 3(a)** shows that AutoNFS almost always selects only original (non‑auxiliary) features in the random and corrupted scenarios and has low error in the second‑order case. **Figure 3(b)** (average drop in performance when removing any selected feature) nicely supports the claim that the selected subset is “tight” in the sense that removing any one feature significantly harms performance. **Figure 6** further illustrates how the balance parameter \(\lambda\) trades off accuracy against number of features in a fairly smooth way.

5. **Useful high‑dimensional biological case study**  
   The metagenomic experiments in **Table 2** show that AutoNFS reduces dimensionality drastically (to about 7.7% of original features on average) while slightly improving mean performance for both MLP and RF. This is a realistic high‑dimensional setting where interpretability and cost of downstream models matter, and the results are practically interesting.

6. **Empirical time‑scaling comparison**  
   The time‑complexity plots in **Figure 4(a)** and **Figure 4(b)** are a useful contribution: they empirically fit exponents \(\alpha\) in \(t \approx D^\alpha\) and show AutoNFS with a very small exponent (\(\alpha \approx 0.08\)). Even if one disputes the “constant” wording, the plots demonstrate that AutoNFS scales much more favorably in their setup than several popular FS baselines.

7. **Interpretability visualizations on MNIST**  
   The MNIST visualizations in **Figures 7 and 8** are helpful sanity checks. They show that selected features (pixels) concentrate in central, digit‑bearing regions, and have higher entropy and more discriminative class‑conditional distributions than non‑selected features, which supports the interpretability aspect.

## Weaknesses

1. **Limited conceptual novelty relative to prior differentiable FS (Hard‑Concrete, STG, INVASE, Concrete AE, etc.)**  
   The core mechanism combines: (i) a global set of feature logits, (ii) Gumbel‑Sigmoid sampling, and (iii) an \(\ell_1\)-style penalty on the (soft) mask. This is extremely close in spirit to Hard‑Concrete \(L_0\) regularization (*Louizos et al., 2017*), Stochastic Gates (*Yamada et al., 2020*), and related differentiable FS methods, and also reminiscent of Concrete Autoencoders (*Balin et al., 2019*). The paper cites these works in Section 2 but does not clearly articulate **what is actually new beyond**:
   - replacing Hard‑Concrete/Concrete with Gumbel‑Sigmoid (which is already standard),
   - using a *single global embedding* to generate feature logits (as opposed to learning them directly), and  
   - using a fixed \(\lambda\) for average‑mask penalty.  
   None of these is argued to fundamentally change capabilities, guarantees, or optimization properties. The novelty claim around “automatic determination of feature count” is essentially the same story as any \(\ell_0/\ell_1\)-penalized mask model; those works also let the model choose how many features to keep under a sparsity penalty. Without a stronger conceptual or theoretical argument, the contribution risks being seen as yet another instantiation of an already well‑explored pattern.

2. **Missing or weakly positioned baselines against closest neural FS methods**  
   The experimental comparison omits several *very* relevant neural FS baselines that the paper itself cites, notably:
   - Hard‑Concrete \(L_0\) (Louizos et al., 2017),
   - STG (Yamada et al., 2020),
   - Concrete Autoencoders (Balin et al., 2019),
   - INVASE (Yoon et al., 2018),
   - GFS‑style Gumbel‑based selectors.  
   Instead, the baselines in **Tables 3–5** are mostly classical (Univariate, Lasso, RF, XGBoost) plus generic neural models with \(\ell_1\) or attention. This makes it hard to assess whether AutoNFS advances the state of the art in differentiable FS *per se* or just outperforms older or less targeted methods. Given that the method is squarely in the same design space as these works, the absence of direct empirical comparison is a serious limitation.

3. **Overstated claims about “nearly constant” computational overhead**  
   The paper repeatedly states that AutoNFS “achieves a nearly constant computational overhead regardless of input dimensionality” (Abstract, Sections 1 and 3.1). However, the method still computes:
   - a linear layer \(f_\phi: \mathbb{R}^{d_e} \to \mathbb{R}^D\),
   - Gumbel‑Sigmoid for every feature dimension,
   - element‑wise masking \(x_m = m \odot x\), and then
   - a dense MLP whose first layer scales with \(D\).  
   All of these are \(O(D)\) per example. **Figure 4(a)** and **4(b)** show that the empirical exponent \(\alpha \approx 0.08\) is small *in the range tested*, but this is still polynomial, not constant, and reflects a particular implementation / hardware regime rather than algorithmic complexity. Moreover, the comparison mixes filter‑style FS (which may be implemented in relatively unoptimized code) with a neural method whose cost is dominated by the shared task network. The complexity analysis is empirical only and does not normalize for implementation details, which makes the “nearly constant” headline somewhat misleading.

4. **The “automatic minimal feature count” claim is not really substantiated**  
   The paper repeatedly says AutoNFS “automatically determines the minimal set of features” (Abstract, Conclusion). What is actually optimized is the tradeoff between \(\mathcal{L}_{\text{task}}\) and \(\lambda \mathcal{L}_{\text{select}}\), for a fixed \(\lambda\) and training budget. There is no theoretical or empirical argument that the resulting subset is *minimal* under any well‑defined criterion, even approximately. **Figure 3(b)**, which measures average performance drop when removing any selected feature, shows that the subset is tight under that particular measure, but that does not rule out alternative, smaller subsets with similar performance. Additionally:
   - \(\lambda\) is fixed to 1 across datasets (Section 3.3 and Appendix C) mostly for convenience, yet **Figure 6** clearly shows that the number of selected features varies dramatically with \(\lambda\), and that excessively high \(\lambda\) harms accuracy. This indicates that the feature count is actually **hyperparameter‑dependent**, not emergent in a parameter‑free way.  
   - There is no discussion of how sensitive results in **Tables 1–5** are to \(\lambda\) or the annealing schedule; all are reported for one configuration.  
   The “automatic minimality” framing is therefore oversold.

5. **Global (data‑independent) mask may limit expressivity and is not justified empirically**  
   AutoNFS learns a **single** embedding \(e\) and mask \(m\) shared across all samples (Section 3.2 and 3.5). This induces a global feature subset, which is appropriate for some settings but can be strictly suboptimal when different regions of the input space require different features. Existing methods like INVASE explicitly exploit instance‑specific masks. The paper does not justify why a global mask is sufficient or desirable in the benchmark tasks, nor does it present any ablation or analysis comparing global vs instance‑wise selection. In fact, the MNIST analysis in **Figures 7–8** uses a global mask that ignores background pixels, which is intuitive but hardly challenges the limitations of global selection. Without exploration of this design choice, the method might under‑serve more complex tabular problems.

6. **Mathematical formulation lacks some important details / possible inconsistencies**  
   - In Section 3.3, \(\mathcal{L}_{\text{select}} = \frac{1}{D}\sum_{j=1}^D m_j\) uses a single mask sample per mini‑batch (Algorithm 1, lines 6–10). This means the effective penalty on each feature is based on one stochastic draw of \(m_j\) per batch. The paper does not discuss the variance of this estimator or whether averaging over multiple samples of Gumbel noise improves stability.  
   - The hard mask at inference (Section 3.5) sets \(m_i = 1\) if \(\sigma(w_i) > 0.5\). This “hard Gumbel‑Sigmoid” discards the learned sampling temperature \(\tau\) and ignores Gumbel noise altogether. In training, however, masks are sampled with \(\sigma((w_i + g_i)/\tau)\). There is no analysis of the mismatch between the training distribution over \(m\) and the deterministic test mask, nor of whether choosing the 0.5 threshold is appropriate given the specific penalty \(\mathcal{L}_{\text{select}}\).  
   - The loss in Algorithm 1, line 13, is written as  
     \[
     \mathcal{L}_{\text{task}} = -\sum_{i=1}^B \sum_{c=1}^C y_{i,c} \log \hat{y}_{i,c},
     \]
     with no normalization by \(B\), which changes the relative scale of \(\mathcal{L}_{\text{task}}\) vs \(\mathcal{L}_{\text{select}}\) depending on batch size. Since \(\mathcal{L}_{\text{select}}\) is averaged over \(D\), the effective sparsity strength is sensitive to batch size, but this is not discussed.  
   While none of these issues is necessarily fatal, they should be clarified and, ideally, supported with ablations or theory.

7. **Time‑complexity experiment design could be better specified and controlled**  
   In **Figure 4(a)**, the y‑axis is “relative time per iteration”, and the fit \(t \approx D^\alpha\) yields \(\alpha \approx 0.08\) for AutoNFS. However:
   - It is unclear whether all methods are implemented in the same framework (e.g., PyTorch / NumPy) and whether all use optimized vectorized operations. For instance, a naive implementation of ANOVA F tests in Python can have quite different scaling behavior than a highly optimized C++ library.  
   - The “time per iteration” for AutoNFS includes both FS and task network training, while filter methods typically run FS once and then train a task model. The comparison does not clearly separate one‑time FS cost from training cost.  
   - There is no mention of the dataset sizes or number of iterations used to measure time, so reproducibility is limited.  
   Given that a major claimed contribution is scalability, the current complexity section, while suggestive, is not rigorous enough to fully support the strong claims.

8. **Benchmark coverage of scenario where feature budget is user‑specified is missing**  
   A key selling point of AutoNFS is avoiding the need to specify the number of features. However, in many practical scenarios users *do* have a budget (e.g., “at most 20 lab tests”). It would be informative to see:
   - how AutoNFS performs under a constraint like “select exactly \(k\) features” (using the ranking induced by logits as noted in Section 3.5),  
   - or how it compares to baselines when tuned to the same number of selected features.  
   Currently **Table 1** and **Figures 3 and 6** focus on AutoNFS’s own emergent feature counts, but the baselines are constrained to a fixed number (original number of real features), which is not a fair budget‑controlled comparison.

9. **Limited analysis of robustness and stability of selected subsets**  
   The paper does not analyze how stable the selected feature set is across random seeds or data splits. In high‑dimensional, correlated‑feature settings (e.g., metagenomics), stability is crucial. The visualization in **Figure 5** on a single dataset illustrates evolution of selection probabilities but does not quantify stability. There is no metric like Jaccard/overlap across runs, which makes the interpretability story weaker.

10. **Metagenomic experiment lacks comparison to strong domain baselines**  
    While **Table 2** is interesting, it only compares “full features vs AutoNFS‑reduced features” for MLP and RF. It does not compare to simpler but widely used biological FS methods (e.g., univariate tests, RF importance, Lasso, tree‑based methods tuned to sparse settings) on these specific datasets. Given the strong prior literature in omics FS, this limits the strength of the claim that AutoNFS is especially effective for biological data.

## Potentially Missing Related Work

1. **Wydmański & Śmieja, “GFSNetwork: Differentiable Feature Selection via Gumbel‑Sigmoid Relaxation” (2025)**  
   This work appears to use Gumbel‑Sigmoid for differentiable FS, which is *very* close to AutoNFS’s core mechanism. It should be discussed in Section 2 as a directly related method and compared conceptually (global vs instance‑specific masks, training objective, complexity), and ideally included as a baseline if feasible.

2. **Nilsson et al., “Indirectly Parameterized Concrete Autoencoders” (2024)**  
   Extends Concrete Autoencoders using Gumbel‑Softmax; clearly relevant to differentiable subset selection with continuous relaxations. It should be cited after the Concrete Autoencoders discussion in Section 2 and contrasted with AutoNFS’s design (no reconstruction objective, global mask).

3. **Peng et al., “scFSNN: A Feature Selection Method Based on Neural Network for Single‑Cell RNA‑seq Data” (2024)**  
   Another neural FS method applied to biological data. It should be mentioned in Section 2 and related to the metagenomic case study in Section 4.2 as an example of domain‑specific neural FS in omics.

4. **Passemiers et al., “A Quantitative Benchmark of Neural Network Feature Selection Methods for Detecting Nonlinear Signals” (2024)**  
   Provides a benchmark of neural FS methods, highly relevant to the positioning and evaluation strategy of AutoNFS. It should be discussed in Section 2, and the authors might adapt some of its benchmark tasks or comparisons to strengthen Section 4.

5. **Fang et al., “Automatic Author Name Disambiguation by Differentiable Feature Selection” (2023)**  
   Uses differentiable FS in a specific application domain; including it in Section 2 would better situate AutoNFS as part of a broad movement towards differentiable FS across tasks.

6. **Alissa et al., “Automated Algorithm Selection: From Feature‑Based to Feature‑Free Approaches” (2023)**  
   Discusses feature‑based algorithm selection and touches on FS in the AutoML context, relevant to the broader motivation in the Introduction and Conclusion. Could be cited in Sections 1 and 5 when discussing AutoNFS as a drop‑in component in larger AutoML pipelines.

7. **Acharya & Zhang, “Feature Selection and Extraction for Graph Neural Networks” (2019)**  
   While focused on graphs, it is still part of the neural FS literature and can be briefly mentioned in Section 2 to acknowledge FS in non‑tabular settings and distinguish AutoNFS’s scope (tabular).

## Questions

1. **Comparison to Hard‑Concrete / STG / INVASE / Concrete AE**  
   Can you provide experimental comparisons against at least one or two of the most relevant differentiable FS baselines (e.g., STG, Hard‑Concrete, Concrete AE, INVASE) on the OpenML benchmark? If not feasible, please explain constraints and at least provide a conceptual and/or synthetic comparison that clarifies where AutoNFS has an advantage.

2. **Clarifying the “constant” overhead claim**  
   Could you clarify what exactly is meant by “nearly constant computational overhead” and how the empirical exponents in **Figure 4(b)** were obtained? In particular,:
   - Are all methods implemented in the same framework and optimized similarly?  
   - Does the timing for filter methods include only FS or also subsequent classifier training?  
   - Would you be willing to tone down the language in the paper to something like “sublinear scaling in the tested range” and explain that the overhead is measured relative to other FS methods, not absolute \(O(1)\) complexity?

3. **Sensitivity to \(\lambda\) and temperature schedule**  
   **Figure 6** focuses on a single dataset. Could you provide aggregate statistics over several datasets (or at least more examples) to show how robust AutoNFS is to the choice of \(\lambda\)? Also, have you tried alternative annealing schedules or fixed temperatures, and how do they affect both accuracy and sparsity?

4. **Batch size and loss scaling**  
   In Algorithm 1, \(\mathcal{L}_{\text{task}}\) is a sum over examples, not an average. Was this the actual implementation? If yes, how does changing the batch size affect the effective sparsity strength, given that \(\mathcal{L}_{\text{select}}\) is normalized by \(D\)? Some clarification (and possibly re‑scaling to a mean loss) would be helpful.

5. **Multiple Gumbel samples vs. one**  
   Did you experiment with drawing multiple Gumbel noise samples per batch and averaging the resulting masks for the penalty and task loss, to reduce variance? If so, what were the effects on training stability and selected feature sets?

6. **Stability of selected features**  
   How stable is the learned mask across different random seeds or training runs? It would be useful to see overlap statistics (e.g., Jaccard index) for selected subsets on a few datasets, especially the metagenomic ones.

7. **Global vs instance‑specific masks**  
   Have you explored a variant where the masking network takes \(x\) as input (i.e., an instance‑specific selector) rather than a global embedding? A small ablation comparing global vs per‑instance masks would clarify the impact of this design choice.

## Flag For Ethics Review

- No ethics review needed.  

## Details Of Ethics Concerns

N/A.

## Soundness Rating

2: fair.  
The method is technically coherent and experiments are substantial, but key baselines are missing, several claims (constant overhead, minimality) are overstated relative to the evidence, and some aspects of the mathematical / experimental setup (loss scaling, variance of mask sampling, stability) are under‑analyzed.

## Presentation Rating

3: good.  
The paper is generally well organized and clearly written, with helpful figures (e.g., **Figures 1–3, 5–8**) and comprehensive tables (especially **Tables 1–5**). However, some claims are phrased too strongly, and the related work and positioning relative to very close prior methods could be sharpened.

## Contribution Rating

2: fair.  
The empirical results on benchmarks and metagenomic data are interesting and likely useful to practitioners, but the conceptual step beyond existing differentiable FS methods is limited, and the absence of direct comparisons to the closest baselines makes it hard to argue for a strong advance.

## Overall Rating

4: Marginally below the acceptance threshold. But would not mind if paper is accepted.  
The paper presents a clean, practically‑useful instantiation of differentiable FS with solid experiments and nice analyses of sparsity and interpretability, which I appreciate. However, the methodological novelty over prior neural FS work appears modest, critical baselines are missing, and some central claims (minimality, near‑constant overhead) are oversold relative to the evidence. With stronger positioning vs existing differentiable FS methods, additional baselines, and toned‑down complexity claims, this could become a solid contribution.

## Reviewer Confidence

4: confident.  
I am familiar with differentiable FS (Hard‑Concrete, STG, Concrete AE, INVASE, etc.), checked the math and experimental setup in detail, and feel confident about the main points of my assessment, though additional baselines could still shift my view on the empirical strength.
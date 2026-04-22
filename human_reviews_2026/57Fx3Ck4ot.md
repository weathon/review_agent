# Enhancing Deep Tabular Models with GBDT-Guided Piecewise-Linear Embeddings

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 8, 4, 4

## Abstract
Tabular data remains central to many scientific and industrial applications. Recently, deep learning models are emerging as a powerful tool for tabular data prediction, outperforming traditional methods such as Gradient Boosted Decision Trees (GBDTs). Despite this success, the fundamental challenge of feature heterogeneity still remains. Unlike in image or text modalities where features are semantically homogeneous, each tabular feature often carries a distinct semantic meaning and distribution. A common strategy to address the heterogeneity is to project features into a shared high-dimensional vector space. Among the various feature types in tabular data, categorical features are effectively embedded via embedding bags, which assign a learnable vector to each unique category. In contrast, effective embeddings for numerical features remain underexplored. In this paper, we argue that piecewise-linear functions are well suited to modeling the irregular and high-frequency patterns often found in tabular data, provided that breakpoints are carefully chosen. To this end, we propose GBDT-Guided Piecewise-Linear (GGPL) embeddings, a method comprising breakpoints initialization using GBDT split thresholds, stable breakpoint optimization using reparameterization, and stochastic regularization via breakpoints deactivation. Thorough evaluation on 46 datasets shows that applying GGPL to a range of state-of-the-art tabular models consistently improves performance—especially on regression tasks—or at least matches their native numerical embeddings. Together with its negligible overhead, this suggests that GGPL can serve as a practical default numerical embedding for future tabular architectures. The code is available in the supplementary material.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes GBDT-Guided Piecewise-Linear (GGPL) embeddings for numerical features in tabular DL. The recipe: (i) initialize breakpoints from XGBoost split thresholds; (ii) optimize breakpoint locations via a simplex reparameterization; (iii) apply stochastic breakpoint deactivation as a regularizer (used only for regression). On 46 datasets, the authors report average-rank gains when swapping GGPL into several backbones (MLP, T2G-Former, ModernNCA, TabM-mini), with TabM-mini-GGPL achieving the best overall average rank (2.96) and small but consistent ablation gains over a uniform PLE baseline. They also present Wilcoxon tests against alternative numerical embeddings and against native embeddings in other backbones, and a sensitivity study showing default XGBoost is “not worse” than tuned (p=0.067)

### Strengths
Tidy, drop-in module; clear problem framing around numerical embeddings. The method is simple and training-only, with no stated inference-time overhead relative to prior PLEs.  

Broad evaluation (46 datasets, 15 seeds), with dataset characteristics analysis and ablations (I/O/R).  

Statistical testing across seeds/datasets is a plus (stratified Wilcoxon).

### Weaknesses
1. Using tree thresholds to seed piecewise segments is very close in spirit to target-aware or supervised breakpoint selection used in prior PLE work; the key twist is to harvest splits from a GBDT. The simplex reparameterization is a standard trick to enforce ordered breakpoints, and “stochastic breakpoint deactivation” is conceptually akin to dropout over segments. The paper reads as engineering polish on known ingredients rather than a conceptually new embedding family.  

2. Main claims are summarized via average ranks over mixed tasks and datasets (Table 2). Ranks can hide small absolute gains and do not convey practical effect sizes. Some gains are modest (e.g., MLP: rank 10.32→7.64 for regression; classification gains are ~1 rank or worse for some backbones), and T2G-Former’s classification rank slightly degrades with GGPL (8.83→8.94). Absolute accuracy/RMSE deltas (with CIs) should be aggregated and reported. 

3. GGPL introduces extra knobs (total breakpoints K, deactivation p) with wide search spaces (up to 48× the number of numeric features), which effectively increases capacity and tuning surface relative to some baselines’ numerical encoders. While authors state non-GGPL hyperparameter spaces were kept identical to prior work, GGPL-specific spaces add degrees of freedom; it’s unclear whether baseline numerical embeddings received equivalently rich capacity/tuning to match. A capacity-matched control (e.g., target-aware PLE with the same K sweep) would be important. 

4. Despite “no inference overhead,” the training pipeline requires fitting an auxiliary XGBoost and tuning extra GGPL hyperparameters (and the overall protocol uses Optuna with 100 trials on most datasets). The paper does not quantify training-time/energy overheads vs. baselines. The “default XGBoost is fine” claim hinges on p=0.067 (borderline), which does not strongly support robustness of the initializer.  

5. The stochastic breakpoint regularizer is deliberately disabled for classification, and the improvements there are minor or negative in places (e.g., T2G-Former). This undercuts the “effective across tasks” narrative. 

6. Beyond periodic/PLE/T2V, the paper omits comparisons to other learnable, high-frequency-friendly encoders (e.g., spline-based or lattice-based numerical transforms). Even one strong representative would help calibrate whether GGPL’s benefits stem from smarter breakpoints or just more flexible piece counts. (The paper’s own framing highlights high-frequency target components as the motivation.)

### Questions
Please report aggregate effect sizes (median % improvement and mean normalized deltas) alongside ranks; include 95% CIs across seeds/datasets. Can you share per-task (regression vs classification) absolute improvements averaged across datasets? (You already have Tables 14–15—aggregate them.)  

Provide training time/compute comparisons (including the XGBoost prepass and Optuna trials) vs. target-aware PLE and periodic embeddings under matched budgets.  

Run a capacity-matched control: target-aware PLE with the same K search as GGPL across backbones, and report whether GGPL still wins.  

Add at least one spline/lattice numerical encoder baseline to test the “high-frequency” rationale.  

For classification, try a breakpoint regularizer that doesn’t enforce smoothness (e.g., piece-drop with re-weighting rather than linear interpolation) or show that disabling R is indeed optimal across datasets.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The authors presented a novel deep learning method for tabular data prediction
The method revolves around embedding numerical features so that they are easier for the deep learning method to use
Building upon previous methods of using fixed breakpoints of piece wise linear embeddings, the authors used gradient boosted decision trees to determine the breakpoints.
The authors showed that the proposed method had better performance compared to not using the GGPL embedding in numerous models.

### Strengths
* Paper is clear and had direct extension of previous work
* GGPL embedding is versatile and could easily be applied to any deep learning architecture
* Results showed clear improvements in performance improvements across multiple models regardless of their architecture

### Weaknesses
* Discretizing numeric features into piecewise linear has been explored in previous works [Gorishniy 2022]. The only innovation is to define the breakpoints with gradient boosted trees 
* Since the discretization is dependent on gradient boosted trees, it might be interesting to understand the performance and memory requirements changes in different datasets. In particular datasets with numerical features with a wide range

### Questions
* Do different gradient boosted tree methods alter performance?
* The final snapshot of the trained XGBoost method was used to get the breakpoints, were there explorations to check the intermediate trees to see if there any possible performance boosts?

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
3

### Summary
This paper tackles the challenge of learning expressive and robust representations for numerical features in deep tabular models. While categorical features are effectively handled through embedding layers, numerical embeddings remain underexplored. The authors introduce an elegant and effective approach, GBDT-Guided Piecewise-Linear (GGPL) embeddings, which integrates three components: initialization of breakpoints using GBDT (XGBoost) split thresholds, stable optimization via simplex-based reparameterization to preserve order, and stochastic breakpoint regularization to prevent overfitting. Comprehensive experiments on multiple tabular datasets demonstrate that GGPL consistently enhances the performance of state-of-the-art tabular models without adding inference-time cost.

### Strengths
1. The paper tackles an important and timely problem in deep learning for tabular data, where numerical feature embeddings remain underexplored.
2. The writing is clear and well-organized, making the paper easy to follow and accessible to both deep learning and tabular data researchers.
3. The experimental setup is comprehensive, covering multiple datasets and multiple baselines, which demonstrates the generality and robustness of the proposed approach.

### Weaknesses
1. The paper appears to make several overclaims regarding novelty. In particular, Contribution 1 (lines 80–82) states that the proposed method introduces piecewise-linear embeddings for numerical features, but this idea has already been explored in prior work, notably by Yury Gorishniy et al., “On Embeddings for Numerical Features in Tabular Deep Learning,” NeurIPS 2022. While the authors cite this paper, the Introduction section (e.g., lines 88–90) does not sufficiently acknowledge or discuss the methodological overlap and differences. I recommend explicitly positioning this work relative to Gorishniy et al. (2022), clarifying what is truly novel (e.g., the GBDT-guided initialization or stochastic regularization), to avoid the impression of rediscovery.
2. Could you clarify how XGBoost and CatBoost were implemented? Were they run with default hyperparameters  (as in lines 224–230) or tuned per dataset? If tuned, could you report detailed parameters per dataset?

### Questions
I have questions about the implementation of tree-based method. see above

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes an improved numerical feature embedding scheme for tabular neural networks. Its design is based on a piecewise-linear embedding from [Gorishniy 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/9e9f0ffc3d836836ca96cbf8fe14b105-Abstract-Conference.html). The authors propose to take the feature binning thresholds from maximal gain XGBoost tree splits instead of taking them uniformly or from a single-feature tree building process. The authors also propose a procedure to finetune the bin edges (based on a reparametrization that keeps the order intact). Authors also propose a dropout-like embedding regularization via dropping bin thresholds at random during training. The efficacy is evaluated on a benchmark from the [TabM 2025](https://arxiv.org/abs/2410.24210) paper and claims that the new "GGPL" embedding scheme maintains the best average rank when combined with SoTA tabular neural networks and is generally better than prior embedding variations.

### Strengths
The experiments are well done and trustworthy. Most of the relevant details for reproduction are present and the paper is clearly written. The motivation of creating a more universal (e.g. without the need to tune the frequency hyperparameter) embedding which is better than the PLE from prior work is also strong and I believe this is an important area for future tabular neural networks.

### Weaknesses
In my view the improvement and its consistency is not very compelling practically speaking. Looking at the raw results table in the appendix, the new embedding may produce slightly better results than the alternatives, but the magnitude of the improvement and its consistency is not good for the proposed complexity.

Another related weakness is the lack of explanations regarding where the effects come from (except some mentions that selecting bin edges is hard, prone to overfitting but helps with representational power). I suggest the authors look deeper into where the improvement actually comes from. I may suggest a few recent papers [1](https://arxiv.org/pdf/2509.04430), [2](https://arxiv.org/abs/2411.00247) that do this for tabular data specifically, maybe these perspectives could shed some more light onto why the proposed modifications help.

### Questions
- Can you provide average % improvement, as a means to judge the scale of the improvement brought by the new embedding scheme?
  - Looking through the results table in the appendix, it seems that most of the classification results are not significant when comparing TabM (I am assuming it uses the prior PLE variation) with TabM-GGPL. For regression problems improvement seems a bit more significant, but the relative gains are still marginal.
- Does dropout-like regularization really only help with regression? This claim seems a bit too high level for the usual state of affairs with tabular data (e.g. very very diverse set of datasets)
- How do you measure "Tortuosity" in Figure 3b?

### Soundness
2

### Presentation
2

### Contribution
2

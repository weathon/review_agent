# Self-supervised Learning to Predict Optimizer Performance: A Trade-off Study Between Ground Truth Requirements and Prediction Quality

- Decision: Reject
- Scores: 6, 6, 4, 2

## Abstract
Predicting optimizer performance typically requires costly ground-truth performance data, since each label requires multiple executions of optimization algorithms, which are computationally expensive to obtain on a large set of problems. We investigate whether established self-supervised learning (SSL) can reduce this labeling burden while preserving performance. Using four landscape representations (ELA, DeepELA, DoE2Vec, TransOptAS), seven learners (Random Forest (RF), XGBoost, LightGBM, SCARF, TabNet, VIME, and SAINT), and four train–test splitting strategies with increasing distribution shift (I, R, PC, P), we quantify the label–quality trade-off for predicting the performance of five Differential Evolution (DE) and five Particle Swarm Optimization (PSO) configurations in a controlled comparative study. We find that across settings, SSL, especially SCARF on learned representations, matches or surpasses supervised RF with only 50–75% labels. Under moderate shift (PC), SSL yields substantial benchmarking gains: for multiple configurations, we can omit around 80% of function evaluations without worsening MAE, with improvements visible already at 25% labels. Computed ELA remains a strong supervised baseline in low-shift regimes (I/R), whereas all representations amplify SSL benefits under shift (PC). The zero-shot split (P) remains challenging, motivating richer problem-space augmentation. The results confirm that SSL benefits are portfolio-agnostic (across DE and PSO). A practical guideline emerges: use RF+ELA for in-distribution tasks, while when facing distribution shifts, apply Doe2Vec with SCARF pre-training and fine-tuning on 50–75% of labels, which achieves performance close to fully supervised models while reducing the need for costly function evaluations.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
1

### Summary
This paper addresses a significant and practical problem in the field of automated algorithm selection (AAS) for black-box optimization (BBO): the prohibitive cost of acquiring ground-truth performance labels. The authors conduct a comprehensive empirical study to investigate whether self-supervised learning (SSL) can mitigate this labeling burden.

The core idea is to decouple representation learning from performance prediction. An SSL model (SCARF or TabNet) is first pre-trained on problem landscape representations (ELA, DeepELA, DoE2Vec, TransOptAS) without performance labels. This pre-trained model is then fine-tuned on a small fraction (25%-75%) of the expensive ground-truth performance data (from DE and PSO portfolios). The goal is to quantify the trade-off between the number of labels used and the final prediction quality, especially under varying degrees of distribution shift (I, R, PC, P splits).

### Strengths
1. The paper tackles a highly practical issue. The cost of running extensive benchmarks (e.g., all DE/PSO variants on all BBOB problems) is a major bottleneck in meta-learning for AAS. The "green benchmarking" motivation is strong and timely.

2. The results are not ambiguous. The paper clearly demonstrates a "label-efficiency sweet spot," finding that SSL (especially SCARF + DoE2Vec) can match or exceed the fully-supervised RF baseline with only 50-75% of the labels.

3. The paper correctly identifies that the real value of SSL is not in easy in-distribution tasks (where RF+ELA is fine) but in the more realistic PC split (moderate shift). Here, SSL provides substantial gains, allowing for ~80% label omission (Table 2), which is a significant finding.

### Weaknesses
1. The paper honestly reports that the zero-shot (P split) scenario remains challenging for all methods. While SSL shows some minor gains, no method truly solves this hardest problem. This suggests that the current SSL pre-training tasks and representations still do not capture the "essence" of problem landscapes well enough to generalize to entirely unseen problem classes.

2. The study focuses on SCARF (contrastive) and TabNet (masking). While these are good representatives, the tabular SSL field is evolving. It's unclear if other methods (e.g., VIME, SubTab) might yield different results.

3. TabNet is consistently unstable at low label ratios (e.g., 25%) across many experiments (Fig 1). The paper notes this but doesn't fully explore why this specific SSL method fails so dramatically in the low-data regime compared to SCARF, which is generally stable.
4. The custom metric $\Delta_{f,s,l,r}$ is slightly convoluted. While the goal is to quantify "savings," the calculation (scaling by $\overline{p}_{f,s,l,r} \cdot (1-l)$) is indirect. A more direct comparison, such as "MAE of SSL at 50% labels vs. MAE of RF at 100% labels," might be clearer. However, Figure 1 and Table 1 (which are clear) support the conclusions.

### Questions
see weakness

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies label efficiency for predicting black-box optimizer performance via self-supervised learning (SSL) on problem-landscape representations. The authors conduct a well-controlled comparative study across four problem representations (ELA, DeepELA, DoE2Vec, TransOptAS), three learners (RF, SCARF, TabNet), and four train-test splits with increasing distributional shift (I, R, PC, P).

### Strengths
- It’s a well-posed, practically relevant question that discerning how many ground-truth evaluations (expensive) are really needed for reliable performance prediction under distribution shift.
- Extensive experimentations: 4 reps × 3 learners × 4 splits × 4 label ratios; multi-target regression over five configurations per portfolio; DE and PSO considered (PSO results in appendix).
- The author defines a clear Δ (omittable evaluations) metric and reports large savings in PC even at 25% labels.

### Weaknesses
- All problems are synthetic BBOB recombinations at fixed dimension (d=10) with a single evaluation budget. The affine blend construction also ties geometry/topology to BBOB priors. I recommend ablations over dimension, budget, and non-BBOB families, and at least one real-world surrogate (e.g., hyperparameter optimization on tabular models) to bound claims.
- Using RF alone (plus a naive baseline) undercuts the decisiveness of conclusions; RF is solid but not SOTA for tabular regression. Stronger baselines (XGBoost/LightGBM/CatBoost and TabR/RealMLP (I might miss some SOTA tab reg. model since I am not in that field)) could alter the crossover points where SSL “wins” and materially change Δ. At minimum, include XGBoost and LightGBM and recompute.
- Though the abstract concedes that P is challenging; Table 1 suggests TabNet can be brittle at low label ratios, with SSL sometimes failing to beat RF. The paper should analyze why: e.g., representation shift (pre-training corpora vs. P split topology), capacity-label mismatch, or optimization noise. Also, for Interpretability: which landscapes drive SSL gains? We see consistent PC improvements, but the mechanism remains opaque.

### Questions
1. How do conclusions change for higher $d$ (e.g., 20/40) ? Do learned reps or ELA degrade differently?
2. If resources are available, how do other SOTA model compare to RF under I/R/PC/P?
3. For the P split, have you tried domain augmentation (e.g., class-level perturbations) or adapter fine-tuning to stabilize SSL?

### Soundness
2

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
3

### Summary
This paper is an empirical study on using existing SSL methods to reduce labeling costs for the AAS task . It does not propose any new method or model. It just applies existing tools to this problem.
The evidence for SSL being effective is weak. The experiments show a small advantage only in the moderate.
All experiments are on low-dimensional d=10 problems, which makes the conclusions highly questionable for any practical use. Given the lack of novelty and major experimental limits, this work is not up to the ICLR standard.

### Strengths
The problem itself is practical (reducing label costs), which is a real bottleneck 
The experimental setup is systematic comparing. 
The discussion of these different shifts is useful.

### Weaknesses
The main weakness is that the paper is lack of novelty.  
The paper just applies existing SSL methods. It does not propose any new method. This is more of a benchmark report.
Besides, the experiments are not sufficient.
All the experiments use d=10. The conclusions cannot be generalized to larger d, so the results are not very useful.
In addition, the analysis in the paper is superficial and lack of details. The paper says SCARF is better than TabNet but it does not explain why in the discussion .

### Questions
Why only use d=10? The results from such a low dimension are not very convincing. Could the authors show any results on higher dimensions?

### Soundness
2

### Presentation
2

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
This paper investigates whether self-supervised learning (SSL) can reduce the number of algorithm performance labels needed to predict optimizer performance in black-box optimization. The authors apply two SSL methods (SCARF and TabNet) to an existing dataset from Cenikj et al. (2025), evaluating on four landscape representations and four data splitting strategies. The main finding is that SSL methods, particularly SCARF on learned representations, can match supervised Random Forest baselines with 50-75% of labels under certain conditions.

### Strengths
- **Timely motivation:** The study addresses a real issue in black-box optimization; the high computational cost of acquiring performance labels—and positions the work within the broader sustainability (“green benchmarking”) agenda.  
- **Comprehensive experiments:** The evaluation spans multiple representations, SSL methods, and both DE and PSO portfolios under four distinct data splits.  
- **Reproducibility:** Code and implementation details (Appendix B) are thorough, and the repository is provided.  
- **Consistent empirical results:** The main trend—that SCARF helps under moderate shift (PC)—is replicated across optimizers and representations.

### Weaknesses
#### 1. **Minimal Novelty and Heavy Dependence on Prior Infrastructure**
Nearly every component is reused from **Cenikj et al. (2025)**:
- identical dataset (8,280 affine recombinations);
- identical representations (ELA, DeepELA, DoE2Vec, TransOptAS);
- identical data splits (I, R, PC, P);
- identical portfolios (5 DE, 5 PSO configurations);
- identical metrics and protocols.

The sole addition is applying off-the-shelf SSL methods (SCARF [Bahri et al., 2022], TabNet [Arik & Pfister, 2021]) without modification or theoretical framing.  
No new SSL pretext task, architectural adaptation, or analytical insight is proposed.  
Thus, while the empirical study is well-executed, it constitutes an **incremental replication**, not a conceptual contribution suitable for ICLR.

*To strengthen the work*, the authors could:
- design domain-specific SSL objectives (e.g., perturbations of landscape features);
- analyze sample-efficiency or generalization bounds;
- visualize or interpret SSL embeddings to reveal *why* they help.

---

#### 2. **Unjustified and Contradictory Definition of “Distribution Shift”**
Section 4 claims that the four evaluation splits follow an *increasing difficulty order*  
**I < R < PC < P**, implying that the *Problem Split (P)* scenario is the strictest zero-shot regime.  
However, this claim is empirically and conceptually unsupported.

From **Table 1 (RLS = False)**:
| Split | Feature | Model | MAE (Mean ± SD) |
|:------|:---------|:------|:----------------|
| **PC** | DeepELA | Baseline | **0.261 ± 0.088** |
| **P**  | DeepELA | Baseline | **0.186 ± 0.014** |

Similar results appear for **DoE2Vec**, **ELA**, and **TransOptAS**, and across **RF** and **SCARF**.  
Thus, the empirically measured errors *contradict* the claimed ordering—PC is consistently *harder* than P according to MAE, despite being defined as “easier.”  
The authors never explain or reconcile this discrepancy.

More fundamentally, the notion of “distribution shift” in **meta-learning and algorithm selection** is itself **ill-defined**.  
There is no universally accepted metric of shift between problem distributions, and the boundaries between in-distribution, partial-combination, and zero-shot settings are heuristic.  
Given this ambiguity, it would be more accurate and scientifically sound to **avoid asserting any hardness ordering** among the four splits altogether.

Instead, the paper should:
- Present the splits simply as *distinct evaluation scenarios* rather than a progression of shift magnitude.  
- Explicitly acknowledge that “distribution shift” in this context is a **loosely defined experimental proxy**, not a measurable quantity.  
- Remove or rephrase claims suggesting that one split is inherently “harder” or “more out-of-distribution” than another.

This change would make the framing both more honest and better aligned with the current state of the meta-learning literature, where distribution shift remains an open and underspecified concept.

---

#### 3. **Superficial Analysis of Results**
Although Tables 1–2 clearly show SCARF’s gains under moderate shift (especially **PC** with **DeepELA** and **TransOptAS**), the paper provides no mechanistic explanation.  
There is no ablation, no embedding visualization, and no investigation into what SSL representations capture beyond supervised baselines.  
As a result, the findings remain *descriptive* rather than *insightful*.

---

#### 4. **Poor Presentation and Writing Quality**
- The prose is verbose, with long technical paragraphs (e.g., the affine recombination formula in §4) that obscure the core ideas.  
- Many sentences are ungrammatical or awkwardly phrased (“We need to mention that…”).  
- Figures and tables are densely formatted, lacking captions that interpret results.  
- The paper reads more like an extended technical report than an ICLR submission, and would benefit from significant language editing and tighter structure.

### Questions
1. Why does the PC split (Table 1, DeepELA baseline = 0.261) appear empirically *harder* than the supposed zero-shot P split (0.186)?  
2. What concrete real-world scenarios correspond to each of the four splits (I, R, PC, P)?  
3. Were any domain-specific SSL pretext tasks attempted (e.g., landscape perturbation or masked-feature prediction)?  
4. Can you visualize the learned embeddings (e.g., t-SNE/PCA) to show what SSL captures?  
5. How sensitive are your results to SSL hyperparameters (corruption rate, batch size, etc.)?  
6. Do the observed gains persist beyond the Cenikj et al. dataset, e.g., in higher-dimensional or noisy problems?

---

The paper delivers a careful empirical study but offers limited novelty and weak conceptual grounding. Its main insight, that generic SSL (SCARF, TabNet) can reduce labeling cost, is interesting yet unsurprising.  

Substantial re-framing, clearer writing, and domain-specific methodological advances would be needed for acceptance at ICLR.

### Soundness
3

### Presentation
1

### Contribution
2

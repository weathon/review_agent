# Hybrid Minority Oversampling via LLM-Generated Seeds and SMOTE Expansion

- Decision: Reject
- Scores: 2, 4, 2, 4

## Abstract
Class imbalance poses a persistent challenge in machine learning, as classifiers often underperform on the minority class when trained on skewed data. Oversampling is a common solution, with methods such as Synthetic Minority Oversampling Technique (SMOTE) offering efficiency but limited representational power, since they rely solely on existing data points. Recent approaches that employ large language models (LLMs) for oversampling overcome this limitation by generating diverse synthetic samples informed by contextual knowledge. However, LLM-only methods are computationally expensive and often impractical at scale. To bridge this gap, we propose LLM-SMOTE Hybrid (LSH), a method that integrates the strengths of both paradigms. In LSH, an LLM acts as a Scout that generates contextually meaningful seed samples for the minority class, while SMOTE serves as the Surveyor that efficiently expands these seeds to generate new samples. This design reduces reliance on repeated LLM calls while preserving diversity and scalability. Extensive experiments on 60 imbalanced tabular datasets, across multiple classifiers and resampling strategies, reveal that LSH consistently outperforms SMOTE and LLM in highly imbalanced datasets, demonstrating particular effectiveness in few-shot and zero-shot scenarios where SMOTE fails. Robustness analysis further shows that LSH achieves stable generalization with lower variance compared to other methods. Finally, LSH provides a practical trade-off, achieving competitive performance to LLM-based methods at substantially lower computational cost. These findings position LSH as an efficient, robust, and broadly applicable oversampling strategy for imbalanced learning problems.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes $\mathrm{LSH}$, a hybrid minority oversampling method that combines an $\mathrm{LLM}$ for generating minority “seed” samples with $\mathrm{SMOTE}$ for subsequent densification. Across $60$ datasets, $\mathrm{LSH}$ delivers small but consistent gains over $\mathrm{SMOTE}$ and an $\mathrm{LLM}$-only baseline, with clearer advantages under severe imbalance and in few-/zero-shot settings (where $\mathrm{SMOTE}$ alone fails). In efficiency, $\mathrm{LSH}$ invokes the $\mathrm{LLM}$ once to create seeds and then expands via $\mathrm{SMOTE}$ in near–constant time.

### Strengths
- Simple, practical hybrid: “$\mathrm{LLM}$ lays the landmarks $\rightarrow$ $\mathrm{SMOTE}$ densifies,” which is easier to operationalize than $\mathrm{LLM}$-only augmentation.  
- Broad evaluation: $60$ datasets, multiple classifiers and resampling ratios, with analyses including the Bayesian Sign Test ($\mathrm{BST}$).  
- Actionable insights: advantages grow with imbalance severity; clear runtime benefits.  
- Code available, which is a good step toward reproducibility.

### Weaknesses
- The paper appears to rely solely on GPT-4o-mini; it remains unclear whether the results are robust to the choice of model or would change across different LLMs.
- There is insufficient analysis of which dataset characteristics favor the proposed method (i.e., where and why it performs best).
- Datasets are listed only as “Data 1..60” without OpenML IDs/names, which makes reproduction practically impossible.
- SMOTE implementation details—
$k$, random seed, handling of mixed continuous/categorical features, etc, are missing. Many OpenML tabular datasets include categorical variables, yet preprocessing/encoding policies are not described.
- There is no audit of the realism of zero-shot LLM-generated minority data (e.g., distributional similarity, constraint violations, statistical distances) and no quality checks beyond downstream performance. This is especially important for domains like healthcare/fraud, where plausible-but-false samples can be harmful.
- The choice of the seed ratio (e.g., first generating to 
1:0.2 with the LLM) lacks justification and sensitivity analysis.
- The concrete thresholds defining “more/mid/less” imbalance are not specified.
- Multiple presentation issues remain; please see the items below and incorporate any fixes that are helpful.

### Questions
- Are there prior hybrid augmentation approaches ($\mathrm{LLM}$ $+$ classical/generative methods)? If so, please include them as baselines.  
- Could techniques other than $\mathrm{SMOTE}$ expand $\mathrm{LLM}$ seeds effectively (e.g., Borderline-$\mathrm{SMOTE}$, $\mathrm{ADASYN}$, variational/GAN-based)? Why is $\mathrm{SMOTE}$ preferable here?  
- Given that public $\mathrm{OpenML}$-style data may appear in $\mathrm{LLM}$ pretraining, can $\mathrm{LSH}$ work on data unlikely to be in the $\mathrm{LLM}$’s training set?  
- Please cite the sources for the $\mathrm{LLM}$-based (LM) baseline and specify the exact model/version used.  
- State at the start of the Experiments that the task is binary classification (currently first noted in the conclusion).  
- Please clarify the primary metric.  
- Precisely define how the “average margin” is computed and justify averaging margins across heterogeneous datasets.  
- For Table $3$, justify why datasets A–D were selected.  
- For Figure $4$, clarify “Resampling strategy (minority ratio target),” provide raw runtimes with $\mu \pm \sigma$, and explain why some segments appear flat (e.g., $0.3 \rightarrow 0.4$). Shouldn’t $\mathrm{LLM}$-only grow with the number of calls?  
- Provide $\mathrm{OpenML}$ IDs/names for all $60$ datasets.

(Other Comment)
- Global: ensure parentheses around citations where appropriate.  
- Add brief how-to-read guidance in figure/table captions.  
- Figure $1$: explain what “$21, 27$” denote; if 2D reduction uses $\mathrm{PCA}$, state it explicitly.  
- Figure $1$ legend: standardize capitalization (Major/Minor).  
- Line $252$: fix typo $\text{ANLAYSIS} \rightarrow \text{ANALYSIS}$.  
- Table $2$: avoid italics for $\text{LM–SM} / \text{LS–LM} / \text{LS–SM}$.  
- Figure $2$: report actual correlation coefficients (even if not significant) and clarify what each plot and the $y$-axis represent (e.g., average margin).  
- Table $3$ vs. Figure $6$: unify dataset identifiers (A–D vs. indices such as $21, 27$) and list the IDs/names for datasets $1\text{–}60$ in the appendix.  
- Classifier inconsistency: the main text lists Random Forest, whereas Appendix A.$3$ (Table $4$) lists kNN—please reconcile.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper introduces LLM-SMOTE Hybrid (LSH), a hybrid oversampling method designed to address class imbalance in tabular data by combining the complementary strengths of Large Language Models (LLMs) and the Synthetic Minority Oversampling Technique (SMOTE). 
The key contributions are: (1) extensive empirical evaluation on 60 datasets showing LSH's consistent superiority, especially in highly imbalanced, few-shot, and zero-shot scenarios where SMOTE fails; (2)demonstrated robustness with lower generalization variance.

### Strengths
1. The method exhibits greater robustness, defined by a lower variance in performance between validation and test sets.
2. Unlike LLM-only methods that require repeated, costly model invocations for each resampling ratio, LSH invokes the LLM only once for initial seed generation.

### Weaknesses
1. The paper lacks critical details necessary for full understanding, verification, and reproducibility. The most significant omission is the absence of a defined strategy for determining the number of seed samples the LLM ("Scout") should generate.
2. The paper's choice to use a single LLM (GPT-4o-mini) and a highly complex, custom prompting strategy limits the generalizability of its findings.

### Questions
1.	The number of seed samples is a fixed number? A percentage of the original minority class? Is it tuned per dataset?
2.	It is unclear if the reported benefits stem from the hybrid concept itself or are an artifact of this specific model and prompt engineering. The claim that LSH is "model-agnostic" remains unsubstantiated. A stronger validation would involve testing with other LLMs (e.g., open-source models like Llama, or other APIs like Claude) to demonstrate the general applicability of the Scout-Surveyor paradigm.
3.	The chosen LLM-only baseline is the same complex pipeline, a more rigorous and actionable comparison would be against a simpler, more direct LLM-based oversampling method to isolate the contribution of the hybrid design from the sophistication of the prompting technique. The gains of LSH over the LLM-only method are modest.

### Soundness
2

### Presentation
3

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
The paper proposes LSH, a hybrid oversampling pipeline where an LLM generates a few minority “seeds,” then SMOTE expands them. Experiments on 60 OpenML tabular datasets and a few extreme few/zero-shot settings report small average F1 gains over SMOTE and an LLM-only variant. The claimed advantage is practicality (one-time LLM cost, cheap SMOTE scaling) and robustness under severe imbalance.

### Strengths
1. Simple idea: “LLM seeds + SMOTE expansion” is potentially practical when minority data are scarce.


2. Wide dataset sweep: 60 datasets + few/zero-shot scenarios cover diverse conditions.

### Weaknesses
**1. Lack of clarity and self-containment.**

Several tables and figures are not clearly explained, which makes the paper difficult to follow and reproduce.
For example, in Table 6, the meanings of columns such as SM_v, SM_t, and SM_a are not explicitly defined.
In Table 3, the datasets labeled Data A/B/C/D are mentioned without description—readers cannot tell what domains these datasets belong to or what kinds of features they include.
Simple statistics like imbalance ratio are not sufficient; it would help to know whether the data are from finance, biology, or other domains.
Overall, the paper could be more self-contained, with clearer explanations of datasets, metrics, and table contents.

**2. Lack of justification and validity discussion**

The paper also lacks a discussion on when and why LLM-based seed generation is valid.
 LLMs could easily produce inconsistent or even impossible feature combinations, especially for structured or domain-specific data.
 It is unclear how such invalid outputs are detected or filtered.
 Moreover, not all domains are equally suitable for generation via LLMs—medical, financial, or safety-critical datasets may pose ethical or factual risks.
 Without addressing these issues, the methodological justification for using LLMs as data generators remains weak; the reader is left uncertain about the boundaries and reliability of this approach.

**3. Missing concrete examples of LLM-generated outputs**

The paper does not provide real examples or validation of the data generated by the LLM.
It is not clear how the authors ensure that the generated features are realistic or consistent.
What if the LLM produces implausible feature combinations? How is this handled in practice?

**4. Weak and unconvincing empirical performance.**

The reported performance gains are quite small—about +0.011 F1 on average compared to SMOTE and the LLM-only variant.
Bayesian sign tests show that results are mostly draws across 80–90% of datasets.
In several scenarios, especially with moderate or low imbalance, LSH performs similarly or even slightly worse than the baselines.
As a result, the experiments do not provide strong evidence that the proposed method offers clear benefits over existing approaches.


**5. Limited comparison baselines.**

The experimental comparison is limited to SMOTE and a simple LLM-only variant.
 The paper does not compare with tabular generation SOTAs (HARMONIC, Llmovertab).
Without these comparisons, it is difficult to understand how the proposed hybrid approach stands relative to state-of-the-art alternatives.

### Questions
Please refer to the weaknesses above.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes LLM-SMOTE Hybrid (LSH), a two-stage method that first uses a large language model (LLM) to generate a small set of seed samples for the minority class, and then expands them using SMOTE. The authors evaluate their approach on 60 imbalanced datasets across multiple classifiers, comparing it with standard SMOTE and LLM-only methods. Experimental results indicate that LSH achieves modest but consistent improvements over baseline methods, particularly in highly imbalanced settings, while maintaining computational efficiency relative to pure LLM-based approaches.

### Strengths
1. The proposed Scout and Surveyor method is intuitive and effectively addresses key limitations of traditional oversampling techniques and the high computational cost associated with LLM-only methods.

2. The evaluation on 60 imbalanced tabular datasets, spanning multiple classifiers, resampling strategies, and few-/zero-shot settings, provides rich and comprehensive empirical evidence.

### Weaknesses
1. Methodological imbalance: LSH makes a single LLM call to reach the 1:0.2 ratio and then scales using SMOTE, whereas the LLM baseline regenerates samples at every target ratio. This design advantage reduces stochastic variance and call overhead for LSH but not for the baseline, meaning that the observed performance gains may partly result from the setup rather than the hybrid mechanism itself.

2. Limited performance gain: The reported average improvement is only about one percentage point, and Bayesian Sign Test results indicate that most comparisons are statistical draws across datasets.

3. Single-model dependency: All minority seed samples are generated exclusively using GPT-4o-mini, with no exploration of alternative LLMs, decoding temperatures, or sampling schemes. This dependence raises questions about generalizability and potential model-specific biases.

### Questions
1. Evaluate the baseline under identical conditions: Re-run the LLM baseline using the same cumulative protocol as LSH: generate once to a 1:0.2 ratio, then scale up without repeated regeneration (e.g., by appending LLM samples or expanding fixed seeds with SMOTE) and report whether it still performs worse than LSH.

2. Analyze sensitivity to parameters: Vary the initial seed ratio (e.g., 1:0.1, 1:0.2, 1:0.3, 1:0.4) and explore different LLM models and decoding settings to assess the robustness of the results.

### Soundness
2

### Presentation
3

### Contribution
2

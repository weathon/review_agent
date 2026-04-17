# Limited Reference, Reliable Generation: Rule-Guided Tabular Data Generation with Dual-Granularity Filtering

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 2, 4

## Abstract
Synthetic tabular data generation is increasingly essential in machine learning, supporting downstream applications when real-world and high-quality tabular data is insufficient.
Existing tabular generation approaches, such as generative adversarial networks (GANs), diffusion models, and fine-tuned Large Language Models (LLMs), typically require sufficient reference data, limiting their effectiveness in domain-specific datasets with scarce records.
While prompt-based LLMs offer flexibility without parameter tuning, they often generate distributionally drifted data with localized redundancy, leading to degradation in downstream task performance.
To overcome these issues, we propose \textit{\textbf{ReFine}}, a framework that (i) derives symbolic \emph{if–then} rules from interpretable models and embeds them into prompts to explicitly guide the generation process toward the domain-specific distribution, and (ii) applies a dual-granularity filtering that suppresses over-sampling patterns and selectively refines rare but informative samples to reduce localized redundancy. 
Extensive experiments on various regression and classification benchmarks demonstrate that \textit{ReFine} consistently outperforms state-of-the-art methods, achieving up to \textbf{0.36} absolute improvement in $R^2$ for regression and \textbf{7.50\%} relative improvement in $F_1$ for classification tasks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposed a LLM powered tabular synthesis model to address the potential issue that LLM generated data maybe drifted by the LLM's pretrained knowledge, and to address the data scarcity issue. The paper contributes to proposing a rule-guided generation strategy powered by random foreset methods, and apply a dual granularity to filter out over sampling datapoints.

### Strengths
- The idea to leverage rules from tree based method to generate tabular data is novel. The rule based method is simple and intuitive that can directly affect the generated sample's correlation among features. 
- The paper has a well designed benchmark and considers the memory of LLM in the design. 
- The paper clearly states each component's contribution to the final synthetic generated data's quality. 
- The study has conducted thoroughly experiments on components in the supplementary, which makes the evaluation solid.

### Weaknesses
- The first concern about the methods is that the rules are based on a data driven method, i.e. random forest. However, the main focus of the paper is to develop a method for generating high quality data when data size is super small, e.g., in your main results, your tried sample size = 30. It is questionable that the rules generated based on such small sample are not biased and overfitting. It is a pivotal problem as the rules directly affect the generated samples in your method. 
- Although the paper has provided evidence and statements that the proposed method can help address the distribution drift problem, but only on the low-dimension projection space. It is still not clear that whether the methods can maintain the column level distribution. At least on the method level, we cannot see there is any injection of the real distribution into the generation. 
- The paper lacks discussion of the hyperparameter setting of the proposed methods. One key factor that may affects the generation quality and diversity is the temperature. But there is no such information and discussion about it. 
- The main results table 1 are based on XGBoost results only. I would suggest the authors to try different methods on each dataset (e.g. Tree-based, linear model, simple nn) and report the best results for each of them. As when data regime is low, it is very possible that the generated sample can have a better performance using some simple methods. 
- Following point 4, I have a concern about whether the benchmark is fair. As when you use a rule based method to generate sample, and then apply a tree-based method only to evaluate the downstream performance, it is easy to inflate the performance compared with other non-rule based methods.

### Questions
- Please see the weakness above.
- It is really interesting to see your methods start from a data-driven based method (random forest) which can definitely contributes more generative rules as sample size increases, but why your model csdoes not perform welll when the sample size goes up.

### Soundness
2

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
4

### Summary
This paper proposes a synthetic data generative model for low sample size. The model is built on LLMs with two additional components: 1) adding rules into the prompt, and 2) filtering data as a post hoc processing.

### Strengths
- The idea of generating rules from a random forest is novel
- This paper identifies the key challenge of localized redundancy.

### Weaknesses
- It's oversimplifying that the data distribution can be captured with some of the rules generated from a random forest. Rules generated from a random forest (particularly when the data size is very small, like 30-120) are not comprehensive and generalizable enough. The rule consolidation method based on LLM and its claim that the consolidated rules generalize beyond individual training instances is overstating.
- The contribution of this work is mainly by adding rules to the prompt (component I). This is rather incremental novelty. 
- This paper fails to cite a previous LLM-based synthetic data generation model to target small sample size data (https://arxiv.org/abs/2406.10521). This previous paper already outperforms the proposed work on the Adult data. 
- This paper tries to avoid data leakage by detecting LLM memorization. However, relying on just one tool’s prediction to determine whether the data is memorized or not is not sufficient to fully determine the impact of LLM memorization. 
- Apparently, the proposed model I+II was never the best-performing model with “unseen” data. With this experimental result, the contribution of component II (dual-granularity filtering) is questionable.

### Questions
See weakness

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper propose a strategy called ReFine which is aimed at reducing the distributional shift and localized redundancy caused when LLMs are used to generate synthetic tabular data. ReFine is a two-stage strategy: (1) extract rules from random forest based classifier to guide the LLM's generation and (2) apply a dual-level filtering mechanism to generated data.

### Strengths
+ the problem is relevant and interesting
+ the rule based method is an interesting way to try to guide the LLM to stay grounded
+ the method seems to produce good improvements on standard datasets

### Weaknesses
- the pipeline is very complex: First train a Random Forest, then run an LLM multiple times for self-consistent rule denoising, generate the data with another LLM, train another reference model (XGBoost) for filtering, and finally applying a multi-step filtering algorithm. All of these implies quite a few hyperparameters that need to be tuned. This raises concerns on robustness of the method.
- The proxy-based density estimation in the dual granularity filtering is built on a very small set of data points. This is a pretty coarse estimate of the density. A few outliers could disrupt the density completely. This error would then propagate to the rest of the filtering process that follows
- there are many heuristics in this method: like the ratioprune = A ln(G(p)) + B. This seems to come from curve fitting to their data. What happens if the model encounters dataset that does not look like this?
- the chunk level and instance level filtering rely on a refernce model, which itself has been trained on a very small dataset (30-60) samples. This could lead to overfitting, which means that the subsequent data filtering can be very biased
- the description of several parameters are very vague. What are the thresholds for support and depth in the "Rule generalization and denoising" step? There is no formal algorithm for this step, so it is not clear what is happening here.
- Table 2, Disease dataset: it looks like the prompt only method has 71% rule compliance, whereas the proposed mehod is only about 31%. Despite this, the downstream performance of the rule-guided approach is far superior. This seems to suggest that rule compliance may actually be a bad thing! Or the rules captured from a small dataset through the random forest approach maybe flawed!

### Questions
1. Can you discuss the error propagation effects of the multiple process and parameters involved in them, per the first point raised in the previous section of the review?
2. For the chunk level and instance level filtering, you use a logarithmic function for the pruning ratio. How confident are you that this specific functional form generalizes to datasets with distributions drastically different from those in your benchmarks? Have you explored simpler, linear relationships or more adaptive, non-parametric methods?
3. The reference model in component II is trained on a very small dataset. How have you ensured that the overfitting in that model does not propagate to the final filtered dataset?
4. Can you provide a formal algorithm for "Rule generalization and denoising" step? How are the thresholds for the support and depth determined?
5. Table 2 results are interesting and seems to suggest that strict adherence to the rules may not be necessary. How do you reconcile that with the central motivation of your paper?
6. Splitting the report between the "seen" and unseen" data is very useful. It appears that the proposed method doesn't win in the "seen" data, but does well in the "unseen" data. Does this mean that when the LLMs have strong prior models of the data, trying to force this rule may be determental to the over all performance?
7. Can you give more details on the computation and fine-tuning costs?

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
The paper proposes a new framework called ReFine, which improves the generation of synthetic tabular data in settings where only a small amount of real data is available. The paper identify two main problems in existing LLM-based approaches: distributional drift (where generated samples deviate from real data distributions); and localized redundancy, (where generated data cluster excessively around common patterns). ReFine addresses these by combining two techniques: (i) Rule-Guided Generation, which extracts symbolic if–then rules from tree-based models to guide LLMs toward domain-consistent generation; and (ii) Dual-Granularity Filtering, which uses density-based filtering to remove redundant samples while preserving rare but informative ones.

### Strengths
The paper is well motivated.

While most synthetic data generation methods require abundant reference data, this work focuses on the low-data regime case, which is both important, underexplored and challenging.

The idea of combining symbolic rules (from decision trees) with LLM-based generative prompting to correct for distributional drift is interesting.  

The paper provides the code implementing the method.

### Weaknesses
The paper’s experimental design and presentation could be strengthened in several key areas: 

Results are reported only as averages, without showing variability across the 10 experimental runs, which is essential given the very small training subsamples used (30–120 samples). Reporting uncertainty measures would improve statistical validity. 

The evaluation should also include additional criteria, especially data privacy, and explore larger datasets (up to 500–1000 samples) to better demonstrate the method’s generality. Comparisons against stronger baselines in the particular low data regime, such as TabPFN, would make the paper considerably stronger. 

Finally, a few methodological sections (particularly 3.3.2 and 3.2) are difficult to follow and would benefit from clearer notation, illustrative examples, and expanded explanations (possibly in the Appendix).

Next, I describe in more detail each one of these issues.

Main issues:

Although line 309 states that the experiments were performed using 10 distinct random train/test splits, Table 1 only reports average F1 and R² scores. The paper should include measures of variability—such as standard deviations or, ideally, full score distributions—across these replications. Given that the experiments use very small training subsamples (sizes 30, 60, 90, and 120), substantial variability across runs is expected. Reporting uncertainty around the F1 and R² estimates is therefore essential for assessing the statistical reliability of the results.

The paper should also evaluate ReFine using additional synthetic data evaluation metrics. At a minimum, the paper should evaluate data privacy. (The data contamination issues might show up more clearly for the privacy metrics.)  Evaluating data fidelity would also make the paper stronger (but is probably not essential, since fidelity metrics tend to correlate well with machine learning efficiency). 

I understand that the focus of the paper is on low data regime but, in my opinion, constraining the evaluations to training datasets containing 30 to 120 samples is a little too restrictive. (These ultra-low data regimes are not that common in practice.) I think the paper would be much more appealing if it could demonstrate the value of ReFine for datasets with up to 500 (or perhaps up to 1000) samples, which can still be quite challenging to other baseline generators, which usually require large training datasets for performing well (e.g., TabDDPM, TABSYN, GReaT, etc).

Also, the paper would benefit from the inclusion of other baselines, in particular, from comparisons against TabPFN [1, 2]. (Reference [2] describes how TabPFN can be used for synthetic data generation.) I think this is a particularly relevant baseline because it has been shown to achieve state-of-the-art performance in classification and regression tasks under low data regimes.  (Most of the current baselines evaluated in the paper are expected to perform poorly in the low data regime. Synthetic data generated by TabPFN might represent a stronger baseline in this setting.)

The paper provides no information regarding the tuning of the other baselines. 

The paper could also benefit from improvements in the presentation. While the Introduction is well written and motivates well the proposed approach, the Methodology Section could be improved. For instance, Section 3.2 would benefit from an illustrative example describing how the association rules are extracted from the random forest trees, and how they are generalized, denoised and then converted to structured prompts for the LLM (such an illustrative example could be included in the Appendix for the sake of space). 

I also found Section 3.3.2 difficult to follow. The paper needs to describe better the notation and elaborate more on the descriptions. For instance, what does $T$ represent in equation 2? I am guessing that $P_{M_t}(y_i | x_i) > 0.5 )$ represents the prediction of a classifier, but what about the regression task case where y_i is continuous? What do $\mu_{conf}$, $\mu_{uncert}$, $\sigma_{conf}$, and $\sigma_{uncert}$ represent in equation 4? (To be honest, I didn’t quite understand how the dual-granularity filtering is implemented. If explaining it in more detail will take space, I would suggest the paper add a detailed explanation in the Appendix, leaving only a high-level description in the main text.)

Additional comments/questions:

Why in Sections 4.3 and 4.4 the paper uses only 3 datasets? Presenting the analyses for all 8 datasets would make the results more compelling. 

I enjoyed that the paper used dataset contamination tests for evaluating potential dataset contamination in the adopted benchmark datasets.  

In line 239 the paper states “let DCR be the mixed-type distance from (Borisov et al., 2023)”. But what do you mean by mixed-type distance? Is it the way that Borisov et al handle categorical variables in their computation of DCR? If that is the case, it would be better to explicitly describe this procedure rather than referencing it indirectly.

In lines 96 to 98 the paper states that one of its main contributions was to identify the two key challenges to LLM-based synthetic data generation (i.e., distributional drift and localized redundancy). But, didn’t the references cited in the paragraphs starting in line 71 and ending in line 84 already pointed out these issues?

Typos:

Line 43: “which used” should be “which is used”

Line 112: “Many work” should be “Many works”

Line 122: “Layers Stoian et al. (2024b)” should be “Layers (Stoian et al., 2024b)”

Line 124: “Stoian & Giunchiglia (2025)” should be “(Stoian & Giunchiglia, 2025)”

Lines 237 to 238: “denote the N = …”  something is missing here

Line 238: “from (Borisov et al., 2023)” should be “from Borisov et al. (2023)”

References

[1] Hollmann et al. (2023). TabPFN: a transformer that solves small tabular classification problems in a second. ICLR 2023.

[2] Hollmann et al. (2025). Accurate predictions on small data with a tabular foundation model. Nature, 637, 319-326.



Initial recommendation:

Overall, I find it difficult to accurately assess the performance of ReFine relative to the baselines in its current form, and therefore I am inclined to recommend rejection at this stage. Nevertheless, the paper presents an interesting idea and addresses a meaningful and timely problem. I would be happy to revisit my assessment if the authors can clarify the issues outlined above and provide stronger empirical evidence supporting ReFine during the discussion period.

### Questions
See above.

### Soundness
2

### Presentation
2

### Contribution
2

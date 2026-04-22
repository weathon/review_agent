# Discovering Hierarchical Latent Capabilities of Language Models via Causal Representation Learning

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Deriving actionable insights from language model evaluations to guide post-training is a central challenge, hampered by complex confounding effects and the prohibitive cost of controlled studies. In this paper, we propose a causal representation learning framework to uncover a hierarchy of LLM capabilities purely from publicly available observational data. Drawing insights from recent factor analysis \citep{ruan2024observational}, we model the observed benchmark performance as a linear transformation of a few latent capability factors. Crucially, we model these latent factors as causally interrelated after appropriately controlling for the base model as a common confounder. Applying this approach to a comprehensive dataset encompassing over 1500 models evaluated across six benchmarks from the Open LLM Leaderboard, we identify a concise three-node linear causal structure that reliably explains the observed performance variations. The hierarchy that we discover --- from general problem-solving to instruction-following, and finally to mathematical reasoning --- is more than an interpretation; it provides a causal roadmap for post-training. We demonstrate through targeted fine-tuning experiments that interventions on parent capabilities propagate to child capabilities as predicted by our model, offering validated, actionable guidance for practitioners.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The authors introduce an $\alpha$-inexact structural causal model (SCM) for LLM benchmark scores. A novel identification procedure that improves upon sensitivity to the inexactness, as compared to prior work, is presented. Utilizing their model and algorithm, the authors produce three latent LLM ``skill" factors and their causal relationship.

### Strengths
* The SCM model allows the authors to uncover an interesting causal relationship between generalist, instruction following, and mathematical skills in LLMs
* The HCA algorithm comes with theoretical guarantees in the exact and inexact SCM settings.

### Weaknesses
* Compared to prior work (Polo et al.) the model is fit and studied on a relatively smaller number of benchmarks/models. The authors could consider utilizing LLM leaderboard v1 and v2.
* Besides the exactness assumptionsThe SCM is identical to that proposed in Jin & Syrgkanis; Zhang et al. Furthermore, I am unable to quantify how HCA is an improvement over the methods in these works. Proposition . shows that if ICA is exact then HCA recovers an $\alpha$-exact SCM. The authors state that prior methods are sensitive to (in)exactness but never explain further (in particular I would be curious if the exact ICA assumption affects this at all).
* I am not sure that the discussion in section 2 is a totally correct assessment of prior work on latent LLM skills. For example, if I understand (Polo et al.) the key assumption is $x \approx Gz(s,t)$, with s, t model size and token count. To me this implicitly allows the latent space to vary based on the model family, which is the main argument for the model the authors introduce. 

**Small writing issues**

* Line 320: ''varisbles''
* Line 265 $z^{(k, i)}$ appears on both sides of the equation, im not sure how this should read

### Questions
* How do the results extend to the benchmarks and models present in llm leaderboard v1/v2?
* Can the authors provide a more thorough comparison between HCA and prior algorithms in the $\alpha$-inexact setting?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces Hierarchical Component Analysis (HCA), a novel framework inspired by causal representation learning, designed to uncover the hierarchical causal relationships among latent capabilities of LLMs using solely publicly available observational benchmark data. Addressing the challenges of confounding effects and the high cost of controlled studies in LLM evaluation, the authors model observed performance as a linear transformation of a few latent capabilities, crucially assuming these capabilities form a DAG after controlling for the base model as a confounder. The study identifies a consistent three-node linear causal hierarchy: general problem-solving influences instruction-following, which in turn influences mathematical reasoning.

### Strengths
1. The paper pioneers the application of causal representation learning using only observational data to understand LLM capabilities, moving beyond correlational scaling laws or factor analysis. 
2. The discovered hierarchy (general problem-solving → instruction-following → math reasoning) provides a potentially practical "causal roadmap" for prioritizing post-training efforts, suggesting interventions on parent capabilities can benefit child capabilities.

### Weaknesses
1. The method relies on strong assumptions like the linearity of capability-performance mapping, the DAG structure of capabilities, and non-Gaussian noise for ICA, which might not fully hold in reality . The impact of violations (e.g., non-linearity) is not deeply explored. 

2. Concerns were raised about the discovered graph's robustness to the choice of base models and benchmarks. While Appendix H provides some sensitivity analysis, the filtering step and potential finite-sample errors impacting ICA/HCA need consideration. The complexity of Algorithm 2's permutation search could be an issue for higher dimensions, though feasible for d=3. 

3. In fact, I was one of the reviewers of this paper submitting to Neurips. At that time, the chair and other reviewers were more concerned about issues around assumption realism, experimental scope/evaluation breadth, and practical utility. However, when I reread the author's submission this time, I did not find that the author had added justification to address these issues, which led to my support for this paper declining.

### Questions
1. How sensitive is the recovered hierarchy ($z_1 \rightarrow z_2 \rightarrow z_3$) to potential non-linear relationships between capabilities or non-linear mappings to benchmark scores? How is the "inexactness" (quantified by MIC) distributed across different causal links or domains? 

2. How likely is this specific 3-node hierarchy to generalize to other base model families (beyond Llama-3/Qwen2.5), different model scales, or architectures like MoEs ? Does the exclusion of models with different PC subspaces (Fig 2c) limit the generality of the findings?

3. Beyond correlation with benchmarks (BBH, IFEval, MATH), what is the deeper semantic meaning of $z_1, z_2, z_3$? Is $z_1$ purely general reasoning or confounded with knowledge? How distinct are these factors, especially given the ambiguities allowed by the identification theory (Theorem 1)?

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces a novel causal representation learning (CRL) framework, Hierarchical Component Analysis (HCA), to discover the hierarchical structure of latent capabilities in LLMs using only observational benchmark data from the Open LLM Leaderboard. By controlling for the base model as a common confounder and analyzing latent factors (Hypothesis 1) through a linear Structural Causal Model (SCM) (Hypothesis 2), HCA recovers a three-node hierarchy: General Problem-Solving ($z_1$) $\rightarrow$ Instruction-Following ($z_2$) $\rightarrow$ Advanced Mathematical Reasoning ($z_3$). This validated causal roadmap offers practitioners clear, actionable insights for strategic post-training interventions.

### Strengths
1. Causal Discovery of Hierarchical Structure: The HCA method moves beyond correlation-based analysis (like PCA) to establish a causally-directed hierarchy among capabilities, providing a validated roadmap for improving LLMs.
2. Robustness by Controlling Confounding: The analysis rigorously accounts for performance heterogeneity across different base models, ensuring the discovered invariant causal structure is robust and not merely a result of pre-training confounding effects.
3. Actionable Insights for Optimization: The hierarchy provides practical guidance, such as demonstrating that enhancing the intermediate Instruction-Following capability ($z_2$) is an effective precursor for boosting the specialized Mathematical Reasoning skill ($z_3$).

### Weaknesses
1. The analysis is constrained by the six benchmarks available, resulting in only three coarse-grained capabilities. A richer set of benchmarks might reveal a more complex hierarchy . 
2. While SFT experiments support the $z_2 \rightarrow z_3$ link, the broader causal claims rely partly on observational correlations and interpretation. The ATE analysis acknowledges unverifiable ignorability assumptions. There could be the potential circularity in naming factors based on correlated benchmarks and then using those benchmarks for validation.

### Questions
1. How exactly would practitioners map real-world fine-tuning strategies (e.g., specific datasets, RLHF techniques) onto interventions on the latent factors $z_1, z_2, z_3$ beyond the examples given (IFEval SFT for $z_2$)? How can one target $z_3$ (math) more directly without catastrophic forgetting, as noted in the paper?
2. What is the justification for the specific definition of MIC in Definition 2? Could alternative CRL algorithms designed for multi-node interventions  yield different results? How sensitive is the result to the specifics of the ICA implementation or the permutation alignment process in HCA (Algorithm 2) ?
3. The paper links $z_1$ to pre-training compute (Fig 8). How exactly does pre-training compute act as a confounder influencing all capabilities, and how does HCA effectively control for this beyond simple regression adjustment?

### Soundness
2

### Presentation
2

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
The paper proposes Hierarchical Component Analysis (HCA), a causal representation learning framework to uncover hierarchical latent capabilities of LLMs using only observational benchmark data (specifically, the Open LLM Leaderboard). The key idea is to model benchmark performance as a linear transformation of latent capability factors, with a shared base model variable treated as a confounder.

### Strengths
1. This work provides a novel causal framing by introducing causal representation learning to interpret model capability hierarchies.

2. The method is based on well-motivated derivation of HCA with identifiable conditions and theoretical grounding.

3. Reliance on public leaderboard datasets ensures reproducibility and relevance for community benchmarking.

### Weaknesses
1. The hierarchy is inferred from only six benchmarks, so the results may not generalize.

2. No robustness study on the hyperparameters. 

3. Table 2 is not referenced in the main text.

### Questions
1. The paper briefly mentions targeted fine-tuning, but the description in the main text lacks sufficient detail. Can the authors elaborate more or point me to related texts?

2. Is it possible to compare performance of the proposed method to that of other latent-factor discovery methods?

### Soundness
3

### Presentation
3

### Contribution
3

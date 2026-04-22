# MPP: MODEL POWER PARITY FOR REASONING LARGE LANGUAGE MODEL COMPARISON ON CROSS-DOMAIN BENCHMARKS

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 6, 6, 2

## Abstract
The ability of Large Language Models is usually evaluated by their average performance scores on various benchmarks. However, this way of comparison does not consider the difficulty difference between benchmarks, and the average score also lacks interpretability. Inspired by the International Comparison Program (ICP), we introduce model power parity (MPP), a principled framework that adapts the idea of purchasing power parity (PPP) to LLM evaluation. MPP performs multilateral, direct comparisons among models and benchmarks without assuming a common difficulty scale. Extensive experiments show that the MPP framework provides reasonable evaluation results and brings new insight into the relation between models and benchmarks. We evaluate over 30 open-source reasoning large language models on 8 prevalent reasoning benchmarks. Our results reveal three key insights: (1) The idea of PPP suits well for building a ranking framework evaluating models' ability and benchmark difficulty. (2) MPP manages to infer missing performance scores matching actual inference results within about 0.5 to 3 points on top-2 benchmarks. (3) MPP helps distinguish whether a benchmark is suitable for evaluating a given model family. We believe that MPP offers a fresh, economically grounded perspective on equitable LLM capability assessment and will facilitate more reliable model selection and benchmark building.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper offers a new method inspired by economics to evaluate language model abilities and benchmark suitability. It adapts Purchasing Power Parity (PPP) and the Country Product Dummy (CPD) method to essentially infer a normalized or equalized version of the raw aggregate score on a benchmark. These methods are used to evaluate many models on several benchmarks. The ranking of models is preserved, the raw score can be recovered from the PPP, and the suitability of a benchmark for evaluating a new model can be determined.

### Strengths
The idea behind this paper is novel. The authors clearly expended considerable effort evaluating and analysing data from some 30 models on 8 reasoning tasks. The fact that raw score can be recovered from the inferred PPP is a useful result, but not unexpected given the simplicity of the regression models.

### Weaknesses
It is not clear how the PPP is computed with the CPD method. As far as I can tell, CPD is a regression with binary dummy input variables encoding country and product, with the aim to predict the price of the product in that country. But of course, each price is perfectly identifiable just from the matrix of dummy variables. The authors do not describe how to overcome this identification problem. As far as I can tell (not being an economist myself), there are some methods for handling this (omitting the reference categories, etc.), but the authors do not describe these methods in detail. This is essential for their method to be taken up by the community. Indeed, throughout, the methodology is described only at a very high-level, making implementation based on the text almost impossible without independent knowledge of these methods. The authors should more clearly describe their methodology step-by-step.

While the idea is innovative, I do wonder what the added utility of computing the MPP is over just reporting the per-benchmark aggregate scores. Both methods gloss over the crucial instance-level variation contained in these benchmarks, which is far more informative with respect to the capabilities of the model than aggregating (Burnell et al., 2023; Raji et al., 2021). The MPP method just adds another, albeit slightly more sophisticated, way to compute the average performance on large datasets. AI Evaluation has moved on these days to the examination of the individual demands of specific items and measuring how that impacts performance, which would give us a much better sense of what models can and cannot do. In that vein, the authors should review and compare to the literature on Item Response Theory and psychometrics for AI evaluation: Polo et al. (2024), Jo and Wilson (2025), Burden et al. (2023), Kipnis et al. (2024), Wang et al. (2023).

In general, the paper is vague and unclear. The notation is quite inconsistent throughout. For instance, PPP is variously indexed by two countries/models or by one (it should properly be with two always), the formatting in equations (4) and (7) is poor, and the authors do not define or justify certain steps, such as how they normalize raw scores or why scores are log-transformed (is it for numerical stability or monotonicity?). I also dislike that the important definitions relating to PPP and CPD are all included in the appendix, demanding the reader to flick back and forth to gain an understanding of the methods and results.


Burden, J., Voudouris, K., Burnell, R., Rutar, D., Cheke, L., & Hernández-Orallo, J. (2023). Inferring capabilities from task performance with bayesian triangulation. arXiv preprint arXiv:2309.11975.

Burnell, R., Schellaert, W., Burden, J., Ullman, T. D., Martinez-Plumed, F., Tenenbaum, J. B., ... & Hernandez-Orallo, J. (2023). Rethink reporting of evaluation results in AI. Science, 380(6641), 136-138.

Jo, N., & Wilson, A. (2025). What Does Your Benchmark Really Measure? A Framework for Robust Inference of AI Capabilities. arXiv preprint arXiv:2509.19590.

Kipnis, A., Voudouris, K., Buschoff, L. M. S., & Schulz, E. (2024). metabench--A Sparse Benchmark of Reasoning and Knowledge in Large Language Models. arXiv preprint arXiv:2407.12844.

Polo, F. M., Weber, L., Choshen, L., Sun, Y., Xu, G., & Yurochkin, M. (2024). tinyBenchmarks: evaluating LLMs with fewer examples. arXiv preprint arXiv:2402.14992.

Raji, I. D., Bender, E. M., Paullada, A., Denton, E., & Hanna, A. (2021). AI and the everything in the whole wide world benchmark. arXiv preprint arXiv:2111.15366.

Wang, X., Jiang, L., Hernandez-Orallo, J., Stillwell, D., Sun, L., Luo, F., & Xie, X. (2023). Evaluating general-purpose AI with psychometrics. arXiv preprint arXiv:2310.16379.

### Questions
1. It seems that one big component of this paper is to simply see if we can recover the ranking and raw scores from the MPP. Why should a practitioner bother computing the MPP at all, if all we end up caring about is the (ranking of the) raw scores on the benchmark?
2. What method is used to avoid collinearity in the CPD regression?
3. How does the TPP method compare to more sophisticated methods like Item Response Theory, which allow the computation of difficulty on a per-item, rather than per-benchmark, basis?
4. Is recovery of raw scores within 3 percentage points good? Kipnis et al. (2025) appear to achieve recover with much lower errors (0.5-1.24%)
5. The title and much of the discussion focuses on reasoning models, but this method seems agnostic to whether they are reasoning models or not. Indeed, you could use it to compare deep RL agents or computer vision systems, mutatis mutandi. Why have the authors focused on reasoning LLMs and reasoning benchmarks?

### Soundness
2

### Presentation
1

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
This paper proposes Model Power Parity (MPP) — a novel evaluation framework for comparing reasoning abilities of large language models (LLMs) across benchmarks of differing difficulty. Inspired by Purchasing Power Parity (PPP) in economics, MPP models the relationship between model capability and benchmark difficulty without assuming a shared difficulty scale.

### Strengths
1, Adapting PPP theory from economics to model–benchmark evaluation is an original and elegant analogy, connecting economic parity with model capability parity.
2, MPP does not assume a single global difficulty scale, unlike average or normalized benchmarks; it enables pairwise and transitive comparisons, enhancing interpretability.
3, Clear derivation using CPD-MTD and Jevons-GEKS-MTD formulations ensures reproducibility and links to established econometric theory.
4, Covers >30 open-source reasoning LLMs and 8 major benchmarks, providing a credible empirical foundation.

### Weaknesses
1， The CPD model assumes linear additive separability of “model ability” and “task difficulty.” Real LLM–benchmark interactions may be nonlinear and interaction-heavy, undermining this assumption.
2， No confidence intervals, variance estimates, or robustness checks are provided for MPP/TPP values — critical for establishing reliability.
3， Since both model and task parameters are estimated from the same performance matrix, there’s risk of self-normalization bias, especially with sparse or correlated data.

### Questions
-

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
A new evaluation framework called Model Power Parity (MPP) has been proposed. This framework considers models as "currency" and benchmark tests as "commodities", using a multilateral comparison method to estimate both the model's ability and the difficulty of the benchmark, without the need to preset a uniform difficulty scale. The author conducted experiments on over 30 open-source inference language models and 8 mainstream inference benchmarks to verify the effectiveness of MPP in model ranking, performance prediction, and benchmark selection.

### Strengths
1: Strong innovation: Introducing PPP ideas from economics into LLM evaluation, with a novel perspective and inspiring methods.

2: Solid technology and rigorous methodology: Two specific calculation methods (CPD-MTD and JEVSON-GEKS-MTD) are provided, both based on standard statistical methods in the PPP field, with a solid theoretical foundation. Reasonable preprocessing (log Odds conversion) was performed on the scores, handling different indicators such as probability and perplexity. The experimental design is systematic, covering multiple dimensions such as ranking, prediction, and correlation analysis.

3: The experiment is comprehensive, covering multiple mainstream open-source models (such as Qwen3, DeepSeek-R1, Phi-4, etc.) and multiple inference benchmarks, with a large amount of data and convincing results.

### Weaknesses
1: Limited benchmark scope: Only 8 inference class benchmarks were used, covering limited domains and task types, which may affect the framework's generalization ability.

2: The model assumption is relatively simple: the current model assumption is that performance is a linear sum of model capability and task difficulty, without considering more complex interaction effects (such as model task adaptability).

3: Lack of comparison with existing methods: No systematic comparison with existing benchmark difficulty assessment methods.

4: The theoretical explanation for why PPP is effective is slightly weak: the paper provides a good description of how to do it, but the theoretical connections and assumptions behind why the analogy of PPP is effective in LLM evaluation can be further elaborated.

5: The readability of some charts is average: for example, the explanations of Figure 3 and Figure 5 are not clear enough, and the reader's understanding cost is high.

### Questions
1: Is the MPP framework suitable for non-inference tasks such as generation, dialogue, and multimodality, and does it need to be adjusted?

2: Is there a non-linear relationship between modeling ability and task difficulty, and are there plans to introduce more complex modeling methods?

3: Is there a plan to quantitatively compare MPP with existing benchmark difficulty assessment methods to further highlight the advantages of MPP?

4: Are there any models or task types that are more prone to significant errors in performance prediction, and are there any patterns to follow?

5: Have you considered extending MPP to dynamic evaluation scenarios?

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
The authors propose an analogy, within the ICP (and PPP) economics model, between currencies and LLMs and goods as benchmarks. In other words, LLMs are the currency that can buy benchmarks. In this sense, one can measure the PPP of an LLM to compare it to others. The higher the PPP, the more powerful a model is at reasoning.

### Strengths
I applaud the authors for trying to propose an unorthodox way to measure the performance of an LLM and compare it to others using well-established methods from macroeconomics.
There may be some value in this approach if expanded and actually grounded to concepts where the analogy makes sense.
With that being said, please read the weaknesses.

### Weaknesses
The framework is very confusing, and it is not clear, from the very beginning, what the actual analogy the authors are posing. If the models are the currencies, who owns and how are the currencies spent? The paper jumps, right after proposing this analogy without giving insights, into the mathematical framework, which is poorly written and exposed. For example, what is p in Eq. 1?

To make it clear, my main concern is with the entire analogy of the paper: a good is something that provides a utility and is scarce. I cannot see the analogy with benchmarks, which are not scarce (I cannot have 100 MATH500, but I can have infinitely many copies of MATH500).
When it comes to currencies, they have 3 key functions: medium of exchange (for trading goods), store of value (they measure the price of things) and unit of account (for savings).
I cannot really see the analogy with LLMs. In what sense do they serve as the currency for benchmarks? Currencies are also fungible, but LLMs are not: I cannot consider 100 tokens of Llama 4 on MATH500 as 100 tokens of GPT-4o on GSM8K; they are different things, and the analogy here is misleading on every aspect, without even mentioning the other aspects of a currency (portable, divisible), which do not hold in the framework.

In terms of results, Table 2 has many missing values (the caption claims every model has a predicted and real score).

The entire paper is built around a framework that, if I am not wrong, could be stated as “the more benchmarks a model can solve in a category and in general, the better it is at reasoning”. But that is precisely what people do in benchmarking, without the necessity to use this framework. One can measure (and researchers do, check HELM by Stanford) the performance on a dataset, as well as the (weighted) average performance on a class/category (e.g., math).

### Questions
Q1. What does it mean for a model (the currency in the PPP framework) to buy a dataset? 
I ask the reviewers to clarify the analogy between LLMs, datasets, and the PPP framework; for me, that is not clear at all, and the analogy misplaced.
I elaborate further on my concerns in the section Weaknesses.

Q2. Table 2 has missing values (the authors say each result consists of two numbers, but only a few have both). Is it intended to be so?

Q3. Can the authors elaborate on the difference and advantages of this framework compared to standard benchmarking (works like HELM or more recent papers on benchmarking models per-class/category of datasets)?

### Soundness
1

### Presentation
1

### Contribution
2

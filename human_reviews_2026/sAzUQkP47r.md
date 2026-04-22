# OpenEstimate: Evaluating LLMs on Reasoning Under Uncertainty with Real-World Data

- Avg Score: 3.50
- Decision: Accept (Poster)
- Scores: 4, 2, 4, 4

## Abstract
Real-world settings where language models (LMs) are deployed --- in domains spanning healthcare, finance, and other forms of knowledge work --- require models to grapple with incomplete information and reason under uncertainty. Yet most LM evaluations focus on problems with well-defined answers and success criteria. This gap exists in part because natural problems involving uncertainty are difficult to construct: given that LMs have access to most of the same knowledge as humans, it is non-trivial to design questions for which LMs will struggle to produce correct answers. As a result, LM performance on reasoning under uncertainty remains poorly characterized. To address this gap, we introduce \textsc{OpenEstimate}, an extensible, multi-domain benchmark for evaluating LMs on probabilistic estimation tasks that require models to synthesize knowledge from pretraining and express predictions as Bayesian priors. We assess these priors for accuracy and calibration. Across six frontier models, we find that LM-elicited priors are worth the equivalent of about five samples from the underlying data distribution, and that posteriors computed using LM priors tend to be more accurate than those computed using a naive prior. At the same time, the relationship between model accuracy and confidence is weak across the board, indicating the value of developing new methods to improve calibration. The \textsc{OpenEstimate} benchmark thus offers a challenging evaluation for frontier LMs and a platform for developing models that are better at probabilistic estimation and reasoning under uncertainty.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces a new benchmark designed to evaluate the ability of LLMs to perform probabilistic estimation, as opposed to merely extracting point estimates from such foundation models. The **OpenEstimate** benchmark involves asking LLMs to generate parametric Bayesian priors, specifically parametrized as Normal or Beta distributions for a set of of "derived variables" from real-world datasets in various fields.

The evaluation is twofold: firstly **accuracy**, which uses a normalized mean absolute error to verify whether the location of the mode of the distribution is correct, while accounting for the probability mass of the 'ground truth' distribution at that point.
Secondly, the authors measure **calibration**, a kind of empirical earth-mover's distance between the observed frequencies and the predicted distribution.

Experiments with several recent general-purpose LLMs suggest no systematic patterns by any particular LLM on any particular task, but that most models tend to be overconfident and do worse than 5 draws from the empirical distribution.

### Strengths
The paper tackles an important and topical problem of evaluating LLM quantification of uncertainty, which is an emerging and until recently underexplored area of research. Generating full, parametrized priors is a relatively novel idea compared to many other works in the literature, which rely on less 'statistical' approaches, such as simple point estimates or series of questionnaires.

The proposal of "derived variables" is a good idea, as it brings the benchmark closer to measuring the utility of priors on realistic tasks, rather than estimating superficial summaries.

If the code and benchmark datasets are made available, this could be a useful resource for other researchers and practitioners. [NB: the code/data was not visible to reviewers at this stage.]

The motivation for the paper is well articulated and overall the writing and presentation of the article are clear. The figure labels are readable (with caveats, see below).

### Weaknesses
The most significant weakness of the paper is the restriction to normal and beta distributions for continuous variables and proportions, respectively. While this is noted in the limitations section, it undermines the validity of the benchmark. For any continuous variable with a skewed distribution or heavy tails---which is indeed likely to include those in one or more of the chosen benchmark datasets---the prior will be fundamentally mis-specified. The use of the mode for MAE should be better justified; while it coincides with the median/median of the normal distribution, this is not true more generally.

On a related note, the beta distribution has more than one parametrization, and so (without seeing the prompt, which is missing from the paper), it will be impossible to disentangle the performance of the LLM from its 'understanding' of the choice of parametrization. Indeed, the first line of the appendix hints at an issue that might be related, and which may have been resolved using 'function calling' capabilities to constrain the format of the output.

The choice of **expected calibration error** (ECE), based on four coarse bins seems like reinventing the wheel. The construction of this metric is ad-hoc, lossy and non-standard for this type of analysis. A graphical method such as a QQ plot would have provided a far more rigorous and informative assessment of calibration. Splitting data into quarters (incorrectly referred to in the paper as 'quartiles', but quartiles/quantiles are *points*, not *intervals*; a common mistake) seems arbitrary and discards information within those bins. The number of models being compared is small enough to allow for a visual method, and if not then other, more granular numerical metrics are available, as well as standard statistical tests for comparison of empirical and expected distributions.

Poor visualization. Bar charts with error bars, sometimes called "dynamite plunger plots", should never be used, and hide the true distribution of performance across the different tasks. Box plots or simple dots and error bars would be better. See Drummond & Vowler (2011; doi:10.1111/j.1476-5381.2011.01251.x).

The distinction between 'reasoning' and 'non-reasoning' models in the ablation study needs to be more clearly defined and justified. What is the basis for classifying, e.g. `GPT-4o` as 'non-reasoning'?

Some related works need to be mentioned in the literature review. This is not the first work to propose eliciting parametric Bayesian priors from LLMs, see for example Selby et al (2025; doi:10.1002/sta4.70054), which explores the problem of evaluating the quality of LLM priors on real-world datasets. Data augmentation approaches (i.e. sampling pseudo-observations from the LLM and then looking at the empirical distribution) should also get a mention: see Huynh et al (2025; https://openreview.net/forum?id=2Q3gFNbpAr).

Finally, there are one or two minor typos: e.g. "empricial" at bottom of page 6.

### Questions
1. The central claim is that LLMs appear to be 'overconfident'. How can we be sure that this is not an artifact of the experimental design? The restriction to Gaussian priors for heavy-tailed or skewed data (e.g. funding) is likely to enforce model mis-specification.

2. Why split the data into four coarse bins to calculate ECE instead of standard, more informative methods for assessing calibration, such as quantile--quantile plots?

3. What is the definition of 'reasoning' and 'non-reasoning'. Can you justify placing powerful GPT models in the 'non-reasoning' category?

4. What is the justification for using the *mode* of the distribution for the MAE calculation?

5. What is the formula or citation used for the correction term from the Jeffreys prior used in the scale-adjusted log-probability metric?

6. How does this work compare with others on eliciting Bayesian priors from LLMs (see above)?

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
3

### Summary
The paper introduces OpenEstimate, a benchmark for testing large language models on probabilistic estimation tasks. Instead of giving point estimates, models must express beliefs as Bayesian priors (Gaussian or Beta distributions) for real-world quantities drawn from datasets in labor economics, finance, and public health. Results show that even advanced models like GPT-4 and LLaMA 3 are poorly calibrated and overconfident, performing no better than simple statistical baselines built from a few real samples.

### Strengths
1. The paper fills a clear gap by evaluating LLMs’ ability to reason under uncertainty, focusing on probabilistic estimation rather than deterministic prediction.

2. The study spans multiple domains and models, using clear metrics for accuracy and calibration and comparing results against interpretable statistical baselines.

### Weaknesses
1. The paper mainly reports performance differences without probing why models fail to calibrate uncertainty.
2. Restricting distributions to Gaussian and Beta forms limits realism for complex or multimodal uncertainty.
3. The benchmark is not used to improve models or study uncertainty learning, missing the opportunity to explore how training on such data could enhance self-calibration.
4. Only zero-shot inference is tested. There’s no attempt to improve performance through structured prompting, self-consistency, or retrieval, which would make the benchmark more actionable.

### Questions
Do you expect that supervised or reinforcement learning using OpenEstimate could improve uncertainty calibration?

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
The paper builds a benchmark dataset OPENESTIMATE, which evaluates LLMs' ability to provide a prior probability distribution over uncertain variables. The variables are constructed from real-world datasets on labor economics, private markets, and human health. LLMs are prompted to specify a Gaussian or Beta distribution for the variables. Results show that state-of-the-art LLMs perform poorly on this benchmark; for example, the LLM-generated priors are often less accurate than posteriors formed from 5 real samples.

### Strengths
1. The paper studies a practically important problem, namely LLMs' ability to give a good prior over random quantities in the real world.
2. The paper gives a reasonable approach to generate derived variables that are unlikely to have been explicitly documented in LLMs' pretraining data, by conditioning on attributes that change the target statistic sufficiently.
3. The performance metrics for accuracy and calibration are reasonable and intuitive.

### Weaknesses
1. My major concern is that the paper uses only the Gaussian and Beta distributions, which may be highly misspecified probabilistic models for the real-world data. This raises the question of whether the LLMs' poor performance is true reflection of its probabilistic reasoning skills, or an artifact of being forced to specify an inappropriate model. I think an important evaluation would be to let an LLM propose a distribution form on its own, and see if the accuracy and calibration improve.
2. The derived variables are constructed by conditioning on randomly sampled attributes. While it can effectively avoid data leakage, I wonder if the derived variables are always practically relevant, e.g., ones that a real-world analyst would actually ever estimate.

### Questions
1. In Figure 2b, how can the expected calibration error (ECE) be as large as 10? By definition, the maximum value that ECE can take is $|1-0.25| / 4 = 0.175$. Are the numbers supposed to be *percentage* ECEs?
2. In constructing the derived variables, the paper conditions only on attributes that alters the target statistic by at least 5%. Would this create overly difficult estimation targets with high variability, and thus make the benchmark more difficult than common real-world tasks?

### Soundness
3

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
4

### Summary
This paper introduces OPENESTIMATE, a benchmark designed to evaluate LLMs on their ability to generate well-calibrated Bayesian priors for real-world quantities in domains such as labor economics, finance, and public health. Models are asked to express uncertainty as Gaussian or Beta distributions, and their outputs are assessed for accuracy and calibration against empirical data.

Experiments across six frontier LLMs (including GPT-4 and Qwen3-235B) show that current models are generally inaccurate and overconfident, often performing no better than using five random samples from real data. While some reasoning models (e.g., o3-mini, o4-mini) better capture probability mass near true values, calibration remains poor and domain-dependent. Ablation studies reveal that elicitation method (how uncertainty is prompted) affects results more than temperature or system prompt settings.

Overall, the study finds that LLMs’ probabilistic reasoning is weak but not random, showing structured, domain-aware uncertainty that could serve as a foundation for improving AI systems that reason under uncertainty.

### Strengths
1. Novel Benchmark Design: The paper introduces OPENESTIMATE, a first-of-its-kind benchmark that evaluates LLMs on probabilistic estimation using real-world tabular data. Unlike prior work focused on deterministic or forecasting tasks, it systematically measures both accuracy and calibration of Bayesian priors.


2. Comprehensive Empirical Evaluation: It provides a cross-domain assessment (labor economics, finance, and public health) across multiple frontier models (GPT-4, Llama 3.1, o3/o4-mini, Qwen3). The inclusion of strong statistical baselines gives the results clear interpretability and robustness.

### Weaknesses
1. The task definition appears to lack clear motivation. Under a zero-shot setting, the LLM has no prior exposure to the specific datasets used in the benchmark, making it unclear how it could produce meaningful estimations. Moreover, it is debatable whether the model’s outputs can truly be considered “priors,” since they effectively reflect the posterior knowledge embedded during pretraining. The way an LLM is trained or fine-tuned likely has a substantial influence on these results, and this issue should be explicitly acknowledged and discussed.

2. The paper focuses on three domains: labor economics, private markets, and public health, but it is unclear why these specific areas were chosen over other possible domains. The authors should clarify the rationale for selecting datasets exclusively from the social sciences and explain why natural sciences, medicine, or engineering data were not considered. In addition, it would be helpful to articulate what types of real-world reasoning or uncertainty these three domains are intended to represent, and what general conclusions can meaningfully be drawn from results confined to these areas.

### Questions
1. The authors could provide a discussion of the differences across models, such as distinctions between open- and closed-source models, and how model size may affect performance on the benchmark.

2. Clarify the motivation and reasonableness of the zero-shot task setting, and whether the model outputs can truly be interpreted as Bayesian priors.

3. Can you justify the choice of focusing only on three social-science domains or expand to other fields.

### Soundness
3

### Presentation
2

### Contribution
2

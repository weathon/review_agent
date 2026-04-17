# Bias Similarity Measurement: A Black-Box Audit of Fairness Across LLMs

- Decision: Accept (Poster)
- Scores: 6, 2, 6, 4

## Abstract
Large Language Models (LLMs) reproduce social biases, yet prevailing evaluations score models in isolation, obscuring how biases persist across families and releases. We introduce Bias Similarity Measurement (BSM), which treats fairness as a relational property between models, unifying scalar, distributional, behavioral, and representational signals into a single similarity space. Evaluating 30 LLMs on 1M+ prompts, we find that instruction tuning primarily enforces abstention rather than altering internal representations; small models gain little accuracy and can become less fair under forced choice; and in our evaluation setting, open-weight models can match or exceed proprietary systems. Family signatures diverge: Gemma favors refusal, LLaMA 3.1 approaches neutrality with fewer refusals, and converges toward abstention-heavy behavior overall. Counterintuitively, Gemma 3 Instruct matches GPT-4--level fairness at far lower cost, whereas Gemini’s heavy abstention suppresses utility. Beyond these findings, BSM offers an auditing workflow for procurement, regression testing, and lineage screening, and extends naturally to code and multilingual settings. Our results reframe fairness not as isolated scores but as comparative bias similarity, enabling systematic auditing of LLM ecosystems. Code is available at https://github.com/HyejunJeong/bias_llm.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces a novel framework called Bias Similarity Measurement (BSM), designed to evaluate the fairness of large language models (LLMs) from a relational rather than an isolated perspective. BSM integrates multi-dimensional signals including scalar performance, distribution similarity, behavioral patterns, and representational similarity (e.g., CKA) to construct a unified bias similarity space. The authors evaluated 30 LLMs from four families—LLaMA, Gemma, GPT, and Gemini—on over one million prompts.

### Strengths
1. Innovative Perspectives and Methodologies: The core contribution of this paper lies in shifting fairness evaluation from traditional isolated scalar scores to relational comparisons between models. The BSM framework offers a novel perspective by employing similarity analysis to trace how biases propagate, inherit, and converge across different model families and versions—an approach rarely seen in existing research.
2. Comprehensiveness and Multi-dimensional Integration of Evaluation: The study was not confined to a single metric but systematically integrated multiple measures at the behavioral, distributional, and representational levels. This multi-faceted assessment helps uncover biases across different dimensions. For instance, the research found that instruction fine-tuning led to behavioral changes (high dropout rates), while internal representations (CKA similarity) remained highly preserved, highlighting the discrepancy between superficial fairness and structural bias.
3. Scale of the empirical study: Although limited in the number of families covered, the evaluation of 30 models and over 1 million prompts indeed represents a substantial scale in terms of model quantity and data volume. This provides rich empirical evidence for observing bias patterns across models of different scales and types (open-source vs. closed-source, base vs. instruction-tuned).
4. Practical Audit Workflow: The paper not only proposes a methodology but also demonstrates the potential of BSM in practical applications through case studies (such as model procurement), integrating it with specific scenarios like model selection and version monitoring, thereby enhancing the practical value of the research.

### Weaknesses
1. The BSM evaluation process involves multiple similarity functions and preprocessing steps, making the pipeline extremely complex. Although the code has been open-sourced, the cost and barriers for other researchers to reproduce the entire evaluation pipeline remain high. The paper lacks discussion on computational resource consumption, API costs (especially for closed-source models), and the potential for simplifying operational steps, which may limit its widespread adoption as a benchmark tool.
2. The coverage of model families is limited: The authors claim to have conducted the "largest-scale study," but the models evaluated come from only four families (LLaMA, Gemma, GPT, Gemini). Currently, the LLM ecosystem is flourishing (with models like Claude, Qwen, etc.), so the generalizability of the conclusions indeed warrants further scrutiny. Different families vary significantly in pretraining data, architectures, and alignment strategies, and the applicability of BSM conclusions to other families requires additional validation.
3. Dataset bias and experimental rigor: I identified a significant imbalance in sample sizes across different bias dimensions in the BBQ dataset (e.g., dimensions like Religion, Disability, and Sexual_orientation had far fewer samples than Gender). The paper mentions "~5000k" samples per dimension in Section 4.2, which may not align with the actual BBQ dataset (likely a typo, possibly meant to be ~5000 samples, but even then, the distribution across dimensions remains uneven). The authors did not address how this sample imbalance might bias evaluation results (such as cosine distance and histograms), nor did they explain whether weighting or normalization was applied to mitigate its impact, which somewhat compromises the rigor of the experiment.
4. The reliability and generalizability of the conclusions: Due to the limitations in model family coverage and dataset biases mentioned above, certain conclusions in the paper (such as "Gemma converges to an abstention strategy" and "open-source models can match closed-source models") should be viewed with caution. It remains questionable whether these conclusions hold true across a broader range of models and more balanced datasets.

### Questions
See the above weakness.

### Soundness
3

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
This paper introduces a conceptual framework called "bias similarity measurement", which aims to compare models across multiple axes, termed bias signatures. These include categorical (accuracy, bias scores), distributional (histograms, cosine distance), behavioral (abstention flips), and representational (centered kernel alignment) dimensions. Rather than asking `Is a model biased?`, the authors reframe the question as `Which models behave similarly with respect to bias, and why?` They claim that this reframing supports practical auditing tasks such as balancing fairness–utility trade-offs under abstention thresholds, tracking model shifts across versions, and identifying suspiciously similar bias profiles in proprietary systems. The paper includes extensive empirical analysis over 1M+ prompts, multiple model families, fine-tuning conditions, datasets, and demographic axes.

### Strengths
* *Comprehensive empirical scope*: The authors conduct meticulous and large-scale evaluations, including a wide range of model families, fine-tuning settings, and bias datasets. This level of breadth is impressive and rare in fairness auditing work.

* *Conceptual clarity*: The decomposition of bias analysis into categorical, distributional, behavioral, and representational dimensions provides a useful organizing lens. This taxonomy helps structure a complex and often fragmented area of research.

* *Thoughtful consideration of abstention*: The distinction between useful abstention (reflecting epistemic uncertainty) and biased abstention (reflecting social or representational bias) is an insightful and nuanced addition to the fairness discourse.

### Weaknesses
* Limited novelty: Beyond the proposed four-part taxonomy, the framework largely repurposes existing metrics (e.g., BBQ bias scores, cosine distance, abstention rates, CKA). While the structure is neat, the underlying analyses could arguably be achieved with standard bias-evaluation tools. The contribution may therefore be more organizational than methodological.

* Utility evaluation is limited: The paper reports accuracy only on the disambiguated BBQ benchmark. Without comparison to broader benchmarks of utility or reasoning (e.g., MMLU, etc), it is difficult to fully assess the fairness–utility trade-off that the framework claims to illuminate.

* Questionable experimental setup (UNQOVER): The experiment comparing ambiguous (BBQ) and under-defined (UNQOVER) prompts for bias measurement may not be conceptually valid. Forcing model responses on inherently ambiguous questions may not reflect realistic use cases; abstention in such cases is arguably the correct behavior rather than reflecting an underlying bias artifact.

* Framing of bias "similarity": The title suggests a unified similarity metric between models, but the actual analyses compare models across separate metrics rather than integrating them into a single similarity measure. This may lead to mislead reader expectations.


Minor comments: 
* Clarify CKA earlier: The acronym and its purpose appear several times before being defined (page 6). I encourage authors to introduce it earlier, with a brief explanation and appropriate citation.

* Ambiguity in terminology: Terms like “small,” “medium,” and “large” models are used without clear quantitative definition or justification.

### Questions
* Utility evaluation scope: Beyond accuracy on the disambiguated BBQ benchmark, could the authors evaluate model utility using broader or standard benchmarks (e.g., MMLU, HELM, BIG-Bench) to strengthen the fairness–utility analysis?

* Experimental validity (UNQOVER setup): How do the authors justify forced responses under under-defined prompts as a meaningful setting for bias auditing?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a bias similarity measure that allows bias trends within models to be compared. The aim is to indicate the ways in which biases in models are related rather than to quantify how biased a particular model is. Bias similarity is broken down into a number of dimensions that capture different aspects of fairness.

### Strengths
- A good number of models are tested to give a sense of bias similarity as a function of model family, model size, open models vs. closed models , and so on.
- The components of the similarity score cover the range of dimensions along which a model might be biased (e.g., accuracy for functionality, cosine similarity for relative preferences, CKA for structural similarities in the representations, and so on). This helps to give an indication of why biased behaviors in models might be similar.
- The analysis considers a good number of attributes along which a model might be biased (gender, age, race, and so on).

### Weaknesses
- The main weakness is that bias is considered only with respect to individual traits. Intersectional bias is especially problematic.
- The paper sometimes provides an observation, e.g., a generational bias trend where older models retain stereotypes whereas new models do not. There is no insight into learning why this is the case though. Sometimes this may not be possible, e.g. models might be closed, but in other instances it may be possible. How do we learn to prevent these observations.
- In the discussions about abstention vs. representation — abstentions will occur with different frequencies for different groups. This is not considered. The paper Bias Runs Deep: Implicit Reasoning Biases in Persona-Assigned LLMs by Gupta et al. comes to mind when I think of this.

### Questions
- Define CKA at first use on line 073.
- Line 258: replace non-“unknown” with known?
- The caption for Figure 2 could better highlight the main takeaways. The Figure is very busy and it takes a lot to read.
- Line 338: the main body refers to dimensions related to physical appearance and sexual orientation being problematic, but I do not see this in Figure 2, especially for sexual orientation.
- Line 349: to avoid “the appearance of neutrality” because of abstentions, is it possible to force the model to make a decision in BBQ or measure the degree of bias from the samples on which the model does not abstain.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Bias Similarity Measurement (BSM), a framework that reframes fairness evaluation from isolated scoring to relational comparison across models. The authors evaluate 30 LLMs on 1M+ prompts, integrating scalar, distributional, behavioral, and representational metrics into unified bias signatures. Key findings: instruction tuning primarily enforces abstention rather than altering representations (CKA > 0.94), small models can degrade with tuning, and open models can match proprietary systems.

Novelty. Individual components (CKA [1], cosine distance) are established, and large-scale bias studies exist [2]. The novelty lies in integration of multi-level signals for fairness-specific relational analysis. UNK Flip Rate and bias signatures seem genuinely new. However, representation preservation during tuning confirms existing hypotheses [3], and similarity analysis is well-documented [4].

Significance. Strong practical value for auditing and procurement. The demonstration that abstention masks rather than resolves bias is important. Small model degradation findings are significant for deployment trends. However, lack of debiasing solutions, reliance on older benchmarks when newer alternatives exist [5,6], and overlap with general frameworks [7] limit impact.

[1] Kornblith et al. (2019). Similarity of neural network representations revisited. ICML.

[2] Kumar et al. (2024). Investigating implicit bias in large language models: A large-scale study of over 50 LLMs. arXiv:2410.12864.

[3] Wolf et al. (2023). Fundamental limitations of alignment in large language models. arXiv preprint.

[4] Klabunde et al. (2025). Similarity of neural network models: A survey. ACM Computing Surveys, 57(9).

[5] Abhishek et al. (2025). BEATS: Bias evaluation and assessment test suite. arXiv:2503.24310.

[6] Fan et al. (2025). FairMT-Bench: Benchmarking fairness for multi-turn dialogue. ICLR.

[7] Dekoninck et al. (2025). Polyrating: A cost-effective and bias-aware rating system for LLM evaluation. ICLR.

[8] Chaudhary et al. (2025). Certifying counterfactual bias in LLMs. ICLR.

[9] Zhao et al. (2018). Gender bias in coreference resolution: Evaluation and debiasing methods. NAACL.

### Strengths
- Novel Integration: First systematic fairness-specific relational framework combining behavioral, distributional, and representational signals with practical auditing workflows.
- Abstention Analysis: Important distinction between fairness-through-caution and fairness-through-representation, demonstrating that high abstention conceals directional bias.
- Counterintuitive Findings: Discovery that small models (LLaMA 3.2 3B, Gemma 3 4B) worsen with tuning; family-specific strategies (Gemma's refusal vs. LLaMA 3.1's neutrality).
- Comprehensive Scale: Largest pairwise fairness comparison with detailed dimension-specific results providing valuable community resource.
- Black-Box Compatibility: Works with API-only models, enabling proprietary system audits.

### Weaknesses
- Limited Component Novelty: Applies established techniques [1,4]; large-scale comparative studies exist [2]; confirms known representation preservation [3] rather than discovering new phenomena.
- No Solutions: Purely diagnostic framework; provides no debiasing methods unlike comparable work [8,9].
- Dataset and Coverage Limitations: English-only; limited to 4 dimensions; high failure rates (up to 85%) in open-ended generation; missing newer multi-modal [5] and multi-turn [6] benchmarks.
- Incomplete Literature: Missing model genealogy, bias propagation in transfer learning, and comparative auditing literature. No comparison with overlapping framework Polyrating [7].
- Interpretive Constraints: Cross-vendor comparisons observational only; overstates claims ("largest study" conflates pairwise with absolute scale [2]); lineage detection unvalidated.

### Questions
1) Validation: Can you validate bias signatures on known model derivatives? What similarity thresholds reliably detect inheritance?
2) Framework Comparison: How does BSM compare with Polyrating [7] for fairness evaluation? Where does fairness-specific analysis add value?
3) Actionable Insights: Can BSM guide debiasing strategies? Why do small models degrade with tuning while larger models improve?
4) Generalization: Do bias signatures remain consistent across benchmarks (BBQ vs. UnQover)? How stable are signatures over time for models receiving updates?

### Soundness
3

### Presentation
2

### Contribution
3

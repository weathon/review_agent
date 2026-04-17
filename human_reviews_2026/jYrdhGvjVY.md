# The interplay between domain specialization and model size

- Decision: Reject
- Scores: 2, 6, 2

## Abstract
Scaling laws for language models have often focused on finding the optimal model size and token count for training from scratch. However, achieving this optimal balance requires significant compute resources due to the extensive data demands when training models from randomly-initialized weights. Continued pretraining offers a cost-effective alternative, leveraging the compute investment from pretrained models to incorporate new knowledge without requiring extensive new data. Recent findings suggest that data quality influences constants in scaling laws, thereby altering the optimal parameter-token allocation ratio. Building on this insight, we investigate the interplay between domain specialization and model size during continued pretraining under compute-constrained scenarios. Our goal is to identify an optimal training regime for this scenario and detect patterns in this interplay that can be generalized across different model sizes and domains. To compare general and specialized training, we filtered a web-based dataset to extract data from three domains: legal, medical, and accounting. We pretrained models with 1.5B, 3B, 7B, and 14B parameters on both the unfiltered and filtered datasets, then evaluated their performance on domain-specific exams. Results show that as model size increases, specialized models outperform general models while requiring less training compute. Additionally, their growing compute efficiency leads to reduced forgetting of previously learned knowledge.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper aims to explore the patterns in domain specialization (i.e., continuing pretraining on corpora in a specific domain). The authors propose the following hypothesis: 
“Domain specialization in Transformer LMs yields increased performance and compute efficiency over general training, as model size increases under a compute-constrained scenario”.
To verify this hypothesis, the authors conduct continued pre-training on LLMs with different sizes from 1.5B to 14B on general-domain corpora and domain-specific corpora (a subset of the general-domain corpus extracted by a classifier) to acquire several “general models” and “specified models”.
Experimental results show that (1) Specialized model exhibits lower perplexity than general models on domain-specific test sets; (2) Specialized models achieve minimum PPL with fewer tokens than general models, and this difference on the number of tokens required increases with model size; and (3) Specialized models are prone to have less forgetting on general knowledge as they are further trained on fewer tokens.

### Strengths
(1)  This paper aims to explore compute-optimal continued pretraining, a significant research direction.

### Weaknesses
(1)  Writing is needed to improve, as it is difficult to understand:
a)	The key concept of this paper, “domain specialization”, is not well explained in this paper. The biggest confusion I had when reading this article was in which aspects a good "domain specification" should be quantified? Is it by comparing the lowest ppl achieved under the same computational budget (i.e., 6ND)?
b)	In line 60 the authors first propose a hypothesis “larger models exhibit greater capacity to retain learned knowledge”, while later in line 68 they deny their own hypothesis, and in line 97 they make another claim “domain specialization in LLMs yields increased performance and compute efficiency over general training as model size increases”. It took me a long time to understand that the real claim this paper is making is the last one.
c)	In section 4.1 the authors list two contradictory conclusions: “the gap” increases with the model size in medical and accounting domains, while it does not increase in the legal domain, without further exploration. I also don’t quite get how the “gap increase” conclusion is reached: though in figure 2 the slopes of the fitted blue and orange lines are slightly different, there doesn't seem to be such a pattern if you only look at the four data points.
(2)  The evaluation data size is too small to draw convincing conclusions. The domain-specific evaluation data only contains 199, 368, and 478 data points. As a result, all experiment results are based solely on the PPL of a few hundred tokens (1 token per data point). To verify a universally applicable scaling law of LLMs you should at least add some experiments with larger evaluation data and in high-resource languages.
(3)  Some of the results are already well-known in the community. For example, (1) training on domain-specific corpora usually yields lower PPL on the corresponding domain than mixing additional unrelated text, and (2) larger models usually need fewer tokens to reach the same performance (in “Scaling Laws for Neural Language Models“ paper). Although experiment (2) was conducted by training from scratch, intuitively this conclusion can also easily be extended to continued training, especially if the data distribution of pre-training data and continued training differs significantly (mainly English+Chinese v.s. Portuguese).
(4)  I wonder the applicability of the conclusions in this paper. For instance, what advice can this paper provide if I want to train a specialized model for the legal domain within a given FLOPS budget? Should I prioritize scaling the model parameters, the amount of training data, or put more effort into data cleaning? I think this is a more important question in continued pretraining at a compute-constrained scenario.

### Questions
Firstly, refer to weakness 2 and 4.
In addition, could you add some figures with curves showing how the test loss changes as training progresses, similar to Figure 2 in the "Scaling Laws for Neural Language Models" paper? This would be beneficial for displaying more information to intuitively compare the differences between various models and data.

### Soundness
1

### Presentation
2

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
The paper investigates the relationship between model size and domain specialization in the context of continued pretraining under compute-constrained scenarios. The authors aim to determine whether filtering training data for domain-specific content yields better performance than general-domain continued pretraining when computational resources are limited.
Key Contributions
Empirical Study on Domain Specialization vs. Model Scale:
The paper systematically evaluates how model size (1.5B, 3B, 7B, 14B parameters) interacts with domain-specialized continued pretraining across three domains: legal, medical, and accounting.
Controlled Training Setup:
All models undergo one epoch of continued pretraining on either:

A large, diverse web-based corpus (general),
Or filtered subsets focused on specific domains.
Evaluation via Domain-Specific Exams:
Performance is measured using standardized multiple-choice exams tailored to each domain (e.g., medical licensing-style questions), enabling precise assessment of knowledge retention and acquisition.
Discovery of a Power-Law Relationship:
Under compute constraints, specialized training follows a power-law improvement over general training.
Improved Compute Efficiency and Reduced Forgetting:
Specialized models achieve lower perplexity faster and exhibit less forgetting of previously acquired knowledge, suggesting more efficient use of compute during continued pretraining.

### Strengths
This paper presents a compelling and timely investigation into the interplay between model size and domain specialization during continued pretraining under compute-constrained settings. Its strengths span multiple dimensions—originality, quality, clarity, and significance—and collectively position it as a valuable contribution to the field of efficient language model adaptation.

1. Originality: High – Novel Problem Formulation with Fresh Insights
The paper’s originality lies in its novel framing of an understudied trade-off: how domain specialization interacts with model scale when compute is limited. While both scaling laws (e.g., Kaplan et al., Hoffmann et al.) and domain adaptation (e.g., Med-PaLM, LlamaLaw) have been widely studied, this work uniquely connects them through the lens of continued pretraining efficiency.
Key aspects of originality include:
Challenging a prevailing intuition: The hypothesis that larger models inherently benefit less from specialization due to greater capacity is intuitive but previously untested. The fact that the authors empirically disprove this assumption—showing instead that larger models gain more from specialization—is a striking and counterintuitive result.
Creative experimental design: By filtering a web-scale dataset using neural topic models to isolate legal, medical, and accounting domains, the authors construct a clean, controlled comparison between general and specialized data regimes—a methodologically sound approach rarely seen in prior specialization studies.
Power-law observation in specialization gains: Identifying a power-law relationship between model size and the relative benefit of domain filtering introduces a new quantitative lens for analyzing specialization strategies—an idea ripe for future theoretical exploration.
Thus, while none of the individual components (scaling laws, domain filtering, continued pretraining) are new, their synthesis into a coherent framework for evaluating data strategy under constraints represents a creative and original contribution.

2. Quality: Strong – Rigorous Methodology and Reproducible Design
The paper demonstrates high scientific rigor across several dimensions:
Controlled variables: Training all models for exactly one epoch on comparable data volumes ensures a fair evaluation under fixed compute budgets—an essential condition for meaningful conclusions about efficiency. 
Model scale diversity: Including four distinct sizes (1.5B to 14B) enables robust trend analysis and supports claims about scalability.
Evaluation on standardized exams: Use of domain-specific multiple-choice benchmarks (e.g., analogous to USMLE or bar exams) provides objective, real-world-relevant metrics, avoiding proxy measures like perplexity alone.
Consistent patterns across domains: The observed trends hold across three disparate domains (legal, medical, accounting), suggesting the findings are not domain-specific artifacts but potentially generalizable principles.
Inclusion of forgetting analysis: Measuring retention of prior knowledge via perplexity on general text adds depth to the evaluation and strengthens claims about reduced catastrophic forgetting in specialized training.
While the paper does not release models or datasets (understandable given anonymity), the methodology is described clearly enough to allow replication by well-resourced labs.

3. Clarity: Excellent – Well-Structured and Accessible Presentation
Despite tackling a complex intersection of scaling, specialization, and transfer learning, the paper is exceptionally clear and logically structured:
Abstract and introduction set up the problem effectively, motivating both practical and theoretical interest in the question: "Should we filter data when continuing to pretrain?"
Hypothesis is explicitly stated, then cleanly refuted by evidence—this narrative arc enhances readability and impact.
Figures (e.g., Figure 1) appear designed to highlight key takeaways (power-law scaling advantage), though full visual access is limited in the blind submission format.
Technical terms (e.g., compute-equivalent replay, re-warming) are briefly contextualized, making the work accessible to a broad NLP audience.
Writing is concise and free of unnecessary jargon; logical flow moves naturally from motivation → hypothesis → experiment → results → implications.
Overall, the clarity significantly enhances the paper's persuasiveness and accessibility.

4. Significance: Broad and Practical Impact Across Communities
The significance of this work extends beyond a single application or architecture—it speaks to fundamental questions about how best to use finite compute resources in the era of large pretrained models.
Its impact can be felt across multiple communities:
Industry & Applied AI: Organizations fine-tuning LLMs for verticals (healthcare, law, finance) will find direct guidance: investing in high-quality, domain-filtered data may yield better returns than further scaling general data.
Efficient ML Research: The paper contributes to the growing body of work on "effective compute" — showing that smarter data selection can outperform brute-force scaling. This aligns with recent interests in data curation, weighting, and pruning (e.g., MinT, LLMLingua).
Scaling Law Theory: By demonstrating that data quality and relevance alter the optimal parameter–token ratio, the work suggests extensions to existing scaling laws to incorporate domain alignment factors—a promising direction for theory.
Sustainability & Equity: Reducing compute needs for high-performance specialized models lowers barriers to entry and reduces environmental costs, supporting more sustainable and inclusive AI development.

### Weaknesses
While the paper presents a compelling narrative with strong experimental design and significant implications, several weaknesses—though not fatal—limit the robustness, generalizability, and depth of its conclusions. Below is a detailed critique focused on specific shortcomings, supported by concrete suggestions for improvement.

1. Narrow Definition of "Specialization": Risk of Confounding Data Quality with Domain Focus
The core claim—that domain specialization improves performance under compute constraints—relies on comparing models trained on:
A broad web corpus (general),
Versus topic-filtered subsets (legal, medical, accounting).
However, filtering by topic may simultaneously improve data quality (e.g., removing low-signal text like social media posts or clickbait), which independently affects scaling laws (Tay et al., 2022; Hoffmann et al., 2022 showed data quality alters scaling constants). 

2. Missing Analysis of Cross-Domain Generalization and Negative Transfer
The paper evaluates models only on in-domain exams, showing that specialized models outperform general ones. However, it omits evaluation on out-of-domain tasks, making it impossible to assess the trade-off between specialization gain and generality loss—a central concern in transfer learning.
For example:
Did the medical-specialized 14B model degrade on commonsense reasoning (e.g., HellaSwag)?
How much did legal specialization hurt performance on programming (e.g., HumanEval)?

### Questions
see above.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper investigates the relationship between domain specialization and model size in continued pretraining of Transformer LMs under compute-constrained scenarios. Using filtered datasets from ClueWeb 2022, the authors train Qwen2.5-based models (1.5B–14B) on both general and specialized (legal, medical, accounting) data.

### Strengths
Compute Efficiency and Forgetting Analysis:Introducing the SGER (Specialized-to-General Efficiency Ratio) metric
The Related Work section is comprehensive and well-situated in recent scaling law and domain-adaptation literature

### Weaknesses
Plots and cross-suite comparisons appear to use the test suite to pick the “minimum perplexity” checkpoint, then report those results. This is classic evaluation leakage; that minimum should be picked on a validation split that is disjoint from the reported test metrics


Power-law claims (and the SGER vs size trend) are fit on four points without confidence intervals, fit method details, or ablations. This risks over-interpreting noise. (You do acknowledge this in Limitations, but the paper still leans heavily on the regressions.) Consider reporting CI bands / bootstrapped SEs and fit diagnostics.

The paper follows Xia et al. and uses PPL on the correct answer letter; however, accuracy (and calibrated accuracy like log-prob of correct minus best distractor) is what many readers expect for MCQA.

### Questions
Did you use any data from the evaluation exams in the topic-classifier training or instruction prompts used for few-shot examples? Please clarify contamination controls.


How many random seeds per run?

### Soundness
2

### Presentation
2

### Contribution
3

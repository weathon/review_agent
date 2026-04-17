# Measuring Meta-Cultural Competency: A Spectral Framework for LLM Knowledge Structures

- Decision: Reject
- Scores: 2, 6, 4, 8

## Abstract
Most cultural evaluation frameworks for Large Language Models (LLMs) compare model outputs with ground-truth answers, capturing mainly factual awareness. This overlooks whether models internalize broader cultural structures and pluralism. In this paper, we introduce a spectral-analysis-based framework to uncover large-scale structural patterns in models' cultural knowledge. We test eight LLMs of different sizes across nine cultural domains (food, religion, language, etc.) spanning 170 countries, comparing their learned structures with human data. Results show that instruction-tuned LLMs align more closely with human patterns than older models like GPT-2 and GPT-J. However, model size is not always an advantage, and performance asymptotes: Llama-8B and Gemma-2B perform as well or better than their larger-sized counterparts: Llama-70B and Gemma-9B. These findings differ from model rankings on existing probing-based cultural benchmarks, showing that our method captures a distinct aspect of cultural competency. Furthermore, initial simulation-based experiments demonstrate that compared to traditional metrics of cultural awareness, the proposed spectral metric is better able to predict a model's ability to serve a user from an unfamiliar background.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes evaluating cultural knowledge in LLMs not as isolated facts but as a broader understanding of relationships across cultures, which the authors term "macrostructures". The approach extracts country-level probability distributions over domain items, builds country×country adjacency matrices (cosine similarity), and summarises those matrices using two spectral metrics: ER and SR. The metric pair is used to categorise domains into four macrostructure types. They evaluate several open models across nine domains and 170 countries and present some interesting findings. They also test these metrics on a recipe-recommendation downstream case study judged by an LLM simulator and a small human validation set. The results claim these spectral metrics capture a distinct, useful notion of “variational awareness” and reveal domain-specific failures.

### Strengths
1. To the best of my knowledge, framing cultural evaluation via macrostructures and linking it to CCT theory is a novel cross-disciplinary idea. 
2. The framework is formally stated (matrix construction, ER and SR definitions) and yields compact, interpretable summaries (ER = effective dimensionality; SR = λ1/λ2). Makes it easy for other researchers to reproduce and build upon. 
3. The finding that macro-structural competency may plateau with scale challenges common assumptions and suggests new evaluation methods.

### Weaknesses
1. The core concept of "macrostructures" remains abstract and lacks intuition, particularly in the introduction. The paper would benefit significantly from a clear visualization (e.g., a simplified t-SNE or UMAP of a "good" vs. "bad" country-similarity matrix) to help the reader see what a "high-consensus" (low ER) or "high-diversity" (high ER) structure actually looks like. 
2. Section 2.1 seems unnecessary since there is repetition with the introduction. There are no clear definitions, and it is written like a literature review section. The third component of negotiation strategies is not used in the rest of the paper. The section reads like a philosophical justification rather than a rigorous definition section, which leaves the reader with little clarity about how these ideas translate into measurable quantities.
3. L216-L226 provides formal definitions (ER/SR) to measure "variational awareness," but this link feels assumed rather than demonstrated. The connection between, for example, a high Effective Rank and a model's awareness of cultural variation is not explicitly proven. The logic needs to be more thoroughly established.
4. The spectral measures are computed from matrices built using very large, and in some cases scraped, candidate lists (e.g., >3700 items per Table 1). The resulting "macrostructure" could be highly sensitive to the noise, bias, or composition of these lists. The paper needs an ablation study to test this.
5. The method for assigning the four-class labels (LH, HL, etc.) relies on binarizing the ER and SR values using their medians. This threshold is dataset-dependent and potentially arbitrary.
6. Many key claims are presented verbally without sufficient statistical backing. For example, the model rankings (Section 4.2), the performance "plateau" claim, and the correlations in the downstream task (Section 5.2) all require more statistical tests.

### Questions
1. Could you provide (or release) the exact candidate lists (CSVs) for each domain and country, and the scraping/cleaning pipeline? This would allow reproducibility and judge whether noisy items influenced ER/SR.
2. To support W4, I recommend the following ablations: compute ER/SR after (i) restricting items to top-k globally frequent items, (ii) removing low-frequency/noisy entries, and (iii) using curated vs. scraped lists to quantify sensitivity. Explicitly report how item selection changed ER/SR for a few representative domains (eg, food, language, currency).
3. To support W5, a few suggestions are (a) show the continuous distributions of ER and SR values so readers can judge separability, (b) test robustness to alternate thresholds (e.g., quartiles or k-means clustering), and (c) use ER and SR as continuous predictors for the downstream task metrics (APR/INT) instead of relying solely on the discretized labels.
4. Please add confidence intervals for the macro-F1 scores, statistical tests (e.g., t-tests) for differences between model pairs, and report effect sizes for the correlations to substantiate these claims. Per-domain confusion matrices would also clarify the evaluation.

### Soundness
2

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
2

### Summary
this paper solved the problem of evaluating cultural competency in LLMs beyond factual knowledge and surface-level cultural awareness. This paper proposed a spectral-analysis-based framework that measures macrostructural patterns in models’ cultural knowledge—capturing how models internalize and organize cultural variation across domains as an indicator of cultural competency. The framework introduces new spectral metrics to assess cultural pluralism and consensus, validated across eight models and nine domains.

### Strengths
1/ This work proposed a novel framework that shifts evaluation from micro-level factual correctness to macro structural cultural understanding. This theoretical move—linking spectral analysis to cultural competency.

2/ The authors conduct comprehensive experiments across models, domains, and human-validated ground truths, demonstrating that the proposed macrostructural evaluation captures distinct aspects of cultural knowledge compared to existing benchmarks.

3/ The simulation-based study connecting macrostructural scores to real-world user alignment (e.g. culturally appropriate recipe recommendations) is a strong practical validation of the framework’s predictive power.

### Weaknesses
1/ I think the paper’s methodology could benefit from clearer intuition—the mathematical formalism is ok but may be difficult for non-technical readers to connect to cultural cognition.

2/ The domain selection (nine cultural areas), though diverse, remains limited in scope; I suggest extending to cross-linguistic or non-Western cultural dimensions to test generality.

3/ While the framework is sound, its interpretability could improve. For example, offering case studies showing what specific macrostructural errors look like in practice.

### Questions
See weakneses

### Soundness
3

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
5

### Summary
This paper introduces a new analysis framework to evaluate LLMs meta-cultural competency, focusing on their ability to understand and navigate cultural diversity beyond factual knowledge. By examining cultural knowledge across 9 domains and 170 countries using two spectral metrics, Effective Rank and Spectral Gap Ratio, the study captures how models represent cultural structures at a macro level. The results show that instruction models align more closely with human cultural patterns, but increasing model size does not always improve performance. A simulation study further demonstrates that these spectral metrics can predict models’ cross-cultural adaptability, offering a novel approach to assessing cultural understanding in LLMs.

### Strengths
The paper presents a new spectral analysis framework that measures how LLMs understand cultural structures rather than just recalling facts. The proposed ER and SR can detect the cultural differences and similarity among countries. 

It combines ideas from computational modeling, anthropology, and cognitive science to build a novel theoretical foundation for evaluating cultural competence. 

The authors test eight models across 9 cultural domains and 170 countries, providing broad and systematic evidence that makes the study both comprehensive and convincing.

### Weaknesses
1. The model selection is limited, such as the GPT-2 is too old, Qwen model is representative but missed. Also any of sota close-sourced models is not tested, such as GPT-5, Grok, deepseek, etc. Which make the results and conclusion not solid.

2. No multilingual evaluation, which is quite essential and basic setting in cultural benchmarking works. Language itself is an important dimension of cultural testing. 

3. The prompt setting is stable or not, is not proved by the authors. The author should report the variance when rephrasing the prompts, cus model's output logits will be changed.

4. The simulation experiments is only in food domain, which makes it hard to conving that the proposed metrics also works in other cultural tasks.

5. In the main text (line 347), the authors refer to a table located in the appendix. Besides, the formatting of tables and figures is inconsistent: some figures include titles (figure 3) while others (figure 2) do not, and some tables use the three-line style whereas others display full grid lines.

### Questions
1. Can the authors provide more theoretical justification or empirical evidence that these spectral properties truly correspond to cultural cognition rather than to distributional variance?

2. Did the authors test prompt consistency quantitatively (e.g., by reporting variance in ER/SR across prompts)? If not, could they comment on potential sensitivity to prompt phrasing?

3. Do the authors expect the ER/SR metrics to generalize to other culturally sensitive tasks such as conversation, education, or healthcare? How to prove it'll also works?

4. The paper finds that model size does not correlate with macrostructural ability. Do the authors have any hypotheses or ablation results explaining why larger models do not necessarily perform better? Is it because logits measurement is not valid setting?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces a novel spectral-analysis-based framework for evaluating the meta-cultural competency of Large Language Models (LLMs). It moves beyond traditional, fact-based microstructural cultural evaluations to capture broader macrostructural patterns in cultural knowledge. The authors propose using spectral metrics—Effective Rank (ER) and Spectral Gap Ratio (SR)—to analyze how models organize knowledge across nine cultural domains and 170 countries. They compare eight LLMs of varying sizes and training regimes, finding that instruction-tuned models align better with human macrostructural expectations, but performance does not scale consistently with model size. A simulation-based experiment further demonstrates that macrostructural alignment predicts a model's ability to effectively serve users from unfamiliar cultural backgrounds.

### Strengths
Originality & Insight: The application of spectral analysis to cultural macrostructures is a highly original and compelling idea.

Clarity: The distinction between micro- and macro-structures is clearly drawn and well-motivated.

Rigor & Comprehensiveness: The experimental design is thorough, supported by human validation, and covers a wide range of models and cultural domains.

Significance: The findings have important implications for how we evaluate, select, and potentially train LLMs for globally inclusive applications.

### Weaknesses
Linguistic & Cultural Generality: The study is limited to English prompts and models. The language-dependence of the identified macrostructures and the framework's generalizability to multilingual or non-English cultural contexts remain open and critical questions.

Domain Coverage: While the nine chosen cultural domains are diverse, they are not exhaustive and may not capture all facets of cultural variation (e.g., social norms, values).

Potential Bias in Simulation: The simulation-based evaluation relies heavily on GPT-4o as a judge. Although validated with human ratings, this still introduces a potential for model-specific bias. Using another strong LLM as an additional judge could further strengthen this part of the argument.

### Questions
1.How might your framework be extended to multilingual or non-English cultural settings? Do you expect the macrostructural patterns to be consistent across different prompt languages, and do you have plans for such an investigation?

2.In the simulation experiment, did you consider using an additional, powerful LLM (e.g., Claude, GPT-4) as a second judge to cross-validate the "appropriateness" and "interaction" scores, thereby mitigating potential judge-specific bias?

3.You position macro- and micro-structures as complementary. Have you explored creating a combined metric that integrates your spectral scores (ER/SR) with performance on existing microstructural benchmarks? If so, what were the results?

### Soundness
4

### Presentation
3

### Contribution
3

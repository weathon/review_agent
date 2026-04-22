# JailNewsBench: Multi-Lingual and Regional Benchmark for Fake News Generation under Jailbreak Attacks

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 6, 4, 2, 6

## Abstract
Fake news undermines societal trust and decision-making across politics, economics, health, and international relations, and in extreme cases threatens human lives and societal safety.
Because fake news reflects region-specific political, social, and cultural contexts and is expressed in language, evaluating the risks of large language models (LLMs) requires a multi-lingual and regional perspective.
Malicious users can bypass safeguards through jailbreak attacks, inducing LLMs to generate fake news.
However, no benchmark currently exists to systematically assess attack resilience across languages and regions.
Here, we propose JailNewsBench, the first benchmark for evaluating LLM robustness against jailbreak-induced fake news generation.
JailNewsBench spans 34 regions and 22 languages, covering 8 evaluation sub-metrics through LLM-as-a-Judge and 5 jailbreak attacks, with approximately 300k instances.
Our evaluation of 9 LLMs reveals that the maximum attack success rate (ASR) reached 86.3% and the maximum harmfulness score was 3.5 out of 5.
Notably, for English and U.S.-related topics, the defensive performance of typical multi-lingual LLMs was significantly lower than for other regions, highlighting substantial imbalances in safety across languages and regions.
In addition, our analysis shows that coverage of fake news in existing safety datasets is limited and less well defended than major categories such as toxicity and social bias.
Our dataset and code are available at https://github.com/kanekomasahiro/jail_news_bench.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces JailNewsBench, a large-scale benchmark for evaluating LLM robustness against jailbreak-induced fake news generation. The benchmark spans 34 regions and 22 languages, incorporating 8 evaluation sub-metrics and 5 jailbreak attacks, totaling around 300k instances. The authors report that the maximum attack success rate reached 86.3%, with notably higher success and generation quality for English and U.S.-related topics, underscoring the need for multilingual and region-aware evaluation.

### Strengths
(1) This paper proposes the first multilingual and regional benchmark specifically for jailbreak-induced fake news generation. It covers an impressive scope: 34 regions, 22 languages, 5 attack types, 8 sub-metrics, and ~300k instances.

(2) This paper provides comprehensive evaluation across 9 state-of-the-art LLMs, revealing meaningful cross-lingual and regional disparities.

(3) This paper presents an evaluation showing strong correlation between LLM-as-a-Judge scores and human ratings

### Weaknesses
(1) Region selection excludes politically sensitive or unstable regions, skewing the dataset toward developed areas and limiting representativeness.

(2) Evaluation relies on GPT-5, Claude, and Gemini—the same families under test—introducing potential model-family bias and circular evaluation.

(3) It is unclear whether evaluation was performed in each native language or after translation; translation-based judging may distort non-English results.

### Questions
(1) Please clarify the two-stage filtering process—what exactly is excluded or retained, and how many samples fall into each category?

(2) How is Attack Success (AS) computed—before or after filtering?

(3) Were judgments made in the original languages or English translations? If translated, how was translation quality verified?

(4) Why do English and U.S. topics show higher attack success and quality?

### Soundness
3

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
This paper proposes JailNewsBench, a benchmark designed to evaluate the vulnerability of large language models (LLMs) to jailbreak attacks in the context of fake news generation. The authors introduce a systematic framework that includes: A set of seven attack strategies (Original, Explicit, Role Play, System Override, Research Front, Negative Prompting, Context Overload) to elicit harmful fake news from LLMs.An LLM-as-a-Judge evaluation framework with eight sub-metrics to assess the harmfulness of generated fake news.Experiments across multiple languages and models to demonstrate the effectiveness of the proposed benchmark.

### Strengths
<1> A large-scale, multilingual, multi-region benchmark specifically targeting jailbreak-induced fake news generation, covering diverse geographies/languages, seed rationales, and attack strategies, and supporting both black-box and white-box evaluations.

<2> An interpretable scoring framework with eight sub-metrics on a 5-point scale that moves beyond a single aggregate score; it improves reliability and diagnostic power, and is validated against alternative judges/human annotations.

<3> Systematic evaluation across strong LLMs uncovers key safety weaknesses, language/region disparities, and gaps in current safety benchmarks, offering actionable insights for model alignment and defense design.

### Weaknesses
<1> The relationships among sub-metrics and their aggregation method require further clarification.
The paper proposes eight evaluation dimensions, but does not provide detailed explanations regarding the independence among these metrics, how weights are assigned, or how the final overall score is computed. Some metrics may be highly correlated (e.g., "Verifiability" and "Faithfulness"), which could affect the efficiency and interpretability of the evaluation.

<2> The credibility of the evaluation results needs to be further strengthened.
The current assessment relies entirely on  models such as GPT-5, Gemini 2.5, and Claude 4 as judges. Although these models are highly capable, their judgments may contain implicit biases or suffer from insufficient consistency. It is recommended to supplement the evaluation with a human annotation experiment to analyze the agreement between human annotators and LLM-based judges.

<3> The presentation details could be further improved.
 The logical flow between certain sentences could be smoother. Additionally, the lack of sufficient figures, tables, and illustrative examples affects the overall readability.

<4>Insufficient discussion of failure cases or edge cases
The paper primarily presents successful jailbreak examples, but lacks analysis of instances where the model successfully resists attacks.

<5>The issue of redundancy among sub-metrics
Regarding Appendix F, Table 8 shows the top three sub-metrics with the highest average scores for each model. It can be observed that most models include Formality, Adherence, and Faithfulness. This raises the question: are the eight metrics potentially redundant?

### Questions
Q1. How do you ensure sub-metric independence and a robust aggregation into the final score?
Please (a) report pairwise correlations (Pearson/Spearman) among the 8 sub-metrics and run VIF, factor analysis or PCA; (b) state the exact weighting/aggregation formula and how weights are chosen (fixed vs. learned; per-dataset or global); (c) conduct sensitivity/ablation by removing or reweighting highly correlated metrics (e.g., Verifiability, Faithfulness, Formality, Adherence) and show whether model rankings remain stable (Kendall’s τ / Spearman ρ with CIs). Directly address the redundancy suggested by Appendix F, Table 8.

Q2. How reliable are the LLM-judge scores without human raters?
Please add a human annotation study and report agreement with LLM judges (Cohen’s κ / Krippendorff’s α / Spearman ρ), plus inter-judge consistency across GPT-5, Gemini 2.5, and Claude 4 (variance, CIs via bootstrap). Provide the judging prompts and decoding settings (temperature, seed) and show that conclusions hold under alternative judges/ensembles or majority-vote calibration.

Q3. What are the failure and edge cases, and can presentation be strengthened with concrete examples/plots?
Please include paired examples where attacks fail (i.e., the model resists) alongside successful jailbreaks: prompt - model response - per-sub-metric scores - final decision. Summarize error/defense taxonomies by attack type/task. Improve readability with figures/tables (e.g., radar plots of 8 metrics, box/violin plots of variance, ranking-stability plots) and illustrative case studies that clarify the logical flow.

### Soundness
2

### Presentation
2

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
This paper introduces JailNewsBench, a large-scale benchmark for testing whether jailbreak attacks can induce LLMs to generate fake news across 34 regions and 22 languages (≈300k seed instructions). Seed instructions are grounded in real news and four malicious “motivations”, then paired with five jailbreak styles (i.e., role-play, system override, research front, negative prompting, context overload). Outputs are scored by an LLM-as-a-Judge that decomposes harmfulness into eight sub-metrics (e.g., Faithfulness, Subjectivity, Agitativeness). On nine LLMs, the authors report high attack success and non-trivial harmfulness, with English/U.S. topics generally more robust than other languages/regions; translating non-English items into English does not close the gap. Models only modestly detect their own generated fakes (best F1 is only 68.2). The authors also argue fake-news generation is underrepresented in existing safety datasets relative to toxicity/bias.

### Strengths
* Clear societal risk focus, broad coverage. The benchmark targets jailbreak-induced fake-news generation, spanning 34 regions/22 languages and ~300k instances, which is substantially broader than typical English-centric setups.

* Documented selection guardrails. The paper explicitly specifies region inclusion criteria (exclude places with special fake-news laws, high instability, or very recent news) to reduce release risk.

### Weaknesses
LLMs’ vulnerability in misinformation-related contexts is a well-established finding. This paper revisits the issue through two main empirical observations: (1) LLMs can be induced to generate misinformation through prompting, and (2) they struggle to detect LLM-generated fake news when relying solely on their internal knowledge and without access to external evidence. However, both observations have been widely reported in prior work: [1] involves the first point, [1-3] involves the second point.

Consequently, the paper’s primary novelty appears to rest on its scale, multilingual scope, and the design of its jailbreaking prompts. Nonetheless, several aspects limit the strength of this contribution:

* Lack of comparative validation. Although the paper positions its jailbreak design as a key contribution, there is no systematic validation of its superior effectiveness, e.g., whether it achieves higher success rates than existing studies that prompts LLMs to generate misinformation [1], produces more harmful or influential misinformation, or yields more deceptive outputs that better evade detection.

* Absence of per-type analysis. The study defines five jailbreak strategies and two baselines, but the results aggregate all into a single “Jailbreak” condition. A per-type comparison would clarify which jailbreaks are most effective, and under what conditions, providing more insight into their distinct mechanisms.

* Under-theorized model compliance. While the paper introduces diverse jailbreak patterns, it does not explore why models comply with deceptive instructions. Analyses on refusal bypassing, instruction-following cues, or safety-policy framing could substantially strengthen interpretability. Similarly, ablations linking prompt design to compliance behavior are missing. This reduces the utility of this work in informing research on LLM safety.

* Focus on attack efficacy, not governance. The paper systematically explores how to increase jailbreak success but offers no parallel study of defenses or governance (e.g., retrieval grounding, output filtering, or post-hoc detection pipelines) nor an evaluation of how these mitigations affect success/harmfulness. As a result, the work advances offense without producing actionable risk-reduction guidance for practitioners.

* This manuscript raises additional ethical concerns. Please refer to Flag For Ethics Review.

[1] Can LLM-Generated Misinformation Be Detected? ICLR 2024.

[2] Bad Actor, Good Advisor: Exploring the Role of Large Language Models in Fake News Detection. AAAI 2024.

[3] Fake News in Sheep's Clothing: Robust Fake News Detection Against LLM-Empowered Style Attacks. KDD 2024.

### Questions
Please address all weaknesses and ethical concerns raised.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper evaluates the robustness of LLMs against jailbreak-induced fake news generation. It introduces JailNewsBench, the first large-scale multilingual and regional benchmark for this task. The benchmark spans 34 regions and 22 languages, incorporates five types of jailbreak attacks, and assesses model outputs using an eight-dimensional LLM-as-a-Judge harmfulness evaluation framework. Experiments on nine prominent LLMs reveal high attack success rates (up to 86.3%) and significant disparities in robustness between English and non-English regions. The study highlights fake news generation as an underexplored yet critical dimension of LLM safety, complementing existing work on toxicity and bias.

### Strengths
1. This paper systematically studies jailbreak attacks that induce LLMs to generate fake news, addressing an important and underexplored safety gap in current research.
2. The proposed JailNewsBench spans multiple regions and languages, making it comprehensive and representative than existing jailbreak benchmarks.
3. The authors conduct extensive experiments across different LLMs, including both black-box models and open-source ones, using four malicious motivations, five jailbreak techniques, and an eight-dimensional harmfulness evaluation. The paper provides detailed descriptions and appendices that should facilitate reproducibility.

### Weaknesses
1. The paper does not provide a grouped analysis of attack performance under different malicious motivations. It would be interesting to see whether current LLMs exhibit varying sensitivity to these different intent types.
2. The assessment relies on LLMs to evaluate outputs, which may introduce circular or model-specific bias.
3. Considering that fake news generation is a highly sensitive and potentially harmful topic, the paper would benefit from a more detailed discussion on ethical safeguards and dataset release protocols.
4. While the study identifies vulnerabilities, it does not explore mitigation strategies or future directions such as defense mechanisms or fine-tuning approaches to resist jailbreak attacks.
5. Minor typos, e.g.,  Appendix B title: “PROMPT FOR GENERATING EED INSTRUCTIONS.”

### Questions
What is the specific criterion or threshold of “successful” attacks?

### Soundness
3

### Presentation
3

### Contribution
3

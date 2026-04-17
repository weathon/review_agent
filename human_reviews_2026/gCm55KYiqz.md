# CAGE: A Framework for Culturally Adaptive Red-Teaming Benchmark Generation

- Decision: Accept (Poster)
- Scores: 4, 6, 4

## Abstract
Existing red-teaming benchmarks, when adapted to new languages via direct translation, fail to capture socio-technical vulnerabilities rooted in local culture and law, creating a critical blind spot in LLM safety evaluation. To address this gap, we introduce CAGE (Culturally Adaptive Generation), a framework that systematically adapts the adversarial intent of proven red-teaming prompts to new cultural contexts. At the core of CAGE is the Semantic Mold, a novel approach that disentangles a prompt's adversarial structure from its cultural content. This approach enables the modeling of realistic, localized threats rather than testing for simple jailbreaks. As a representative example, we demonstrate our framework by creating KoRSET, a Korean benchmark, which proves more effective at revealing vulnerabilities than direct translation baselines. CAGE offers a scalable solution for developing meaningful, context-aware safety benchmarks across diverse cultures.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces CAGE (Culturally Adaptive Generation), a framework for generating culturally-grounded red-teaming benchmarks to evaluate LLM safety across different languages and cultures. The authors argue that directly translating English safety benchmarks fails to capture socio-technical vulnerabilities rooted in local laws, social norms, and historical contexts. CAGE addresses this through "Semantic Molds"—slot-based representations that separate adversarial structure from cultural content—enabling systematic adaptation of English red-teaming prompts to new cultural contexts. The framework collects seed prompts from existing English benchmarks, refines them into abstract slot-tagged forms, and instantiates these molds with culturally-specific content curated from local sources such as laws, news, and social discourse. As a demonstration, the authors create KoRSET, a large-scale Korean benchmark, and show that CAGE-generated prompts achieve substantially higher Attack Success Rates and quality scores compared to direct translation baselines.

### Strengths
- Novel semantic mold framework. The core idea of separating adversarial structure from cultural content through slot-based semantic molds is creative and provides a systematic approach to cultural adaptation.

- Strong and consistent quantitative results. CAGE demonstrates substantial improvements over direct translation across all tested models and attack methods. The results are comprehensive, covering multiple models and four automated attack frameworks.

- Demonstrated generalizability. The Khmer case study (Section 4.4) provides evidence that the framework is language-agnostic and works for low-resource languages, with similar patterns of quality and ASR improvements. This suggests the approach can scale beyond Korean.

- Large-scale benchmark contribution. KoRSET provides 7,161 culturally-grounded Korean prompts across 12 risk categories, representing a valuable resource for the Korean LLM safety community.

### Weaknesses
- Missing critical baselines and comparisons with existing cross-cultural adaptation methods. The paper discusses three categories of existing approaches in Section 2.3—direct translation, template adaptation (e.g., KoBBQ), and native construction (e.g., KorNAT)—and claims CAGE integrates their benefits while avoiding their limitations. However, experiments only compare against direct translation, the weakest baseline. Notably absent is comparison with template-based methods like KoBBQ, which also use slot-filling with culturally-specific entities and are conceptually similar to CAGE's semantic molds. Additionally, the paper does not test against a simple but powerful baseline: prompting frontier language models to "adapt this English prompt to Korean cultural context with local laws and current events." Without these comparisons, it remains unclear whether semantic molds provide meaningful advantages over existing template approaches, whether the complex pipeline outperforms simple LLM-based adaptation, and whether CAGE achieves its claimed goal of combining the "cultural fidelity of native dataset construction" with the "scalability of template-based methods." These missing comparisons prevent proper positioning of CAGE within the landscape of existing methods and leave the necessity of the framework's complexity unvalidated.

- Unvalidated Fundamental Premise. The paper's central claim is that culturally-grounded prompts are necessary because "real-world threats are deeply rooted in local laws, social conflicts, and historical contexts that cannot be conceived in one language and simply translated". However, the paper never validates whether English and Korean prompts actually reveal different vulnerability types or merely test the same underlying capabilities with different surface details. The critical missing experiment is comparing vulnerability distributions: does a multilingual model exhibit qualitatively different failure modes when tested with original English prompts versus CAGE-generated Korean prompts, or do both reveal the same vulnerability categories with higher Korean ASR simply reflecting more effective jailbreaking?

### Questions
**What explains the dramatic ASR improvements from cultural adaptation, and what does this reveal about model safety mechanisms?**

Your results show substantial ASR increases (50-200%) when adding culturally-specific details to prompts. However, the paper does not explain the mechanism behind these improvements. Several competing explanations exist, each with different implications for your framework's contribution:

**(A) Culture-specific knowledge gaps**: Models lack training on Korean cultural contexts (laws, historical events, social structures), creating blind spots that culturally-grounded prompts exploit. This would validate your core claim about culture-specific vulnerabilities.

**(B) English-centric safety alignment**: Models are primarily safety-aligned on English corpora, so Korean prompts bypass filters not because they test different vulnerabilities but because safety mechanisms weren't trained to recognize Korean patterns. This would suggest the issue is training data imbalance rather than fundamental cultural differences in vulnerability types.

**(C) Specificity effect**: The original English seed prompts may be deliberately generic (for broad coverage), while CAGE prompts are highly specific. The improvements might reflect specificity rather than cultural grounding—would equally specific English prompts achieve similar gains?

**(D) Naturalistic framing**: Cultural details make prompts appear more like legitimate informational queries, reducing adversarial signal. This would be a universal jailbreaking technique rather than culture-specific vulnerability discovery.

Could you provide analysis distinguishing these explanations? Specifically:

1. Compare CAGE Korean prompts against **highly-specific English prompts** (not generic translations) to isolate the cultural vs. specificity effect
2. Test CAGE prompts on **Korean-specialized models** (like EXAONE) vs. English-primary models to assess whether improvements stem from English-centric training
3. Analyze **failure cases** showing what cultural knowledge the model lacks versus where it simply fails to detect adversarial intent

This would clarify whether your framework reveals genuinely culture-specific vulnerabilities or primarily exploits artifacts of English-dominated safety training, which is crucial for evaluating the paper's core contribution.

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
4

### Summary
The paper introduces a new framework called CAGE (Culturally Adaptive Generation) to generate culturally adaptive red teaming benchmark. Authors focus on Korean for this specific study and use their framework to generate a Korean red teaming benchmark which they call it KoRSET. Authors show that their framework coupled with their benchmark is more effective at revealing vulnerabilities than direct translation baselines.

### Strengths
1. The paper studies an interesting, timely, and important problem which can be exciting to the community. 
2. Generating culturally aware red teaming benchmarks can benefit the community. The benchmark dataset that authors created in this paper (KoRSET) along with the taxonomy can be really useful.
3. Authors perform various ablations and use various models to study the effectiveness of their work.

### Weaknesses
1. The paper is limited in its scope as it focuses on Korean only. It would be more interesting if authors could expand their studies across more diverse cultural contexts.
2. There were some terminologies that were used in the paper, such as mold and slot, that were not well defined in the beginning of the paper, so it may take sometime for the reader to understand what they really are after reading the paper and going over some examples. It would be nice if authors define these terms upfront in the paper when they are first introduced.
3. In section 3.3, some important details are missing on how exactly things are done (e.g., "all collected materials were pre-processed into valid slot replacements and manually reviewed to ensure semantic fidelity and linguistic fluency"). It would be good if more details are put into the manual review and statistics on that. In addition, more details would be good to be provided for Taxonomy and Trend driven approaches/pipelines. 
4. For the experiments, analysis on more languages and cultures would have been interesting.
5. Comparison to vanilla translation baseline was interesting, but it would be good if more baselines are studied or even studies on the effect of using various translation approaches/tools.

### Questions
1. In section 3.3, it seemed like the Taxonomy and Trend driven approaches/pipelines would be really time consuming to be utilized for each culture separately. Is this the reason why authors focused on Korean only? In general, how much effort is required to construct such a benchmark and what is its trade-offs compared to a simple translation based baseline?

2. How large is KorSET? It would be good if some statistics are provided about the benchmark.

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
4

### Summary
This paper presents a framework CAGE that generate multilingaul red teaming prompts from existing prompts. Compared to translation based methods, the proposed framework claims to be more effective at adapting existing prompts to prompts in a different language due to its ability to consider cultural specificity. Based on this framework, the paper constructs KoRSET, a Korean benchmark. The experimental results show that this benchmark is more effective at red teaming LLMs than translation based baselines.

### Strengths
1. The experiment results show that the prompts generated by the proposed framework CAGE achieved higher ASR compared to translation based approach.
2. This work introduces a Korean red teaming dataset, KorSET.
3. The proposed framework is more friendly to low-resource language, as translation quality is lower compared to high-resource language.

### Weaknesses
The proposed framework seems to need a lot of manual work such as the manual construction of taxonomy, and contury specific content gathering such as keywords and topics.

### Questions
Will the dataset KorSET be made public?

### Soundness
2

### Presentation
3

### Contribution
2

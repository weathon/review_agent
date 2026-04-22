# Adaptive Generation of Bias-Eliciting Questions for LLMs

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 2, 6, 6

## Abstract
Large language models (LLMs) are now widely deployed in user-facing applications, reaching hundreds of millions worldwide. As they become integrated into everyday tasks, growing reliance on their outputs raises significant concerns. In particular, users may unknowingly be exposed to model-inherent biases that systematically disadvantage or stereotype certain groups. However, existing bias benchmarks continue to rely on templated prompts or restrictive multiple-choice questions that are suggestive, simplistic, and fail to capture the complexity of real-world user interactions. In this work, we address this gap by introducing a counterfactual bias evaluation framework that automatically generates realistic, open-ended questions over sensitive attributes such as sex, race, or religion. By iteratively mutating and selecting bias-inducing questions, our approach systematically explores areas where models are most susceptible to biased behavior. Beyond detecting harmful biases, we also capture distinct response dimensions that are increasingly relevant in user interactions, such as asymmetric refusals and explicit acknowledgment of bias.Leveraging our framework, we construct CAB, a human-verified benchmark spanning diverse topics, designed to enable cross-model comparisons. Using \bench, we analyze a range of LLMs across multiple bias dimensions, revealing nuanced insights into how different models manifest bias. For instance, while GPT-5 outperforms other models, it nonetheless exhibits persistent biases in specific scenarios. These findings underscore the need for continual improvements to ensure fair model behavior.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes a framework for evaluating bias in LLMs through adaptive generation of counterfactual questions. The core approach employs a genetic algorithm-inspired optimization process where questions are iteratively mutated and selected based on their ability to elicit biased responses from target models. The authors introduce a multi-dimensional evaluation scheme. The authors also construct CAB, a human-verified benchmark. They evaluate various state-of-the-art models on this benchmark, finding that most of the sex-related questions elicit biased responses from at least one model.

### Strengths
The paper attempts to address limitations in existing bias evaluation benchmarks, particularly the reliance on templated prompts and multiple-choice formats that poorly reflect real-world interactions. The multi-dimensional evaluation framework is thoughtful, especially in distinguishing between models that exhibit bias versus those that acknowledge it, which is indeed a common conflation in prior work. The human verification step adds credibility to the final benchmark. The comprehensive evaluation across various frontier models with detailed categorization of bias types provides practical value for understanding current model behaviors.

### Weaknesses
My primary concern with this work is that the counterfactual evaluation paradigm is far from novel and has been extensively employed in existing bias evaluation literature (WinoQueer, CrowS-Pairs, WinoBias, BBQ, etc.). The authors fail to adequately justify how their application of counterfactuals constitutes a meaningful technical contribution. Simply generating questions with placeholder attributes like {man/woman} and comparing responses is standard practice in this field. The claim that their approach is innovative because it generates "realistic, open-ended questions" rather than templates is undermined by the fact that their questions are still built around the same counterfactual template structure, just with more elaborate contexts around them.

The genetic algorithm optimization aspect, while interesting, raises significant questions about what is actually being learned. The framework essentially tunes questions to maximize bias in specific target models, but it's unclear whether this produces questions that genuinely reflect real-world bias scenarios or merely adversarial examples that exploit model-specific quirks. The fact that questions are optimized on five models but then evaluated on nine different ones partially addresses this, but the paper doesn't adequately explore whether the learned questions transfer well or simply represent a form of overfitting to the target models' failure modes.

The reliance on LLM judges for both fitness evaluation during generation and final bias assessment is problematic. While the authors acknowledge this limitation briefly, they don't sufficiently address how errors in the judge's assessment propagate through the entire framework. If the judge misidentifies bias during the optimization process, the genetic algorithm will evolve questions based on incorrect fitness signals. The paper lacks any systematic validation of judge reliability or inter-annotator agreement beyond the final human filtering step, which only verifies that questions are well-formed rather than that the bias scores are accurate.

The benchmark construction methodology also has concerning aspects. Using only 405 questions total across three attributes seems quite limited for making broad claims about model bias. The human filtering process removed roughly 35% of generated questions, but the paper provides minimal detail about what made these questions unsuitable beyond brief categories. Were they syntactically malformed, or did they fail to actually elicit bias? This distinction matters significantly for understanding whether the generation process works as intended. Furthermore, the heavy filtering suggests the generation process produces substantial amounts of unusable content, raising questions about the practical utility of the approach for scaling to larger benchmarks.

The experimental design choices are not well justified. Why were these specific five target models chosen for generation rather than others? Why is k=3 samples adequate for determining fitness? The paper sets various hyperparameters (gamma=0.5, specific mutation probabilities) without ablation studies or sensitivity analysis. The implicit question results showing a 40% drop in bias are presented as validation, but could equally suggest the framework is overfitted to explicit phrasings and doesn't generalize to more realistic interaction patterns.

### Questions
Please see weaknesses.

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
4

### Summary
This paper proposes an automated counterfactual question generation framework to discover realistic prompts that elicit biased responses from LLMs and then uses that framework to build a human-verified benchmark of bias-inducing questions across three sensitive attributes (sex, race, religion). The method uses LLMs in three roles (1) proposes or mutates candidate questions, (2) queries target models multiple times per attribute value, and (3) has an LLM judge score responses on four dimensions (Bias, Relevance, Acknowledgment, Refusal). The paper analyzes many state-of-the-art models on CAB and categorizes the kinds of biased behavior found.

I think that the main contribution of this paper is its strong execution and resulting dataset that can be used by subsequent research and bias detection. The "LLM-as-data-generator-for-bias-evaluation" main idea I don't think is that novel.

### Strengths
Practical and scalable approach to bias detection and strong execution:
The adaptive generation loop addresses a real gap: templated prompts are brittle and unrealistic. Automating iterative question mutation and selection is an effective way to surface diverse, lifelike prompts that stress models in ways hand-crafted datasets often miss. While some of the ideas are not conceptually new, the method is implemented cleanly and systematically. The workflow is easy to follow, with explicit prompt templates, judge dimensions, and selection criteria, which enhances transparency and reproducibility.

Dataset useful to the community: 
The resulting CAB dataset appears to contain realistic, natural-language prompts rather than artificial templates, addressing a genuine limitation of prior bias benchmarks. If released with appropriate documentation and safety controls, it could become a useful testbed for bias evaluation and mitigation research.

Strong reproducibility and high levels of details: 
The authors include prompt examples, scoring guidelines, and discuss their data verification process. Even though much of the system relies on LLM components, the procedural clarity allows replication and potential extension.

The evaluation is comprehensive: 
The authors apply the framework to three sensitive attributes (gender, race, religion) and multiple strong target models. The breadth of tested systems and the consistent analysis across them make the study empirically solid. The inclusion of explicit and implicit variants of questions is a nice design choice that highlights differences between overt and subtle bias triggers.

### Weaknesses
Limited novelty: 
The core idea, using LLMs to generate probes for bias detection, is not that novel. The proposed adaptive generation loop is a modest extension of existing LLM-driven dataset generation methods, without introducing fundamentally new technical insights or modeling mechanisms. As a result, the paper reads more as a well-executed systemization of known techniques than a conceptual advance.

Judge reliability and "overfitting of bias": 
The evaluation pipeline hinges on a single LLM judge to assess bias, relevance, acknowledgment, and refusal. Without quantitative calibration against human annotation or independent judges, it’s unclear how reliable or stable these automated scores are. This undermines the scientific validity of both the adaptive optimization and the benchmark’s ground truth. Furthermore, when generator and judge are both LLMs (and particularly if the judge and generator share architecture or training data), there is a bias amplification risk. 

Limited generalization, scalability, and external validity: 
The adaptive generation process optimizes questions for a specific target model or a narrow set of models, raising the risk that CAB primarily captures model-specific artifacts rather than general bias phenomena. The paper does not include transfer or robustness analyses to demonstrate that high-fitness questions generalize across models or settings. Furthermore, the benchmark is constrained to three attributes and English, U.S.-centric contexts, limiting cross-cultural and intersectional coverage.
The human filtering stage was performed solely by the paper’s authors, without independent annotators or inter-annotator agreement reporting, leaving potential subjectivity and selection bias unquantified. Finally, despite claims of automation, the approach still depends on substantial human review and on well-defined bias categories and cultural priors. These dependencies constrain scalability and hinder the framework’s applicability to real-world, diverse, or multilingual deployments.

Insufficient statistical and interpretive rigor in evaluation: 
The evaluation reports only average fitness scores without confidence intervals, significance testing, or analysis of variability due to judge randomness or sampling (k=3 responses per attribute value is likely too small). As a result, it is unclear whether observed differences between models or attributes are statistically meaningful. Beyond descriptive comparisons, the paper offers little interpretive depth connecting specific question characteristics to types or mechanisms of bias, limiting the scientific insight drawn from the results.

### Questions
Validity of the LLM judge and evaluation reliability: 
How did you validate the reliability of these judgments against human annotations?
Have you measured agreement between the LLM judge and multiple human raters, or tested robustness across different judge models or prompt formulations?
Given that small prompt or model changes can shift judgments substantially, how stable are your reported fitness scores?

Generalization and robustness of discovered questions: 
Since questions are optimized on specific target models, to what extent do they transfer to other architectures or checkpoints?
Have you evaluated cross-model generalization to verify that CAB captures systemic bias rather than model-specific artifacts?

Human verification and dataset objectivity: 
Can you clarify how many annotators participated, what criteria were used, and whether inter-annotator agreement was measured?
Would independent or crowdsourced verification yield consistent results?
Why is the question drop rate sort of high (like 130 something / 250, nearly half)?
Could this pipeline realistically operate as a continuous evaluation system for production models, or is it mainly for controlled benchmark creation?

Statistical and interpretive significance: 
Have you performed significance testing, or estimated variance across random seeds and sampling?
Beyond numeric scores, can you provide qualitative analyses or case studies linking specific prompt features to bias mechanisms?

### Soundness
3

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
4

### Summary
This paper introduces an adaptive genetic algorithm that iteratively generates open-ended questions designed to elicit biased responses from large language models. The approach optimizes questions based on a multi-dimensional fitness score that distinguishes between bias exhibition and acknowledgment across four dimensions: bias strength, relevance to the question, explicit acknowledgment of societal biases, and asymmetric refusals. The authors construct CAB, a human-verified benchmark of 405 questions spanning three sensitive attributes (sex, race, religion), and evaluate nine frontier models.

### Strengths
Unlike static templated prompts, the approach iteratively refines questions through generate/replace/refine mutations, systematically searching for bias-eliciting scenarios. 

The four-dimension fitness (bias/relevance/acknowledgment/refusal) enables nuanced analysis beyond aggregate metrics.

Full prompts, detailed hyperparameters, and human filtering statistics support reproducibility.

### Weaknesses
Questions were optimized against Claude-Haiku-3.5, Gemini-Flash-Lite, LLaMA-4-Maverick, Hermes-3, and GPT-4-Mini, yet the evaluation includes models from the same model families. Models within families may share training data and architectural choices. The genetic algorithm explicitly evolved questions to maximize bias in these specific models, meaning CAB may measure family-specific training artifacts rather than general societal biases. The claim that using slightly different models ensures fairness lacks supporting evidence. No analysis shows questions optimized on GPT-4-Mini generalize equally to Claude/Gemini families. This fundamentally affects the paper's central claim of enabling fair cross-model comparison. Cross-family transfer analysis quantifying whether optimization advantages specific families may strengthen the paper.

All findings rest on judgments from a single LLM judge (GPT-5-Mini), which is also from the generating model's family, creating a circular system where questions are evolved to trigger responses a specific model will judge as biased. The absence of human validation and insufficient multi-judge reliability analysis means systematic judge errors could propagate throughout the benchmark.

Minor issues:
1. No evaluation of the same nine models on prior benchmarks exists. Cannot assess whether CAB reveals new biases or rediscovers known ones, or whether model rankings correlate with established benchmarks.

### Questions
see above

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
Authors propose a counterfactual bias evaluation framework that generates realistic, open-ended question pairs over sensitive attributes such as sex, race and religion. The framework uses LLM-as-a-Judge to evaluate the target model's answers along four dimensions (bias, relevance, acknowledgment and refusal). The scores obtained during this evaluation is formulated as a fitness score to further refine the dataset and build a high quality benchmark dataset. Authors have designed the policies and rubric for the various prompts very meticulously (manually developing categories for various attributes). The final benchmark dataset is human verified (by authors). Authors show experimentally that framing the questions is very important to elicit the biases in models (and show how explicit vs. implicit framing changes the results by a significant margin). Using the dataset, authors have analyzed various frontier models and show various interesting patterns (GPT5 shows low bias compared to other models).

### Strengths
- the paper expands on the recent directions and generates more open-ended realistic interactions with counterfactual pairs (this goes beyond templates and MCQ style assessment) 
- focuses on disentangling bias acknowledgement from actual bias in response and tries to identify those; similarly, the fitness scoring mechanism penalizes irrelevant information (focus on relevance) and refusals. This is critical to building the high quality dataset. 
- the adaptive dataset expansion approach is AI classical (like spreading activation networks and search) that seem to work well in this context and produce good dataset which is a strong contribution of this research.

### Weaknesses
- this work utilizes a completely automated pipeline of prompt generators and evaluators heavily relying on LLM-Judges which seems like a recursive problem - using LLMs to judge biases in LLMs - although the authors validate the dataset that the system generates, it still requires many human assessments, using intern-annotator agreements to understand various correlations, use of other metrics (like sentiment or other proxy) to also compare and validate the use of LLM judges here. 
- while the fitness score looks reasonable, this work should include more studies to better understand the choice of function used in fitness metric, the stability of results, use of ablation studies, etc. 
- no comparison to other datasets, baselines, benchmarks on the same models; authors discount use of other popular benchmarks like BBQ that are very popular (even though being quite old now), there are generative, multilingual and multimodal extensions of BBQ already in literature, there is no comparative studies. 
- Also, other benchmarks use many more attributes (beyond sex, race and religion); this work is limited in this context.

### Questions
- Does the fitness scoring mechanism correlate with human judgments?
- Any insights on stability of the fitness function (choice of normalization function, sampling size, etc) 
- Do models that score well in this research also score well on other benchmarks like Open-BBQ ( or others)?

### Soundness
2

### Presentation
3

### Contribution
3

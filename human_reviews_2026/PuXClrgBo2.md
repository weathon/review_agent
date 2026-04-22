# T2VPhysBench: A First‑Principles Benchmark for Physical Consistency in Text‑to‑Video Generation

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 6, 6, 4

## Abstract
Text-to-video generative models have made significant strides in recent years, producing high-quality videos that excel in both aesthetic appeal and accurate instruction following, and have become central to digital art creation and user engagement online. Yet, despite these advancements, their ability to respect fundamental physical laws remains largely untested: many outputs still violate basic constraints such as rigid-body collisions, energy conservation, and gravitational dynamics, resulting in unrealistic or even misleading content. Existing physical-evaluation benchmarks typically rely on automatic, pixel-level metrics applied to simplistic, life-scenario prompts, and thus overlook both human judgment and first-principles physics. To fill this gap, we introduce T2VPhysBench, a first-principled benchmark that systematically evaluates whether state-of-the-art text-to-video systems, both open-source and commercial, obey twelve core physical laws including Newtonian mechanics, conservation principles, and phenomenological effects. Our benchmark employs a rigorous human evaluation protocol and includes three targeted studies: (1) an overall compliance assessment showing that all models score below 0.60 on average in each law category; (2) a prompt-hint ablation revealing that even detailed, law-specific hints fail to remedy physics violations; and (3) a counterfactual robustness test demonstrating that models often generate videos that explicitly break physical rules when so instructed. The results expose persistent limitations in current architectures and offer concrete insights for guiding future research toward truly physics-aware video generation.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposed an text prompt benchmark *T2VPhysBench* for evaluation of video generation model (T2V) 's obedience to physical rules. The proposed prompt benchmark contains 84 prompts divided into 12 specific physical laws. The benchmark employs pure human evaluation, results are reported in Table 2 for 10 models either closed or open-sourced.

### Strengths
1. The paper is clearly written
2. Besides the normal evaluations, authors also proposed an *counterfactual* evaluation protocol in section 4.3: instead of prompt with plausible physical descriptions, authors intentionally use prompt with implausible physical descriptions and then check whehter videos obeying physical laws are generated.

### Weaknesses
1. The proposed benchmark contains only text prompts, whereas the mainstream of video generation models are I2V styled, this would limit the actual usage of the benchmark.
2. The proposed benchmark needs pure human judgement. It would be better that authors had studied the variance and confidence of human performance, and even more, might compose an automatic evaluation method based on the proposed benchmark.

### Questions
1. About the *counterfactual* evaluation protocol in section 4.3, the stated observation 4.5 in page 8 is too superficial. Results in table 3 are quite different from table 1, is there any other conclusion could we get from the differnet ranking?

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
3

### Summary
This paper introduces T2VPhysBench, a human-evaluated benchmark designed to assess whether state-of-the-art text-to-video (T2V) models conform to fundamental physical laws. The benchmark uses 84 human-annotated cues to evaluate the performance of 10 models across 12 physical principles (Newton's laws, conservation laws, and phenomenological principles). The authors conducted three studies: (1) overall conformity assessment; (2) cue removal; and (3) counterfactual robustness testing.

### Strengths
1. Physical consistency in T2V generation is crucial but under-researched. First-principles-based methods are more systematic than scenario-based evaluations.

2. The evaluation is thorough and comprehensive, with detailed human evaluations and benchmarks covering 10 open-source and commercial models.

3. The paper is well-written with good visual examples.

### Weaknesses
1. Only 7 prompts per physical law (84 total) is quite limited, while similar benches such as Vbench or ChronoMagic-Bench have over 1000 prompts.

2. Although the experiment covered the vast majority of video generation models, the evaluation process only involved three annotators, raising questions about its reliability.

3. While the Prompts are divided into three main categories—Newton's Principles, Conservation Principles, and Phenomenon Principles—they do not consider more important physical processes such as seed germination and ice melting, which are mentioned in MagicTime and ChronoMagic-Bench.

### Questions
None

### Soundness
3

### Presentation
2

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
This paper introduces T2VPhysBench, a new benchmark designed to evaluate whether modern text-to-video generation models obey fundamental physical laws. Unlike prior benchmarks that rely on pixel-level or scene-based evaluations, T2VPhysBench follows a first-principles design, covering twelve core physical laws across three categories:

1. Newtonian mechanics (e.g., inertia, action-reaction),

2. Conservation laws (energy, mass, momentum, angular momentum), and

3. Phenomenological effects (Hooke’s law, Snell’s law, reflection, Bernoulli’s principle).

The benchmark involves human evaluation of videos generated by ten leading models (e.g., Sora, Wan 2.1, Kling, Mochi-1, Pika 2.2).

### Strengths
1. Unlike heuristic or pixel-based benchmarks, T2VPhysBench is grounded in explicit physical laws, making the evaluation interpretable and scientifically principled.
2. Manual scoring across multiple annotators ensures that evaluations align with human perception, avoiding pitfalls of purely automated metrics.
3. The paper is well-organized, easy to follow

### Weaknesses
1. While human evaluation is valuable, combining it with physics-informed automatic checks (e.g., trajectory analysis, optical flow consistency) could strengthen reproducibility.
2. A comparison with existing efforts such as VideoPhy or Physics-IQ using identical prompts would clarify what new insights T2VPhysBench uniquely provides.
3. The evaluation relies on only three annotators and coarse-grained scoring (0, 0.25, 0.5, 1). Statistical measures of inter-annotator agreement (e.g., Cohen’s κ) are missing.

### Questions
1. The paper does not clearly specify whether T2VPhysBench will be open-sourced, nor how prompts, human evaluation guidelines, or reference outputs will be shared.

### Soundness
3

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
4

### Summary
This paper introduces T2VPhysBench, a new benchmark for evaluating the physical consistency of text-to-video (T2V) generation models. The authors argue that existing benchmarks often rely on automated, pixel-level metrics or overly simplistic scenarios, failing to capture the nuances of physical laws from a first-principles perspective. T2VPhysBench is designed around 12 fundamental physical principles, categorized into Newtonian mechanics, conservation laws, and phenomenological effects. The core of the benchmark is a rigorous human evaluation protocol where annotators score generated videos on a 4-level scale based on their adherence to the specified law. The study evaluates ten state-of-the-art T2V models and presents three key findings: (1) all models perform poorly, with average compliance scores below 0.60; (2) adding explicit hints and detailed descriptions to prompts provides minimal to no improvement, and can even degrade performance; (3) models often generate videos that explicitly violate physical laws when prompted to do so, suggesting their behavior is based on surface-level pattern matching rather than genuine physical understanding.

### Strengths
The primary strength of this work is its grounding in fundamental, first-principles physical laws rather than just ad-hoc, life-scenario prompts. This provides a more structured and fundamental way to probe the physical reasoning capabilities of T2V models. The paper goes beyond a simple compliance test. The inclusion of the prompt-hint ablation study and the counterfactual robustness test are particularly insightful. The finding that more detailed prompts do not help (and sometimes hurt) is a crucial insight, suggesting that the problem is not a lack of prompt understanding but a core limitation in the models' world knowledge. The counterfactual tests compellingly support the hypothesis that models rely on pattern memorization.

### Weaknesses
1.Limited Scale and Statistical Significance: The benchmark is based on 84 distinct prompts, which are distributed across 12 different physical laws. This averages to only 7 prompts per law. This sample size is quite small and raises concerns about the statistical significance of the results for any single law. A model's performance on a specific principle could be heavily skewed by its success or failure on just a few specific instantiations of that law, making it difficult to draw robust conclusions about its general understanding of that principle.
2.Subjectivity and Reproducibility of Human Evaluation: The core of the benchmark relies on manual evaluation by three annotators. This methodology, while well-intentioned, introduces significant challenges for reproducibility and objectivity.
￮Potential for Bias: With only three annotators, their collective interpretation could introduce a systematic bias. Furthermore, the 4-level scoring system, particularly the distinction between "largely correct but contains minor inaccuracies" (Level 3, score 0.5) and "exhibits a clear violation of the law" (Level 2, score 0.25), is inherently subjective. Different teams of annotators could easily arrive at different conclusions.
￮Lack of Reported Agreement: The paper does not report the inter-annotator agreement (IAA), for example using metrics like Fleiss' Kappa. This is a critical omission for a benchmark that relies on human judgment, as it is the primary way to demonstrate the reliability and consistency of the annotation process. Without a high IAA, the conclusions drawn from the scores are less trustworthy.
￮Reproducibility Concerns: A key goal of a benchmark is to be a stable tool for the community. With a human-in-the-loop protocol, it is very difficult for other researchers to reproduce the exact evaluation scores. For the benchmark to have lasting value, the authors must provide extremely detailed guidelines, annotated examples, and a clear protocol for resolving disagreements to ensure that other researchers can apply the evaluation framework in a consistent manner. As it stands, the paper does not sufficiently address how this can be achieved.

### Questions
1.What was the inter-annotator agreement (e.g., using Fleiss' Kappa) for the 4-level scoring? How were disagreements between the three annotators resolved to produce the final scores reported in the paper? Providing this information is crucial for assessing the reliability of the human evaluation protocol.
Regarding the limited number of prompts per law (an average of 7), how did the authors ensure that these prompts were representative and provided sufficient coverage of the diverse ways a physical law can manifest? Is it possible that the poor performance is partly due to the selection of particularly challenging or ambiguous scenarios?

### Soundness
3

### Presentation
3

### Contribution
3

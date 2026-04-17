# CaughtCheating: Is Your MLLM a Good Cheating Detective? Exploring the Boundary of Visual Perception and Reasoning

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 4, 6

## Abstract
Recent agentic Multi-Modal Large Language Models (MLLMs) such as GPT-o3 have achieved near-ceiling scores on various existing benchmarks, motivating a demand for more challenging test tasks. These MLLMs have been reported to excel in a few expert-level tasks for humans, e.g., GeoGuesser, reflecting their potential as a detective who can notice minuscule cues in an image and weave them into coherent, situational explanations, leading to a reliable answer. But can they match the performance of excellent human detectives? To answer this question, we investigate some hard scenarios where GPT-o3 can still handle, and find a common scenario where o3's performance drops to nearly zero, which we name CaughtCheating. It is inspired by the social media requests that ask others to detect suspicious clues from photos shared by the poster's partner. We conduct extensive experiments and analysis to understand why existing MLLMs lack sufficient capability to solve this kind of task. CaughtCheating provides a class of challenging visual perception and reasoning tasks with great value and practical usage. Success in these tasks paves the way for MLLMs to acquire human-level detective perception and reasoning capabilities.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces CaughtCheating, a new benchmark designed to challenge advanced multi-modal large language models (MLLMs) like GPT-o3, which have great performance on existing benchmarks. Inspired by real-world “detective” scenarios from social media, where users analyze photos for subtle clues of suspicious behavior—the benchmark tests models’ ability to integrate fine-grained visual cues with contextual reasoning. Through experiments, the authors find that while GPT-o3 performs well on many hard tasks, it fails almost completely on CaughtCheating scenarios. The study highlights current MLLMs’ limitations in nuanced perceptual reasoning and argues that success on such tasks would mark a step toward human-level detective-like understanding.

### Strengths
1. The paper introduces CaughtCheating, a creative and practically relevant benchmark that targets subtle, context-dependent visual reasoning—an underexplored yet important aspect of real-world perception tasks.

2.The study systematically evaluates advanced MLLMs’ visual reasoning processes, revealing specific scenarios where even top-tier models like GPT-o3 fail, thus providing valuable diagnostic insight into current model limitations.

3. By connecting model failures to Guided Search theory, the paper offers an interpretable explanation of why MLLMs struggle, contributing both conceptual depth and guidance for future benchmark and model design.

### Weaknesses
1. Section 2.1 provides an informative overview of how modern MLLMs perform visual reasoning. However, the section is unnecessarily verbose (particularly lines 148–160) and contains excessive descriptions of specific model responses that distract from the main narrative. It would be more effective to distill these observations into general patterns or response tendencies across mainstream MLLMs, in line with the section title “Exploring the Boundary of Visual Perception and Reasoning.” Moreover, focusing on a single closed-source model limits the robustness and generality of the analysis.

2. The examples used to demonstrate task difficulty and model behavior are not entirely convincing. For instance, in Figure 4(a), the supposed clue (a finger) appears nearly impossible for a human to detect, raising concerns about the fairness or clarity of the ground truth. Meanwhile, GPT’s response to Figure 4(b) seems reasonable, yet it is judged as incorrect, suggesting potential ambiguity or inconsistency in the labeling criteria.

3. The dataset’s narrow focus on “catching cheating” scenarios undermines its generalizability. While the paper claims to assess general visual reasoning abilities, the data distribution is highly specific and socially contextualized. As such, conclusions such as “failing to identify deterministic clues” are not well supported by a sufficiently diverse or representative sample of reasoning tasks.

### Questions
Weakness mainly reflects my concerns.

some typos:
1. Line 037 : unnecessary line breaks
2. Line 091-092: `` detection-level'' should be ``detective-level'' as mentioned before.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces CaughtCheating, a new benchmark designed to test whether advanced multi-modal LLMs possess detective-level visual perception and reasoning skills. Motivated by real social-media scenarios where users ask others to find subtle clues contradicting a partner’s claim, the benchmark focuses on identifying extremely inconspicuous, context-dependent visual evidence, such as reflections, small objects, or social inconsistencies, that current models often overlook. The authors analyze why even state-of-the-art models like GPT-o3 fail on these tasks, drawing on Guided Search theory to show that the clues have low bottom-up salience, lack top-down guidance, and require nuanced social interpretation. They construct both a carefully curated real subset and a diverse synthetic subset, and provide automatic evaluation procedures. Experiments reveal stark limitations: even the strongest models achieve only modest accuracy on real-world cases, frequently hallucinating suspicious clues or missing decisive ones.

### Strengths
- The dataset captures subtle, real-world visual clues that existing benchmarks overlook, providing a meaningful test of advanced multimodal reasoning.
- The analysis identifies why state-of-the-art models fail, linking errors to low-salience signals and lacking top-down guidance, offering actionable insights for future model design.
- The benchmark includes both real and synthetic subsets, plus automatic scoring procedures, enabling rigorous and scalable assessment of MLLM capabilities.

### Weaknesses
- The benchmark focuses on a specific “detective-style clue finding” scenario, which may be partially included in other work and limits its generalizability to other multimodal reasoning tasks.
- Some clues in real-world images may be open to interpretation, making the correctness labels somewhat subjective.
- The prompt design is too simple, which cannot cover enough requirements in the real world for fine-grained perception and reasoning.

### Questions
- How were ambiguous or borderline “suspicious clues” handled during annotation, and what measures ensured consistency across annotators?
- Could the task design unintentionally encourage models to hallucinate suspicious clues due to its “detective-style” framing?
- Did you study whether models improve with more explicit top-down instructions (e.g., “look for reflections”)?

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
4

### Summary
The paper introduces CaughtCheating, a benchmark for detective visual perception with annotated clues and decomposed questions. It aims to find subtle, context-dependent clues in images that contradict the given claim. Using this benchmark, state-of-the-art models perform poorly, suggesting a gap between current MLLMs and detection capabilities.

### Strengths
1.	Interesting, socially relevant task that focuses on subtle visual perception and context-sensitive reasoning on image-claim pairs.
2.	The annotation schema distinguishes deterministic and non-deterministic clues in the detective task and decomposes visual reasoning questions, aiming to evaluate the performance of MLLMs.

### Weaknesses
1.	The scale of the constructed real dataset is quite small, with only 100 samples. A small real dataset raises concerns about the reliability of the benchmarking results and also weakens the contributions of benchmarking.
2.	The paper primarily constructs a benchmark and reports the limitations of current MLLMs. It does not discuss the potential direction or insightful methods to address the identified failures. 
3.	Synthetic subset generation relies on stereotypes and templated clues. The paper acknowledges that it is easier, which can distort model comparisons and may encode social biases.
4.	Results on real and synthetic datasets are significantly different, raising concerns about whether the proposed dataset can benchmark current methods and whether it can truly improve the performance of current methods.

### Questions
1.	See above.
2.	Can you report breakdowns by scene, gender presentation, and cue types to identify biased failure modes?

### Soundness
3

### Presentation
2

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
This paper addresses the gap in evaluating detective-level visual perception and social reasoning in MLLMs by proposing **CaughtCheating**, a benchmark inspired by real-world social media requests to detect suspicious clues contradicting a claim. The benchmark includes a Real Subset (100 manually curated social media images, split into Clued/Unclued) and a Synthetic Subset (3700 GPT-Image-1 generated images) to balance realism and scalability. Grounded in cognitive science’s Guided Search theory, the paper explains MLLM failures: clues have low bottom-up salience, lack top-down feature guidance, and require social context interpretation—rendering MLLMs’ exhaustive search ineffective. Extensive experiments on 17 MLLMs (open/closed-source) show top models like GPT-o3 achieve only 26.0% Clued Accuracy, 17.2% IoU, 8.0% Unclued Accuracy, and 23.9% F1 on the Real Subset, exposing limitations like hallucination in unclued images. Contributions include the benchmark, theoretical failure analysis, and insights into MLLMs’ detective capability boundaries.

### Strengths
* The paper proposes a genuinely novel and highly challenging task that pushes MLLMs beyond standard VQA or reasoning benchmarks. 

- The benchmark is thoughtfully constructed with two complementary subsets: a small, extremely difficult Real Subset that captures real-world subtlety and a large-scale Synthetic Subset for diversity. The annotation scheme is very thorough, distinguishing between "Deterministic" and "Non-deterministic" clues, and including "Decomposed" (Perception vs. Reasoning) questions, which enables a fine-grained failure analysis.

- The paper provides crucial insights into SOTA model failures. The key finding is twofold: models are not only bad at finding the *correct* clue (e.g., 26.0% Clued Acc for o3) but are *also* prone to hallucinating *incorrect* clues on innocent images (e.g., 8.0% Unclued Acc for o3). The decomposition analysis (Table 2) brilliantly supports the theoretical claims by showing that models *can* perceive the clues if prompted (high Dec. P Acc) but fail in the *search and discovery* process.

### Weaknesses
* The Real Subset (100 images) is small, dominated by hotel scenes (69%), and biased toward cisgender heterosexual couples—failing to represent diverse scenarios (e.g., workplaces) or relationships, which limits conclusions about MLLMs’ performance across real-world contexts.
* Clues in the Synthetic Subset are overly prominent (e.g., obvious lipstick, two glasses) due to prompt design, probably making it less effective at mimicking the subtlety of real-world clues (e.g., faint reflections) and reducing its value for training/testing MLLMs on authentic challenges.
* The authors rightly acknowledge the ethical limitations in Section 7. The "Real Subset" is sourced from public data that overwhelmingly represents cisgender, heterosexual couples. While the synthetic data attempts to balance gender, the entire benchmark is framed around a single, socially charged, and potentially harmful application (detecting infidelity). This narrow, biased framing limits the generalizability of the "detective-level" reasoning claims.
* The data and code not be publicly released.

### Questions
* Can you expand the Real Subset to include more scenes (e.g., offices, public transit) to address current scenes' biases?
* Did you measure difficulty (e.g., inter-model accuracy variance, human annotation time) across dimensions/sub-fields? Are there imbalances (e.g., over-representation of easy Daily Life questions) that might skew model rankings?
* Have you fine-tuned MLLMs on the CaughtCheating dataset (especially the Real Subset) to test if it improves their detective capabilities?
* Advanced models like GPT-o3 hallucinate more on unclued images—have you analyzed their reasoning traces to identify root causes? 

I look forward to an active discussion with the authors during the rebuttal phase and will revise my score accordingly.

### Soundness
2

### Presentation
3

### Contribution
3

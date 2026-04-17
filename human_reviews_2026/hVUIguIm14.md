# Off-Trajectory Reasoning: Can LLMs Collaborate on Reasoning Trajectories?

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 4

## Abstract
Reasoning LLMs are trained to verbalize their thinking process, yielding strong gains on reasoning tasks. This transparency also opens a promising direction: multiple reasoners should directly collaborate on each other's thinking on a shared trajectory, yielding better inference efficiency and exploration. A key prerequisite, however, is their abilities to assess usefulness of and build on other models' partial thinking traces -- we call this *off-trajectory reasoning*. Our paper investigates a critical question: can standard *solo-reasoning* training pipelines yield desired *off-trajectory* behaviors? To this end, we propose twin tests that capture the two extremes of the spectrum: **Recoverability**, which tests whether LLMs can backtrack from "distractions" induced by misleading reasoning traces, and **Guidability**, which tests their ability to build upon correct reasoning from stronger collaborators. Our study evaluates 15 open-weight LLMs (1.5B–32B) and reveals a counterintuitive finding -- "stronger" LLMs on benchmarks are often more fragile under distraction. Moreover, all models tested fail to effectively leverage guiding steps from collaborators on problems beyond their inherent capabilities, with solve rates remaining under 9.2% for math. Finally, we conduct control studies to isolate the effects of three factors in post-training on these behaviors: the choice of distillation teacher, the use of RL, and data selection strategy. Our results provide actionable insights for training natively strong reasoning collaborators; e.g., we find that sub-optimal recoverability behaviors of teacher models are transferred to distilled students even if the distilled data trajectories are correct. Taken together, this work introduces the framework for evaluating multi-model collaborations under shared reasoning, while revealing limitations of off-the-shelf reasoning LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
1. It investigates whether large language models (LLMs) trained for solo reasoning—i.e., generating complete reasoning chains independently—are capable of effective collaborative reasoning, where they must build on or recover from partial reasoning traces produced by other agents (e.g., other models or humans).
2. Stronger solo-reasoning models are often more fragile under distraction—high-benchmark models like AM-Thinking-32B show poor recoverability.
3. Reinforcement Learning (RL) after supervised fine-tuning (SFT) significantly improves both recoverability and guidability.

### Strengths
1. This work is among the first to systematically probe LLMs’ robustness and adaptability in multi-agent reasoning contexts.
2. The twin-test framework—Recoverability and Guidability—is conceptually original. It cleanly isolates two extremes of collaborative interaction (resisting distraction vs. leveraging guidance) and operationalizes them in a reproducible, controlled manner.
3. The stronger solo reasoners are more fragile under distraction, and no model effectively uses correct guidance to exceed its solo capability.

### Weaknesses
1. It simulates collaboration via a static, one-shot injection of an off-trajectory steer (either distracting or guiding), followed by continuation from the main model. This simplification overlooks more realistic collaborative dynamics—e.g., iterative back-and-forth exchanges, multi-turn corrections, or mixed-quality steers (partially correct, partially wrong).
2. All experiments are conducted on mathematical reasoning benchmarks. While math is a strong testbed for structured reasoning, it may not generalize to other domains (e.g., commonsense, legal, or scientific reasoning) where reasoning trajectories are less formalized or more ambiguous.

### Questions
1. Since the models used in exp are trained with different training recipe, could you use open-sourced datasets and the numbers of training dataset to study the Recoverability and Guidability at the SFT-stage? 
2. The models listed in exp are almost trained with sft-loss, could you use open-sourced datasets and RL-training to study the impact of RL for Recoverability and Guidability?

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
3

### Summary
This paper systematically investigates the capabilities of large language models in multi-model collaborative reasoning, introducing the novel paradigm of "off-trajectory reasoning." The authors design a dual-test framework of Recoverability and Guidability, revealing that current models trained for solo reasoning exhibit fragility in collaborative scenarios - stronger benchmark performers are more susceptible to interference, and all models struggle to effectively leverage external correct reasoning traces. This work exposes limitations in conventional training paradigms and provides both evaluation tools and improvement directions for building genuinely collaborative reasoning systems.

### Strengths
1. Innovative methods: The paper's innovative dual-test framework of Recoverability and Guidability provides the first systematic tools for evaluating LLMs in collaborative reasoning. This novel and practical methodology effectively uncovers model deficiencies that remain hidden in traditional solo-reasoning benchmarks.
2. Rigorous experiments: Through extensive experiments across multiple models and tasks, the paper presents a counterintuitive finding: strong solo-reasoning models do not equate to strong collaborators. This reveals an orthogonal relationship between benchmark performance and collaborative robustness, posing a significant challenge to the prevailing training paradigm that over-optimizes for leaderboard rankings.
3. Controlled variable experiments: Through controlled variable experiments, the study clearly demonstrates how teacher model selection, RL training, and data filtering strategies directly impact collaborative reasoning capabilities. These findings provide concrete guidance for industrial practitioners to optimize model training.

### Weaknesses
1. Limitations of the experimental scope: The paper's experiments are confined to mathematical reasoning tasks, limiting the generalizability of its findings to other collaborative domains like coding or scientific inquiry. Furthermore, the evaluation relies on simulated two-model interactions rather than testing in real multi-agent environments, leaving its practical efficacy unverified.
2. The analysis of model defects is relatively shallow: While the study successfully identifies failures in collaborative reasoning, it does not delve into the underlying causes. It remains unclear whether models fail to "recognize the relevance of guidance" or "cannot integrate it into their own reasoning process," and the analysis lacks mechanistic investigations (e.g., attention weights or intermediate step tracing) to pinpoint these cognitive bottlenecks.

### Questions
1. When constructing distracting reasoning traces, did the authors consider incorporating statements with logical fallacies as distractors, in addition to sampling from different questions? This would help evaluate the models' ability to discern and reject fundamentally unsound reasoning.
2. There is a spelling error in the comments below Table 1 on page 5. The word "bechmark" in "the best bechmark model" should be corrected to "benchmark".

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
2

### Summary
This paper investigates whether reasoning LLMs trained for solo reasoning can effectively collaborate by working on shared reasoning trajectories—a capability the authors term "off-trajectory reasoning." The authors propose twin evaluation tests: (1) **Recoverability**, which measures whether LLMs can backtrack from distracting reasoning traces, and (2) **Guidability**, which tests whether they can build upon correct partial reasoning from stronger models to solve problems beyond their inherent capabilities. The study evaluates 15 open-weight LLMs (1.5B–32B parameters) across five math benchmarks (AIME2024/2025, MATH-500, Minerva, OlympiadBench).

### Strengths
- First systematic study examining whether solo-reasoning training yields collaboration-ready models
- Comprehensive evaluation across 15 open-weight LLMs from 4 distinct families
- Clear problem motivation connecting to practical collaboration scenarios (efficiency, exploration, safety)
- Provides actionable insights for training better collaborative reasoners

### Weaknesses
- All experiments confined to mathematical reasoning benchmarks
- The assumption that reasoning for question q' inserted into question q constitutes a "strong distractor" is not validated
- No analysis of what specific data characteristics lead to poor/variable recoverability
- No discussion of computational costs of multi-model collaboration vs. solo reasoning

### Questions
- How do results change if steers are from different model families rather than the same model?
- Can you show cases where models successfully integrate guidance? What makes them different?
- Does synthetic data generation for off-trajectory scenarios help?

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
3

### Summary
The paper proposes two synthetic tests to assess how well LLMs can collaborate on a shared reasoning trajectory, which the authors refer to as off-trajectory reasoning. The tests are as follows:

- Recoverability: tests whether models can backtrack from incorrect reasoning to arrive at the correct answer. This is done by adding a distractor in the middle of a reasoning trace.
  - A distractor is constructed by taking a reasoning path from a *different question* and truncating it to 20% of the target reasoning length.
  - Recoverability tests are performed on questions that the model can already answer correctly.
- Guidability: tests whether the model can successfully use the guidance of a larger guide model to arrive at the correct answer. This is done by adding a portion of the teacher's reasoning at the beginning of the reasoning trace. The length is selected between 20% - 80% the guide's reasoning length.
  - Guidability tests are performed on questions where the model's solve rate is less than 2/8.

The authors perform these tests on a wide variety of models and show that the test metrics are largely orthogonal to the original *solo* reasoning capability of the models. They also perform various ablation on test parameters, and perform control studies on the post-training recipes of the models under evaluation, i.e., SFT distillation and RL.

### Strengths
- Novel approach: the authors acutely identify recoverability and guidability as two key pre-requisite abilities for LLM collaboration, and target these aspects through novel probing schemes.
- Simplicity: the two test methodologies are simple and easy to implement
- Novel findings: The main finding is counter-intuitive and novel.
- Actionable insights: The authors present clear, actionable insights from the control studies.

### Weaknesses
- W1. Contrived test protocol: the test protocol involves truncating reasoning traces and inserting these partial traces at arbitrary token locations. Also, as I understand, this happens within the internal thinking traces of the models within the, e.g., <think> tags, within the *assistant* message.
  - This introduces misalignment with the training distribution, which is unrelated to the model abilities which the authors intend to evaluate. Firstly, truncating and inserting at arbitrary token indices can lead to incoherent sentences. Also, in standard SFT pipelines, the training data for the *assistant* message portion of the samples are meant to be outputs from the model itself. Furthermore, in RL, the *thinking* part of the assistant message is only trained on outputs generated by the model itself. The proposed test protocol introduces contexts that are significantly misaligned with this training distribution. I.e., placing the external reasoning in the thinking part of the assistant message indicates to the model, perhaps deceivingly, that it originated from the model itself, rather than some external model. The input distribution shift due to the *protocol* will likely cause significant performance degradation, unrelated to the *semantic* distribution shift of distractor reasoning or guidance from an external collaborator LLMs, on which the author *intend* to evaluate the models.
  - While some amount of input distribution shift is inevitable in any inference-time probing scheme, I believe the contrived nature of the  specific approach proposed by the authors may cause excessive distribution shift, which would be avoided in potential real-world methods used for LLM collaboration. For example, it may be possible to include external distractor or guidance reasoning as a user message, indicating the that the reasoning does not originate from the current model. The model will then be inclined to identify potential errors or hints and act accordingly.
- W2. Limited task domain: The study is limited to a single domain, math reasoning. Therefore, the findings may not generalize to other domains such as coding, creative writing, etc., where distractions and guidance may have different characteristics.

### Questions
Regarding W1, I'm not sure if we necessarily want or need models to perform well on this test for them to collaborate well with other models. I do not believe the test scores will align with the model's ability when applying realistic inference-time or training-based collaboration methods.

I'm willing to raise my score if the authors can provide a strong argument against this concern.

### Soundness
3

### Presentation
3

### Contribution
2

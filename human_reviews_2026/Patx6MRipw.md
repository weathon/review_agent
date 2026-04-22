# ENACT: Evaluating Embodied Cognition with World Modeling of Egocentric Interaction

- Avg Score: 4.80
- Decision: Accept (Poster)
- Scores: 6, 4, 4, 6, 4

## Abstract
Embodied cognition argues that intelligence arises from continuous sensorimotor interaction with the world. It raises an intriguing question: do modern vision-language models (VLMs), trained largely in a disembodied manner, exhibit signs of embodied cognition? To investigate this, we introduce ENACT, a benchmark that probes this question through world modeling from egocentric interaction. Grounded in a partially observable Markov decision process (POMDP) framework, ENACT comprises two complementary sequence reordering tasks: forward world modeling (predicting an ordered sequence of future states from actions) and inverse world modeling (inferring an ordered sequence of actions from state changes). Correctly solving these tasks indicates that the model has a solid understanding of how the environment will evolve given one's actions. Our scalable dataset contains 8,972 QA pairs derived from diverse, long-horizon household activities in the BEHAVIOR simulator. Experiments reveal a significant performance gap between state-of-the-art VLMs and humans, which widens dramatically as interaction horizons lengthen. We find that models consistently solve the inverse problem better than the forward one and exhibit strong embodied biases, showing a preference for right-handed actions and performance degradation with camera perspectives that deviate from those of human vision.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces ENACT, a benchmark for testing embodied cognition in Vision-Language Models via world modeling from egocentric interaction. Using BEHAVIOR simulator trajectories, it builds scalable sequence-reordering tasks: Forward (order future images given actions) and Inverse (order actions given images), with a semantic verifier over symbolic scene-graph deltas. Across 8,972 QAs (29 activities; horizons L=3–10), state-of-the-art VLMs lag far behind humans, degrade sharply with longer horizons, and perform better on inverse than forward. Probing shows robustness to rendering realism but sensitivity to non-human camera intrinsics and a notable right-handed bias. Error analysis reveals failures dominated by omissions and hallucinations of state changes, positioning ENACT as a scalable, diagnostic stress test for consequence-aware, long-horizon reasoning.

### Strengths
S1. 8,972 QAs over 29 household activities, balanced lengths (L=3…10) and 11 predicate classes, with visibility checks and de-duplication—good breadth while keeping semantics precise. Also, evaluation spans 7 proprietary and 22 open-weight VLMs with unified prompts/decoding, plus a robust human baseline – this breadth makes results meaningful to the community.

S2. Converting predictions to atomic state-change signatures and breaking errors into omission, hallucination, polarity inversion, predicate/entity substitution gives rare, fine-grained insight into failure modes.

S3. Scalable, well-engineered data pipeline:  The DAG+DP path-counting and weighted backtracking for key-frame synthesis is technically neat and makes long-horizon QA generation tractable without manual curation.

### Weaknesses
W1 . ENACT reduces forward/inverse dynamics to sequence reordering under a symbolic action space and an online subset-based verifier. A model can succeed by learning robust pairwise visual diffs (relying on dection of local difference and adjacent changes) and text matching without learning an internal, action-conditioned dynamics model (e.g., no counterfactual rollouts, no latent transition consistency beyond local adjacency). That undermines the claim that scores reflect “world modeling” rather than discriminative sorting.

W2. Several benchmarks already test temporal ordering, causal clip inference, or egocentric state changes. ENACT’s main novelty is pairing reordering with symbolic scene-graph deltas in a simulator. But it still evaluates static post-hoc reasoning over pre-rendered frames—not interactive rollouts, planning, or closed-loop control. For ICLR standards, this may be too incremental without a stronger theoretical or algorithmic contribution. Meanwhile, the “image realism” study shows no significant differences, but all variants are style transforms of the same rendered content with the same camera geometry. That cannot test the core sim-to-real generalization problem (novel textures, clutter, domain shift in object distributions, sensor noise, real lens artifacts). Thus in terms of data quality, there’s no significant advantage over previous works.

### Questions
The semantic verifier uses subset inclusion rules. How often do accepted predictions omit/hallucinate parts of transitions? 

Limiting to 11 predicates and “visible” deltas excludes many dynamics (forces, latent preconditions). Do you have a plan to include non-visible or continuous variables, and how would the verifier adapt?

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
The authors propose a framework called ENACT to evaluate if vision language models (VLMs) can exhibit embodied cognition. The framework uses 2 main sequence reordering tasks (forward and inverse world modelling) and data collected from the egocentric robot manipulation simulator BEHAVIOR from a diverse set of household scenarios. Authors conduct an extensive analysis on 29 VLMs (both open and closed source) on this framework and compare it with human evaluation from 3 annotators, with strong agreement reported. They also present and discuss failure modes of VLMs.

### Strengths
- The authors conduct an extensive evaluation on 29 VLMs (both open and closed source), on increasing horizons lengths. They then use the most promising models (one open InternVL3.5, one closed GPT5-mini) and conduct a further set of experiments to explore the VLMs sensitivity to image realism, camera configurations and embodied biases.
- The paper is clearly presented and easy to follow, with descriptive and detailed figures, comprehensive discussion on each dimension evaluated and a good overview of the data collection process.
- The work offers a good overview of the limitations of current VLMs models and future directions for research.
- The failure modes analysis in section 3.5 is really informative, highlighting the challenges of hallucinations and omissions in extending VLMs to embodied tasks.
- The authors are happy to opensource the code and the QA dataset to benefit the research community.

### Weaknesses
- I'm unsure about the claim the authors make on VLMs being trained on vast static datasets. Recent VLMs, including those used as benchmarks in this work make extensive use of video data during training which include temporal dynamics (motion, causality, progression), as well as context across frames.
- I would've liked to see more details about the online verifier, which is key in computing the 2 main metrics: task and pairwise accuracy.

### Questions
- Have you considered discussing how more recent pretrained world models can act in embodied settings, as opposed to world modelling via VLMs?
- In Fig 2, should N, the number of trajectories, be replaced with R, as described on line 132?

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
3

### Summary
### ENACT: Evaluating Embodied Cognition with World Modeling of Egocentric Interaction

#### Main Research Problem

* **Embodied Cognition (EC)**: The theory posits that intelligence arises from continuous sensorimotor interaction with the world.
* **VLM Limitation**: Modern Vision-Language Models (VLMs) are trained **disembodiedly** on vast static datasets.
* **Core Question**: **To what extent does embodied cognition emerge** in these VLMs despite their non-interactive training, specifically their ability to understand the **consequences** of actions and the dynamics of the environment?

#### Research Methodology (The ENACT Benchmark)

* **Framework**: Introduced the **ENACT** benchmark, grounded in a Partially Observable Markov Decision Process (POMDP) framework, to evaluate EC through **egocentric** (first-person) interaction world modeling.
* **Task Types**: Two complementary **sequence reordering** tasks test a model's understanding of how the world evolves with its actions:
    1.  **Forward World Modeling**: Given a sequence of abstract actions, the model must predict and correctly order the sequence of **future states**.
    2.  **Inverse World Modeling**: Given a sequence of state changes (observations), the model must infer and correctly order the corresponding **actions**.
* **Data**: A scalable dataset of 8,972 QA pairs derived from diverse, long-horizon household activities in the BEHAVIOR simulator.

#### Key Experimental Findings

1.  **Significant Gap**: State-of-the-art VLMs (including GPT-5) show a **substantial performance gap** compared to human performance.
2.  **Horizon Failure**: Model accuracy **drops sharply** as the interaction time horizon (number of steps) increases, indicating a lack of robust, **long-term** understanding of action consequences.
3.  **Embodied Biases**: VLMs exhibit distinct embodied biases, such as a preference for recognizing **right-side actions** and a significant performance drop when camera views deviate from typical human perspectives.
4.  **Error Analysis**: The primary error modes for top models are **Omission** (failing to recognize a state change that occurred) and **Hallucination** (recognizing a state change that did not occur).

#### Conclusion

The ENACT benchmark demonstrates that while powerful, current VLMs lack the necessary robust, long-term understanding required for embodied world modeling, indicating that **key elements of embodied intelligence have not fully emerged** from their static training paradigm.

### Strengths
### Main Strengths of the ENACT Paper

* **1. Novelty and Comprehensiveness in Evaluation (Consequence-Aware World Modeling)**
    * **Summary**: ENACT elevates the evaluation of embodied AI beyond simple perception or isolated interactions to **consequence-aware world modeling** over **extended time horizons**.
    * **Detail**: It introduces complementary **Forward** (Action $\rightarrow$ State) and **Inverse** (State $\rightarrow$ Action) sequence reordering tasks, providing a holistic test of a VLM's ability to reason about the **causal effects** of its actions, unlike prior static or limited benchmarks.

* **2. Focus on Egocentric Interaction and Fidelity (Robust POMDP Framework)**
    * **Summary**: The benchmark is explicitly designed around **egocentric (first-person)** interaction trajectories within a **Partially Observable Markov Decision Process (POMDP)** framework.
    * **Detail**: By utilizing data from the **high-fidelity BEHAVIOR simulator**, ENACT ensures the evaluation tasks are realistic and relevant to the challenges faced by embodied agents operating in complex, dynamic household environments.

* **3. Effective Diagnosis of SOTA VLM Failures (Long-Term Reasoning and Biases)**
    * **Summary**: ENACT provides a crucial diagnostic tool by successfully revealing fundamental weaknesses in the **state-of-the-art VLMs** (e.g., GPT-5) concerning long-term embodied reasoning.
    * **Detail**: Experimental findings pinpoint specific failure modes—such as a **sharp performance drop** with increasing horizon, and prevalent **Omission** and **Hallucination** errors—which are essential for directing future research efforts in embodied AI development.

### Weaknesses
### Main Limitations of the ENACT Benchmark (Weaknesses)

* **1. Reliance on Simulation and the Sim-to-Real Gap**
    * **Summary**: The entire ENACT dataset and evaluation are restricted to high-fidelity trajectories within the **BEHAVIOR simulator**.
    * **Detail**: This introduces the inherent **sim-to-real gap** limitation. The model's strong performance (or failures) in the clean, deterministic physics and visuals of the simulation may not directly translate to the stochastic, noisy, and complex dynamics of a real-world robotic agent.

* **2. Abstraction of States and Actions (Missing Low-Level Embodiment)**
    * **Summary**: The benchmark evaluates understanding based on high-level **symbolic scene graphs** and **abstract action primitives** (e.g., `Open(object)`), rather than raw sensorimotor data.
    * **Detail**: By removing the necessity to process raw pixels, continuous motor commands, and solve low-level control problems, ENACT fails to test critical aspects of true embodied cognition related to continuous planning and fine-grained sensor processing.

* **3. Limitation to Sequence Reordering (Absence of Generative Modeling)**
    * **Summary**: The core evaluation is a **sequence reordering task**, which tests recognition and temporal alignment against a limited set of choices.
    * **Detail**: This task is less demanding than **free-form generative world modeling**, where the VLM would have to *generate* novel, unconstrained future states or action sequences from scratch, which is the ultimate goal of a robust world model.

* **4. Focus on Functional Dynamics (Neglects Fine-Grained Physics)**
    * **Summary**: The benchmark primarily assesses object-level, functional state changes (e.g., object is open/closed, object is grasped/placed).
    * **Detail**: It may insufficiently test complex, fine-grained **physical reasoning** such as geometric constraints, stability, momentum, or soft-body dynamics, which are integral to sophisticated manipulation in unstructured environments.

### Questions
1. I still have some doubts about the selection of keyframes. The author emphasized during the construction of the dataset to avoid having repetitive frames and overly similar frames as adjacent keyframes, and proposed a series of methods for this purpose. However, what readers are concerned about is whether it is no longer similar to the previous frame. This is particularly crucial throughout the entire trajectory. In the readers' understanding, a key frame contains sufficient event information. Can the author propose a series of criteria to measure this?

2. In my opinion, this is actually a piece of work that assesses the model's ability to understand the embodied world. Since the author attempts to propose a benchmark or dataset, it seems that more measurement criteria are necessary in the eyes of the readers. The current embodied tasks do not seem to be satisfied with merely evaluating the "ordering tasks" of forward and backward modeling. And if such a benchmark only includes these two evaluation tasks, then a series of video generation models such as Sora would be excluded (we also observed this in the experimental part). However, we know that many of the current world models are actually constructed based on image/video generation models. This would significantly limit the widespread use of benchmarks.

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
The paper proposes ENACT, a simulator-derived benchmark that tests whether VLMs exhibit embodied cognition through sequence reordering tasks grounded in a POMDP. The benchmark includes two tasks: forward world modeling (ordering future images given the action sequence) and inverse world modeling (ordering actions given the image sequence).

The authors derive their data from BEHAVIOR demos, which they convert to key-frame trajectories by detecting abstract scene graph deltas. They form QAs by shuffling future states or actions and employ a parser-friendly output along with an online verifier.
Testing across 29 long-horizon household activities (8,972 QAs, a passing number for benchmark of this kind), the results show that SOTA VLMs underperform humans, particularly as horizons lengthen. The models consistently find the inverse task easier than forward modeling. The probing experiments reveal interesting patterns: the models show little sensitivity to rendering realism but experience sizable performance drops with non-human-like camera intrinsics and heights. They also exhibit a clear right-handed bias.

Overall, ENACT provides a scalable and reasonably clean probe of consequence-aware reasoning in egocentric interaction. While the benchmark offers meaningful signals about embodiment, these insights remain simulator-bound in nature.

### Strengths
- The benchmark design is solid and principled. It uses POMDP framing with two tasks formulated as permutation prediction to avoid hand-crafted distractors. The forward task requires reordering images to match a given action sequence, while the inverse task requires reordering actions to match a given image sequence. An online verifier allows multiple valid permutations, and the metrics include Task Accuracy (exact match) and Pairwise Accuracy (adjacent consistency). The benchmark achieves solid human IAA ($α = 0.83$).

- The probe studies are focused and actionable, examining camera priors and handedness rather than just presenting leaderboard tables. The findings are particularly interesting. For image realism, there's no significant effect across path tracing, style-transfer, or ray-tracing variants. However, camera intrinsics and height matter considerably: performance drops with large apertures, fisheye lenses, and high camera positions, with best results using human-like intrinsics. The models show clear embodied biases, including a right-handed advantage with higher precision/recall and lower mixing. The error taxonomy reveals that failures are dominated by Omissions and Hallucinations of state changes.

- The main evaluation is mostly solid, testing 7 proprietary and 22 open-weight VLMs against human performance as reference. Results consistently show that models perform better on inverse tasks than forward tasks, accuracy falls monotonically with L, and there's a large gap between human and model performance at longer horizons.

### Weaknesses
- The central research question, in my view, would benefit from clarification. The phrasing “to what extent does embodied cognition emerge from such training?” seems to suggest an over-time evaluation across training stages, yet the current evaluation (on end models; based solely on benchmark performance) cannot adequately address that. VLMs typically undergo multiple distinct training phases (e.g., large-scale language pretraining, multimodal alignment, instruction tuning, reinforcement learning for human preference / reasoning). Therefore, the results can at best reveal how well these models are capable of solving embodied tasks, rather than whether or how embodied cognition itself emerges during training.
- The benchmark relies heavily on simulator semantics. Scene-graph deltas and visibility constraints are simulator-privileged, which simplifies supervision and evaluation but may bake in ontology choices that models can exploit while limiting transfer to real-world scenarios. **The authors should consider adding a real-video subset (such as curated egocentric clips with weak labels) to stress test ontology mismatch.** The scene graph itself also represents a strong choice by assuming graph structure, which may not fully capture the complexity of real-world visual reasoning.
- (I am less sure about this, authors can correct me if I'm wrong.) Key-frame sampling by predicate deltas seems to underweight fine motor dynamics and overemphasize symbolic toggles (Open/Inside/OnTop). Authors should consider an ablation with low-level pixel change gates or dense optical-flow checkpoints to verify that results aren't an artifact of symbolic sparsity.

### Questions
**[Q1]**: The paper asks “to what extent does embodied cognition emerge from such training,” but the benchmark measures sequence ordering accuracy. How exactly does success on this proxy reflect embodied cognition rather than just temporal correlation learning?
- Action: Include an operational mapping, e.g., a table or diagram linking each task (forward/inverse) to specific cognitive constructs (for example, causal modeling, temporal abstraction, affordance understanding, ....).

**[Q2]**: The inverse task may benefit from language priors, aka models are better at mapping visuals to familiar verbs than to unseen symbolic actions. How robust is this advantage when action descriptions are out of vocabulary?
- Action: Re-run a subset of evaluations with non-linguistic action encodings (e.g., icon sequences or structured tokens) and a held-out predicate set. Report whether the inverse > forward gap remains when language grounding is removed.

**[Q3]**: Because multiple QAs come from the same trajectory, the reported averages might conflate model ability with activity-specific bias. Are results consistent across different activities and noise levels in the abstract deltas?
- Action: Provide a per-activity breakdown of Pairwise Accuracy, plus variance across random seeds where $∆Vis$ predicates are randomly perturbed or dropped. This would clarify whether conclusions hold beyond a few well-behaved trajectories.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
To investigate whether embodied cognition emerges in VLMs, the authors introduce ENACT, a benchmark designed for two sequence reordering tasks: 1. Forward World Modeling: Predicting an ordered sequence of future states (egocentric RGB images) from an initial observation and a set of actions. 2. Inverse World Modeling: Inferring an ordered sequence of actions from an ordered sequence of states.

Using both proprietary and open-weight models, the authors find:
(1) a significant performance gap across VLMs,
(2) all models perform better on inverse than forward modeling, and
(3) models exhibit embodied biases.

### Strengths
- Compares the ability of VLMs to perform forward vs. inverse world modeling on the Behavior1k dataset with egocentric inputs.

- Provides several novel discussions and insights.

- Demonstrates consistent trends between open- and closed-source baselines.

### Weaknesses
1. Novelty:

How does this approach compare to GVL[1]? GVL also performs zero shot reordering task using closed-source VLMs and reports high success rates on existing embodied benchmarks. I wonder if ENACT is just evaluating different models using GVL's proposed methods? Also, a citation is missing. 

2. Limited Discussion of Model-Specific Differences:

The paper mentions that the tested VLMs are trained on “static” datasets, and I think the static here is unclear what this means (image-only/dis-embodied data?). For instance, Cosmos-Reason1, which is trained on embodied datasets, performs the worst. If ENACT is designed to measure embodied cognition, I would expect Cosmos-Predict1 to outperform other baselines trained not in the embodied domain.

3. Real-World Comparisons:

The “real-world” comparisons appear to rely on GPT-image-1 performing style transfer, and comparing across ray tracing and other rendering effects disabled in the simulator, and the authors conclude certain baselines are indifferent to rendering changes, thus they are robust to the sim-to-real gap. I think there's still a gap between real-world captured data and style-transfered data, and degrading rendering effects do not account for real-world challenges such as lighting / physics simulation difference etc.

[1] Ma, Y. J., Hejna, J., Wahid, A., et al., Vision Language Models are In-Context Value Learners, 2024.

### Questions
- If inverse world modeling is consistently easier than forward modeling, what implications does this have? What directions do the authors foresee for future work? Does this asymmetry apply specifically to VLMs, or might it generalize to generative models or unified VLM architectures?


I would consider raising my score accordingly if my questions are sufficiently answered.
I also encourage the authors to consider open-sourcing the ENACT benchmark.

### Soundness
3

### Presentation
3

### Contribution
2

# Semantic World Models

- Decision: Reject
- Scores: 6, 4, 4, 6

## Abstract
Planning with world models offers a powerful paradigm for robotic control. Conventional approaches train a model to predict future frames conditioned on current frames and actions, which can then be used for planning. However, the objective of predicting future pixels is often at odds with the actual planning objective; strong pixel reconstruction does not always correlate with good planning decisions. We posit that instead of reconstructing future frames as pixels, world models only need to predict task-relevant _semantic_ information about the future. To do this, we pose world modeling as a visual question answering problem, about semantic information in _future frames_. This perspective allows world modeling to be approached with the same tools underlying vision language models. We show how vision language models can be trained as "semantic world models" through a supervised finetuning process on image-action-text data, enabling planning for decision-making while inheriting many of the generalization and robustness properties from the pretrained vision-language models. We demonstrate how such a semantic world model can be used for policy improvement on open-ended robotics tasks, leading to significant generalization improvements over typical paradigms of reconstruction-based action-conditional world modeling.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces Semantic World Models (SWM), which frame world modeling as predicting high-level semantic outcomes via future VQA instead of reconstructing pixels. The authors trained a VLM (PaliGemma-3B) to answer questions about the future given current observations and actions. Experiments on planning benchmarks (LangTable and OGBench) show clear gains over pixel-based and offline RL baselines.

### Strengths
- Predicting high-level task-relevant info instead of generating pixels makes a lot of sense for effective world model-based planning
- The methodology is simple, clear, and sound.
- The paper is well written and quite easy to follow.

### Weaknesses
- The translation from goals to QA pairs is manual. Trying some LLM prompting for this step would be interesting. Otherwise, I suggest acknowledging this point in *Limitations*. For the current benchmarks, manual translation may be acceptable, but it will become a bottleneck for more diverse scenarios in the future.

- I am unsure about SWM’s task generalization. Given a new task, do we always need to train a separate SWM? Some discussion and exploration of this would be valuable for future work that goes beyond LangTable and OGBench. For example, one additional experiment could be to fine-tune PaliGemma jointly on the SAQA samples from both domains and then compare the effectiveness.

- In Section 3, it would be nice to frame the Semantic World Model as a more general framework, with the current VLM + yes/no VQA setup presented as one particular instantiation. This is a minor point—the current style is clear and straightforward.

- One additional potential advantage to highlight over generative world models is computational efficiency. Including some brief quantitative results would be strong.

Suggested related works for better coverage:

- *Efficient Exploration and Discriminative World Model Learning with an Object-Centric Abstraction*: abstract world models for planning. There are strong connections between SWM and the semi-MDP frameworks reviewed there, which also provide examples of how to formalize SWM.  
- *Planning with Reasoning using Vision Language World Model*: planning with a VLM-based world model. The idea of language-based abstraction is shared.  
- *Predicate Invention for Bilevel Planning*: related to the proposed future-VQA setup—their predicates act as verifiers of desired states.

### Questions
See weakness

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
The paper reframes world-model planning as answering semantic questions about the future. A Semantic World Model (SWM) is built by fine-tuning a VLM (PaliGemma-3B) to answer future-conditioned questions given an image, an action sequence, and a natural-language query. Actions are mapped into the LLM token space by a learned projection inputs concatenate image embeddings, action embeddings, and question tokens; training uses cross-entropy on the answer token. 

For planning, each task is specified as a small set of questions, desired answers, and weights. The value of a candidate is the (weighted) likelihood that SWM produces the desired answers, optionally summed over sub-chunks to reward earlier completion (Eqs. (1)–(2)). The paper instantiates both MPPI (sampling-based; Eq. (3)) and a gradient-based improvement over a diffusion base policy. 

At test time, the planner uses a fixed, per-task question set (with weights) to score sequences; Table 7 lists the question templates used for LangTable/OGBench. During training, future-QA labels are programmatically generated from oracle future state across multiple horizons to form SAQA supervision.

Empirically, MPPI on SWM attains near-perfect success on simpler tasks (e.g., 100% on LangTable reach/separate; 97% on OGBench reach). Gradient-based planning provides substantial policy improvement over IDQL and an action-conditioned video-diffusion world model (AVD) while being far faster in wall-clock than AVD.

### Strengths
- Originality — “future QA” as control objective. Defining task value via SWM answer likelihoods aligns the model objective with decision-making and avoids pixel prediction; the action-token projection turns a pretrained VLM into an action-conditioned model with minimal surgery. 

- Quality — clear task specification and planners. Tasks are explicitly defined via question/answer/weight sets (Table 7), and the value function includes a sub-chunking mechanism to encourage earlier completion (Eqs. (1)–(2)). Both MPPI (with a softmax-weighted update; Eq. (3)) and gradient-based optimization are implemented. 

- Clarity — datasets and questions are concrete. SAQA generation from oracle futures is described with multiple horizons and phrasings; Appendix A.3 enumerates question types and variations. Figures/tables make the setup legible. 

- Significance — strong planning and efficiency results. MPPI planning achieves near-perfect success on simple tasks; gradient-based planning improves a diffusion base policy and is orders of magnitude faster than AVD in time per action chunk (1.56 s vs. 676 s; MPPI 4.48 s).

### Weaknesses
- Privileged-state supervision limits real-world portability. SAQA labels depend on oracle future state; acquiring comparable labels on hardware is challenging. A path to weaker supervision (pseudo-labels from frozen VLMs, success detectors) would strengthen applicability. 

- Baseline coverage for latent world models is narrow. Comparisons focus on AVD (pixel-level) and IDQL; omitting modern latent world-model planners (e.g., Dreamer/TD-MPC-style) makes it hard to isolate the benefit of semantic QA vs. latent prediction, under matched wall-clock. 

- Calibration and reward design are underexplored. The value function sums (log-)likelihoods of desired answers across sub-chunks; language-model probabilities can be miscalibrated. An ablation on temperature/label smoothing and sensitivity to the chunk size and weights would test robustness. 

- Execution details for MPPI are implicit. The paper specifies the MPPI distribution update (Eq. (3)) and reports speeds “per action chunk,” but it does not explicitly state whether the controller executes the updated mean plan (common in MPPI) or the best sample; please clarify. (For the gradient-based planner, they do state “execute 4 of 16 actions and replan.”) 

- Compute trade-offs and reliance on a base policy. MPPI with large models is computationally heavy; the main gains on harder tasks come from gradient-based refinement of a diffusion base policy. Reporting matched wall-clock performance “from scratch” would sharpen the picture. 

- Semantic scope is mostly binary. Table 7 shows binary QA templates; a few non-binary or numeric queries (counts, distances) would demonstrate broader semantic control.

### Questions
- Task questions at test time. Confirm that, for each task, you use the fixed per-task question set (with weights) in Table 7 to score candidates. Have you tried automatically deriving questions from a natural-language instruction at test time? 

- MPPI action execution. Given Eq. (3), do you execute the updated mean sequence (standard MPPI) or the best-scoring sample in the receding-horizon loop? Please describe the exact selection rule used during roll-out. 

- Calibration & value design. How calibrated are answer probabilities? Any temperature scaling or label-smoothing results? Sensitivity of performance to sub-chunk size and weights in Eqs. (1)–(2)? 

- Weaker supervision for SAQA. Any preliminary results on generating future-QA labels without oracle state (e.g., pseudo-labels from a frozen VLM on future frames, or success detectors)? 

- Baselines under compute parity. Can you include a compact comparison against a latent world-model planner (e.g., Dreamer or TD-MPC-style) under matched wall-clock and horizon? 

- From-scratch planning on complex tasks. Beyond the simple cases where MPPI succeeds, can SWM plan without a base policy on multi-step or harder OGBench tasks under moderate compute? If not, what dominates the failure (search, query design, calibration)? 

- Comparison to automatic question discovery: Work like https://arxiv.org/abs/2410.23156 and https://arxiv.org/abs/2501.00296 seems highly relevant where they automatically find task-relevant questions and use it for planning. Could you add a comparison to these approaches, clarifying similarities/differences in how questions are generated, how they interface with the planner, and the supervision required?

### Soundness
2

### Presentation
2

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
This paper reframes world modeling as predicting task-relevant semantics rather than pixels, casting it as a visual question answering problem about future scenes. This allows the use of vision-language models (VLMs) as “semantic world models,” trained via supervised finetuning on image–action–text data. These models support planning and policy improvement with stronger generalization and robustness than reconstruction-based methods.

### Strengths
1. New framework: The core idea of shifting from pixel-level prediction to semantic, question-based prediction is novel and compelling. It directly addresses a known weakness of many video-based world models

2. Empirical results: The SWM achieves impressive results (Figure 5, Table 8), significantly outperforming the base policy and other baselines (IDQL, AVD).

3. Effective use of suboptimal data: The paper shows (Table 2) that model performance improves when trained on a combination of expert and suboptimal data, compared to using expert data alone.

### Weaknesses
1. The paper posits that a VLM could be used to decompose a high-level goal into these QA pairs (Section 2), but this is not demonstrated. The method requires a human to meticulously design a "curriculum" of questions to define a task, which is not scalable.

2. Comparisons: The comparison to the "Action Conditioned Video Diffusion" (AVD) baseline is not a fair "apples-to-apples" comparison. The AVD model is used to predict a future frame, and then the authors' own SWM model is used to perform VQA on that predicted frame to get a reward. This setup unfairly benefits the SWM's semantic reward structure. A more appropriate baseline would be to use a standard pixel-based model (like Dreamer or PlaNet) with a standard planning algorithm (like CEM) optimizing a task-specific reward (e.g., L2 distance to goal), rather than forcing it into the SWM's VQA-based reward framework. Furthermore, the reported planning time for AVD (676 seconds, Table 9) is so high it suggests a non-optimized implementation, further calling the comparison into question.

3. Lack of real-world application: The entire evaluation is conducted in simulation (LangTable, OGBench). The paper makes broad claims about a "powerful paradigm for robotic control" but provides zero evidence of the method's feasibility in the real world.

### Questions
All of my qeustions are listed in the weakness section. If my concerns are well addressed, I will raise my rating.

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
3

### Summary
The paper proposes Semantic World Models (SWM), reframing world modeling as future-tense visual QA instead of future pixel/latent prediction. A pre-trained VLM is adapted to answer questions about the outcomes of candidate action sequences; these answer likelihoods are aggregated to score trajectories. A simulation-generated SAQA dataset provides future Q/A labels (using privileged state). SWM is used for planning via (i) MPPI sampling and (ii) gradient-based refinement atop a base policy. On LangTable and OGBench, SWM improves task success over a pixel-diffusion world model and an offline RL baseline, shows some OOD robustness, and benefits from mixed-quality data.

### Strengths
- Novel formulation: The paper introduces a novel conceptual framing of world models as future-tense semantic predictors rather than pixel predictors. This reframing is non-trivial and departs meaningfully from both pixel/latent-based world modeling and language-conditioned policy architectures (e.g., VLAs), offering a new axis for research in decision-making with foundation models.

- Planning experiments: Compatible with both sampling and gradient refinement; the latter gives strong policy-improvement behavior.

- Empirical signal: Consistent gains over a pixel-diffusion world model and an offline RL baseline; robustness in some OOD settings; positive results with mixed-quality data.

- Clarity: The paper is generally well-written with helpful diagrams and ablations that support the core message.

### Weaknesses
- SAQA depends on privileged state: The central data engine uses oracle simulation state to label future Q/A tuples; there’s no empirical demonstration of a real-data pipeline (weak/self-supervised QA labels, multi-view, proprio/contact cues, etc.). This is a potential barrier for training this method on real-world data and further deployment and, IMO, the main limitation.

- From-scratch planning is not yet practical: Sampling with a large VLM is slow; the most effective mode is gradient refinement seeded by a competent base policy. The method thus acts as a policy-improvement operator rather than a standalone planner on harder tasks.

- Limited scope of evaluation: Simulated tabletop tasks with relatively short horizons; no real-robot trials and limited analysis of long-horizon multi-object goals.

- Baselines could broaden: Missing comparisons to language-space/CoT-style planners or semantic reward models trained directly from pixels; these would clarify where SWM’s semantic scoring helps most.

- Under-reported engineering details: No clear compute/latency budgets for MPPI vs gradient planning (per-timestep wall-clock, VRAM, #VLM passes), and limited sensitivity analyses (prompt phrasing, chunk length, weight selection).

### Questions
1) Real-world SAQA without privileged state: How do you envision generating future QA labels on real robot data where ground-truth object state is unavailable? (e.g., weak labels, VLM consistency, multi-view cues)

2) Planning compute & latency: What is the per-timestep compute cost for MPPI and gradient refinement (forward/backward passes, approximate ms per step)? This will clarify practical control rates.

3) Prompt / language robustness: How sensitive is planning performance to rephrasing of the questions at test time (synonyms, negations, multi-object phrasing)? 

4) Dependence on base policy: Can the gradient-based planner make meaningful progress with a weaker or absent base policy, or is a strong initialization required?

5) Task spec & weights: How sensitive is performance to the manually chosen question weights, and do you see a viable path to automatically learning or deriving them?

### Soundness
3

### Presentation
3

### Contribution
3

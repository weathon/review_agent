# Examination Feedback Simulation: Beyond Static and Unique Clinical Trajectories

- Decision: Reject
- Scores: 4, 6, 4, 2

## Abstract
Large language models (LLMs) have shown significant promise in healthcare applications. To better mirror real-world settings for LLM evaluation, dynamic longitudinal diagnosis-and-treatment simulation with virtual patients has recently emerged as a focal point of research. However, existing simulation frameworks are constrained by the limitation of clinical trajectory uniqueness, where virtual patients can only provide feedback based on information available in static electronic health records (EHRs). This limitation leads to simulation failures when unrecorded but medically sound examinations are ordered. In this paper, we formulate the task of Examination Feedback Simulation to address this limitation, which aims to dynamically augment the unique trajectory by simulating medically plausible examination results in response to clinical orders. To support this largely unexplored research, we construct ClinTrack, a dataset curated from MIMIC-IV. ClinTrack is organized in a hierarchical, chronologically-ordered structure to facilitate sequential clinical tasks. We further propose a structure-aware evaluation metric SimScore to quantitatively assess the quality of simulated results, which shows promising initial alignment with expert judgment. Building on this framework, we develop ClinSim, a new generative model specifically designed for this task. Experiments demonstrate that our 4-billion-parameter ClinSim model significantly outperforms flagship models up to 235B parameters on this task, achieving an improvement of over 10 percentage points in SimScore, providing a critical foundation for creating more dynamic and realistic virtual patients.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work makes three major contributions in an effort to allow more realistic evaluation of LLMs for sequential clinical decision making, particularly in cases where the desired test by the LLM is not present in the patient's record and as such needs to be generated. It introduces ClinTrack, a new dataset which structures clinical sequences hierarchically and chronologically. It introduces a new metric, SimScore, by which to evaluate this task of novel event generation. Lastly, it introduces ClinSim, a model based on Qwen3-4B which is trained using ClinTrack to generate realistic and appropriate clinical events given a patient's record. With this set-up, the authors report an 11.86% improvement of SimScore using ClinSim to generate new patient events.

### Strengths
This work has many strengths. First, it is well motivated that as we evaluate LLMs on real-world tasks, we face the problem that we may incorrectly penalize these agents for not matching real world data although their counterfactual recommendation is valid in its own respect. LLMs are used a lot in healthcare now but we can’t do proper counterfactual analysis when an LLM orders a test without a result, so ClinSim's ability to generate realistic events with an improved performance compared to base models is significant in regards to evaluating models in the future. Further, ClinTrack, ClinSim, and SimScore could each serve as high-significance contributions in their own right, creating a novel dataset, model, and evaluation scoring method for novel data types. Lastly, the authors integration of clinical feedback into the evaluation of their model is a valuable sanity check, allowing users to verify that a clinician deems the output generations of ClinSim to be higher quality than base models, aligning with SimScore judgement.

### Weaknesses
The paper suffers from a lack of clarity in the writing/organization as well as a lack of strong baselines for comparison. Generally, the paper would benefit from a running example throughout the text such that the reader can understand what the structure of the data at each point looks like.  For example, Figure 2 is generally confusing and does not clearly explain how ClinTrack is generated. Further, I assume that ClinTrack is used for training and evaluation but even this is not clearly delineated by the authors. Relatedly, all of section 3.2 is difficult to follow without an example of data structure to accompany how ClinSim would actually generate scores in practice. In line with weaknesses of writing/organization, many of the results seem ancillary and either not fully supportive of any main claims and/or not immediately relevant. For example, Figure 4 shows the delta between two Qwen models across different evaluation methods, but the significance of this with respect to SimScore is not fully explained. Similarly for Table 3, if the result of a training ablation is significant to the take-aways of the paper, further discussion/analysis of this pattern is required.

### Questions
Is there truly no prior work on simulating missing diagnostic steps or counterfactuals in LLM-based clinical reasoning?

What does “generate samples to be simulated” mean in Figure 2 at step 4?

Why were Levenshtein and ROUGE selected as primary baselines for SimScore?

Why do the authors need SimScore instead of just training a model to generate associated test scores given a test type? It seems that SimScore is a result of the project formulation, but it would simplify the set-up (and thus isolate the experiments) to use a next-event prediction model that is trained to output high-fidelity test scores and then, in deployment, when a non-existent test is ordered it can generate a test result after being trained on what a realistic test result in that setting may be. This seems a simpler approach that does not have as many abstractions-- can you explain why this approach is not sufficient?

Could the authors show a concrete ClinTrack example with a corresponding SimScore computation?

How does the sigmoid function process raw predictions to yield the final similarity score? Generally unsure of how the SimScore is calculated and what the benefits of this recursive approach are as opposed to more general methods of similarity analysis, mean squared error between predicted test result and ground truth test result, or LLM-as-judge?

Given that only one medical expert participated, how reliable is the validation of SimScore alignment?

How does this work connect to broader machine learning interests beyond healthcare (e.g., simulation fidelity, counterfactual reasoning)?

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
The study tackles a common failure in virtual patient systems: when a clinician orders a reasonable test that wasn’t done in the historical record, the simulator cannot respond. It proposes “Examination Feedback Simulation,” a task that generates plausible results for such tests, and supports it with a curated dataset - ClinTrack, a structure-aware metric - SimScore, and a dedicated model - ClinSim. The scope centers on adult inpatient data from MIMIC-IV and three exam families i.e., radiology, microbiology, labs.

### Strengths
- The paper clearly formulates a practical, under-served problem and turns it into a concrete, reproducible task with well-defined inputs and outputs. This clarity makes the contribution easy to evaluate and build upon.
- It contributes an end-to-end pipeline and dataset that convert raw EHR tables into chronological, structured cases, enabling large-scale supervised training for the new task. The json event format is well aligned with clinical exam outputs.
- The SimScore metric evaluates structure, text, and numbers rather than surface text overlap alone. This better matches the nature of medical exam results and encourages faithful formatting.
- A compact, specialized model ClinSim outperforms much larger general models on this task, supported by useful ablations on training strategies. This suggests specialization and metric alignment matter more than sheer scale here.
- The paper provides implementation details, prompts, and an initial blinded clinician preference study improving transparency and offering additional validity for the metric.

### Weaknesses
Ordered from the most severe: 
- The main evaluation is a proxy, meaning that models are judged against held‑out events that actually happened, while the stated goal is off‑path plausibility. Without large‑scale clinician validation of true counterfactuals, the central claim remains partially untested.
- The same metric - SimScore is used both for optimization (RL) and evaluation, increasing the risk of reward hacking and overfitting to metric quirks. The small human study is not large enough to fully mitigate this concern.
- Ground truth is partially produced via LLM parsing and classification of notes, but there is no reported human audit of extraction accuracy. Label noise could propagate through training and evaluation.
- The project’s scope is limited, the addressed problem is important but narrow, and the data/domain (adult inpatient MIMIC-IV; three exam types) restrict generalizability. It is unclear how well the approach extends to outpatient, pediatrics, or other modalities.

### Questions
- How would you evaluate plausibility (clinician‑designed tests that were not performed) to validate the proxy of test on‑path events at scale?
- What safeguards do you use to detect/prevent reward hacking when SimScore is both the training signal and the test metric?
- Since the model labeled parts of the dataset, how accurate are those labels when checked by humans, and how much noise might be in them?
- Why include exact timestamps as inputs for the simulated event and how does performance change if they are removed or perturbed to better show off‑path use?
- Please fix the grammar: "the second best scores are highlighted in underlined".

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses a limitation in current LLM-based clinical simulations: their reliance on static EHRs, which causes simulations to fail when an LLM orders a medically valid but unrecorded examination. The study formulates a new task "Examination Feedback Simulation" to dynamically generate plausible results for such off-path orders. To support this, the study introduces three main contributions: (1) ClinTrack, a dataset curated from MIMIC-IV; (2) SimScore, an evaluation metric to assess the quality of the simulated JSON-based examination results; and (3) ClinSim, a generative model to perform the task. The paper demonstrates that ClinSim significantly outperforms baseline models on multiple tasks, providing a foundation for dynamic virtual patient simulations.

### Strengths
1. The paper addresses multiple facets of the challenge. It introduces a new benchmark dataset (ClinTrack), a novel evaluation metric (SimScore), and a specialized model (ClinSim) to perform the defined task. The connection between these three components is logical and well-executed.
2. The methodology is clear. The process for building the dataset and the SimScore metric is clear and easy to follow.
3. The proposed ClinSim model shows significant performance improvement over existing baseline LLMs. The inclusion of a wide range of baseline models in Table 1 provides a strong empirical validation.

### Weaknesses
1. I find the distinction between the “static” and “dynamic” aspects of the work somewhat unclear. The proposed task involves dynamic simulation of new feedback, yet the underlying dataset (ClinTrack) is inherently static, representing historical records. It would be helpful for the authors to clarify how this dynamic component is operationalized. Specifically, how much historical data is required to generate reliable simulation results, and how many simulated trajectories are produced per patient record?
2. I'm wondering if the proposed SimScore fully captures clinical utility. For example, is a simulation with a high SimScore guaranteed to lead to a better diagnosis? The alignment with expert judgment (89%) is very promising, but I'd be interested in the authors' thoughts on how fidelity (which SimScore measures) relates to downstream clinical settings or diagnostic accuracy.
3. The paper introduces a model called ClinSim, described as being built upon Qwen3, using Qwen3-4B for training and Qwen3-30B for parsing clinical notes. From the description, it seems that ClinSim primarily involves fine-tuning or adapting existing Qwen models. I am not very convinced if this is a significant contribution of model development.

### Questions
1. The paper notes that the authors treat the output as plain text, potentially overlooking the intrinsic structure and semantics of the JSON data. Could the authors elaborate on why preserving the JSON format might be important or beneficial in this context?
2. For certain applications, such as general simulations, generating synthetic outputs may be acceptable. However, for more sensitive areas like disease diagnosis, could such simulations pose risks or lead to misleading conclusions?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces Examination Feedback Simulation, a new task for generating plausible exam results to overcome the static nature of clinical trajectories in virtual patients. The authors build ClinTrack, a hierarchical, time-ordered dataset from MIMIC-IV, and propose SimScore for evaluating simulation quality. Their model, ClinSim (4B parameters), outperforms LLMs up to 235B parameters by over 10 points in SimScore, enabling more dynamic and realistic healthcare simulations.

### Strengths
1. The newly curated ClinTrack dataset and benchmark is valuable for the research of clinical trajectory simulation.
2. A new rule-based metric is proposed considering expert judgement in human evaluation.

### Weaknesses
1. The novelty of this work seem limited. The major contributions are the dataset and benchmarks. Please further clarify the motivation and novelty.
2. In introduction, the third contribution is overclaimed. There seems no change in architecture. The model is only pretrained on more specialized data. This is unfair to directly compare it with other existing models.
3. It is unclear why the proposed ClinSim, which is based on Qwen3, outperform Qwen3 by a large gap. It seems only the curated data is contributing.
4. There is no reference or detailed introduction of the database MIMIC-IV, which confuses readers.
5. No new training paradigm is proposed specialized for the EXAMINATION FEEDBACK SIMULATION in this work. The author is only validating the contribution of existing paradigms, such as SFT, CoT, GRPO.
6. The presentation of Figure 3 is poor, such as the organization of trainset and valset, the lack of detailed description of different event.
7. There is no discussion on how to solve hallucinations in clinical trajectory simulation.

### Questions
Please see the weaknesses

### Soundness
2

### Presentation
2

### Contribution
2

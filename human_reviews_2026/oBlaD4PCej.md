# Are Vision Language Models Blind Thinkers on Zero-Shot Multimodal Planning Tasks?

- Decision: Reject
- Scores: 6, 6, 4, 4

## Abstract
The advancement of vision language models (VLMs) has empowered embodied agents to accomplish simple multimodal planning tasks, but not long-horizon ones requiring long sequences of actions. In text-only simulations, long-horizon planning has seen significant improvement brought by repositioning the role of LLMs. Instead of directly generating action sequences, LLMs translate the planning domain and problem into a formal planning language like the Planning Domain Definition Language (PDDL), which can call a formal solver to derive the plan in a verifiable manner. In multimodal environments, research on VLM-as-formalizer remains scarce, usually involving gross simplifications such as predefined object vocabulary or overly similar few-shot examples. In this work, we present a suite of five VLM-as-formalizer pipelines that tackle one-shot, open-vocabulary, and multimodal PDDL formalization. We evaluate those on an existing benchmark while presenting another two that for the first time account for planning with authentic, multi-view, and low-quality images. We conclude that VLM-as-formalizer greatly outperforms end-to-end plan generation. We reveal the bottleneck to be vision rather than language, as VLMs often fail to capture an exhaustive set of necessary object relations. While generating intermediate, textual representations such as captions or scene graphs partially compensate for the performance, their inconsistent gain leaves headroom for future research directions on multimodal planning formalization.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The authors explore the application of the LLM-as-formalizer paradigm to VLMs. They consider 2 VLMs (GPT4.1 and Qwen-VL) and 5 VLM-as-formalizer pipelines. They introduce two novel multimodal benchmarks by annotating ground truth PDDL files for Blocksworld-Real and Alred-Multi. The evaluation considers 3 levels of success: 1st) executable PDDL syntax, 2) solver can find a plan with exhaustive search and 3rd) executing the plan leads to a success in the simulation. They find that VLM-as-formalizer is particularly helpful in long horizon tasks but visual capabilities remain the main bottleneck.

### Strengths
- well designed benchmark setup with well motivated design decisions
- useful adaption of existing datasets for VLM-as-formalizer
- Exhaustive related work section

### Weaknesses
- novelty is limited but great starting point for future research in this direction

### Questions
- Does the max token count of 1024 mentioned in 4.2. include thinking tokens of the VLM?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces 5 VLM-as-Formalizer (\vaf) pipelines for one-shot, open-vocabulary, multimodal PDDL formalization, evaluated on two existing and two new datasets (Blocksworld and ALFRED) with realistic visual conditions. Results show \vaf surpasses end-to-end plan generation and highlights that current VLMs are constrained more by vision than language, particularly in modeling complete object relations. Scene graph–based inference offers partial mitigation, pointing to future directions in visual PDDL generation.

### Strengths
1. New Benchmarks: The paper’s key strength lies in introducing the Blocksworld and ALFRED datasets, which move beyond oversimplified benchmarks. By incorporating real photos, multi-view scenes, occlusion, and motion blur, these datasets offer a much more realistic and impactful testbed for multimodal planning.

2. The results strongly support the idea that current VLMs reason well but see poorly. Low F1 scores in visual grounding versus text-based grounding (Table 1) clearly reveal vision, not language, as the main bottleneck—an insightful and memorable finding.

3. Instead of a single model, the paper systematically compares five formalizer pipelines, showing that intermediate representations (e.g., captions, scene graphs) significantly outperform direct PDDL generation. This highlights that inference-time reasoning can partly offset visual limitations.

### Weaknesses
1. Token Inefficiency: Figure 5 shows that top-performing pipelines like CAPTION-P achieve higher accuracy but at a huge token cost, making them far less efficient than direct planners. This limits scalability and practicality for real-world, resource-constrained systems.

2. Limited Scope of “Thinking”: The “blind thinker” metaphor somewhat exaggerates the VLM’s role—its function is mainly to translate scenes into PDDL state descriptions, while actual multi-step planning is handled by an external solver. The model formalizes states, not plans.

### Questions
All of my qeustions are listed in the weakness section. If my concerns are well addressed, I will raise my rating.

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
3

### Summary
This paper systematically investigates the paradigm of using Vision Language Models (VLMs) as PDDL formalizers (VLM-AS-FORMALIZER) for long-horizon, multimodal planning tasks, in contrast to end-to-end plan generation. The authors devise five modular pipelines for open-vocabulary PDDL problem generation, benchmarked against a VLM-as-planner baseline. Evaluation covers existing datasets and two new challenging, realistic, multi-view benchmarks (BlockSWORLD-REAL and ALFRED-MULTI) emphasizing imperfect real-world visual conditions. Extensive empirical results, supported by quantitative and qualitative analysis, reveal strengths and persistent limitations in VLM-based multimodal planning, especially in visual groundings.

### Strengths
- **Systematic, Comparative Framework:** The paper provides a thorough and well-structured empirical comparison of five distinct VLM-AS-FORMALIZER pipelines and a planner baseline, all on the identical, ambitious long-horizon planning task (Section 4.1 and Figure 1). This comprehensiveness goes beyond much prior work, which often examines a single method in isolation or under-simplifies visual scenarios.
- **Realistic Benchmarking:** The introduction of BlockSWORLD-REAL and ALFRED-MULTI as multi-view, real-image, challenging planning benchmarks (Section 3.2, Figure 2) advances evaluation realism and addresses key shortcomings in prior datasets (single-view, synthetic, or trivially structured). This enables new insights regarding VLM capabilities in noisy, partially observed environments.
- **Detailed Analysis and Insights:** Both quantitative (Table 1, Figure 4, Figure 5) and qualitative (Figure 6) evaluations are presented, and the discussion draws sharp, data-driven conclusions about why and where VLMs fail—especially the gap in visual scene grounding versus goal and object recall, and the role of prompt/intermediate representation strategy.
- **Mathematical and Technical Rigor:** The paper offers a precise formal definition of the Vision-PDDL-Planning task (Section 2.1), specifying entity sets, state representations, and evaluation criteria. This provides a sound basis for reproducibility and deeper analysis.
- **Clarity in Limitations:** The authors do not shy from articulating where VLMs fall short (e.g., "blind thinkers" bottlenecked by vision). Failure patterns are meticulously mapped in the results (Table 1, Figure 6), helping inform where future research should target.
- **Comparative Token Efficiency Analysis:** By introducing token efficiency metrics (Figure 5), the paper highlights important tradeoffs relevant to practical deployment in resource-constrained settings.

### Weaknesses
1. **Limited Novelty in Core Paradigm:** The VLM-as-formalizer concept—and generating PDDL from multimodal input—has close precedent in several recent works, including Shirai et al. (2024), Kwon et al. (2025), Dang et al. (2024), and multiple others reviewed in the related work section. The degree of novelty primarily lies in the systematic empirical benchmarking and pipeline analysis, not in radically advancing the methodology itself.
2. **Weaknesses in Related Work Positioning:** Several highly relevant directly related works (see “Potentially Missing Related Work” below) are not cited or discussed, notably those that handle similar formalization tasks or propose comparable systems (e.g., Image2PDDL or reflective planning with VLMs). For instance, Dang et al. (2024)'s 'Image2PDDL' mirrors aspects of the problem definition but is only implicitly addressed. The broader context of reflective or structured VLM planning is underexplored.
3. **Insufficient Baseline/Method Diversity:** The baseline and compared pipelines all involve either single VLMs or simple prompting variations. Despite claims of zero-shot and open-vocabulary capabilities, there is little inclusion or discussion of advanced compositional, neuro-symbolic, or hybrid methods that could integrate external object detectors, or explicit multi-step visual reasoning, as done in some related works or through more advanced architectural solutions (e.g., see Gu et al., 2024; Jiao et al., 2022). More diversity in competing approaches is desirable.
4. **Shallow Theoretical Analysis:** Beyond basic formalization, the paper provides little insight or formal modeling of the failure modes. For example, there is no mathematical dissection of how visual ambiguity or multi-view inconsistency propagates through PDDL generation. While the F1 breakdown in Table 1 is informative, deeper theoretical or optimization analysis—such as loss decomposition or an explicit error budget—would improve the scientific rigor.
5. **Equation and Notation Clarity:** The paper gives a succinct formal statement of the task (Section 2.1), but does not specify sampling/selection strategies for negative versus positive object relations, handling of ambiguous vision cues, or precise mechanisms for entity disambiguation across views in its mathematical form. For instance, it is unclear how VLMs resolve conflicts or occlusions in entity matching—critical for applications relying on cross-image consistency.
6. **Qualitative Analysis Covers Only a Few Representative Cases:** While Figure 6 provides qualitative illustration, these are limited to BlockSWORLD-REAL and capture only a small subset of error types. Broader visualization across datasets and tasks—and perhaps a taxonomy of common errors—would make the findings more robust.
7. **Reproducibility and Implementation Details:** Although the high-level approach and evaluation metrics are well described, many practical details are not spelled out, such as the prompt wording, exact hyperparameters for each VLM pipeline, and the PDDL parsing/compilation protocol. Additionally, there is no statement regarding code/data release, which impedes open comparison.
8. **Reliance on Proprietary Models and Limited External Validation:** GPT-4.1 performance may not generalize to non-commercial settings, and only one open-source model is tested. There is no systematic sensitivity analysis for model version, size, or training data variations. The findings could benefit from validation on a broader suite of models or architectures.
9. **Token Efficiency Framing Could Be Deepened:** While Figure 5 is a welcome inclusion, the discussion does not explore whether token count differences actually impact planning failure/intermediate errors, or if efficient summarization tricks would yield comparable performance.
10. **No End-to-End Embodied Agent Evaluation:** The pipelines are validated only through planning success, not actual execution on embodied agents in closed-loop settings. This reduces the operational significance of the findings, as simulation performance may not fully mirror real-world embodied planning challenges.
11. **Qualitative Differences Between Pipelines Not Probed Deeply Enough:** For instance, the differences in intermediate representations between CAPTION-P and SG-P are discussed, but the underlying cognitive/attention or error propagation mechanisms are not explored in depth, nor are possible remedies tried.

## Potentially Missing Related Work

1. **Yang, Z., Garrett, C. R., Fox, D. (2024): "Guiding Long-Horizon Task and Motion Planning with Vision Language Models"**
   - Directly integrates VLMs with hierarchical task and motion planning for long-horizon tasks, mirroring the vision-language-planning intersections here. Should be discussed in Section 6 and compared as a baseline or reference pipeline in empirical sections.
2. **Shirai, K., Beltran-Hernandez, C. C., Hamaya, M. (2024): "Vision-Language Interpreter for Robot Task Planning"**
   - Proposes using VLMs for generating problem descriptions for robotic task planning—directly paralleling VLM-as-formalizer logic. Needs explicit discussion and comparison in the related work section and empirical pipeline design.
3. **Feng, Y., Han, J., Yang, Z. (2025): "Reflective Planning: Vision-Language Models for Multi-Stage Long-Horizon Robotic Manipulation"**
   - Introduces test-time computation enhancements for VLM-based planning, relevant to the inference scaling ideas here. Should be referenced and positioned relative to novel aspects of the pipelines introduced.
4. **Dang, X., Kudláčková, L., Edelkamp, S. (2024): "Planning with Vision-Language Models and a Use Case in Robot-Assisted Teaching"**
   - Presents the 'Image2PDDL' framework, which closely aligns with the paper's formalization goals. Should be cited and compared in both the methodology and empirical results.

### Questions
1. **Entity Disambiguation and Multi-View Consistency:** How exactly does each pipeline resolve ambiguities or conflicts when the same entity appears in multiple images (e.g., under occlusion or motion blur)? Are there explicit mechanisms for cross-image entity fusion, or is this delegated entirely to VLM's latent capacity? Further details here—both in text and formula—would clarify the implementation.
2. **Negative Sampling and Initial State Recall:** Section 5 and Table 1 highlight that initial state recall is a primary failure point. Could the authors share which negative or ambiguous object relations are most frequently missed, perhaps with additional figures or even a confusion matrix? Would a different loss formulation or pipeline targeting high-recall make a significant difference?
3. **Prompt Engineering and Pipeline Robustness:** Are the observed differences in performance between CAPTION-P, SG-P, etc., mainly due to prompt wording or the inherent structure of intermediate representations? Is there evidence that better prompts would allow the simpler pipelines (e.g., DIRECT-P) to catch up? Sharing prompts or performing ablation over prompt structure would be valuable.
4. **Choice of Models:** Why was only GPT-4.1 and Qwen2.5-VL72B chosen? Were other VLMs or multimodal models considered, and if so, how did results compare, especially in lower-resource or real-time settings?
5. **Code/Data Availability:** Will the benchmarks (BlockSWORLD-REAL and ALFRED-MULTI), pipeline code, and evaluation scripts be publicly released to promote reproducibility and follow-up research?
6. **Full Embodied Evaluation:** Do the findings on planning success translate to real-world embodied agent execution, or is there a substantial sim2real gap? Any evidence (even anecdotal) here would be useful.

### Soundness
2

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
3

### Summary
The paper introduces two new benchmarks, BLOCKSWORLD-REAL and ALFRED-MULTI, which for the first time establish partially observable, multi-view environments within the Vision-PDDL-Planning framework. It further designs five VLM-as-Formalizer pipelines that generate PDDL problem files defined by the objects, initial states, and goal states of the given environment, and then use a PDDL solver to compute valid plans. Compared with the end-to-end baseline, the proposed VLM-as-Formalizer pipelines demonstrate significantly stronger performance on both benchmarks. Moreover, the paper shows that the primary weakness of current vision-language models lies not in their language reasoning ability but in their visual understanding, as they struggle to accurately capture the relationships between objects within the environment.

### Strengths
- The proposed benchmarks bridge the gap between synthetic and real-world embodied environments.

- The paper provides a comprehensive comparison by evaluating five structured VLM-as-Formalizer pipelines, which helps clarify the role of intermediate representations in enhancing visual understanding.

- The evaluation metrics are well designed.

### Weaknesses
- All proposed pipelines are primarily prompt-engineering variants rather than algorithmically distinct methods, without introducing fundamentally new learning mechanisms.

- The absence of concrete prompt templates of these baselines somewhat limits reproducibility and interpretability.

### Questions
- The generation process of the VLM-as-Planner baseline seems somewhat vague. It is unclear how the model actually produces the plan. Does it first generate a caption or a structured scene description and then predict a sequence of actions defined in the PDDL domain file, or does it directly output the action sequence? Additionally, would introducing intermediate representations, as done in pipelines B–E, help this process? For instance, if the environment is first formalized through a Caption- or Scene-Graph-based representation and then used as input for planning, would this improve its performance?

- Have the authors considered evaluating their pipelines on additional models (e.g., Claude or Gemini) to examine the generality of the findings?

### Soundness
3

### Presentation
3

### Contribution
3

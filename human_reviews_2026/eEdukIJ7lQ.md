# Remote Sensing-Oriented World Model

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 2, 4, 6

## Abstract
World models have shown potential in artificial intelligence by predicting and reasoning about world states beyond direct observations. However, existing approaches are predominantly evaluated in synthetic environments or constrained scene settings, limiting their validation in real-world contexts with broad spatial coverage and complex semantics. Meanwhile, remote sensing applications urgently require spatial reasoning capabilities for disaster response and urban planning. This paper bridges these gaps by introducing the first framework for world modeling in remote sensing. We formulate remote sensing world modeling as direction-conditioned spatial extrapolation, where models generate semantically consistent adjacent image tiles given a central observation and directional instruction. To enable rigorous evaluation, we develop RSWISE (Remote Sensing World-Image Spatial Evaluation), a benchmark containing 1,600 evaluation tasks across four scenarios: general, flood, urban, and rural. RSWISE combines visual fidelity assessment with instruction compliance evaluation using GPT-4o as a semantic judge, ensuring models genuinely perform spatial reasoning rather than simple replication. Afterwards, we present RemoteBAGEL, a unified multimodal model fine-tuned on remote sensing data for spatial extrapolation tasks. Extensive experiments demonstrate that RemoteBAGEL consistently outperforms state-of-the-art baselines on RSWISE.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper’s goal is to (i) define “remote sensing world modeling” as direction-conditioned spatial extrapolation (i.e., generate the tile north/south/east/west of a given satellite tile), (ii) introduce an evaluation benchmark and metric suite (RSWISE), and (iii) introduce a model (RemoteBAGEL) fine-tuned for that task, showing SOTA results over baselines drawn from strong multimodal/editing models.

### Strengths
1. Clear new task framing: turning spatial extrapolation in real satellite imagery into a “world modeling” problem is compelling and well motivated by real-world needs like flood response and urban planning.

2. Benchmark contribution: RSWISE is carefully constructed (4 scenarios, 1,600 tasks) with standardized prompts and directional supervision derived from overlapping 3×3 grids. This feels like infrastructure the community can reuse.

3. Metric design: The combined RSWISE score (FID + GPT-4o spatial compliance) addresses a real failure case of standard generative metrics, where a model can “win” by repeating the input instead of extrapolating. The paper also analyzes the weighting and shows ranking stability.

4. Strong empirical gains: RemoteBAGEL strongly outperforms strong baselines on RSWISE across all four scenarios (average 88.8 vs 62.4 for BAGEL and 53.2 for Qwen-Image-Edit, etc.), and the qualitative samples support that it is actually extrapolating spatial structure rather than copying textures.

### Weaknesses
1. Limited OOD / robustness analysis. The benchmark scenarios (general, flood, urban, rural) are diverse, but all data still come from a few public datasets and similar sensors. We don’t see tests like cross-city, cross-year, different sensor modality, or unseen geography, which would be critical for claims about “world modeling” rather than “tile completion on these datasets.”

2. Reliance on GPT-4o for scoring without human validation. The GPT judge is clever, but there’s no study of reliability vs expert human raters, nor variance across multiple evaluations. If GPT-4o hallucinates that a plausible-looking but geographically wrong continuation is “correct,” that could inflate scores. 

3. Ablation depth. The model is described as BAGEL-7B fine-tuned on (tile, direction→neighbor tile) pairs with a simple reconstruction loss and directional conditioning. There’s no ablation isolating: effect of direction token vs none, effect of 66.7% overlap vs less/more overlap, effect of different prompt wordings (“look right” vs “to the east” etc.).
This makes it hard to tell what design choices are essential vs incidental.

4. Novelty: The novelty is primarily in task definition + benchmark + evaluation metric, not in a radically new architecture.

### Questions
1. Scope of “world model.” While I agree with the framing that spatial extrapolation in satellite imagery is an instance of “world modeling,” the paper sometimes leans on the broader hype around world models (robotics, autonomous driving, temporal prediction). Some readers may feel that what’s actually demonstrated is a strong, task-specific spatial outpainting benchmark/model, not yet a generalized “world model of Earth.”

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
3

### Summary
This paper introduces a new paradigm for world modeling in remote sensing. The authors propose applying world modeling to remote sensing imagery, which demands understanding and extrapolating complex spatial structures, arguing that this is a crucial capability for applications such as disaster response and urban planning.

The authors formulate remote sensing world modeling as a direction-conditioned spatial extrapolation problem, where a model must generate semantically consistent neighboring image tiles based on a given central tile and a directional instruction (e.g., up, down, right or left).

To evaluate this task, they introduce RSWISE (Remote Sensing World-Image Spatial Evaluation), the first benchmark for world modeling in remote sensing. RSWISE comprises 1,600 evaluation tiles across four scenarios: general, flood, urban, and rural. The idea is to assess visual fidelity and instruction compliance, and uses GPT-4o as a semantic evaluator to ensure models truly understand spatial relationships rather than performing low-level pattern copying.

Building on this benchmark, the paper presents RemoteBAGEL, a unified multimodal model fine-tuned on remote sensing data specifically for spatial extrapolation. Experimental results show that RemoteBAGEL outperforms the compared baselines across all RSWISE scenarios.

Although the task is novel and a dataset is always a valuable contribution, the evaluation protocols and methodological grounding remain weak.

### Strengths
1. Novel dataset and task formulation. The paper introduces a new problem setup: direction-conditioned spatial extrapolation in remote sensing—which is conceptually framed as an instance of "world modeling". 
The accompanying RSWISE benchmark, covering 1,600 tasks across diverse conditions (general, flood, urban, rural), provides a structured evaluation protocol that could be useful to the research community. 

2. Overall clarity and writing quality
The paper is well-organized and clearly written, with a logical flow from motivation to methodology, experiments, and discussion.

### Weaknesses
1. **Unjustified formulation of “world modeling” for remote sensing.** The paper does not justify why world modeling in remote sensing should take the specific form of direction-conditioned spatial extrapolation. While generating adjacent image tiles might reflect a kind of spatial prediction, the link between this operation and broader notions of world modeling—which typically involve learning dynamic, causal, or physical structure—is insufficiently explained. As a result, the formulation feels somewhat arbitrary, and its conceptual connection to existing world modeling frameworks remains weak.


2. **Weak connection to high-impact applications.**
The paper motivates the study by citing socially relevant applications—such as infrastructure forecasting and flood prediction, which require reasoning beyond observed regions. However, the proposed task (generating adjacent tiles) is not clearly linked to these applications. There is no demonstration of how the generated outputs could actually inform decision-making in such domains. The task thus seems artificial and detached from practical use cases, reducing the claimed real-world significance.

3. **Vague dataset construction details.** The dataset creation process lacks technical clarity. For instance, the authors mention that “_large_ satellite images are divided into 3×3 overlapping grids” but do not define what qualifies as “large,” nor do they specify image resolution, spatial coverage, or grid size. These omissions make it difficult to assess the representativeness and quality of the dataset, or to replicate the benchmark.

4. **Unfair evaluation setup.** The evaluation protocol appears biased in favor of the proposed model, since RemoteBAGEL is fine-tuned on the same dataset used for testing, while all competing methods are evaluated in a zero-shot manner. This discrepancy conflates the effects of task adaptation with model capability, making it unclear whether RemoteBAGEL’s superior performance reflects genuine spatial reasoning or merely data familiarity. A fair comparison would require at least partial fine-tuning or domain adaptation for all baselines.

5. **Lack of alternative evaluations.**
Beyond the RSWISE benchmark, no additional quantitative or qualitative evaluations are provided. The authors do not test model robustness, generalization to unseen geographies, or practical downstream utility. This narrow evaluation scope makes it hard to judge whether the model generalizes beyond the controlled benchmark conditions or truly captures meaningful spatial dependencies.

6. **Questionable reliance on GPT-4o for evaluation.**
The use of GPT-4o as a “semantic judge” introduces significant uncertainty and potential bias in the evaluation process. Large language models are not guaranteed to assess visual or spatial coherence reliably, particularly for remote sensing imagery where domain knowledge is required. The lack of human or domain-expert evaluation seriously limits confidence in the reported results and calls into question the claimed semantic understanding of the model.

7. **Lack of methodological innovation beyond fine-tuning BAGEL.** If I understood correctly, the proposed RemoteBAGEL model is largely a fine-tuned version of the existing BAGEL architecture, adapted to the proposed remote sensing dataset. Beyond conditioning the model on directional inputs and predicting neighboring tiles, there do not appear to be substantive architectural or algorithmic innovations introduced for the specific challenges of spatial extrapolation or world modeling.  If this interpretation is accurate, the paper’s contribution lies mainly in problem framing and dataset creation, rather than in developing new methods or insights into spatial reasoning itself.

7. **Mischaracterization of the remote sensing literature.**
The authors explicitly claim that prior remote sensing work has been “limited to classification and semantic segmentation,” which is factually inaccurate and overly reductive. The field includes a broad spectrum of tasks such as object detection, change detection, super-resolution, spatiotemporal forecasting, generative modeling, VQA, image retrieval, etc [1]. This oversimplification undermines the novelty claim, since the proposed direction-conditioned generation task fits naturally within existing paradigms of spatial image synthesis and extrapolation already explored in remote sensing contexts.

[1] Tuia, D., Schindler, K., Demir, B., Zhu, X. X., Kochupillai, M., Džeroski, S., ... & Camps-Valls, G. (2024). Artificial Intelligence to Advance Earth Observation: A review of models, recent trends, and pathways forward. IEEE Geoscience and Remote Sensing Magazine.

### Questions
- Could the authors clarify why world modeling in remote sensing is formulated specifically as direction-conditioned spatial extrapolation?
- How does this formulation relate conceptually to the broader definition of world models in AI, which often involves learning dynamics, causality, or temporal reasoning?
- How do the proposed contributions go beyond existing generative or extrapolative approaches already explored in the field?
- The paper motivates the task through applications like flood prediction and infrastructure forecasting. Could the authors elaborate on how the proposed spatial extrapolation task concretely relates to or supports these applications? Are there examples or experiments showing that the generated adjacent tiles can be practically useful in such contexts?
- How reliable is GPT-4o as a semantic evaluator for spatial consistency in remote sensing imagery? Have the authors validated GPT-based assessments against human or domain-expert evaluations?
- Could the authors provide inter-rater agreement or consistency analysis to support the robustness of the GPT-based evaluation?
- How does RemoteBAGEL differ architecturally or conceptually from the original BAGEL model?

### Soundness
1

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
This paper introduces the first framework for world modeling in remote sensing, aiming to extend the idea of world models, originally developed for synthetic or constrained settings, to real-world geospatial contexts. The authors:
1. Formulate remote sensing world modeling as direction-conditioned spatial extrapolation, where the model predicts adjacent image tiles given a central observation and directional instruction.
2. Develop a new benchmark, RSWISE (Remote Sensing WorldImage Spatial Evaluation), containing 1,600 evaluation tasks across general, flood, urban, and rural scenarios.
3. Propose a unified multimodal model, RemoteBAGEL, fine-tuned for spatial extrapolation tasks.
4. Design evaluation metrics combining visual fidelity (FID) and instruction compliance using GPT-4o as a semantic judge.
Experiments show that RemoteBAGEL outperforms state-of-the-art baselines on RSWISE.

### Strengths
- The paper presents an interesting and novel perspective by framing spatial reasoning as a world modeling problem in remote sensing.
- Section 3.1 gives a precise mathematical definition of the directional spatial extrapolation task, and the three evaluation axes are well thought out.
- The RSWISE benchmark could stimulate follow-up work if more carefully validated.

### Weaknesses
- “Meanwhile, remote sensing applications urgently require spatial reasoning capabilities for disaster response and urban planning.” Please give a clearer definition of spatial reasoning capabilities in this context.

- "We formulate remote sensing world modeling as direction-conditioned spatial extrapolation (defined in the image-grid frame with up, down, left, and right, rather than geographic cardinal directions)." Why? The motivation needs to be further clarified.

- The evaluation design raises concerns: using GPT-4o as a semantic judge introduces non-determinism (“GPT4o can give different answers every time”), raising questions about fairness, reproducibility, and scientific rigor.

- Table 1 mixes benchmarks from unrelated domains (robotics, video, games), making the comparison difficult to interpret.

- In Section 2.2: “However, in contrast to world models, current RS methods seldom attempt spatial continuation or reasoning over large geospatial structures.” Is this empirically verified or based on community consensus?

- The statement “our analyses therefore study anisotropy in grid-aligned continuations independent of geographic orientation” is unclear - what does this imply, and is it desirable?

- In Section 3.2, more details about the datasets (e.g., sensors, resolution) should be provided.

- The data construction pipeline would benefit from an illustration. Also, why was a 66.7% overlap chosen? Is it too large? Was it experimentally tested?

- The connection between the three evaluation axes and the two metrics (distributional fidelity and spatial reasoning) is not clearly explained; more illustrations would help.

- Table 2 lacks details about how baselines are used and why they were chosen.

- Figure 3 contradicts the preceding statement: "FID values are globally normalized into sfid ∈ [0, 1], standardizing units across scenarios and **inverting the metric** so that higher values correspond to better performance."

- Several experimental settings are underspecified:
1. No clear train/test split; unclear how 10,080 instances are derived from 4,000 images, which possibly introduces data leakage.

2. Why is inference more computationally expensive than training?

3. How were baseline models (e.g., BAGEL) used, zero-shot or few-shot?

4. The statement “These failures underscore that visual realism alone is insufficient…” suggests domain-specific fine-tuning; should this be considered a fine-tuning method rather than a new one?

5. The explanation for Figure 5(b) (“Horizontal continuations...”) is not convincing.

- The conclusion mentions capturing “large-scale geospatial structures” and future extensions to cloud removal, weather forecasting, and 3D flood visualization. These claims are vague and not clearly supported by current experiments.

### Questions
Please refer to Weaknesses. 

The paper raises an interesting and timely idea: introducing world models to remote sensing, but the current version lacks clarity and rigor in its experimental design, benchmark construction, and evaluation methodology. The use of GPT-based scoring and an unclear dataset setup undermines the reliability of the conclusions. Overall, while the topic is promising, the paper requires significant work to be considered solid.

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
This paper introduces a new framework for world modeling in remote sensing, addressing a gap where existing world models have been tested mostly in synthetic or constrained settings.
The authors formulate remote sensing world modeling as a direction-conditioned spatial extrapolation task, where a model generates semantically consistent adjacent tiles (up, down, left, right) based on a given central image and directional instruction.
To evaluate this setting, they propose RSWISE (Remote Sensing World-Image Spatial Evaluation) — a benchmark combining distributional fidelity (FID) and spatial reasoning (GPT-based semantic scoring), covering four scenarios (general, flood, urban, rural).
They further develop RemoteBAGEL, a remote-sensing-adapted multimodal model fine-tuned from BAGEL-7B, demonstrating superior performance on RSWISE compared to various baselines (e.g., Qwen-Image-Edit, FLUX.1-Kontext-Dev, Step1X-Edit). The results show that RemoteBAGEL achieves both improved realism and geospatial coherence, highlighting the value of domain-specific adaptation for spatial reasoning.

### Strengths
Novel problem formulation: The paper is among the first to formalize world modeling for remote sensing, introducing a direction-conditioned extrapolation setup that aligns with real-world geospatial reasoning needs (e.g., flood forecasting, urban planning).

Benchmark contribution: RSWISE is a well-motivated and comprehensive benchmark, integrating both distributional and semantic evaluation. It goes beyond standard FID-only metrics by quantifying instruction compliance and spatial continuity, which is convincingly illustrated in Figure 1.

Clear empirical advantage: RemoteBAGEL significantly outperforms both foundation and editing baselines (up to +25 points in RSWISE average), showing that specialized training for spatial reasoning leads to meaningful gains.

Rigorous experimental setup: The study uses 1,600 evaluation tasks across multiple environmental scenarios, includes ablation-like discussions on directional anisotropy and scenario difficulty (Fig. 5), and analyzes evaluation weighting (Fig. 6) with methodological transparency.

High-quality writing and presentation: The paper is clearly written and well-structured, with informative diagrams (Figures 1–4) and transparent methodology. 

Potential impact: By connecting world models to remote sensing, this work could influence future research in spatial reasoning, data-efficient Earth observation, and model evaluation beyond synthetic benchmarks.

### Weaknesses
Limited conceptual novelty in modeling: The RemoteBAGEL model itself is mainly a domain-adapted fine-tuning of BAGEL rather than a fundamentally new world-modeling architecture. The innovation lies more in benchmarking than in model design.

Dependence on GPT-based evaluation: While GPT-4o is used as an automated semantic judge, its objectivity and reproducibility in scientific benchmarking could be questioned. The paper mentions prompt consistency (Appendix A) but lacks an independent human validation or inter-rater reliability check.

Lack of interpretability or failure diagnostics: Although failure modes are discussed (Fig. 4), quantitative analyses of why certain directional biases (up/down vs left/right) occur remain shallow. Deeper insights into model attention or representation behavior would enhance scientific understanding.

Dataset scale and coverage: The dataset is composed of 1,600 tasks drawn from a few public datasets (Sky-SA, FloodNet, LoveDA), which may limit generalization to other modalities (SAR, hyperspectral) or unseen geographies.

Overemphasis on quantitative scores: While RSWISE provides structured evaluation, the qualitative interpretation of what constitutes “good spatial reasoning” remains somewhat heuristic and tied to metric design.

### Questions
See weakness

### Soundness
3

### Presentation
2

### Contribution
3

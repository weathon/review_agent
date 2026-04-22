# Scaling Agents via Continual Pre-training

- Avg Score: 4.67
- Decision: Accept (Poster)
- Scores: 2, 4, 8

## Abstract
Large language models (LLMs) have evolved into agentic systems capable of autonomous tool use and multi-step reasoning for complex problem-solving. However, post-training approaches building upon general-purpose foundation models consistently underperform in agentic tasks, particularly in open-source implementations. We identify the root cause: the absence of robust agentic foundation models forces models during post-training to simultaneously learn diverse agentic behaviors while aligning them to expert demonstrations, thereby creating fundamental optimization tensions. To this end, we are the first to propose incorporating Agentic Continual Pre-training (Agentic CPT) into the deep research agents training pipeline to build powerful agentic foundational models. Based on this approach, we develop a deep research agent model named AgentFounder. We evaluate our AgentFounder-30B on 10 benchmarks and achieve state-of-the-art performance while retains strong tool-use ability, notably 39.9% on BrowseComp-en, 43.3% on BrowseComp-zh, and 31.5% Pass@1 on HLE.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes Agentic Continued Pre-Training for having better base pre-trained models for capabilities requiring tool calls and multi-hop reasoning like DeepResearch. The authors develop a model named AgentFounder-30B using this continued pre-training setup by generating synthetic pre-training data using FAS (First-order Action Synthesis) and HAS (Higher-order Action Synthesis); and further apply SFT on top of the base model with agentic capabilities. They achieve SoTA scores (across open models/deepresearch agents) on various agentic benchmarks like BrowseComp, HLE, GAIA etc.

### Strengths
- The paper tackles an important problem of building better base models for agentic capabilities where standard pre-training data is not easily available for multi-hop reasoning, long trajectories, etc.
- The paper's synthetic data generation pipeline is interesting and captures various insights like collecting discarded rejection sampling and historical tool use data, correlation of quality of first-step reasoning with final task completion rates.
- The empirical results are impressive.

### Weaknesses
I feel there are quite a lot of points to address before this paper is ready for publication.

- The entire paper reads more like a high-level blog rather than a technical white paper with concrete methodology, steps/details, and experiments.
- The paper tries to present that they are the first to do agentic CPT, but a lot of mid-training/pre-training works do this step before post-training for example: https://arxiv.org/abs/2507.20534
- Lines 65-67 "Consequently, both SFT and RL training depend on limited deterministic
supervisory signals that lock models into replicating specific behavioral patterns rather than develop
flexible decision-making capabilities". The authors don't show any evidence for this and neither do they cite any works highlighting this phenomenon. SFT does mimic specific patterns, but I've not seen this case happening with RL.
- A lot of key technical details are missing (at least in the main paper), like where are the questions sourced from, how many parallel steps are generated for the HAS strategy, what models are used for the rejection sampling, number of tokens for SFT and what's the source of the SFT data, etc.
- How is the model performance affected on non-agentic benchmarks by adding the agentic CPT data? There's no analysis on that.
- What's the effect of the model used for rejection sampling? Can weaker models help as well to eliminate the "distillation" effect of using more capable models to get the rejection sampling data?
- The effect of HAS strategy is minimal on top of FAS, and the evaluation gains seem to be just noise.
- Since I don't know the source of the questions used to generate the synthetic data, how did the authors ensure no contamination with the evaluation sets like HLE.

Although not a weakness, there are a few typos here and there: AgentFoudner instead of AgentFounder in various places, line 224: high-orider instead of high-order.

### Questions
I have asked most of my questions in the weaknesses sections above. I am willing to increase my score if the authors are able to address the issues.

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
This paper introduces Agentic Continual Pre-Training as an intermediate training state between pre-training and post-training for building deep research agents. The authors propose a two-stage training pipeline: stage 1 involves standard next-token prediction training on synthetic data, while stage 2 focuses on long-context training. To support the training pipeline, authors propose two key data synthesis methods: First-order Action Synthesis (FAS) for generating planning and reasoning data and Higher-order Action Synthesis (HAS) for transforming wasted trajectories into multi-step multi-choice decision problems. 

The trained model achieves strong performance across 10 benchmarks, including BrowseComp, HLE, GAIA, etc. However, the paper does not provide details of some of the claims and doesn't provide enough details about the generated data.

### Strengths
- Strong Empirical Performance: The proposed model achieves state-of-the-art results on multiple challenging benchmarks and beats stronger and larger commercial models on most datasets.
- Scalable Data Synthesis: The FAS approach for generating diverse question-answer pairs from entity-anchored knowledge without requiring expensive API calls or human supervision is practical and cost-effective.
- Model Size and Data Scaling: The Authors show that their method scales with the model size and data volume effectively.
- Adaptability to different SFT data: Table 3 shows that the authors' model consistently improves performance across different post-training dataset setups, suggesting the CPT does provide a better foundation.

### Weaknesses
- Unverified Claims: The Authors claim that general-purpose foundation models create "inherent optimization conflicts", but this is not verified in this work, and no supporting evidence in the form of a citation is provided. Similarly, authors also claim that "the quality of first-step reasoning exhibits strong positive correlation with final task completion rates". Although this statement may seem intuitive, no experimental validation is provided to support it.
- Unclear distinction from domain adaptation: While the authors claim that the trained model provides a better foundation model for post-training, the paper doesn't verify this claim outside the domain of web-search-based benchmarks. This is understandable because the data was collected using web-search-specific tools; however, the authors should not claim it as a foundation model.
- Limited Transparency on data composition: While the work mentions using 200B and 100B tokens for Stage 1 and 2, respectively, the actual composition of the FAS/HAS data or the data sources is never discussed in the work. How did the authors verify that there was no potential test-data leakage during the training data collection?
- Zero Supervisory Signal: In section 2.2, the authors claim that the data is collected using a zero supervisory signal. However, in section 2.2.2, the authors employ rejection sampling using an LLM-as-a-judge. This is a type of supervision. Authors should clarify this claim.
- Limited Error Analysis: The paper does not provide details error analysis showing what types of tasks/questions benefit most from Agentic CPT versus post-training alone, making it difficult to understand the method's limitations.

### Questions
- GLM 4.5 Comparison: Please add some details on how your strategy differs from the GLM 4.5 strategy. What is the difference between the agentic data sizes used for GLM 4.5 and the AgentFounder model?
- FAS vs Trajectory Collection: What is the performance comparison between FAS-generated data and actual collected trajectories (e.g., from rollouts)? Is FAS primarily a cost-saving measure or does it provide qualitative benefits?
- HAS Design Choices: In HAS data construction, you generate N alternatives. What is the value of N here? Please add details about the dataset construction to the paper.
- In the first stage, is there any general-purpose reasoning corpus added to the data mix? If not, does the model suffer from forgetting? Some evaluations on general-purpose language reasoning benchmarks like MMLU would be appreciated and would make the claims stronger.
- Can authors please provide error analysis of the trained model on the evaluation datasets? What type of errors are still prevalent after Agentic CPT? What type of errors are targeted in this training paradigm? This would add significant depth to the work.
- It is not clear to me which data is used for what stage. Some clarification on this would be appreciated.
- In Table 4, what happens if you skip Stage 1 training? 
- Evaluation Protocol: The Agentic CPT data is collected using the same set of tools as used in the evaluation. However, I am not sure if the other models were trained to use the same set of tools. This gives an inherent advantage to the AgentFounder model. If my understanding is correct, how does it affect the tool use for AgentFounder vs other models? Are the majority of the errors in baseline models because of incorrect tool use?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes Agentic Continual Pre-Training, an intermediate training stage between pre-training and post-training to bridge the optimization conflict between pre-training and agent post-training. To test the effectiveness of the proposed method, the authors further synthesized large-scale data for model training. The resulting model, AgentFounder-30B, achieves state-of-the-art results on 10 benchmarks, including 39.9% on BrowseComp-en, 31.5% on HLE, and 72.8% on GAIA. Ablations show consistent improvements, and scaling analyses indicate steady gains with larger models and data. Key contributions include:

(1) First-order Action Synthesis (FAS): reformulates static knowledge into complex questions and synthesizes planning/reasoning behaviors without API costs.

(2) Higher-order Action Synthesis (HAS): reuses suboptimal trajectories by expanding each step into contrastive multi-choice decision processes.

(3) Two-stage CPT strategy: progressive training with 32K and 128K contexts (200B + 100B tokens) to enhance long-horizon reasoning.

### Strengths
(1) The data synthesis pipeline is scalable and efficient.  FAS enables large-scale agentic data generation through knowledge-to-question transformation without costly API calls, while HAS reuses suboptimal trajectories by converting them into step-level decision processes. Together, they improve data efficiency and coverage. The addition of reject sampling with knowledge alignment verification further enhances data quality.

(2) The experiment results are promising. AgentFounder-30B achieves SOTA across ten benchmarks, outperforming all open-source agents and even surpassing some commercial systems. It is the first open-source model to exceed 30% on HLE (31.5%), with consistent gains across all SFT configurations, demonstrating strong robustness.

### Weaknesses
(1) Limited Theoretical Evidence for “Optimization Conflict” 
The claimed conflict between capability learning and alignment is intuitively appealing but empirically unsupported. It would be great if the authors could provide more evidence for this claim, like gradient interference analyses (e.g., cosine similarity between CPT and SFT objectives), visualize loss landscapes, and probe capability retention during SFT with vs. without CPT.

(2) Lack of analysis regarding the influence of the model on data synthesis.  It would be great if the authors could provide more details on the models used in data synthesis. Ideally, it would be great if the authors could further provide an oblation study regarding model performance against the quality of the model for data synthesis.

### Questions
The paper's core contribution relies heavily on the synthetic FAS and HAS data synthesis. Could you specify the data release plan, like whether the whole CPT stage training data will be released? If it's not, it's possible to release a sample subset for community inspection.

### Soundness
3

### Presentation
3

### Contribution
3

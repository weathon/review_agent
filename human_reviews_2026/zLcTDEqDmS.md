# SpaceR: Reinforcing MLLMs in Video Spatial Reasoning

- Avg Score: 4.67
- Decision: Reject
- Scores: 6, 4, 4

## Abstract
Video spatial reasoning, which involves inferring the underlying spatial structure from observed video frames, poses a significant challenge for existing Multimodal Large Language Models (MLLMs). This limitation stems primarily from 1) the absence of high-quality datasets for this task, and 2) the lack of effective training strategies to develop spatial reasoning capabilities. Motivated by the success of Reinforcement Learning with Verifiable Reward (RLVR) in unlocking LLM reasoning abilities, this work aims to improve MLLMs in video spatial reasoning through the RLVR paradigm. To this end, we introduce the **SpaceR** framework. First, we present **SpaceR-151k**, a dataset with 91k questions spanning diverse spatial reasoning scenarios with verifiable answers, and 60k samples for maintaining general multimodal understanding. Second, we propose **Spatially-Guided RLVR (SG-RLVR)**, a novel reinforcement learning approach that extends Group Relative Policy Optimization (GRPO) with a novel map imagination mechanism, which encourages the model to infer spatial layouts in the thinking process, thereby facilitating more effective spatial reasoning.
Extensive experiments demonstrate that SpaceR achieves state-of-the-art performance on spatial reasoning benchmarks (e.g., VSI-Bench, STI-Bench, and SPAR-Bench), while showing competitive results on video understanding benchmarks (e.g., Video-MME, TempCompass, and LongVideoBench). 
Remarkably, SpaceR surpasses the advanced GPT-4o by 11.6\% accuracy on VSI-Bench and is on par with the leading proprietary model Gemini-2.0-Flash, highlighting the effectiveness of our SpaceR-151k dataset and SG-RLVR in reinforcing spatial reasoning ability of MLLMs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the limitations of existing Multimodal Large Language Models (MLLMs) in video spatial reasoning (inferring 3D spatial structures from video frames), which arise from the lack of high-quality task-specific datasets and effective training strategies. Inspired by Reinforcement Learning with Verifiable Reward (RLVR), the authors propose the SpaceR framework, consisting of two core innovations: SpaceR-151k Dataset: A large-scale dataset with 91k spatial reasoning QA pairs (SR-91k, derived from the 3D ScanNet dataset, covering 6 tasks like relative direction and object size) and 60k general multimodal samples (from Video-R1-260k) to preserve general video understanding.
Spatially-Guided RLVR (SG-RLVR): An RL approach extending Group Relative Policy Optimization (GRPO) with a map imagination mechanism and task-specific verifiable rewards (e.g., format, multi-choice, numerical, map rewards).
Extensive experiments show SpaceR achieves state-of-the-art performance on spatial reasoning benchmarks (VSI-Bench, STI-Bench, SPAR-Bench)—surpassing GPT-4o by 11.6% on VSI-Bench—and competitive results on general video understanding benchmarks (Video-MME, TempCompass, LongVideoBench). It also matches the proprietary Gemini-2.0-Flash, validating the framework’s effectiveness.

### Strengths
Targeted Dataset Solution: SpaceR-151k fills a critical gap in video spatial reasoning resources by providing 91k high-quality, verifiable QA pairs across diverse spatial tasks (e.g., relative distance, room size), addressing the field’s data scarcity.

Innovative RL Enhancement: The map imagination mechanism explicitly guides models to infer spatial layouts, a novel design that strengthens deep spatial reasoning (ablation studies confirm it boosts performance on spatial benchmarks).

Strong Generalization: SG-RLVR outperforms supervised fine-tuning (SFT) across both spatial reasoning and general video tasks, avoiding SFT’s "memorization" limitation and demonstrating broader applicability.

Comprehensive Validation: Experiments cover 3 spatial and 3 general video benchmarks, with comparisons to state-of-the-art models (GPT-4o, Gemini series, open-source MLLMs like Qwen2.5-VL), ensuring robust evaluation of effectiveness.

Scalability: SG-RLVR works across model sizes (3B, 7B parameters) and architectures (dense VLMs like Qwen2.5-VL, MoE models like Kimi-VL-Thinking), showing flexibility for different MLLM frameworks.

### Weaknesses
Limited Failure Analysis: While the paper identifies two failure modes (visual perception errors, location misidentification), it lacks deeper investigation into root causes (e.g., why object recognition fails for occluded items) or actionable solutions to address them.

Reasoning Efficiency Tradeoffs: The "think mode" (structured reasoning) improves spatial performance but introduces noise in general video tasks (slight accuracy drops). The paper does not propose adaptive mechanisms to decide when reasoning is necessary, limiting practicality.

Dataset Dependency on ScanNet: SR-91k relies heavily on the ScanNet 3D dataset (indoor scenes), which may restrict the framework’s generalization to non-indoor scenarios (e.g., outdoor, dynamic environments) not covered by ScanNet.

All performance metrics are automatic (accuracy, ROUGE scores), with no human assessment of reasoning interpretability (e.g., whether cognitive maps align with human spatial intuition), which is critical for trust in real-world use.

### Questions
Please refer to the weaknesses part.

### Soundness
3

### Presentation
2

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
This paper introduces SpaceR, a reinforcement learning framework designed to enhance spatial reasoning in multimodal large language models (MLLMs). The core idea is to move beyond traditional visual question answering and captioning supervision by explicitly rewarding models for spatial consistency between generated textual descriptions and visual layouts.

SpaceR builds a spatial reward model that measures geometric alignment between object relations described in text and their actual visual configurations. The authors use a combination of synthetic spatial reasoning datasets and real-world multimodal data to train and evaluate the approach.
During RL fine-tuning, the policy model (based on InternVL2 or Qwen-VL2) receives feedback from the spatial reward to iteratively improve reasoning accuracy. Experimental results show consistent improvements on MIRAGE, GQA-Spatial, and CLEVR-Humans, achieving up to +5–8% gains over baseline MLLMs without introducing additional visual encoders.

### Strengths
Clear motivation and novelty:
The paper identifies a real gap—MLLMs’ weak spatial reasoning—and proposes a reward-driven framework (SpaceR) that directly targets this limitation.

Methodological soundness:
The spatial reward function is well-designed, combining both geometric similarity and relational consistency, which goes beyond naive L2 distance or caption overlap metrics.

Empirical improvements:
Results across multiple benchmarks are significant and consistent, and ablations demonstrate the distinct contribution of the reward model compared with pure SFT or contrastive losses.

Broader relevance:
The idea of structured, task-specific reward modeling could inspire new RL pipelines for multimodal reasoning beyond spatial understanding.

### Weaknesses
Model dependency and potential bias:
All experiments are conducted using Qwen-VL2 as the base model, which may already encode spatial priors due to large-scale visual pretraining. This makes it difficult to disentangle how much improvement comes from SpaceR itself versus Qwen-VL2’s inherent bias or possible data leakage from spatially rich corpora.

Limited domain coverage:
The evaluation focuses narrowly on spatial perception and video reasoning datasets (e.g., VSI-Bench, STI-Bench, SPAR-Bench, GQA). It remains unclear whether SpaceR enhances higher-level spatial reasoning involving dynamics (e.g., motion prediction, affordances) or multimodal reasoning with more abstract textual context.

Reward interpretability:
The reward formulation combines multiple geometric terms, but the paper does not analyze the stability or sensitivity of these signals—e.g., how robust the reward is to visual noise, occlusion, or linguistic ambiguity.

Lack of theoretical grounding:
While the results are promising, the paper provides no formal justification for why reinforcement with spatial rewards leads to better compositional generalization compared to supervised spatial grounding.

Reproducibility concern:
Implementation details for reward computation (e.g., normalization, projection heuristics, and parameter settings) are only briefly mentioned, making reproduction difficult for non-affiliated teams.

### Questions
Model dependency:
All experiments use Qwen-VL2 as the base model. Could the authors clarify whether this model’s strong spatial pretraining might bias the results? Have they tried SpaceR on another architecture to verify its independence?

Evaluation scope:
The experiments focus mainly on spatial perception and video reasoning benchmarks (e.g., VSI-Bench, STI-Bench, SPAR-Bench). Can the authors comment on whether SpaceR generalizes to more abstract spatial reasoning, such as motion or temporal relations?

Reward robustness:
How sensitive is the spatial reward to noise or ambiguity? For instance, if object detection is slightly inaccurate or the textual relation is underspecified, does the reward remain stable?

Theoretical justification:
Can the authors provide more insight into why reinforcement with spatial rewards improves compositional reasoning compared with supervised grounding? Is there an underlying intuition or theoretical perspective?

Reproducibility:
Could the authors release or further specify the implementation details of reward computation (e.g., normalization, projection functions, hyperparameters) to support external replication?

### Soundness
3

### Presentation
2

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
The paper addresses the significant challenge of video spatial reasoning in Multimodal Large Language Models (MLLMs), which typically struggle to infer 3D spatial structures from video frames. A new dataset called SpaceR-151k, which combines 91,000 question-answer pairs for spatial reasoning derived from the ScanNet dataset, and 60,000 samples for general multimodal understanding. A training strategy named Spatially-Guided Reinforcement Learning with Verifiable Reward (SG-RLVR). This method extends the Group Relative Policy Optimization (GRPO) framework by introducing a "map imagination mechanism". Experiments, using Qwen2.5-VL-7B-Instruct as the base model, show that SpaceR outperforms its base model and SFT-only counterparts on spatial reasoning benchmarks like VSI-Bench, STI-Bench, and SPAR-Bench.

### Strengths
1. The paper targets a well-known and significant weakness in MLLMs: video spatial reasoning.
2. The paper contributes a new dataset (SpaceR-151k) specifically for this task, with verifiable answers derived from 3D scene data. This is a valuable asset for future research.

### Weaknesses
1. The SpaceR-151k dataset is constructed entirely from ScanNet, which also serves as one of the benchmark datasets under evaluation. Although the authors claim to have removed ‘overlapping videos,’ the shared data domain introduces significant risks of data leakage. This may lead to inflated results on benchmarks, thereby undermining the persuasiveness of the model's generalisation capabilities.
2. The proposed method primarily adds spatial rewards and a ‘map imagination’ module to the GRPO framework of DeepSeek-R1 and Video-R1. Whilst this design is conceptually sound, it lacks substantive algorithmic innovation. The overall framework continues to employ existing reward designs, with adjustments made only at the task level, rendering it more akin to an engineering extension than a theoretical breakthrough.
3. The core contribution of the paper, the ‘map imagination mechanism’, lacks sufficient validation of its efficacy. The ablation results in Table 2 demonstrate only a marginal improvement of approximately 1.6%, and the paper fails to provide analyses of map generation accuracy, the correlation between map quality and performance, or visual diversity. The absence of such quantitative evidence leaves it unclear whether this mechanism genuinely enhances spatial reasoning.
4. The layout of the first page is extremely poor. It is distracting, wastes space, and looks unprofessional.
5. The 91k spatial reasoning samples feel more like a proof-of-concept than a large-scale dataset.
6. The idea of encouraging a model to "think" by generating spatial layouts is not new, even if the specific RLVR implementation is. The paper could do a better job of positioning this contribution relative to prior work on structured chain-of-thought and intermediate representations.
7. The paper honestly reports that enabling the think mode can decrease performance on general video understanding benchmarks. This suggests the model is learning a specialized, task-specific reasoning process that may introduce "unnecessary or inaccurate reasoning"  and noise for other tasks. This trade-off is a significant weakness that is not fully resolved.

### Questions
1. Would the authors be willing to reformat Page 1 to follow a standard ICLR layout? The current version with the large, isolated radar chart and excessive white space is a major barrier to readability.
2. Given that performance scales with data (Figure 8), do the authors consider the 91k spatial samples to be sufficient for learning this skill, or is this primarily a proof of concept? Are there plans to scale the dataset, perhaps using other 3D-aware sources?
3. The paper states a $10 \times 10$ map is used. This seems like a coarse-grained representation for complex indoor scenes. What was the rationale for this choice? Was it to reduce the token count of the generated map? Did you experiment with finer-grained grids, and how did that affect performance versus token overhead?

### Soundness
3

### Presentation
1

### Contribution
2

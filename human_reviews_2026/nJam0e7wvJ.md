# Geo-R1: Unlocking VLM Geospatial Reasoning with Cross-View Reinforcement Learning

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 6, 2, 4

## Abstract
We introduce Geo-R1, a reasoning-centric post-training framework that unlocks geospatial reasoning in vision-language models by combining thinking scaffolding and elevating. In the scaffolding stage, Geo-R1 instills a ``geospatial thinking paradigm" via supervised fine-tuning on synthetic chain-of-thought exemplars, enabling models to connect visual cues with geographic priors without costly human reasoning annotations. In the elevating stage, it uses GRPO-based reinforcement learning on a weakly-supervised cross-view pairing proxy. This design supplies a verifiable and scalable reward signal: teaching models to capture and reconcile features across modalities, and harnessing reasoning for accurate prediction. Geo-R1 extends geospatial modeling from domain pretraining / supervised finetuning to reasoning-first post-training, and achieves state-of-the-art performance across various geospatial reasoning benchmarks. Our code is available at https://anonymous.4open.science/r/Geo-R1-ICLR.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes Geo-R1, a post-training framework designed to equip general-purpose vision-language models with geospatial reasoning capabilities. The authors designed a two-stage training pipeline: first, infusing a geographic reasoning paradigm through scaffolding-style supervised fine-tuning, followed by reinforcement learning using verifiable rewards based on a cross-view image matching task to optimize the accuracy and structure of the reasoning. Experiments show that Geo-R1 not only achieves over 60% performance improvement on in-distribution cross-view tasks but also excels on multiple zero-shot geographic reasoning benchmarks, while largely maintaining its general multimodal capabilities without degradation.

### Strengths
1.	The paper accurately identifies the core challenges faced by VLMs in the geospatial reasoning domain. The proposed two-stage training framework, "SCAFFOLDING WITH SFT + RLVR," is well-designed. Experiments also demonstrate the framework's effectiveness.

2.	The evaluation system is very comprehensive, covering in-distribution performance, zero-shot out-of-distribution generalization, capability retention, and detailed ablation studies.

3.	The reward function is reasonably designed. The presentation and analysis of phenomena in the training dynamics, such as the "Aha Moment" and "dual ascent," validate the training's effectiveness.

### Weaknesses
1.	The core techniques of this work are a direct adaptation of paradigms from pioneering works like DeepSeek-R1, constituting combinatorial innovation. No innovation in algorithms or training methods.
2.	As mentioned in the introduction, geospatial reasoning encompasses various tasks. However, the main experiments lack a fine-grained breakdown of performance across different sub-tasks. Is performance consistent across different tasks? 
3.	The article can add some case studies, especially failure case studies.

### Questions
1.	The core techniques of this work are a direct adaptation of paradigms from pioneering works like DeepSeek-R1, constituting combinatorial innovation. No innovation in algorithms or training methods.
2.	As mentioned in the introduction, geospatial reasoning encompasses various tasks. However, the main experiments lack a fine-grained breakdown of performance across different sub-tasks. Is performance consistent across different tasks? 
3.	The article can add some case studies, especially failure case studies.

### Soundness
3

### Presentation
3

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
Geo-R1 proposes a reasoning-centric post-training framework designed to enhance the capabilities of Vision-Language Models in geospatial reasoning tasks. The framework consists of two stages, the scaffolding stage utilizes supervised fine-tuning on synthetic chain-of-thought data to instill a "geospatial thinking paradigm" in the model. and the elevating stage employs GRPO-based reinforcement learning using a cross-view image pairing task as a verifiable, weakly-supervised reward signal to refine the model's reasoning quality and generalization.The method achieves state-of-the-art performance on multiple geospatial reasoning benchmarks while effectively avoiding catastrophic forgetting and preserving the model's capabilities on general-purpose tasks.

### Strengths
1.The paper is well written and easy to follow.
2.The cross-view pairing task is both challenging and easily verifiable, making it highly suitable for weakly-supervised RL.
3. Extensive evaluation covering in-distribution, out-of-distribution generalization, and preservation of primitive abilities show the strength of the method.

### Weaknesses
A significant limitation is that the use of GRPO to enhance MLLM reasoning is no longer novel. The paper does not introduce substantive innovations in the core algorithm or reward mechanism, resulting in a lack of clear technical contribution to the methodology itself.but consider the written quality and solid experiments, I incline to give a positive rate.

### Questions
See weakness

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes Geo-R1, a reasoning-first reinforcement learning framework for geospatial understanding tasks. It uses synthetic chain-of-thought supervision and cross-view alignment as weak rewards to fine-tune a base VLM (Qwen2.5-VL-7B). Experiments on several benchmarks show improvements over open-source general-purpose VLMs.

### Strengths
Adapts an RL-based approach for geospatial reasoning.

Builds a large and diverse synthetic dataset and applies CoT reasoning signals effectively.

Achieves measurable gains on public benchmarks like GeoChain and IMAGEO-Bench.

### Weaknesses
Baseline coverage is incomplete: The paper only compares against general VLMs (Qwen, LLaMA, Claude) but omits numerous strong task-specific geolocalization and remote-sensing models (e.g., GeoCLIP). These are directly relevant to the core tasks evaluated.

All experiments are conducted solely on Qwen2.5-VL-7B, no results are provided for other architectures. This makes it difficult to assess the generalizability or stability of the proposed RL pipeline.

### Questions
Have the authors attempted to apply the same RL pipeline to other backbones (e.g., LLaVA-Next, InternVL, Phi-3-V)?

Why were task-specific geolocalization baselines not included? Many of them have public weights and could provide stronger comparisons.

An anonymous GitHub repository is provided, but it is currently empty. The authors should not claim code availability or include the link unless the source code is indeed provided at submission time.

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
4

### Summary
This paper adopts an SFT stage as a cold-start to warm up the MLLM model to handle geospatial CoT capacities by curating a synthetic dataset. Then, the authors design a cross-view selection reward pipeline to perform the GeoSpatial RL post-training based on the post-training dataset. Meanwhile, this paper conducts SFT, then RL and direct RL manner to train MLLM, like Qwen2.5VL, to evaluate the geospatial generalization abilities of the model. Through careful reward function adaptations, this paper obtains competitive cross-view geospatial evaluation performances.

### Strengths
1. The application of adopting MLLM like Qwen2.5VL to geospatial tasks sounds like an interesting application.

2. The post-training pipeline based on MLLM seems consistent with most RL-needed tasks.

3. The final zero-shot generalization in terms of in-distribution and out-of-distribution performances is competitive and promising.

### Weaknesses
1. Though the final geospatial performances are competitive, the overall training receipts or designs are not that novel, which sounds like an application from the RL GRPO to these new downstream tasks.

2. I am confused about the reward score assignments for the cross-view selection objective, while the correct one gets a 1.0 score, and all the others get -0.8. Does the author demonstrate well about this choice? What if we set 0 for the intermediate options?

3. As from the main paper, the visual appearances among the geo image and the satellite images, does the paper also make use of other metadata to help the model learning, since the large differences are not that reasonable to help the model learning towards mapping between the correct geospatial images and the satellite candidates?

4. Regarding Figure 8, the figure here looks quite blurry. I suggest that the authors consider inserting better presentations.

### Questions
The experimental results are limited, and the comparisons with other MLLMs (either open or closed sources) are very limited in this paper. What if compared with Intern3VL, GPT-4O, and O3? And also, the ablative studies seem very few to understand the effective post-training results in this paper, for example, some ablations about the cross-view between spatial image and satellite images, by changing the satellite images with other mixed spatial images?

### Soundness
3

### Presentation
3

### Contribution
3

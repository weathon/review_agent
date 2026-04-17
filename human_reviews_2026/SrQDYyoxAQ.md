# ReasonGen-R1: Cot for Autoregressive Image Generation Models Through SFT and RL

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 4, 4

## Abstract
Although chain-of-thought (CoT) reasoning and reinforcement learning (RL) have driven breakthroughs in large language models(LLMs), their integration into generative vision models remains underexplored. We introduce ReasonGen-R1, a two-stage framework that first imbues an autoregressive image generator with explicit text-based "thinking" skills via supervised fine-tuning (SFT) on a newly generated reasoning dataset of written rationales, and then refines its outputs using Group Relative Policy Optimization (GRPO).
To enable the model to reason through text before generating images, We automatically generate and release a corpus of model-crafted rationales paired with input prompts, enabling controlled planning of object layouts, styles, and scene compositions.
Our GRPO algorithm uses reward signals from a pretrained vision–language model to assess overall visual quality, optimizing the policy in each update. We further design an adaptive entropy loss to prevent model collapse in this relatively complex task.
Evaluations on GenEval, DPG, and the T2I benchmark demonstrate that ReasonGen-R1 consistently outperforms strong baselines and prior state-of-the-art models.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes ReasonGen-R1, a two-stage training framework to imbue autoregressive text-to-image models with chain-of-thought reasoning abilities. During sft cold start, the authors generate a dataset of textual rationales for image prompts, and fine-tune the image generator to produce these rationales before drawing the image. During GRPO, they are using a single overall visual quality VLM assessment as feedback. Extensive experiments are conducted to demonstrate the method’s effectiveness, along with some in-depth analysis.

### Strengths
* The motivation is clear and easy to grasp and the method demonstrates improved performance on tasks that standard image generators struggle with.
* Good ablation study showing SFT and adaptive entropy loss matters and boost final performance.
* Well documented and transparent training disclosure and data disclosure.

### Weaknesses
* Several highly related work such as GoT-R1, T2I-R1, all uses chain-of-thought plus RL on AR image generation models, is not mentioned or compared in any way at all.
* The abstract suggests the RL reward is mainly about “overall visual quality” as judged by a VLM and a rather small one . This is a very high-level and coarse signal with potential of hallucinations and hacking.
* The experiments seem mostly focused on the compositional benchmark, evaluation on broader text-to-image tasks (COCO etc.) are not existent.

### Questions
Was there any signs of reward hacking observed?

### Soundness
2

### Presentation
3

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
ReasonGen-R1 proposes a two-stage framework designed to enhance the capabilities of autoregressive image generation models (based on Janus-Pro-7B). Its core contribution lies in integrating the successful Chain-of-Thought (CoT) reasoning mechanism from LLM into a visual generation pipeline, enabling a "think-and-generate" process.

The framework employs a Supervised Fine-Tuning (SFT) stage, utilizing 200k high-quality CoT trajectories annotated by GPT-4.1, to teach the model to explicitly generate reasoning plans. Subsequently, a reinforcement learning (RL) stage based on Group Relative Policy Optimization (GRPO) is used for training, incorporating an adaptive entropy loss to ensure training stability and address entropy explosion in multimodal sequences. 

Experimental results demonstrate that this method consistently outperforms baseline models on multiple benchmarks, significantly improving both image quality and instruction alignment.

### Strengths
1.  The key contribution of this work lies in introducing the Chain-of-Thought (CoT) and Reinforcement Learning (RL) paradigms, which have proven effective in the LLM domain, into autoregressive image generation models. By enabling the model to generate a reasoning plan before creating an image, it effectively decomposes complex instruction-following tasks into manageable intermediate steps. 

2.  The two-stage training framework ensures that the model learns the correct reasoning structure and format while continuously improving its performance.

3.  The paper provides a comprehensive quantitative evaluation on multiple generation benchmarks. The results demonstrate that ReasonGen-R1 surpasses the strong baseline model, Janus-Pro-7.

4.  Applying RL to autoregressive generative models with interleaved modalities (mixed text/image tokens) is highly prone to training instability. The proposed Adaptive Entropy Loss design effectively mitigates issues of entropy explosion or entropy collapse.

### Weaknesses
1.  The reward model (RM) is built upon Qwen-2.5-VL and provides binary scores. The current binary scoring can be quite extreme – minor deviations in text or image quality might result in a reward of 0, which could pose challenges for training. 

2.  The autoregressive generative model must generate an entire CoT text sequence during inference, which inevitably increases inference latency. Although performance is improved, the additional computational overhead presents a challenge for real-time or high-throughput application scenarios. The paper should provide a quantitative analysis discussing the trade-off between the extra latency introduced by the "thinking" process and the corresponding performance gains.

### Questions
1. The reward model (RM) is built upon Qwen-2.5-VL and provides binary scores. Were more fine-grained scoring schemes explored? More analysis regarding the RM would be beneficial.

2. Janus-Pro is inherently a unified model. After this targeted training, how are its original capabilities, such as general understanding, affected?

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
It proposes a two-stage pipeline for autoregressive image generation. On GenEval, DPG-Bench, and T2I-Benchmark, REASONGEN-R1 outperforms Janus-Pro-7B and surpasses many diffusion and autoregressive baselines.

### Strengths
1. The motivation is clear.
2. The combination of textual reasoning and image tokens is novel.
3. The ablation study is comprehensive.

### Weaknesses
1. RL reward is provided by a single VLM judge (Qwen2.5-VL-7B). Is the policy overfitting that judge?
2. The evaluation benchmarks (GenEval, DPG-Bench, T2I-Benchmark; Tables 1–3) mostly test object count, color binding, spatial relations, etc. What about the human preference evaluation benchmark? For instance, MM-RewardBench.
3. Human evaluation is missing.
4. Figure 4 shows RL is unstable without adaptive entropy loss. The theoretical justification could be proposed.
5. The work does not achieve optimal performance on T2I-Benchmark. It is nice to give further analysis.
6. The setting appears oversimplified, as the inference process seems limited to single-step generation. It would be valuable to examine its effectiveness in multi-round iterative refinement or video generation scenarios.

### Questions
Please refer to the Weaknesses.

### Soundness
2

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
5

### Summary
The paper presents ReasonGen-R1 which a wo-stage training paradigm combining supervised fine-tuning (SFT) with chain-of-thought (CoT) and RL. Although the paper addresses proposes some interesting approaches and questions, please find my detailed comments on weakness and strengths.

### Strengths
1. The method is well motivated

2. Current results show that out of the methods considered here the proposed method outperforms

3. The paper uses a simple yet straightforward methodology

### Weaknesses
1. Without using some standard large scale benchmarks like Imagenet it is very hard to judge the quality of the model.

2. Although the authors evaluate on DPG bench, genval and compbench. There lies a very inherent bias and noise specific to these benchmarks, they use methods like object detectors which can throw out a lot of false negatives and cannot detect classes beyond a fixed vocabulary. What are steps taken to make sure that the this method does not have these biases

3. The comparisons are outdated, I would have liked to see some better competitor models like GPT-4o, Seedream, Nano Banana, Imagen 4/4-ultra

4. In tab 4 I would have liked to see more models and bigger models

5. Any insights on things like reward hacking or potential biases and issues can be an interesting addition

### Questions
See weaknesses

### Soundness
3

### Presentation
4

### Contribution
3

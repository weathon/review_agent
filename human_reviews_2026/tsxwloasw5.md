# Vision-Language-Action Instruction Tuning: From Understanding to Manipulation

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
To operate effectively in the real world, robots should integrate multimodal reasoning with precise action generation. However, existing vision-language-action (VLA) models often sacrifice one for the other, narrow their abilities to task-specific manipulation data, and suffer catastrophic forgetting of pre-trained vision-language capabilities. To bridge this gap, we introduce **InstructVLA**, an end-to-end VLA model that preserves the flexible reasoning of large vision-language models (VLMs) while delivering leading manipulation performance with the help of embodied reasoning. InstructVLA introduces a novel training paradigm, *Vision-Language-Action Instruction Tuning (VLA-IT)*, which employs multimodal training with mixture-of-experts adaptation to jointly optimize embodied reasoning and action generation on both standard VLM corpora and a curated 650K-sample VLA-IT dataset. On in-domain SimplerEnv tasks, InstructVLA achieves 33.3% improvement over SpatialVLA. To evaluate generalization, we introduce SimplerEnv-Instruct, an 80-task benchmark requiring closed-loop control and high-level instruction understanding, where it outperforms a fine-tuned OpenVLA by 96% and an action expert aided by GPT-4o by 29%. Additionally, InstructVLA surpasses baseline VLMs on multimodal tasks and exhibits inference-time scaling by leveraging textual reasoning to boost manipulation performance in both simulated and real-world settings. These results demonstrate InstructVLA's potential for bridging intuitive and steerable human-robot interaction with efficient policy learning.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors argue that prior iterations of VLA models have not placed as much focus on embodied reasoning as they should. By embodied reasoning, we mean the ability of the model to move from being told exactly what manipulation to do, to a robot that can use context clues from the environment to decide what manipulation to do.

To address this, the authors introduce a new evaluation benchmark, based on augmenting the SimplerEnv setup with multilingual instructions, more indirect descriptions of objects (i.e. "I need something to clean with"), etc. They also train the VLA in a 2 stage process. First, an action expert is pretrained on a large manipulation dataset, to predict both learnable latent actions and language descriptions of the actions. This  is tuned against a fixed VLM backbone, using a flow matching loss. This approach was popularized by works like pi_0, but then as a 2nd stage, the authors train LoRAs on the LLM backbone against the fixed action expert. (Some experiments were done with finetuning the action expert too, and results were generally similar / slightly worse). The authors argue this 2nd stage can be viewed as a form of instruction tuning, as mediated by the action expert, to answer more complex questions.

This model is shown to perform reasonably on two prior benchmarks, a Google Robot and WidowX Robot. On the new SimplerEnv-Instruct benchmark, InstructVLA-Expert (the model trained only with the 1st stage) is only around on par with finetuned OpenVLA, but performs signficantly better after the stage 2 LLM backbone finetuning.

### Strengths
Paper does a good job of arguing that existing VLAs perform subpar at more complex queries, and the idea that the LLM backbone needs to be finetuned to exploit the action expert for embodied understanding seems reasonable. The proposed SimplerEnvInstruct benchmark is generally reasonable and gains on this benchmark are pretty significant. Ablations on different parts of the architecture, dataset, etc. are pretty extensive.

### Weaknesses
I think the paper is somewhat overreaching on how much it contributes to embodied reasoning. Even in the pre RT-1 days, there were some papers trying to do embodied reasoning. In my opinion the main difference these days is that the smaller, low latency VLMs are now more capable of answering these questions on their own.

Similarly, I find it difficult to attribute how much gain comes from the general evolved understanding that MoE backbones are better than dense backbones (many baseline models date to before this time).

Last, there is always some benchmark bias, where in general you expect a paper to do better on its self proposed benchmark compared to other models, the largest gains coming from the self-proposed benchmark gives me some pause.

nit: please label the y-axes in Figure 5, Figure 6. Please make it clearer how many trials were done for each eval number and/or include confidence intervals (why do these only exist in the SimplerEnv-Instruct columns?). In Table 1/2, I think including robot state is a significantly more important variable than the rest and would prefer if that were separated from the other results more in some way. (It is also unclear which baselines methods use robot state and which do not.)

### Questions
When creating SimplerEnv-Instruct, what variations were considered? I see that there are around 80 tasks + 1.1k trials which necessarily limits coverage for practicality reasons, but I am curious what things were decided to not be covered.

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
5

### Summary
This paper introduces InstructVLA , a model designed to solve "catastrophic forgetting" in Vision-Language-Action (VLA) models, a common problem where models lose their pre-trained multimodal reasoning abilities after being fine-tuned on manipulation tasks. The authors propose a novel training paradigm called "Vision-Language-Action Instruction Tuning (VLA-IT)" , which uses a Mixture-of-Experts (MoE) framework and a new 650K-sample dataset to jointly optimize both textual reasoning and action generation. As a result, InstructVLA successfully preserves its powerful language and reasoning capabilities while delivering state-of-the-art manipulation performance , significantly outperforming baselines like OpenVLA on the new SimplerEnv-Instruct benchmark.

### Strengths
1. VLA-IT Paradigm: It introduces a novel training paradigm called "Vision-Language-Action Instruction Tuning (VLA-IT)". This paradigm uses a Mixture-of-Experts (MoE) adaptation framework to jointly optimize textual reasoning and action generation, trained on standard VLM corpora and a new 650K-sample VLA-IT dataset.
2. Two-Stage Training: The training is divided into two stages: (1) Action Pretraining, which trains a VLM-driven "action expert" ; and (2) VLA Instruction Tuning, which freezes the action expert and fine-tunes only the VLM backbone via the MoE module to handle complex instructions and multimodal reasoning.
3. New Dataset and Benchmark: The paper contributes two significant resources: (a) a 650K VLA-IT dataset with diverse instructions, scene captions, and Q&A pairs ; and (b) a new benchmark, SimplerEnv-Instruct, featuring 80 tasks designed to evaluate generalization, closed-loop control, and high-level instruction understanding in VLAs.

### Weaknesses
1. This work is the combination of MoE and the fine-grained instruction finetuning. It seems so incremental. Especially, I can't see any advantages from the design. The biggest contribution is that the author gives the a 650K VLA-IT dataset with diverse instructions, scene captions, and Q&A pairs.
2. More related works containing the step by step reasoning ability should be added and discussed like CoT-VLA.
3. Authors should give more attention to the motivation. Particularly when the paper said their problem about ''how to train robotic manipulation skills without suffering "catastrophic forgetting" of the powerful multimodal reasoning capabilities inherent in pre-trained Vision-Language Models (VLMs)'', I don't know how instruction tuning can solve this kind of problems.

### Questions
See weakness.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces InstructVLA, which is a VLA model that preserves large vision-language reasoning capabilities while achieving state-of-the-art manipulation performance. To achieve this, it introduces VLA-IT training paradigm with a 650K dataset. To evaluate generalization, this paper also proposes SimplerEnv-Instruct benchmark. InstructVLA achieves good performance on multimodal, Simpler, and SimplerEnv-Instruct benchmarks.

### Strengths
This paper shows a good example of how to effectively keep the capability of LLM/VLM in a VLA system, which is a very critical problem for current VLA models. It also demonstrates how to further enhance the reasoning ability of VLA. The proposed dataset and benchmark can also be a good reference for later works.

### Weaknesses
Please see the questions.

### Questions
**1.** The description of the architecture and training paradigm is quite confusing.

   **1.1** For example, they mention an action lora, a language lora, MoE adaptation, and a scalar head. However, it is hard to find details about them in Figure 2. Are the MoE and language lora only activated in stage 2? 

   **1.2** I'm also confused about the training details of stage 1. The authors mention that they use the data of RT-1, and the model needs to predict both action and language, so what is the input of the model in stage 1?  If the task prompt and images are the input, is the output language generated by the authors? 

   **1.3** In stage 1, what is the meaning of the action lora? Where exactly is this action lora applied? Do both language tokens and action tokens pass through the action lora? If only action tokens go through it, then what’s the point of the language loss? Is it just used as a simple regularizer?

**2**. The metric of the SimplerEnv-Instruct benchmark is success rate? 
**3.** Table 2 shows that the training stage 2 is harmful to the normal simpler tasks, and there is little explanation about it. Since the necessity of reasoning VLA data is critical, can the authors provide more discussion?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The article proposes a novel VLA model architecture that uses MoE architecture to effectively address the critical challenge of catastrophic forgetting of pre-trained VLA capabilities. The work is exciting and solid, providing a clear narrative that successfully addresses the challenge of integrating multimodal reasoning ability into VLA models. The introduction of a dedicated evaluation task, SimplerEnv-Instruct, is also a substantial contribution to the field.

### Strengths
Effective Resolution of Core VLA Challenges and Strong Performance: The work successfully addresses major VLA obstacles, namely catastrophic forgetting of multimodal skills and the difficulty of tightly integrating high-level reasoning with precise low-level control. Also itsresults are compelling, validating the model’s efficacy.

Novelty in Training Paradigm and Architecture Design: The core technical strength lies in the Vision-Language-Action Instruction Tuning (VLA-IT) paradigm and its architectural implementation. This approach skillfully unifies autoregressive language generation with flow-based action generation using a Mixture-of-Experts (MoE) adaptation framework within a two-stage training strategy. This design allows for the dynamic alternation between textual reasoning and action execution , providing an efficient solution to task interference while preserving general VLM knowledge. and the diagram of this part is very clear.

The authors also curate a large-scale, customized 650K-sample VLA-IT dataset featuring detailed annotations for embodied scene understanding and planning, essential for bridging VLM knowledge with embodied scenes.

### Weaknesses
Lack of explaination of Architecture part. The proposed architecture, particularly the reliance on the Mixture-of-Experts (MoE) adaptation and the design of the latent action space, is central to the paper's claims of solving catastrophic forgetting and enabling unified reasoning-guided manipulation. However, key mechanistic details and the representational quality require further clarification to substantiate the claims.

Please provide a detailed description of the Scale Head (gating network) component of the MoE. Specifically, what is its architecture, what precise input does it take (e.g., is it the hidden state of the VLM?), and how is it trained during the Vision-Language-Action Instruction Tuning (VLA-IT) stage?

Also, the core challenge in co-training VLM capabilities and action skills is catastrophic interference. Did the training process, besides the standard $\mathcal{L}_{LM} + \mathcal{L}_{FM}$ loss3333, include any regularization loss (e.g., sparsity or orthogonality constraints) specifically designed to encourage or enforce the dynamic switching between the Language Adapter and the Action Adapter? Such detail is crucial to validate the claim that MoE effectively enables adaptation while preserving pre-trained knowledge.

### Questions
Please clarify the 31.7% increase in the performance part is a relative percentage or an absolute percentage increase. In my calculation its a relative percentage, but you sould clarify it for easier understanding.

### Soundness
4

### Presentation
4

### Contribution
3

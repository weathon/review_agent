# GoT-R1: Unleashing Reasoning Capability of Autoregressive Visual Generation with Reinforcement Learning

- Decision: Accept (Poster)
- Scores: 2, 4, 6, 6

## Abstract
Visual generation models have made remarkable progress in creating realistic images from text prompts, yet struggle with complex prompts that specify multiple objects with precise spatial relationships and attributes. Effective handling of such prompts requires explicit reasoning about the semantic content and spatial layout. We present GoT-R1, a framework that applies reinforcement learning to enhance semantic-spatial reasoning in autoregressive visual generation models. Leveraging the natural affinity between autoregressive architectures and sequential reasoning, our approach builds upon the Generation Chain-of-Thought framework to enable models to autonomously discover effective reasoning strategies beyond predefined templates. To achieve this, we propose a dual-stage multi-dimensional reward framework that leverages MLLMs to evaluate both the reasoning process and final output, enabling effective supervision across the entire generation pipeline. The reward system assesses semantic alignment, spatial accuracy, and visual quality in a unified approach. Experimental results demonstrate significant improvements on T2I-CompBench and GenEval benchmark, particularly in compositional tasks involving precise spatial relationships and attribute binding. GoT-R1 advances the state-of-the-art in autoregressive image generation by successfully transferring sophisticated reasoning capabilities from language models to the visual generation domain. Code is available at https://github.com/gogoduan/GoT-R1.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes GoT-R1, an RL framework to improve autoregressive image generation models on complex text prompts by incorporating explicit reasoning. The approach introduces a dual-stage, multi-dimensional reward scheme: one stage evaluates the reasoning process itself, and another evaluates the final image, using vlm as judges. On multiple benchmarks, GoT-R1 shows significant improvements in placing objects correctly and binding attributes, outperforming the previous sota in complex prompt fidelity.

### Strengths
* The method shows significant improvement on compositionally of image gen tasks.
* Clear ablation study that supports the dual stage and reward design, RPI‑only helps but full reward performs best, while RRI‑only is harmful

### Weaknesses
* The idea of injecting chain-of-thought into image generation and refining with RL is not unique to this paper. For example: ReasonGen-R1, T2I-R1 are both very similar works and comparative discussions are limited. 
* The experiments seem mostly focused on the compositional benchmark, evaluation on broader text-to-image tasks are very limited.

### Questions
For equation (3), the total reward is a sum (R_sem + R_spa) which is [0,2] times a bunch of other [0,1]s, wouldn't this make the total reward be [0,2]?

### Soundness
3

### Presentation
4

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
This paper introduces ​​GoT-R1​​, a novel framework that enhances the ability of autoregressive visual generation models to handle complex, compositional text prompts. The key innovation is a ​​dual-stage, multi-dimensional reward function design for GRPO training​​ that uses a Multimodal LLM (MLLM) to evaluate both the reasoning processand the final image.

### Strengths
- The paper leverages the proven success of RL (GRPO) in enhancing reasoning for language models like DeepSeek-R1and applies it to the challenging domain of compositional visual generation. The motivation is clear: predefined reasoning templates are a bottleneck, and RL is a natural solution for discovering better strategies.

### Weaknesses
- No code and data is provided to reproduce the experiments.
- The approach is computationally expensive. Training involves sampling multiple candidates (N=16) per prompt, each evaluated by a 7B parameter MLLM (Qwen2.5-VL) as the reward model.
- The paper directly applies the original GRPO algorithm with an improved reward function to the new model. As a technical contribution, the effort remains insufficient, with the primary novelty being several rewards.
- The paper mentions T2I-R1 as concurrent work but does not provide a detailed comparison. Since T2I-R1 also uses RL for CoT in image generation, a clearer distinction of GoT-R1's unique contributions.

### Questions
- Why the total reward is defined as the product of individual rewards? Further exploration is warranted regarding why addition yields less than multiplication, ideally with more specific case studies on rewards.
- How well does the reward model generalize to prompts that are out-of-domain or contain concepts not well-represented in the MLLM's training data?
- Could you provide some failure cases for GoT-R1? I think analyzing a case where even the RL-trained model fails could reveal valuable insights into the remaining limitations of the approach, such as the fundamental constraints of the base autoregressive architecture or the reward design.
- Can the rewards designed in this paper be used in other GRPO-like methods such DAPO [1]?


[1] DAPO: An Open-Source LLM Reinforcement Learning System at Scale

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduce GOT-R1 a framework apply reinforcement learning to enhance semantic special reasoning in autoregressive visual generation models. This method is build based on the generation chain-of-thought framework, which first generated the reasoning process given the prompt, and then generates images following the reasoning process. The motivation of incorporating reinforcement learning into the reasoning change generation is that the author observe that GoT-generated reasoning chains can be unfaithful to the prom, despite following templates well. Thus authors introduce GoT-R1, adapting reinforcement, learning, advances to enhance semantic spatial reasoning in autoregressive vision generation. Specifically, in GoT-R1, there is a comprehensive reward framework to ensure that effective of training, including rewards for prompt to reasoning semantic alignment, prompt-reasoning spatial alignment, reasoning–to–image alignment, and prompt-to-image alignment. Experiment results demonstrate that GoT-R1 achieves state of the art on T2I-CompBench and Geneval.

### Strengths
1. This paper introduces an effective way to enhance the chain-of-thought reasoning capability of autoregressive visual generation model. Experiment results directly support the effectiveness of the proposed method. 
2. The reward framework is comprehensive, cross alignment between prompt, reasoning chain and images are considered. This rewards are general, and could provide inspiration to the rewards design of other visual generation model. 
3. The paper is well organised, clearly written and easy to follow and understand.

### Weaknesses
1. The experiments are conducted with Janus-Pro-1B and Janus-Pro-7B as the base model. It'll be interesting to see how the GoT-1 framework would perform with other base models. (I know it is not a necessary experiment.)
2. The experiments really demonstrates that the GoT-R1 method help improve the quality of the final generated image. While the quality of chain of thought is not evaluated, especially compared to the original GoT. 
3. Some of reference of the table in the main text are wrong, for example,  table 4.4 in line 433, table 4.5 and line 445.

### Questions
1. In line 322, it is said that the parameter of MLLM are updated through LoRA finetune. As I understand, the NLLM is used as the reward model which should be fixed. Why the MLLM should be finetuned? 
2. During the experiment, Qwen2.5VL-7B is selected as the reward model. Have you evaluate how reliable Qwen2.5VL-7B is as the reward model?

### Soundness
3

### Presentation
3

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
This paper proposes a reinforcement learning framework for training Generation Chain-of-Thought (GoT) capabilities in image generation. The base model used in this framework is a unified generative understanding model, which leverages the base model's ability to generate text, coordinates, and images. A dual-stage multi-dimensional reward framework is designed and trained using the GRPO algorithm. Experimental results show that the proposed framework delivers significant performance improvements on T2I-CompBench and GenEval.

### Strengths
1. This paper uses reinforcement learning to enhance Generation Chain-of-Thought (GoT) capabilities, which is an interesting and meaningful perspective in the field of multimodal generation.
2. This paper proposes a dual-stage multi-dimensional reward framework that comprehensively utilizes the prompts, reasoning content, and images in spatial and semantic alignment. This design effectively utilizes the capabilities of the unified generative understanding model.

### Weaknesses
1. The paper does not specify the number of GRPO training steps or the amount of data used. No curves illustrating the GRPO training process are shown.

2. The authors used a model that unifies generative understanding, but did not evaluate the performance of the trained model in multimodal understanding. Why is that? I am very interested in how the multimodal understanding capability changes after training with reinforcement learning on GOT.

3. The ablation experiments shown in Table 4 are puzzling. For example, row 5 (w $R_{PI}$) plays a crucial role, but row 8 (w $R_{RI}$ & $R_{PI}$) with the addition of $R_{RI}$ performs worse, and the penultimate row further weakens the gain. However, adding $R_{sem}$ at the end reverses this and becomes the best performance. The ablation experiment results here are strange, making me doubt the reliability of the experiment.

### Questions
1. Refer to the issues raised in the weakness section.

2. Why is multiplication used between rewards in Equation 3? Summation is generally used now. Experiments are needed to verify this. The curves showing the changes in rewards during training should also be added.

3. Why not utilize the generative understanding of the unified model's own understanding capabilities to obtain rewards, thus avoiding reliance on an additional large multimodal model as the reward model?

### Soundness
3

### Presentation
3

### Contribution
3

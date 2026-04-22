# MoDA: Modulation Adapter for Fine-Grained Visual Understanding in Instructional MLLMs

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 2, 6, 4

## Abstract
Multimodal Large Language Models (MLLMs) have achieved remarkable success in instruction-following tasks by integrating pretrained visual encoders with large language models (LLMs). However, existing approaches often struggle with fine-grained visual grounding due to semantic entanglement in visual patch representations, where individual patches blend multiple distinct visual elements, making it difficult for models to focus on instruction-relevant details. To address this challenge, we propose MoDA (Modulation Adapter), a lightweight module that enhances visual grounding through instruction-guided channel-wise modulation. Following the standard LLaVA training protocol, MoDA operates in the second stage by applying cross-attention between language instructions and pre-aligned visual features, generating dynamic modulation masks that emphasize semantically relevant embedding dimensions while de-emphasizing irrelevant information. This targeted refinement enables more precise visual-language alignment without architectural modifications or additional supervision. We conduct comprehensive evaluation across 13 diverse benchmarks spanning visual question answering, vision-centric reasoning, and hallucination detection. MoDA demonstrates substantial improvements, achieving notable gains of +12.0 points on MMVP hallucination detection and +4.8 points on ScienceQA reasoning, while consistently outperforming baselines on 12 out of 13 benchmarks with minimal computational overhead (<1% FLOPs). Our results establish MoDA as an effective, general-purpose enhancement for improving fine-grained visual grounding in instruction-tuned MLLMs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces a lightweight module MoDA to address the limitation of semantic entanglement in MLLMs in visual representations. MoDA tackles this by performing instruction-guided, channel-wise modulation on visual features. It uses a cross-attention mechanism to attend language instructions with pre-aligned visual features and generates a dynamic modulation mask. The module is designed to be plug-and-play into the MLLM instruction tuning. Experiments across diverse benchmarks show the promising results of MoDA.

### Strengths
1. The writting is clear.
2. The proposed method is easy to implement, lightweight, and indicates promising results.
3. The experimental settings and benchmarks evaluated are thorough.

### Weaknesses
1. Even though the proposed method achieves promising results among multiple benchmarks, it remains unclear how the proposed method achieves disentanglement. The author should further discuss this.
2. MoDA seems to bring more gains for Vicuna-7B backbone, while smaller performance improvement for LLaMA 3.1-8B (Table 2/3). Is MoDA become more effective when the language backbone is weaker? 
2. How does this work different from existing work such as AdaLink [1]?

[1] Wang, Yaqing, et al. "Non-intrusive adaptation: Input-centric parameter-efficient fine-tuning for versatile multimodal modeling." arXiv preprint arXiv:2310.12100 (2023).

### Questions
Please refer to the Weaknesses section.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors introduce MoDA (Modulation Adapter) , a lightweight module designed to enhance fine-grained visual grounding in Multimodal Large Language Models (MLLMs). The core motivation is to address the problem of semantic entanglement in ViT patch representations, where a single patch blends multiple distinct visual elements, hindering the model's focus on instruction-relevant details.

### Strengths
MoDA achieves substantial and decisive performance gains on tasks requiring fine-grained visual discrimination and robustness against hallucination, notably MMVP (+12.0 points) and ScienceQA (+4.8 points).

### Weaknesses
W1: Lack of Robustness and Fragility in Generalization.The catastrophic performance drop on MMBench-Cn (from 68.0 to 63.6 with SigLIP-S2) fundamentally undermines the claim of MoDA being a "general-purpose enhancement." This regression, attributed by the authors to data distribution, demonstrates that the core instruction-guided feature selection is brittle and tightly coupled to the linguistic space of the training data. If the visual feature disentanglement were truly robust, minor linguistic shifts (e.g., Chinese instructions) should not cause such a collapse, implying a failure in the cross-attention mechanism to generalize the correct channel modulation knowledge.

W2: Architectural Incrementalism and Redundancy with Existing Methods.MoDA's core is a standard Cross-Attention block performing query-conditioned feature re-weighting. This is not novel; models like InstructBLIP and Q-Former already utilize instruction-conditioned attention to select visual features. MoDA is merely a post-hoc, lightweight application of this established mechanism placed at the LLM input. It constitutes an incremental refinement on feature alignment rather than a breakthrough in architectural design, failing to justify its classification as a distinct "fourth paradigm" against existing attention/masking strategies.

W3: Inadequate Solution for Semantic Entanglement (Soft Modulation vs. Hard Sparsity).The stated problem is "semantic entanglement." However, MoDA only performs soft re-weighting (modulation) and cannot enforce "explicit sparsity" (setting weights to zero). If a channel truly encodes an entangled mixture of the relevant dog ear color and the irrelevant plush toy texture, merely attenuating that channel (soft re-weighting) is a weak solution. A truly disentangling mechanism requires a hard, instruction-conditioned feature selector to enforce channel sparsity, a mechanism MoDA structurally lacks.

W4: Superficial Cross-Attention Design Choice.The model relies on tokens ($\mathbf{T}$) from the "initial layers of the LLM." This design is suspicious, as later LLM layers contain more abstract, semantically rich representations of the instruction. The paper fails to provide a robust ablation comparing this choice against using tokens from the LLM's final, more refined layers. This suggests the design was chosen for convenience rather than optimality, potentially limiting the quality of the modulation mask $F(\cdot)$ for complex reasoning instructions.

W5: Insufficient Ablation on Optimal Cross-Attention Depth.The most critical ablation (Table S2) focuses on the performance collapse of the Linear MLP variant when increasing depth (2 layers to 4 layers), leading the authors to choose 2 layers. However, the superior Cross-Attention architecture is only tested at 2 layers. There is no systematic ablation to show that a 1-layer or 3-layer Cross-Attention MoDA wouldn't be more optimal for the superior model variant. The conclusion to use 2 layers is based on the failure of a structurally different, inferior architecture.

W6: Insufficient Discussion of Precedent Work on Instruction-Guided Feature Manipulation. The paper's related work section exhibits a critical lack of discussion regarding highly relevant and precedent methods that also explicitly use language instructions to regulate and refine the visual representation flow:

Instruction-Guided Multi-Layer Fusion: The paper ignores work like Instruction-Guided Fusion (arXiv:2501.08443), which uses instruction to dynamically fuse features from multiple encoder layers. This demonstrates a broader, co-emergent effort to use instruction to select the most relevant visual information source (whether layer, token, or channel) based on the query.


Modality Linear Steering (MoReS): The authors fail to discuss LLaVA Steering (arXiv:2412.12359), which introduces Modality Linear Representation-Steering (MoReS). MoReS addresses modality imbalance by applying linear transformations across multiple model layers to steer visual features. This idea of using linear modulation to explicitly re-balance modality influence across the visual subspace is highly pertinent to MoDA’s core mechanism and challenges its unique architectural contribution.


Pre-LLM Grounding Enhancement (EAGLE): EAGLE (arXiv:2501.02699) focused on visual feature refinement before LLM integration to mitigate hallucinations. MoDA's positioning and objective are a direct parallel to this pre-LLM refinement paradigm, and the omission of this related school of thought significantly weakens the argument for MoDA's architectural novelty in combating fine-grained visual issues.

No Open Source: Unverifiable Claims: Code and Weights Are Not Provided.Despite strong claims regarding computational efficiency (<1% FLOPs) and highly specific architectural gains (e.g., Cross-Attention's superiority over Self-Attention), the lack of public code and model weights renders these key findings unverifiable.

### Questions
see above

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses MLLMs’ poor fine-grained visual grounding due to semantic entanglement in visual patch representations. It proposes MoDA, a lightweight module integrating into LLaVA’s two-stage training. MoDA uses cross-attention between language instructions and pre-aligned visual features to generate dynamic channel-wise masks, emphasizing relevant embeddings while de-emphasizing irrelevant ones—no architectural changes or extra supervision needed. Evaluation on 13 benchmarks show the advantages of MoDA.

### Strengths
S1. This paper is well written and is easy-to-follow.

S2. Accoding to the evaluation results, the performance of MoDA is strong.

S3. MoDA is able to plug-in play with various LLMs.

S4. Performance evaluation side is good. Extensive.

### Weaknesses
The major weaknesses from my point of view are as follows:

W1. Lack of visual comparisons. The authors propose MoDA to refine the tokens. Thus, the representations of the tokens should be compared and discussed.

W2. The method is performed in an implicit manner. Although is easy-to-use, the performance will be limited.

### Questions
How about plug-in play with QWEN-VL and InternVL families?

As a easy-to-use method, without complex design, computation overhead, I have no major reasons to reject this paper. Therefore, I vote for a positive rating. I will update my final score based on the rebuttal and other reviewers' comments.

Typos:

Tab. 4, POPE, 87.9 should be bolded.

### Soundness
3

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
3

### Summary
A new multi-modal adapter experiment to do multi-modal llm. 

It seemed the most selling point was a 'modulation' factor rather than previous BLIP work. And the motivation and experiment setup are clean and showed it was not a result of pure data set scaling.

### Strengths
There are clean ablations going on. 
They have abilated across self-attn and cross-attn, different llm backbone and vision encoder, even though the differences between the scores are not significant

### Weaknesses
Not sure to me how this would scale to multiple images and videos. 
Most benchmarks and work feels about 1.5 year old. Llava 1.5 was last updated in late Dec 2023. There are no comparsion to any model new in 2024 and onwards, so it felt like this submission is unfortunately not current anymore.

### Questions
See weakness section.

### Soundness
3

### Presentation
2

### Contribution
4

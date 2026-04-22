# Video-STAR: Reinforcing Open-Vocabulary Action Recognition with Tools

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 8, 4, 6

## Abstract
Multimodal large language models (MLLMs) have demonstrated remarkable potential in bridging visual and textual reasoning, yet their reliance on text-centric priors often limits their ability to disentangle semantically similar actions in open-vocabulary scenarios. To address this, we propose Video-STAR, a framework that harmonizes contextual sub-motion decomposition with tool-augmented reinforcement learning for open-vocabulary action recognition (OVAR). Unlike prior methods that treat actions as monolithic entities, our approach innovatively decomposes actions into discriminative sub-motions for fine-grained matching while dynamically invokes domain-specific tools for cross-modal interleaving, thereby enabling category-specific reasoning capacity and reducing cross-modal hallucination. Moreover, by designing a hierarchical reward that balances tool-usage efficiency, sub-motion relevance, and structural coherence in reasoning, our method autonomously leverages external tools to prioritize sub-motion patterns without explicit supervision, transmitting from text-centric reasoning to visually grounded inference. Extensive evaluations on HMDB-51, UCF-101, SSv2, Kinetics-400, and Kinetics-600 datasets demonstrate our state-of-the-art performance, outperforming existing methods in distinguishing fine-grained actions and handling cross-modal hallucination, while maintaining computational efficiency.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes Video-STAR, a framework for open-vocabulary action recognition that integrates contextual sub-motion decomposition with tool-augmented reinforcement learning on top of MLLMs. Actions are no longer treated as atomic labels; instead, they are decomposed into discriminative motion primitives, while external tools are invoked to reduce cross-modal hallucinations and enable category-specific reasoning. A hierarchical reward is designed to jointly optimize tool-usage efficiency, structural coherence, and sub-motion relevance. Extensive experiments demonstrate substantial gains over CLIP-based baselines and vanilla MLLMs across multiple settings.

### Strengths
The combination of sub-motion decomposition and tool-augmented RL constitutes a meaningful departure from static cross-modal alignment pipelines commonly used in OVAR.

The method delivers consistent and large improvements across diverse benchmarks and evaluation settings.

### Weaknesses
1. While the integration is well-designed, the core building blocks (tool-augmented CoT, RL-based post-training, and sub-action decomposition) are all known paradigms; the novelty is primarily at the system-level composition rather than at the level of a fundamentally new principle.

2. The approach depends on specific external tools, yet the paper does not analyze robustness to tool inaccuracies or the portability of the method under alternative tool choices.

3. The computational overhead of repeated tool invocation and multi-round RL inference is not reported nor compared against CLIP-based or purely SFT-based OVAR pipelines. Furthermore, the training cost of GRPO fine-tuning is also not quantified or compared with prior methods.

4. Although standard CLIP-based OVAR methods are included, the paper does not benchmark against recent LLM-augmented CLIP paradigms that explicitly incorporate generative priors for action understanding, such as [1–3].

5. The evaluation is conducted primarily against mid-scale or earlier-generation MLLMs (e.g., Qwen2.5-VL), without comparison to state-of-the-art frontier models (e.g., Qwen3-VL, GPT-5, Claude, Gemini), many of which already demonstrate strong video reasoning capabilities.

[1] Building a Multi-modal Spatiotemporal Expert for Zero-shot Action Recognition with CLIP

[2] Generating Action-Conditioned Prompts for Open-Vocabulary Video Action Recognition

[3] VTD-CLIP: Video-to-Text Discretization via Prompting CLIP

### Questions
Please see the weakness.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes, implements and investigates a novel strategy and framework, i.e., Tools Augmented CoT reasoning for zero-shot fine-grained human action recognition. The framework employs VFMs, VLMs and Visual RAG to extract sub-concept vision representations such as human, pose, and video explanation, and generated a combined prompt on proposed format for CoT to MLLM for final stage prediction. The core innovations are the format definition and implementation of sub-action decomposition, candidate selection, and matching scoring for CoT and reinforcement learning. Concrete evaluations are performed on five formal benchmarks and the results shown significant improvements over the SOTA.

### Strengths
A novel Tool Augmented CoT framework and implementation approach for open-vocabulary action recognition (OVAR), concrete evaluations of 5 benchmarks and new SOTA performance.

### Weaknesses
There are still some uncertain issues. (1) the experiments on only one base MLLM model, Qwen2.5-VL, are reported, where the training prompts of reasoning chain for CoT are generated by Qwen2.5-VL-72B, and then fine-tune the small-size model Qwen2.5-VL-3B and Qwen2.5-VL-7B for experiments, is teacher-student knowledge distillation on the proposed reasoning chain format able to achieve similar effectiveness? May be better to add more results on other leading frontier MLLMs such as InternVL2.5, Gemini-2.5-Pro, Llama, etc. (2) is it applicable to professional actions such as FineGym, Diving? Where expertise sub-action concepts might not well be learned for AGI models. (3) On lines 356-360, are the protocols defined for previous benchmarks? Please cited them. If novel classes Y_N are completed unknown in MLLM, how to generate the novel nouns of the new classes? So that the base VLM and MLLM have been trained on related concepts, and may be not strictly zero-shot performance. (4) As the training sub-concepts and reasoning chains are generated by VFM, VLM, and MLLM, are there hallucinated chains which lead to final correct answers on ground truth? Maybe the discussion on the Visual Grounded CoT is helpful.

### Questions
See Weaknesses.

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
The paper discusses this idea: don’t let the model guess in one shot. Make it first break the action into sub-motions (arms, torso, legs, contact), then match those to candidate actions, and finally score them — and let the model call tools (YOLO human detection, pose estimation, Qwen-based action/video explanation) when it thinks vision cues are not enough.

### Strengths
1. The paper names two real OVAR pain points, cross-modal hallucination and similar-action confusion, and every design choice (tools, sub-motions, hierarchical reward) points back to those, making a coherent motivation.
2. Treating “shoot ball” as “bend → jump → arm extend → release” matches how actions are actually separable in video; it’s more plausible than pure text-CoT on top of global video tokens.
3. Tools aren’t a fixed pipeline — the model decides whether to call pose / human / RAG /video description, and the reward penalizes useless tool calls. That’s better than many “agentic VLM” papers that just always run pose.

### Weaknesses
1. They say open-vocab, but the system leans on online RAG / Qwen API to pull category-specific definitions at inference. That narrows the search space. It’s closer to “recognition with external label dictionary + video grounding” than to “truly open” recognition.
2. YOLO 11 for human + pose, Qwen API for explanation / video description — that’s a very specific tool stack.
3. First round: “which tool(s)?” Second round: “do sub-motion reasoning.” Plus GRPO sampling 4–6 responses. That might not be cheap for real-time video, and the paper doesn’t talk about latency / streaming.

### Questions
The questions are the same as weakness.

### Soundness
3

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
4

### Summary
This paper addresses two fundamental limitations of current multimodal large language models (MLLMs) for open-vocabulary action recognition:
(1) the over-reliance on textual priors that neglect domain-specific visual cues, and
(2) the inability to distinguish semantically ambiguous actions in open-vocabulary settings.
These issues are indeed of great importance and widely exist across multiple MLLM-based video understanding tasks. The proposed Video-STAR framework attempts to mitigate these problems by constructing multimodal Chain-of-Thought (CoT) data and introducing tool-augmented reasoning with reinforcement learning optimization.

### Strengths
1.The paper is motivated by a well-defined and practically significant problem—bridging the gap between text-centric reasoning and visually grounded inference.

2.The use of a multimodal CoT to train MLLMs for structured, tool-guided reasoning is a valuable direction.

3.The framework integrates detection, pose estimation, and semantic reasoning tools with a hierarchical reward design, which is novel and effectively demonstrated across multiple benchmarks.

### Weaknesses
Major Comments

1.During data construction, the model is trained using decomposed motion sequences (sub-actions).
How does the framework behave when encountering previously unseen actions in the open-vocabulary test set?
Is there any mechanism ensuring compositional generalization beyond the motion patterns observed during training?
Without such a mechanism, the model might overfit to the seen sub-motion combinations.


2.How exactly is each action decomposed into sub-actions?
Is the decomposition manually annotated, automatically generated, or derived from an existing motion ontology?
Moreover, how do the authors ensure that the set of sub-actions can comprehensively cover unseen action categories during testing?
Since this decomposition is central to the model’s reasoning ability, more transparency on this process is necessary for reproducibility and understanding its generalization scope.

3.Video-STAR performs task-specific supervised fine-tuning (SFT) and reinforcement learning (RL), while most baselines (e.g., Qwen2.5-VL,) are evaluated without any fine-tuning.
This introduces a fairness issue: the superior performance of Video-STAR may partly result from extra supervision rather than the proposed method itself.
A fairer comparison would include a fine-tuned Qwen2.5 baseline trained on the same dataset but without tool usage and sub-motion decomposition, to isolate the true contribution of the proposed framework.

4.The ablation studies only compare the presence vs. absence of tool usage but do not analyze which tool or combination contributes most.
How is the tool selected in practice?
What is the performance when all tools are used simultaneously (pose, detection, action explanation, and video description)?
A more detailed comparison of tool selection strategies would clarify whether the proposed policy is optimal or if simpler combinations yield similar gains.

5.Since Video-STAR relies on multiple external tools, the inference pipeline likely introduces additional computational overhead.
The paper should report both the overall inference latency and the module-wise cost (e.g., tool invocation vs. model reasoning).
Comparing the efficiency of Video-STAR with standard MLLMs would help quantify the trade-off between accuracy improvement and computational expense.

6. Line 213: The symbol T_r appears for the first time without explicit definition. Please clarify its meaning and source.

### Questions
1. The paper claims to mitigate “text-centric reasoning,” but the explanation of how this is achieved is somewhat abstract.
Please explicitly describe the mechanism by which visual grounding is enforced during training and inference.

### Soundness
3

### Presentation
3

### Contribution
3

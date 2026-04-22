# DeepEyes: Incentivizing "Thinking with Images" via Reinforcement Learning

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 8, 6, 6, 6

## Abstract
Large Vision-Language Models excel at multimodal understanding but struggle to deeply integrate visual information into their predominantly text-based reasoning processes, a key challenge in mirroring human cognition. To address this, we introduce \nameshort{}, a model that learns to ``think with images'', trained end-to-end with reinforcement learning without requiring pre-collected reasoning data for cold-start supervised fine-tuning (SFT). Notably, this ability emerges natively, leveraging the model's own grounding capability as an intrinsic function rather than relying on external specialized models or APIs. We enable this capability through active perception, where the model learns to strategically ground its reasoning in visual information, guided by a tailored data selection and reward strategy. \nameshort{} achieves significant performance gains on general perception and reasoning benchmarks and also demonstrates improvement in grounding, hallucination, and mathematical reasoning tasks. Interestingly, we observe the distinct evolution of active perception from initial exploration to efficient and accurate exploitation, and diverse thinking patterns that closely mirror human visual reasoning processes. Code is available at \url{https://github.com/Visual-Agent/DeepEyes}.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper presents DeepEyes as one of the first open-source solutions that follow the thinking-with-images paradigm and uses image cropping as a tool to incentivize fine-grained visual perception capabilities in modem VLMs.

Two key contributions of this paper are: 
- Interleaved Multi-modal Chain-of-Thought (iMCoT) that reasons beyond text-only trajectories and provides seamless integration of zoom-in operations and textual reasoning. 
- Conditional tool reward that promotes the usage of zoom-in actions. 

While those key contributions are common practice in agentic literature, I think it is not a trivial attempt to make them work in the thinking-with-images paradigm. Considering the comprehensiveness of ablation studies in this paper, I lean to recommend acceptance pending the discussion with fellow reviewers/ACs.

### Strengths
1. DeepEyes comes with clear motivation – using RLVR-style post-training methods to activate the precise grounding capabilities of VLMs.
2. The proposed iMCoT and conditional tool reward prove to be effective in ablation studies in various high-res & visual reasoning benchmarks. 
3. The staged evolution (exploration -> high-frequency engagement -> efficient utilization) is compelling and supported with metrics like tool-call counts, response length, and grounding IoU curves.

### Weaknesses
1. Both iMCoT and conditional tool incentives are now fairly standard in agentic RL literature (interleaving actions + CoT, rewarding tool use). The novelty is mainly in making them work cleanly for “thinking with images”. 
2. While the authors claim to put DeepEyes in an agentic framework, the available tools are relatively limited – seems to be zoom-ins only. See the following Q1.

### Questions
1. Can you anticipate the generalization beyond image crop? Say various image manipulation tools in related work, such as scale, contrast, denoise, or keypoint heatmaps.
2. Can you quantify the benefit of (i) difficulty curation, (ii) verification, and (iii) perception-utility filtering in your data curation pipeline?

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
The paper introduces DeepEyes, a vision-language model trained end-to-end with reinforcement learning to “think with images” by interleaving textual chain-of-thought with active visual perception. The model autonomously decides when to zoom into image regions by generating bounding boxes; the resulting crops are fed back as observation tokens and used for subsequent reasoning. Training uses agentic reinforcement learning with GRPO, sparse outcome rewards, and a conditional bonus that incentivizes correct answers achieved via active perception. A data curation pipeline selects samples likely to benefit from grounding-assisted reasoning. DeepEyes achieves strong gains on fine-grained, high-resolution benchmarks, improves general perception and reasoning, reduces hallucination and slightly improves grounding, and yields consistent math reasoning gains. The paper analyzes training dynamics, showing a three-stage evolution from exploration to efficient use of active perception, validates the conditional tool reward, demonstrates scalability to larger models, positive data scaling effects, and zero-shot tool extension.

### Strengths
- Originality:
  - Proposes an interleaved multimodal CoT (iMCoT) that natively integrates active perception into reasoning with end-to-end RL, avoiding pre-collected reasoning SFT and external specialized models/APIs (Abstract; Sections 1, 3.1). 
  - Conditional reward that only bonuses correct tool-using trajectories effectively incentivizes perception-aware reasoning while discouraging gratuitous tool calls (Section 3.2; Table 5; Figure 4).
  - A targeted data curation pipeline that filters for perception-utility to bootstrap active perception without SFT (Section 3.3; Appendix B; Table 10).
- Quality:
  - Comprehensive evaluations across high-resolution, grounding/hallucination, and reasoning benchmarks with strong improvements (Tables 1–4).
  - Ablations isolate key components: conditional tool reward (Table 5), iMCoT vs text-only RL (Table 9), data composition (Table 10), model scaling (Table 6), and zero-shot tool generalization (Table 8).
  - Insightful training-dynamics analysis revealing emergent stages of active perception and behavioral patterns (Figure 3; Section 4.3), plus qualitative case studies (Figures 7–10).
- Clarity:
  - Clear agentic RL formulation with observation tokens (Section 3.2), system/tool interface (Appendix A), and pipeline overview (Figure 2).
  - Training details provided (Section 4.1) and code link for reproducibility.
- Significance:
  - Large, consistent gains on fine-grained high-res understanding (Table 1) and reductions in hallucination (Table 3), suggesting a practical path to multimodal test-time scaling via native visual thinking.
  - Demonstrates scalability to larger models with stronger grounding IoU and longer reasoning chains (Table 6), and extensibility to new tools at inference (Table 8).

### Weaknesses
- Technical detail gaps:
  - Reward magnitudes/normalization, exact coefficients for accuracy/format/tool rewards, and sensitivity analyses are not fully specified (Section 3.2; Table 5 references but no hyperparameters), limiting reproducibility and insight into stability.
  - Limited description of how image crops are tokenized/encoded, how many visual tokens per crop, and how observation tokens are interleaved with text for different tools (Section 3.2 mentions loss masks but not encoder specifics).
- Evaluation scope and fairness:
  - High-resolution gains are compelling, but comparisons to workflow baselines may be confounded by different training data and compute; stronger apples-to-apples controls are desirable (Table 1). Statistical significance and variance are not reported.
  - Grounding improvements on refCOCO family are modest (+0.6–1.0; Table 3); ReasonSeg gains are small. It would help to quantify how often crops improve IoU versus distract.
- Generality of “thinking with images”:
  - The system currently centers on cropping; while rotate shows zero-shot extensibility (Table 8), the toolset remains narrow. Some failure cases highlight grounding drift and reasoning limits (Figures 11–12), suggesting brittleness for complex visual workflows.
- Compute and efficiency:
  - Training uses 80 iterations on H100 with 256 prompts × 16 rollouts and max 6 perceptions, KL=0 (Section 4.1). Sample efficiency and cost-benefit vs SFT or hybrid methods are not deeply analyzed; KL=0 may risk policy drift/collapse.
- Claims around “no cold-start SFT”:
  - Although no reasoning SFT is used, the base model (Qwen2.5-VL) is already instruction-tuned; clarifying the exact initialization and any prompt engineering dependencies (Appendix A) would temper the claim.

### Questions
- Reward design and stability:
  - What are the exact numerical weights for R_acc, R_format, and the conditional R_tool? How sensitive are results to these weights and to the correctness indicator threshold (Eq. 2)? Please provide a sensitivity study or at least ranges (Section 3.2; Table 5).
  - Why was KL set to 0? Did you observe policy collapse or reward hacking at any point? Would a small KL or reference model improve stability/generalization (Section 4.1)?
- Observation/crop handling:
  - How are cropped images preprocessed and encoded (e.g., AnyRes vs fixed resolution), and how many visual tokens do they add? Is there an explicit cap on total image tokens across multiple crops? Any latency analysis (Section 3.1)?
  - Do you condition bounding-box generation on a learned coordinate head or pure text token decoding? How is the bbox format enforced and validated during rollouts (Appendix A)?
- Data and splits:
  - For V* and HR-Bench, please detail train/val/test splits used for RL vs evaluation to rule out leakage (Sections 3.3, 4.1). Are any evaluation images seen during RL?
  - Can you release the curated indices and the perception-utility labels (Section 3.3; Table 10)?
- Comparisons and metrics:
  - For Table 1, can you report confidence intervals or variance over multiple seeds? Similarly, could you add a compute-normalized comparison against Pixel-Reasoner (Su et al., 2025) and ZoomEye (Shen et al., 2024a)?
  - For grounding, can you report IoU distributions and success vs failure breakdown for cases where iMCoT invoked cropping vs not, to substantiate causal benefits (Section 4.2; Table 3)?
- Generalization and tools:
  - Beyond rotate, have you tried text-conditioned detection/segmentation or simple measuring tools (e.g., rulers/lines)? Any no-retrain extensions besides rotation (Table 8)?
  - What are the failure modes that lead to grounding drift (Figure 11)? Could a recurrent state or memory help prevent drift across sequential crops?

### Soundness
3

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
5

### Summary
This paper proposes a new CoT pipeline by integrating both the text and visual cues into the intermediate reasoning steps. Specifically, without SFT training, this paper firstly curates a post-training dataset and then leverages the RL-like GRPO algorithm to incentivize the model's internal grounding abilities to crop and embed the local visual information into the next reasoning trajectory. Trained by several reward functions, this paper then achieves powerful performances on general perception tasks and also the reasoning benchmarks, which imitates our human beings' zoom-in ability to investigate the mostly related visual regions and does not solely rely on once visual embedding sequencers and text-based reasoning trajectories.

### Strengths
1. Combing both the textual and visual cues into the MLLM's intermediate reasoning trajectories sounds like an interesting and critical exploration, despite the openai-o3 has applied the very similar methods to empower the MLLM thinking with images abilities. 

2. The fully released pipeline including codes and datas contributes the community which can be a good point for the community to develop the O3 frameworks.

3. The overall pipeline is not that complicated and somehow simple but effective, in which the authors also conduct various experiments to explore and exploit the overall RL training dynamics.

### Weaknesses
1. Regarding the iterative MCoT steps, does this paper explore multiple object grounding at the same time?

2. If the model needs more than twice MCoT, then the model needs more than twice visual embedding extraction, which sounds like a computational overhead. Can the authors also provide more inference latency analysis?

3. What if the model predicts all possible grounding objects' bounding boxes and extracts all visual embeddings, then concatenates all these new local visual tokens together, saying all in once?

4. What about the hard image-text samples, but with a small resolution? Does the author study these low-resolution scenarios?

5. I also notice that the implemented codes adopted a stronger MLLM models using vllm to serve as the judger, but the paper claims internal grounding post-training, which raises confusions.

### Questions
I am also curious what if using these thinking with images idea into video tasks or 3D scenarios, by adopting and improving the similar pipeline into other more difficult tasks?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces DeepEyes, a vision–language model (VLM) endowed with active perception capabilities through the integration of an image zoom-in tool. DeepEyes exhibits substantial performance gains over existing open-source and workflow-based baselines across a range of high-resolution visual tasks and reasoning benchmarks. Furthermore, the paper provides an in-depth analysis of the model’s training dynamics and emergent reasoning behaviors, uncovering patterns reminiscent of human visual cognition.

### Strengths
1. The paper is technically solid. The results (Tables 1–9) demonstrate consistent and notable improvements over relevant baselines in high-resolution perception (e.g., HR-Bench-8K), general reasoning, visual grounding, hallucination mitigation, and multimodal mathematical reasoning. The empirical rigor is further supported by comprehensive ablation studies and scaling experiments.
2. The paper conducts an insightful analysis of the training dynamics and emergent reasoning behaviors of **DeepEyes**, offering readers a deeper understanding of how active perception shapes visual reasoning.

### Weaknesses
1. Within the broader context of Agentic Reinforcement Learning (RL), this work can be regarded primarily as a **multimodal agents** equipped with a zoom-in tool, contributing limited methodological novelty.
2. The current implementation falls short of fully realizing the concept of **“Thinking with Images.”** This paradigm should encompass not only zoom-in operations but also more diverse capabilities—such as image editing, spatial manipulation, or compositional visual reasoning.
3. **Limited tool generalization.** Section 4.4 briefly asserts that the method generalizes to new tools; however, only a trivial extension to rotation is quantitatively validated. No systematic exploration of tool compositionality or multi-tool chaining—both central to agentic multimodal systems—is presented.

### Questions
1.Can the authors clarify how the model parameterizes and samples coordinates for zoom-in actions? Additionally, is any mechanism employed to prevent the repeated selection of non-informative or overlapping regions during active perception?

### Soundness
4

### Presentation
3

### Contribution
2

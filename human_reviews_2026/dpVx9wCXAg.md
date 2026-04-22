# Beyond Textual CoT: Interleaved Text-Image Chains with Deep Confidence Reasoning for Image Editing

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 6, 4, 4

## Abstract
Image editing with natural language has gained significant popularity, yet existing methods struggle with intricate object intersections and fine-grained spatial relationships due to the lack of an explicit reasoning process. While Chain-of-Thought (CoT) has been explored to enhance reasoning, purely textual CoT or CoT augmented with coordinate information is fundamentally limited in its ability to represent intricate visual layouts and lacks the necessary visual cues to guide the generation of fine-grained, pixel-level details. To address these challenges, we propose $\textbf{Mu}$ltimodal $\textbf{R}$easoning $\textbf{E}$dit ($\textbf{MURE}$), a novel framework that $\textit{shifts the visual editing process from purely text-based reasoning to a series of interleaved textual and visual rationales}$. Our framework performs image editing using a natively multimodal, interleaved text-image CoT. This approach generates a step-by-step chain of reasoning where a textual description is followed by a corresponding visual cue, such as a positional mask that defined intended edited regions or a representation of new content. Furthermore, to mitigate the hallucination phenomenon of large language models, we introduce $\textbf{M}$ulti$\textbf{m}$odal $\textbf{D}$eep $\textbf{C}$onfidence ($\textbf{MMDC}$) reasoning paradigm. This paradigm explores a tree of visual reasoning paths at each step. By pruning low-quality branches using a deep confidence score from a reward model, it ensures the model consistently follows a high-quality trajectory towards the final edited result. The proposed method decomposes complex editing tasks into interdependent sub-tasks, achieving greater precision at each stage and yielding high-fidelity edited results. We define the formulation for interleaved text-image chains and release the first CoT-Edit-14K dataset, comprising 14K high-quality editing examples. Extensive experiments show that our method yields significant improvements across three image editing benchmarks, establishing a more effective reasoning framework for visual editing.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes an interleaved CoT formation with confidence-based pruning, along with a new dataset CoT-Edit-14K to fine-tune an existing MLLM to enhance its image editing capability.

### Strengths
1. The proposed CoT-Edit-14K can be useful to fine-tune existing MLLMs for interleaved multi-modal CoT for the image editing task.

2. The proposed interleaved CoT and MMDC achieves state-of-the-art performance across MagicBrush, Emu, and SmartEdit benchmarks.

### Weaknesses
1. Lack of failure case analysis. For example, in Fig. 14, the first row doesn't show a **backwards** baseball cap; 4th row, the style of the helicopter doesn't match the original image (ICEdit's result seems better).

2. Lack of training/inference efficiency information. The proposed MMDC requires generating multiple candidates (5 as reported in the paper) and feeding them to VLM to generate reward scores, which may bring a drastic slowdown to the generation process.

3. Unclear reliability of the reward model based on Qwen-2.5 VL-7B. In the sample shown in Fig. 17, the rightmost mask generation adheres to the prompt best (as the editing prompt is to remove the green plant); however, the reward model selects the inferior leftmost path. Also, it is unclear if the proposed inference-time reward-based pruning is better than reward fine-tuning methods such as DPO and GRPO.

### Questions
1. According to the Appendix, the training was conducted with a batch size of 1, which is uncommon. Is it the effective batch size or batch size per GPU? Other unclear points: Is it full-parameter fine-tuning or LoRA? How many and what kind of GPUs were used?

2. What is the inference overhead of introducing MMDC? In Figure 4, the line's slope appears to decrease, suggesting a diminishing improvement in scores as the token count increases.

3. The authors used a 0-100 scoring scheme for the reward model, which, according to the qualitative samples shown in Fig. 17, is not reliable enough. How about using a coarser scheme, such as 1-5, or even pairwise comparison?

4. Instead of introducing inference-time MMDC, which introduces extra inference overheads, how about directly fine-tuning the interleaved CoT with DPO / GRPO?


[1] Wang, Yibin, et al. "Unified reward model for multimodal understanding and generation." arXiv preprint arXiv:2503.05236 (2025).

### Soundness
2

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
The paper proposes MURE, a unified editing model that replaces purely textual chain-of-thought with an interleaved text–image reasoning chain. Each step alternates between textual reasoning and visual generations (e.g., a mask for the edit region or an image of new content). The authors also add an inference-time selection method, Multimodal Deep Confidence (MMDC), which samples multiple candidates at each visual step and greedily keeps the branch with the highest score from a reward VLM (Qwen2.5-VL). Training uses cross-entropy for text tokens and a rectified-flow MSE for image latents. The authors collected a new dataset, CoT‑Edit‑14K, to support interleaved editing chains.

### Strengths
- [S1] The interleaved text–image paradigm is an intuitive way to ground “where/what to edit,” with explicit masks/new‑object images that are easy to inspect and debug; the pipeline diagram and walkthrough make the idea clear.
- [S2] Consistent benchmark results with sensible ablations: improvements on MagicBrush/Emu and SmartEdit, plus ablations that separate interleaved CoT and MMDC contributions; search‑width increases yield reasonable gains.
- [S3] CoT‑Edit‑14K could be a useful resource for step‑wise editing with interleaved chains across 10 edit types; construction details and distributions are provided.

### Weaknesses
- [W1] Evaluation is insufficient for the paper’s core claims. There is no human study and no targeted measures of (i) mask correctness (e.g., IoU/precision/recall) and (ii) physical consistency (e.g., shadows/reflections/occlusions). Given the stated motivation, these are significant omissions.
- [W2] Technical novelty is moderate relative to visual‑reasoning work that already makes the generation/editing process explicit via reasoning steps. GoT [1] formulates textual reasoning (with semantic–spatial guidance) for both generation and editing, and GoT‑R1 [2] extends it with RL; ImageGen‑CoT [3] and T2I‑R1 [4] add textual CoT and scale‑up/RL for text‑to‑image; MM‑R1 [5] introduces cross‑modal CoT for personalized synthesis. MURE’s distinct aspect is producing editing‑specific visual artifacts inside the chain plus a greedy step scorer which is useful engineering, but not an algorithmic novelty.
- [W3] Related‑work positioning should be clearer. The paper should explicitly contrast what MURE can do (and cannot do) relative to GoT/GoT‑R1, ImageGen‑CoT/T2I‑R1, and MM‑R1, as well as earlier multimodal/visual‑only reasoning (Multimodal‑CoT [6]; CCoT [7]; Visual Planning [8]).
- [W4] Mixed metrics are not discussed. Emu shows a slight CLIP‑Out drop versus Bagel, and SmartEdit “reasoning” has a small LPIPS regression; these deserve analysis of trade‑offs (Tables 1–2, p.7).
- [W5] Failure cases are not demonstrated. Showing failure cases can help analyze what is missing and motivate future research.


[1] Fang et al., “GoT: Unleashing Reasoning Capability of Multimodal Large Language Model for Visual Generation and Editing,” arXiv, 2025.  
[2] Duan et al., “GoT‑R1: Unleashing Reasoning Capability of MLLM for Visual Generation with Reinforcement Learning,” arXiv, 2025.  
[3] Liao et al., “ImageGen‑CoT: Enhancing Text‑to‑Image In‑Context Learning with Chain‑of‑Thought Reasoning,” arXiv, 2025.  
[4] Jiang et al., “T2I‑R1: Reinforcing Image Generation with Collaborative Semantic‑level and Token‑level CoT,” arXiv, 2025.  
[5] Liang et al., “MM‑R1: Unleashing the Power of Unified Multimodal Large Language Models for Personalized Image Generation,” arXiv, 2025.  
[6] Zhang et al., “Multimodal Chain‑of‑Thought Reasoning in Language Models,” arXiv, 2023.  
[7] Mitra et al., “Compositional Chain‑of‑Thought Prompting for Large Multimodal Models,” CVPR, 2024.  
[8] Xu et al., “Visual Planning: Let’s Think Only with Images,” arXiv, 2025.

### Questions
1) What is the latency/compute overhead of MMDC as search width increases? Any sensitivity to the reward prompt/model, and have you tried non‑greedy (e.g., beam/global) selection?  
2) For edit types that currently omit visual steps, does forcing masks or object images help or hurt? An ablation (mask‑only / object‑only / both) would clarify.

### Soundness
3

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
This paper introduces MURE, a novel framework for image editing that shifts from purely text-based Chain-of-Thought (CoT) reasoning to interleaved text-image CoT sequences. The key innovation is decomposing complex editing tasks into a series of sub-tasks, where each textual reasoning step is paired with corresponding visual outputs such as segmentation masks or content representations. The paper also introduces a CoT-Edit dataset with significant improvements on three benchmarks.

### Strengths
- The idea of interleaved CoT for image editing, along with the MMDC reasoning paradigm, is intuitive and interesting.
- The proposed approach is effectively demonstrated through experiments.
- The presentation is clear and easy to follow.

### Weaknesses
- The latency of the approach could be a significant concern, since the model is required to generate intermediate images. In other words, the number of output tokens per sample (in Figure 4) would be much higher than in non-CoT or textual CoT frameworks. I don’t think the approach or paradigm is scalable, especially when we extend it to longer search trajectories.
- The proposed approach is limited to object-oriented image editing, as it heavily relies on extracting objects in the approach and the dataset construction process. It could not be applied to other types of editing, e.g., changing the background, or changing other visual features, such as color or shape.
- From Table 5 which compares textual CoT and interleaved CoT, the improvement in the interleaved CoT is quite marginal.

### Questions
- Is it possible to extend the framework to more types of editing, such as changing the background?

### Soundness
2

### Presentation
3

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
In this paper, the authors proposed a new approach that enahnces the purely textual reasoning to interleaved text–image reasoning chains that alternate between textual reasoning steps and visual cues (e.g., masks, synthesized intermediate content)

Additionally, it presents Multimodal Deep Confidence (MMDC) — a reward-model–driven pruning mechanism that evaluates multiple visual reasoning paths, selects high-confidence branches, and mitigates hallucinations.

The authors also construct CoT-Edit-14K, a dataset with 14 K high-quality interleaved text–image CoT examples, covering 10 editing subtasks (e.g., object replacement, removal, color change).

### Strengths
1. The high-quality dataset (CoT-Edit-14K) would be a contribution to the image editing community.

2. The evaluation and solid performance are comprehensive. Demonstrates consistent gains across CoT-Edit-14K, MagicBrush and Emu benchmarks on multiple metrics (CLIP, DINO, PSNR, SSIM, LPIPS). Ablations isolate contributions of interleaved CoT and MMDC convincingly.

### Weaknesses
1. The MURE framework works very well for object swapping or adding-- it did text reasoning, mask predition and new object generation. However, there might be some concerns when conducting other editing types, especialy for eidting exisiting objects. For example, changing the object location, size or shape. Such method might not be able to keep the identity consistent when generation the new object.

### Questions
It would be great for the reviewer to understand the strength or weakness when the authors report the performance of MURE in different editing types.

### Soundness
3

### Presentation
3

### Contribution
2

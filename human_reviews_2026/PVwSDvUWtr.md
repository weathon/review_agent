# GOLD: Global Overview to Local Detail in Efficient Visual Grounding for GUI Agents

- Avg Score: 4.67
- Decision: Reject
- Scores: 6, 4, 4

## Abstract
Graphical User Interface (GUI) agents powered by Vision-Language Models (VLMs) have recently emerged as a promising direction for multimodal automation. However, VLM-based GUI grounding incurs substantial computational overhead, making deployment on edge devices infeasible and leading to prohibitively high cloud serving costs. Prior attempts to reduce background or history vision tokens partially alleviate this issue, but either rely on sparsity in foreground elements or require extensive fine-tuning. In this work, we present GOLD, Global Overview to Local Detail for efficient GUI grounding that is tuning-free and robust across varying interface densities. GOLD operates in three stages. At the Global Pruning Stage, we downsample GUI screenshots and feed them into the VLM to identify relevant regions, thereby achieving efficient context reduction. In the Local Refinement Stage, only crops of detected regions are passed to the VLM at high resolution. To retain broader contexts, we aggregate the outputs of both stages to integrate both global and local information in Global-Local Context Fusion Stage. Experimental results show that GOLD reduces TFLOPs by 78%, while even improving accuracy by 0.7%p when it is integrated into the state-of-the-art GUI grounding method on the ScreenSpot-V2 benchmark. These findings highlight the efficiency of our global-to-local grounding framework.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents **GOLD (Global Overview to Local Detail)**, a training-free framework for efficient visual grounding in GUI agents. GOLD operates in three stages: (1) a global pruning stage that downsamples the input screenshot and uses attention maps to identify candidate regions, (2) a local refinement stage that processes these regions at original resolution to recover fine-grained details, and (3) a global-local fusion stage that integrates both coarse and fine information to produce the final prediction. Evaluated on ScreenSpot-V2 and Multimodal-Mind2Web benchmarks, GOLD achieves a **78% reduction in TFLOPs** while improving grounding accuracy by 0.7%p when integrated with GUI-Actor. The method is plug-and-play, requiring no additional training, and demonstrates consistent improvements across multiple backbones and GUI environments (mobile, web, desktop).

### Strengths
**1. Practical and Training-Free**  

The method requires no additional fine-tuning and can be seamlessly integrated into existing VLM-based GUI agents, making it highly practical for deployment in real-world systems.  

**2. Clear and Lightweight Design**  

The three-stage framework (global pruning, local refinement, global-local fusion) is simple, intuitive, and lightweight. The fusion stage adds virtually no computational overhead since it only performs score lookups on precomputed coordinates.  

**3. Strong Empirical Results**  

GOLD achieves up to 78% TFLOP reduction while slightly improving accuracy, even setting new state-of-the-art results on ScreenSpot-V2 when combined with GUI-Actor, with gains consistent across different backbones (GUI-Actor-2B/3B, Qwen2.5-VL) and platforms (Mobile, Web, Desktop). The paper also provides a comprehensive set of ablation studies to support the design choices.  

**4. Great Presentation** 

The paper is well-written with clear figures and organization, making the methodology easy to follow and supporting reproducibility and interpretability.

### Weaknesses
**1. Baseline Selection**

Most comparisons are made against other GUI grounding models (OS-Atlas, UGround, etc.) rather than efficiency-oriented methods. Since GOLD is positioned as an efficiency framework, it would be more informative to demonstrate its generalizability by applying it to a wider set of models reported in the paper and analyzing the resulting performance–efficiency trade-offs.

**2. Limited Comparison with Training-Free Acceleration for VLMs**

The evaluation only includes FastV as a training-free baseline, without clarifying whether its hyperparameters were carefully tuned for GUI grounding tasks. A broader comparison would strengthen the paper, particularly with recent token pruning approaches on VLMs (e.g., SparseVLM for adaptive sparsification) as well as grounding-specific optimizations such as GAP [arXiv:2506.21873], which addresses position ID misalignment after pruning, and FEATHER [arXiv:2412.13180], which proposes ensemble criteria for localization tasks. Including such baselines would help position GOLD more clearly within the growing space of efficient grounding methods.

**3. Novelty Concerns**

The core mechanism of attention-guided region selection followed by refinement bears some resemblance to the attention-based candidate selection already used in GUI-Actor. The paper would benefit from an explicit discussion of how GOLD differs. 

More generally, the methodology follows standard coarse-to-fine grounding paradigms, which raises questions about (i) what is fundamentally new beyond hierarchical region selection and (ii) whether the framework could or should be validated on broader grounding tasks beyond GUIs.

**4. Practical Efficiency Metrics**

While reductions in TFLOPs are clearly reported, the multi-stage pipeline introduces additional forward passes. The paper does not provide latency or end-to-end runtime measurements (e.g., similar to UI-Agile’s reporting of wall-clock gains). Without such measurements on representative hardware, it is difficult to assess whether the theoretical FLOP savings translate into practical speedups for real-time GUI agents.

### Questions
1. Since GUI-Actor already employs attention-based action heads for grounding, could you explicitly clarify how GOLD's attention-driven region selection differs from GUI-Actor's existing mechanism? What specific advantages does the global-to-local pipeline provide beyond what GUI-Actor already implements?

2. The evaluation only compares with FastV as a training-free baseline. For FastV, were hyperparameters (pruning ratio, layer selection) carefully tuned for GUI grounding tasks? Could the authors include broader comparisons to other pruning/acceleration methods such as SparseVLM, GAP, or FEATHER?

3. While TFLOP reductions are reported, could the authors provide end-to-end latency/runtime measurements on representative hardware to show whether the theoretical savings translate into practical speedups for real-world GUI agents?

4. The methodology follows a general coarse-to-fine paradigm without apparent GUI-specific components. Could the authors clarify: (1) whether GOLD can be applied to standard visual grounding benchmarks like RefCOCO/RefCOCO+, (2) which design choices, if any, specifically exploit GUI properties such as structured layouts or small elements, and (3) whether evaluation on broader grounding tasks would strengthen the contribution, or if GOLD is intentionally tailored to GUIs?

5. Table 1, 2, 3 shows GOLD applied primarily to GUI-Actor and Qwen2.5-VL. Could you report results when applying GOLD to other models in the table (e.g., OS-Atlas, ZonUI, UGround)? This would better demonstrate the framework's generalizability as a plug-and-play efficiency enhancement.

### Soundness
3

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
2

### Summary
This paper proposes GOLD, a "global overview - local refinement" three-step method, which is used to significantly reduce the computational cost of VLM in GUI grounding without the need for fine-tuning. By first screening candidate regions at low resolution, then cropping the regions of the original image for high-resolution re-encoding, and finally fusing the global-local attention scores, a 78% TFLOPs reduction is achieved, while the accuracy is improved by 0.7% on benchmarks such as ScreenSpot-V2. The method is simple, training-independent, and suitable for deployment on mobile devices and cloud sides.

### Strengths
1.Training is irrelevant, plug-and-play, friendly to existing VLMs

2.The three-step pipeline has a clear structure, balancing both global context and local details

3.It achieves significant acceleration and improvement in accuracy simultaneously on multiple public benchmarks, with thorough experiments

4.It remains robust in foreground-intensive scenarios, outperforming previous methods such as background pruning or token dropping

### Weaknesses
1. Currently, τ ∈ (0, 1] is a fixed relative threshold, and C = 3 is an empirical upper limit. The optimal values under different GUI densities, resolutions, or screen sizes have not been explored. Figure 2 shows that when s = 0.3, the desktop accuracy has decreased by 4%, indicating that the attention peaks at low resolutions are easily "overwhelmed" by large-sized controls, and the 8-neighborhood clustering easily merges controls with small spacings into the same area. The paper does not provide adaptive estimation of τ (such as Otsu, Pareto front) or online feedback strategies, resulting in the need for manual adjustment of hyperparameters, thereby weakening the "no-training" advantage.

2.When the instructions involve "selecting three checkboxes simultaneously" or "clicking the fifth item in the menu", the model needs to focus on ≥3 areas. GOLD first performs hard truncation and then fusion. Once the target control element does not enter the Top-C, it cannot be restored. The experiment only reports the success rate of single-click, does not calculate the Recall@C curve, and does not provide an ablation study for C > 3. It is difficult to prove that C = 3 is sufficient for complex tasks.

3.The author assumes that "the low-resolution attention peak is sufficient to indicate the high-resolution target location", but when image compression causes text blurring and icons to have jagged edges, VLM attention often disperses to adjacent text or blank areas. Figure 4 is an example that fails due to blurring, but the paper only qualitatively presents this, without quantifying the failure rate, nor providing secondary verification (such as edge gradients, OCR boxes) to correct the candidate regions. Once the attention map is unreliable, the three-step pipeline of GOLD will go completely wrong, lacking rollback or self-check mechanisms.

4.To reduce the number of forward passes in VLM, the authors combined C high-resolution cropped images into multiple images for input at once. Although this approach is efficient, the self-attention mechanism of VLM only computes within a single image, and there is no interaction between regions. For instructions like "Click the 'Save' button in Panel A instead of the 'Save' button in Panel B", which require comparing two similar controls, the model cannot utilize cross-region attention for disambiguation. It can only rely on subsequent score fusion, but the fusion stage lacks an inter-region comparison mechanism.

### Questions
See Weaknesses

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper aims to address the computationally burdensome nature of GUI grounding in high-resolution scenarios. It proposes GOLD, a tuning-free approach that enhances grounding performance by combining global pruning with local refinement and coupling their predictions. Experiments demonstrate that the method substantially reduces TFLOPs while achieving improved performance.

### Strengths
1. The overall presentation is clear.

2. The final experimental evaluation is fairly comprehensive.

### Weaknesses
1. The motivation of the method is somewhat unclear. For example, Section 3.3 argues that local refinement recovers details that may be lost at low resolution. However, in Figure 1, the local refinement appears to be inaccurate, and the final prediction relies on fusion via α. From Figure 3, it also seems that performance is still primarily driven by the global view. Therefore, the motivation for local refinement feels weak.

2. The method’s novelty appears limited. For instance, Line 196 notes that the proposed approach borrows ideas from prior work.

### Questions
1. In Section 3.2, which layer’s attention map is used? How would using attention maps from different layers affect the results?

### Soundness
2

### Presentation
3

### Contribution
2

# GUI-KV: Efficient GUI Agents via KV Cache with Spatio-Temporal Awareness

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 6, 4, 2

## Abstract
Graphical user interface (GUI) agents face severe efficiency bottlenecks when processing long sequences of high-resolution screenshots, making inference costly and memory-bound. Existing KV cache compression methods, designed for natural images, remain suboptimal as they fail to exploit the unique spatial and temporal redundancies of GUIs. In this work, we first demonstrate that unlike natural images, GUI attention sparsity is uniformly high (>0.99) across all transformer layers, invalidating complex layer-varying budget strategies. Building on this insight, we introduce GUI-KV, a training-free compression method that allocates a uniform budget driven by two novel mechanisms: (1) spatial saliency guidance, which augments attention with residual stream L2 norms to preserve semantic visual tokens; and (2) temporal redundancy scoring, which employs subspace projection to identify and prune historical frames that are linearly redundant with the current view. Across six benchmarks, GUI-KV outperforms competitive baselines, often recovering near-full-cache accuracy at 10-20% budgets. Notably, on AgentNetBench, it reduces decoding FLOPs by 38.9% while increasing step accuracy by 4.1% over the full-cache baseline.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors propose GUI-KV, a smart and efficient KV-cache compression framework for GUI agents that deal with tons of screenshots and long tasks. Instead of retraining the models, GUI-KV just plugs into existing agents and makes them faster and lighter by pruning redundant cache states. The key idea is that GUI screenshots are full of repetitive visuals, so the paper introduces a spatio-temporal theorem-guided method: (1) spatial saliency guidance, which scores tokens based on their L2-norm (how much “information energy” they carry) and attention weights; and (2) temporal redundancy scoring, which uses QR decomposition to detect overlapping key subspaces between frames and drop repetitive tokens. This simple yet well-reasoned design leads to large speedups and even slight accuracy gains over full-cache models.

### Strengths
The paper  builds an argument from the observation (Fig. 1, lines 108–215 and 162–215) that GUI attention sparsity is uniformly above 0.99 across all layers. This leads to a clean, theory-backed conclusion that uniform budget allocation is optimal for GUI workloads, overturning prior assumptions from methods like PyramidKV and VL-Cache that rely on varying per-layer budgets.

he proposed saliency + redundancy scoring (lines 270–323) ties practical algorithmic components to theoretical motivation. It’s easy to implement, works plug-and-play without retraining.

### Weaknesses
The core theorem and sparsity analysis (lines 162–215) hinge on an assumption of uniform high attention sparsity (> 0.99). This is empirically true for GUI screenshots but might not hold for other multimodal agents (e.g., video or natural-scene reasoning). The paper doesn’t clarify how GUI-KV behaves when this uniformity breaks down.

The temporal redundancy mechanism (lines 270–323) assumes low-rank subspaces (r ≪ dₕ), but the computational overhead of repeated QR per frame could become significant in real-time agents or larger contexts. While the authors mention negligible prefill cost (lines 432–485), no timing breakdown is provided for larger r or more screenshots.

Although the authors reason that uniform budget allocation is “optimal,” there is no formal proof or convergence guarantee. The argument is empirical rather than mathematical, so the claimed “theorem-like” property is more observational than theoretically rigorous.

Minor weakness: (1) Some parameter choices (e.g., α = 0.5, τ = 0.1 in Eq. 5) are not justified. (2) The interaction between spatial and temporal modules isn’t theoretically characterized (just shown empirically in Fig. 4).

### Questions
How sensitive is GUI-KV’s performance to the rank r used in QR decomposition—does lowering r hurt accuracy, and can adaptive r per layer improve efficiency?

Could the authors formally characterize conditions (beyond uniform sparsity) under which the uniform-budget theorem holds, and how it might adapt to multimodal domains with variable sparsity (e.g., GUI + video)?

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
3

### Summary
This paper addresses the computational inefficiency of GUI agents when processing long sequences of high-resolution screenshots by proposing GUI-KV, a plug-and-play, training-free key-value cache compression method tailored for GUI environments. The authors first conduct a thorough empirical analysis and reveal a key insight: unlike natural images, GUI screenshots exhibit uniformly high attention sparsity across all transformer layers. This finding invalidates existing layer-wise adaptive budget allocation strategies, which rely on inter-layer sparsity variation and thus misallocate cache in GUI contexts. Consequently, the authors advocate for a simple yet effective uniform budget allocation across layers.
Building on this, GUI-KV introduces two novel components: (1) spatial saliency guidance, which augments attention scores with the L2 norm of visual token hidden states to better preserve semantically important regions; and (2) temporal redundancy scoring, which uses QR decomposition to project historical keys onto the current frame’s key subspace and preferentially prune tokens that are redundant over time. Extensive experiments across six established GUI agent benchmarks—including visual grounding, offline action prediction, and online evaluation—demonstrate that GUI-KV consistently outperforms strong baselines. Notably, it recovers near-full-cache accuracy with only 10–20% cache budget, and in some cases, it improves step accuracy by +4.1% while reducing decoding FLOPs by 38.9% compared to the full-cache baseline. Ablation studies confirm the complementary benefits of the spatial and temporal components.
The paper is well-motivated, technically sound, and empirically rigorous. The proposed method is simple, effective, and directly applicable to real-world deployment of efficient GUI agents. The writing is clear, and the analysis is insightful. This work makes a valuable contribution to efficient multimodal inference and is suitable for publication at ICLR.

### Strengths
（1）	The paper addresses a critical efficiency bottleneck in GUI agents, KV cache explosion when processing long sequences of high-resolution screenshots, a key barrier to real-world deployment.

（2）	The authors conduct the first systematic analysis showing that GUI screenshots exhibit uniformly extreme attention sparsity across all transformer layers, in stark contrast to natural images. This invalidates layer-wise adaptive budgeting strategies used in prior KV compression methods.

（3）	Clear presentation and strong reproducibility: The method is well-described, with pseudocode, hyperparameter details, and thorough ablations provided in the appendix, ensuring transparency and ease of replication.

### Weaknesses
（1）All experiments use pre-recorded screenshots from benchmarks like ScreenSpot and AgentNetBench. However, real-world GUIs often contain dynamic components that violate the assumption of high temporal redundancy, potentially undermining the QR-based subspace projection.

（2）The paper reports aggregate accuracy gains but provides no error case studies. For instance: in which tasks does GUI-KV underperform? Are failures caused by premature pruning of small but critical UI elements? Such analysis is essential to delineate the method’s failure modes.

（3）The paper includes insufficient visualizations, which limits the reader’s ability to fully grasp the core mechanisms and empirical findings of the proposed method. The authors are encouraged to include additional illustrative figures or case studies to improve clarity and strengthen the presentation.

### Questions
（1）The current abstract lacks conciseness and focus. We suggest that the authors revise and condense the abstract to better highlight the core contributions and technical innovations of the method.

（2）The paper fixes the subspace rank at . How was this value selected? Does the optimal rank vary significantly across tasks or screenshot resolutions? Could an adaptive rank selection strategy further improve robustness?

（3）The paper employs different evaluation metrics across datasets. Please clarify why a unified metric is not used and, more importantly, specify for each benchmark whether the chosen metric follows the standard established in prior work or is newly introduced in this paper. If any metric is newly designed, please justify its validity and appropriateness for the corresponding task.

（4）The paper currently validates GUI-KV only on two models: UI-TARS-1.5-7B and OpenCUA-7B. How can the authors demonstrate that the method is effective on other models as well, rather than being limited to these two specific architectures?

（5）I notice that when budget=100%, all methods have the same evaluation metrics. Is it reasonable in this field that different methods yield identical experimental results when applied to the same model?

（6）The paper does not include ablation studies for the hyperparameters α and τ. How were their values determined, and is there evidence that the current setting is robust across different tasks or models?

### Soundness
2

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
This paper addresses the high-cost and high-memory challenges of running VLM-based GUI agents. The core problem is that these agents must process long sequences of high-resolution screenshots, making the KV cache prohibitively large. The paper first analyzes and emphasizes the attention sparsity in GUI agents and leads to a simple uniform budget strategy. Based on this, the paper introduces GUI-KV, including Spatial Saliency Guidance and Temporal Redundancy Scoring. Experiments across six benchmarks and two VLM agents show GUI-KV  outperforms existing baselines.

### Strengths
1. The analysis of attention sparsity in GUI agents is interesting. The finding of attention distributions explains why existing methods underperform on GUI tasks. It also provides an empirical motivation for the design choice of adopting a uniform budget.
2. The two designs in the method are straightforward and intuitive. Spatial Saliency Guidance effectively alleviates attention sink issue, while Temporal Redundancy Scoring removes cross-frame redundancy.
3. The experiments are comprehensive, demonstrating GUI-KV outperforms existing KV compression methods.

### Weaknesses
1. The presentation is somehow poor. It is a pain to real and understand this paper, as it focuses too much on the details while failing to clearly present the motivation and high-level idea.
2. In Figure 4, in the 7 screenshot setting, GUI-KV matches the performance of full-cache. Then with more screenshots, e.g., 10, why does GUI-KV significantly drop? From 7 to 10 screenshots, both full-cache and SanpKV do not have obvious change. The paper claims that GUI-KV can reduce long-context distraction, which is not aligned with the experiment result.
3. The paper claims the overhead of temporal redundancy scoring is "negligible" during the prefill phase. However, it's less clear what the latency cost is during the auto-regressive decoding phase, as the scoring function (which uses attention from "observation tokens") needs to be re-evaluated.

### Questions
See weakness.

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes GUI-KV—a plug-and-play KV cache compression method for GUI agents built on vision-language models (VLMs), addressing inefficiencies in processing long high-resolution screenshot sequences.

First, it reveals a key insight: unlike natural images, GUI screenshots exhibit uniformly high attention sparsity (>0.99) across all transformer layers, making uniform KV cache budget allocation across layers more effective than complex layer-varying schemes.

Then, GUI-KV integrates two novel mechanisms:
1. Spatial saliency guidance: Augments attention scores with the L2 norm of visual token hidden states to retain semantically important tokens.
2. Temporal redundancy scoring: Uses QR decomposition on the current frame’s key matrix to create a subspace, pruning redundant tokens from previous frames via projection residual analysis.


Evaluated on 6 benchmarks with 2 GUI models (UI-TARS-1.5-7B, OpenCUA-7B), GUI-KV recovers near-full-cache accuracy with 10–20% budgets, reduces decoding FLOPs by 38.9% (5-screenshot, 40% budget) with negligible prefill overhead, and outperforms baselines (SnapKV, PyramidKV, VL-Cache).

### Strengths
1. The systematic analysis of attention patterns in GUI agent workloads and the effectiveness of KV cache budget allocation strategies—particularly the discovery of uniformly high attention sparsity across all transformer layers in GUIs—holds significant importance. It not only fills the gap in understanding GUI-specific characteristics for KV cache optimization but also provides valuable insights that contribute to the advancement of the broader GUI agent and KV cache research community.

2. The experimental results of GUI-KV are highly promising, especially its ability to recover near-full-cache accuracy with modest KV cache budgets (typically 10–20%) and even exceed full-cache performance in some scenarios. The quantitative results across six benchmarks (e.g., ScreenSpotV2, AgentNetBench) and two leading GUI agent models (UI-TARS-1.5-7B, OpenCUA-7B) effectively validate its superiority over competing baselines, while the efficiency analysis—showing substantial decoding FLOP reductions (e.g., 38.9% in the 5-screenshot, 40% budget setting) and negligible prefill overhead—further demonstrates its practical value.

### Weaknesses
1. The core technical contribution is GUI-KV’s design. While the systematic analysis of GUI attention patterns (e.g., uniformly high sparsity across layers) is valuable for guiding GUI-KV’s uniform budget strategy, it may be better framed as a foundational insight rather than a standalone main contribution.

2. Regarding GUI-KV itself, its main innovation centers on integrating spatial saliency guidance and temporal redundancy scoring to form a complete compression pipeline. However, the novelty of this integration and its individual components could be further emphasized, as the current framing may contribute seem relatively limited. For instance, while the temporal redundancy scoring mechanism retains non-redundant tokens from earlier frames by projecting their keys onto the current frame’s key subspace, the description does not fully highlight how this approach differs from existing methods that leverage historical frame information.

3. The paper is highly readable, but its structure has room for improvement. Typically, the "Related Work" section (Section 5) should not be placed near the end of the paper. Related work serves to contextualize the GUI problem setting, highlight gaps in existing methods (e.g., suboptimal KV cache compression for GUIs), and justify the design of GUI-KV—all of which would better inform readers before presenting detailed analysis (Section 2) of current GUI agent challenges and the proposed methodology. If the related work has limited relevance to the core design or analysis, it could be relocated to supplementary materials; otherwise, positioning a comprehensive literature review prior to Section 2 would make the logical flow more coherent. 

4. The analysis section (Section 2) of the paper is relatively lengthy, which has constrained the space available for detailed descriptions of the proposed GUI-KV model. Notably, the methodology section (Section 3) dedicated to explaining GUI-KV’s design—including its spatial saliency guidance, temporal redundancy scoring, and their integration—spans only approximately one page. This imbalance gives the impression that the analysis part is emphasized more heavily than the methodology, which may raise concerns about whether the novelty and complexity of the proposed GUI-KV model are sufficiently articulated and highlighted

5. There’s no overall pipeline for your proposed GUI-KV. This pipeline is important and straightforward to let readers easily understand the main contributions and novelties of your paper. 

6. In Figure 3, the caption should not overlap the figure(check the right sub-figure).

7. In the ablation study (Figure 4), the y-axis is not explicitly labeled to clarify what it represents, leaving ambiguity about the specific performance indicator being measured for each method (e.g., GUI-KV, spatial-only guidance, temporal-only guidance, and SnapKV baseline).

Overall, the paper is well-written and readable, with a strong final performance. However, concerns remain regarding its novelty, structural organization, and lack of detailed clarifications. It is recommended that the authors revise the paper structure and expand discussions on the technical novelty and contributions of GUI-KV.

### Questions
Please reply to the 7 weakness points accordingly, as I mentioned before. 

I think the main question that authors should address is the novelty of this paper. 
Analysis can be the motivation for this paper, but the contribution should focus on "what is totally new in your proposed pipeline design".

### Soundness
2

### Presentation
2

### Contribution
2

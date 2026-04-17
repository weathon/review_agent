# APT: Towards Universal Scene Graph Generation via Plug-in Adaptive Prompt Tuning

- Decision: Accept (Poster)
- Scores: 6, 4, 4, 6

## Abstract
Scene Graph Generation (SGG) is pivotal for structured visual understanding, yet it remains hindered by a fundamental limitation: the reliance on fixed, frozen semantic representations from pre-trained language models. These semantic priors, while beneficial in other domains, are inherently misaligned with the dynamic, context-sensitive nature of visual relationships, leading to biased and suboptimal performance. In this paper, we transcend the traditional one-stage v.s. two-stage architectural debate and identify this representational bottleneck as the core issue. We introduce Adaptive Prompt Tuning (APT), a universal paradigm that converts frozen semantic features into dynamic, context-aware representations through lightweight, learnable prompts. APT acts as a plug-in module that can be seamlessly integrated into existing SGG frameworks. Extensive experiments demonstrate that APT achieves +2.7 improvement in mR@100 on PredCls, +3.6 gain in F@100 and up to +6.0 gain in mR@50 in open-vocabulary novel splits. Notably, it achieves this with less than 0.5M additonal parameters (<1.5\% overhead) and reduced 7.8\%-25\% training time, establishing a new state-of-the-art while offering a unified, efficient, and scalable solution for future SGG research. The source code of APT is available at <https://github.com/CGCL-codes/APT>.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper points out that the core bottleneck of SGG is the mismatch between pre-trained semantic priors and visual relationship dynamic contexts, rather than a one-stage/two-stage dispute. Propose an APT lightweight plugin that converts frozen features into context sensitive representations, with 0.5M parameters and a training acceleration of 7.8% -25% plug and play. With the highest score increase of+6.0 on PredCls, SOTA is refreshed with<1.5% overhead, providing SGG with a unified, efficient, and scalable new paradigm.

### Strengths
1. The paper is written clearly and organized, especially in the introduction section, where the author provides a detailed explanation of the problem of frozen features using three figures.

2. The APT model not only achieves a significant improvement in accuracy relative to SOTA, but also ensures a limited increase in parameter count and saves training time.

3. The relevant work has been thoroughly sorted out, and the source code of the paper has been provided.

### Weaknesses
1. The main objective of this work is to address the shortcomings of the one-stage and two-stage methods, but these shortcomings are not unique to SGG tasks, and the designed method can also be applied to other tasks. This will bring two sub-issues: 1) Have you tried using it on other tasks; 2) The analysis of the specificity of SGG tasks in this work needs to be further explored.

2. The ablation study is comprehensive only under the top-50 setting, while it is incomplete under the top-20/100 setting. In the experimental section, ablation research is the core experiment, and it is strongly recommended to supplement it completely. In addition, it is recommended to try removing the one-stage, two-stage, and Open Vocabulary parts of APT-enhanced methods in sequence to eliminate random factors caused by model combinations.

3. Partial modifications needed: 1) Spelling error in line 045 for "Fiugre"; 2) Some images have font sizes that are too small and do not match the main text.

4. One-stage methods lack the latest methods for 2025.

### Questions
See [Weaknesses].

### Soundness
3

### Presentation
2

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
The paper proposes APT, a plug-in module for Scene Graph Generation (SGG) that keeps language backbones frozen while learning lightweight prompts to modulate object/predicate semantic embeddings using visual context. Concretely, APT introduces (i) a Detection Prompt and Relation Prompt fused with static word vectors via small MLPs, and (ii) a Compositional Generalization Prompter (CGP) with context gating, basis‐prompt synthesis, and feature refinement to improve open-vocabulary generalization. The authors motivate APT by arguing that frozen embeddings (GloVe/BERT/CLIP-text) are rigid and misaligned with context-dependent visual relations, illustrated via t-SNE plots and discussion of the one- vs two-stage SGG divide. On VG (with appendix results on OI-V6/GQA), they report consistent gains in mR/F across several backbones and claim <0.5M extra parameters and 7.8%–25% training-time reduction per epoch.

### Strengths
1. Clear problem framing. The paper articulates a real pain point in SGG—static semantic priors—and gives intuitive illustrations (Figures 2–3).
2. The idea of drop-in prompts for both one- and two-stage SGG and OV-SGG is straightforward and potentially practical.
3. Compute friendly. Claims of small parameter overhead and faster epochs are attractive for the community.

### Weaknesses
1. The core idea (i.e., learn prompts on frozen backbones) has strong prior art: VPT (vision prompts), MaPLe (joint vision+text prompts for CLIP), DualPrompt (complementary/global-vs-expert prompts), and adapter-style PEFT (CLIP-Adapter / Tip-Adapter). APT reads like a tailored application of these ideas to SGG, but the paper does not articulate a new principle beyond that specialization. A side-by-side comparison (same budgets) against these methods is missing.
2. Motivation evidence is mostly qualitative. The "frozen semantics are rigid" claim is intuitive, but support rests on t-SNE/intuition. Provide quantitative diagnostics that isolate the bottleneck (e.g., probing tasks for role asymmetry, controlled swaps of subject/object labels, ablate text vs. vision prompts separately).
3. The information-bottleneck framing is descriptive; provide any measurable prediction (e.g., MI/CKA proxies showing APT features are "more minimal yet sufficient") that the experiments then corroborate.

### Questions
1. In Table 5, Why do some+APT variants show fewer parameters than their baselines? Provide a breakdown and add inference latency/memory/FLOPs.

### Soundness
2

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
The proposed method of using learnable text embeddings for SGG is a good direction. The results on benchmarks confirm the effectiveness of this approach.

### Strengths
Focusing on text embeddings is a great way to solve this problem. The paper does a good job of analyzing this, and the method is designed in a way that makes sense. The results are good, and it's also a bonus that the method is efficient.

### Weaknesses
My main concern is that the paper doesn't compare its method to the right baselines. The best embedding it compares against is from CLIP. Why not use embeddings from much more powerful modern models like Qwen-VL? The introduction claims that existing embeddings can't tell the difference between "standing on" and "walking on," but I'm not convinced this is true for today's large VLMs. This makes the problem seem less important than it's presented. The authors need to add an experiment comparing their learnable embedding directly against an embedding from a model like Qwen. They must prove their method is actually better.
There are also some small problems, including:
1. The claim that Figure 2 illustrates a "richer substructure" from left to right is not visually evident.
2. Does the structure in Figure 2(b) only appear in the scenario of closed-set SGG? What about for open-vocabulary SGG?

### Questions
Please refer to the issues detailed in the Weakness part.

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
This paper identifies a key limitation in current Scene Graph Generation (SGG) approaches—the reliance on frozen semantic embeddings from pre-trained language models such as GloVe and BERT. These static features, while useful in other domains, fail to adapt to the dynamic and context-sensitive nature of visual relationships. To address this issue, the authors propose Adaptive Prompt Tuning (APT), a lightweight and universal plug-in framework that injects learnable prompts to modulate frozen semantic representations into context-aware features. APT can be seamlessly integrated into both one-stage and two-stage SGG architectures, as well as open-vocabulary variants. Extensive experiments on Visual Genome, Open Images V6, and GQA show that APT significantly improves mean recall (up to +6.0 in mR@50 on novel splits) with minimal parameter overhead (<0.5M, <1.5%) and reduced training time (7.8%–25%). The authors further analyze the approach through ablation studies, efficiency evaluations, and theoretical grounding using the Information Bottleneck principle, demonstrating that APT effectively filters semantic noise and enhances contextually relevant representations for relational reasoning

### Strengths
The paper introduces a well-motivated and original problem formulation by diagnosing the representational rigidity of frozen semantic features as a fundamental bottleneck across all SGG paradigms. This diagnosis reflects deep insight beyond superficial performance concerns and unifies multiple architectural directions under a single, representation-level perspective. The proposed APT is elegant and conceptually sound, leveraging prompt tuning to dynamically adapt pre-trained embeddings without retraining large models. The design’s universality and modularity—applicable to both one-stage, two-stage, and open-vocabulary frameworks—demonstrate strong methodological generality. The experiments are comprehensive and convincing, covering multiple datasets, models, and metrics, and include detailed ablations and efficiency analyses that support the claimed advantages. Moreover, the authors’ theoretical explanation via the Information Bottleneck principle adds interpretability and rigor, showing awareness of both empirical and conceptual coherence.

### Weaknesses
While the paper presents solid experimental evidence, it would benefit from stronger empirical isolation of causal effects—for example, disentangling the relative influence of prompt conditioning versus visual feature fusion. 

The description of certain components (e.g., Basis Prompt Synthesis and Feature Refinement) could be more mathematically formalized to ensure reproducibility and conceptual clarity. 

Although the authors frame APT as a “universal paradigm,” the current validation remains centered around SGG; testing on broader relational reasoning tasks (e.g., visual question answering or video relation detection) would further substantiate its universality. 

The reliance on pre-trained language backbones and learned prompts also raises questions about potential semantic drift or bias amplification, which the paper acknowledges only implicitly. 

While the Information Bottleneck discussion provides an elegant theoretical perspective, it remains largely qualitative without direct quantitative verification (e.g., mutual information analysis). Addressing these aspects would make the argument more rigorous and strengthen the generalization claims.

### Questions
The paper identifies frozen semantic embeddings as the core bottleneck in Scene Graph Generation. Could the authors provide additional empirical evidence (e.g., controlled feature ablation or attention visualization) showing how adaptive prompts modulate semantic spaces differently from fine-tuning approaches?

How stable is APT when applied to different pre-trained language models (e.g., BERT vs. CLIP-text)? Does prompt initialization significantly affect convergence or generalization?

### Soundness
3

### Presentation
3

### Contribution
2

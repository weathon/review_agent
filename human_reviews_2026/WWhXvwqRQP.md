# Cognitive-Awakening Chain-of-Surgery for Compositional Zero-Shot Surgical Triplet Recognition

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 2, 2, 6

## Abstract
Compositional Zero-shot Surgical Triplet Recognition (CZSTR) is a challenging task that requires models to recognize novel combinations of <instrument, verb, target> that never co-occurred during training. This task captures the inherent generalization requirement in real surgical procedures. Large Vision-Language Models (LVLMs) with Chain-of-Thought (CoT), as one of the most advanced methods, are limitedly exposed to sufficient surgical semantics, leading to a shortage on the CZSTR task. To tackle this, we explore a more intuitive and natural human-like reasoning framework, which is introduced as Cognitive-awakening Chain-of-Sugery (CoCoS). CoCoS mirrors the way surgeons think: it starts by glancing at the scene, then gazing at the operation process over time, and finally drawing structured conclusions. Such a step-by-step cognitive-awakening process reflects how we naturally interpret surgical procedures and instruct large vision-language models (LVLMs) to deeply understand surgical scenes. Observing that LVLMs often hallucinate on relatively simple subtasks, e.g., identifying instruments, we further propose a Multimodal image–Sequence–Text (MiST) fusion module to reinforce the stability of the framework. To evaluate our framework, we also develop a strategy to reorganize existing surgical triplet datasets into a compositional zero-shot benchmark. Experiments show that our framework improves generalization to unseen triplets, outperforming both traditional models and LVLMs under this challenging task.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces the "Cognitive-awakening Chain-of-Surgery" (CoCoS) framework to tackle Compositional Zero-Shot Surgical Triplet Recognition (CZSTR), a novel and highly significant research topic for surgical AI generalization. While the problem itself is a valuable contribution, the paper's technical contribution is limited, as the method primarily combines existing models (LVLMs, SAM, MViT) with a new prompting strategy. The work's primary weaknesses are its insufficient experimental validation and poor reproducibility. The ablation study is critically incomplete. Furthermore, crucial implementation details such as hyperparameters and manual filtering rules are omitted, and no code has been released to compensate.

### Strengths
1. The paper introduces a novel and challenging task. It addresses a realistic gap by requiring models to recognize unseen combinations of known components, which is crucial for developing truly generalizable and trustworthy surgical AI systems.
2. It creatively leverages VLMs through the proposed "Cognitive-awakening Chain-of-Surgery" (CoCoS) framework. This method simulates a surgeon's cognitive process in three stages (Glance, Gaze, Think) to guide the model toward a deeper, context-aware understanding of complex surgical scenes.

### Weaknesses
1. The core idea of adapting a "Chain-of-Thought" reasoning process is not entirely novel in this domain. Even if the proposed 'Chain-of-Surgery' concept differs in its specifics, the authors should cite and discuss the relevant prior work, "Chain-of-look prompting for verb-centric surgical triplet recognition in endoscopic videos."
2. Regarding time efficiency, the use of both SAM, MViT and VLMs must result in a speed trade-off. However, real-time performance is critical in surgery. The paper should report what the exact inference time is, compare it against RDV, and quantify how much speed was sacrificed by the proposed method.
3. Reproducibility is a major concern. It took me a long time to find that 'T' (defined in Line 236) is 16 (in Line 360). The 'Implementation Details' section is almost empty; it has no hyperparameters, and the code has not been released.
4. In section 3.2.1, the authors mention using "manually design prior rules based on domain knowledge." However, these specific rules could not be found in the appendix. Furthermore, based on my knowledge, relying on such rules is unreliable. The size of tools and targets varies significantly with endoscopic zoom or their position at the periphery, meaning they can appear either very large or very small.  Also, since CholecT50 is annotated at 1 fps, it is normal for tools not to appear continuously, which contradicts the rule.
5. Using only one dataset for surgical triplets makes the experiment seem less reliable and limits the findings to the single task of cholecystectomy. Based on my search, the "prostatd: a large-scale multi-source dataset for structured surgical triplet detection" also provides triplet labels and could have been used.

### Questions
1.  Based on your Fig. 4, it looks like neither 'glance' nor 'gaze' is of much help for target recognition. Some of the associated phrases for the target seem inaccurate, and the link between the text and the visual target appears tenuous or forced. The results in Table 2 seem to reflect this phenomenon as well.
2.  The choice of 16 as the clip length is not justified. Surgical actions can be long or short, having variable durations. If this value was determined empirically, where is the ablation study to support it?
3.  In Table 2, what are the results for using only the 'Machine Encoding' module? Because the final judgment is based on the combination of both components' predictions, this baseline is essential to understand their individual contributions.
4.   Although the model is designed for unseen (rare) triplets (γ=80), it clearly also has the ability to recognize common triplets in a zero-shot manner. Why not use more diverse splits to test its capabilities?

### Soundness
2

### Presentation
2

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
The paper proposes CoCoS (Cognitive-awakening Chain-of-Surgery): a staged prompting scheme that queries large VLMs and fuses their text with spatial (ResNet-18+SAM) and temporal (MViT) features via a MiST fusion module to predict surgical triplets <instrument, verb, target> under a compositional zero-shot (CZSL) split of CholecT50. Results report gains over prior triplet models and recent VLM pretraining baselines.

### Strengths
- Important problem: zero-shot compositional triplets are clinically relevant for generalization.

### Weaknesses
- Baselines are potentially not all capable of composing unseen triplets (e.g., RDV) we makes some of the comparisons quite unfair  
- There seems to be a fundamental capacity mismatch between the models used for comparisons, while some use tiny ResNet-18, others essentially do a mixture of powerful foundation models such as Qwen-VL-Max and QVQ-Max with extra encoders  
- The approach mainly seems to benefit from the combination of out of the shelf models with limited finetuning.There is a lot of complexity in the method and I don’t think it is necessarily justified. Potentially a much simpler approach that still benefits from Qwen-VL-MAX and QVQ-Max would perform similarly. Generally the ablations don’t isolate the claimed contribution of this paper such as CoS well.  
- Training is unclear: which modules are trained, MiST only?  
- Under-specified SAM details and reporting: mask prompting/selection, instrument–anatomy mapping etc.

### Questions
- Trainable modules & loss: Exactly which components are trained/frozen? What are the objectives for I/V/T and full triplet scoring?  
- It would be good to add a split where test triplets are frequent (not just rare) to show true compositionality beyond long-tail effects.  
- Add a machine-only baseline: ResNet-18 \+ MViT, no LVLM text, I/V/T heads \+ optional compatibility.  
- Capacity-matched comparison: Either replace Qwen-VL-Max/QVQ-Max with an open model of comparable capacity (e.g., Qwen2-VL) or upgrade baselines to similar parameter/compute budgets.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper tackles compositional zero-shot surgical triplet recognition, predicting <instrument, verb, target> combinations that never co-occurred during training. It proposes a chain-of-surgery pipeline (glance -> gaze -> think) that stages an LVLM's perception and reasoning, then fuses this with a machine encoding branch that supplies grounded spatial and temporal cues. This paper also proposes MiST fusion module aligns text prompts and dynamics via tailored cross-attention.

### Strengths
1. Introducing compositional zero-shot recognition for surgical triplets is timely and realistic.
2. A clear, human-inspired scheduling of perception and reasoning for LVLMs.
3. On CholecT50 CZSL, the method outperforms prior VLM baselines; the cumulative gains in the ablation (mAP_ivt from 3.90→9.39) substantiate the contribution of each component.

### Weaknesses
1. The Chain-of-Surgery is compelling, but the paper should articulate more sharply how its staged prompting differs in principle from existing multi-step/multi-view LVLM prompting beyond domain adaptation (e.g., why exactly three stages; what information bottlenecks each stage resolves).
2. THis paper argues that there are 3 contributions. But I think the first two are actually one contribution. Moreover, the proposed chain-of-surgery is no big difference with former methods and has no contribution for the community. The improvement on performance comes from the tuning process and not the prompt.
3. Why you design MiST? what is the purpose of it? And lack the corresponding ablation studies.

### Questions
What is the model scale used in your model and the compared methods?

### Soundness
2

### Presentation
2

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
This paper proposes Cognitive-Awakening Chain-of-Surgery (CoCoS), a framework for compositional zero-shot surgical triplet recognition. It introduces a three-stage Glance–Gaze–Think prompting scheme that mimics human surgical reasoning to enhance large vision-language models’ understanding of surgical videos. A Machine Encoding module provides stable spatial-temporal cues, while the MiST fusion module aligns visual and textual features for robust triplet prediction. A compositional zero-shot split strategy is designed to evaluate generalization to unseen instrument–verb–target combinations. Experiments on CholecT50 demonstrate that CoCoS significantly outperforms existing baselines in accuracy and compositional generalization.

### Strengths
1. This work proposes the Cognitive-Awakening Chain-of-Surgery (CoCoS), a three-stage “Glance–Gaze–Think” prompting scheme that mirrors how surgeons observe, analyze, and decide during operations. This staged reasoning introduces an original multimodal adaptation of chain-of-thought, enabling large vision-language models to perform structured and clinically aligned interpretation of surgical videos.

2. This work proposes a new benchmark and data split strategy where each component of a surgical triplet (instrument, verb, target) is seen during training but their combinations are unseen at test time. This setting realistically captures clinical generalization needs and allows systematic evaluation of compositional reasoning in surgical AI.

3. This work proposes the MiST fusion module that effectively aligns spatial (segmentation-based), temporal (video-level), and textual representations, improving model robustness against hallucination. Extensive experiments on the CholecT50 dataset demonstrate clear gains across all triplet components, validating both methodological soundness and practical feasibility.

### Weaknesses
1. The method is only tested on CholecT50; it remains unclear whether the framework generalizes to other surgeries or datasets.

2. The multi-stage Glance–Gaze–Think process may be computationally heavy, yet the paper reports no inference time or feasibility for real-time use.

3. Although described as “human-like reasoning,” the work does not measure how well generated reasoning aligns with visual evidence or handles noisy scenes.

Ref:

[1] OphNet: A Large-Scale Video Benchmark for Ophthalmic Surgical Workflow Understanding

[2] OphCLIP: Hierarchical Retrieval-Augmented Learning for Ophthalmic Surgical Video-Language Pretraining

[3] Ophora: A Large-Scale Data-Driven Text-Guided Ophthalmic Surgical Video Generation Model

[4] Towards Dynamic 3D Reconstruction of Hand-Instrument Interaction in Ophthalmic Surgery

### Questions
1. Could the authors clarify how the Glance–Gaze–Think prompting is operationalized during inference? For example, is each stage conditioned on the previous stage’s textual output, or are they run independently and then fused by MiST? The paper would benefit from explicitly defining the interaction mechanism.

2. The paper mentions that Machine Encoding stabilizes perception when LVLMs hallucinate, yet it remains unclear how frequent or severe such hallucinations are. Could the authors provide quantitative evidence or failure examples to show how much this module reduces visual hallucination or improves grounding accuracy?

### Soundness
3

### Presentation
3

### Contribution
3

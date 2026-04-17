# CARE: Towards Clinical Accountability in Multi-Modal Medical Reasoning with an Evidence-Grounded Agentic Framework

- Decision: Accept (Poster)
- Scores: 6, 6, 6

## Abstract
Large visual language models (VLMs) have shown strong multi-modal medical reasoning ability, but most operate as end-to-end black boxes, diverging from clinicians’ evidence-based, staged workflows and hindering clinical accountability. Complementarily, expert visual grounding models can accurately localize regions of interest (ROIs), providing explicit, reliable evidence that improves both reasoning accuracy and trust. In this paper, we introduce **CARE**, advancing **C**linical **A**ccountability in multi-modal medical **R**easoning with an **E**vidence-grounded agentic framework. Unlike existing approaches that couple grounding and reasoning within a single generalist model, CARE decomposes the task into coordinated sub-modules to reduce shortcut learning and hallucination: a compact VLM proposes relevant medical entities; an expert entity-referring segmentation model produces pixel-level ROI evidence; and a grounded VLM reasons over the full image augmented by ROI hints. The VLMs are optimized with reinforcement learning with verifiable rewards to align answers with supporting evidence. Furthermore, a VLM coordinator plans tool invocation and reviews evidence-answer consistency, providing agentic control and final verification. Evaluated on standard medical VQA benchmarks, our **CARE-Flow** (coordinator-free) improves average accuracy by **10.9%** over the same size (10B) state-of-the-art (SOTA). With dynamic planning and answer review, our **CARE-Coord** yields a further gain, outperforming the heavily pre-trained SOTA by **5.2%**. Our experiments demonstrate that an agentic framework that emulates clinical workflows, incorporating decoupled specialized models and explicit evidence, yields more accurate and accountable medical AI.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes a framework for multi-modal reasoning specifically designed for medical VQA tasks. The task is split into three separate stages: (i) identifying medical entities relevant to the question, (ii) segmenting those entities in the image, (iii) visual reasoning using different views generated from the segmentations. They further propose the inclusion of a coordinating agent for planning and review. They report in-domain and out-of-domain improvements over comparable methods and include a comprehensive ablation study evaluating various components of their method.

### Strengths
The paper proposes a novel framework for agentic visual reasoning with tool usage in medicine. They identify core workflow steps helpful for medical visual reasoning and train dedicated agents for these tasks. While the core concept of tool-supported visual reasoning exists for generalist VQA with similar training strategies, the originality and significance of this work arises from a well-executed specialization for the medical domain that I can see to be easily built upon.  
The method, modules and training strategy is well motivated and thoroughly evaluated and ablated.  
The manuscript is well structured and explanations and figures are clear.

### Weaknesses
1. ##### **Missing related work:**

   The paper claims to beat state-of-the-art on VQA-RAD and SLAKE, however, there are multiple methods with better performance on VQA-RAD and SLAKE not mentioned in the comparison, e.g. \[1,2,3\]. These works should be included for a complete contextualization of the work and the claim adapted.

2. **Unclear contribution of the coordinator**

	In figure 3 and 18 we can see that the coordinator (GPT-5) is able to overwrite the answers of the medical entity proposal module as well as the final answer of the system. Especially on the OOD dataset, where GPT-5 alone outperforms CARE, I am wondering how often the coordinator makes use of this power, essentially circumventing the contributions of all the proposed modules and degenerating to an API call to ChatGPT (albeit with the input of the CARE modules). A different reading of the OOD results would be that GPT-5 reaches an accuracy of 62.20 but if we include all the info from the CARE pipeline, GPT-5 becomes worse.

\[1\] Yu, Ting, et al. "Fine-grained Adaptive Visual Prompt for Generative Medical Visual Question Answering." *Proceedings of the AAAI Conference on Artificial Intelligence*. Vol. 39\. No. 9\. 2025\.  
\[2\] Cui, Hejie, et al. "Biomedical visual instruction tuning with clinician preference alignment." *Advances in neural information processing systems* 37 (2024): 96449-96467.  
\[3\] Lin, Weixiong, et al. "Pmc-clip: Contrastive language-image pre-training using biomedical documents." *International Conference on Medical Image Computing and Computer-Assisted Intervention*. Cham: Springer Nature Switzerland, 2023\.

### Questions
I would like to see some statistics on how often the final answer is overwritten by GPT-5 and the accuracy of these overwrites. 

What is the binary modality token?

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
3

### Summary
This paper proposes CARE, an agentic framework that mirrors the clinical diagnostic workflow. Instead of treating medical visual question answering (VQA) as a monolithic black-box task, the authors decompose it into interpretable sub-steps: (1) a medical entity proposal stage, (2) a segmentation model to localize regions of interest (ROIs), and (3) an evidence-grounded VQA model that reasons over the full image along with different evidence views (zoom-in, masked, or global). Additionally, an LLM-based coordinator performs planning, reasoning, and quality control over these steps, improving performance and interpretability.

Overall, I find this to be a promising and well-motivated direction. The framework reflects clinical reasoning in a meaningful way, and the results appear solid across multiple datasets.

### Strengths
- Well-motivated workflow design. The authors tackle the core problem—controlling hallucinations in large medical VLMs—through a structured, interpretable workflow rather than pure end-to-end training. This decomposition aligns well with clinical reasoning and is conceptually elegant.

- Coordinator design. The introduction of an LLM-based coordinator for planning and quality control is intuitive and methodologically sound. It also adds an additional layer of reliability that most prior medical VQA systems lack.

- Strong empirical performance. The reported results show consistent improvement over prior works, demonstrating the practical value of the agentic workflow.

- Extensive experimentation. The authors conduct a large number of experiments, including multiple baselines and ablations, which adds credibility to the study.

### Weaknesses
See questions

### Questions
- Reproducibility and openness. Since the paper claims superior performance, open-sourcing the models, weights, and codebase would be critical for community validation. Without this, it’s difficult to verify the robustness and generality of the framework.

- Justification for reinforcement learning. The ablation study supports the inclusion of RL fine-tuning, but the paper should better articulate why reinforcement learning is necessary for both the entity proposal and the VQA stages. From the description, it appears to combine symbolic and correctness-based rewards; please clarify why supervised fine-tuning (SFT) would not suffice.

- Demonstration of diagnostic effectiveness. If the paper aims to position CARE as a clinically meaningful diagnostic model, the current visual examples (e.g., Figure 3) feel too trivial. The task shown (“Which organ has the largest area?”) is not representative of real diagnostic challenges and could likely be handled by standard detection models (e.g., YOLO). Consider adding more realistic examples—such as differentiating between cancerous vs. non-cancerous lung images—and illustrating how the coordinator or CoT reasoning contributes to that decision.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces CARE, an agentic framework for medical VQA that explicitly mirrors clinical workflows. CARE decomposes the task:
(i) a medical entity proposal VLM (trained with RLVR/DAPO using verifiable rewards), (ii) an entity-referring segmentation module (SA-Med-2D) providing pixel-level evidence (ROI masks/boxes), and (iii) an evidence-grounded VQA (EG-VQA) reasoner that answers using one of three evidence views: Zoom-in, Mask, or Global (a full-context “no localization needed” indicator).

Two operating modes are studied: CARE-Flow (deterministic pipeline) and CARE-Coord (a planner–reviewer agent that selects tools, iterates, and performs answer reviews). Across four benchmarks, the 10B CARE-Flow establishes strong new results; CARE-Coord further improves performance, surpassing prior 32B baselines. However, it's heavily dependent on using relatively bigger proprietary models such as GPT-5.

### Strengths
The paper's primary strength lies in its principled and clinically inspired architecture. The decomposition of medical reasoning into discrete, specialist-driven stages [hypothesize (entity proposal) → localize (segmentation) → reason (grounded VQA)] is a significant conceptual advance. This design directly addresses the "black box" problem by providing explicit, pixel-level evidence for each decision, thereby enhancing both interpretability and debuggability. Empirically, the work is robust, demonstrating strong performance across multiple datasets and supporting its claims with thorough ablation studies that dissect the contribution of each component, from the matching strategy to the agentic coordinator. The use of reinforcement learning with verifiable rewards (RLVR/DAPO) is a sophisticated technical choice that aligns model incentives with checkable signals, avoiding the need for costly human preference data. Finally, the paper is commendably detailed in its implementation description, providing prompts and tool schemas that greatly exceed the reproducibility standards of many contemporary agentic AI papers.

### Weaknesses
### **Baseline Comparison Fairness and Transparency**
The validity of the primary results in Table 1 depends critically on understanding each baseline’s training regime. It remains unclear which comparison models were fine-tuned on the same in-domain datasets (OmniMedVQA, VQA-RAD, SLAKE) and which were evaluated zero-shot. It is indeed unjust to call the columns in-domain and OOD solely based on your configurations and then compare them against other models. The authors should explicitly disclose the training exposure for every baseline—ideally in a small supplementary table—and group results into “Fine-tuned” vs. “Zero-shot/Unknown” categories. Without this disclosure, the magnitude of CARE’s improvement cannot be properly attributed to architectural design versus training-data advantage. The current Table also needs to be modified or at least clearly states the differences and point to the more complete Table which would be in the appendix.

### **Reward Function Analysis and Sensitivity**
While the EG-VQA reward is ablated, the Entity Proposal reward R_{\text{Entity}} = R_{\text{sim}} + R_{\text{count}} + R_{\text{repetition}} + R_{\text{format}} lacks empirical justification for its composition.
A component-wise ablation analysis or a reference to another paper that is using the same reward with clear ablations would clarify which sub-rewards most influence stability and downstream performance. This is particularly important since R_{\text{Entity}} governs the quality of the entity grounding that underpins the rest of the pipeline.

### **Isolating Framework Contributions (CARE-Coord-Open Variant)**
The performance of CARE-Coord depends in part on its proprietary coordinator (GPT-5). To distinguish architectural benefit from raw model capacity, the paper should include a CARE-Coord-Open variant using the strongest open-source VLM employed elsewhere (e.g., InternVL3-8B). Reporting this gap would isolate how much of the gain stems from the agentic framework itself versus the closed-model advantage.

### **Misleading Parameter-Scaling Visualization**
Figure 1(e) plots proprietary models with unknown parameter counts on a “Billions of Parameters” axis, inadvertently implying knowledge of their scale and suggesting a false comparison. The authors should redesign the plot; for instance, use a separate panel, shaded region, or qualitative marker to avoid misrepresenting scaling relationships.

### Questions
Also refer to **Weaknesses**.

### **Baseline Training Exposure and Evaluation Fairness**
For each baseline reported in Table 1, please explicitly indicate whether the model was:
(a) fine-tuned on any of the in-domain datasets (OmniMedVQA, VQA-RAD, SLAKE),
(b) fine-tuned on other medical VQA datasets, or (c) evaluated zero-shot using public checkpoints. This transparency is critical to disentangle architectural advantages from training-data advantages. A compact supplementary table summarizing this information would substantially improve interpretability and fairness.

### **Quantitative Evaluation of Accountability**
Beyond qualitative case studies, have you considered measuring accountability quantitatively (e.g., the overlap between predicted answer entities and the segmented ROI, or expert ratings of reasoning trace correctness)? Even a limited-scale human evaluation or automatic alignment metric would strengthen the accountability claim.

### **Coordinator Error and Review Conservatism**
In Figure 18, the coordinator overrides a correct specialist answer, illustrating a potential hallucination at the review stage.
How frequently do such coordinator-induced reversals occur across the evaluation set? Have you explored conservative review heuristics (e.g., vetoing coordinator changes when specialist confidence is high) to mitigate these errors?

### **Evaluation Stability and LLM-Based Scoring**
For open-ended questions evaluated by GPT-4-class models, have you measured inter-rater agreement (e.g., multiple LLM or human graders) or compared results with deterministic metrics like normalized exact match? Providing a measure of scoring variance would improve confidence in reported accuracy differences.

### Soundness
3

### Presentation
3

### Contribution
3

# Weakness Patterns from ICLR 2025 Reviews Relevant to AdaIB

## Overview
This document identifies common weakness patterns from human reviews of papers on related topics (multimodal/vision-language interpretability, attribution methods, information bottleneck, CLIP analysis) that could apply to AdaIB (Adaptive Information Bottleneck for Multimodal Attribution).

---

## 1. EVALUATION METHODOLOGY ISSUES

### 1.1 Limited Baseline Comparisons
**Pattern**: Insufficient comparison with existing attribution/interpretability methods
- **From review Zggz6seq6F (FIOVA)**: "There are no experiments that demonstrate why the authors' dataset is superior to other datasets"
- **From review 2bEjhK2vYp (SSL Attribution)**: "The paper only compares SSLA to a random masking baseline. There are no comparisons with other existing attribution methods adapted to SSL, making it hard to gauge the effectiveness of SSLA."
- **From review rpbzBXdo4x (Multimodal CoT)**: "Many of the expectations seem very unintuitive... criterion (ii), 'human constraints generalize to AI', has a massive amount of room for interpretation and post-hoc result fitting"

**Implication for AdaIB**: Must compare against established attribution methods (e.g., Integrated Gradients, attention-based methods) across multiple vision-language models, not just CLIP variants.

### 1.2 Single or Limited Datasets
**Pattern**: Evaluation limited to one or few datasets/models
- **From review KnYsdgeCey (SSL Attribution)**: "Although experiments are conducted on the ImageNet dataset using ResNet-50, the evaluation lacks diversity in both datasets and model architectures. The claim that SSLA is architecture-agnostic is not fully supported without experiments on different architectures."
- **From review 0iAZYF9hrl (Disentangled Rep Learning)**: "The scope of this work appears too narrow, focusing solely on microscopy images... The proposed approach might be more convincing if demonstrated on natural images as well."
- **From review L9j8exYGUJ (Multi-hop Reasoning)**: "The primary weakness of the paper is that it only considers one clean, synthetic dataset... It is unclear whether such a clean result would hold on a more realistic reasoning dataset."

**Implication for AdaIB**: Need evaluation on diverse vision-language datasets (Flickr30K, COCO, CC3M variants), multiple VLMs (CLIP variants, BLIP, LLaVA), and varied types of misalignment (text errors, image degradation, irrelevant pairs).

### 1.3 Evaluation Metric Issues
**Pattern**: Limited or arbitrary evaluation metrics; inadequate justification for metric choice
- **From review Zggz6seq6F (FIOVA)**: "No new metrics are proposed... How can GPT-3.5 evaluate dimensions like Context, Correctness for video captions without accessing the visual parts?"
- **From review Zggz6seq6F (FIOVA Reviewer 4)**: "AutoCQ only provides event-level evaluation. Why were the five dimensions (e.g., consistency, context, etc.) from Sec. 2.2 not considered in the model response evaluation metrics?"
- **From review 2bEjhK2vYp (PbP)**: "The decision of using Levenshtein distance as the main metric for the evaluation is arbitrary"
- **From review Tepaft7632 (Time Series Anomaly)**: "The authors use the wrong evaluation metrics... many works have demonstrated that PA can lead to faulty performance evaluations"

**Implication for AdaIB**: Metrics must be justified theoretically. For attribution, should use: (1) faithfulness metrics (e.g., insertion/deletion curves), (2) correlation with human annotations, (3) information-theoretic measures beyond standard IB metrics.

---

## 2. GENERALIZATION AND SCOPE LIMITATIONS

### 2.1 Generalization to Diverse Conditions
**Pattern**: Limited evidence for generalization across different data distributions, model types, or real-world conditions
- **From review Zggz6seq6F (FIOVA Reviewer 3)**: "The evaluation model has limitations. This article evaluates some selected LVLMs, but does not explore broader models, such as the business model Gemini-1.5-Pro"
- **From review Tepaft7632 (MADCluster)**: "The title of this paper seems to suggest that MADCluster can broadly be applied across various data types. However, the experimental validation is limited to time-series datasets... creating a discrepancy between the title's generality and the paper's scope."
- **From review 2bEjhK2vYp (PbP Reviewer 3)**: "Implicit preference ambiguity: while the hierarchical structure aims to maintain consistency, the authors didn't adequately ensure that preferences are truly implicit and understood across different scenes."

**Implication for AdaIB**: Must demonstrate robustness across:
- Different types of image-text misalignment (not just the specific misalignment types in training data)
- Different VLM architectures and training objectives
- Different types of images (different resolutions, visual complexity, domains)
- Different text conditions (length, language, semantic complexity)

### 2.2 Real-World Applicability Concerns
**Pattern**: Gap between method assumptions and real-world conditions
- **From review kFsWpSxkFz (MetaUrban)**: "The main goal of such a large-scale simulator is to eventually transfer learned models to the real world. However, there is limited experimentation with real robots"
- **From review 2bEjhK2vYp (PbP Reviewer 2)**: "How would the agent get access to egocentric observations of the user? Ideally the robot should learn to pick up preferences by watching a human perform the task from its own camera."

**Implication for AdaIB**: Need to validate on realistic misalignment scenarios, not artificial constructions. Consider: (1) natural distribution shifts in real data, (2) how the method performs on hard examples, (3) computational cost in practical deployment.

---

## 3. THEORETICAL GAPS AND INFORMAL CLAIMS

### 3.1 Lack of Theoretical Justification
**Pattern**: Methods proposed without theoretical grounding or clear formal analysis
- **From review rPup1cWk4d (Bezier Curves)**: "'Extraordinary claims require extraordinary evidence.' Proposing a new family of neural networks needs strong evidence in their advantages with existing methods, in either theoretical or empirical aspect, or more ideally, in both. Unfortunately, this paper fails to persuade me they are better."
- **From review dxMffCAd4w (Bezier)**: "The paper claims that the CLF is 'fully interpretable'... This is groundless. To render a solid discussion, please (1) define what is explainable, (2) explain why activation make it unexplainable, and (3) show how to explain the results of CLF."
- **From review KnYsdgeCey (SSL Attribution)**: "The two main components of the method lack motivation... The correlation (or even causality) of this and 'SSL as learning representation' is not clear."

**Implication for AdaIB**: Must clearly justify why information bottleneck is the right framework for this problem. Need formal analysis of:
- Why IB helps with misaligned data specifically
- How the adaptive mechanism ensures better attribution than fixed-weight IB
- Theoretical guarantees or bounds on attribution quality

### 3.2 Informal or Unjustified Design Choices
**Pattern**: Key design decisions not well motivated or justified
- **From review IqGVIU4rvM (SEED Tokenizer)**: "The overall pipeline is too complicated and probably not a scalable approach... For learning the semantic embedding, why not simply learn continuous embeddings from Q-former and not perform any quantization at all? What's the use of quantization and then de-quantization?"
- **From review KnYsdgeCey (SSL Attribution)**: "The iterative method. The author may consider justify why we need an iterative method to attribute the importance."
- **From review 2bEjhK2vYp (PbP)**: "It is unclear for me why scale and texture are more relevant than 'scale and shape', as it is the case before fine-tuning."

**Implication for AdaIB**: Every algorithmic choice (e.g., how information bottleneck is adapted, how misalignment is detected/weighted) must be clearly justified with ablations showing its necessity.

### 3.3 Unvalidated Assumptions
**Pattern**: Core assumptions of the method not validated
- **From review rpbzBXdo4x (CoT)**: "Condition 'B', however, is very vague and hard to apply... Whether such constraints are mirrored by LLMs is, again, a separate research question"
- **From review L9j8exYGUJ (Multi-hop Reasoning)**: "This analysis suggests that the suggest-then-narrow-down reasoning approach can be **induced** from the model's intermediates, but that is not the same thing as the model necessarily **using** that approach. How can we have confidence that the model is actually using this reasoning chain?"

**Implication for AdaIB**: Must validate the core assumption that misaligned pairs harm attribution quality. Need evidence that: (1) the method actually detects misalignment, (2) the adaptation mechanism improves robustness, (3) the IB approach is actually the mechanism driving improvements.

---

## 4. COMPUTATIONAL COST AND EFFICIENCY CONCERNS

### 4.1 Insufficient Analysis of Computational Cost
**Pattern**: Lack of discussion about computational overhead, scalability, or runtime efficiency
- **From review rPup1cWk4d (Bezier)**: "The experiments are limited to small-scale datasets (with <10k samples and <1k dimension), and it's unclear whether the proposed method is scalable (for computational reasons) to more complex datasets."
- **From review zcTLpIfj9u (Medical Imaging)**: "There is no comparison evaluation between the proposed method and prior methods in model pre-training... Baselines without the pre-training process should also be reported."
- **From review dxMffCAd4w (Bezier)**: "I believe the operational complexity of this approach could be significant when dealing with high-dimensional inputs and outputs... we suggest that the authors include a detailed complexity analysis section"

**Implication for AdaIB**: Need explicit discussion of:
- Computational overhead of the adaptive IB mechanism vs. standard attribution
- Scalability to high-resolution images and long text sequences
- Memory requirements during training and inference
- Comparison of runtime with simpler baseline methods

### 4.2 Missing Ablation Studies
**Pattern**: Key components not ablated to demonstrate individual contributions
- **From review Tepaft7632 (MADCluster)**: "There is no ablation study in the paper to demonstrate the independent function of learnable cluster center and one-directed adaptive loss."
- **From review dxMffCAd4w (Bezier)**: "A more direct baseline is to train the full model with all blocks during quantization"
- **From review 2bEjhK2vYp (PbP)**: "They pass in various kinds of image-based observations - are all these observations necessary? For example, would an agent still work as well if the third-person view is not provided? This ablation seems to be missing."

**Implication for AdaIB**: Must include ablations for:
- IB mechanism (with/without information bottleneck)
- Adaptation mechanism (fixed weights vs. adaptive)
- Misalignment detection component
- Different weighting schemes for misaligned pairs

---

## 5. DATA QUALITY AND TRAINING DATA CONCERNS

### 5.1 Ground Truth Quality Issues
**Pattern**: Concerns about the quality or validity of ground truth/annotations
- **From review Zggz6seq6F (FIOVA Reviewer 3)**: "There are doubts about the collection of groundtruth in FIOVA. GPT-3.5-Turbo cannot directly see the video, induction based on human text order alone can easily bring errors such as illusions to groundtruth"
- **From review Zggz6seq6F (FIOVA Reviewer 4)**: "Using an LLM instead of a VLM to summarize the five human captions is insufficient because an LLM cannot properly handle conflicting information in the five human captions"
- **From review 0iAZYF9hrl (Disentangled)**: "Important metrics are either not explained in the text or lack adequate definitions in the captions, leaving readers uncertain of their meaning. This omission impacts the study's reproducibility"

**Implication for AdaIB**: For training on misaligned data:
- Need clear definition and validation of what constitutes "misalignment"
- Ensure human annotation of misalignment is reliable (inter-annotator agreement)
- Document the construction process thoroughly
- Provide examples and statistics of different misalignment types

### 5.2 Dataset Bias Issues
**Pattern**: Potential biases in dataset construction or selection not addressed
- **From review 2bEjhK2vYp (PbP Reviewer 3)**: "Biases in few-shot demonstrations? The reliance on rule-based generation of few-shot examples might introduce potential biases. The scenarios may be constructed in a way that fits the developers' understanding of user preferences"

**Implication for AdaIB**: Must address:
- Potential bias toward specific types of misalignment in training data
- Whether the misalignment distribution reflects real-world conditions
- How the method performs on underrepresented misalignment types

---

## 6. COMPARISON AND FAIRNESS ISSUES

### 6.1 Unfair Comparisons
**Pattern**: Baseline comparisons conducted under different conditions or settings
- **From review L9j8exYGUJ (EfficientQAT)**: "The proposed method only did weight quantization, but many of baselines were using both activation and weight quantization... This paper adopts 4096 samples in RedPajama datasets with 2048 sequence length... Thus, the comparison with those PTQ works (AWQ, OminiQuant and GPTQ) is unfair"
- **From review IqGVIU4rvM (SEED Tokenizer)**: "The authors compare their work with SEED/LaVIT while acknowledging different metrics: 'SSIM and PSNR are not particularly meaningful for those models'... Nevertheless use these comparisons to claim improvement"

**Implication for AdaIB**: Ensure fair comparison by:
- Using same input data and preprocessing for all methods
- Comparing methods in same computational budget/training setup
- Using metrics appropriate for all compared methods
- Clearly stating any differences in experimental setup

### 6.2 Limited Related Work Discussion
**Pattern**: Insufficient engagement with existing literature and related methods
- **From review Tepaft7632 (MADCluster)**: "The claimed novelty in this paper is Model-Agnostic. But the authors did not compare with any Model-Agnostic anomaly detection models... The authors should include these related works in one section, compare these prior works by experiments"
- **From review 0iAZYF9hrl (Disentangled)**: "There is a lack of comparative experiments... The paper appears to lack comparative experiments... the absence of comparisons for classification performance is particularly concerning and unreasonable"

**Implication for AdaIB**: Must discuss:
- Related attribution methods for vision-language models
- Existing approaches to handling data quality issues
- Information bottleneck applications in other domains
- How AdaIB differs from and improves upon prior work

---

## 7. PRESENTATION AND CLARITY ISSUES

### 7.1 Presentation and Clarity
**Pattern**: Papers lacking clear exposition of methods or poor presentation quality
- **From review rPup1cWk4d (CoT)**: "The main text would benefit from more detailed task descriptions & exposition beyond Figure 1"
- **From review 0iAZYF9hrl (Disentangled)**: "The paper's presentation suffers from numerous issues... Sections like Section 2.2 resemble output generated by ChatGPT and lack rigorous academic polish... Figures appear low-resolution, with inadequate explanations in captions"
- **From review L9j8exYGUJ (Multi-hop)**: "Although the analysis provided in the paper is quite interesting... Figure 4 does not include legends for the plots, making them hard to interpret"

**Implication for AdaIB**:
- Clearly articulate the problem being solved and why it matters
- Use concrete examples for misaligned image-text pairs
- Include high-quality visualizations of attributions
- Provide algorithmic pseudocode for reproducibility

---

## 8. GENERALIZATION CLAIMS VS. EVIDENCE

### 8.1 Overstated Generality Claims
**Pattern**: Paper claims broad applicability but provides limited evidence
- **From review IqGVIU4rvM (SEED)**: "Lack of novelty or interestingness - On a high level, the paper mainly augments the SEED tokenizer with an additional low-resolution copy of the input image... The choices made for different component of this pipeline rather made it much more complicated to be useful in practice"
- **From review 0iAZYF9hrl (Disentangled Reviewer 2)**: "Given the lack of compelling insights, this work appears to be primarily an application of existing DRL methods without significant methodological or theoretical innovation"

**Implication for AdaIB**:
- Clearly delineate scope (which types of misalignment are handled, which models are supported)
- Don't claim general adaptivity if only shown on CLIP
- Acknowledge limitations explicitly

---

## Summary of Key Recommendations for AdaIB

### Critical Areas to Address:
1. **Evaluate on multiple datasets** (Flickr30K, COCO, CC3M) with multiple VLM architectures
2. **Use proper baseline comparisons** with established attribution methods (Integrated Gradients, attention-based)
3. **Provide theoretical justification** for why IB is the right framework
4. **Include comprehensive ablations** of all major components
5. **Validate core assumptions** that misaligned data harms attribution and that adaptation helps
6. **Analyze computational cost** and scalability
7. **Use appropriate evaluation metrics** with theoretical justification
8. **Test real-world generalization** on naturally occurring misalignment
9. **Ensure fair comparisons** with equal setup and resources for baselines
10. **Discuss limitations clearly** rather than overselling generality


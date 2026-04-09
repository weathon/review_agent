# Detailed Evidence: Common Weakness Patterns from ICLR 2025 Reviews

## Review Source Information

Papers reviewed (topics related to AdaIB):
- **Zggz6seq6F**: FIOVA - Video Description Benchmark (Vision-Language Evaluation)
- **zcTLpIfj9u**: Medical Imaging with EHR (Multimodal Learning)
- **rPup1cWk4d**: Poset-based Data Augmentation & CoT Effects on LLMs/VLMs
- **IqGVIU4rvM**: SEED - Semantic Diffusion Image Tokenization
- **gc8QAQfXv6**: (Multimodal-related)
- **Tepaft7632**: MADCluster - Anomaly Detection (Model-Agnostic)
- **kFsWpSxkFz**: MetaUrban - Urban Simulation Platform
- **2bEjhK2vYp**: Preference-based Planning for Embodied AI (Multimodal)
- **0iAZYF9hrl**: Disentangled Representation Learning for Microscopy
- **KnYsdgeCey**: Self-Supervised Learning Attribution (SSLA)
- **rpbzBXdo4x**: When Does CoT Hurt LLMs/VLMs?
- **dxMffCAd4w**: Bezier Curve Fitting Networks (Interpretability)
- **6Mdvq0bPyG**: EfficientQAT - Quantization for LLMs
- **L9j8exYGUJ**: Multi-hop Reasoning in LLMs (Interpretability)

---

## Weakness Pattern 1: Evaluation Methodology Issues

### 1A. Limited Baseline Comparisons

**Evidence Quote 1 - FIOVA (Zggz6seq6F, Reviewer 1):**
> "There are no experiments that demonstrate why the authors' dataset is superior to other datasets. For instance, is the FIOVA dataset better than the DREAM-1K dataset used in Tarsier for evaluation? Do the models that perform better on FIOVA also perform better in human evaluations?"

**Evidence Quote 2 - SSL Attribution (KnYsdgeCey, Reviewer 2):**
> "The paper only compares SSLA to a random masking baseline. There are no comparisons with other existing attribution methods adapted to SSL, making it hard to gauge the effectiveness of SSLA."

**Evidence Quote 3 - CoT Effects (rpbzBXdo4x, Reviewer 1):**
> "While presented results with standard CoT prompts support the thesis that naive CoT can be harmful, additional prompting variations would make the findings more robust."

**Why it matters for AdaIB:**
- Attribution methods have established baselines: Integrated Gradients, Gradient*Input, attention mechanisms
- Need comparison to show AdaIB improves over these established methods
- Random baselines are insufficient

### 1B. Single/Limited Dataset Evaluation

**Evidence Quote 1 - SSL Attribution (KnYsdgeCey, Reviewer 2):**
> "Limited Dataset and Model Diversity: Although experiments are conducted on the ImageNet dataset using ResNet-50, the evaluation lacks diversity in both datasets and model architectures. The claim that SSLA is architecture-agnostic is not fully supported without experiments on different architectures."

**Evidence Quote 2 - Disentangled Representation (0iAZYF9hrl, Reviewer 2):**
> "The scope of this work appears too narrow, focusing solely on microscopy images. The proposed approach might be more convincing if demonstrated on natural images as well."

**Evidence Quote 3 - Multi-hop Reasoning (L9j8exYGUJ, Reviewer 2):**
> "The primary weakness of the paper is that it only considers one clean, synthetic dataset with one type of question. It is unclear whether such a clean result would hold on a more realistic reasoning dataset."

**Why it matters for AdaIB:**
- Different VLMs have different architectures: ViT-based, CNN-based backbones
- Different datasets have different characteristics of misalignment
- Claiming robustness requires testing on multiple models and datasets

### 1C. Evaluation Metric Issues

**Evidence Quote 1 - FIOVA (Zggz6seq6F, Reviewer 2):**
> "While this work has adopted multiple metrics to demonstrate the video caption performance, it lacks analysis of how those metrics align with human preference."

**Evidence Quote 2 - FIOVA (Zggz6seq6F, Reviewer 4):**
> "AutoCQ only provides event-level evaluation. Why were the five dimensions (e.g., consistency, context, etc.) from Sec. 2.2 not considered in the model response evaluation metrics?"

**Evidence Quote 3 - Anomaly Detection (Tepaft7632, Reviewer 2):**
> "The authors use the wrong evaluation metrics. The authors use the point adjustment (PA) for evaluation. Many works [5, 6, 7] have demonstrated that PA can lead to faulty performance evaluations, where PA use true labels from the test datasets to adjust the outputs of models, and it is known that using PA can result in state-of-the-art performance even with random scores or random initialized non-trained models [6, 7], making it impossible to conduct a fair comparison."

**Evidence Quote 4 - Planning (2bEjhK2vYp, Reviewer 2):**
> "The decision of using Levenshtein distance as the main metric for the evaluation is arbitrary and penalises the agent for not exactly matching the training data format."

**Why it matters for AdaIB:**
- Standard attribution metrics: faithfulness (insertion/deletion), ROAR, correlation with human annotations
- Metrics must be justified theoretically, not arbitrary choices
- Need multiple metrics to assess different aspects of quality

---

## Weakness Pattern 2: Generalization & Scope Limitations

### 2A. Limited Model/Architecture Coverage

**Evidence Quote 1 - FIOVA (Zggz6seq6F, Reviewer 3):**
> "The evaluation model has limitations. This article evaluates some selected LVLMs, but does not explore broader models, such as the business model Gemini-1.5-Pro, which has a strong understanding of long videos."

**Evidence Quote 2 - SSL Attribution (KnYsdgeCey, Reviewer 2):**
> "The claim that SSLA is architecture-agnostic is not fully supported without experiments on different architectures."

**Why it matters for AdaIB:**
- CLIP has many variants: CLIP ViT-B/32, ViT-L/14, ViT-H/14, ViT-g/14
- Other VLMs: BLIP, BLIP-2, LLaVA, EVA-CLIP, OpenCLIP variants
- Each may respond differently to misalignment and adaptation

### 2B. Discrepancy Between Claimed Scope and Tested Scope

**Evidence Quote 1 - MADCluster (Tepaft7632, Reviewer 4):**
> "The title of this paper seems to suggest that MADCluster can broadly be applied across various data types. However, the experimental validation is limited to time-series datasets, with no testing on other data types (e.g., structured tabular data, graphs, images). This creates a discrepancy between the title's generality and the paper's scope."

**Evidence Quote 2 - Planning (2bEjhK2vYp, Reviewer 3):**
> "Implicit preference ambiguity: while the hierarchical structure aims to maintain consistency, the authors didn't adequately ensure that preferences are truly implicit and understood across different scenes. Variability in scene context and object interactions could lead to unintended changes in how a preference is perceived for same person."

**Why it matters for AdaIB:**
- Don't claim "adaptive information bottleneck for vision-language models" if only tested on CLIP
- Document which types of misalignment are handled vs. not handled
- Test across vision-language domains: web data, academic datasets, user-generated content

---

## Weakness Pattern 3: Theoretical Gaps & Informal Claims

### 3A. Lack of Theoretical Justification

**Evidence Quote 1 - Bezier Curves (rPup1cWk4d, Reviewer 1):**
> "'Extraordinary claims require extraordinary evidence.' Proposing a new family of neural networks needs strong evidence in their advantages with existing methods, in either theoretical or empirical aspect, or more ideally, in both. Unfortunately, this paper fails to persuade me they are better. The experiments are too simple – only comparing the method with MLP using MNIST."

**Evidence Quote 2 - Bezier Curves (dxMffCAd4w, Reviewer 2):**
> "I don't see how the proposed CLF is more interpretable than MLP. Instead of learning linear combinations of input in MLP, CLF is learning quadratic functions over each dimension and then sum over all dimensions. The nonlinear function of each dimension is itself not interpretable."

**Evidence Quote 3 - SSL Attribution (KnYsdgeCey, Reviewer 3):**
> "The two main components of the method lack motivation... The first one is using cosine-similarity of features before/after transformation as a measure of usefulness of SSL model. The correlation (or even causality) of this and 'SSL as learning representation' is not clear."

**Why it matters for AdaIB:**
- Why does information bottleneck specifically help with misalignment?
- How does the adaptive mechanism ensure better attribution?
- What are the theoretical guarantees or approximation bounds?

### 3B. Unvalidated Core Assumptions

**Evidence Quote 1 - CoT Effects (rpbzBXdo4x, Reviewer 2):**
> "Condition 'B', however, is very vague and hard to apply... Whether such constraints are mirrored by LLMs is, again, a separate research question. It is especially moot since LLMs are known to essentially retrofit their explanations to the answer, so it's not clear to what extent and in which conditions LLM explanations dictate the actual flow of their reasoning."

**Evidence Quote 2 - Multi-hop Reasoning (L9j8exYGUJ, Reviewer 2):**
> "This analysis suggests that the suggest-then-narrow-down reasoning approach can be **induced** from the model's intermediates, but that is not the same thing as the model necessarily **using** that approach. How can we have confidence that the model is actually using this reasoning chain and not rely upon other forms of reasoning instead/in addition?"

**Why it matters for AdaIB:**
- Validate that misaligned data actually harms attribution quality
- Show that the adaptation mechanism actually detects and handles misalignment
- Prove that IB-based approach is what drives improvements (not just having more samples)

### 3C. Unjustified Design Choices

**Evidence Quote 1 - SEED Tokenizer (IqGVIU4rvM, Reviewer 1):**
> "The overall pipeline is too complicated and probably not a scalable approach for learning compressed embeddings: For learning the semantic embedding, building on top of prior work SEED, the proposed approach first learns quantized 1D embeddings using Causal Q-Former, but then also performs the task of reverse Q-former... Why not simply learn continuous embeddings from Q-former and not perform any quantization at all? What's the use of quantization and then de-quantization?"

**Evidence Quote 2 - SSL Attribution (KnYsdgeCey, Reviewer 3):**
> "The iterative method. The author may consider justify why we need an iterative method to attribute the importance."

**Why it matters for AdaIB:**
- Every component needs justification (e.g., why IB over other weighting schemes?)
- Ablation studies must isolate the contribution of each design choice

---

## Weakness Pattern 4: Computational Cost & Efficiency

### 4A. Missing Computational Analysis

**Evidence Quote 1 - Bezier Curves (rPup1cWk4d, Reviewer 1):**
> "The experiments are limited to small-scale datasets (with <10k samples and <1k dimension), and it's unclear whether the proposed method is scalable (for computational reasons) to more complex datasets."

**Evidence Quote 2 - Bezier Curves (dxMffCAd4w, Reviewer 4):**
> "I believe the operational complexity of this approach could be significant when dealing with high-dimensional inputs and outputs... we suggest that the authors include a detailed complexity analysis section, comparing the time and space complexity of CLF to traditional neural networks for various input dimensions."

**Evidence Quote 3 - Medical Imaging (zcTLpIfj9u, Reviewer 1):**
> "Interpretability: The TTE pretraining's impact on specific pixel-level biomarkers is less clear; additional analysis on feature attribution could help."

**Why it matters for AdaIB:**
- Information bottleneck computation can be expensive
- Adaptation mechanism must not be prohibitively slow
- Need to report runtime relative to standard attribution methods

### 4B. Missing Ablation Studies

**Evidence Quote 1 - Time Series (Tepaft7632, Reviewer 1):**
> "There is no ablation study in the paper to demonstrate the independent function of learnable cluster center and one-directed adaptive loss."

**Evidence Quote 2 - Planning (2bEjhK2vYp, Reviewer 4):**
> "They pass in various kinds of image-based observations - are all these observations necessary? For example, would an agent still work as well if the third-person view is not provided? This ablation seems to be missing."

**Evidence Quote 3 - EfficientQAT (6Mdvq0bPyG, Reviewer 2):**
> "The ablation for scale optimization (E2E-QP) with weights from post-training quantization methods compared to Block-AP weights is needed."

**Why it matters for AdaIB:**
- Ablate IB loss component
- Ablate misalignment detection component
- Ablate adaptive weighting mechanism
- Show that adaptation actually helps (not just random chance)

---

## Weakness Pattern 5: Data Quality & Training Data

### 5A. Ground Truth Quality Issues

**Evidence Quote 1 - FIOVA (Zggz6seq6F, Reviewer 3):**
> "There are doubts about the collection of groundtruth in FIOVA. FIOVA carefully designed manual annotations composed of five human annotator annotations, and merged and rewrote human annotations with GPT-3.5-Turbo. However, since GPT-3.5-Turbo cannot directly see the video, induction based on human text order alone can easily bring errors such as illusions to groundtruth."

**Evidence Quote 2 - FIOVA (Zggz6seq6F, Reviewer 4):**
> "Using an LLM instead of a VLM to summarize the five human captions is insufficient because an LLM cannot properly handle conflicting information in the five human captions. For example, in Figure 4, Human3 notes that the little boy cries at the end, while Human5 states that the boy smiles at the end. Since an LLM cannot 'see' the video, it may simply guess that the boy smiles at the end."

**Evidence Quote 3 - Disentangled (0iAZYF9hrl, Reviewer 3):**
> "Important metrics are either not explained in the text or lack adequate definitions in the captions, leaving readers uncertain of their meaning. This omission impacts the study's reproducibility and overall clarity."

**Why it matters for AdaIB:**
- Need clear definition of "misalignment" with examples
- Validate human annotations of misalignment (inter-annotator agreement)
- Document the exact annotation protocol
- Provide statistics on misalignment types

### 5B. Dataset Bias Issues

**Evidence Quote 1 - Planning (2bEjhK2vYp, Reviewer 3):**
> "Biases in few-shot demonstrations? The reliance on rule-based generation of few-shot examples might introduce potential biases. The scenarios may be constructed in a way that fits the developers' understanding of user preferences, which might not capture the diversity and unpredictability of real-world user behaviors."

**Why it matters for AdaIB:**
- Check if misalignment dataset covers diverse types equally
- Validate that synthetic/constructed misalignment reflects real conditions
- Test on naturally occurring misalignment (web data, user-generated content)

---

## Weakness Pattern 6: Comparison & Fairness

### 6A. Unfair Comparisons

**Evidence Quote 1 - EfficientQAT (6Mdvq0bPyG, Reviewer 2):**
> "The novelty of this paper is limited. The block-wise QAT is not novel as the block-wise methods are commonly used in quantization and pruning methods for LLMs... The work focuses on the weight-only quantization, while the comparison works contain many weight and activation both quantized methods including OmniQuant and LLM-QAT, which shows unfairness."

**Evidence Quote 2 - EfficientQAT (6Mdvq0bPyG, Reviewer 2):**
> "This paper adopts 4096 samples in RedPajama datasets with 2048 sequence length for the Block-AP and 4096 sequence length for the E2E-QP. Thus, the comparison with those PTQ works (AWQ, OminiQuant and GPTQ) is unfair. The paper did not explain the setting of those PTQ works, if they are also use such amount of data with such sequence length for calibration?"

**Evidence Quote 3 - SEED Tokenizer (IqGVIU4rvM, Reviewer 3):**
> "The authors compare their work with SEED/LaVIT while acknowledging different metrics: 'SSIM and PSNR are not particularly meaningful for those models'... Nevertheless use these comparisons to claim improvement."

**Why it matters for AdaIB:**
- Use same data, preprocessing, hyperparameters for all methods
- Don't compare methods developed for different objectives
- Use metrics appropriate for all compared methods
- Explicitly state any differences in setup

### 6B. Insufficient Related Work Discussion

**Evidence Quote 1 - MADCluster (Tepaft7632, Reviewer 2):**
> "The claimed novelty in this paper is Model-Agnostic. But the authors did not compare with any Model-Agnostic anomaly detection models... The authors should include these related works in one section, compare these prior works by experiments, and provide a detailed comparison highlighting the key technical differences and innovations."

**Evidence Quote 2 - Disentangled (0iAZYF9hrl, Reviewer 2):**
> "The absence of comparisons for classification performance is particularly concerning and unreasonable."

**Why it matters for AdaIB:**
- Discuss existing VLM attribution methods in detail
- Discuss how AdaIB differs from information bottleneck literature
- Position relative to data quality/robustness work
- Cite recent related papers (2024-2025)

---

## Weakness Pattern 7: Presentation & Clarity

### 7A. Method Clarity Issues

**Evidence Quote 1 - CoT (rpbzBXdo4x, Reviewer 1):**
> "The main text would benefit from more detailed task descriptions & exposition beyond Figure 1. E.g. for the grammar task, I was initially quite surprised that CoT resulted in worse performance there, but got a lot less surprised once I saw task examples & prompts in the appendix."

**Evidence Quote 2 - Disentangled (0iAZYF9hrl, Reviewer 3):**
> "The paper's presentation suffers from numerous issues that impede readability and clarity... Sections like Section 2.2 resemble output generated by ChatGPT and lack rigorous academic polish... Figures appear low-resolution, with inadequate explanations in captions."

**Evidence Quote 3 - Multi-hop (L9j8exYGUJ, Reviewer 4):**
> "Figure 4 does not include legends for the plots, making them hard to interpret."

**Why it matters for AdaIB:**
- Provide concrete examples of misaligned image-text pairs
- Use clear algorithmic descriptions (pseudocode)
- Include high-quality visualizations of attribution masks
- Make figures self-contained with comprehensive captions

---

## Weakness Pattern 8: Generalization Claims vs. Evidence

### 8A. Overstated Generality

**Evidence Quote 1 - SEED Tokenizer (IqGVIU4rvM, Reviewer 1):**
> "Lack of novelty or interestingness - On a high level, the paper mainly augments the SEED tokenizer with an additional low-resolution copy of the input image. The choices made for different component of this pipeline rather made it much more complicated to be useful in practice."

**Evidence Quote 2 - Disentangled (0iAZYF9hrl, Reviewer 2):**
> "Given the lack of compelling insights, this work appears to be primarily an application of existing DRL methods without significant methodological or theoretical innovation. This level of contribution may not align with ICLR's focus on novel methodological and theoretical advances in machine learning."

**Why it matters for AdaIB:**
- Avoid claiming "adaptive" unless truly tested on diverse conditions
- Acknowledge which VLMs and datasets are covered
- Be honest about limitations of current approach
- Scope title and abstract to match evidence

---

## Cross-Cutting Themes

### Theme 1: Limited Experimental Rigor
Many papers fail to:
- Test on multiple datasets and models
- Provide proper baselines
- Include ablation studies
- Report confidence intervals or statistical significance

### Theme 2: Gap Between Theory and Practice
Many papers:
- Make theoretical claims unsupported by analysis
- Lack rigorous definitions of key concepts
- Don't validate core assumptions empirically

### Theme 3: Reproducibility Concerns
Many papers:
- Lack implementation details
- Don't report hyperparameters clearly
- Provide insufficient dataset documentation
- Miss important experimental settings

### Theme 4: Evaluation Appropriateness
Many papers:
- Use arbitrary evaluation metrics
- Don't justify metric choice
- Mix evaluation paradigms that aren't comparable
- Lack ground truth or human validation


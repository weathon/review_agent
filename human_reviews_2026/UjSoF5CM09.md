# AttTok: Marrying Attribute Tokens with Generative Pre-trained Vision-Language Models towards Medical Image Understanding

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 4

## Abstract
Recent generative pre-trained vision–language (GPTv) models have achieved remarkable success in multi-modal understanding, inspiring their adaptation to medical imaging tasks such as disease diagnosis and visual question answering (VQA). However, current instruction-tuned GPTv models suffer from two key challenges: (1) medical attributes (e.g., disease names, severity grades) are encoded as plain text tokens, collapsing semantically distinct concepts into nearly identical textual sequences; and (2) inadequate textual supervision weakens visual representation learning, leading to severe inter-attribute confusion and misaligned vision–language embeddings. To address these limitations, we introduce attribute tokens (AttTok), a set of pre‑defined special tokens that uniquely encode clinical attributes (e.g., imaging modality, diagnosis, severity) within a structured token space. Complemented by attribute‑centric embedding books, AttTok serves as anchor points for aligning both visual and textual modalities into a shared, discriminative representation space. Building on this foundation, we design two key components: an attribute‑centric cross attention (ACC) adapter, which breaks the vision‑to‑text information‑flow bottleneck and enriches the visual encoder with discriminative attribute knowledge, and an attribute‑centric matching (ACM) loss, which enforces robust multi‑modal alignment centered on the attribute tokens. Extensive experiments on five medical classification benchmarks and three VQA datasets demonstrate that AttTok substantially improves both discriminative accuracy and medical knowledge reasoning, establishing a new paradigm for medical GPTv models with clinically discriminative understanding.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This study focuses on enhancing the fine-grained attribute discrimination capabilities of pre-trained GPTv models through "instruction tuning" (i.e., post-training). It addresses a limitation where existing GPTv models struggle with semantically distinct, fine-grained medical concepts due to their encoding as plain text tokens, leading to diminished discriminative power and misaligned vision-language embeddings. To overcome this, the authors propose **AttTok**, a framework that introduces specialized attribute tokens and associated attribute-centric embedding books for unique clinical attribute encoding. AttTok further integrates ACC adapter to infuse "discriminative attribute knowledge" into the visual encoder, and ACM loss to enforce robust "multi-modal alignment" centered on these attribute tokens. Extensive experiments across five medical classification benchmarks and three VQA datasets consistently demonstrate the effectiveness of AttTok. However, the analysis does not sufficiently clarify whether the observed performance gains genuinely stem from its attribute-centric design.

### Strengths
AttTok addresses the practical limitation that current GPTv models struggle with fine-grained medical attribute recognition, which is an important component of clinical reasoning. 

- The proposed attribute-centric post-training strategy offers an efficient way to improve attribute sensitivity without modifying the base architecture. 

- The combination of attribute tokens, embedding tables, an ACC adapter, and ACM loss is well motivated and leads to consistent performance gains across multiple medical benchmarks.

### Weaknesses
1. Ambiguity in Baseline Configuration: It is not clearly stated whether the "Instruction-Tuned GPTv Models" baselines (e.g., Lingshu-7B (r=16)) are trained on the exact same datasets as AttTok-enhanced models, but without AttTok’s attribute modeling components (attribute tokens, ACC adapter, ACM loss). Without explicit confirmation, one cannot fully exclude the possibility that performance gains originate from differences in the post-training setup rather than AttTok’s specific contributions.

2. Lack of Dedicated "Fine-grained Attribute Discrimination" Evaluation: Although AttTok aims to capture subtle inter-attribute distinctions, the paper does not provide a quantitative evaluation that directly assesses fine-grained discrimination capabilities. More detailed analyses such as confusion matrices or direct comparisons of highly similar clinical attributes (e.g., mild DR vs. severe DR) would help verify that AttTok truly improves sensitivity to fine-grained variations.

3. Limited Qualitative Evidence and Visual Insights: The paper lacks qualitative demonstrations that illustrate how AttTok influences attribute-aware perception within the visual encoder. Visualizations such as heatmap examples showing shifts in attention guided by attribute tokens could provide more interpretability. Case studies highlighting where baseline models fail but AttTok correctly identifies fine-grained findings would further reinforce the method’s clinical relevance.

### Questions
N/A

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes AttTok, a unified framework to enhance medical GPTv models through learnable attribute tokens and embedding books that explicitly encode clinical concepts. It introduces an Attribute-Centric Cross Attention (ACC) adapter and Attribute-Centric Matching (ACM) loss to improve visual–text alignment, achieving consistent gains across multiple medical VQA benchmarks.

### Strengths
1) This manuscript clearly addresses the semantic collapse problem in medical GPTv models, motivation is clear.
2) The methodologically seems sound, with explicit formulation of attribute tokens, ACM loss, and prototype stabilization.
3) The end-to-end integration supporting both discriminative and generative paradigms.
4) Empirical evidence across diverse modalities and clear ablation validation supports the claims.

### Weaknesses
1) The approach builds on existing prototype-based and token-alignment ideas without substantial theoretical innovation, which limits its technical novelty 
2) The method resembles prototype-guided learning. The paper should clarify how the attribute-token design and embedding book differ from prototype representations.
3) The paper lacks details discussion on the effort, criteria, and consistency in defining attributes, e.g., whether they are disease-specific or dataset-dependent. Furthermore, attribute tokens may encode explicit cues, inflating results.
4) Missing comparisons / discussions of the difference with existing medical multimodal models, such as BioMedCLIP and MedCLIP.
5) ACM loss lacks formal analysis or guarantees on improved alignment.
6) Evaluation is limited to classification and closed-form VQA, may perform additional tests on open-ended or noisy data.
7) Minor - next-token prediction paradigm (Radford et al.), lack of publication year.

### Questions
A few questions below, details refer to weakness.
1) How does AttTok differ conceptually and technically from prototype-based learning methods?
2) How are attribute tokens and embedding books defined—disease-specific or dataset-dependent—and what effort is required to construct them?
3) How sensitive is the model to attribute definition errors or GPT-generated keyword noise?
4) Can the authors provide ablation or scalability analysis on key hyperparameters (λ, γ) and large attribute sets?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the failure of generative vision-language (GPTv) models in medical imaging, where encoding clinically distinct attributes (e.g., "mild" vs. "severe" disease) as similar plain text causes semantic ambiguity and poor vision-language alignment. The proposed solution, AttTok, introduces a set of pre-defined, special "attribute tokens" to uniquely represent these clinical concepts. This framework uses these tokens as anchors in two new components: an Attribute-centric Cross Attention (ACC) adapter to enrich the visual encoder with attribute knowledge, and an Attribute-centric Matching (ACM) loss to enforce robust multi-modal alignment via contrastive learning. Trained jointly with the standard next-token prediction loss, experiments across five medical classification and three VQA datasets demonstrate that AttTok substantially improves discriminative accuracy over baselines.

### Strengths
1. The paper clearly identifies and targets a core weakness of medical GPTv models: the poor discriminative ability of plain-text encodings for clinically distinct attributes.
2. The proposed solution is well-designed. The two main components, ACC and ACM, directly target the two identified weaknesses (information bottleneck and misalignment). The ACC adapter is an effective mechanism to inject top-down attribute knowledge into the visual encoder . The ACM loss is a well-founded contrastive objective to enforce multi-modal alignment
3. The evaluation is moderately thorough. The authors test on varied datasets covering diverse medical modalities. 
4. The analysis in Table 5, showing that AttTok improves F1-scores for low-sample categories on an imbalanced dataset, is a strong point.

### Weaknesses
1. This is the most significant weakness. In Table 1, the average results for the Lingshu-7B + Ours model show a decrease in closed-ended accuracy (from 76.9% down to 71.6%) while improving open-ended accuracy (from 67.2% up to 80.6%). This directly contradicts the text, which claims "consistent and significant gains". This discrepancy is also seen in individual results (e.g., Lingshu-7B on Derma, where 'open' drops, and on Fundus, where 'close' drops). This confusion undermines the central claims and must be clarified.
2. The entire framework hinges on a set of "pre-defined special tokens". This raises critical questions about scalability. The method for defining attributes for VQA (using a GPT to extract keywords) seems ad-hoc and potentially brittle; the paper itself calls these attributes "relatively coarse". It is unclear how this approach would scale to thousands of fine-grained attributes or complex attribute combinations. The attribute books $\mathcal{B}$ would become very large, potentially making the ACC cross-attention and ACM matching computationally expensive.
3. As the authors acknowledge, the performance improvements on the three VQA benchmarks are modest (e.g., +0.8% average gain for Lingshu-7B). While the provided explanation (the baseline is already heavily pretrained) is reasonable, it limits the demonstrated impact of AttTok on tasks requiring more complex reasoning beyond classification.
4. The paper focuses on tasks with short, conclusive answers (classification and VQA). The authors concede in Appendix A.4 that the method's utility for long-text generation or chain-of-thought reasoning is "largely unexplored". It is unknown if adding the strong discriminative ACM loss harms the generative fluency, coherence, or nuance of the base LLM.

### Questions
1. Can you please address the apparent contradiction in Table 1? For the Lingshu-7B model, your method improves the average open-ended accuracy (67.2% $\rightarrow$ 80.6%) but seems to harm the closed-ended accuracy (76.9% $\rightarrow$ 71.6%). This contradicts the claim of "consistent improvements". Is this a typo in the table, or does AttTok introduce a trade-off between closed-ended and open-ended performance for this specific model?
2. The method relies on pre-defining $K$ attribute tokens. How do you see this approach scaling to a much larger and more complex attribute space (e.g., thousands of findings, locations, and modifiers)? Would the ACC module and ACM loss remain effective and computationally tractable as $K$ increases significantly?
3. The process for defining VQA attributes uses a GPT model to extract and assign keywords . How robust is this? Did you manually verify the quality of these extracted attributes? Given the "coarse" nature of these attributes, do you believe the modest gains on VQA are a limitation of the AttTok method itself or a result of noisy/imprecise attribute definitions?
4. Your method adds a strong discriminative objective (ACM loss) to the generative NTP loss. Did you perform any qualitative or quantitative analysis on whether this discriminative pressure harms the model's generative fluency, coherence, or its ability to generate nuanced, long-form explanations?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes AttTok, a framework that introduces attribute tokens (e.g., <|fundus_sdr|>) and a multi-modal attribute dictionary (embedding book) containing textual keywords, learned embeddings, and EMA-updated visual prototypes.
Two main modules are designed:

* Attribute-Centered Cross-Attention (ACC): Allows visual features to cross-attend to attribute embeddings, breaking the traditional unidirectional “vision → text” flow in generative vision-language (GPT-v) models.

* Attribute-Centered Matching (ACM): A triplet-style objective aligning visual, textual, and attribute embeddings to improve intra-class consistency and inter-class separability.

### Strengths
* The attribute-token and dictionary act as explicit anchors; ACC and ACM target information flow and alignment respectively. The design is intuitive and easy to integrate into existing GPT-v pipelines (e.g., LoRA fine-tuning with frozen backbones).

* Demonstrates stable improvements across diverse medical modalities (fundus, dermoscopy, OCT, X-ray, pathology) and VQA tasks, showing strong generalizability and robustness even on advanced medical VL backbones.

### Weaknesses
* As the attribute space grows (rare diseases, fine-grained grades, comorbid combinations), the dictionary (~3K entries) and ACC computation may become computationally expensive. The paper lacks analysis of complexity, efficiency, or EMA prototype stability as K increases.

* For VQA, attribute keywords are extracted using Qwen-3. The paper does not assess label noise, propagation effects, or how mis-specified keywords may misguide alignment.

* Results mainly report accuracy without confidence intervals, AUROC, calibration, or significance testing. The use of LLM-based semantic scoring for VQA may introduce bias.

* It remains unclear how introducing attribute tokens affects text fluency, interpretability, or controllability, especially when encountering unseen or composite attributes.

### Questions
* Can new attributes (e.g., new diseases or grading levels) be added incrementally to the dictionary without retraining the entire model? How is representation drift prevented? 

* Does AttTok handle combinations (e.g., “chest X-ray + mild infiltration + suspected heart failure”) via atomic tokens or compositional embeddings? How is combinatorial explosion managed?

* Since VQA accuracy is determined via LLM-based semantic matching, has the team validated with human or rule-based evaluation? How consistent are scores across evaluators?

* How does AttTok behave under domain shifts, image noise, or out-of-distribution (OOD) cases? Can attribute tokens support uncertainty estimation or refusal to answer?

### Soundness
3

### Presentation
2

### Contribution
2

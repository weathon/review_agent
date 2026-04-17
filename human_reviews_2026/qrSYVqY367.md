# ERA: Evidence-Based Reasoning and Augmentation for Open-Vocabulary Medical Vision

- Decision: Reject
- Scores: 2, 4, 4, 2

## Abstract
Vision-Language Models (VLMs) have shown great potential in the domain of open-vocabulary medical imaging tasks. However, their reliance on implicit correlations instead of explicit evidence leads to unreliable localization and unexplainable reasoning processes. To address these challenges, we introduce ERA (Evidence-Based Reasoning and Augmentation), a novel framework that transforms VLMs from implicit guessers into explicit reasoners for medical imaging. ERA leverages Retrieval-Augmented Generation (RAG) and Chain-of-Thought (CoT) to construct a traceable reasoning path from evidence to results. This framework requires no additional training and can be readily applied on top of any existing Vision-Language Model. Evaluated across multiple challenging medical imaging benchmarks, ERA's performance is comparable to fully-supervised specialist models and significantly surpasses current open-vocabulary baseline methods. ERA provides an effective pathway for building reliable clinical Vision-Language Models.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents ERA, a novel framework addressing the critical need for reliable, evidence-based reasoning in medical VLMs. The proposed training-free approach, which synergizes Retrieval-Augmented Generation (RAG) and Chain-of-Thought (CoT), is well-motivated and technically sound. The framework demonstrates impressive robustness, achieving state-of-the-art zero-shot performance and even outperforming supervised specialists on complex tasks where other generalist models fail completely. However, this strong performance comes at the cost of extremely high inference latency, which is orders of magnitude slower than baselines. While the results are strong, the practical viability is limited by this efficiency trade-off.

### Strengths
- **Important Problem:** The work tackles a critical and timely limitation of modern VLMs: their lack of reliability and interpretability, which is a major barrier to adoption in high-stakes domains like medicine.
- **Novel Methodology:** The framework’s design, which combines RAG and CoT to enforce explicit, evidence-based reasoning in a *training-free* manner, is a novel and elegant approach. It smartly redesigns the inference process rather than relying on costly model fine-tuning.
- **Exceptional Robustness:** The empirical results are strong, particularly in complex scenarios. ERA maintains robust performance on MSD tasks where powerful zero-shot baselines (e.g., YOLO-World, Grounding DINO) suffer a "catastrophic performance collapse" with near-zero scores.

### Weaknesses
- **Computational Cost:** The most significant weakness is the inference efficiency. As shown in Table 3 and acknowledged in Section 4.3, ERA's inference time is orders of magnitude higher than the baselines. This severe latency is a major barrier to practical clinical deployment.
- **Knowledge Base Dependency:** The framework's effectiveness is fundamentally dependent on the quality and comprehensiveness of the external knowledge base (K). The main paper provides limited detail on its scale and composition (deferring to Appendix C). It is unclear how the model performs on "out-of-distribution" queries that lack close exemplars in the knowledge base.
- **Unclear Baseline Configuration:** In Table 1 and Table 2, results are shown for both "ERA + SAM2" and "ERA + MedSAM". The baseline "MedSAM"  scores exceptionally poorly in Table 1, which is highly counter-intuitive for a model specialized in medical segmentation. This discrepancy is not explained and makes it difficult to assess the true contribution of ERA when paired with MedSAM.
- **Missing Ablation of Reasoning Policies:** The Deliberative Reasoning Engine (Sec 3.3.2) uses a crucial three-policy CoT (Adopt, Search, Zero-shot). However, the paper does not include an ablation study on the impact of these policies. Analyzing the activation frequency of each policy or the performance impact of removing one (e.g., always adopting evidence) would significantly strengthen the validation of this core component. No direct evidence found in the manuscript.

### Questions
See Weakness.

### Soundness
2

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
4

### Summary
This work leverages an external database from MedIMeta to construct a knowledge base for an RAG system in medical tasks. The image, text and corresponding bounding box annotations are retrieved and a multi-stage prompting strategy is designed to decide whether or not to leverage this retrieved image/bounding box locations in the prompt of another multimodal large language model. Experiments are conducted in multiple datasets to evaluate the performance.

### Strengths
- This work explicitly incorporates the evidence as its reasoning base in a multimodal large language model’s setting.
- A multi-stage prompting strategy is designed to better incorporate the retrieved information.
- It’s interesting to retrieve spatial annotations with the images to improve the localization capability.

### Weaknesses
*Contribution* 
-  The contribution is overclaimed. [line 82] It’s at most encouraging the usage, instead of forcing it to check evidence.

-	[Line 96] it’s hard to say this framework “transforms the decision process into a transparent process”. Any reasoning model can achieve this.

-	[line 53] Claiming “CoT is traceable” is too strong. It at most offers some insights. Whether the generated chains are correct requires expert review and whether CoT is the real trace is also arguable. 

*Implementation* 

Many implementation details are missing
- [Line 187] what is a spatial prompt? What text and what image is retrieved?
- [Line 222] how exactly is a a query made of an image and a text encoded? What is the text in the knowledge base /where do they come from? Simply class name?
- [Line 209] many medical images are not annotated with bounding box or segmentation masks. Are you retrieving the whole image or only the cropped part?
- How do you handle resolution, ratio difference between retrieved and tested images?

*Method and Data*

- [Table 8] it seems directly applying groundingDINO is better than the proposed approach? Since this framework is training-free, can this approach be combined with groundingDINO? If not, why do people not directly apply GroundingDINO?

-	[line 21] It’s unclear whether the compared specialist model is trained on the same external database used to construct RAG. If not, the comparison is unfair. 

-	Is there any data leackage checked? Does the database potentially include the test set datasets?

### Questions
- [line 252] Why is it necessary to check the ground-truth bounding box of retrievd images again?

- [Line 259] Under what scenario will it happen that a bounding box from the knowledge base’s image is directly useable in the tested images? This design seems somewhat unreasonable to me. 

- How is SAM2 used in this framework? Usef to offer the bounding box of the tested image or the retrived image?

- [line 771] Why is it necessary to compute another bounding box as “geometric anchor” when ground-truth mask is already present?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces ERA, an evidence-based framework that enables vision–language models to perform explicit reasoning for medical imaging tasks. It encodes images and text using BLIP2 and leverages retrieval-augmented generation (RAG) to search through a large, encoded knowledge base of medical imaging datasets to identify relevant items, which are then provided as input to the deliberative reasoning engine. The core of this reasoning engine employs a chain-of-thought (CoT) strategy, enabling the vision–language model Qwen to output a spatial prompt. The authors evaluated ERA across a diverse set of datasets, including ISIC 2018, MSD, and BraTS 2021, and compared its performance against various baseline model architectures.

### Strengths
- Appreciate the vision to develop evidence-based frameworks, addressing a crucial problem with an innovative solution. The integration of RAG and CoT techniques within a tiered decision engine effectively incorporates relevant evidence for downstream tasks like segmentation.

- The evaluation is comprehensive, encompassing both quantitative and qualitative analyses across diverse datasets and benchmarking models on 2D and 3D tasks.

- Also appreciate the inclusion of a computational efficiency analysis, which adds important practical insight to the study.

- Well presented qualitative results.

### Weaknesses
- Methodology: 
   - RAG details needed: 
     - Further clarification is needed regarding the size and composition of the medical knowledge base. It would be valuable to motivate the rationale for using a combined database of CT, MRI, and X-ray images for tasks focused on a single imaging modality. For example, if the task involves CT-based segmentation, it may be more effective, and computationally efficient, to restrict retrieval to CT-only data, potentially improving both relevance and speed.
     - It would be important to provide more details on the Evidence Candidate set and, if possible, include a quantitative evaluation of the evidence retrieval. For example, assessing how many of the top-k retrieved items match the query modality or anatomy (e.g., if the query is a head CT, what proportion of the retrieved items are also head CTs) would offer valuable insight into the relevance and precision of the evidence search.
     - Why do you need an evidence candidate set of length 6 (number found in the appendix), when you are selecting the top-ranked exemplar? 
     - Should cite the database used in the main paper and not just in the appendix.
  - Chain-of-Thought Reasoning uncertainties:
The reasoning engine is based on the Qwen2.5 vision–language model. It would be valuable to indicate whether other vision–language models were tested or compared. 
- Evaluations: 
   - Why are the Zero-Shot Generalist Models evaluated with only SAM2 and not with MedSAM? (Tables 1 and 2)
  - Which tier within the tiering strategy was most used during evaluation? Should have those numbers quantified to provide a better understanding of how effective RAG is.

- In the contributions, the authors state: “By generating a clear reasoning path, it builds a foundation of trust for AI in high-stakes clinical settings.” It would be important to validate this claim through clinician evaluation, as interpretability alone does not necessarily translate to trust. Assessing whether ERA genuinely enhances clinician understanding and confidence would strengthen this contribution.

- Title is too general: Since the task is evaluated is only segmentation, maybe consider “...open-vocabulary medical vision segmentation” 

- Minor: There are many cases where acronyms are being re-defined (VLMs, SAM, RAG, CoT, ERA). Ensure that these acronyms are being defined once. 

- Typo: Table 8: Prostate, groundingDINO IoU = 8.8, should be 0.88

### Questions
Table 2: The evaluation with Zero-Shot Generalist Models, many of the other baselines exhibit surprisingly dismal performance. Is this real? Can you explain why that is the case?

Fig.2: 
- What are the weights (w1 and w2) and why is it used?
- Why are the text encoder and image encoder shapes flipped?
- If only the top exemplar (E*) is used, then why are “k exemplars” present in Tier 1 and Tier 2?
- What is the memory bank and memory attention?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces ERA, a zero-shot framework to address the unreliability of Vision-Language Models (VLMs) in medical imaging. It combines Retrieval-Augmented Generation (RAG) to source visual evidence from an external medical knowledge base with a Chain-of-Thought (CoT) module to force the VLM to validate this evidence before making a decision . The authors claim this "evidence-based reasoning" approach significantly outperforms other zero-shot baselines and can even match fully-supervised specialist models in certain tasks.

### Strengths
1. The paper presentation is easy to understand.
2. The motivation is clear: VLM can use external knowledge for boosting performance on special domains.
3. The qualitative examples in Figure 1 and Figure 3 do show a clear visual improvement over the catastrophic failures of generalist baselines like FG-CLIP and YOLO-World.

### Weaknesses
1. Missing baselines and experiments: The experimental comparison is not enough. The proposed ERA framework uses Qwen2.5-VL-7B as its reasoning engine, but it is compared against non-MLLM, open-vocabulary detector models like YOLO-World and Grounding DINO. 
A proper baseline would be to use the same Qwen-7B model with a simpler prompting strategy (e.g., a simple kNN-retrieval prompt), which is missing. 
Also, it is unknown if this framework would work at all with other models like LLaVA, InternVL, making the generality of the framework contribution unproven.
2. "Zero-Shot" claim: The paper claims a "zero-shot framework" but Appendix B.2 and Table 6 state that the tier1_similarity and tier2_similarity thresholds are "presented as effective ranges"and that "the optimal value is determined on a validation set for each domain." This is, by definition, a tuning process that requires labeled validation data for each new task/domain, and is not zero-shot, and possibly not fair for the comparisons on other methods. I wonder what would the performance be if we set a single threshold for all tasks.

### Questions
1. In your figure 4, for each tier there is workflow with many steps. Is the workflow being done as multi-round conversation, or just one round?
2. Do you have failure cases on VLM query? The model may not output exactly as your json format. How will you handle that?
3. Can the authors justify the 3.2-hour inference time for the Hippocampus dataset? What specific part of the pipeline (retrieval, or the CoT deliberation) causes this massive bottleneck?
4. For table 4, what is the difference of "w/o Retrieval" and "w/o Tier2"? My understanding is w/o Retrieval means NO retrieval at all (Tier 3 only). The ablation w/o Reasoning means retrieval happens, but only the simple Tier 1 logic is used.

### Soundness
3

### Presentation
3

### Contribution
3

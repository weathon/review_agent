# KEPIL: Knowledge-Enhanced Prompt-Image Learning for Prompt-Robust Disease Detection

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 6, 4, 4

## Abstract
Vision–language models (VLMs) show promise for clinical decision support in ra-
diology because they enable joint reasoning over radiological images and clinical
text, thereby leveraging complementary clinical information. However, radiologi-
cal findings are long-tailed in practice, leaving some conditions underrepresented
and making zero-shot inference essential. Yet current CLIP-style medical VLMs
are sensitive to prompt variations and often lack trustworthy external knowl-
edge at inference time, which hinders reliable clinical deployment. We present
KEPIL, a prompt-robust framework that integrates curated medical knowledge
to stabilize zero-shot generalization. KEPIL comprises: (i) dynamic prompt en-
richment using ontologies with LLM assistance, (ii) a semantic-aware contrastive
loss aligning embeddings of equivalent prompt variants via a dual-embedding ob-
jective, and (iii) entity-centric report standardization to yield ontology-aligned
representations. Across seven benchmarks, KEPIL achieves state-of-the-art zero-
shot/finetuning performance in classification and segmentation; under prompt-
variation tests, it improves AUC by 6.37% on CheXpert and by 4.11% on average.
Ablations and qualitative analyses validate the contributions of enriched prompts
and semantic alignment, while attention maps highlight clinically relevant regions.
These results show that structured knowledge and robust prompt design are key to
clinically reliable radiology-facing VLMs. Code will be released at ***.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes KEPIL, a knowledge-enhanced vision-language framework aiming to improve prompt robustness and zero-shot generalization in medical imaging tasks. It integrates curated ontologies (UMLS, Radiopaedia) with LLM-generated descriptions, introduces a semantic-aware contrastive loss (Lsc) to stabilize embeddings across prompt variants, and standardizes radiology reports via entity-centric preprocessing. Experiments on seven public chest X-ray datasets show improved zero-shot and finetuning performance compared to CLIP-style and medical-specific baselines.

### Strengths
- Addresses a practically relevant issue: prompt sensitivity and lack of knowledge grounding in medical VLMs.

- Combines knowledge curation and contrastive learning in an interpretable manner.

- Provides comprehensive experiments with both quantitative and qualitative analyses.

- Demonstrates cross-modality transfer and prompt perturbation robustness, which are meaningful for clinical deployment.

### Weaknesses
- The work is largely incremental and engineering-oriented. Its main components—ontology-guided prompt design, adapter alignment, and cross-attention fusion—closely follow prior work such as KAD, MAVL, and MedKLIP. The proposed “semantic-aware contrastive loss” is a simple consistency objective, and the “Knowledge Query Module” mainly reuses standard cross-attention. Overall, the paper does not introduce substantial novelty or new conceptual understanding of prompt robustness or knowledge integration.

- The paper also relies heavily on ChatGPT-4o for generating and refining prompts. While this provides flexibility, it introduces potential issues with factual accuracy and consistency, as large language models can produce hallucinated or unstable medical text. The paper does not include verification or expert validation of these outputs. Since the generated text directly affects training, it is unclear whether the reported gains stem from true model robustness or from uncontrolled variations in GPT-generated data.

- The experimental analysis lacks statistical rigor. The paper reports no standard deviations, repeated trials, or significance testing. The reported 1–3% performance gains may fall within normal variance. In addition, the prompt robustness experiments mainly test minor typos rather than diverse or semantically rephrased prompts, providing limited evidence of genuine robustness.

- The claims of clinical generalization appear overstated. All experiments are conducted on public datasets rather than real-world or prospective clinical data. The CXR-to-CT transfer is a simplified setting that does not demonstrate true cross-modality adaptation. Without human or expert validation, the claim of “trustworthy clinical deployment” is not sufficiently supported.

### Questions
Please refer to the Weaknesses section.

**I am willing to raise my score according to the rebuttal.**

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The study introduces **KEPIL** (Knowledge-Enhanced Prompt Image Learning), a framework designed to improve VLMs for medical disease detection in radiology. The paper includes 2 main contributions:  **Knowledge-Grounded Prompt Enrichment**, and  **Semantic-Aware Contrastive Loss**. The proposed loss function make the model robust to prompt variations. It uses a dual-embedding objective to align the embedding of equivalent prompt variants, teaching the model that different phrasings of the same concept are related. **KEPIL** achieved SOTA performance in zero-shot setting classification. It demonstrated robustness to prompt variations.

### Strengths
1) The proposed method re-defines prompt sensitivity problem in medical VLMs as a knowledge  alignment problem, aligning ontology-grounded text with visual features to stabilize predictions.

2) Demonstrating robustness to real prompt noise, improving performance on rare/unseen diseases, indicates meaningful impact  in clinical setting.

2) The study demonstrates consistent in zero-shot settings and smaller drops under prompt perturbations then competing baselines.

### Weaknesses
1) I have one concern about the improvement from the vision encoder being pretrained on chest X-ray data. The current gains might partially reflect this pretraining rather than the proposed knowledge components.

2) I am very confused because the experiment setting section mentions SIIM-ACR for the task of segmentation, but I could not find relevant report for this dataset in segmentation setting.

3) The study generates prompt variants including rephrasing, typos, omissions, and incorrect punctuation. However, it is not enough for realistic clinical settings such as abbreviations, multilingual terms, or clinician-specific jargon.

4) The study's introduction emphasis on long-tail distribution, but the results are not depicted the results for rare diseases. The results mainly focus on CXR diseases.

### Questions
1) Could you provide experimental results to clarify the concern (1) in the **weakness** section.

2) Could you provide experimental results on a rare disease dataset to support the paper claims?

3) Could you provide experimental results to clarify the concern (3) in the **weakness** section.

4) Could you provide additional subsection to explain the results for the segmentation task?

5) How does the Knowledge Query Module (KQM) enhance localization compared to baseline?

### Soundness
3

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
This paper proposes KEPIL (Knowledge-Enhanced Prompt–Image Learning) to mitigate two challenges of medical-imaging VLMs in zero-shot settings: prompt sensitivity and lack of external domain knowledge. The method has three key components: (i) ontology-constrained prompt expansion and standardization using external medical knowledge bases (e.g., UMLS, Radiopaedia) with LLM assistance; (ii) a semantics-aware contrastive loss that enforces representation consistency across text-side views via a lightweight adapter with dropout; and (iii) entity-centric report normalization using RadGraph to reduce free-text noise. The architecture uses a frozen clinical text encoder (e.g., BioClinicalMPBERT) with a trainable adapter, a ViT-B/16 visual encoder, and a Knowledge Query Module (KQM) for token-level cross-attention alignment between image patches and text tokens. Experiments cover seven chest X-ray benchmarks (classification/segmentation/localization), including seen/unseen/rare categories and cross-modality transfer (CXR→CT). KEPIL outperforms or matches strong baselines and exhibits reduced performance degradation under diverse prompt perturbations; ablations attribute gains primarily to knowledge enrichment and the proposed loss.

### Strengths
- Strong empirics across seen/unseen/rare and cross-modality settings; consistent zero-/few-shot gains, including with limited labels for segmentation.
- Robustness focus is carefully evaluated with multi-source LLM-generated and perturbed prompts; UMAP suggests tighter intra-class clusters.
- Interpretability: entity-centric text and Radiopaedia cues produce attention maps that align with clinical findings.

### Weaknesses
- Train–test gap in semantic alignment: The loss aligns two stochastic views of the same text, not explicit cross-variant positives (paraphrases, synonym mappings). Training with explicit variant pairs would better match the robustness claim.
- Theory is light: “Semantic-aware” is broad; a perspective via invariance subspaces, information bottleneck, or generalization bounds maybe would strengthen the conceptual grounding.

### Questions
- What is the size of the entity set E and its coverage per disease category? How are conflicts between UMLS and Radiopaedia resolved; what fraction is human-audited?
- Did you train with explicit paraphrase/synonym/noisy variant pairs as positives? If not, can you add such a loss and report gains vs. the current two-view objective? I think by incorporating cross-variant positive pairs during the training phase could further support the claim of being "variant-robust". 
- At inference, do you require LLM calls to generate prompts, or rely on a pre-built normalized library? 
- In robustness plots, are max token length and template structure matched across models?

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
This paper introduces KEPIL, a knowledge-enhanced prompt-image framework designed to address the issues of prompt sensitivity and limited generalization in medical vision-language models. It incorporates a large amount of medical knowledge based on ontologies and leverages dynamic prompt enhancement guided by large language models (LLMs) to improve understanding. In addition, a semantics-aware contrastive loss is proposed to enhance prompt robustness, and entity-centered report standardization is employed to optimize information representation. Experiments on seven benchmark datasets demonstrate that KEPIL achieves state-of-the-art performance in zero-shot classification and segmentation tasks.

### Strengths
This work is the first to integrate medical ontological knowledge, dynamic prompt enhancement, and semantics-aware contrastive learning to improve prompt robustness in medical vision-language models. With comprehensive experiments across multiple datasets and tasks, it demonstrates clear methodology and achieves significant performance gains, highlighting its practical value for medical AI.

### Weaknesses
1.The LLM-based prompt enrichment lacks transparency and rigorous validation against raw knowledge sources, risking unquantified hallucinations.

2.The superior segmentation scores lack qualitative validation (e.g., mask visualizations), leaving the clinical precision of improvements unproven.

3.​​The complex inference-time prompts increase computational overhead, but efficiency (latency) is not benchmarked, hindering practicality assessment.

### Questions
1.Provide radiologist-evaluated proof that LLM-enriched prompts are more clinically valuable than raw knowledge-base text.

2.​​Was robustness tested beyond typos (e.g., clinical synonyms like "opacity" vs. "consolidation")?

​3.​Why was dropout chosen over explicit text augmentation for creating positive pairs in the contrastive loss?

​​4. Is performance on rare diseases due to unique feature learning or merely semantic proximity to common diseases in the knowledge graph?

If my main concerns are properly addressed, I would be willing to raise my evaluation.

### Soundness
3

### Presentation
3

### Contribution
3

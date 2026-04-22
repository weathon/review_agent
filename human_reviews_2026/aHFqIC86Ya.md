# CortiLife: A Unified Framework for Cortical Representation Learning across the Lifespan

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 4, 6, 6

## Abstract
The human cerebral cortex encodes rich neurobiological information that is essential for understanding brain development, aging, and disease. Although various cortical representation learning methods have been proposed, existing models are typically restricted to stage-specific cohorts and lack generalization across the lifespan. While recent vision-language models offer a promising direction, building a unified framework for cortical representation faces three key challenges: (1) the non-Euclidean manifold structure of cortical surfaces, (2) homogenization of individual folding patterns induced by registration, and (3) distribution shifts of cortical features across the lifespan. To address these issues, we present CortiLife, the first unified vision-language framework for lifespan-aware cortical representation learning. Specifically, CortiLife introduces a surface tokenizer that integrates icosahedron-based surface patchification with multi-level patch encoding to transform complex cortical manifolds into compact token representations. The multi-level encoding incorporates three complementary streams that capture local topology, global interactions, and patch-wise distributional patterns, effectively mitigating the challenges of homogenization and distribution shifts. Furthermore, CortiLife integrates masked self-distillation with metadata language prompting, embedding information such as age, sex, health status, and attribution type into the text encoder to better capture individual-specific cortical representations while enabling both age-aware and modality-aware modeling. Extensive experiments on downstream tasks, including two encoder-frozen tasks (age prediction and cortical parcellation) and four encoder fine-tuning tasks (brain disorder diagnosis), demonstrate that CortiLife consistently outperforms state-of-the-art baselines across different age stages and modality types, underscoring its effectiveness and generalization ability.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses cortical representation learning across the lifespan, collecting an impressive dataset spanning 26 weeks to 95 years (13,928 subjects) and proposing CortiLife with multi-level surface tokenization. The experimental scope is comprehensive and results show competitive performance.

However, I have significant concerns about the claimed contribution. The paper positions itself as "the first lifespan-aware framework," but the evidence primarily demonstrates strong performance from CLIP trained on diverse age data rather than architectural innovation for cross-age generalization. Critical issues include: (1) Figure 4's circular reasoning (age/gender embeddings reflect explicit text inputs, not learned capability), (2) unclear whether baselines used the same pretraining data (unfair comparison?), (3) no direct evidence that scale-adaptive encoding addresses distribution shift vs. adding features, and (4) evaluation gaps (no adults 30-55, no per-age-group breakdown). The paper needs to clarify what specifically makes this lifespan-aware beyond data aggregation and provide ablations isolating architectural contributions. With revisions addressing these fundamental questions, this could become a solid contribution. Currently, the novelty and validity of claims are insufficient.

### Strengths
- Significant data collection effort (9 datasets, 13,928 subjects across lifespan)
- Comprehensive experimental scope (multiple tasks, modalities, settings)
- Extensive baseline comparisons
- Generally competitive empirical performance
- Technically sound surface patchification approach

### Weaknesses
**W1. Unclear Contribution Beyond SOTA Performance**
The paper claims to be "the first unified vision-language framework for lifespan-aware cortical representation learning," but the actual technical novelty separating this from standard multi-dataset training is unclear. The key question remains: Is this truly lifespan-aware architecture innovation, or simply a CLIP-based model trained on diverse age datasets? The performance gaps over baselines are often marginal (e.g., Table 3a: 0.806 vs 0.792 for CHD), raising concerns about whether the architectural components specifically enable lifespan generalization or if aggregate data is the primary factor.

**W2. Circular Reasoning in Figure 4**
Figure 4 visualizes embeddings separated by age, gender, and diagnosis - but these exact attributes are explicitly provided as text inputs via PubMedBERT (Section 2.3: "The age of the subject is [age]. The gender of the subject is [gender]..."). This is not evidence of learned representation quality but rather a trivial reflection of the input. The paper presents this as validation of the model's capability, which is misleading.

**W3. Distribution Shift Solution Lacks Validation**
The paper identifies distribution shifts across lifespan as a key challenge (Figure 1c) and proposes scale-adaptive encoding (Eq. 1-2) as the solution. However:

- The ablation study (Table 4) only shows that removing "statistical" encoding reduces performance
- No evidence that this specifically addresses distribution shift vs. simply adding more features
- No analysis showing that embeddings from different age groups are actually aligned in the representation space

**W4. Inconsistent Experimental Setup**

- Baseline models' pretraining data is not specified - did they use the same 8 datasets?
- Fine-tuning only evaluates 3 specific age groups (infancy, adolescence, elderly), missing adult ages 30-55
- Table 1 shows HCP covers 22-36 years, but it's only used for zero-shot evaluation, not fine-tuning
- The claim of "across the lifespan" is not systematically validated across all age ranges

**W5. Poor Presentation Quality**

- Related Work relegated to Appendix without justification (highly unusual)
- Inconsistent terminology: "ADHD" (disease) vs "ADNI" (dataset) mixed in Table 3 headers
- Acronyms (CHD, ADHD, ADNI) undefined until Appendix A.3, first appearing in main text without explanation
- Figure 2 is overly complex and difficult to parse

**W6. Unmotivated and Underexplored Teacher-Student Design**
The paper employs masked self-distillation with teacher-student framework (Section 2.2), masking 75% of patches based on attention scores. However:

- **No ablation study** comparing with/without this design (Table 4 doesn't include it)
- **No computational efficiency metrics** reported (training time, inference speed, memory)
- **No justification for 75% masking ratio** - why not 50%, 60%, or 80%? This appears borrowed from MAE without domain-specific validation
- The claimed motivation ("information redundancy") is not validated
- Appears to be standard MAE-style masking without cortex-specific justification

This is concerning because:

1. If efficiency is the goal → no speedup metrics provided
2. If better representation is the goal → no ablation showing improvement
3. The 75% ratio may be suboptimal for cortical surfaces where spatial anatomical structure matters

### Questions
**Q1.** What specific architectural components enable lifespan generalization beyond simply training on multi-age datasets? Can you provide ablation studies showing performance when age information is removed from text prompts?

**Q2.** Were baseline methods (CLIP, ACLIP, DetailCLIP, CARZero) pretrained on the same 8 datasets as CortiLife? If not, how is this a fair comparison?

**Q3.** Figure 4 shows embeddings clustered by attributes that were explicitly input. How does this demonstrate learned capability rather than input reflection? Can you show embeddings on data where age/gender were NOT provided as text input?

**Q4.** How does the scale-adaptive encoder (Eq. 1-2) specifically address distribution shift? Can you visualize embedding distributions across age groups to show alignment?

**Q5.** Why is fine-tuning only performed on 3 datasets spanning limited age ranges? What about performance on young/middle-aged adults (30-55 years)?

**Q6.** Table 3 shows many baseline methods with comparable performance. For example, in Table 3c (SD), WSSADN achieves 0.935 vs your 0.972 on ADNI - but WSSADN had no pretraining. How much gain is from pretraining vs. lifespan-aware architecture?

**Q7.** In zero-shot evaluation (Table 2), CARZero performs catastrophically poorly (DICE ~0.01). Was this implementation correct? This seems like an implementation error.

**Q8.** What is the specific purpose of the teacher-student masked self-distillation framework? The paper employs this design masking 75% of patches, but the motivation is unclear. Is this for computational efficiency (if so, where are the speedup metrics?), or for learning better representations (if so, where is the ablation comparing single-model vs. teacher-student)? Table 4's ablation study does not include this comparison. The paper cites "information redundancy" in cortical data, but provides no evidence that this design specifically addresses it. This appears to be standard MAE-style masking - what makes this cortex-specific?

**Q9.** Where is the computational cost analysis? If the student processes only 25% of patches during training, what is the actual speedup in training time, inference time, and memory consumption compared to processing all patches? Does this enable scaling to higher-resolution cortical meshes? Without efficiency metrics, the teacher-student design appears unmotivated - if there's no computational benefit and no representation quality benefit (per Q8), why use it?

**Q10.** Why specifically 75% masking ratio? Section 2.2 states "Patches with the lowest 75% attention values are masked out," but no justification or ablation study for this choice is provided. Was this borrowed from MAE (He et al., 2022) which used 75% for natural images? Cortical surfaces have fundamentally different properties - high spatial autocorrelation, fixed anatomical structure (motor cortex, visual cortex locations), and critical geometric features. Shouldn't the masking ratio be validated for this domain? Can you provide ablations testing 50%, 60%, 80%, or 90% masking to show 75% is optimal for cortical surfaces?

### Soundness
2

### Presentation
2

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
This paper presents a framework for extracting information from cortical mesh data. Among the information, there is the reconstruction of the cortical surface and metadata about the subject. The experiments are done using several classical datasets in neuroscience.

### Strengths
The framework extract much information from surface cortical mesh with very important metadata. The text is clear and the model well presented. The experiments show interesting results. While the model is not always the best, it is the most reliable. The potential impact of such in the neuroscience community is great since the cortical surface important information about the health of the subject.

### Weaknesses
I see several weaknesses in such paper.

- There exists strong fairness issues with cortical data, especially on the gender. Do such biases affect the model?
- I am not sure of the impact of such in the ICLR community. While such framework can have a large audience in the neuroscience community, it is not clear that the classical machine learning community will be intereted. Especially there exists now several frameworks of this kind in medical sciences (see [1]).
- Extracting the age and gender using a LLM seems a little overkill and misleading. Why not using a regression model?
- Similar frameworks have been proposed for other kind of medical imaging (for example, see [1, 2, 3]). What is the position of this model compared to these in terms of capabilities?

### References

- [1] Khan, W., Leem, S., See, K. B., Wong, J. K., Zhang, S., & Fang, R. (2025). A comprehensive survey of foundation models in medicine. IEEE Reviews in Biomedical Engineering.
- [2] Lu, M. Y., Chen, B., Williamson, D. F., Chen, R. J., Liang, I., Ding, T., ... & Mahmood, F. (2024). A visual-language foundation model for computational pathology. Nature medicine, 30(3), 863-874.
- [3] Zhang, K., Zhou, R., Adhikarla, E., Yan, Z., Liu, Y., Yu, J., ... & Sun, L. (2024). A generalist vision–language foundation model for diverse biomedical tasks. Nature Medicine, 30(11), 3129-3141.

### Questions
See the weakness section for the questions.

### Soundness
4

### Presentation
3

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
The authors present a method called CortiLife, a unified vision-language framework for lifespan-aware cortical representation learning. They design a surface tokenizer to capture local topology, global interactions, and patch-wise distributional patterns. During training, CortiLife uses masked self-distillation to avoid substantial information redundancy and metadata language prompting to embed extra information such as age, sex and so on.

### Strengths
- The method can handle data from any lifespan.
- The authors build a surface tokenizer and training framework tailored to cortical data.
- The method outperforms existing methods, indicating potential practical application value.

### Weaknesses
- In Table 2, CARZero’s DICE performance appears surprisingly poor—can authors provide an explanation?
- The “zero-shot” experiments seem to require additional training; doesn’t this resemble “pretraining + linear probe” rather than conventional zero-shot generalization?
- The authors could consider to add more experiments on ablation study—for example, varying the masking ratio and whether to include $L_{CLIP}$—to more comprehensively show each module’s marginal contribution.

### Questions
Three encoding types in Figure 2 can be drawn more specifically to facilitate readers' understanding

### Soundness
3

### Presentation
3

### Contribution
3

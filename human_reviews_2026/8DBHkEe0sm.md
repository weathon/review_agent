# Rethinking Cross-lingual Alignment: Balancing Transfer and Cultural Erasure in Multilingual LLMs

- Decision: Reject
- Scores: 8, 4, 6, 2

## Abstract
Cross-lingual alignment (CLA) aims to align multilingual representations, enabling Large Language Models (LLMs) to seamlessly transfer knowledge across languages. While intuitive, we hypothesize, this pursuit of representational convergence can inadvertently cause "cultural erasure"—the functional loss of providing culturally-situated responses that should diverge based on the query language. In this work, we systematically analyze this trade-off by introducing a holistic evaluation framework, the transfer-localization plane, which quantifies both desirable knowledge transfer and undesirable cultural erasure. Using this framework, we re-evaluate recent CLA approaches and find that they consistently improve factual transfer at the direct cost of cultural localization across all six languages studied. Our investigation into the internal representations of these models reveals a key insight: universal factual transfer and culturally-specific knowledge are optimally steerable at different model layers. Based on this finding, we propose Surgical Steering, a novel inference-time method that disentangles these two objectives. By applying targeted activation steering to distinct layers, our approach achieves a better balance between the two competing dimensions, effectively overcoming the limitations of current alignment techniques.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
Under the assumption that language represents culture, this paper investigates the dual effects of cross-lingual alignment on universal and culture-specific knowledge in LLMs. This paper proposes a holistic evaluation framework that comprehensively assesses how LLMs’ mastery of shared and culture-specific knowledge changes across languages before and after cross-lingual alignment. The results reveal a clear trade-off between the two. Furthermore, the paper examines the representation spaces and introduces a representation engineering (RE) approach, in which language alignment and cultural differentiation REs are applied at different layers. Taken together, these two investigations can provide insights for future research.

Two main limitations remain. First, the discovered trade-off is validated on only one model and a pair of datasets (one for universal knowledge and the other for culture-specific knowledge evaluation). Second, the assumption that language represents culture is overly strong, which further weakens the validity of the representation explorations.

### Strengths
- The topic (intersection of multilingual and culture) is practical; and the research question (relationship between cross-lingual alignment and cultural alignment) is intuitive and important.

- The methodology is straightforward and easy to follow: applying cross-lingual alignment (CLA) techniques to LLMs and evaluating changes in the models’ mastery of shared and culture-specific knowledge in multilingual contexts.

- The experimental results are intuitive: although CLA improves the cross-lingual transfer of shared knowledge, it leads to erasure of cultural-specific knowledge, revealing a clear trade-off between cross-lingual transfer and cultural preservation.

- The representation analysis in Section 5 is interesting — it identifies distinct layer distributions for shared versus cultural knowledge. The proposed representation engineering (RE) method based on this observation offers useful insights for mitigating this trade-off in future work.

### Weaknesses
- Despite testing different CLA methods and languages, the experiments are still limited: results are based only on a single model (Gemma3 12B), with shared knowledge evaluated solely on the GMMLU dataset and cultural-specific knowledge on BLEND.

- The paper adopts a strong language-as-culture-proxy assumption and removes explicit localization context from BLEND (Section 3). This assumption is debatable, yet its rationale and limitations are not explicitly discussed.

- The motivation behind the representation analysis in Sections 4.4 and 5 is insufficient. To directly test the question, “If CLA suppresses a model’s ability to use language as a cultural cue, is that knowledge permanently erased or merely inaccessible?”, the most straightforward approach would be to reintroduce explicit cultural localization cues in the prompts (as done by Veselovsky et al., 2025). Without such comparisons, the effectiveness of the proposed RE method is also not fully validated.

### Questions
- The GMMLU dataset does not appear to be a parallel multilingual dataset and includes cultural nuances — does this affect the validity of the results?
- It would be better if the representation similarity in Section 4.4 is quantitatively measured.
- How to further explain the phenomenon in Section 4.4? Is it just because BLEND datasets are more surface-level different across languages, or does the LLMs genuinely encode shared and cultural knowledge in different layers?
- Could more details be provided on EN-steering and LOC-steering? For example, from which datasets are they derived? (From Figure 4b, it seems both datasets were used for vector extraction)

### Soundness
2

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The study aims to disentangle knowledge transfer and cultural knowledge erasure. They propose an inference-time method to steer activations for knowledge and cultural information at different transformer layers. The study finds that middle layers work best for knowledge transfer, while cultural information is encoded in deeper layers.
Knowledge is categorized into universal knowledge (language agnostic knowledge) with the objective of best possible transfer and culturally-adaptive knowledge based on universal concepts but different instantiation through cultural norms, regulation etc., with the objective of best possible localization. The proposed method is evaluated on global mmlu and BLEND. The paper proposes SUR-steering method that applies localized activation steering inspired by related steering and representation alignment methods.

### Strengths
- The study is well motivated
-  The evaluation results of the proposed steering method show that localization can be achieved without negatively affecting global mmlu performance
- The paper makes a good contribution by providing a unified view on knowledge transfer and cultural knowledge erasure

### Weaknesses
- It's unclear how multilingual instruction-tuning, midalign, and steering are encouraging localization, as they map target languages to English representation space
- The proposed SUR-steering method lacks innovation as it applies existing steering to existing cross-lingual alignment methods. While the insights are interesting, performance differences are low for the best performing methods (MidAlign, CLO)
- The underlying technical mechanisms negatively affecting cultural knowledge are not discussed in detail

### Questions
- How could the benefits of SUR-steering be applied when training multilingual models?
- Knowledge localization extends beyond languages, i.e. user can ask for emergency number in a different country. How to adapt the method to localize knowledge beyond language?
  - Different languages are an arbitrary boundary for localized knowledge
- typo line 97

### Soundness
3

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
The paper investigates how cross-lingual alignment in multilingual models improves factual transfer but causes cultural erasure. Using the proposed transfer-localization plane, the authors reveal this trade-off across six languages. Moreover, they introduce Surgical Steering, an inference-time method that targets different model layers to better balance factual transfer and cultural localization.

### Strengths
1. This paper addresses a compelling topic—the trade-off between knowledge transfer and cultural localization—pioneering research in this area. By exploring this trade-off, multilingual alignment can achieve better semantic consistency across languages while preserving culturally relevant differences.
2. The paper offers deep insights into cross-lingual alignment through the transfer-localization plane, demonstrating that current multilingual methods enhance transfer performance at the cost of cultural localization, highlighting opportunities for improvement.
3. The authors propose Surgical Steering, an inference-time method that adjusts model representations to achieve a better balance between knowledge transfer and cultural localization.

### Weaknesses
1. The insights and contributions of this paper are impressive. However, the evaluation is limited to a single model and one culturally specific dataset. It would be important to demonstrate the method’s generalizability across other models and datasets.
2. The proposed Surgical Steering appears to be an adaptation of existing English Steering methods, which raises questions about presenting it as a wholly novel approach for cross-lingual alignment. Nevertheless, the authors provide a valuable insight into the trade-off between transfer and cultural localization in multilingual models.

### Questions
1. Do you have results on other models and cultural-specific datasets?
2. How do you calculate the LOC-steering vector? I didn't find a clear explanation for it in the paper.

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper is about the dark side of cross-lingual alignment or transfer. The authors conducted empirical studies for four common cross-lingual alignment frameworks (CLA)  to observe culture erasure after training, and they suggest considering both transfer/alignment and localization in CLA. Then, the authors propose a representation patching method that surgically steers shallow layers to “English” and middle-to-deep layers to the “target” language.

### Strengths
1.	The presentation is clear. 

2.	The authors provide some valuable, negative results for CLA.

### Weaknesses
1.	While the negative results are valuable, the overall contribution of the paper is not sufficient. The authors re-run existing frameworks on existing datasets, and the findings are not new. For example, the cultural bias is discussed in GLOBAL_MMLU, which the authors consider and use.

2.	The work is not complete. The method, which relies on surgery of the model, is not easy to reproduce on other models. The authors only conduct experiments on one model, making the generality unclear. Is there a principled way or an automated way to configure the method? Do you have any ablation studies for this?

### Questions
Please refer to Weaknesses

### Soundness
1

### Presentation
3

### Contribution
1

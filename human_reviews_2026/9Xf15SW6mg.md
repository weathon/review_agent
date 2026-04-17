# Unmasking LAION-5B: Age, Gender, Race, and Emotion Biases in Large-Scale Image Datasets

- Decision: Reject
- Scores: 2, 2, 2, 4

## Abstract
Large-scale image-text datasets, such as LAION-5B, are foundational to modern AI systems, yet their vast scale and uncurated nature raise significant concerns about demographic and stereotypical biases. This study presents a comprehensive analysis of the demographic composition and representational, stereotypical, and intersectional biases in LAION-2B-en and LAION-2B-multi, the two main components of the LAION-5B dataset. Using state-of-the-art models---FairFace, DeepFace, and Emo-AffectNet---we analyze faces detected in the dataset to identify biases across age, gender, race, and expressed emotion. Our findings reveal substantial overrepresentation of young adults (20--39), White individuals, and males, alongside consistent underrepresentation of minority racial groups and middle-aged or older women across both dataset components. We also observe stereotypical associations between demographic attributes and emotions, such as “Anger” being predominantly linked to males and “Happiness” to females, pointing to systemic imbalances in the data. The consistency of these patterns across two demographic models and both components of LAION-5B demonstrates that these biases are deeply embedded in one of the most widely-used training datasets. Given the scale at which LAION-5B is used to train generative models, these demographic imbalances could shape the behavior and outputs of numerous downstream AI systems.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
- The paper addresses a critical point in the field of generative AI: large-scale uncurated image-text datasets (e.g., LAION-5B) serve as foundational training data for modern AI systems, but their inherent biases remain insufficiently explored, potentially distorting downstream model fairness. 
- The paper systematically analyze LAION-5B’s two main components (LAION-2B-en and LAION-2B-multi), and then has the several conclusions : 1) Both dataset components exhibit significant representational biases, with overrepresentation of 20–39-year-olds, White individuals, and males, and underrepresentation of minority racial groups and middle-aged/older women; 2) Strong stereotypical biases exist (e.g., males linked to “Anger”/“Disgust,” females to “Happiness”); 3) Gender-age intersectional biases are the most consistent (younger females and older males overrepresented); 4) LAION-2B-multi shows slight improvements in race/age diversity but worse gender balance. 
- The paper conclude that these biases are deeply embedded in LAION-5B and may propagate to downstream generative AI systems.

### Strengths
1. The paper systematically examines three key types of biases (representational, intersectional, stereotypical) using well-validated tools and metrics. This design ensures the reliability of bias detection, as evidenced by consistent results across both models and dataset components.
2. Prior studies on LAION-5B focused on harmful content or model output bias, while this paper targets the under-explored “dataset intrinsic demographic composition” . It explicitly links dataset biases (e.g., male overrepresentation, gender-emotion stereotypes) to potential risks in downstream generative models (e.g., may reinforce gender disparities), complementing existing research and providing a foundational reference for dataset curation in fair AI.

### Weaknesses
* 1. The paper acknowledges reliance on FairFace, DeepFace, and Emo-AffectNet but lacks in-depth analysis of how these models’ own biases may confound results. 
* 2. The paper states that “balanced composition is treated as desirable” (Section 6.1) but does not explore alternative fairness definitions. For example, it does not explain whether a “balanced gender ratio” is necessary for LAION-5B, given that some generative tasks (e.g., medical image generation) may require domain-specific demographic distributions. This leads to a one-sided fairness framing. An improvement direction is to add a subsection in Discussion: “Contextual Fairness Considerations for LAION-5B,” contrasting demographic balance with task-specific fairness goals.

### Questions
>1. The paper uses ~1 million URLs (0.02% of LAION-5B) and justifies it via MOE (Section 3.1), but LAION-5B is web-scraped, which may have geographic/language-based data clustering (e.g., LAION-2B-multi has non-English content). Is there evidence that the sampled URLs are distributed uniformly across geographic regions or language families? For example, does the LAION-2B-multi sample include sufficient content from low-resource languages to represent their demographic characteristics? This is critical to validating whether sample biases reflect the full dataset.
>2. The paper excludes groups with <1% prevalence (in Section 4.3) due to unreliable Z scores. May this exclude marginalized subgroups (e.g., elderly Black females) that are critical for fair AI. Does the paper have plans to expand the sample size or use alternative statistical methods to analyze these low-prevalence groups? 
>3. **The paper suggests “careful training dataset curation” (Section 7) but provides no specific actionable strategies. How to address the identified biases? Providing such guidance would enhance the paper’s practical value.**

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
The authors analyze demographic and emotional bias to approximately 80K human images from two LAION-5B datasets, using pretrained computer vision models from FairFace, DeepFace, and Emo-AffectNet. They estimate demographic attributes (age, gender, race) and emotion labels, and report representational, intersectional, and stereotypical bias statistics.

### Strengths
The paper presents a rigorous statistical analysis of multiple bias dimensions (representational, intersectional, stereotypical).

### Weaknesses
- The choice of models (DeepFace and Emo-AffectNet) is questionable, as they are not state-of-the-art models but rather libraries primarily intended for easy use.
- The accuracy of the models is relatively low (e.g., DeepFace reports ~68% accuracy on race and 60% on age; Emo-AffectNet achieves ~66% on emotion). These inaccuracies likely propagate into the bias estimates, making the conclusions unreliable. For instance, Figure 1 shows inconsistent gender ratios (60–40 in FairFace vs. 70–30 in DeepFace).
- The claim in the contributions section about analyzing “a substantial sample of 1,000,000 image URLs” is misleading, since the effective analyzed sample is 80K images.
- The analysis largely confirms known facts, that web-scraped datasets are biased, without providing new insights or novel mitigation strategies.

### Questions
1. What is the added value of this analysis, given that it reiterates the well-known fact that web-scraped datasets contain bias?
2. Are there more curated datasets currently in active use by major AI developers? To your knowledge, do companies typically rely on uncurated LAION-5B subsets for training models?
3. Given that generative models are not typically deployed in high-stakes contexts like face recognition, what are the practical implications of the observed bias?

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This study provides an analysis of the demographic and stereotypical biases present in the LAION-2B-en and LAION-2B-multi datasets. Using the FairFace, DeepFace, and Emo-AffectNet models for face detection and attribute classification, the analysis reveals biases such as a strong overrepresentation of young adults, White individuals, and males, alongside a consistent underrepresentation of minority racial groups and middle-aged or older women.

### Strengths
1. The fundamental idea of this paper is technically correct.
2. The paper is well written and easy to follow.
3. The rationale behind each section and the overall motivation are clearly presented and easy to understand.

### Weaknesses
1. The study's entire analysis is fundamentally dependent on the outputs of third-party, pre-trained models: RetinaFace for face detection, and FairFace, DeepFace, and Emo-AffectNet for demographic and emotion classification. This methodology introduces a potentially fatal flaw. These models are known to exhibit their own inherent biases. The generated biased labels compromises the accuracy and validity of the final analysis. While the authors acknowledge this limitation and attempt to mitigate it by using two independent models, this does not resolve the core issue.
2. While the auditing of large-scale datasets is a valuable service to the community, the contribution of this paper may not meet the standard of ICLR. The paper stops at quantifying these biases without proposing or testing novel methods for bias mitigation. As such, the work feels more like a descriptive technical report than a novel research contribution.
3. The paper is primarily descriptive statistical analysis, lacking deep theoretical exploration of bias origins. For the discovered bias patterns (such as the "angry male-happy female" stereotype), there is insufficient theoretical explanation of why these patterns emerge and persist in web-scraped data.
4. While the paper notes potential impacts of dataset biases on downstream models, it lacks empirical validation. The authors do not actually test how these biases propagate to the models trained on LAION-5B.
3. From Figure 1, we can see that the demographic distributions detected by FairFace and DeepFace models are quite different. What types of samples typically show disagreement between the models? These details could be analyzed in greater depth.

### Questions
1. The analysis is flawed because it uses biased models to measure bias, making the results unreliable.
2. The paper lacks novelty for ICLR, as it only reports on known biases without proposing any new methods.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper audits **LAION-5B**—specifically **LAION-2B-en** and **LAION-2B-multi**—for **representational, intersectional, and stereotypical** biases along **age, gender, race, and emotion**. The authors sample ~1M URLs, successfully download ~464k images, detect **79,902** faces (≥48×48) via RetinaFace, and infer demographics with **FairFace** & **DeepFace** plus **Emo-AffectNet** for expressions. Using **Ducher’s $Z$** to quantify association biases, they find overrepresentation of **young adults (20–39)**, **White**, and **male** faces; consistent **gender–age** skew (younger females, older males); and stereotypical **emotion–gender** links (male↔Anger/Disgust, female↔Happiness). Bootstrapped CIs (FairFace) suggest these $Z$ effects are statistically stable. Code and derived CSVs (URLs + attribute predictions) are provided.

### Strengths
* **Methodological triangulation:** Two independent demographic models + FER model; intersectional and stereotypical analyses via a normalized $Z$.
    
* **Representative sampling scale:** ~**1M** URLs attempted; ~**80k** faces after quality gating; MOE quantified.
    
* **Consistent patterns:** Overrepresentation (20–39, White, male), robust gender–age skew, and emotion–gender stereotypes across components.
        
* **Statistical robustness:** Bootstrapped CIs for hallmark $Z$ findings.
    
* **Reproducibility:** Code + derived CSVs (URLs + predictions) to replicate figures/tables.

### Weaknesses
* **Tool-induced bias not disentangled.** No human-labeled audit subset to calibrate FairFace/DeepFace/Emo-AffectNet errors within LAION; sensitivity of conclusions to classifier confusion (e.g., race or age misestimates) is not quantified.
    
* **Sampling/attrition analysis is thin.** ~54% download failure and face-quality filtering could induce selection bias; limited evidence that final 79,902 faces remain representative of the starting partitions beyond MOE on proportions. (MOE ignores non-response mechanisms.)
        
* **Partial CI coverage.** CIs focus on exemplar $Z$s (FairFace only); a fuller table—spanning both demographic models and major cells—would strengthen claims.
        
* **Privacy surface.** Publishing per-image URLs paired with inferred sensitive attributes may raise re-identification risk even without image redistribution. The ethics discussion acknowledges risks in general, but concrete mitigations for released CSVs are not detailed.

* **Causality caveat.** The link from dataset bias to downstream generative behavior is discussed qualitatively; no bridging experiment (e.g., controlled training subset swaps) is provided.

* **Missing Related Works.** Especially in the case of face recognition bias, several related works should be incorporated to tell the complete story of bias in data and how we arrived at our current state. ** For example, but not limited to,
   - Robinson, J. P., Livitz, G., Henon, Y., Qin, C., Fu, Y., & Timoner, S. (2020). Face recognition: too bias, or not too bias?. In Proceedings of the ieee/cvf conference on computer vision and pattern recognition workshops (pp. 0-1).

I expect this would strengthen the story while better representing the start of research in this area. Especially the work mentioned above, it is just too relevant to overlook (it questions the overall completeness of this work).

### Questions
1. **Non-response bias.** Can you analyze **download success vs. failure** by metadata (e.g., language/domain) and report whether face-presence/quality differs materially, to bound attrition bias?
    
2. **Calibration subset.** Would you include a **human-labeled audit** (say, 2–5k faces) to estimate and correct classifier biases (age bin drift, race confusion) and re-report adjusted proportions/$Z$s?
    
3. **Dual-model reconciliation.** Where FairFace and DeepFace diverge (e.g., Asian vs Middle Eastern shares), can you provide a **consensus estimator** and uncertainty bands across models?
    
4. **Emotion reliability.** Given modest FER accuracy in-the-wild, how sensitive are the emotion–demographic $Z$ patterns to plausible label noise (e.g., ±10–20% symmetric/biased flips)?
    
5. **Downstream linkage.** Could you run a **mini ablation**: fine-tune a diffusion backbone on stratified, re-weighted subsets to test whether reducing identified imbalances measurably changes prompt-conditioned outputs?
    
6. **CSV privacy.** What safeguards (hashing, rate-limited access, per-row k-anonymity checks) accompany the released URL+attribute CSVs? Would you consider releasing **aggregated** counts only?

### Soundness
3

### Presentation
3

### Contribution
3

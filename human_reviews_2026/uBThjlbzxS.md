# Doxing via the Lens: Revealing Location-related Privacy Leakage on Multi-modal Large Reasoning Models

- Decision: Accept (Poster)
- Scores: 6, 4, 8, 6

## Abstract
Recent advances in multi-modal large reasoning models (MLRMs) have shown significant ability to interpret complex visual content. While these models possess impressive reasoning capabilities, they also introduce novel and underexplored privacy risks. In this paper, we identify a novel category of privacy leakage in MLRMs: Adversaries can infer sensitive geolocation information, such as users' home addresses or neighborhoods, from user-generated images, including selfies captured in private settings. To formalize and evaluate these risks, we propose a three-level privacy risk framework that categorizes image based on contextual sensitivity and potential for geolocation inference. We further introduce DoxBench, a curated dataset of 500 real-world images reflecting diverse privacy scenarios divided into 6 categories. Our evaluation across 13 advanced MLRMs and MLLMs demonstrates that most of these models outperform non-expert humans in geolocation inference and can effectively leak location-related private information. This significantly lowers the barrier for adversaries to obtain users' sensitive geolocation information. We further analyze and identify two primary factors contributing to this vulnerability: (1) MLRMs exhibit strong geolocation reasoning capabilities by leveraging visual clues in combination with their internal world knowledge; and (2) MLRMs frequently rely on privacy-related visual clues for inference without any built-in mechanisms to suppress or avoid such usage. To better understand and demonstrate real-world attack feasibility, we propose GeoMiner, a collaborative attack framework that decomposes the prediction process into two stages consisting of clue extraction and reasoning to improve geolocation performance. Our findings highlight the urgent need to reassess inference-time privacy risks in MLRMs to better protect users' sensitive information.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper identifies a new privacy risk in multimodal large reasoning models (MLRMs): geolocation inference leakage, where models can deduce users’ home locations from images like selfies. The authors propose a three-level privacy risk framework and introduce DOXBENCH, a 500-image dataset to evaluate such risks. They also develop GEOMINER, an attack framework that demonstrates how MLRMs can effectively infer locations using visual clues, underscoring the urgent need for privacy safeguards in multimodal AI systems.

### Strengths
- This paper presents a comprehensive study, covering the motivation, evaluation of existing methods, as well as attack and defense aspects.

- I particularly like Table 1, which maps the identified risks to specific legal and regulatory provisions.

- The experiments are also quite thorough and well-conducted.

- threat model defined is aligned with the practice.

### Weaknesses
First, I would like to clarify that I have read the entire main body of the paper carefully, including some of the appendices that are relevant to my interests, as well as the figures and tables in those appendices. I may have skipped certain parts of the appendix that are either unrelated to the main text or not of direct interest to me. Therefore, if any of my questions have already been addressed in the appendix, please kindly point that out.

- The prompt structures used in the benchmark evaluation may reflect different risk capabilities. How do you evaluate the potential capability or upper bound of risk represented by each prompt?

- I noticed that in Table 2, Claude’s VRR is quite low, which seems to significantly affect the overall evaluation outcome. Can your evaluation strategy mitigate or correct this imbalance caused by low VRR values?

- Many of the result explanations in the paper do not clearly indicate which figure or table they refer to. For example, the statement “Prediction difficulty increases with the annotated levels”, which figure is this referring to?

- How was Figure 25 generated? From which dataset or classification process did it originate, and what method was used for the classification?

- Regarding Figure 4, could there be potential information leakage, since the “clue” might have been extracted from the same dataset used for evaluation?

### Questions
- Can your evaluation strategy mitigate or correct this imbalance caused by low VRR values?

- Many of the result explanations in the paper do not clearly indicate which figure or table they refer to. For example, the statement “Prediction difficulty increases with the annotated levels”, which figure is this referring to?

- How was Figure 25 generated? From which dataset or classification process did it originate, and what method was used for the classification?

- Regarding Figure 4, could there be potential information leakage, since the “clue” might have been extracted from the same dataset used for evaluation?

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
This paper investigates a novel privacy risk associated with recent multi-modal large reasoning models (MLRMs). It finds that these models possess sophisticated reasoning capabilities enabling them to infer sensitive geolocation information, such as home addresses or neighborhoods, from user-generated images like selfies, even those taken in private settings. To evaluate this risk, the authors propose a three-level privacy risk framework and introduce DOXBENCH, a benchmark dataset of 500 real-world images representing various privacy scenarios.

### Strengths
1. This paper focuses on evaluating the risk of geolocation privacy leakage in large models and introduces a novel benchmark dataset for this purpose.

2. The use of the GLARE metric is well-justified, offering a more comprehensive assessment than simple accuracy measures alone.

3. The experiments conducted are extensive, covering a wide range of mainstream large models.

### Weaknesses
1. The paper anchors its geographic scope almost exclusively to California, rendering the dataset incomplete and limiting its persuasiveness. This raises concerns about potential bias, possibly stemming from an overrepresentation of California data in the large models' training sets. While the authors acknowledge this limitation in Appendix F, the discussion provided is far from sufficient to address the concern.

2. Critical questions regarding defense mechanisms and the utility of geolocation data remain unanswered. How can this identified privacy vulnerability be effectively mitigated? Furthermore, what is the quantifiable benefit or utility gained by large models from leveraging geographic coordinates?

3. The distinction between "Privacy Space" and "Personal Imagery" lacks clarity. What are the precise operational differences between these two concepts? They appear highly similar and ultimately converge on the fundamental issue of personal privacy, making their practical differentiation ambiguous.

### Questions
See above.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper reveals that multi-modal large reasoning models (MLRMs) can infer users’ private locations from ordinary photos, including selfies in personal spaces. The authors build DOXBENCH, a 500-image dataset of real-world scenes annotated into three legal privacy-risk levels, and evaluate 13 MLRMs and MLLMs, showing that most outperform non-expert humans in geolocation inference. The paper further proposes CLUEMINER to identify which visual clues drive location reasoning, and GEOMINER, a two-stage attack combining clue extraction and reasoning to amplify leakage.

### Strengths
S1: The problem studied in this paper is interesting and well-motivated. 

S2: The paper introduces a purpose-built dataset of 500 privacy-sensitive, real-world images representing personal spaces rather than public landmarks, making the evaluation realistic and legally grounded.

S3: The paper benchmarks 14 leading multimodal models using reproducible metrics. 

S4: The paper develops CLUEMINER (to identify visual clue categories) and GEOMINER (a two-stage clue-assisted attack) that together reveal how and why leakage occurs. 

S5: Overall, the paper is well written, well-organized, and easy to follow.

### Weaknesses
W1: DOXBENCH primarily includes images from California and nearby areas, which may limit geographic, cultural, and environmental diversity; generalization to other regions remains unclear.

W2: It seems that the experiments focus on image-based inputs; the approach and findings may not fully extend to other modalities (e.g., video or text-image pairs).

### Questions
Q1: Can the authors briefly discuss whether their findings generalize to regions beyond California, and what additional geographic or environmental factors might influence model performance?

Q2: Can the proposed analysis and metrics be extended to other modalities, such as video or text–image pairs, and what challenges might arise in doing so?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents the first systematic study of location-related privacy leakage in Multi-modal Large Reasoning Models. The authors identify a novel privacy risk where adversaries can infer sensitive geolocation information from user-generated images. Key contributions include:They constructed DoxBench, the first benchmark dataset specifically designed to evaluate this risk. They also introduced a three-tier privacy risk taxonomy grounded in legal frameworks. Furthermore, it innovatively proposes GLARE, an information-theoretic metric to quantify the extent of privacy leakage, and develops analytical tools named ClueMiner and GeoMiner to trace the root causes of the risk and demonstrate attack feasibility.

### Strengths
The authors have meticulously constructed a privacy dataset containing real-world scenarios, proposed a highly innovative evaluation metric, and demonstrated the pervasiveness and severity of the risks through extensive experiments. They even showcased how their attack tools could enable ordinary users to achieve this with ease. The entire research framework is comprehensive, progressing logically from problem definition and analysis to verification, with robust evidence throughout.

### Weaknesses
The current dataset primarily focuses on California, USA, which naturally raises the question: would this methodology remain equally effective when applied to European or Asian streetscapes and architectural styles? Furthermore, in the defense section, while several methods were tested, the underlying reasons for their failures haven't been thoroughly explored. A deeper analysis of how the models circumvent blurring and noise-based defenses would provide more valuable insights.

### Questions
Could you validate your findings on images from other geographic regions？

### Soundness
3

### Presentation
3

### Contribution
3

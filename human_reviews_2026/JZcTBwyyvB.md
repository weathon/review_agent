# Counting Hallucinations in Diffusion Models

- Decision: Reject
- Scores: 2, 4, 4, 4

## Abstract
Diffusion probabilistic models (DPMs) have demonstrated remarkable progress in generative tasks, such as image and video synthesis. However, they still often produce hallucinated samples (_hallucinations_) that conflict with real-world knowledge, such as generating an implausible duplicate cup floating beside another cup.
Despite their prevalence, the lack of feasible methodologies for systematically quantifying such hallucinations hinders progress in addressing this challenge and obscures potential pathways for designing next-generation generative models under factual constraints.
In this work, we bridge this gap by focusing on a specific form of hallucination, which we term **counting hallucination**, referring to the generation of an incorrect number of instances or structured objects, such as a hand image with six fingers, despite such patterns being absent from the training data.
To this end, we construct a dataset suite **CountHalluSet**, with well-defined counting criteria, comprising ToyShape, SimObject, and RealHand.
Using these datasets, we develop a standardized evaluation protocol for quantifying counting hallucinations, and systematically examine how different sampling conditions in DPMs, including solver type, ODE solver order, sampling steps, and initial noise, affect counting hallucination levels. 
Furthermore, we analyze their correlation with common evaluation metrics such as Fréchet Inception Distance (FID), revealing this widely used image quality metric fail to capture counting hallucinations consistently.
This work aims to take the first step toward systematically quantifying hallucinations in diffusion models and offer new insights into the investigation of hallucination phenomena in image generation.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces a framework for quantitatively measuring a specific type of error in image diffusion models called counting hallucinations: cases where models generate an incorrect number of objects (e.g., a hand with missing fingers). The authors build a dataset suite called CountHalluSet and analyze how sampling conditions like solver type, ODE order, number of steps, and initial noise affect hallucination rates. They find that standard quality metrics like FID fail to capture these errors and propose a simple mitigation method called joint-diffusion models to reduce hallucination.

### Strengths
1. Hallucination has long been difficult to quantify in generative models. This paper makes progress by isolating and rigorously evaluating one facet of hallucination (counting hallucination) where the number of generated objects is different from the ground truth.
2. The experimental setup, evaluation metrics, and analyses are clearly described and well controlled, which enhances reproducibility and robustness of the results.

### Weaknesses
1. The paper does not evaluate state-of-the-art diffusion models (e.g., FLUX, Stable Diffusion), but instead relies on models trained or fine-tuned on relatively small datasets. As a result, it remains unclear whether CountHalluSet meaningfully captures hallucination behavior in modern large-scale diffusion models, or whether the reported effects of sampling parameters generalize to them.
2. The study focuses narrowly on counting hallucinations, whereas real-world hallucinations in image generation include missing/wrong subjects, incorrect spatial relationships, geometry, depth, color, and style inconsistencies. This narrow scope limits the paper’s overall contribution and novelty.
3. The only real-image dataset used, RealHand, consists of single-hand images occupying most of the frame. This dataset lacks diversity in object categories and does not test models on multi-object types or cluttered scenes.

### Questions
1. The proposed Joint-Diffusion Model involves classifier guidance from a detector at intermediate denoising steps, which may substantially increase inference time. Did the authors benchmark the impact of the additional guidance on wall-clock generation time?

### Soundness
4

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
In this paper, the authors discuss the problem of hallucination in image generation by diffusion models, focusing on a specific subtype termed counting hallucination, where generated images contain a physically impossible or dataset-inconsistent number of objects. To quantify this phenomenon, the authors construct a benchmark suite, CountHalluSet, consisting of three datasets of increasing visual complexity, each defined by explicit counting rules. Using these datasets, they systematically analyze how different diffusion sampling conditions (e.g., number of steps, solver type, initial noise) affect counting hallucination. The study further compares hallucination rates with FID, showing that while generation quality metrics like FID correlate with general visual failures, they do not correlate with counting hallucinations, highlighting a gap between perceptual quality and factual correctness. Finally, motivated by these findings, the authors introduce a conceptual Joint Diffusion Model (JDM), which purportedly reduces hallucinations by jointly denoising visual representations and factual constraints in a shared latent space.

### Strengths
- The paper is nicely motivated with a relevant and important issue: image generation models often generate images that does not folow phyciscal law. 
- The authors focus on counting hallucination and define measureable metrics, makes the analysis quantifiable. 
- The proposed CountHalluSet provides a controlled progression of datasets from synthetic toy datasets to real-world datsets, enabling systematic study of hallucination under increasing realism.
- The paper explores some important dimensions of diffusion models including solver types, integration order, and step counts influence hallucination rates.
- Section 5.4 propose interesting future work direction on how to reducte hallucination can insipre future work on constraint-aware diffusion generation.

### Weaknesses
Overall, the scope of the studied problem seems limited. The paper focuses narrowly on counting hallucinations but offers little explanation of why diffusion models shows counting hallucination or how diffusion dynamics contribute. The proposed dataset suite is mostly synthetic with arbirtray counting rules. The proposed Joint Diffusion Model (JDM) in Section 5.5 is an interesting idea but discussed too briefly and without sufficient methodological detail. 

- Paper writing lack clarity on some experiments. For example, there is no clear definition on the NCFR mentioned in Table 1.It is unclear what hat constitutes a “non-counting hallucination” and how such samples are detected. 
- Similarly, there is no clear explanation for the proposed JDM in table 3. Whereas the motivation makes sense and seems promising, the paper lacks discussion on what techniques and architecture was adopted for Table 3.
- The paper motivates with the problem of hallucination as violations of physical laws or semantic consistency. However, the toy datasets primarily enforce arbitrary counting rules rather than true physical or causal constraints. It is questionable how representitive the datasets are for real image generation counting constraints. This can also be confirmed with the inconsistent trend between the datasets in the result section.

### Questions
- How exactly are NCFR and TFT measured?
- Can you provide mored detail on the JDM in section 5.5? 
- The motivation mentions “violations of physical laws.” How do the synthetic datasets (ToyShape, SimObject) reflect such physical constraints rather than dataset-specific counting rules?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper investigates the problem of counting hallucinations. To this end, the authors construct CountHalluSet, a benchmark comprising three components: ToyShape, SimObject, and RealHand, ranging from simple synthetic shapes to real-world hand images. Using this dataset, the paper examines how factors such as solver type, sampling steps, and initial noise influence counting hallucinations. Interestingly, the study reveals that counting hallucinations are not directly correlated with overall generation quality. Finally, the authors propose a method to mitigate these hallucinations, contributing a practical solution to the issue.

### Strengths
1. This paper is the first to systematically study counting hallucinations in diffusion models. In large-scale text-to-image diffusion models, counting hallucinations present a genuine challenge. Therefore, this work has the potential to provide a systematic and quantitative understanding of counting hallucinations in such models.

2. The paper is well-structured: the definition of counting hallucination is clearly articulated, the experimental setup is well explained, and the proposed dataset is thoughtfully constructed. Moreover, the experimental findings, such as the observation that counting hallucinations are not intrinsically related to generation quality, are both interesting for the research community.

### Weaknesses
1. The paper claims a contribution in proposing a method to reduce counting hallucinations. However, the description of this method lacks sufficient detail. For instance, what exactly are the "joint-diffusion models (JDM)" mentioned in line 464? More clarification and elaboration are necessary.

2. There is a noticeable gap between the contributions of this paper and the issues that people care about in practice. Specifically, the authors evaluate primarily on synthetic datasets and a small-scale hand dataset to draw conclusions. It is unclear how these findings generalize to large-scale models, which is the community’s main concern. I recommend the following additional experiments to strengthen the paper:

      a. Study counting hallucinations in large-scale text-to-image diffusion models such as Stable Diffusion or Flux. For example, generate images containing hands and evaluate how solver type, sampling steps, and initial noise affect counting hallucinations. Additionally, investigate the relationship between generation quality and counting hallucinations in these settings.

      b. Evaluate the performance of the proposed joint-diffusion models on large-scale text-to-image diffusion models, to assess their practical effectiveness.

If the authors address these concerns, I would be inclined to raise my score.

### Questions
1. How to obtain the "ground-truth initial noise" in line 372.
2. How many samples are generated to evaluate results in Table 1 and Table 2?

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper builds on top of the hallucination investigation by Aithal et. al. by introducing a benchmark suite and a complementary method that allows quantifying “counting hallucinations” in diffusion models. These are cases where diffusion models generate incorrect number of objects (such as extra fingers). Three datasets are used with different complexity: ToyShape, SimObject, and RealHand. 

The key findings of the paper are that: (1) standard image quality metrics like FID fail to measure counting hallucinations; (2) While increasing sampling steps reduces hallucinations in the simple datasets, it often increases them in complex ones; and (3) a Joint Diffusion Model (JDM) method helps reduce hallucinations across the board.

### Strengths
1. The problem has been studied in a very clean way. The authors articulate how counting hallucinations is very well positioned because of its quantifiable nature. On top of that, the three levels of complexity allow for a very clear comparison.
2. JDM provides a promising initial direction for enforcing factual constraints in generative models.
3. The lack of correlation of count hallucinations with FID reflects how perceptual and factual quality may not always agree. This is a nice point.
4. Really like the pre-filtering with counting-ready indicator! This makes so much sense.

### Weaknesses
1. The experiments on JDM are rather short. I would have liked to see a broader discussion of the same across datasets.
2. How does JDM impact FID? How is JDM implemented? The section is very underdevleoped and not conference ready.
3. Since the FID correlations change so much across solvers and datasets, it really calls for more experiments: either more finetuning runs, or more datasets, or more solvers. We need more evidence before making any statistically significant claim

### Questions
1. I do not think I followed how NCFR is computed. What are the other failures for each dataset?
2. Which dataset is Table 3 on?

### Soundness
3

### Presentation
3

### Contribution
2

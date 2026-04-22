# MontageAug: Enhancing Long-tail Robustness And Semantic Consistency of VLMs

- Avg Score: 4.67
- Decision: Reject
- Scores: 4, 4, 6

## Abstract
Vision-Language Models (VLMs) have made significant strides in multimodal understanding tasks, yet their robustness faces severe challenges when dealing with the long-tail data distributions common in the real world, especially in high-stakes domains like medical image analysis.
To address this challenge, we propose MontageAug, a compositional data augmentation approach designed specifically for long-tail vertical domains.
It strategically composes images (particularly from head and tail classes) to construct a novel visual scene (a montage image) and synchronously generates a perfectly corresponding compositional text description.
This method not only fundamentally guarantees the semantic fidelity of the augmented samples but also effectively alleviates the long-tail data problem by creating information-rich hard positive samples.
We conducted extensive experimental validation on a model based on the InternVL architecture using ophthalmic medical benchmarks.
The results show that MontageAug significantly enhances the model's recognition performance and generalization on tail classes, achieving state-of-the-art (SOTA) performance that surpasses existing augmentation methods on several benchmarks.
Furthermore, to explore the approach's extensibility, we validated it on  Mathematical Expression Recognition (MER), achieving consistent improvements.
Our work ultimately demonstrates that MontageAug, as an efficient, low-cost, and semantics-preserving VLM augmentation strategy, holds practical value in solving the long-tail problem in specialized domains.
We plan to open-source our code, benchmark data, and models upon paper acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes MontageAug, an augmentation method to enhance VLMs' long-tail robustness. MontageAug comprises three major components: (1) Hard Sample Prioritization Sampler, which prioritizes the sampling of rare and hard samples; (2) Visual Montage Composition, which vertically concatenates images from head and tail classes to form an augmentation image; (3) Textual Montage Composition, which concatenates captions through a template to ensure semantic consistency. Experimental results show that MontageAug surpasses simple oversampling and single-modality-based augmentation (mixup and caption-based image generation) in medical and general-purpose VLM benchmarks.

### Strengths
1. It is sound to consider semantic consistency in long-tail augmentation methods for VLM, instead of single-modality augmentation.
2. The experimental results show that MontageAug surpasses simple oversampling and single-modality augmentation.

### Weaknesses
1. **Over-claimed generalizability.**  The paper has claimed its effectiveness in the general domain. However, the experimental results on general-purpose benchmarks are only left in the appendix, with only one baseline (simple SFT), which is not convincing.

2. **Confusing writing.** Some important details are missing, such as training data and MLLM for report generation. And the organization of the paper is also confusing. For detailed questions, please see below.

3. If the evaluation benchmarks are from the medical domain, why does this paper choose a general-purpose VLM as the base model instead of widely-used medical VLMs like LLaVA-Med?

### Questions
- Line 265: Which powerful MLLM is used for report generation?
- Sec 4.1.1, what open-source data has been used?
- In Figure 3, how does the model learn to differentiate part 1 and part 2? My concerns stem from the experimental results on general benchmarks that MontageAug suppresses performance on tasks that rely on fine-grained perception (e.g., MME). Does this really relate to resolution, or due to the misalignment between different parts?
- Line 130, it is suggested to replace "comprehensively surpasses" with "consistently surpasses"
- Missing '.' in the captions of Figure 3 and Figure 4.

### Soundness
2

### Presentation
1

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
The authors present a novel augmentation technique called MontageAug specifically created for VLMs and long-tailed problems, which conserves the semantic coherence of image-text pairs by leveraging “hard-positive” samples.
MontageAug creates a montage of multiple images and a text template to pair it with the respective textual descriptions. Further, the method chooses images based on the class abundance and on the instance difficulty, which is assessed by the performance of a base model. MontageAug achieves SOTA performance on Ophthalmology datasets outperforming simple SFT as well as other augmentation approaches like generative augmentations, sampling techniques or mixing techniques.

### Strengths
MontageAug convinces with its simplicity and broad applicability to most VLMs and tasks.
The method is well motivated in the need of improving models for long-tailed problems.

### Weaknesses
1.) It is unclear how an epoch is defined for each method and therefore it is unclear whether the presented performance gain is solely due to the fact that MontageAug sees more image-text pairs e.g. compared to SFT. While Table 2 indicates an overfitting with more epochs for SFT (i.e. 2 epochs), it remains to be shown if sampling based methods would benefit from more epochs here. Sampling techniques could benefit from more epochs in this case, e.g. increasing the numbers of samples seen to the amount of image-text pairs the MontageAug sees. 

2.) Further, due to the montage the "effective" batch size could be larger in the case of MontageAug which could be the source of improvement, an ablation with SFT and Oversampling with a 40% (since the best $\alpha$ for MontageAug was 0.4) increased batch size would be interesting as well. To be more explict: one could conduct an experiment where instead of the montaged images those images which would make up the montage will end up in the same batch with a batch size accomodating the additional images of the montage so that the information exposure per training step is the same.

3.) Due to the fact that the epoch for each method is not properly defined, the training times presented in table 3 are against expectation i.e. why the training time for “Oversampling” should be higher than the “MontageAug”/”Vanilla SFT” method.

4.) It is unclear (cf lines 265ff) how the training dataset was generated. Both the used MLLM/VLM and the used templates are not specified. Further, to be more clear the authors should already link to the listing of the training datasets within the appendix around lines 265.

5.) Regarding the textual montage the exact template being used is not provided.

6.) The captions of tables 1, 2 and 4 could be improved as they do not mention the metrics being used as well as what bold values are supposed to mean (i.e. best perfromance). In table 3 it could be specified which "medical dataset" was being used (e.g. referring to the main body).

7.) While there is a small hyper-parameter grid-search done for the MontageAug method the other ablated methods were only done with one set of possible sub-optimal hyper-parameters which could make MontageAug superior due to a better set of hyper-parameters with respect to the other methods. E.g. for the hard oversampling one could find a more suitable $\alpha$ (oversampling frequency).

8.) Line 144: The authors state that their method demonstrates practical value for medical image analysis in general while it is only shown for Ophthalmology. I.e. this claim is too broad for what is shown in the paper.

9.) Line 48ff: How many private samples were collected? Should also be included in the caption of Figure 1.

10.) Line 270ff: In listing their validation data they don't state the number of classes for all datasets (i.e. GMAIMMBench and FundusMMBench).

11.) Line 411: The authors state "... the MontageAug method shows continuous performance improvement with more training epochs" whereas it is only shown for 1 and 2 epochs so the statement is too general.

12.) Why did the authors chose LLaVA-1.5 to ablate the method on other tasks, as it is clear that the method benefits strongly from a VLM which supports dynamic resolution? Overall section 4.2 feels rushed.

13.) The authors should explicitly refer to figure 1 in line 130.

14.) Regarding the reading flow the listing starting from line 095 feels abrupt and a little out of context. Further in line 337 there should be some transition between the listing of the comparison methods and the baseline methods.

15.) The statement in line 030/031 and 144 is too broad "demonstrating its practical value for medical image analysis.", while it is only shown for ophthalmology images.

### Questions
How exactly are epochs defined in your paper? If an epoch is one run through all training cases how can it be that in Table 3 the training time of Vanilla SFT and Oversampling differ so much? If the Oversampling technique additionally sees 40% more rare cases on top than one would expect a training time 40% larger (around 53h) than SFT which is not the case.

Why is the increase in training time of the MontageAug method only marginal compared to SFT in table 3, if InternVL uses dynamic resolution there should be 40% more tokens being processed?

### Soundness
1

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
3

### Summary
This paper aims to tackle a well-posed problem: instruction-tuned VLMs struggle on long‑tail categories, particularly in ophthalmic fundus imaging, where rare diseases (“tail classes”) have few labeled instances. In this problem, the training data are instruction–answer pairs and the objective is to improve downstream multiple‑choice diagnostic accuracy especially on tail categories, while preserving image–text semantic consistency. Specifically, the proposed method composes $k$ images into a grid montage and synchronously composes the paired texts with a deterministic template, yielding hard positive examples that are visually richer but semantically perfectly aligned with their new text. A hard-sample prioritization sampler biases secondary images toward (i) rare categories and (ii) samples mispredicted by a strong VLM, identified by an evaluator LLM. A training item is replaced by a montage with a predefined probability $\alpha$. 

Empirically, the method is evaluated using the dataset constructed using ~200k fundus images and ~300k instruction pairs with 11/4 ratio of normal and harm examples and on four fundus benchmarks. Under equal budgets on InternVL‑Chat‑V3.0‑8B, MontageAug improves average accuracy by 2.5% percents. In contrast, oversampling matched for tail exposure gives +0.4 on average, while Mixup and RetinaLogos-based generation reduce performance. Against specialized medical VLMs and closed models (e.g., GPT-4o), the InternVL model fine-tuned with the proposed method is competitive or better on this benchmarking suite. Finally, a generality probe on LLaVA‑1.5 shows benefits on compositional reasoning but degradation on fine‑grained perception, attributed to montage downsampling and spatial‑label incompatibilities.

### Strengths
1. Figure 1 indeed ties head–tail skew to accuracy collapse, justifying a tail‑focused augmentation that preserves alignment.
2. Composing both vision and text enables the proposed method to avoid label noise that plagues baselines Mixup/generation.
3. The comparisons are generally fair and extensive.
4. The method does not introduce significant computing overhead, as shown by Table 3, where the proposed method shows nearly identical training time to vanilla and a lower cost than generation.

### Weaknesses
1. Weak matching is currently dependent on GPT-4o without prompt/post-processing details. Meanwhile, the robustness to other open evaluators is unreported.
2. The hard-sample pool depends on Qwen2.5-72B judgments. The sensitivity to thresholding and the evaluator are not analyzed.
3. The strongest results are in fundus VQA-style tasks, while the general-domain LLaVA results are relatively mixed.
4. The authors do not provide control for image-only montage or alternate templates. In this case, the relative role of textual scaffolding is not that clear.

### Questions
1. Can you release the GPT-4o prompts and normalizers used for weak matching and replicate with other open-sourced models to quantify the evaluator variance?
2. Can you compare full montage, image-only montage (i.e., keep original text), text-only concatenation (i.e., no visual montage), and alternate templates to isolate the effects?
3. It is also suggested to add a public long-tail benchmark outside medicine (i.e., single-image evaluation) to assess the cross-domain utility of this method.
4. For fixed-resolution encoders (e.g., LLaVA-1.5), have you attempted with tiled cropping/higher-resolution encoders/etc to recover MME performance?

### Soundness
3

### Presentation
2

### Contribution
3

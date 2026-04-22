# On the Diminishing Reliability of Reference-Free Memorization Detection in Modern Diffusion Models

- Avg Score: 3.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 0, 4, 4, 4

## Abstract
Diffusion models have been observed to memorize and regurgitate portions of their training data, which raises potential copyright and privacy concerns. To quantify and mitigate this phenomenon, various reference-free metrics that operate without training data access have become an effective tool for detecting memorization in text-to-image systems. As diffusion models expand beyond the familiar text-to-image paradigm to encompass multi-modal and multi-stage training for 3D and video synthesis, the reliability of existing detection methods in these novel domains remains unclear. In this work, (1) We find that metric efficacy declines when applied to models that are fine-tuned in multiple stages from a text-to-image base to support additional modalities, where more varied training protocols may obscure memorization signals from existing detection techniques. (2) We demonstrate that these metrics have limited reliability in distinguishing between successful and failed memorization mitigation attempts, risking false judgments in model sanitization efforts. (3) We trace this performance degradation to violations of assumptions underlying current detection frameworks and conduct factorized analysis. Our findings call for caution when applying existing memorization detection metrics beyond text-to-image models and point toward the need for more robust evaluation methods tailored to a wider range of emerging diffusion models with diverse training protocols.

## Human Reviews

## Human Reviewer 1

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
This paper investigates why reference-free memorization detection methods, such as CFG- and attention-based metrics, lose reliability when applied to modern multi-stage and multi-modal diffusion models, revealing that violations of their underlying geometric assumptions lead to degraded performance and unreliable evaluation of unlearning methods.

### Strengths
1. The paper presents a clear and logically structured analysis that systematically connects empirical observations with theoretical explanations.

2. The experimental results are well-visualized, with comprehensive figures that effectively illustrate the degradation trends across models and metrics.

### Weaknesses
1. Flawed baseline and misimplementation of CAE
Regarding CAE (Cross-Attention Entropy) and SSCD, the entire experimental foundation seems incorrect. The baseline results for both CAE and SSCD are far below those reported in the original papers — both methods achieve over 0.99 AUROC on Stable Diffusion v1.4, yet this paper reports much lower values. In addition, the implementation of CAE is clearly wrong. The authors claim that “lower entropy indicates concentrated attention on a few tokens, suggesting memorization.” However, according to the original paper, only certain heads focus on the trigger token, while others focus on the beginning token. The competition between the trigger and the beginning token causes dispersed attention, i.e., higher entropy, when memorization occurs. Conversely, when there is no memorization, attention concentrates on the beginning token, leading to lower entropy. This fundamental misunderstanding makes the implementation invalid and casts doubt on the reliability of all experimental results. Since the entire paper’s conclusions rely on these experiments, I recommend a strong rejection.

2. Missing annotation in Appendix E
Appendix E lacks clear labeling, making it impossible to distinguish which images are training samples and which are generated samples. This seriously affects the interpretability of the visual results.

3. Poorly described experimental settings
The experimental settings are described with insufficient detail. For example, the paper does not specify how many images were used for SSCD, how many steps were taken, which specific CAE metric variant was used, or how many denoising steps were involved. These omissions make the experiments non-reproducible and the results unverifiable.

### Questions
As mentioned in Weakness 1 and Weakness 3, the experiments are unclear and terrible.

### Soundness
2

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
This paper systematically examines the reliability of reference-free memorization detection metrics for diffusion models in multi-stage/multimodal settings, and further evaluates whether these metrics can reliably verify dememorization success. It provides a geometric/statistical failure analysis with diagnostics and controlled interventions.

### Strengths
1.The paper systematically examines the applicability limits of T2I-era reference-free memorization detection metrics in video/3D/multi-stage pipelines, aligning with where diffusion models are rapidly expanding. 

2.Evaluates both naturally trained models and multiple post-hoc dememorization methods across image, video, and 3D, improving external validity.

2.AUROC shrinkage and distribution overlap visualizations are intuitive.

### Weaknesses
1. While the paper diagnoses why reference-free metrics fail in video/3D settings, it does not translate its theoretical analysis into new or adapted metrics tailored to these modalities.

2. The compared models differ in resolution, sampling steps, CFG strength, scheduler, and latent space; without rigorous alignment and sensitivity analyses, the conclusions may be driven by implementation details rather than underlying phenomena.

### Questions
1.Have you evaluated the interaction between CFG and the number of sampling steps, and also performed  different CFG analyses?

2.Do you provide a systematic attribution of misclassified cases? Which categories most commonly cause reference-free metrics to fail?

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
3

### Summary
This paper investigates the reliability of reference-free memorization detection metrics on complex, multi-modal diffusion models such as LaVie, MVDream, and DiffSplat. The authors find that these metrics degrade significantly under multi-stage or domain-shifted training, and analyze this through a geometric framework that links metric failure to violations of underlying theoretical assumptions (e.g., Gaussian locality, covariance alignment).

### Strengths
1. Timely and important topic. Memorization detection and model safety are high-interest issues in diffusion models.
2. Strong empirical scope. The experiments cover multiple metrics and models, providing a broad empirical overview.
3. The analysis connecting detection performance degradation to assumption violations is interesting.

### Weaknesses
1. **Low contribution.** The paper proposes no new detection metric, only diagnoses why existing ones fail. Since prior metrics were never designed for these complex training settings, observing degraded accuracy is somewhat trivial. While confirming this empirically and analyzing the cause is mildly interesting, a paper focused only on the limitations of existing methods without offering new methodology feels incomplete.
2. **Memorization bias is uncontrolled.** In Section 3, model-level memorization bias is a major confounding factor, but it is not considered. That is, the readers do not have information about how frequently a model reproduces training samples or how close generated outputs are to those samples. However, this is an important factor in the accuracy of detection methods. For instance, in SD 1.4, a prompt that reproduces memorized content 1/10 times versus 10/10 times would lead to very different detection accuracies, with the latter naturally yielding higher accuracy. Without quantifying or controlling this bias across different models, the evaluation lacks validity. Moreover, while diversity-based metrics such as Median SSCD and TL2 do not directly measure the proportion of generated but memorized images, they can implicitly indicate how strongly the model is biased toward a single image. The fact that MVDream, LaVie, and DiffSplat show low AUROC scores in these diversity metrics suggests that they are not strongly biased toward specific images — meaning their inherent memorization bias is low. Therefore, using such low-bias models to argue that detection metrics fail under complex training is misleading. A small-scale controlled experiment that systematically varies memorization bias under complex training settings would make this claim more convincing.
3. **Confounded experimental design.** Section 3 mixes modality and complex training, making it unclear which factor drives the performance drop. RL-fine-tuned models (e.g., GRPO, DPO) could also violate the assumptions in Section 4, but are not tested. Although Section 5 adds small-scale experiments, they are insufficient to support general conclusions about large-scale experiments.

To be acceptable, the work needs to go beyond diagnosing old metrics—it should either (a) propose a new detection method robust to complex training or (b) present a truly novel and concrete theoretical analysis. The contribution is too incremental now.

### Questions
1. Can authors conduct a controlled experiment to test how detection accuracy changes under different models with the same memorization bias?
2. Can authors isolate the effects of modality vs. complex training in large-scale models?

### Soundness
2

### Presentation
2

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
This paper investigates the reliability of reference-free memorization detection metrics when applied to modern diffusion models (DMs) that go beyond standard text-to-image (T2I) training. While these metrics have shown promise for identifying memorization in Stable Diffusion, the authors show their performance degrades substantially in multi-modal or multi-stage models such as LaVie, MVDream, and DiffSplat. The study (1) systematically evaluates multiple CFG- and diversity-based metrics (NDN, HED, BE, SSCD, TL2, CAE) across diverse architectures, (2) examines their reliability in assessing post-training “unlearning” or memorization-mitigation methods, and (3) links the drop in reliability to violations of core geometric assumptions underlying these metrics. Empirical analysis and controlled experiments demonstrate that common training modifications can systematically invalidate these assumptions, explaining the degradation. The paper concludes by emphasizing the need for training-aware detection frameworks to ensure dependable safety evaluation of next-generation diffusion models.

### Strengths
1. The paper identifies a previously underexplored limitation: most memorization detection metrics are validated only on vanilla T2I diffusion models, yet are now routinely applied to much more complex systems (multi-modal, multi-stage, 3D/video). 
2. The authors’ systematic evaluation and theoretical diagnosis represent a new perspective on the robustness and transferability of safety-critical metrics. The mapping between training-protocol design choices and theoretical assumption violations (Table 4) is an especially creative conceptual bridge.
3. The work carries high relevance for both safety and the evaluation of generative models. As diffusion architectures diversify, understanding when reference-free metrics fail is crucial for privacy auditing, model release, and regulatory compliance. 
4. The paper is well written and easy to follow. Figures are informative and visually consistent.

### Weaknesses
1. The current experiments are insufficient. It is suggested to include more baseline methods in the comparative results to make the claim of “As diffusion models expand beyond the familiar text-to-image paradigm to encompass multi-modal and multi-stage training for 3D and video synthesis, the reliability of existing detection methods in these novel domains remains unclear” more convincing. Currently, only 3 methods (LaVie, MVDream, and DiffSplat) are being experimented with in total for these two additional modalities. For example, a recent work [1] that focuses on memorization in the video diffusion models has conducted experiments on ModelScope, VideoCrafter, and RaMViD, in addition to LaVie, and validated that they are all subject to the memorization problem. Thus, it would be interesting to see if the metrics tested in this paper also underperform on these additional video diffusion models, which could potentially strengthen the claim. [1] Investigating Memorization in Video Diffusion Models. arXiv, 2025.
2. Some diagnostics (Table 2) rely on proxy estimators that can be noisy or implementation-dependent. The authors acknowledge this, but could better quantify uncertainty or validate proxies on controlled synthetic models.
3. The analysis in Sec. 3.3 tests twelve mitigation techniques, but this section lacks clear grouping or interpretation of which categories fail and why. A summarized taxonomy (gradient-based vs. attention-based vs. concept-ablation) and correlation with metric families would strengthen Section 3.3.
4. While the paper calls for “training-aware detection strategies,” it does not sketch concrete formulations or early prototypes. Including one proof-of-concept adjustment could make the contribution more forward-looking.
5. The paper fails to completely follow the ICLR template, as the abstract is over-wide.

### Questions
Please check out the strengths and weaknesses sections.

### Soundness
2

### Presentation
3

### Contribution
3

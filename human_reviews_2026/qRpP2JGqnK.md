# Prominence-Aware Artifact Detection and Dataset for Image Super-Resolution

- Avg Score: 3.33
- Decision: Reject
- Scores: 2, 4, 4

## Abstract
Generative image super-resolution (SR) is rapidly advancing in visual quality and detail restoration. As the capacity of SR models expands, however, so does their tendency to produce artifacts: incorrect, visually disturbing details that reduce perceived quality. Crucially, their perceptual impact varies: some artifacts are barely noticeable while others strongly degrade the image. We argue that artifacts should be characterized by their \emph{prominence} to human observers rather than treated as uniform binary defects. Motivated by this, we present a novel dataset of 1302 artifact examples from 11 contemporary image-SR methods, where each artifact is paired with a crowdsourced prominence score. Building on this dataset, we train a lightweight regressor that produces spatial prominence heatmaps and outperforms existing methods at detecting prominent artifacts. We release the dataset and code to facilitate prominence-aware evaluation and mitigation of SR artifacts.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a new dataset and detection method for artifacts in single-image super-resolution (SISR). The authors introduce a dataset consisting of 1302 artifact examples labeled with human-annotated prominence scores and propose a lightweight MLP-based regressor to predict spatial prominence heatmaps. The method uses features from existing quality metrics (e.g., DISTS, LPIPS) and demonstrates improved performance over prior work in detecting and mitigating prominent artifacts. Evaluations are conducted across multiple SR models, with applications shown in artifact detection and model fine-tuning.

### Strengths
1. The paper addresses a practical issue in SISR, i.e., artifact perceptual prominence, and proposes a quantifiable formulation to capture it.
2. The dataset construction is thorough with a well-documented crowdsourcing setup and a clear comparison to existing datasets like DeSRA.
3. The method is simple and effective, combining known perceptual features with a lightweight regressor, and shows consistent improvements across several baselines.
4. The evaluations are extensive, covering detection metrics, subjective studies, SR model fine-tuning, and ablation experiments.

### Weaknesses
1. The paper presents an incremental work over the DeSRA framework, with the main change being the shift from binary artifact labeling to annotated prominence scores. While more detailed, this change does not introduce fundamentally new ideas or techniques.
2. The method is heavily engineering-driven and lacks technical novelty. It relies on a linear regression-style approach using features from existing methods without proposing new representations or insights into artifact formation.
3. The framing of "prominence" as a perceptual score is intuitive but not conceptually deep. It does not provide theoretical understanding or modeling of artifact causes or their interactions with SR architectures.
4. Generalization is a major concern. The model regresses directly on human-annotated prominence values based on a relatively small dataset, raising the risk of overfitting. There is no convincing evidence that the method generalizes to unseen artifact types or datasets beyond DeSRA and their own collection.
5. The reliance on pseudo-ground-truth (e.g., RLFN) as a reference in full-reference setups introduces potential biases, especially given that these reference models may fail to preserve valid fine textures.

### Questions
1. Given the limited scale and diversity of the dataset, how do the authors ensure that the model generalizes beyond the specific SR models and artifact types seen during training?
2. Artifact prominence is subjective and context-dependent. How robust are the predictions to variations in viewer demographics, viewing conditions, or image content categories?

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
5

### Summary
This paper aims to address visual artifacts in super-resolution (SR). The authors first introduce a new artifact dataset in where each image is annotatated with a prominence score. Building on this, they propose a detector to predict the prominence score. Finally, they demonstrate the detector's utility by using the predicted results to guide the tuning of existing SR models.

### Strengths
1. The core idea of the paper is reasonable. The introduction of a "prominence score" for artifacts is a valuable contribution, offering a more flexible and nuanced evaluation metric than traditional binary masks.

2. The proposed artifact detector is validated effectively, as the experimental results confirm its utility and performance.

### Weaknesses
1. The paper's core premise is contradictory. It advocates for the nuanced "prominence score", yet the proposed model fine-tuning (ft) process appears to rely on a binary mask to guide its replacement strategy. The authors fail to explain how this binary mask is derived from the prominence score (e.g., what threshold is used?). This is a critical omission that undermines the necessity of the prominence score itself. If the end application only requires a binary decision, the justification for not using a simpler, binary-annotated dataset is weak.

2. The paper's evaluation is kind of misaligned: it mainly focuses heavily on artifact detection accuracy, but fails to demonstrate that a better detector (or by using prominence score) actually leads to a better final SR image. Therefore, I recommend the author to:

a) Show more qualitative SR results between different artifact detection methods.

b) Provide standard SR metrics (e.g., PSNR, SSIM, LPIPS or others) and/or a user study on these final SR images to quantify the improvement.

3. The DeSRA result in Table 7 is highly confusing. Why does fine-tuning with DeSRA's detected binary mask drastically increase artifacts? This is baffling, especially since the authors' own method also uses a binary mask for fine-tuning. This result needs a clear explanation, preferably supported by qualitative examples

### Questions
1. Why is "Ours" labeled as "automatic" in Table 1? Doesn't line 160 state that the artifact masks are "manual annotations"?

2. What does "PixFrac" in Table 8 represent?

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
4

### Summary
The paper proposes a new prediction task, “prominence” score, for super-resolution (SR) artifacts detection. In addition to the binary mask of the artifact region of a SR result, the paper estimate its prominence score via a protocol to average over 30 raters with quality control. The paper introduces not only a bigger dataset than the prior art, but also a lightweight baseline method to regress the prominence score. The experiments show that the proposed baseline achieves SOTA performance and drives effective SR fine-tuning for artifact reduction.

### Strengths
- Task novelty: Traditional binary annotation can be ambiguous given the nature of the task is to some degree subjective. This paper introduces the prominence score to alleviate the annotation issue and to effectively drive SR fine-tuning for artifact reduction.
- Good label quality. The annotation protocol and the quality control analysis are solid, demonstrating the reliability of the prominence score estimation despite the ambiguity nature of the task.
- Comprehensive experiments: Practical impact for benchmarking SR artifacts and for reducing artifacts via fine-tuning; the method achieves top average PR-AUC ranking and best fine-tuning rank.

### Weaknesses
- Model novelty is limited. The core is an MLP over existing features. This is a reasonable engineering contribution but may underwhelm for ICLR. Its improvement over DITS is incremental. It'll be great to add more analysis (qualitative examples) on where it's better or worse than other methods.
- Dataset image selection: it can be better to do a stratified selection of images from the 9M images in Open Images. E.g. sampling from different semantic categories (objects, stuff) or different frequency spectrum patterns. Right now, the dataset is biased toward low prominence due to seeding, which not only skew training and evaluation but also lack insights on where do artifacts happen.
- Need sensitivity studies on \kappa and mask dilation, which seem to affect the rankings much. It'll be good to quantify their impact on PR-AUC/IoU and fine-tuning outcomes. 
- Need sensitivity studies on Pseudo-GT. Using RLFN as pseudo-GT may introduce bias. It'll be good to compare with other SR methods and provide failure cases where pseudo-GT introduces artifacts.

### Questions
- Feature choices: why this exact 3-feature set? Provide a principled selection study and show performance when adding complementary cues (e.g., texture statistics) or removing each component beyond the provided ablations.
- Robustness and generalization: In Appendix E, DITS seems to have better generalization results. Does it mean that the proposed baseline overfits to the training dataset? What are the possible causes for the performance gap?

### Soundness
3

### Presentation
3

### Contribution
3

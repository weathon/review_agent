# Only Brains Align with Brains: Cross-Region Alignment Patterns Expose Limits of Normative Models

- Decision: Accept (Poster)
- Scores: 6, 4, 4

## Abstract
Neuroscientists and computer vision researchers use model–brain alignment benchmarks to compare artificial and biological vision systems. These benchmarks rank models according to alignment measures such as the similarity of representational geometry or the predictivity of neural responses from model activations. However, recent works have raised a number of problems with these rankings, most critically their lack of discriminative power, raising the conceptual question of what it means for a model to be ''brain-aligned''.
Here we introduce *alignment patterns* - characteristic functional relationship profiles of each brain region to all others - and propose that models should reproduce these patterns to qualify as brain-aligned. 
First, we apply a standard benchmarking pipeline to a broad spectrum of vision models on the BOLD Moments video fMRI dataset across visual regions of interest (ROIs). 
We find diverse models appear *equivalent* in their brain alignment, reflecting the lack of discriminative power of conventional alignment benchmarks.
Conventional alignment evaluation is a pointwise similarity test: it assesses whether a model is aligned to an individual ROI. It is therefore sensitive to the specific invariances and scaling properties of the chosen metric. In contrast, *alignment pattern analysis (APA)* is a second-order *structural consistency* test: a model aligned to a given ROI should reproduce that ROI’s characteristic cross-region alignment profile. 
Applying this test, we find that, while these patterns are highly stable across brains of different subjects, even top-ranked models often fail to capture them. Notably, models that appear effectively equivalent in alignment diverge sharply under the relational criterion, demonstrating the added discriminative value of APA.
Finally, we argue for a clearer distinction between the criteria a model must meet to serve as a tool versus as a computational model. Conventional alignment measures may be sufficient for identifying neurally predictive models, but claims about computational or algorithmic similarity may require a stronger basis of evidence, including the reproducibility of relational alignment patterns.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper benchmarks 47 image and video models on BOLD-Moments fMRI with RSA and linear predictivity, finding that rankings depend on the metric and many models are practically equivalent within subject variability. It proposes Alignment Pattern Similarity, grounded in structural connectivity, to test whether models preserve each ROI’s cross-region similarity profile, and reports that brains align with brains while models generally fail to match these patterns, arguing for stricter anatomy-informed evaluation.

### Strengths
- The authors raise a very important and timely issue regarding AI-model alignment.
- The authors propose a solution that I think is quite novel, which is checking the inter-regional similarities between brain regions and with one region replaced by its best-fit model.
-  Showing that most models fall within subject-level variability is a much-needed report.
-  I personally find the high APS for brain-brain alignment very interesting (but could the author please provide the baseline elaborated in the weaknesses section below?).

### Weaknesses
- I have a philosophical, high-level concern. I am not sure if requiring the model representation (that has a best fit to one brain region) to have relationships to other brain regions in a way that one brain region relates to those brain regions is an unnecessarily strict requirement.  But I do think it is still an interesting thing to check and report, which could be the author’s point.

- In this line of thought, I think another interesting analysis would be
1. First, (within a single model) measure the similarity between layers that have high alignments to brain regions of interest. Call that s’_model_RSA(i,j) (or s’_model_LP) for brain regions i and j. For example, say layer 3 corresponds to ROI 1 of the brain, and layer 5 corresponds to ROI 2 of the brain. Then s’_model_RSA(1,2) measures the similarity between layer 3 and layer 5 of the networks.
2. Measure and report R^2 between S’_model_RSA and brain-to-brain S_RSA (the latter one is from the paper e.g. Figure 5d top left red line).
Would it be easy to perform this analysis? This is less strict than the author’s requirement. Qualitatively speaking, it is like saying we don’t care exactly how layers talk to each other (and how the ROIs talk to each other), and the only thing we care about is the relationship (RSA/LP value) between these layers and the relationship (RSA/LP value) between these ROIs. We might want these relationships to be similar (high R^2) between the model and the brain.

- Perhaps more importantly, I think the authors should report the baseline for APS, where the ROI connectivity graph is randomly generated.

- I understand that the paper's idea and methodology are not easy to explain, but I think the clarity of the paper can be dramatically improved. Figure 1 is well-intended, but it only makes sense after reading the entire paper. As a reader, it is especially unclear what the cartoon plots under "Alignment Patterns" indicate (one would wonder that the x-axis and y-axis are), unless they have read section 3.5 (which I don't think is also clearly written). I am not clear what the best way is to explain the overall pipeline, but I don't think Figure 1 helps much at all.  In section 3.5, the last sentence was nearly impossible to understand.

### Questions
Please see the weaknesses section for the question.

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
This paper evaluates 47 vision models on the BOLDMoments video fMRI dataset using RSA and linear predictivity (LP) to assess brain-model alignment. The authors identify V-JEPA2 as achieving strongest alignment but show that many models are "practically equivalent" within subject-level variability. They introducxe Alignment Pattern Similarity (APS), a novel metric that compares cross-region alignment patterns to anatomical connectivity patterns. The key finding: while normative models can align well with individual brain regions, they fail to reproduce brain-to-brain cross-region alignment patterns, revealing a fundamental limitation of current approaches.

### Strengths
1. **Important conceptual contribution:** APS is a genuinely novel idea that addresses a real gap in the field. The insight that models should preserve relational structure across regions (not just match individual regions) is valuable and biologically motivated.
2. **Comprehensive model evaluation:** Testing 47 models spanning diverse architectures (CNNs, Transformers), objectives (supervised, self-supervised, multimodal), and modalities (image, video) on a large video dataset is thorough.
3. **Strong empirical results:** The finding that brain-to-brain APS is high while model-to-brain APS is near-zero is an interesting result.

### Weaknesses
1. **Figure quality is severely lacking:** The sketch-based design of figure 1 is not ideal for a publication at a top tier conference. I would recommend redoing it with vector graphics. The other figures have almost illegible axis labels with overlapping/compressed model and ROI names. 
2. **Limited to RSA and LP Despite Known Limitations:** Paper correctly identifies that "conclusions depend strongly on choice of metric" yet only uses RSA and LP. Missing modern metrics like CKA[1], RTD[2] and NSA[3]
3. **APS Validation is Incomplete:** No null model is presented. What is expected APS for random alignment patterns? No p-values or significance tests. Could spatial proximity or retinotopy explain results instead of connectivity? Does APS generalize to NSD or HCP 7T?
4. **Model-Brain APS Results Are Hard to Interpret:** Low APS could mean: (a) models fundamentally limited, (b) wrong layer selection, or (c) APS too strict but the authors assert (a) without ruling out the other options. Layer-wise analysis of APS is also missing.

### Questions
Please refer to weaknesses.

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
This paper benchmarks tens of image and video models on the BOLD-Moments fMRI dataset using Representational Similarity Analysis (RSA) and Linear Predictivity (LP), argues that many models are practically equivalent despite their rankings because their model→brain alignment falls within the distribution of brain↔brain alignment, and introduces Alignment Pattern Similarity (APS)—a connectivity-grounded test of whether a model preserves each ROI’s cross-region similarity pattern. The key claim is that, although some models score well on RSA/LP, they generally fail to reproduce brain-to-brain cross-region patterns under APS, whereas brains do.

### Strengths
- The paper tackles the growing concern that conclusions about “brain alignment” depend on metric choice, and proposes a relational criterion that goes beyond local fits.
- The practical-equivalence analysis is a strong way to assess when model-ranking differences are statistically meaningful.
- Creative attempt to incorporate anatomical priors and cross-region dependencies into model evaluations.
- It's nice that the paper includes a diverse set of image and video models across supervised and self-supervised objectives.

### Weaknesses
- My biggest concern is that the three aims—(1) benchmarking image/video models on the BOLD Moments dataset, (2) demonstrating practical equivalence of several models (in the sense that they lie within the brain-brain similarity range), and (3) introducing APS—feel only loosely connected. Each of these components could stand as an independent contribution, but when presented together, the narrative lacks a clear causal or conceptual throughline. For instance, the benchmark and equivalence analyses establish the limitations of RSA/LP metrics but are not explicitly framed as motivating APS; instead, APS appears as a parallel idea rather than a methodological extension.
- The practical-equivalence conclusion hinges on the brain↔brain baseline, which is constructed by averaging voxel responses across subjects before comparing to a held-out subject. This raises concerns about functional alignment (are voxels/topographies aligned across individuals?) and the arbitrariness of cross-subject averaging. The authors should: (i) report within-subject baselines, (ii) test hyperalignment or response-based alignment, and (iii) evaluate subject→subject prediction without averaging. Related ideas have appeared in the NeuroAI Turing Test (Feather et al., 2025), which formalizes a similar distributional criterion for model evaluation; citing and differentiating from that work would clarify novelty.
- The phrase “shifting the focus from single-metric rankings to the stability of rankings” is confusing, since two metrics (RSA, LP) are still used. The authors actually examine whether model orderings are stable given subject variability—i.e., when multiple models are within the brain↔brain alignment range. This should be explicitly defined early to avoid misinterpretation.
- The statement that “recent benchmarks of video models have relied on a single alignment metric, while metric-comparison studies rarely include modern video architectures” should be supported with citations

### Questions
- How sensitive are the practical-equivalence findings to the brain↔brain baseline (averaged vs. individual subject predictors; with/without hyperalignment; within-subject splits)?
- Could you compare your equivalence criterion to the NeuroAI Turing Test (Feather et al., 2025) to delineate conceptual differences and overlapping assumptions?

### Soundness
3

### Presentation
2

### Contribution
3

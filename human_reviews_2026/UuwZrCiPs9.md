# ScoreMix: Synthetic Data Generation by Score Composition in Diffusion Models Improves Recognition

- Avg Score: 4.67
- Decision: Reject
- Scores: 4, 4, 6

## Abstract
Synthetic data generation is increasingly used in machine learning for **training and data augmentation**. Yet, current strategies often rely on external foundation models or datasets, whose usage is restricted in many scenarios due to policy or legal constraints. We propose **ScoreMix**, a **self-contained** synthetic generation method to produce hard synthetic samples for recognition tasks by leveraging the score compositionality of diffusion models. The approach mixes class-conditioned scores along reverse diffusion trajectories, yielding domain-specific data augmentation without external resources. We systematically study class-selection strategies and find that mixing classes distant in the discriminator’s embedding space yields larger gains, providing **up to 3\% additional average improvement**, compared to selection based on proximity. Interestingly, we observe that condition and embedding spaces are largely uncorrelated under standard alignment metrics, and the generator’s condition space has a negligible effect on downstream performance. Across **8 public face recognition benchmarks**, ScoreMix improves accuracy by **up to 7 percentage points**, without hyperparameter search, highlighting both robustness and practicality. Our method provides a simple yet effective way to maximize discriminator performance using only the available dataset, without reliance on third-party resources. _Code and synthetic datasets are available._

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces ScoreMix, a self-contained synthetic data augmentation method that mixes class-conditioned diffusion scores along the reverse trajectory to synthesize hard in-domain examples for recognition. The setup trains both generator and discriminator on the target data without external models or datasets, then augments the real training set with ScoreMix samples. Experiments focus on face recognition. The idea is clear and practical, but the empirical scope is narrow.

### Strengths
1 This paper proposes practical and simple recipe once a class-conditional diffusion model is trained, avoiding external data or models.

2 Empirical results show consistent improvements across multiple face benchmarks with an actionable rule for choosing distant class pairs in embedding space. 

3 Useful analysis that clarifies why reproducing samples are not effective for improving downstream recognition performance and why embedding geometry should guide mixing.

### Weaknesses
1. The paper’s scope is confined to face recognition, leaving transfer to other recognition domains untested. I understand the focus is low-data regimes, but that still includes domains like medical imaging, retail product IDs, species recognition, and industrial parts where manifold structure is different and the current evidence does not generalize.

2. The finding that mixing more than two classes brings little benefit is reported and interesting but there is little analysis on top of it. It would be better if include some analysis on the failure mode.

3. Robustness to sampler configuration is unclear. As a key factor in diffusion model inference, parameters such as the number of sampling steps, noise schedule, and the guidance strength can affect generation quality, yet their impact in this setting remains unclear

### Questions
1. How sensitive are results to guidance strength, noise schedules, and the pair-selection heuristic under fixed compute?
2. Does the method transfer to non-face domains with potentially different data manifolds?

Please also refer to the weakness section

### Soundness
3

### Presentation
3

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
This propose ScoreMix, a self-contained synthetic generation method to produce hard synthetic samples for recognition tasks by leveraging the score compositionality of diffusion models. The approach mixes class-conditioned scores along reverse diffusion trajectories, yielding domain-specific data augmentation without external resources.

### Strengths
1. This paper develops a self-contained augmentation strategy—that is, one that does not rely onexternal datasets, commercial APIs, or third-party models—to maximize the performance of state-of-the-art discriminators solely with the available data.
2. This paper demonstrates that convex combinations of classconditioned scores yield synthetic samples that consistently improve discriminator training.

### Weaknesses
My main concern regarding this paper's motivation lies with its core premise: leveraging synthetic data for augmentation, particularly within the sensitive domain of facial data. I question the fundamental viability of this approach. Specifically, wouldn't introducing synthetic data risk confusing the model and ultimately compromise its robustness, especially when considering critical applications like face anti sproofing?
1. Given the widespread availability of high-quality, pre-trained generative models (e.g., Stable Diffusion, Midjourney), what is the efficacy of a simpler baseline approach that uses these off-the-shelf models directly for sample generation? A comparative analysis would be valuable to contextualize the performance gains of the proposed method.
2. The proposed methodology draws a strong parallel to techniques in generative model-based image editing. On that note, have the authors considered leveraging established image editing methods—which utilize techniques like model inversion or inversion-free editing—as an alternative for data augmentation? A discussion on the relative merits and performance of such an approach would be insightful.
3. Regarding the computational cost trade-offs: The paper states that ScoreMix has approximately double the computational cost of AugGen. To facilitate a comprehensive evaluation of the method's practical viability, could the authors provide a more detailed quantification of the end-to-end pipeline's cost?
4. The images have 'AI look'. I have two questions about that. First, I doubt these images would pass a face anti sproof test? Second, and more importantly, does training your classifier on these 'AI-flavored' images make it easier for other AI-generated pictures to fool it in a real-world application. 
5. Have you considered applying this method to long-tailed recognition problems? It sounds like it would be a great way to generate high-quality images for the rare classes that have very few samples.
6. You state that your method creates 'hard samples,' but what exactly makes them 'hard'? Are they hard because they look blurry or distorted, or are they hard because they are confusing for the classifier, sitting right on the decision boundary?

### Questions
Please see the weakness.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes ScoreMix, a self-contained synthetic data augmentation method for recognition that linearly mixes class-conditioned scores inside a diffusion model during reverse sampling. Both generator and discriminator are trained from scratch on the same dataset, avoiding external models/datasets. On 8 FR benchmarks, ScoreMix improves accuracy (up to +7 pp) and even beats scaling the backbone in some cases. The authors analyze which class pairs to mix, finding that selecting distant classes in the discriminator’s embedding space yields the biggest gains, whereas distances in the generator’s condition space are weakly correlated and largely unhelpful. They also provide a robustness argument connecting CKA alignment to order preservation for pair selection, and report that aligning generator outputs to class centers can reduce diversity and hurt performance.

### Strengths
* Clear, simple mechanism with strong intuition. Convex score mixing is well-motivated to preserve score magnitude and remain on-manifold; qualitative grids and discussion illustrate why non-convex weights can fail.

* Self-contained augmentation. Training both generator and discriminator only on the available dataset is practically appealing in sensitive domains.

* Consistent empirical gains. Across FR benchmarks, ScoreMix improves over training on real data alone and beats a larger IR101 baseline on the same data; gains up to +7pp are reported without hyper-parameter search.

* Negative/neutral results are also reported. Mixing >2 classes and explicitly aligning to class centers are shown to be ineffective or harmful (Lines 444-446), with analyses of intra-class similarity vs. fidelity.

### Weaknesses
* The authors note ScoreMix “roughly doubles” sampling cost vs. AugGen (Line 269). Please explicitly quantify: GPU hours for generator training + sampling per 0.2M synthetic images, versus baselines (e.g., AugGen), and the cost of computing embedding distances & m-plet mining. Without cost curves, practicality is hard to judge.

* Class-pair selection uses distances from a trained discriminator. How sensitive are gains to the quality/architecture of that initial model? If we re-select pairs using a different backbone/head, do results hold (beyond the CKA bound), and does iterative re-selection risk overfitting to the embedding geometry?

* The CKA result is insightful but a bit informal here. What concrete guidance does it give a practitioner (e.g., target CKA>$\tau$ before selecting pairs)? If assumptions (energy-matched Gaussian misalignment) are violated, do we still see monotone trends? A short empirical map from CKA values to expected gains would help.

* The authors argue that ScoreMix’s gains come specifically from mixing in score space during diffusion sampling. Beyond the compatitive AugGen baseline, consider within-model mixup/cutmix in feature space, latent space interpolations in the diffusion prior, or curriculum-hardening on real pairs (no synthesis). Also consider compare to label-mixing in condition space tuned by grid search (your critique of AugGen) under equal compute. Current baselines make it hard to attribute gains purely to score-space mixing.

* Minor issue: The paper focuses specifically on FR, yet the title and abstract adopt a broader “recognition” framing until the empirical results part (around line 025). This over-generalization may lead readers to infer applicability to broader recognition domains (e.g., fine-grained species or object recognition), which is not substantiated by the presented results. A more precise and domain-specific framing in the title and abstract is therefore recommended.

### Questions
Please address all issues raised in Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

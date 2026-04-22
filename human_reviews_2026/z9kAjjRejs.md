# Decoding Dynamic Visual Experience from Calcium Imaging via Cell-Pattern-Aware Pretraining

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 2, 4, 6, 6

## Abstract
Neural recordings exhibit a distinctive form of heterogeneity rooted in differences in cell types, intrinsic circuit dynamics, and stochastic stimulus–response variability that goes beyond ordinary dataset variability, mixing statistically regular neurons with highly stochastic, stimulus-contingent ones within the same dataset. This heterogeneity poses a challenge for self-supervised learning (SSL)—learnable statistical regularity—thereby destabilizing representation learning and limiting reliable scaling.
We introduce POYO-CAP (Cell-pattern Aware Pretraining), a biologically grounded hybrid pretraining strategy that first trains with masked reconstruction plus lightweight auxiliary supervision on statistically regular neurons—identified via skewness and kurtosis—and then fine-tunes on more stochastic populations.
On the Allen Brain Observatory dataset, this curriculum yields 12–13\% relative improvements over from-scratch training and enables smooth, monotonic scaling with model size, whereas baselines trained on mixed populations plateau or destabilize. By making statistical predictability an explicit data-selection criterion, POYO-CAP turns neural heterogeneity into a scalable learning advantage for robust neural decoding.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
- This paper considers the problem of decoding relevant signals from neuronal activity recordings. The studied tasks aim to decode the presented visual input or its statistics from concurrent recordings of neuronal activity from the mouse visual cortex.

- The paper argues that pre-sorting of neurons into 'predictable' and 'unpredictable' sets is useful for curriculum learning: the proposed deep model is first trained on a simple task and using the predictable subset. Then, the unpredictable subset is used to learn a harder task. A neuron is labeled as predictable if the 3rd and 4th order moments of its temporal activity statistics are low.

- The paper proceeds to show that this way of curriculum training improves task performance and conducts ablation studies to support this finding.

### Strengths
- Curriculum training is not a new idea. Beyond a curriculum of tasks (easier to harder), the manuscript also argues for a curriculum of samples during training. This is a nuanced point and can perhaps still be considered within the classical curriculum training idea. However, its application to large-scale neural data analysis is novel.

- The manuscript classifies a neuron as predictable and unpredictable based on moments of its own activity, which is easy to compute. It also shows that it corresponds to certain neuronal subsets with molecular underpinnings.

- The paper suggests that it will be harder to decode the signal of interest from neurons whose activity is more variable (high 3rd, 4th moments), both because the task loss is larger (I didn't understand why only the first quadrant of top-2 PCs is shown in Fig. 2.) and those neurons have higher Fisher information for the task.

### Weaknesses
- A main weakness of the manuscript is its claim on high-fidelity movie reconstruction from neural recordings. This would indeed be a major advance. However, my understanding is that the reconstruction is for previously seen video fragments. For such a claim, reconstruction should be demonstrated on previously unseen scenes. As is, I believe the manuscript significantly over-claims. (If my understanding is wrong, I am willing to substantially increase my score.)

- The nuance introduced to curriculum training in this manuscript (a curriculum of data samples) is not a significant enough contribution that warrants publication.
    - The generalizability of this approach is not tested. It is not clear whether this would be helpful in analyzing other neuroscience datasets (e.g., non-visual) or accomplishing benchmark machine learning tasks.
    - It can be considered as a natural variant although I don't know if this curriculum was explicitly studied before. This does not strike me as a significant enough contribution to the curriculum training paradigm.

- I don't think the scalability argument surrounding Figure 5 is established in a convincing way. (All plots trend upwards and the y-axis is zoomed-in.) I think there is insufficient evidence to conclude that the slope of the linear fit will be larger for any one model. (e.g., the conclusion depends strongly on the set of points used for such a fit.)

- Statistical Regularity Hypothesis: I believe this is a hypothesis that the authors are putting forth. (Please clarify.) Self-supervised learning will obviously and clearly work better with data with statistical regularity. (e.g., it is not possible to learn noise.) That is, masked reconstruction accuracy will be higher for more predictable neurons. However, whether this extends to supervised (task) performance following masked learning is not clear and I believe that is what the authors want to propose. Predicting the orientation of the drifting grating or the reconstruction of movie frames are both supervised tasks. I think this hypothesis needs a major rethinking and multiple qualifying statements may need to be added.

- Section 3.3 presents various numerical analyses, not theoretical analyses. Please consider renaming.

### Questions
- Could you please expand the field of view of the two plots in Figure 2 so the loss at the boundaries reaches high values?

- In Figure 2, does the loss correspond to the task reconstruction loss (drifting gratings? movie frame reconstruction?) or the masked loss (self-supervised)?

- Was the knee-detection algorithm applied for each Cre-line? If so, why not apply it to individual neurons and admit neurons into the predictable set based on their own scores rather than based on the Cre-line they belong to?

- (line 147) What does "near-Gaussian activity" mean?

- How would the proposed POYO-SSL model perform if 'Finetune Data' is set to 'All' in Table 3?

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
The paper proposes POYO-SSL, a cell-pattern-aware self-supervised pretraining scheme for decoding dynamic visual experience from mouse calcium imaging. The key idea is to pretrain only on “predictable” neurons, identified a-priori via low skewness/kurtosis of calcium traces (knee thresholds: skew≤3.51, kurt≤22.62; CRE lines SST/VIP/PVALB/NTSR1), then fine-tune on the remaining “unpredictable” neurons for downstream tasks (movie reconstruction; drifting-grating orientation). On the Allen Brain Observatory, the method reports SSIM 0.593 for direct neural-to-movie reconstruction and 55.5% accuracy on gratings, outperforming a strong from-scratch POYO+ baseline with identical capacity. Ablations (architecture, data selection, masking/aux loss) and scaling plots are provided.

### Strengths
A priori selection of predictable neurons using skewness/kurtosis is transparent (knee-based thresholds) and applied consistently without leakage (animals/sessions/neurons disjoint).

Architecture capacity controls; data-diet variants (inhibitory-only, reverse, mixed); objective variants (temporal vs random masking; CE-only; weight sweeps). These help attribute where gains come from.

### Weaknesses
This paper relies mainly on POYO+-from-scratch leaves room for skepticism about general SOTA claims. Even if CEBRA/Neuro-BERT aren’t pixel decoders, an adapted masked-autoencoding baseline over calcium with the same U-Net decoder, or a temporal contrastive baseline (e.g., CPC-style) would help. At minimum, include a “random neuron subset (size-matched)” pretraining control to show skew/kurtosis selection matters beyond sample count and CRE composition (the paper has “mixed” and “reverse” but not “random size-matched”). 

The authors state that SSIM = 0.593 is the highest reported to date for direct visual reconstruction from cellular-resolution neural recordings. Note they contrast with fMRI works (SSIM 0.19/0.365) which are different modalities and not directly comparable.

In addition, the comparison to a capacity-matched POYO+ trained from scratch is fair and clean, and there are thorough ablations (encoder/decoder variants; inhibitory-only / reverse / mixed data diets; masking vs random masking; CE-weight sweeps). However, external SSL or generative baselines adapted to calcium-to-image decoding are not included (the paper argues popular SSL methods like CEBRA/Neuro-BERT don’t target pixel-level generation). For a flagship result, adding one or two adapted published methods (or a strong masked-autoencoder baseline over calcium with the same U-Net decoder).

### Questions
You define “predictable” neurons via a knee-detection on skewness/kurtosis and set fixed thresholds (skew ≤ 3.51, kurt ≤ 22.62). How sensitive are results to these cutoffs? Please provide a sweep (±10–20%) or cross-validated thresholds.

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
3

### Summary
●	This paper studies how to improve the benefits derived from pre-training for downstream decoding. The authors discover that curating the type of data used for pre-training is crucial to obtaining good scaling. Specifically, they propose to pre-train on neural data that can be said to be more regular, and then fine-tuning is done on data that is less regular. Here, statistical regularity is defined by the authors in terms of skewness and kurtosis. The pretraining objective is masked reconstruction. They find that this pretraining results in better downstream performance on a movie reconstruction task and in classifying a drifting grating.

### Strengths
●	This answers a question about why scaling sometimes stalls even as more pretraining data is added. This is useful for the community to know.
●	This paper provides an actionable lesson for anyone doing self-supervised learning: train on the easy-to-model data first.

### Weaknesses
●	I'm missing something very basic: why should this method work at all? If pre-training is done on the regular data first, how can the model ever learn good representations of the irregular data? Moving from one neuron type to another represents a distribution shift. What's the explanation for how the model is able to handle this shift?
●	If possible, could the authors please discuss connections to other domains where this strategy would be useful. For example, in language modeling, would it be better to pre-train on regular strings first?
●	The writing could be clearer. See questions below. The methods section, particularly 3.2.2. could be more clear in many places if plainer language were used. For example, take line 205: "We employ a curriculum learning approach combining masked reconstruction with weak supervision for stable representation learning. Our weakly-supervised auxiliary loss relies on simple visual primitives (drifting gratings) as a curriculum warm-up before moving to a complex downstream movie decoding task." In my opinion, it would be easier to understand the following sentence: "During pre-training, the model is trained on a joint objective, consisting of self-supervised masked reconstruction and fully-supervised classification." I don't believe it is correct to use the term "weak-supervision" here, since labels are available for all the training examples. And I think it's more common to simply refer to this type of training as "pre-training and fine-tuning" rather than "curriculum learning".

### Questions
●	Line 146: This is a basic question, but what are the skewness and kurtosis being taken with respect to? The distribution of calcium values? Across what period of time?
●	Section 4.1: When results are reported, for example in Table 3, what neurons are being decoded from? Only the unpredictable neurons? What I'm trying to get at is this: Is the same evaluation data used across all experiments? If not, how can performances between experiments be compared as in Table 3?
●	I wonder, if regular activity is more beneficial for pre-training, would it be even better to produce synthetic calcium traces for pre-training that have even more regularity? Do you think this would further improve performance?

Minor points:
●	line 190: is this a typo? Should it read: "the latent representation of the unmasked view is then used as a target for the latent representation of the unmasked variant." ? That would make it fit with line 204.
●	Figure 1: What should the axis titles be on the calcium trace plots?
●	Figure 5: The caption mentions "orange", but the corresponding line is yellow.

### Soundness
3

### Presentation
2

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
This paper introduces POYO-SSL, a self-supervised learning framework that addresses neural data heterogeneity by pre-training on statistically predictable neurons identified via skewness and kurtosis metrics before fine-tuning on unpredictable populations. Applied to calcium imaging data from the Allen Brain Observatory, the method achieves 12–13% relative gains in decoding dynamic visual experiences, enables high-fidelity movie reconstruction without external stimuli, and demonstrates stable scaling with model size, turning neural variability into an asset for robust representation learning.

### Strengths
1. Biologically Grounded Data Selection: The method innovatively leverages higher-order statistics to prioritize predictable neurons (e.g., inhibitory interneurons like SST/VIP), transforming neural heterogeneity from a challenge into an advantage for stable pre-training and improved data efficiency (1.98×gain).  

2. Scalable and Task-Adaptive Architecture: POYO-SSL enables monotonic performance scaling with model size and supports diverse decoders (e.g., Skip-Connection U-Net for movie reconstruction), achieving high-fidelity results without task-specific labels.

### Weaknesses
1. The method uses skewness ≤ 3.51 and kurtosis ≤ 22.62 as thresholds to partition neurons into predictable and unpredictable groups via a knee-detection algorithm. However, these thresholds are applied as a fixed, universal criterion without sensitivity analysis or validation across diverse datasets. The paper states: "These thresholds were determined a priori as a single, fixed criterion to partition the dataset, not as a tunable hyperparameter, which is why a sensitivity analysis was not performed". This risks overfitting to the Allen Brain Observatory dataset and limits generalizability. A robustness analysis (e.g., varying thresholds) would strengthen the approach.

2. The skip-connection U-Net decoder is introduced for high-fidelity movie reconstruction but lacks ablation studies comparing it to other decoder designs (e.g., transformers). The description focuses on architectural choices without quantifying their individual contributions: "Our new U-Net-inspired decoder generates frames from a single neural embedding... See Appendix F for more details". Without isolating the decoder’s impact, it is unclear whether gains stem from the architecture or the pre-training strategy.

3. The experiments compare POYO-SSL to a from-scratch baseline and POYO+ but omit broader comparisons with state-of-the-art methods like CEBRA or Neuro-BERT, arguing their architectures are not suited for direct high-fidelity visual reconstruction. However, this justification appears in an appendix, and the main text does not discuss adaptations or partial comparisons (e.g., feature extraction). This narrow scope may overstate POYO-SSL’s advantages.

4. The scaling analysis (Figure 5) shows performance gains with model size but uses a limited range of capacities. The paper notes: "Our main approach (red) unlocks consistent performance gains as model capacity increases" , yet no details are provided about the maximum size tested or computational constraints. Expanding the scale range would better validate the claimed monotonic scaling.

5. The theoretical analysis (e.g., loss landscape, Fisher Information) relies on projections and approximations without uncertainty quantification. For instance, the loss landscape roughness metrics ($\sigma_L$) are derived from smoothed visualizations , which may hide variability.

6. The paper emphasizes biological grounding but does not validate neuronal predictability against ground-truth cell-type properties beyond statistical correlations. While skewness/kurtosis align with inhibitory/excitatory roles, causal experiments are absent.

### Questions
Weaknesses

### Soundness
3

### Presentation
3

### Contribution
3

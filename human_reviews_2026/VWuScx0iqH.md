# Cross-Subject Integration of Multi-Region Neural Signals via Functional Embedding.

- Decision: Reject
- Scores: 2, 4, 4, 6

## Abstract
Aggregating intracranial recordings across subjects is challenging since electrode count, placement, and covered regions vary widely. Spatial normalization methods like MNI coordinates offer a shared anatomical reference, but often fail to capture true functional similarity, particularly when localization is imprecise; even at matched anatomical coordinates, the targeted brain region and underlying neural dynamics  can differ substantially between individuals. We propose a scalable representation-learning framework that (i) learns a subject-agnostic functional identity for each electrode from multi-region local field potentials using a Siamese encoder with contrastive objectives, inducing an embedding geometry that is locality-sensitive to region-specific neural signatures, and (ii) tokenizes these embeddings for a transformer that models inter-regional relationships with a variable number of channels. We evaluate on a 20-subject dataset spanning basal ganglia–thalamic regions collected during flexible rest/movement periods with heterogeneous electrode layouts. The learned functional space supports accurate within-subject discrimination and forms clear, region-consistent clusters; it transfers zero-shot to unseen channels. The transformer, operating on functional tokens without subject-specific heads or supervision, captures cross-region dependencies and enables reconstruction of queried channels, providing a subject-agnostic backbone for downstream decoding. Together, these results indicate a path toward large-scale, cross-subject aggregation and pretraining for intracranial neural data where strict task structure and uniform sensor placement are unavailable.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes a data-driven functional embedding method for intracranial neural recordings, aiming to facilitate large-scale, cross-subject modeling where electrode placement is heterogeneous. Traditional approaches relying on MNI coordinates often fail to align functionally similar sites across subjects. The authors instead learn embeddings using two contrastive learning frameworks supervised by the anatomical region labels of sEEG channels. They show that the resulting embeddings can effectively separate channels from different regions and improve cross-region reconstruction compared to using MNI coordinates alone within a functional transformer model.

### Strengths
The paper addresses an important and timely problem — how to enable large-scale, cross-subject pretraining for intracranial neural data without relying on consistent electrode placement or strict task structure. The motivation is clearly articulated, and the approach is well-motivated within the context of current larger-scale model trends. The paper is clearly written and systematically explores the embedding space through multiple analyses, showing that the learned representations are meaningful with respect to anatomical regions and can be helpful with the cross-region reconstruction.

### Weaknesses
While the proposed method is carefully executed, the experiments primarily demonstrate that a contrastive model can learn to distinguish signals from different regions — a somewhat expected outcome given the supervision setup. It remains unclear whether such region-discriminative embeddings actually benefit the intended downstream goal of cross-subject model pretraining and decoding.

The relatively poor performance on held-out channels (Fig. 4) raises concerns about generalization, particularly across subjects, which is critical for the stated motivation. The apparent trade-off between overall improvement and degraded performance on high-performing samples in the cross-subject aggregation experiment suggests that the observed “gain” might merely reflect a redistribution of performance rather than a genuine improvement.

Finally, the evaluation remains indirect: improved reconstruction loss does not necessarily translate into better decoding or behavioral prediction performance. Demonstrating downstream improvements using the pretrained functional transformer would greatly strengthen the paper’s claims.

### Questions
- Could the authors clarify the training setup for the masked-region reconstruction experiment? Specifically, were the functional embedding and functional transformer trained on the same dataset, and what was the training/validation/test split? If embeddings were computed using test data, this could lead to data leakage and overestimated performance.

- Have the authors evaluated the method on held-out subjects to assess cross-subject generalizability on real data?

- Would a model with learnable region embeddings or MNI-coordinate–to–embedding mapping perform comparably in the reconstruction task? Such baselines might test whether the benefit arises from data-driven embedding learning or simply the inclusion of region information.

- Since reconstruction loss is not a direct measure of downstream decoding ability, can the authors provide results on a representative decoding task using the pretrained functional transformer?

Minor questions:

- How correlated are the learned functional embedding similarities with MNI-coordinate distances across subjects?

- What performance difference, if any, arises if the encoder is trained in a fully supervised manner (e.g., directly predicting region labels from the 32D embedding) instead of using a contrastive loss?

### Soundness
2

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
2

### Summary
The authors present a method to learn subject/region agnostic functional coordinates from low-field potential recordings in patients undergoing deep brain stimulation. The method computes functional embeddings using contrastive learning, with an objective function that keeps signals from the same brain region close in the latent space implemented in two different ways: pairwise siamese contrastive and modified SupCon. For region classification, a kNN classifier operates on the functional embeddings. 
To experiment with the model, the authors study the region classification performance on simulated data where functional properties mach regions annotations, showing a near perfect classification. The authors then move on to real data and present two cross-validation strategies for region classification on single subject: held out time-segments and held-out channels with ~75% and 44% accuracy respectively. Training on multiple subjects the prediction accuracy increases by 5%.

The learned embeddings are then used in tokenization for a transformer model solving a reconstruction task, where it is shown that tokenization using functional embeddings outperform MNI, and where the Modified SupCon contrastive approach is recommended over Pairwise Siamese method.

### Strengths
- Precise localisation of electrodes is a clear motivation.
- The relationship between anatomical ontologies and functional mapping is not a simple nor a solved problem, and the authors correctly state the potential gains from creating reliable functional embeddings.
- Code and datasets are available in a clear reproducibility statement

### Weaknesses
The gap in accuracy from held-out time segments 75% versus held-out channels 44% should raise some alarm. Here there is a large discrepancy with the synthetic dataset that could be explained either by the fact that the region labels are not reliable enough to train the model, or just by the fact that the anatomical ontology does not map to functional properties as neatly as in the simulation. In both cases the region labels may not be the best choice for contrastive objective. It thus seems the model learns something else that is not really related to position but related to a given subject/channel, which is passed on through the functional embeddings.



Minor Comments
- Figure 2, 3: confusion matrices really hard to read and poor resolution

### Questions
- In figure 6 A, on the right side, what explains the much higher frequency of reconstructed Func1 and Func2 signals versus the MNI reconstruction ?
- In section 4.4, authors state "in each evaluation window, all channels from a target region (here VO) were withheld" which refers to the downstream task of reconstruction. But where they also withheld from the functional encoding model ?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes FunctionalMap, a contrastive method for learning functional embeddings of sEEG channels over multiple patients (and hence multiple electrode configurations). The method consists of a Siamese convolutional encoder on neural recordings and a contrastive objective where embeddings of channels from the same brain region are pulled together, while those from different regions are pushed apart. The method is evaluated on synthetic data to uncover curated region-specific signatures of oscillatory activity from generated time series, and on a real dataset with brain region decoding and cross-region channel reconstruction as primary tasks. From the experiments, the authors suggest that the proposed method learns channel-level embeddings that can uncover continuous spectral features, spatial information, and results in improved reconstruction performance when compared to selected baselines. The paper argues that data-derived functional coordinates are a more reliable backbone for large-scale intracranial recordings than anatomical alignment.

### Strengths
(S1) Clear motivation and relevant work. The reliance on spatial coordinates in previous works modeling sEEG reflects a clear gap in the ability to generalize beyond spatial priors and to datasets where coordinates are unavailable or unreliable. Learning a data-driven, region-structured embedding first, then using it for tokenization downstream is clean, modular, and well-justified for heterogeneous sEEG setups.

(S2) By removing reliance on subject-specific heads or IDs, FunctionalMap enables zero-shot transfer to new subjects without having to lose channel identity. As suggested by the authors, this would be critical for large-scale pretraining on sEEG data, so subsequent works can focus on developing robust methods for representational learning on, or decoding from, neural signals without the constraint of having to learn new identities during transfer.

(S3) The paper presents interesting analyses that highlight properties of the method. In particular, I appreciate the study on synthetic data which demonstrates that the embeddings recover oscillatory features in a continuous space. Further, the comparison and analysis of contrastive objectives is useful for understanding how different formulations affect the geometry of the learned embedding space and downstream performance.

### Weaknesses
(W1) The real dataset on which the method was evaluated is limited, especially in terms of spatial coverage. While the isolated study of 5 brain regions is useful for analyzing the method in a controlled setting, the extent of the results is limited without extensions to other datasets. It would be ideal to see how the model fares on a larger dataset with more coverage of brain areas, such as Braintreebank \[1\] which is rich in terms of spatial coverage of sEEG recordings and would create a connection to previous works:

1. To build confidence that FunctionalMap can generalize, it would be nice to see how well it does on brain region decoding / reconstruction tasks on a new dataset, i.e. pretrain on one dataset and transfer to another.  
2. For the proposed application of the method on downstream large-scale pretraining / decoding to be viable, it should be demonstrated that FunctionalMap can itself scale up to pretraining on multiple datasets. I would suggest jointly pretraining on multiple datasets that span many brain areas.

(W2) The approach still depends on region labels for the contrastive pretraining, hence the motivation to move away from localization-based labels is not fully addressed. For instance, localization information can be imperfect due to topographic variability (e.g. due to resected areas \[2\]), electrode straddling over time, and since brain region labeling is a process done post-hoc it further invites the possibility of label noise.

(W3) Further, while the method does enable zero-shot transfer at inference-time, it assumes sufficient coverage during pretraining. While there are large datasets available with localization, there are also abundant corpora of data with no spatial coordinate labels. The method is fundamentally limited in its ability to scale up on these datasets which represent a large proportion of the publicly (and often privately) available sEEG data (e.g. \[3\]).

(W4) The paper lacks any sort of scaling analysis that could characterize how best to utilize the method. For instance, it would be useful to see how much data (along axes like \# of subjects, \# of sessions, number of recording hours, number of channels) is necessary to achieve the generalization capabilities highlighted in the results. Such an analysis would inform future studies and clarify the minimum data regime required for stable cross-subject alignment.

(W5) The paper could benefit from clearer framing and additional editing. For instance, sections 4.2 and 4.3 do not introduce the brain region decoding task at all. While it can be inferred from Figures 3 and 4 that the task in question is indeed brain region decoding, it was not set up or specified at all in the sections. There are also a number of minor grammatical and stylistic mistakes throughout the text.

### Questions
(Q1) How sensitive is the contrastive method to imperfect region labels, as detailed in (W2)? Could the authors perform a robustness analysis where label noise is artificially introduced in the brain region labels to approximate more realistic clinical localization?

(Q2) To clarify, as suggested at the beginning of section 3.2, were functional embeddings inferred on a new subject on only a single 10s segment, then the rest of the signals were used arbitrarily in the downstream decoding (reconstruction) task? If so, this reveals a very promising property of the zero-shot capability of the model. However, I’m interested in understanding the scope of this a bit more: (1) Does it matter which 10s segment is considered? (2) Does this property hold when generalizing to new brain regions? (3) Would the decoder model benefit from accepting a dynamic embedding, e.g. pass each context window being inputted into the decoder through FunctionalMap to zero-shot derive per-context-window, per-channel functional embeddings?

(Q3) The simulation study is limited only to oscillatory patterns. While these are very commonly strong signals for various stimuli / behaviors coded in the brain, there are other types of patterns that may be relevant. Of course one cannot exhaustively curate all possible patterns, and this isn’t necessarily required, it would be nice to see some diversity. Examples such as transient bursts \[4\] or cross-frequency coupling \[5\] could be considered. How would the model fare in being able to model these types of patterns, and would we observe the same sort of continuous mapping? Also, the study on synthetic data offers a nice opportunity to scale up the number of patients without worrying about data scarcity.

(Q4) Can confusion matrices on brain region decoding and PCA plots from the embedding space be provided for the other subjects, e.g. in the appendix? It’s not clear whether the observed spatial structure in S4’s learned embeddings generalizes to the others.

(Q5) To clarify, sections 4.2 and 4.3 do refer to brain region decoding as a task?

(Q6) What would the authors expect from including subject-specific components (e.g. subject ID/embeddings, decoder head, learnable channel embeddings), i.e. is the line of thinking that FunctionalMap would enable on–par or better downstream performance due to a better channel-level prior? Or do the authors expect performance to degrade, yet the ability to generalize in a zero-shot manner offsets the gap? It would be insightful to analyze this tradeoff.

(Q7) Can the authors comment on the choice to not include projection heads before the contrastive loss? Since this is standard in modern contrastive learning methods \[6\], it would be good to validate this design choice.

Lastly, a quick stylistic suggestion: on Figure 2C, it might be better to color the frequency on a log-scale.  

\[1\] Wang, C., Yaari, A., Singh, A., Subramaniam, V., Rosenfarb, D., DeWitt, J., ... & Barbu, A. (2024). Brain treebank: Large-scale intracranial recordings from naturalistic language stimuli. Advances in Neural Information Processing Systems, 37, 96505-96540.

\[2\] Roberts, D. W., Hartov, A., Kennedy, F. E., Miga, M. I., & Paulsen, K. D. (1998). Intraoperative brain shift and deformation: a quantitative analysis of cortical displacement in 28 cases. Neurosurgery, 43(4), 749-758.

\[3\] Carzaniga, F., Hersche, M., Sebastian, A., Schindler, K., & Rahimi, A. (2025). A foundation model with multi-variate parallel attention to generate neuronal activity. arXiv Preprint arXiv:2506. 20354\.

\[4\] van Ede, F., Quinn, A. J., Woolrich, M. W., & Nobre, A. C. (2018). Neural oscillations: sustained rhythms or transient burst-events?. Trends in neurosciences, 41(7), 415-417.

\[5\] Canolty, R. T., & Knight, R. T. (2010). The functional role of cross-frequency coupling. Trends in cognitive sciences, 14(11), 506-515.

\[6\] Gupta, K., Ajanthan, T., Hengel, A. V. D., & Gould, S. (2022). Understanding and improving the role of projection head in self-supervised learning. arXiv preprint arXiv:2212.11491.

### Soundness
2

### Presentation
2

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the challenge of aggregating heterogeneous, multi-subject intracranial (SEEG) recordings.  This task is notoriously complicated due to variable electrode placements.  The authors propose a framework that learns a subject-agnostic functional identity for each electrode, effectively creating a learned "functional coordinate system".   To my knowledge, this appears to be the first time such a learned functional coordinate system was attempted for Ephys data.  The training approach uses contrastive learning with annotated brain functions in the training data.  Afterwards, this learned functional embedding (called functional tokens) was fed into downstream "functional transformer".  The validation dataset is a 20-subject dataset, showing improved performance on models that rely on more rigid coordinate systems.

### Strengths
The paper's primary strength lies in its novel approach (functional tokens) for dealing with unreliable anatomical coordinates in Ephys data.  The empirical evidence supports the benefits of this approach, including both analyzing whether the functional coordinate system is correlated with ground truth functional annotations, as well as the benefits of it for downstream decoding.

### Weaknesses
I think the paper over-weights interpreting the functional coordinate system (e.g., all the various confusion/correlation analyses), and under-weights understanding how this embedding improves downstream decoding performance.

The main result for downstream decoding is Figure 6, which I found difficult to interpret.  Moreover, I would've liked to see more analysis teasing apart how and when the functional tokens improve decoding performance.

### Questions
No additional questions, beyond commenting on my weakness items.

### Soundness
3

### Presentation
2

### Contribution
3

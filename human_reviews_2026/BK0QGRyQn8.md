# Joint training does not transfer information between EEG and image classifiers

- Decision: Reject
- Scores: 0, 2, 4, 2

## Abstract
Caution is necessary with machine-learning methods, and especially
computer-vision methods, to support brain processing claims from
neuroimaging data.  Recent papers propose (i) a joint-training process
that does not use class information and (ii) a bidirectional transfer
of (a) image information to an EEG classifier and (b) brain-activity
information to an image classifier, such that the joint embedding
includes the shared image and brain-activity information.  These
claims cannot be maintained: the training process is initialized with
class information, and joint training with EEG degrades rather than
improves the performance of the image encoder.  Moreover, theoretical
solutions exist that entail no transfer beyond class information in
the joint embedding space.

## Human Reviews

## Human Reviewer 1

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
This paper is very worrisome. 

**Ulterior motive**. The introduction attacks Palazzo et al. from the very start. The authors use strong language such as "a large body of flawed work" and "This is further egregious". Many research groups are listed and explicitly attacked in page 3. 

**Presentation**. The paper uses way too many of checklist notations:  (i) and (a) in the short abstract, (II) and (B) in the introduction. The introduction starts with a long attack on specific research papers before transitioning with "Here we focus on a separate issue". Page 3 is almost exclusively citations of papers without actual sentences. Page 6 is almost exclusively bullet points.

### Strengths
None.

### Weaknesses
None.

### Questions
None.

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper focuses on an interesting question of whether jointly training the image and the EEG encoder enables the bi-directional information transfer between the two modalities. The author also performed some evaluations with an image-evoked EEG dataset. However, the writing and grammar are very hard to understand. Besides, the analysis is very limited to extensively support the hypothesis. I suggest full improvement after the submission.

### Strengths
The main critique in this paper is important in the field of brain decoding. It also raises attention to the fair and strict evaluation in the field.

### Weaknesses
N/A

### Questions
N/A

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper re-examines the validity of EEG–image joint training approaches originally proposed by Palazzo et al. (2018–2024).

Through PCA variance and classification analyses, they find that pretrained image encoders already contain class information,
and joint training neither enables cross-modal information transfer nor improves performance.
Instead, it degrades image encoder quality, thereby challenging prior claims.

### Strengths
- Provides a transparent replication of the EEG–image joint training framework with precise control over confounded and nonconfounded datasets (Li et al., 2021; Spampinato et al., 2017).
- Clearly isolates methodological factors — pretraining, triplet loss, and joint vs. separate training — showing that class information dominates the embedding space.
- The PCA-based decomposition offers an interpretable quantitative view of representational variance and supports the paper’s central logical claims (A–D).
- The writing is coherent and analytical, effectively communicating why observed classification gains stem from class information rather than cross-modal transfer.

### Weaknesses
- The PCA-based linear variance analysis may overlook nonlinear dependencies between modalities.
- Variance is used as a proxy for information content without validating its alignment with discriminative information.
- Claims of “class-only” encoding lack information-theoretic validation (e.g., mutual information, RSA).
- Empirical analyses rely on single-subject data (Li et al., 2021) and a pooled dataset (Spampinato et al., 2017), limiting cross-subject generalization.
- The criticism of saliency-map validity is conceptual rather than empirically demonstrated.

### Questions
- Q1. Were any analyses performed across multiple subjects or sessions to confirm generalizability?
- Q2. Could additional metrics (e.g., CCA, mutual information) further support the PCA-based conclusions?
- Q3. How did you empirically validate that the saliency/activation maps reflect brain-activity information rather than class priors or ImageNet features?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This paper reexamines a previously proposed method for jointly training EEG and image encoders and claims that it does not achieve meaningful information transfer between modalities. Using several architectures and datasets, the authors claim that the joint model’s performance is largely driven by preexisting class information rather than shared brain–image representations. Their analyses suggest that such training can even degrade image-model performance and fails to generalize on properly controlled data. The work provides a methodological critique of a widely cited line of research.

### Strengths
* the work performs a large number of analyses to investigate different claims made in prior work in a variety of settings
* it is important to critically evaluate and correct where necessary any mistaken approaches present in the literature

### Weaknesses
I found the writing hard to follow:
* it lacks a clear structure, unnecessarily repeating  the same content (e.g., Introduction: "Other work questioned these claims due to confounds...", same point again in 2 ("Several independent lines have refuted [...] suffer from temporal confound..."))
* using up almost a whole page for lists of citations
* mixes logical argumentation with general arguments on the importance of refutation or the importance of the release of raw data in distracting ways
* Long bulleted lists without clear internal structure with long texts inside each bullet point

Some of the space would be better used to more clearly convey the motivation behind some analysis choices or illustrative figures explaining the analysis pipelines. In the current form, I do not find it suitable for publication as I do not believe it efficiently communicates the claims it tries to make.

Overall, I have to admit I still don't really get the point. If one wants to check whether an EEG encoder and an image encoder have had some information transfer, I would expect it should be enough to
1) Learn some matching of EEG and image encodings
2) Evaluate whether unseen unconfounded EEG and corresponding image encodings are mapped closer together than non-corresponding ones. Images could also be of classes not present in the training set if one wants to remove more confounds.

Am I misundersanding that? If that had already been done in previous work, I do not see the point of deep analysis of in how far (only) class information is learned or not.

Especially as such analyses are hard:
* PCA analysis may miss nonlinear relationships obviously
* "Prior to joint training, the pretrained image encoders contain close to perfect class information, and likely very little information other than class information." -> The second part seems at odds with prior literature that was able to invert a lot of semantic details from images even when just using output logits of pretrained image encoders. 
* "have the representational capacity to memorize one-hot EEG and image encodings that minimize the loss function on the training set." -> We know that there are many settings in which deep networks have the capacity for memorization yet learn generalizable representations, so this does not seem such a strong argument to me

### Questions
I do not understand the top-40 components analysis and the motivation behind it? Because there are 40 classes?

### Soundness
2

### Presentation
1

### Contribution
2

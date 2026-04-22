# Beyond Conflict: Subspace-Alignment as the Missing Piece of Model Merging

- Avg Score: 3.33
- Decision: Reject
- Scores: 4, 4, 2

## Abstract
Model merging integrates task knowledge by combining task vectors (differences between pre-trained and fine-tuned weights) across models for both vision and large language models (LLMs).
Existing methods improve merging performance solely by mitigating conflicts between task vectors. However, we find that conflict alone is not the full story: 
when conflict-oriented remedies are applied, severe interference still persists. This observation motivates a novel perspective: analysing the interference from the standpoint of alignment.
Experiments reveal that task vectors exhibit high alignment in subspaces with large singular values. After merging, these aligned subspaces gather more singular values and produce activations with higher magnitude compared to others. The resulting spectral imbalance substantially degrades model performance.
Inspired by this, in contrast to previous methods that focus solely on conflicts, we propose the Subspace-Alignment Aware Merging (AlignMerge), which quantifies alignment by projecting task vectors onto the shared singular subspaces of the merged task vector and attenuates overly aligned components.
AlignMerge is training-free and requires no auxiliary data. Across vision and language benchmarks, it achieves state-of-the-art performance and narrows the 
gap to traditional multi-task learning to 3.6\%.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Paper proposes a new method for model merging. It argues that previous work has focused on preventing drift, and this paper instead analysis conflict from the view of alignment between task vectors. This can lead to spectral imbalance and the method proposes to address this problem. The paper follows recent works on SVD based mering (Gargiulo 2025, Marczak 2025).  Results show good results on vision and language model merging tasks.

### Strengths
- The authors might have come up with  a better explanation for the functioning of SVD based model merging methods (although they do not claim their explanation could be applied to previous methods, I think that might be true). The alignment story provides a good motivation why the SVD methods actually improve model merging. 
- ablation of the method is good showing the impact of the several factors on final performance.

### Weaknesses
- The paper is building on recent works of SVD based model merging (these methods saw a significant jump with respect to TA). The authors do a poor job in explaining the main difference of their method with respect to these methods (much of their alignment story could be hold for these methods ? ). It is the task of the authors to clearly distinguish from the most relevant related work, and the authors have not done that. 
- the results in Table one for TSV-M, Iso_C, Iso-CTS are significantly lower than in original papers (often outperforming the proposed method). The authors should be clear about the reason for this difference (beyond different checkpoints); any bugs in ISO-C ?
- Figure 1 too complex, not clear for me. Needs improvement.

### Questions
I would really want the authors to do a much more detailed discussion of the differences of their method with the previous SVD based methods. Alignment plays an important role in the motivation of Marczak as well. It is not the task of the reviewer to find the differences, but the authors should provide that.

Please address the drop in performance of ISO-C in tables in more detail.

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
4

### Summary
This paper proposes and analyzes subspace alignment as a key but overlooked factor causing interference in model merging. The authors observe that although task vectors are approximately orthogonal overall, they exhibit significant directional alignment in the top-level singular subspaces sorted by singular values. During merging, the singular values ​​of these aligned subspaces accumulate and amplify, leading to over-activation of these subspaces during inference and thus degrading overall performance. Based on this, the authors propose two training-free methods: AlignMerge-B (which calculates calibration coefficients per subspace after merging and scales/suppresses over-aligned components); and AlignMerge-O (which extracts and preserves the "aligned subspaces" before merging while orthogonalizing/whitening the white noise/remaining subspaces to avoid cumulative amplification). The paper shows that AlignMerge achieves or approaches state-of-the-art performance on multiple benchmarks and narrows the gap with traditional multi-task learning to approximately 3.6%.

### Strengths
1. The writing is quite easy to read and it was well-written
2. The gains from their proposed method of sampling are convincing and comprehensive
3. The article shifts the focus of model fusion from "conflict" to "subspace alignment," and points out that the alignment of top-level singular subspaces will produce a cumulative amplification of singular values/activations after merging, filling a major blind spot in the existing explanation.

### Weaknesses
1. The paper mentions that "the top subspace alignment is significant" and that the top-1 is treated specially, but it does not provide a more detailed explanation of how to select which subspaces in different layers or modules, how to select the number of singular values, and the application of this selection to each task.
2. The authors primarily treat "alignment" as a new factor besides conflict, but do not fully explore how to combine conflict-mitigation with AlignMerge to achieve complementary gains. I am curious whether merging the two can further narrow the gap with MTL.

### Questions
1.  Theories on orthogonalizing task vectors have been proposed in AWD[1]. Whether there is a fundamental difference between orthogonalizing in the singular value space after SVD and directly orthogonalizing task vectors is something that needs further explanation in this paper.
2. The methods compared in the article are limited. I would like to know whether the method proposed in the article has any further advantages compared to some recent SOTA studies, such as [2][3][4].
3. The article states that SVD only applies to some linear layers. Could you please specify which layers were selected and the selection criteria?

[1] Multi-task model merging via adaptive weight disentanglement

[2] EMR-Merging: Tuning-Free High-Performance Model Merging

[3] Whoever Started the Interference Should End It: Guiding Data-Free Model Merging via Task Vectors

[4] Task Singular Vectors: Reducing Task Interference in Model Merging

If the author can solve the question and the weakness well, i will raise my score.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper deals with the problem of Model Merging, which consists of merging multiple task vectors obtained from multiple fine-tuned models on each task to obtain a final multi-task task vector. The authors argue that Subspace Alignment is another factor, beyond conflict, contributing to the interference among task vectors — a factor overlooked by previous model merging methods. Hence, they analyze the alignment between the task vectors and notice that top singular subspaces are strongly aligned, and that this alignment causes performance degradation and should therefore be removed. The authors propose a metric called Subspace Cross-Influence ($\upxi^{r}$), measuring the alignment between two tasks, and define the Subspace Alignment Index (SAI), measuring the interference as the deviation of $\upxi^{r}$ from its optimal value of one. The authors propose two complementary strategies, called Subspace-Alignment Aware Merging Through Balancing (AlignMerge-B) and Subspace-Alignment Aware Merging Through Orthogonalization (AlignMerge-O), which remove alignment in the post-merging and pre-merging phases, respectively. The method is evaluated across vision and language tasks.

### Strengths
- By removing the subspace alignment between task vectors the approach reduces over-amplication improving the performance. 
- The approach provides  good performance with respect to the baselines.

### Weaknesses
1) The papers suffers from serious presentation issues that make it difficult to assess the soundness of the derivations in Section 3.2 and 3.3, which are necessary for understanding the methodologies:
- The notations used throughout the paper are heavy and significantly reduces  readability. The mathematical formulations lack intuitive explanations — especially geometric interpretations –  and many symbols are not clearly defined. For instance, the expression $$Proj_{P^r_{col}\,\Delta W_i}\left(P^r_{col}\,\Delta W_j\right) $$  is not properly introduced or explained in the text, and it represents an example of excessively heavy notation. Moreover, the use of  bold font  for all the  mathematical symbols  makes most of the formulas even harder to read.  

- The paper is not self-contained. Understanding key components of the paper requires consulting the appendix, which should not be necessary.
For example, at Line 210, the Subspace Cross-Influence index (one of the main contributions) involves a term $W_{\text{res}}$​, whose definition appears only in the appendix. Since this index is crucial to understanding the proposed method, it should be clearly defined in the main text. The appendix should serve to provide clarifications or supplementary material – not essential definitions.

- The figures are difficult to interpret. In particular, Figure 1 is extremely intricate, and it is unclear how the various components are related. A clearer, more structured visual explanation would greatly improve comprehensibility. 

2) The contribution seems largely incremental over [1], with several overlapping ideas and methodologies: 

- The  Subspace Alignment as an overlooked  source of interference (the first contribution emphasized in the title and mentioned at Line 98 in the introduction), has already been introduced in [1]. The authors are aware of  this work,  as they cite and compare it in the experimental section, however this reference should appear in several points of the paper. For instance,  the discussion of Subspace Alignment as a motivation of interference (Section 3.2) is not novel. Many of the arguments presented between lines 180 to 190 (e.g. orthogonal task vectors and alignment highlights) have already been discussed in [1]. 

- Moreover,  [1]  introduced the concept of *Subspace Alignment Ratio* (SAR) to quantify subspace alignment, while the present paper proposes Subspace Cross-Influence and  *Subspace Alignment Index* (SAI). The authors should directly compare their metrics to SAR and clarify the differences.

-  (Line 324) Selecting the top-$k$ singular vectors to identify the most aligned, shared subspace is similar to the approach used in Iso-CTS [1], although in Iso-CTS this is computed starting from Task Arithmetic rather than from each task-specific matrix

- (Line 329) Averaging the singular values (Eq. 11) is the same operation performed in Iso-C and Iso-CTS [1].

- (Line 337) Performing orthogonal projection to obtain the residual (Eq. 12) is also done in Iso-CTS [1].

- (Line 347) Using whitening to obtain orthogonal bases is employed as well in both TSV-M [2] and Iso-CTS [1]

The authors must clearly justify why their proposed metrics are superior to those in [1], and explain why their method achieves performance improvements over [1], as reported in the experimental section.

[1] Marczak, D., Magistri, S., Cygert, S., Twardowski, B., Bagdanov, A.D. &amp; Van De Weijer, J.. (2025). No Task Left Behind: Isotropic Model Merging with Common and Task-Specific Subspaces. Proceedings of the 42nd International Conference on Machine Learning.

[2] Gargiulo, A. A., Crisostomi, D., Bucarelli, M. S., Scardapane, S., Silvestri, F., & Rodolà, E. (2025). Task singular vectors: Reducing task interference in model merging. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

### Questions
The paper requires substantial reformatting to improve clarity in notation, mathematical presentation, and overall readability. It would also benefit from the inclusion of intuitive geometric figures to illustrate the geometrical operations the authors apply to the task matrices. In its current form, the manuscript is not ready for publication. Furthermore, the narrative should be significantly revised. Since Subspace Alignment is a central concept of the work—one that has already been introduced in [1]—the preliminary section should discuss [1] in detail. Throughout the manuscript, the authors should clearly articulate how their approach and findings extend or improve upon [1], and explain why the proposed method achieves better performance.

### Soundness
2

### Presentation
1

### Contribution
1

# A-TPT: Angular Diversity Calibration Properties for Test-Time Prompt Tuning of Vision-Language Models

- Decision: Accept (Poster)
- Scores: 4, 6, 4, 6

## Abstract
Test-time prompt tuning (TPT) has emerged as a promising technique for adapting large vision-language models (VLMs) to unseen tasks without relying on labeled data. However, the lack of dispersion between textual features can hurt calibration performance, which raises concerns about VLMs' reliability, trustworthiness, and safety. Current TPT approaches primarily focus on improving prompt calibration by either maximizing average textual feature dispersion or enforcing orthogonality constraints to encourage angular separation. However, these methods may not always have optimal angular separation between class-wise textual features, which implies overlooking the critical role of angular diversity. To address this, we propose A-TPT, a novel TPT framework that introduces angular diversity to encourage uniformity in the distribution of normalized textual features induced by corresponding learnable prompts. This uniformity is achieved by maximizing the minimum pairwise angular distance between features on the unit hypersphere. We show that our approach consistently surpasses state-of-the-art TPT methods in reducing the aggregate average calibration error while maintaining comparable accuracy through extensive experiments with various backbones on different datasets. Notably, our approach exhibits superior zero-shot calibration performance on natural distribution shifts and generalizes well to medical datasets. We provide extensive analyses, including theoretical aspects, to establish the grounding of A-TPT. These results highlight the potency of promoting angular diversity to achieve well-dispersed textual features, significantly improving VLM calibration during test-time adaptation. Our code is available at https://github.com/MB-Shihab-Aaqil-Ahamed/A-TPT/.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
normalized textual features on the unit hypersphere, ensuring uniform distribution an A-TPT enhances test-time prompt tuning (TPT) by introducing angular diversity calibration. Instead of maximizing dispersion or orthogonality alone, it maximizes the minimum pairwise angular distance between d better

### Strengths
1.	Elegant mathematical framing grounded in Tammes best-packing problem.
2.	Clearly articulated motivation: poor calibration under low angular diversity.
3.	Demonstrates lower Expected Calibration Error (ECE) with negligible accuracy loss.
4.	Theoretically principled and easy to integrate.

### Weaknesses
1.	Incremental relative to O-TPT and C-TPT; lacks major conceptual leap.
2.	No runtime or convergence analysis for numerical optimization.
3.	Experiments mostly on classification; unclear utility for generative tasks.

### Questions
1.	How sensitive is A-TPT to initialization of textual prompts?
2.	Can angular diversity loss degrade accuracy for semantically overlapping classes?
3.	Is there any theoretical bound on ECE improvement from uniform angular spacing?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper describes a new criterion for test-time prompt tuning in order to reduce calibration error.  It has previously been noted that calibration error can be reduced by designing a prompt template so that, when the different class labels are inserted into the template, the resulting prompts have maximally dispersed text vectors.  C-TPT measured dispersion using mean L2 distance, which does not guarantee pairwise separation; O-TPT guaranteed pairwise dispersion of 90 degrees if the number of classes is less than twice the number of embedding dimensions, but not otherwise.  The proposed A-TPT minimizes the maximum pairwise cosine similarity of classes, thus maximizing pairwise dispersion.

### Strengths
Derivations are interesting and clear. 

Equations and derivations seem correct. The point about \arccos normalizing gradient magnitudes is quite interesting.  I find multi-letter variable names aesthetically displeasing in general, but the use of "Cos" as a variable name does not impair legibility or correctness in this case.

Results show significant consistent reduction in calibration error, with small and inconsistent changes in accuracy, across 15 datasets, in comparison to TPT, C-TPT, and O-TPT.

### Weaknesses
Minor: Fig. 3 clearly shows that the prompts with the highest ECE ("the nearest shape in this image is" and TPT) are clustered in the center, while other prompts are distributed.  This does not show, however, that the prompts with high ECE have low angular diversity, because t-SNE does not show the angles of vectors: it only shows their cluster structure.

### Questions
On p. 4, what does it mean when the same prompt appears in both the list "Hard prompts" and the list "Tuned prompts," but with different Accuracy, ECE and AD?

Eq. (1) min_{j,j\ne i} should be min_{i,j\ne i}

### Soundness
4

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
4

### Summary
This paper targets the calibration problems that arise when doing test-time prompt tuning (TPT) on vision-language models and argues that existing fixes (like text feature dispersion (C-TPT) and orthogonality constraints (O-TPT)) don’t actually guarantee that class-wise text features are well separated, especially when the number of classes exceeds the embedding dimension. To address this, the authors propose A-TPT (Angular Test-time Prompt Tuning), which adds an angular diversity regularizer that, for each class embedding, maximizes its minimum angular distance to any other class, encouraging a more uniform packing of text features on the unit hypersphere. According to experiments on fine-grained, distribution-shifted the proposed method reduces expected calibration error (ECE) while largely preserving TPT’s accuracy gains over zero-shot CLIP and prior TPT variants.

### Strengths
* The paper clearly identifies the shortcomings of prior text-feature dispersion approaches and uses Figure 2 to illustrate them effectively.

* The reported ECE gains over baselines such as C-TPT and O-TPT are also encouraging.

* The authors show the method also works not only on standard benchmarks used to evaluate CLIP performance, but also on 'calibration critical applications' such as medical domain in Table 4.

### Weaknesses
* Although the paper proposes angular diversity regularization as a new metric, the method still operates within the existing C-TPT and O-TPT test-time adaptation paradigm, so the contribution feels more incremental than fundamentally novel in terms of theory or technique

* Could the proposed method be a complementary to previous methods (e.g., C-TPT or O-TPT). That is, could we for example enforce the proposed angular diversity on top of textual dispersion proposed by C-TPT or the orthogonality constraints of O-TPT. It would be interesting to see such an ablation.

* Since the ECE metric could suffer from bias, could the authors report calibration metrics other than ECE as well?

### Questions
See weaknesses above.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes Angular Diversity for Test-time Prompt Tuning (A-TPT), instead of pushing text prompts to be dispersed by l2 distance or cosine similarity, it maximizes the minimum pairwise angle between normalized class-wise prompt vectors (a maximin objective using θ=arccos of cosine similarity). This directly spreads prompts on the unit hypersphere to promote more uniform coverage and better calibration during inference. The authors also argue why this angle-based objective has stable gradients even when vectors are very close—unlike the orthogonality loss whose gradient vanishes as angles go to 0.

### Strengths
- Rather than optimizing L2 distance or cosine similarity, the paper optimizes the **angle itself,** which better captures geometric separation on the unit sphere and compensates for the shortcomings of previous work. This paper shows the limitation of previous work well.
- The paper includes extensive analyses that illuminate the method’s behavior from multiple perspectives, aiding interpretation and practical use.
- It explicitly examines the calibration differences between N > |D| and N ≤ |D|, a case prior work largely overlooks, and clarifies where the proposed method offers the biggest gains over O-TPT.

### Weaknesses
- When we increase λ, we understand this as trading some accuracy for improved ECE (better calibration). This trend aligns with Flowers102, but Food101 shows a contrasting pattern. Could you provide insight into why the two datasets behave differently? Also, are these curves averaged over multiple seeds, and how large is the variance across runs?
- How did you choose the λ term?
- In the main performance table, could you report results separately or make them explicitly distinguishabl for the N>|D| and N≤|D| regimes? This would help isolate where your method provides the most benefit over O-TPT.
- How do you ensure numerical stability when computing arccos?
- While increasing the minimun θ can encourage dispersion, it doesn’t seem sufficient to prevent localized density (clustering) in certain regions. Do you have any guarantees or empirical evidence that your method avoids such partial clustering?

### Questions
See weakness section

### Soundness
3

### Presentation
3

### Contribution
2

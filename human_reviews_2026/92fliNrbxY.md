# CheckMate! Watermarking Graph Diffusion Models in Polynomial Time

- Avg Score: 6.40
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 8, 6

## Abstract
Watermarking provides an effective means for data governance. However, conventional post-editing graph watermarking approaches degrade the graph quality and involve NP-hard subroutines. 
Alternatively, recent approaches advocate for embedding watermarking patterns in the noisy latent during data generation from diffusion models, but remain uncharted for graph models due to the hardness of inverting the graph diffusion process.
In this work, we propose CheckWate: the first watermarking framework for graph diffusion models embedding checkerboard watermark and providing polynomial time verification. 
To address NP-completeness due to graph isomorphism, CheckWate embeds the watermark into the latent eigenvalues, which are isomorphism-invariant. 
To detect the watermark through reversing the graph diffusion process, CheckWate leverages the graph eigenvectors to approximately dequantize the discrete graph back to the continuous latent, with theoretical guarantees on the detectability and dequantization error. 
We further introduce a latent sparsification mechanism to enhance the robustness of CheckWate against graph modifications. 
We evaluate CheckWate on four datasets and four graph modification attacks, against three generation time watermark schemes. CheckWate achieves remarkable generation quality while being detectable under strong attacks such as isomorphism, whereas the baselines are unable to detect the watermark. 
Code available at: https://github.com/r-gheda/checkwate.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes CheckWate, a novel sampling-time watermarking framework for graph diffusion models. By embedding checkerboard patterns into the eigenvalues of noisy latent representations, CheckWate achieves isomorphism-invariant, polynomial-time watermark verification. To enable detection from quantized (i.e., discrete) graphs, the authors develop an approximate dequantization method based on spectral properties, and introduce a latent sparsification mechanism to suppress false positives under adversarial graph perturbations. The method is thoroughly evaluated across multiple synthetic and real-world graph datasets under diverse attacks, showing strong generation quality and watermark detectability.

### Strengths
- This is the first paper to explore watermarking in graph diffusion models, extending prior image-based techniques to a much more complex domain.
- The use of graph eigenvalues as watermark carriers is well-motivated by their isomorphism invariance, and the checkerboard embedding leverages known spectral laws to ensure detectability.
- The approximate dequantization and sparsification modules are theoretically grounded and practically effective, enabling detection even under isomorphism and heavy graph edits.

### Weaknesses
- Z-score does not reflect full detection performance    
While Z-score is useful for summarizing distributional separation, it does not directly reflect practical detection metrics such as true positive rate (TPR), false positive rate (FPR), or ROC curves. As a reviewer, I would prefer to see:
    - Detection threshold selected to ensure low FPR (e.g., 1%), then report TPR.
    - Alternatively, a ROC/AUC analysis on the per-graph detection scores.
This is especially relevant for real-world deployment or fair method comparison.

- The use of “2” as the bulk-blip threshold lacks justification    
The watermark detection pipeline considers eigenvalues with magnitude >2 to be outside the bulk (Line 237). While this is motivated by the Wigner semicircle law, the choice of 2 as a hard threshold can be brittle since real data distributions are finite and noisy.
I recommend reporting sensitivity analysis of Z-scores under varying thresholds (e.g., 1.8/2.2) or using quantile-based definitions.

- Tradeoff between watermark strength and dequantization accuracy is underexplored     
According to Line 295 and Theorem 3.3, the dequantization error increases with eigenvalue multiplicity, which itself depends on watermark strength (larger W, smaller k). This raises a natural question: Does stronger watermarking hurt detectability due to worse latent reconstruction? The authors should analyze this tradeoff explicitly, possibly via a plot of detectability vs. reconstruction error for varying k and W.

- Line 360 has a type of "Appendix ??".

### Questions
Please refer to the weakness part. In short,
- Can you report ROC/AUC curves or TPR/FPR under fixed thresholds, in addition to Z-scores?
- Why was 2 chosen as the cutoff between bulk and blip eigenvalues? Have you tested threshold robustness?
- How does watermark strength (via k and W) affect the tradeoff between detectability and dequantization error?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces CheckWate, a watermarking framework for graph diffusion models that enables polynomial-time watermark verification. The method embeds a checkerboard pattern in the eigenvalues of the latent diffusion space.  The benefit of this approach is that eigenvalues are isomorphism-invariant. Prior techinques based on graph isomorphism (or graph edit distance) run into the issue that these problems are NP-hard. This approach avoids that issue.  

Experiments on four datasets show strong performance under various attacks (isomorphism, edge/node perturbations), outperforming prior diffusion watermarking methods such as Gaussian Shading and TreeRing.

### Strengths
This is a watermarking method specifically designed for graph diffusion models, a growing area in generative AI and data governance.

There are mathematical results with proofs.

There's a good set of experiments: multiple datasets, attacks, comparison approaches.

### Weaknesses
Assumptions (continuous latent spaces?).  (The paper states it could be applied to discrete latent spaces in Appendix D, but doesn't discuss fully.)

Watermark is non-blind.  

Polynomial time here is O(N^3), which is non-trivial;  accordingly, it seems, experimental tests are for graphs with less than 500 nodes. More information on running time would be useful.  

Some writing things to check (I see ?? for missing references, e.g., Appendix ??, please check).

### Questions
What is the empirical running time for watermark detection?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes CheckWate, the first watermarking framework for graph diffusion models that enables verification in polynomial time. Traditional methods are challenged by the computationally hard Graph Isomorphism (GI) problem. To bypass this, CheckWate embeds a "checkerboard" watermark into the latent eigenvalues, which are isomorphism-invariant. The framework uses an approximate dequantization mechanism to revert the discrete graph to its continuous latent for verification and a latent sparsification method to improve robustness against attacks. Experiments demonstrate that CheckWate maintains high generation quality and is robust against attacks like isomorphism, where baselines fail completely.

### Strengths
1.	It introduces CheckWate, the first watermarking framework specifically designed for graph diffusion models that provides exact and polynomial time verification. 
2.	The framework demonstrates strong robustness against various graph modification attacks, including isomorphism, edge deletion, edge addition, and node deletion.

### Weaknesses
1.	The paper's primary claim of efficiency is supported by theoretical complexity analysis but lacks empirical validation. A direct comparison of wall-clock runtime against baseline methods, particularly on large-scale graphs, would be necessary to fully substantiate the practical performance benefits of the proposed polynomial-time approach.
2.	The paper could benefit from a careful proofreading to correct minor typographical errors (e.g., around line 360 in the provided draft), which would improve the overall clarity and presentation quality.
3.	The non-blind nature of the watermark raises significant scalability concerns. The paper does not address the retrieval problem when the database of original watermarked graphs (c) is large. Verifying a suspect graph would seem to require a linear scan through all c candidates, leading to a total complexity of $c \cdot O(N^3)$, which may be impractical for large-scale auditing or copyright enforcement scenarios.
4.	The paper lacks a sensitivity analysis for the key hyperparameters (k, W). A thorough ablation study is needed to demonstrate how different choices of k and W affect the fundamental trade-off between watermark robustness and generative quality, which is crucial for understanding the method's practical applicability across different datasets and requirements.
5.	The reliance on Z-score as the primary metric for detectability is insufficient and potentially misleading. The paper's own results in Table 2 show the Bipartite baseline achieving Z-scores that are an order of magnitude higher than CheckWate, making the proposed method appear far weaker. To provide a true measure of effectiveness, the authors must report standard classification metrics. Reporting the watermark detection ACC and AUC is essential to clarify whether CheckWate's Z-score, while lower, is already sufficient for near-perfect detection, thus framing Bipartite's superior score as a result of severe, quality-damaging overkill.

### Questions
See the weakness.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors introduce CheckWate, a novel graph watermarking method for diffusion models. Instead of trying to watermark graph features directly, it is much more efficient to plant the watermark into graph properties that are invariant to isomorphism, i.e., into eigenvalues. This circumvents the watermark detector's need to handle different representations of the same graph, leading to compute performance gains. The authors also introduce dequantization and sparsification mechanisms, which they experimentally validate to be important. The CheckWate system appears to strike a balance between good detection (robustness) and graph quality in comparison to major watermarking systems.

### Strengths
The watermarking method is novel and its empirical performance is backed by theoretical underpinnings. Clarity on limitations is to be commended - potential future research paths are important and interesting problems. Paper is well-written and scientific with only a few points of concern (mentioned below).

### Weaknesses
- Polytime detector is presented as a major feature of CheckWate, but it is not emphasized in the body. The method appears clearly better than the prior work on graph watermarking that do not circumvent superpolynomial subroutines, but it is unclear how performance compares to the baseline (image) watermarking schemes evaluated.
- I believe graph isomorphsim is known to be in NP (e.g., Babai's quasipolynomial algorithm [1]), but not known to be NP-complete. There are statements in the paper that contradict this, e.g., row 50.
- The selected baselines are predominantly image domain watermarks. There is limited comparison to graph domain watermarks.
- "exact" detection mentioned in the abstract is perhaps misleading. Might be worth rephrasing.

[1] Babai, László. "Graph isomorphism in quasipolynomial time." Proceedings of the forty-eighth annual ACM symposium on Theory of Computing. 2016.

### Questions
- CheckMate (instead of CheckWate) in the title - is this intentional?
- How is it possible that Gaussian Shading is provably lossless, yet generation quality is at times worse than lossy watermarks (Table 1)? On row 396: "CheckWate achieves state-of-the-art generative quality" despite it being lossy.
- One of the major upsides of CheckWate seems to be efficient detection (i.e., with respect to "conventional" exponential time approaches). How does the scheme compare concretely to the other baselines in raw compute cost? Apologies if I missed this in the paper.
- Do you expect it to be possible to force keys (K) to be the same? This would lead to forgery attacks. It would be nice to have some kind of threat model to make it clear how "secure" the watermark needs to be. I suspect this is not as much of a concern as it would be for generic mode watermarks since the graph domain may be less adversarial, but I'd like clarity on this.

Presentation remarks
- Row 21/22: approximately dequantizes - > approximately dequantize
- Row 360: Appendix ??

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes a new graph watermarking approach CheckWate with three main contributions:
(1) a checkerboard watermark technique that embeds signals into noisy latent eigenvalues; since eigenvalues are isomorphism-invariant, detection can be done in polynomial time without solving NP-hard GI/GED problems;
(2) an approximated dequantization mechanism that projects discrete graphs to a continuous space for accurate latent reconstruction and watermark verification;
(3) a robust detection mechanism aimed at reducing false positives.

The experiments show that CheckWate can improve watermark detectability while maintain the graph quanlity.

### Strengths
* The paper is well written, logically structured, and easy to follow. The problem is important and well motivated for graph diffusion watermarking.
* Leveraging eigenvalues for watermark embedding/detection is a reasonable way to bypass GI/GED and keep verification polynomial.
* The method addresses graph watermark embedding, approximate dequantization, and robust detection in one framework, largely solving the targeted problem at a conceptual level.

### Weaknesses
* Sec. 3.1 states that for high enough $k$, checkerboard ensembles approximate regular Gaussian noise "while forcing limited modifications", but there is no quantitative analysis of how does the distribution change with $k$. Since regular Gaussian noise is a key assumption for diffusion latents, this gap weakens the claim.
* Sec. 3.4 argues that eigenvalues of perturbed GOE may leave the bulk and cause false positives and proposes sparsification to fix this, but experiments do not present the baseline FPR (without robust detection) versus post-fix FPR to show the reduction magnitude.
* An important baseline, Bipartite, is used but lacks a clear literature reference and detailed explanation in the main text; given its very high Z-scores yet relatively low quality metrics, it would be informative to see tuned variants that trade detectability for better quality, potentially making it a stronger baseline.
* The paper states that disabling the robust detection mechanism reduces Z-scores, but comparing Table 2 and Table 4 shows both increased and decreased Z-scores, which is inconsistent with the paper's statement.
* Line 360: “Appendix ??” appears in the draft and should be fixed.

### Questions
1. Please report quantitative divergence between checkerboard ensembles and Gaussian across $k$. Where is the operational $k$ range in which deviations remain “limited”?
2. What false positive rates arise from perturbed GOE without the robust step, and what are the FPRs with it? A per-dataset breakdown would substantiate the robustness claim.
3. Please add a citation and methodological details for the baseline Bipartite, and provide quality-controlled tuning where detectability is traded for higher quality to assess how strong Bipartite can be under matched quality.

### Soundness
3

### Presentation
3

### Contribution
3

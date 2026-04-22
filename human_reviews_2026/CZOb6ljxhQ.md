# UNSUPERVISED MEMBRANE SUBTRACTION IN CRYOGENIC ELECTRON MICROSCOPY IMAGES

- Avg Score: 3.20
- Decision: Reject
- Scores: 4, 4, 2, 2, 4

## Abstract
Cryogenic electron microscopy (cryo-EM) of membrane proteins often requires extracting them from their membrane to simplify downstream image processing. While this step reduces the influence of membranes on 3D reconstruction, it also prevents proteins from being observed in their natural state. To overcome this limitation, we propose a two-step machine learning framework that avoids protein extraction: (1) membrane detection, which identifies the bilayer membrane, and (2) membrane subtraction, which digitally removes the detected membrane from the cryo-EM micrograph. Recent work has introduced supervised algorithms for membrane detection, but membrane subtraction remains relatively underexplored. Here, we present a novel unsupervised approach to membrane subtraction that models membranes using a general representation and computes a smooth estimate, which can then be subtracted from the original cryo-EM micrograph. Experimental results show that our method outperforms existing membrane subtraction alternatives and enables reliable 3D reconstruction of membrane proteins using cryo-EM without protein extraction.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper titled “Unsupervised Membrane Subtraction in Cryogenic Electron Microscopy Images” introduces a new unsupervised framework for subtracting membranes from cryo-EM micrographs, given their segmentation is known. The authors introduce two main novelties. First a mathematical framework for representing biological membranes and second based on that representation an iterative unsupervised algorithm that estimates each unique membrane within micrographs. By estimating and  removing the membrane signal, the method produces micrographs denoised from membrane interference. The proposed framework is evaluated against existing methods, demonstrating better performance in terms of the membrane subtraction fraction. Finally, the authors validate their method’s effectiveness by reconstructing a membrane protein at high resolution.

### Strengths
The main strength of the paper lies in its solid mathematical foundation for defining and estimating membrane structures in cryo-EM micrographs. Building on this framework, the authors develop an iterative approach to estimate membrane signal contributions in experimental data and remove them in order to obtain higher-resolution membrane protein reconstructions. The theoretical rigor and clear formulation demonstrate strong technical expertise, along with improved evaluation and presentation. The method has the potential to become a valuable contribution to the field.

### Weaknesses
1. The claim of an “unsupervised membrane subtraction algorithm” is not entirely accurate, as the membrane detection step within the pipeline relies on a supervised U-Net model. This dependence weakens the claim of unsupervision.

2. The evaluation is based on limited data diversity, while similar micrographs are used both for training the U-Net and for comparison with other methods. This setup makes the evaluation potentially biased and less fair to the baselines.

3. The experimental analysis could be strengthened through ablation studies (e.g., iteration count, grid spacing, weighting function) and by including more 3D reconstruction results, especially from unseen experiments. Additionally, the validity of the proposed evaluation metrics (the membrane similarity index and subtraction fraction) remains somewhat uncertain, as their correlation with reconstruction quality is not demonstrated.

### Questions
1. Please clarify whether the U-Net segmentation output is applied uniformly across all compared methods (SA-algorithm, BM3D, SwinIR) or used only for the proposed approach. This point is essential to assess fairness in the comparison.
2. Could the authors provide evidence or discussion on how the membrane similarity index and subtraction fraction correlate with reconstruction quality? It is unclear whether these metrics meaningfully reflect improvements in 3D reconstruction outcomes.
3. Minor: In line 319 “ pixel sizes of 1.06, 1.09, and 0.825” please add the unit of length.

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
This paper presents an “unsupervised membrane subtraction” approach for cryo-electron microscopy (cryo-EM) images of membrane proteins, aiming to reduce interference from membrane signals during single-particle reconstruction. The method first detects membrane regions using a U-Net model, then performs local geometric modeling with iterative smoothing to estimate and subtract membrane intensity. Experiments on three representative protein systems (Kv1.2, KCNQ1, and Na,K-ATPase) demonstrate that the approach effectively suppresses membrane artifacts while preserving protein details. Overall, the paper proposes a mathematically grounded and reproducible framework that can improve the preprocessing quality of cryo-EM data.

### Strengths
The paper tackles a practical problem in cryo-EM image preprocessing with a clear and well-formulated approach.

The method combines geometric modeling and optimization in a coherent framework. 

Experimental results clearly show reduced membrane artifacts.

### Weaknesses
My background differs somewhat from the authors’, so the following comments are offered from a broader, cross-disciplinary perspective. 

Overall, the topic is meaningful and relevant, but from a machine learning and modeling standpoint, the paper’s originality appears moderate. The method mainly systematizes existing geometric and variational techniques rather than introducing new learning or representation mechanisms, and the term “unsupervised” is somewhat misleading given that membrane detection still relies on a supervised U-Net. The experimental dataset is relatively small (around 200 micrographs for training and 5000 for testing in one system) and not publicly available, limiting the assessment of generalization. The comparison includes BM3D, SwinIR, and SA, but omits recent self-supervised or EM-specific baselines. The theoretical derivations are thorough, but the link to implementation details is unclear, and there are minor inconsistencies between text and figures (e.g., missing subpanel in Figure 4). 

In summary, the paper demonstrates solid engineering work and numerical stability, yet from a broader ML audience’s viewpoint, it would benefit from stronger methodological originality, more comprehensive experiments, and clearer narrative integration.

### Questions
1) Please clarify the scope and assumptions of unsupervised. The paper states supervised membrane detection and “unsupervised” subtraction, but does not formally specify the learning or optimization assumptions for the unsupervised stage (priors, regularizers, observables).

2) Please map theory to implementation explicitly. While implementation details and iterative optimization in PyTorch are described, the correspondence from specific equations (core objectives and regularizers) to actual loss terms and update steps is not clear. An equation to code/loss term mapping would help.

3) Have you considered adding self-supervised or EM-specific baselines? The current comparisons (SA, BM3D, SwinIR) are reasonable but omit domain-relevant unsupervised approaches, which could contextualize the method’s advantages more fairly.

4) Any plans for data release and larger-scale validation? The paper describes two datasets but does not mention public availability or evaluation across more diverse imaging conditions. Releasing sample data or testing on larger datasets would strengthen reproducibility and generality.

BTW, as my expertise is outside the cryo-EM field, these questions are raised from a broader “non-specialist reviewer” perspective. I would appreciate clarifications from the authors on these points and would be interested to see how their responses and the ensuing discussion with other reviewers might address these concerns.

### Soundness
2

### Presentation
2

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
The paper addresses the problem of membrane subtraction in cryo-EM images of membrane proteins. To this end, it proposes a mathematical framework to represent membranes in an image and an iterative unsupervised algorithm to subtract them. The paper argues that the proposed method results in better subtraction of membranes than alternate ways.

### Strengths
+ A novel mathematical framework is presented for membranes, which to the best of my knowledge, is the first of its kind
+ Experiments were performed on real cryo-EM datasets instead of just simulated data

### Weaknesses
- I do not think the problem is significant enough for ICLR conference. Furthermore, the membrane subtraction did not result into significant discoveries, such as, increasing resolution of membrane proteins or discovery of novel membrane proteins. 

- The paper itself mentions that proposed mathematical framework for membranes can be used for other membrane-like image features which are not membranes. This is contradicting, in such a case, the definition is wrong. Without taking into account the scale of the image, I do not think it is possible to mathematically define membranes. Membrane-like structures inside the cell are not membranes. Just because the method treats them as so does not make them membranes.

- The reconstruction aspect of the evaluation is not clear. Figure 6 does not demonstrate the benefit of doing membrane subtraction using the proposed method

### Questions
Please explain the reconstruction aspect of evaluation. How Figure 6 is showing the need of your membrane subtraction method? Why the alternate methods can not be used?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
Authors propose an image artifact detection method that first estimates the artifact signal by iteratively fitting an artifact-specific mathematical model to the noisy input images. Authors then remove the artifacts by simple subtraction. The scope of the study is limited to membranes in CRYO-EM images.

### Strengths
Clarity: The paper is clearly written.
Results: Proposed method outperforms the reported baselines.

### Weaknesses
The main issue of the paper is its extremely limited scope. While the work is worthwhile, it is extremely niche to be in the main ICLR conference.


Originality: 

Domain specific modelling of objects is a valid approach but iterative smoothing and subtraction steps of the core algorithm is very similar to expectation maximization-based methods like Richardson-Lucy. As far as I can see, the difference here is that the authors' “prior” is their membrane model which is a Gaussian mixture. I do not see this as a significant novelty.


Significance: 

The scope of the paper is extremely limited. Authors only address membrane removal in cryo-em. While this is a valid problem to address, I doubt the method is applicable to other tasks (authors do not make such a claim either)  nor is it interesting to the ICLR community.


Experiments: 

This is tied to the scope of the paper but results in 3 cryo-em datasets does not tell the reader much.

### Questions
Can the proposed smoothing–subtraction procedure be formulated as an optimization problem or probabilistic model to justify the “unsupervised” claim?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes an unsupervised method to remove membranes from cryo-EM micrographs while keeping the embedded membrane proteins intact. The goal is to enable 3D reconstruction of membrane proteins without physically extracting them from the lipid bilayer, which often alters their native structure. The method has two main parts: membrane detection (done with a standard supervised U-Net) and membrane subtraction (the new part). The subtraction part models the membrane using local basis functions aligned with membrane curvature, and iteratively smooths and subtracts it. The authors test the method on several datasets of ion channel and ATPase proteins, comparing it to semi-automatic and denoising-based methods. Results show that their approach produces smoother, more realistic membrane estimates and improves 3D reconstructions.

### Strengths
* The problem is meaningful for cryo-EM and real biological research.
   * The approach is original and mathematically well grounded.
   * The iterative smoothing framework is intuitive and effective.
   * The results clearly beat older baselines like SA and BM3D.
   * The authors validate their method through both visual results and reconstruction quality.
   * The writing is careful and figures are clear.

### Weaknesses
* The unsupervised part still depends on supervised membrane detection.
This makes the claim of “fully unsupervised” less accurate. A discussion on how segmentation errors affect subtraction would help.
   * Besides, the evaluation is quite narrow. Only a few proteins are tested, and all data are from a small number of labs. There is no test on synthetic data or other public cryo-EM datasets. It would be nice to see quantitative results on a broader range of conditions.
   * The mathematical theory section is long and difficult to read. Many parts of Section 3 could be simplified. Readers might find it hard to connect the math to the algorithm.
   * The visual comparison with other methods is convincing but limited.
Adding a quantitative measure of protein preservation would make the results stronger.
   * Finally, computation time is not discussed. The iterative process may be slow on high-resolution images, but this is not reported.

### Questions
1. How sensitive is the subtraction to errors in membrane segmentation?
For example, what happens if part of the membrane is missed or mislabeled?
      2. How long does the full process take for a 5760×4092 micrograph? Can it scale to thousands of images efficiently?
      3. Could your algorithm handle multiple membranes or overlapping vesicles in the same image?
      4. Is it possible to use this approach directly on tomograms or 3D cryo-ET data?
      5. What would happen if the membrane is not bilayer-like, for example in irregular cell membranes?
      6. Do you plan to release the code and dataset so that others can reproduce your results?
      7. Could your iterative smoothing be replaced or combined with modern unsupervised denoising networks for speed improvement?

### Soundness
3

### Presentation
3

### Contribution
3

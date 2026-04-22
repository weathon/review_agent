# Certifying the Full YOLO Pipeline: A Probabilistic Verification Approach

- Avg Score: 5.20
- Decision: Accept (Poster)
- Scores: 6, 4, 4, 8, 4

## Abstract
Object detection systems are essential in safety-critical applications, but they are vulnerable to object disappearance (OD) threats, in which valid objects become undetected under small input perturbations, creating serious risks. This paper addresses the problem of verifying the robustness of YOLO (You Only Look Once) networks against OD by proposing a three-step probabilistic verification framework: (1) estimating output ranges under a distribution of input perturbations, (2) formally verifying the Non-Maximum Suppression (NMS) process within these ranges, and (3) iteratively refining the results to reduce over-approximation. The framework scales to practical YOLO models. Both theoretical analysis and experimental results demonstrate that our method achieves comparable probabilistic guarantees and provides tighter Intersection-over-Union (IoU) lower bounds while requiring significantly fewer samples than existing methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes a three-step probabilistic verification framework to assess the robustness of YOLO object detection systems against object disappearance attacks. By estimating output ranges, verifying NMS, and refining results, the method provides strong probabilistic guarantees and tighter IoU lower bounds with fewer samples, scaling effectively to practical models.

### Strengths
1. The authors propose the first scalable probabilistic verification framework for YOLO models, effectively addressing the object disappearance threat in safety-critical scenarios.
2. Combines solid theoretical analysis with comprehensive experiments, showing tighter IoU lower bounds and improved robustness.
3. Achieves strong probabilistic guarantees with significantly fewer samples compared to existing methods.

### Weaknesses
1. The method is very focused on object disappearance, which is an important threat, but also a pretty narrow one. It’s not clear how well the framework would generalize to other attack scenarios or tasks.

2. The experimental comparison feels a bit limited. Most of the analysis is against one baseline (RCPN), and a broader set of baselines would make the claims more convincing.

3. There’s little discussion on actual computational cost. While the paper shows faster verification times, it doesn’t really explain how the method would scale in a real deployment or resource-constrained setting.

### Questions
1. How does the method generalize under different perturbation distributions?
2. Could this verification framework be integrated with robustness training methods to further improve security?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces a PAC-based local verification procedure for YOLO networks---object detection networks that produce annotation boxes---specially for object disappearance (OD) threats (see, e.g., [Eykholt et al., 2018](https://www.usenix.org/conference/woot18/presentation/eykholt)).
These networks have very high-dimensional input and especially output spaces and use non-trivial post-processing, which makes verification with off-the-shelf methods difficult.

The approach combines multiple sampling-based probabilistic techniques to solve the task in multiple steps.
1. Approximate (from a sample) a hyper-rectangle as a high probability output "bounding box" for the relevant region in the input space.
2. (Repeatedly):
    - Detect unsafe output points $\mathbf y$ with Quadratic programming, if none found, return **safe**.
    - Attempt (with random sampling) to find inputs $\mathbf x$ such that $\lVert F(\mathbf x) - \mathbf y\rVert$ is minimal
    - If no close enough input found, refine output space, else return unsafe
The resulting guarantees are relaxed depending on multiple probabilistic hyperparameters, that dictate various sample sizes.

### Strengths
- **(S1) Significance and Motivation**
    - The problem is important for gauging the robustness of YOLO networks, and existing methods cannot be trivially adapted to this domain.
    - The Introduction and Related Work motivate the work well and give a nice overview of the neural net verification landscape, although some additional sources for more scalable PAC approaches can be discussed (e.g., [Baluta et.al., 2021](https://ieeexplore.ieee.org/abstract/document/9402111), [Blohm et al., 2025](https://openreview.net/forum?id=UKHlXpiFMy)).

- **(S2) Theoretical Contribution**
    - The obtained sample complexities are realistic, and proofs of the main results seem correct after surface-level checks.
    - Neural network verification with PAC methods has been widely explored, yet the unique output modality and post-processing seem to require more involved methods for verification.
    - The authors leverage multiple probabilistic techniques in tandem, leading to low sample complexity for the overall procedure; samples appear quasi-independent of the network dimensionality.
    - The general idea of integrating PAC methods with counterexample-guided refinement in verification is interesting and implemented in a novel way.

- **(S3) Empirical Results**
    - The procedure seems able to deal with model sizes common in related literature and issues robustness certificates that are close to the results of adversarial methods.

While I have several questions regarding the details of the procedure, I think they can be addressed, and their inclusion in the manuscript would strengthen the contribution.

 I would be happy to see a refined version of this submission accepted.

### Weaknesses
The main weakness of this manuscript is the complicated presentation of an already complex procedure.
Important details of the probabilistic procedure are difficult to gauge from the manuscript. 

Many theoretical issues may be easily clarified or adapted by the authors and do not greatly impact the theoretical contribution. 
However, a clear statement of the limitations and failure modes is necessary for a probabilistic procedure like this.

- **(W1) Readability and Notation**: The notation is very dense, which is partially difficult to avoid in this output modality. 
While not the main concern of the manuscript, the readability of the manuscript would improve a lot if:
  - Abbreviations were introduced more consistently (e.g., YOLO, RCP, PGD, ...).
  - Citations were wrapped in parentheses when they are not part of a sentence (i.e., use `\citep{}`-style formatting if you use LaTeX/natbib).
  - Tables avoided heavy rules and used a cleaner booktabs style, as is typical for ML conferences like ICLR.

- **(W2) Opaque Sample Complexity**: The procedure combines sampling-based approaches in a “nested” way, resulting in four hyperparameters $\alpha,\beta,\delta,\epsilon$, as well as empirically chosen sample sizes for parts of the procedure. 
This makes it difficult to gauge how precisely the sample complexity and runtime scale with each parameter, and what trade-offs different choices bring to the overall procedure. 

  The fact that the likelihoods of failure events depend on multiple parameters amplifies this, as multiple choices of different parameters can lead to the same sample sizes.

  **Actionable request:** Can you provide a mapping of each parameter to its failure event, sample complexity (big-O), or its impact on the overall runtime?
  This might not be necessary to include in the main text, but it potentially gives a nice overview of the procedure.

- **(W3) Partially Empirical Network Output Approximation**: The output domain of the network is estimated by scaling $\mathbf v_{\max}$, which, in my understanding, is essentially an empirically chosen proposal vector.
  The scaling coefficient is chosen with a probabilistic guarantee with $O(\frac{1}{\alpha} (\ln \frac{1}{\beta}+\ln\frac{1}{\alpha}))$ samples. 
  While this scaling coefficient is chosen optimally, the proposal vector seems to be chosen based on a heuristic, with the statement in Proposition 3 in Appendix D being opaque.

  There is a lot of established theory on obtaining bounding rectangles (see alternatives below). 
  These methods do not give the dimension-independent sample complexities that are presented here.
  However, the presented method consequently relies on a scaling trick and might overapproximate the tightest bounding rectangle if the proposal vector was not chosen well.

  The potential looseness of the initial output-space approximation is amplified by eliminating $L\_2$ balls from counterexamples, instead of using $L_\infty$.
   Over-approximations in a single dimension may require many refinement steps to eliminate, especially in very high-dimensional spaces.
  **Actionable request:** see **(Q1)**

 - **(W4) $L_2$ Counterexample Refinement and MIQP Runtime in Higher Dimensions**: Refinement steps of the output space are performed with a Mixed Quadratic Integer Program. 
  While this idea is novel and interesting, it is unclear how effective the refinement steps are, and whether the found counterexamples, in general, significantly shrink the output domain to be searched.
A discussion/investigation of this seems important, as each refinement step brings the cost of increased uncertainty.
- **(W5) Unclear Empirical Results in Table 2 and Figure 5**:

  - It is unclear what results precisely are presented in Table 2. The Baseline Selection and Appendix N state that 37,000 samples are used to certify the proposed method, with the baseline $\mathrm{RCP}_N$ on $10^6$ samples. However, the caption of Table 2, as well as *Safety Guarantee*, states $10^6$ uniform perturbations. In either scenario, the results are presented in a slightly confusing manner: either the runtime is compared to a method using significantly more samples (clarify in Table 1), or Table 2 shows results for significantly larger sample complexity (and thus tighter parameters).  
  - The caption states $\epsilon = 1/255$, yet $\epsilon$ is varied in the table.  
  - The false negative rate appears high (up to 15%). It is not stated on how many robust/non-robust boxes these results were computed; standard deviations are also missing.  
  - Figure 5’s x-axis is not labelled; it is not stated which model/experiment produced this data. Code does not appear to be available, so details cannot be checked.

### Questions
I would appreciate it if the authors briefly addressed my concerns in **$W2$** as well as answered my questions below to address the remaining weaknesses.

 - **(Q1) Initial Approximation of Output Domain**:
Many methods exist to give an \(\alpha,\beta\) guarantee on estimating a hyper-rectangle. How does your Part 1 method compare in terms of the tightness of the obtained rectangle? How much of an issue do over-approximations of the output domain present in practice?

   There is established theory for estimating non-parametric confidence regions from samples, e.g., the **DKW inequality** ([Dvoretzky–Kiefer–Wolfowitz, 1956](https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-27/issue-3/Asymptotic-Minimax-Character-of-the-Sample-Distribution-Function-and-of/10.1214/aoms/1177728174.full); [Massart, 1990](https://projecteuclid.org/journals/annals-of-probability/volume-18/issue-3/The-Tight-Constant-in-the-Dvoretzky-Kiefer-Wolfowitz-Inequality/10.1214/aop/1176990746.full); for multivariate exact constants, see [Naaman, 2021](https://www.sciencedirect.com/science/article/pii/S016771522100050X)). 
   Similarly, one can learn the smallest hyper-rectangle with empirical risk minimization (ERM).

   Alternatively, there exist dimension-dependent bounds with **$\epsilon$-nets**—see [Haussler & Welzl, 1987](https://link.springer.com/article/10.1007/BF02187876) and the standard reference [Mitzenmacher & Upfal, 2017](https://www.cambridge.org/core/books/probability-and-computing/3A5B47DB315FC64B9256C5C8131C5EFA). One could consider the class of *inverse* (half-)hyperrectangles with VC-dimension $d$. After a sample of size $\tilde{O}(\frac{d}{\epsilon})$, one could use $\mathbf v_{\max}$ directly as a bound, without the need to scale by a constant. 
   These alternatives trade dimension-independence for transparency; a short discussion comparing your scaling heuristic to empirical rectangles/$\epsilon$-nets would help position your choice.

- **(Q2) Counterexample Refinement with MIQP**: From reading the manuscript, it is not obvious that refinement steps significantly reduce the volume of the output domain (i.e., that the produced $\mathbf y$ will be far from the actual codomain of $F$). Consequently, refinement might weaken the probabilistic guarantees in Theorem 2 without real advantage. Why eliminate an $L_2$ neighborhood of each counterexample rather than, for example, an $L_\infty$ hypercube (or hyper-rectangles)? 
Wouldn’t that allow eliminating much larger volumes in high dimensions, especially for images?

   It would be very useful to investigate how much each successive refinement step not only increases the bounds but actually decreases the size of the output domain, as well as how much it relaxes the resulting bounds. A discussion of when further refinement is “not worth it” in terms of cost in confidence would be valuable.

- **(Q3) Counterexample Validation and Refinement**: The theoretical idea of Theorem 1 is opaque from the main text. A brief mention of the probabilistic idea behind the “more conservative estimates” in §5.3 (with a citation or name of the invoked bound/inequality) would help communicate the approach. 
  The proof of Theorem 1 in the appendix mentions Hoeffding’s inequality; explaining its role in one sentence in the main text would help.

- **(Q4) Motivation versus PGD**: If I understand correctly, Figure 5 shows that PGD often finds tighter lower bounds than probabilistic verification. 
   Is this an issue for the motivation of the procedure? 

   With an attack that is presumably cheaper than the proposed method, one can seemingly get tighter bounds. 
   What advantage does the proposed method offer over performing a PGD attack over an \(\tilde{O}(1/\epsilon)\) sample and reporting the tightest counterexample as a bound (cf. [Blohm et al., 2025](https://openreview.net/forum?id=UKHlXpiFMy))?

   In general, can adversarial methods be integrated into the approach instead of relying on uniform random samples? If such integration is out of scope, a motivation for using a probabilistic procedure when a cheap attack can immediately provide a counterexample would help.

- **(Q5)Clarification of Experimental Setup**:
  - What is the precise sample complexity of the results in Table 2---37,000 or $10^6$?
  - What is the precise certificate that the procedure issues for the instances, in terms of the probabilistic bounds and their interpretation?
  - Which column in the table reflects actual robustness behaviour?
  - Approximately, what is the certificate (confidence) provided by $\mathrm{RCP}_N$ at the used sample complexity?

---

## Minor Recommendations

**Self-Contained Theorems** In §5.3–5.4, including Theorems 1–2, some notation is not reintroduced (e.g., $A', B', C$). In the theorems, restate the meanings of $N, M, M_2$ for self-containment. In Theorem 2, “the algorithms defined above” should be referenced specifically (reffing Algorithm 1 might suffice if others are subroutines).

**Naming of Subroutines.** The manuscript would be easier to follow if “Part 1/2/3” were replaced by names (e.g., *Output-Box Estimation*, *Unsafe-$\mathbf y$ Search*, *Counterexample Validation/Refinement*). The algorithm captions would read more cleanly without the repeated “Part Y” phrasing.

**Typos.** One of the OpenReview keywords reads “guaranteen.” In Algorithm 4, use $\mathbb{Z}^+$ rather than $Z^+$.

---

## References

- Blohm, P.; Indri, P.; Gärtner, T.; Malhotra, S. (2025). *Probably Approximately Global Robustness Certification.* ICML 2025. OpenReview: <https://openreview.net/forum?id=UKHlXpiFMy>  
- Baluta, T.; Chua, Z. L.; Meel, K. S.; Saxena, P. (2021). “Scalable Quantitative Verification for Deep Neural Networks.” Proceedings of the 2021 IEEE/ACM 43rd International Conference on Software Engineering (ICSE), 312–323. <https://doi.org/10.1109/ICSE43902.2021.00039>  
- Dvoretzky, A.; Kiefer, J.; Wolfowitz, J. (1956). “Asymptotic Minimax Character of the Sample Distribution Function and of the Classical Multinomial Estimator.” *Annals of Mathematical Statistics*, 27(3), 642–669. <https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-27/issue-3/Asymptotic-Minimax-Character-of-the-Sample-Distribution-Function-and-of/10.1214/aoms/1177728174.full>  
- Massart, P. (1990). “The Tight Constant in the Dvoretzky–Kiefer–Wolfowitz Inequality.” *Annals of Probability*, 18(3), 1269–1283. <https://projecteuclid.org/journals/annals-of-probability/volume-18/issue-3/The-Tight-Constant-in-the-Dvoretzky-Kiefer-Wolfowitz-Inequality/10.1214/aop/1176990746.full>  
- Naaman, M. (2021). “On the Tight Constant in the Multivariate Dvoretzky–Kiefer–Wolfowitz Inequality.” *Statistics & Probability Letters*, 173, 109088. <https://www.sciencedirect.com/science/article/pii/S016771522100050X>  
- Haussler, D.; Welzl, E. (1987). “\(\varepsilon\)-nets and Simplex Range Queries.” *Discrete & Computational Geometry*, 2, 127–151. <https://link.springer.com/article/10.1007/BF02187876>  
- Mitzenmacher, M.; Upfal, E. (2017). *Probability and Computing: Randomization and Probabilistic Techniques in Algorithms and Data Analysis* (2nd ed.). Cambridge University Press. <https://www.cambridge.org/core/books/probability-and-computing/3A5B47DB315FC64B9256C5C8131C5EFA>

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
3

### Summary
This paper proposes a probabilistic verification framework (ODPV) to certify YOLO object detectors against the Object Disappearance (OD) problem. The framework consists of three modules: output range approximation, NMS verification, and probabilistic refinement, providing experiments on multiple YOLO variants showing strong robustness.

### Strengths
1.	The work presents a scalable verification framework for YOLO object detectors, incorporating the non-differentiable NMS post-processing into the certification process.
2.	The method requires far fewer samples to achieve strong probabilistic guarantees, enabling the verification of large-scale models.
3.	The experimental evaluation is comprehensive, covering diverse YOLO object detectors and testing robustness under various configurations. Results demonstrate the effectiveness and superiority of ODPV compared to existing probabilistic baselines, establishing a solid empirical foundation for detection verification research.

### Weaknesses
1.	The PAC guarantees rely on a uniform sampling distribution over the perturbation set. However, the paper does not explore how this assumption might affect the robustness of the guarantees in more realistic scenarios with non-uniform perturbations.
2.	This framework is developed and evaluated only on the YOLO family of detectors, and its robustness certification is limited to the OD threat. The generality of the approach to other detection architectures or to other robustness concerns remains unverified.

### Questions
NA

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper presents a probabilistic method to verify the full YOLO pipeline, from a specified input set C (a hypersphere of images around a nominal image) through non-maximum suppression (NMS). The goal is to check that, for all x in C, the post-NMS output y does not exhibit object disappearance: there is no perturbation such that,   after NMS on x, the best box with the correct class has the IoU below a fixed detection threshold.

The procedure has three parts. First, over-approximate the detector outputs F(C) by building Z. In practice, Z is an axis-aligned hyperrectangle estimated from samples x ~ C, with a PAC guarantee: with probability at least 1 - beta over the construction, a random x ~ C satisfies F(x) in Z with probability at least 1 - alpha. Second,  verify NMS over all y in Z. This is framed via a safe set Q that identifies boxes which, across Z, both meet the IoU and class requirements and cannot be suppressed into failure by NMS; if Q is nonempty, disappearance cannot occur. Third, if a candidate y in Z appears to violate safety, refine Z by trimming unreachable regions until either a real counterexample is confirmed or the candidate is shown unreachable. Then they provide the end-to-end probabilistic guarantee: for the chosen perturbation set C, object disappearance does not occur with the specified confidence and coverage parameters.

### Strengths
- The paper addresses the difficult and relevant problem of assuring end-to-end robustness of YOLO under perception noise, including the NMS stage. This is especially relevant to safety-critical systems that employ these black box detectors at runtime.

- The formalism encodes object disappearance but is general enough to express other anomaly types (e.g., misclassification, spurious appearances, duplicate suppression)

- Synthesizing Z only requires the ability to draw samples from C and does not assume a parametric form for the perturbations

- The PAC-style guarantees are nice because they provide calibrated confidence and coverage claims as opposed to simple binary claims, so the guarantees are generally more interpretable and can help inform upstream design decisions.

- The paper states definitions precisely, proves lemmas and propositions (including the soundness of the NMS safe set argument), and relates the algorithms to the formal guarantees they provide.

### Weaknesses
- The method certifies robustness only within a small epsilon-ball around a single image, and doesn't necessarily reflect practical YOLO deployments (e.g., traffic monitoring) where scenes change continuously and unpredictably across frames.

- If ground truth is already available for the target image, the value of verifying that the detector recovers it is questionable; this makes the result feel more like a labeled-scene sanity check than a deployment-relevant guarantee.

- The safety specification is narrow and ignores other important failure modes under perturbations, such as false appearances (spurious detections), class misidentification, or other anomalies.

### Questions
1. For a deployment like traffic monitoring, where the scene evolves continuously (new vehicles appear, others leave), how tractable is it to verify formal robustness guarantees over a short temporal horizon (multiple frames)?

2. What shape would C take on to make such guarantees meaningful? For instance, naively stretching the hypersphere C would begin to include semantically broken images that always invalidate safety.

3. Is it possible for verification to work with weaker supervision? E.g., rather than having complete ground truth bounding boxes, you have some sort of a priori map over time.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes a new method for PAC-based verification of the YOLO object detection network. Importantly, the method accounts for the Non-Maximum Suppression (NMS) post-processing step that is often used in practice. The first contribution is formalizing the verification problem. The second contribution is a certification pipeline based on the sample-based scenario approach [Campi, 2009]. Results show certification results that are faster than baselines and not too conservative.

### Strengths
- The paper tackles an ambitious problem, as YOLO is a large object detection model.
- The method accounts for the NMS post-processing stage, which appears to be novel and practically useful.
- By using a PAC-based sample-based analysis, the proposed verification method is less conservative and more practical than deterministic formal verification techniques (at the expense of weaker guarantees).
- The method is substantiated with a theoretical analysis.
- Results show that the error bounds are not too conservative.

### Weaknesses
- The sample complexity derived for the $RCP_N$ method (Appendix C.1) is incorrect: It should be computed with $d=1$ and not with $d_0=640 \times 640 \times 3$ (Appendix N), so $RCP_N$ likely requires fewer than 560'000'000 samples. The dimension $d$ corresponds to the optimization variable dimension, which is scalar for $RCP_N$, see (4). This error affects the sample efficiency and speedup claims.

- The appendix and proofs of the theoretical results are long, yet they are sometimes not polished, unclear, and have typos. Given the emphasis on theoretical results, this is a serious limitation. In particular:
1) Section D would greatly benefit from clearer exposition, e.g. "Then main result" (line 788). Also, "by algorithm should not far beyond" (line 789) does not specify what algorithm is considered and misses a verb.
2) Section G (proof of lemma 2) is unclear and not rigorous: The proof starts with "There are ...", but how it leads to the conclusion is unclear, the sentences on lines 903-905 and 909-910 are unclear and miss verbs and nouns.
3) In Section J.1., the proof relies on the sets $\mathcal{Q}_k$ and $\mathcal{T}$ whose definitions are unclear (the $\mathcal{Q}_k$ are only subsets of $2^{\mathcal{C}}$, and the definition of $\mathcal{T}$ is not rigorously written), and on an independence assumption of the events $\mathcal{T}\in\mathcal{Q}_k$ that is unclear.

- In Definition 1, $P_{x\sim\mathcal{C}}$ is unclear. The probability distribution $P$ is undefined. Also, $\mathcal{C}$ is a set, not a distribution, so $x\sim\mathcal{C}$ is unclear. This notation should be clarified before Section 5.

### Questions
- Please clarify the sample complexity of the $RCP_N$ method.

- Please revise the appendix and its proofs that are sometimes unclear or suffer from poor grammar.

### Soundness
2

### Presentation
2

### Contribution
3

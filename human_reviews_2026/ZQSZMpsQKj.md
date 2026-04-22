# SMOTE and Mirrors: Exposing Privacy Leakage from Synthetic Minority Oversampling

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 4, 4, 4, 6

## Abstract
The Synthetic Minority Over-sampling Technique (SMOTE) is one of the most widely used methods for addressing class imbalance and generating synthetic data.
Despite its popularity, little attention has been paid to its privacy implications; yet, it is used in the wild in many privacy-sensitive applications.
In this work, we conduct the first systematic study of privacy leakage in SMOTE:
We begin by showing that prevailing evaluation practices, i.e., naive distinguishing and distance-to-closest-record metrics, completely fail to detect any leakage and that membership inference attacks (MIAs) can be instantiated with high accuracy.
Then, by exploiting SMOTE's geometric properties, we build two novel attacks with very limited assumptions: DistinSMOTE, which perfectly distinguishes real from synthetic records in augmented datasets, and ReconSMOTE, which reconstructs real minority records from synthetic datasets with perfect precision and recall approaching one under realistic imbalance ratios.
We also provide theoretical guarantees for both attacks.
Experiments on eight standard imbalanced datasets confirm the practicality and effectiveness of these attacks.
Overall, our work reveals that SMOTE is inherently non-private and disproportionately exposes minority records, highlighting the need to reconsider its use in privacy-sensitive applications and as a baseline for assessing the privacy of modern generative models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors study the privacy leakage of SMOTE in the cases of data augmentation on imbalanced datasets with minority classes, and synthetic data generation. They propose two privacy attacks DistinSMOTE and ReconSMOTE, both achieving perfect precision while naive methods completely fail to detect any privacy leakage. The authors also claim to be the first to run a membership inference attack (MIA) against SMOTE. The state-of-the-art MIAs achieve high AUC, especially in the synthetic data generation case, which provides further evidence that SMOTE is fundamentally non-private.

### Strengths
* The paper studies an important problem, namely the privacy leakage of SMOTE.
* The paper proposes two privacy attacks DistinSMOTE (Algorithm 2) and ReconSMOTE (Algorithm 3) for the cases of data augmentation and synthetic data generation, respectively. 
* The strength of the proposed privacy attacks are supported by both theoretical results (Theorems 1 - 4) and empirical evidence (Section 4).

### Weaknesses
* The proofs can be written more clearly and carefully. There are some gaps in the proofs and they are somewhat informal. 
* There are no correctness proofs for algorithms (2 & 3), and the proofs of the theorems (1 - 4) are not based on the steps of algorithms (2 & 3).

### Questions
* Wouldn't the proof of Theorem 2 also require Assumption 4 (which is found in appendix A) to support the claim "the only point lying on three or more such lines is the shared real endpoint."?
* It can be stated more clearly in both the statement and proof of Theorem 3, that the segments/edges are directed, which is required for computing the probabilities.
* In Theorems 3 and 4, the proofs rely on the conclusion that each $C_{i j}$ is distributed according to the binomial distribution. This skips a step in the proof that the vector of all $C_{i j}$ has a multinomial distribution with $\sum_{i j} C_{i j} = n_0 - n_1$ which is exactly the number of synthesized points, and then we can conclude that $C_{i j}$ are marginally binomial.
* The proofs can be made stronger and more formal if they were based on the steps of the algorithms.

### Soundness
2

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
The paper investigates the empirical privacy of the widely used data augmentation/synthetic data generation method, SMOTE. They argue that common empirical privacy mettrics such as DCR severely underestimate privacy leakage of SMOTE and introduce two new attacks that exploit SMOTES geometric interpolation to perfectly distinguish real from synthetic samples and reconstruct minority records from synthetic data. These attacks have almost perfect recall on a number of benchmark datasets highlighting the privacy problems of relying on SMOTE in sensitive domains.

### Strengths
- The paper is clearly and concisely written, making it easy to follow.
- The proposed attacks have strong empirical results across a range of benchmark datasets and attack settings.
- The proposed attack, is as far as I am aware novel in the sense of exploiting SMOTEs interpolation algorithm with the added benefit that it is relatively “lightweight” as it avoids the use of shadow models.

### Weaknesses
- The main contribution -- that SMOTE leaks information and is non-private, is not conceptually new. Most practitioners already recognize that SMOTE, based on linear interpolation, offers no privacy guarantees and a long-line of work in tabular diffusion models (which is cited by this paper) even shows this empirically via DCR.
- The paper stops short of offering mitigation strategies of SMOTE, beyond briefly mentioning differential privacy approaches as future work.
- The framing (“SMOTE is inherently non-private”) overemphasizes a result that, while empirically validated, is expected and offers modest new understanding to the synthetic data community without suggesting defenses.

### Questions
1. I don’t believe the main finding, that SMOTE is non-private is particularly novel/insightful. It seems well known that SMOTE is not (empirically) private as shown by a number of DCR results in the line of literature on tabular diffusion models such as [2,3] which already note SMOTEs poor privacy characteristics. Therefore the central claim of the paper is that these DCR results "underestimate" the true privacy leakage which again is not a new insight i.e, it is shown in [4]. Can the authors expand on why their work is novel?
2. The claim that “SMOTE remains one of the most widely applied algorithms for medical synthetic data” seems overstated. Kaabachi et al. (2025) report SMOTE as only a small fraction of methods used, with GAN-based approaches dominating. I find it hard to believe the motivation for this paper i.e., 1) SMOTE is used a lot in practice for privacy-sensitive synthetic data and 2) is claimed as a ‘privacy preserving' method.
3. Wouldn’t adding a small amount of noise to the synthetic point generated from SMOTE hinder your proposed attack since the synthetic points will no longer be collinear? The paper should include more results using naive defenses such as this to show it is robust.
4. The datasets used are small-scale in both the number of rows and columns. Can you discuss the the computational scalability for larger, high-dimensional datasets?
5. How realistic is the assumption that k >= 3 for low-dimensional datasets like Abalone where d=6?

[1] Kaabachi, Bayrem, et al. "A scoping review of privacy and utility metrics in medical synthetic data." *NPJ digital medicine* 8.1 (2025): 60.

[2] Kotelnikov, Akim, et al. "Tabddpm: Modelling tabular data with diffusion models." *International conference on machine learning*. PMLR, 2023.

[3] Zhang, Hengrui, et al. "Mixed-type tabular data synthesis with score-based diffusion in latent space." arXiv preprint arXiv:2310.09656 (2023).

[4] Ganev, Georgi, and Emiliano De Cristofaro. "On the Inadequacy of Similarity-based Privacy Metrics: Reconstruction Attacks against``Truly Anonymous Synthetic Data''." (2023).

### Soundness
3

### Presentation
4

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
The paper studies the privacy vulnerability of SMOTE, a technique for augmenting imbalanced data or generating synthetic data. The paper presents two attacks against SMOTE that exploit the fact that SMOTE generates new points by interpolating real points. The first can perfectly distinguish SMOTE-generated datapoints from real datapoints in an augmented dataset under suitable assumptions. The second reconstructs real points from SMOTE-generated data. Both attacks are empirically evaluated and compared with existing MIAs and distance-based attacks. Both attacks have extremely high accuracy, while the baselines do not.

### Strengths
The paper clearly shows that SMOTE is not suitable to any privacy-sensitive application, whereas previous evaluations, especially distance-based ones, are less convincing. The writing of the paper is good, and the principles of the attacks are explained clearly.

While SMOTE is not considered a serious synthetic data generation method in the machine learning community, the citations in the paper show that SMOTE is seriously considered in other important research communities. The results of the paper show that it should not be. The new attacks are so accurate that releasing SMOTE-generated synthetic data is not far from releasing real data, and if releasing real data is acceptable, generating synthetic data does not make sense.

The results also demonstrate a flaw in most empirical privacy evaluations. Typical evaluations only use generic metrics like DCR or attacks like MIAs that can be applied to many methods without modifications. However, these do not rule out the possibility that an attack targeted at a specific method could lead to a much higher privacy loss than suggested by generic metrics, which is what this paper shows for SMOTE.

In summary, though the scope of the paper is limited to SMOTE, there are enough wider implications in the work to warrant publication in ICLR once some technical issues with the theory (described below) are resolved.

### Weaknesses
There are some incorrect technicalities in Theorems 1 and 2. For Theorem 1, it is possible that a SMOTE-generated point happens to be collinear with two real points other than the points it was interpolated from. In this case, one of the unrelated real points could be the midpoint of the line between the three, and it would be classified as a synthetic point, while the generated points would be classified as real. 

Theorem 2 has a similar issue: it is possible that three generated points from different real points just happen to be collinear. If that happens multiple times, ReconSMOTE could reconstruct points that are not real. Also in Theorem 2, global collinearity does not guarantee that any intersection of more than three lines is a real point. It is possible to draw three lines intersecting at one point, and pick 2 points from each line to be real points, such that no three of the 6 real points are collinear. 

However, all of these issues are very unlikely to happen in practice, so they can likely be fixed by adding some reasonable assumption. For example, I would not be surprised if the probability of these happening is 0 when the real datapoints are sampled from a continuous distribution in $d$-dimensions.

In addition, the proofs of Theorems 1 and 2 should show that looking at a subset of nearest neighbours of a point is sufficient instead of looking at all points (line 9 in Algorithm 1 and line 8 in Algorithm 2).

### Questions
- Lines 181-182: how can the number of neighbours and imbalance ratio be approximated?
- Would setting $k = 2$ be an effective defence against the attacks?
- Do the attacks work if applied against one-hot encoded categorical data? Does SMOTE even make sense in this case?

### Soundness
2

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents a systematic privacy analysis of SMOTE (Synthetic Minority Oversampling Technique), one of the most widely used methods for addressing class imbalance in machine learning. The authors show that existing privacy evaluation metrics, including naive classifiers and DCR, completely fail to detect leakage. While standard membership inference attacks achieve a high success rate against SMOTE-generated data, they still substantially underestimate the true extent of privacy leakage. They further introduce two novel geometry-based attacks: DistinSMOTE, which perfectly distinguishes real minority records from synthetic ones in augmented datasets, and ReconSMOTE, which reconstructs the original minority records from only the synthetic data with theoretically perfect precision and high recall. Supported by formal proofs, theoretical bounds, and experiments on eight benchmark datasets, the paper concludes that SMOTE is inherently non-private, exposing the very minority individuals it aims to protect, and calls for a reevaluation of its use in privacy-sensitive applications.

### Strengths
- The paper tackles an important and long-overlooked issue in synthetic data privacy by revealing that SMOTE’s interpolation mechanism can inherently leak information about real minority records. This problem is well-motivated, timely, and relevant to both theoretical and applied ML communities.
- The paper is clearly written and visually well-presented. The pseudocode, figures, and tables effectively communicate both the attack logic and empirical findings, making the complex geometric reasoning easy to follow.
- The proposed attacks are original and analytically grounded rather than heuristic. DistinSMOTE and ReconSMOTE exploit the geometric structure of SMOTE in a principled way, offering both theoretical depth and conceptual elegance.
- The experimental results, though limited in data complexity, convincingly demonstrate that existing privacy metrics underestimate leakage while the proposed methods reveal substantial vulnerabilities. This provides strong empirical support for the paper’s central claim.

### Weaknesses
- The three core assumptions of this work are too idealized to hold in most real-world settings. Most real datasets (especially in medicine or finance) contain categorical or mixed-type features, violating Assumption 1. Even in continuous data, collinearity or near-collinearity often occurs due to correlated features or duplicated records, which could break Assumption 2.

- Theorem 3’s recall bound relies on overly idealized assumptions of independence and uniform sampling that do not hold in real SMOTE graphs, where neighbor relationships overlap and edges share endpoints. The uniform segment probability ignores heterogeneous data densities, and in high-dimensional spaces, line intersections are rarely exact and must rely on numerical thresholds that introduce errors. As a result, the bound offers useful intuition but overestimates recall under realistic, dependent, and noisy conditions.

- No experiments show how the attacks behave when collinearity, floating-point noise, or categorical features are introduced.

- The datasets used in the experiments are relatively simple and do not fully reflect the complexity or real-world settings of the SMOTE applications discussed in the introduction.

### Questions
- Given that many real-world datasets are mixed-type, do you believe these attacks constitute a realistic privacy threat, or mainly a theoretical warning?
- The paragraph describing the assumptions of adversary model is inconsistent (Section 4.1). It first states that the adversary can approximate (k) and (r) from the released dataset, but then claims that no access to public data or published aggregate statistics is required.
- The table reference in line 97 is incorrect; it should refer to Table 2, not Table 1.
- Please also address my concerns raised in the weaknesses section, as resolving them could lead me to raise my score.

### Soundness
3

### Presentation
3

### Contribution
3

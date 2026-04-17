# EnsembleSHAP: Faithful and Certifiably Robust Attribution for Random Subspace Method

- Decision: Accept (Poster)
- Scores: 4, 8, 2

## Abstract
Random subspace method has wide security applications such as providing certified defenses against adversarial and backdoor attacks, and building robustly aligned LLM against jailbreaking attacks. However, the explanation of random subspace method lacks sufficient exploration. Existing state-of-the-art feature attribution methods, such as Shapley value and LIME, are computationally impractical and lacks security guarantee when applied to random subspace method. In this work, we propose EnsembleSHAP, an intrinsically faithful and secure feature attribution for random subspace method that reuses its computational byproducts. Specifically, our feature attribution method is 1) computationally efficient, 2) maintains essential properties of effective feature attribution (such as local accuracy), and 3) offers guaranteed protection against privacy-preserving attacks on feature attribution methods. To the best of our knowledge, this is the first work to establish provable robustness against explanation-preserving attacks. We also perform comprehensive evaluations for our explanation's effectiveness when faced with different empirical attacks, including backdoor attacks, adversarial attacks, and jailbreak attacks. The code is at https://github.com/Wang-Yanting/EnsembleSHAP. WARNING: This document may include content that could be considered harmful.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper studies explanations for models built with the random subspace method. It proposes an attribution method that reuses random subspace method inference byproducts to produce importance and develops a certified lower bound for detecting explanation-preserving attacks. Experiments focus mainly on text classification and LLM jailbreak detection across different metrics.

### Strengths
1. The paper is well organized and easy to follow.
2. The certified lower bound for detecting explanation-preserving attacks provides a certified guarantee that links attribution to robustness.
3. It studies attribution under adversarial, backdoor, and jailbreak settings, which is a novel perspective.

### Weaknesses
1. The proposed method is proposed for only random subspace methods, which limits its generalizability to other ensemble methods or broader applications.

2. The experiments emphasize a small set of attribution methods (e.g., Shapley/LIME/ICL). It is suggested to include more diverse families, e.g., propagation-based methods (LRP/DeepLIFT) or gradient-based variants (IG + smoothing), to strengthen the empirical case and reduce experiment bias.

3. Section 5.2.3 shows hyperparameter sensitivity. The analysis mainly compares with Shapley. More attribution baselines and detailed analysis would strengthen this study.

### Questions
1. Can the method work beyond RSM?
2. How does your method compare to other attribution families?
3. What causes the hyperparameter sensitivity, and how can it be mitigated?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This work proposes an enhancement for arbitrary partition-based certified defenses that (1) provides explanability and (2) is in itself certifiably robust.

For context, partition-based defenses, here referred to as "random subspace methods", are a commonly used flavor of randomized smoothing that performs self-ensembling over (random) subsets of features or training samples. If the number of subsets that a single element can appear in is bounded, then it can only have limited influence on the ensemble prediction. This allows for the derivation of robustness certificates.

The authors propose a method for computing per-feature importance scores given the ensemble model's prediction.
They then show that this method has three desirable properties:
* The importance scores can be computed from the same feature subsets that are already used for making smoothed predictions. Thus, the computational overhead is small.
* The importance scores inherit desirable properties of Shapley values, which are a powerful (but in this setting intractable) feature attribution method
* Using the standard argument for partition-based defenses, the method is in itself certifiably robust. Specifically, if an adversarial attack on the ensemble model is successful, then many of the modified features will be assigned a high importance score.

In the experiments, the method is first empirically evaluated based on its ability to provide useful (as measured by "faithfulness") explanations under adversarial attacks on standard classification datasets, as well as an LLM jailbreaking benchmark. Afterwards, the certificates are evaluated in terms of certified detection rate (essentially certified ratio for explanations).

### Strengths
* The proposed method is very broadly applicable, with partition-based defenses being a standard approach to certified robustness at both training time (poisoning attacks) and inference time (sparse attacks on images, graphs, point clouds etc.)
* The approach is incredibly elegant, achieving multiple goals via relatively straight-forward procedure (see desirable properties above)
* Section 4.3 provides provable utility guarantees in addition to provable robustness guarantees. This is somewhat unusual for randomized smoothing papers and definitely a positive
* The chosen range of datasets and models appears adequate for a paper that is primarily focused on provable robustness
* Provably robust explanation is mostly underexplored, i.e., novelty appears high

### Weaknesses
The work is primarily held back by its presentation. In particular:
* There are various typos and grammatical errors (tense, conjugation, duplicate words like "is a is a" in l.075 etc.). The manuscript could be significantly improved by running it through grammarly, the Copilot grammar checker, or a similar tool
* The main theorem is somewhat awkwardly forwarded (see Eq. 12-16) and incredibly dense. If the theorem itself cannot be further simpliified, I would encourage the authors to at least expand its intuitive explanation in l. 332ff.
* There are no explanatory figures. Adding one or two could help readers not familiar with randomized smoothing to more easily follow the paper.

Other than that, I only see two issues with the experimental evaluation:
* When varying parameters of the explanation method (see, e.g., Fig. 16), only the effect on certified detection rate is shown. However, it is not clear how varying these parameters impacts model utility. It would be better to additionally visualize the trade-off between utility and provable robustness (similar to certified accuracy in classification).
* Assuming the authors do, in fact, only want to show certificate strength: Constraining the experiments to a specific dataset and model as in Fig. 1 leads to a reductive view on the certification procedure (it fixes the dataset size and the model's confidence etc. to a particular value). It would be more informative to just treat these as additional parameters to vary (as is already done with $\beta$, $\rho$, $N$, etc.). Varying these additional parameters could be a nice experiment for the camera-ready version.

### Summary
Overall, this work makes a novel, elegant, and broadly applicable contribution to the field of provably robust machine learning There is 
 minor room for improvement in the experimental evaluation.
Assuming that the authors will improve the presentation for the camera-ready version (at least fixing most of the grammatical errors and typos), I recommend acceptance.

### Questions
### Other suggestions:

The following paper appeared ca. three months before the submission deadline. I would encourage the authors to discuss it as concurrent work:

Anani et al.. Pixel-level Certified Explanations via Randomized Smoothing. ICML 2025

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes EnsembleSHAP, an explanation method for Random Subspace Method ensembles that is efficient and certifiable. During the inference of an ensemble method, EnsembleSHAP uses the model’s prediction to assign credit to features, then derives a certified detection lower bound of explanation-preserving attacks that modify at most T features. Experiments show that EnsembleSHAP achieves higher faithfulness with small overhead.

### Strengths
1.	The method suits well with RSM since it explains the ensemble by reusing its votes, so it is efficient and well-motivated.
2.	The method is model agnostic since it only requires votes and some ablations, making it easy to adapt.
3.	Various types of attacks are evaluated such as jailbreaking, token edits, trigger based.

### Weaknesses
1.	The certification is limited to at most T feature perturbations, which may be limited since many real-life attacks can modify an arbitrary number of features (e.g., sinusoidal signal poisoning).
2.	The tools used in theoretical analysis are fairly elementary, such as frequency counting, confidence intervals, and binary search, so the novelty of the theorems and proofs appears limited.
3.	The threat model is limited for explanation-preserving attacks only, which is overly restrictive: in practice, many attackers will at least partially disrupt the explanation (e.g., by shifting saliency), so this limited certification guarantee seems limited to real-world scenarios.

### Questions
1.	This method assumes equal contribution among a subset of features. What is the rationale behind this choice?

### Soundness
2

### Presentation
3

### Contribution
2

# A Fair Bayesian Inference through Matched Gibbs Posterior

- Decision: Accept (Poster)
- Scores: 6, 4, 4

## Abstract
With the growing importance of trustworthy AI, algorithmic fairness has emerged as a critical concern. 
Among various fairness notions, group fairness - which measures the model bias between sensitive groups - has received significant attention. 
While many group-fair models have focused on satisfying group fairness constraints, model uncertainty has received relatively little attention, despite its importance for robust and trustworthy decision-making. 
To address this, we adopt a Bayesian framework to capture model uncertainty in fair model training. 
We first define group-fair posterior distributions and then introduce a fair variational Bayesian inference. 
Then we propose a novel distribution termed matched Gibbs posterior, as a proxy distribution for the fair variational Bayesian inference by employing a new group fairness measure, the matched deviation. 
A notable feature of matched Gibbs posterior is that it approximates the posterior distribution well under the fairness constraint without requiring heavy computation. 
Theoretically, we show that the matched deviation has a strong relation to existing group fairness measures, highlighting desirable fairness guarantees. 
Computationally, by treating the matching function in the matched deviation as a learnable parameter, we develop an efficient MCMC algorithm.
Experiments on real-world datasets demonstrates that matched Gibbs posterior outperforms other methods in balancing uncertainty–fairness and utility–fairness trade-offs, while also offering additional desirable properties.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces a Bayesian framework for fair machine learning, addressing group fairness and uncertainty quantification  simultaneously. Traditional group-fair models ensure fairness by constraining optimization objectives but rarely account for predictive uncertainty, which is crucial for trustworthy AI. The authors propose a matched Gibbs posterior, derived from a novel fairness-aware penalty called the matched deviation, which upper-bounds the Wasserstein and total-variation fairness measures. This formulation avoids adversarial training and provides a computationally efficient approximation to the fair Bayesian posterior. They design an MCMC algorithm where both model parameters and the matching function 𝑇 are inferred jointly, ensuring fairness without heavy computation. Experiments on tabular, image, and text datasets show that the matched Gibbs posterior achieves better trade-offs between utility vs fairness and uncertainty vs fairness than baselines such as Reduction, GapReg, and Adversarial fairness methods, while also improving individual fairness.

### Strengths
1. Well written and clearly structured: Despite heavy mathematics, the paper is well organized; intuitions precede theorems, notation is consistent, and experiments visually support claims.

2. Novel combination of fairness and Bayesian inference: The paper is among the first to integrate group fairness constraints directly into Bayesian inference, explicitly addressing both uncertainty quantification and fairness.

3. Avoids adversarial optimization: By replacing adversarial discriminators with a learnable matching function 𝑇, the approach sidesteps instability and 𝑂(𝑛^2) cost typical in IPM-based fairness.

4. Improved fairness–utility trade-off: Across datasets, matched Gibbs posterior consistently outperforms prior baselines on accuracy, NLL, Brier, and calibration error (ECE).

### Weaknesses
1. Scalability concerns: Joint inference of 𝑓 and 𝑇 may become costly for high-dimensional or non-metric input spaces (e.g., text embeddings). No discussion on large-scale efficiency.

2. Restricted to binary sensitive attributes: The method currently handles only 𝑆∈{0,1}; multi-group or intersectional fairness remains unexplored.

3. Empirical scope and baselines: Experiments are thorough but limited to medium-sized datasets; modern large-scale deep architectures (e.g., BERT, ResNet-50) are absent.

4. Unclear robustness under complex priors: The framework assumes tractable Gaussian priors; how matched Gibbs behaves with non-Gaussian or hierarchical priors is not tested.

My main concerns focus on scalability and applicability to modern large-scale architectures. Specifically, the paper lacks validation on high-capacity models such as BERT or ResNet-50, and it remains unclear whether the proposed matched Gibbs posterior remains computationally feasible in high-dimensional or non-metric spaces (e.g., text embeddings). In such settings, stability and sensitivity to noise could become significant issues.

### Questions
Impact of imperfect matching: How sensitive is fairness performance to suboptimal or noisy matching functions 𝑇? Can the authors quantify how deviation from optimal 𝑇 affects fairness bounds?

Scalability: How would the proposed MCMC perform on large neural models (e.g., transformers) or high-dimensional text embeddings?

Connection to uncertainty geometry: Can the authors relate matched Gibbs fairness constraints to curvature of the posterior (e.g., Hessian eigen structure) or persistent-homology-based fairness landscapes? (This is not necessary for the paper, just something that came to mind during the review.)

### Soundness
3

### Presentation
4

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
The paper develops a Bayesian route to group fairness by defining fairness for posteriors and proposing a matched Gibbs posterior: a Gibbs posterior with a new matched deviation penalty that avoids adversarial IPM/MMD inner loops, comes with bounds relating it to Wasserstein/TV, and is trained via an HMC+MH sampler that jointly infers the predictor and a matching function $T$. Experiments on ADULT, DUTCH, CRIME, CELEBA, and CIVIL suggest improved utility–fairness and uncertainty–fairness trade-offs.

### Strengths
1. Clear formulation of fairness for posteriors (average DP to strong DP via rejection) and a practical proxy—the matched Gibbs posterior—that circumvents adversarial critics and offers $O(n)$ updates with an explicit sampler.
2. Sound theory + strong experiments. Bounds linking matched deviation to Wasserstein/TV give intuition, and the image/text/tabular results show stronger Pareto fronts.

### Weaknesses
1. Fairness target & positioning. The work focuses on demographic parity. Is there any justification for this choice against other metrics (Equal opportunity/equalized odds/calibration)?
2. Fairness is measured on score distributions (W2(P_{f,0}, P_{f,1})); the relation to thresholded decisions (rate gaps) is unclear. Could you provide a more detailed analysis connecting score-level DP to rate-level DP, with sensitivity to thresholds or post-hoc calibration? 
3. The prior and MH proposal for T (swap k matches) are reasonable, but (i) what distance d is used for images/text (esp. CIVIL)? (ii) How does mixing/acceptance scale with n, class imbalance, and high-dimensional X? 
4. Complexity claims. The paper argues $O(n)$ per update vs. $O(n^2)$ MMD; it would be better if there were some running time/memory tables on large splits and report the cost of the HMC step (leap-frogs, step size) and MH over $T$.
5. Concerns about the dataset. The empirical studies are very strong. However, the adult dataset has some known issues (https://arxiv.org/abs/2108.04884), it would be better to try out a more robust dataset or mention these caveats.
6. Limitations. The current proposed method seems only applicable to binary sensitive attributes. Are there any discussions on how to extend to multi-categorical/continuous ones? Also, how could it be generalized to conditional fairness notions like equalized odds?

I'm willing to raise the score if my concerns are discussed and addressed, thank you.

### Questions
Please see weakness.

### Soundness
3

### Presentation
3

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
This paper addresses the underexplored challenge of integrating model uncertainty into algorithmic fairness. To this end, the authors propose a group-fair posterior distribution and develop a fair variational Bayesian inference framework that embeds fairness constraints into probabilistic learning. To improve computational efficiency, they introduce a novel matched Gibbs posterior, which approximates the fair posterior under fairness constraints using a newly defined metric, matched deviation. This measure is theoretically shown to be closely related to established fairness notions, thereby offering strong fairness guarantees. Empirical evaluations on real-world datasets demonstrate that the proposed approach achieves superior trade-offs between fairness, model uncertainty, and predictive utility compared to baseline methods.

### Strengths
1. The topic of group fairness is highly relevant and socially significant, making this study timely and impactful.

2. The authors effectively integrate group fairness and model uncertainty within a Bayesian inference framework and provide solid theoretical justification for their approach.

3. Experiments conducted on three distinct modalities of datasets demonstrate the robustness and practical applicability of the proposed method.

### Weaknesses
The paper’s writing and structure could be improved for clarity. After reading, several conceptual issues remain ambiguous:

a. What is the motivation for jointly modeling group fairness and uncertainty? Are there concrete real-world applications that benefit from this combination?

b. What are the main technical challenges in combining fairness and uncertainty? Have there been prior studies exploring this intersection? Why can’t existing fairness and uncertainty methods simply be combined?

c. Why is the Bayesian inference framework particularly appropriate for this setting? What specific advantages does it offer compared to prior non-Bayesian approaches?

The work only considers DP as the fairness definition, which is too limited. Other widely used metrics, such as EO and EOdds, should also be discussed. It remains unclear whether the proposed theory and framework can generalize to these metrics.

In the experimental section, fairness evaluation is restricted to the Wasserstein distance under DP. The empirical validation could be strengthened by including additional fairness metrics to better demonstrate the generalizability and robustness of the proposed method.

### Questions
See weaknesses part.

### Soundness
2

### Presentation
2

### Contribution
2

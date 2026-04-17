# A General Framework for Black-Box Attacks Under Cost Asymmetry

- Decision: Accept (Poster)
- Scores: 6, 8, 4, 6

## Abstract
Traditional decision-based black-box adversarial attacks on image classifiers aim to generate adversarial examples by slightly modifying input images while keeping the number of queries low, where each query involves sending an input to the model and observing its output. Most existing methods assume that all queries have equal cost. However, in practice, queries may incur *asymmetric costs*; for example, in content moderation systems, certain output classes may trigger additional review, enforcement, or penalties, making them more costly than others. While prior work has considered such asymmetric cost settings, effective algorithms for this scenario remain underdeveloped. 
In this paper, we introduce asymmetric black-box attacks, a new family of decision-based attacks that generalize to the asymmetric query-cost setup. We develop new methods for boundary search and gradient estimation when crafting adversarial examples. Specifically, we propose *Asymmetric Search (AS)*, a more conservative alternative to binary search that reduces reliance on high-cost queries, and *Asymmetric Gradient Estimation (AGREST)*, which shifts the sampling distribution in Monte Carlo style gradient estimation to favor low-cost queries.
We design efficient algorithms that reduce total attack cost by balancing different query types, in contrast to earlier methods such as *stealthy attacks* that focus only on limiting expensive (high-cost) queries. We perform both theoretical analysis and empirical evaluation on standard image classification benchmarks. Across various cost regimes, our method consistently achieves lower total query cost and smaller perturbations than existing approaches, reducing the perturbation norm by up to 40\% in some settings.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces two techniques for creating adversarial examples against black-box image classifiers when queries have asymmetric cost. These techniques, AS and AGREST, can be plugged into existing attack frameworks (e.g. HSJA, GeoDA...) to substantially increase their performance under asymmetric query costs. AS is a variant of binary search where the interval is split in two parts of length proportional to the cost of the respective queries, and can also be seen as a better version of multi-stage line search. AGREST estimates the gradient around a boundary point by sampling from a hypersphere that is not exactly at the boundary, but shifted toward the low-cost class in order to make fewer high-cost queries, and adjustments are made to estimate the gradient at the original boundary point.

### Strengths
* solid theoretical grounding
* the techniques can be plugged into existing attack frameworks
* introduction of asymmetric gradient estimation, which addresses a question left as an open problem in prior work
* good experimental results
* source code provided

### Weaknesses
* introduction of a new hyperparameter (m) which is set by evaluating the performance of the attacks under different values, and presumably would result in degraded performance when attacking an unknown classifier for which the value of m may not be optimal.
* no tests on binary classifiers, only on multi-class classifiers

### Questions
1) In Table 1, are you able to report confidence intervals?

2) Table 1: intuitively, your method must increasingly outperform other methods as c* grows. That's why I find the HSJA results on ViT-B/32 at c* = 2 surprising: 18.3, 18.3, 2.7 and 2.7. I would have expected 4 values instead of just 2, and results more similar to HSJA on ResNet-50 at c* = 2 or c* = 5. In most other cases, the ResNet-50 and ViT-B/32 results are roughly the same. Is the experiment underpowered? What is the sample size? As a sanity check, did you verify that the performance is the same when you set c* = 1?

3) When you mention the median L2 distance, what is the sample size?

4) Table 5: what do you mean by "Higher queries"? What is the value of c*?

5) Figure 1: what is the cost ratio?

6) 107: there's no Fig. 1 (left). I believe this should be Fig. 3 (right).

7) Do your results hold with a binary classifier, consistent with the motivation of the paper (evading a content moderation classifier) and similar to Debenedetti et al. (e.g. ImageNet-Dogs)?

8) Do you take query discretization into account, as recommended for future work by Debenedetti et al.?

9) Does your method retain appropriate performance when the hyperparameter m is estimated on the same attack but a different classifier than the target one, as would be the case in practice?

Suggestions:
* You could consider converting Table 1 into a plot.

Typos:
 * 377: duplicate Chen et al.
  * 244: I believe that what you call a random variable is instead a sample / realization from a random variable.
    Rating 6
    Confidence

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper addresses the practical and important problem of asymmetric query costs in decision-based black-box adversarial attacks, where queries to a model may incur different costs depending on their output. The authors challenge the common assumption of uniform query costs and propose a general framework that can handle arbitrary cost ratios between "low-cost" (e.g., benign) and "high-cost" (e.g., flagged) queries. The core contributions are two novel techniques: Asymmetric Search (AS), a cost-aware alternative to binary search for finding the decision boundary, and Asymmetric Gradient Estimation (AGREST), a method that shifts the gradient sampling distribution towards the low-cost region and uses importance weighting to maintain estimation accuracy. Through extensive experiments on models like ResNet, ViT, and CLIP, the authors demonstrate that their framework significantly outperforms standard attacks and prior "stealthy" attacks, achieving lower perturbation norms for a given total cost budget.

### Strengths
1. The work tackles a highly significant problem with direct real-world implications. In many practical scenarios, such as content moderation systems or pay-per-query APIs, the cost of a query is not uniform. By generalizing the problem from the niche "stealthy attack" setting (where benign queries have zero cost) to one with arbitrary cost ratios, the authors dramatically increase the practical relevance and applicability of decision-based attacks.

2. The proposed techniques, AS and AGREST, are both novel and technically sound. While AS is an intuitive and clever modification of a standard procedure, AGREST is a more profound contribution. The idea of shifting the sampling center away from the boundary into the low-cost region and compensating with importance sampling is an elegant solution to the core challenge of balancing cost reduction with gradient estimation quality. This is supported by solid theoretical analysis (Theorems 2 and 3), which adds a layer of rigor and confidence to the method.

3. The experimental evaluation is comprehensive and convincing. The authors validate their framework across multiple modern and diverse architectures (CNNs, Transformers, VLMs), demonstrating its general applicability. The comparisons are thorough, including not only vanilla baselines but also a direct and fair comparison against the most relevant prior work, Stealthy HSJA. The ablation studies (Table 1) are particularly effective, clearly isolating the individual and combined benefits of AS and AGREST. The reported performance gains—reducing perturbation norms by up to 40% in some cases—are substantial, not marginal.

4. The paper is exceptionally well-written. The motivation is clearly established using a tangible example (NSFW detection). The methodology is presented logically, aided by clear notation and insightful illustrations (Figure 2). The paper is easy to follow from problem formulation to experimental results, making the complex ideas accessible to the reader.

### Weaknesses
1. The paper's contributions and evaluations are focused exclusively on improving gradient-based decision attacks (HSJA, GeoDA, etc.). However, there exists another class of decision-based attacks that do not rely on explicit gradient estimation, such as evolutionary algorithms[1] or random walks[2]. These methods might be naturally resilient to query-intensive steps and could potentially be adapted to the asymmetric cost setting with simple modifications to their search strategy. The lack of discussion or comparison to this family of attacks makes the evaluation feel slightly incomplete. While gradient-based methods are the state-of-the-art in terms of query efficiency in the symmetric setting, acknowledging and contextualizing these alternative approaches would strengthen the paper's positioning.

2. AGREST introduces a new hyperparameter, m (the overshooting scheduler rate). According to Appendix D.2, this parameter is tuned on a small set of images and then fixed for all experiments, irrespective of the cost ratio c*. This simplifies the method's application but raises concerns about its robustness. The performance might be sensitive to this choice, and an optimal m for c*=100 may not be optimal for c*=10,000.

3. Reliance on the Local Linearity Assumption: The theoretical justification for AGREST relies on the assumption that the decision boundary is locally linear. While the authors acknowledge this is a common assumption in the literature, and the strong empirical results suggest the method is robust in practice, a deeper discussion of the limitations would be beneficial. For instance, on a highly curved boundary, the shifted sampling region of AGREST could lead to a biased gradient that is not fully corrected by the re-weighting, potentially impacting performance more than a standard estimator.

[1] Efficient decision-based black-box adversarial attacks on face recognition. CVPR, 2019.

[2] Aha! Adaptive History-Driven Attack for Decision-Based Black-Box Models. ICCV, 2021.

### Questions
1. The paper focuses on enhancing state-of-the-art gradient-based attacks. Could you elaborate on why non-gradient-based methods, such as a modified Boundary Attack or an evolutionary strategy designed to favor low-cost regions, were not considered as baselines? Do you hypothesize that your gradient-based framework would still be superior, and if so, why?

2. Regarding the hyperparameter m: How sensitive is the attack's performance to the choice of this value? Could you provide a brief sensitivity analysis? Furthermore, have you considered an adaptive schedule where m could be adjusted dynamically based on the cost ratio c* or the attack's progress?

3. The framework is presented for non-targeted attacks. What do you foresee as the primary challenges in extending AS and AGREST to the targeted attack setting, where there may be multiple "high-cost" classes (the original class and non-target classes) and one "low-cost" class (the target class)?

### Soundness
3

### Presentation
3

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
This paper addresses the practical limitation of decision-based black-box attacks that assume all queries have equal cost. The paper proposes a new general framework for asymmetric black-box attacks to minimize the total query cost when certain queries are more expensive. The framework introduces two core components: Asymmetric Search, a boundary search method that splits the search interval based on the query cost ratio to minimize the expected cost; and Asymmetric Gradient Estimation, which shifts the sampling distribution to favor low-cost queries and applies differential weighting for variance reduction. Empirically, the proposed asymmetric framework consistently achieves lower total query costs and smaller adversarial perturbations than existing methods, including prior stealthy attacks.

### Strengths
- The paper tackles the highly practical and often overlooked problem of cost asymmetry in black-box attacks, making the research highly relevant to real-world deployment scenarios.
- The proposed method can be integrated with existing decision-based attacks, offering flexibility. 
- The methods consistently and substantially outperform existing methods by achieving a significantly lower total query cost while maintaining the quality of the adversarial examples.

### Weaknesses
- The claim that the method is superior to existing black-box attacks is unconvincing to me. More specifically, asymmetric cost in Eqn. 3 is optimized in the proposed method and also used in the evaluation, while other methods are optimized with **different objectives** ($c^*=0$ or $\infty$). Incorporating the new objective seems straightforward. I acknowledge the novelty in the problem setting and theoretical analysis, but the novelty in the method is quite trivial to me.
- The main weakness of this paper is the limited evaluation. The experiments only contain three models (ResNet50 and two ViT variants), one dataset (ImageNet), and **no defense**. Especially, there are many adaptive defenses for black-box attacks, which should be considered to support the claim that the method is effective and reduces attack cost compared to other stealthy attacks. Furthermore, the paper should dedicate a section to discuss possible adaptive strategies to defend against AS and AGREST.
- The experiments report the performance with different values of $c^* $, but the paper does not propose any method or heuristic to choose $c^*$, limiting the practicality.

### Questions
Please see Weaknesses.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a general framework for decision-based attacks in the setting where flagged queries have a different cost from non-flagged queries. The main difference from prior work is that their framework does not force the attacker to completely rewrite the attack, and instead leverages the same primitives as standard decision-only attacks.

### Strengths
- The paper introduces general abstractions for black-box attacks that might be useful to apply in different ways, and provide a clean conceptual model for thinking about these attacks
- The paper is easy to read, and the experiments are thorough
- The theoretical analysis is convincing and not completely unrealistic, actually shedding some light on the design of the estimators.

### Weaknesses
- Given that the motivation is to increase the practicality of black-box attacks, it would be interesting to see a real-world study of an adversarial attack pulled off using this method.
- In terms of the actual benefit over Debenedetti et al., it seems like the main thing is just the conceptual advantage of keeping the same primitives as normal black-box attacks, rather than an actual practical advantage
- Conceptually, the AS algorithm seems to be just a weighted binary search, so I'm similarly unsure of the actual novelty there. 

Overall, the paper is a conceptually interesting contribution, but I'm not sure of the practical significance, especially without a real-world case study.

### Questions
- Is this attack more effective against models that have been defended, either via adversarial training or via gradient masking-style defenses?
- What do the distortions look like qualitatively? I couldn't find any examples in the paper.

### Soundness
4

### Presentation
4

### Contribution
2

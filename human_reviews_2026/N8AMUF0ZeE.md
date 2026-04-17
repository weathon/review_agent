# Sample Efficient Corrective Deep Unlearning

- Decision: Reject
- Scores: 4, 2, 6, 4

## Abstract
Machine unlearning enables machine learning models to selectively forget a subset of training data, ensuring compliance with privacy laws and allowing for the efficient removal of outdated or harmful data samples. Current machine unlearning algorithms are restricted to specific models or are applicable only to a subset of learning and unlearning settings, while requiring full knowledge of data points to unlearn. In this paper, we propose a sample efficient corrective deep unlearning algorithm that achieves competitive empirical performance across various unlearning settings without degrading model performance. Our experiments demonstrate that our algorithm achieves strong unlearning performance while requiring only a small computation budget and a small unlearning sample size, thus making it a viable solution for scalable and practical machine unlearning.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper focuses on corrective machine unlearning, where the objective is to restore a model’s performance by removing corrupted or poisoned training data, rather than pursuing privacy-oriented unlearning. The authors propose Selective Parameter Adjustment and ReCalibration (SPARC), a two-stage unlearning framework. In the first stage, SPARC selectively adjusts model parameters based on an activation-difference influence measure; in the second stage, it recalibrates low-importance parameters using orthogonal gradient descent inspired by PCGrad. Experiments cover five unlearning scenarios—backdoor attacks, label confusion, data poisoning, classification errors, and regression tasks—across multiple datasets (CIFAR-10/100, Lacuna-10, AgeDB) and architectures (ResNet-18, AllCNN, ViT-Tiny).

### Strengths
1. The method works across CNNs, ResNets, and Transformers without architecture-specific modifications, demonstrating good generalizability.

2. The paper covers five distinct corrective unlearning scenarios with consistent evaluation across multiple datasets and architectures, providing strong empirical evidence.

3. The proposed method demonstrates robust performance with forget set sizes as low as 2%.

### Weaknesses
1. The paper claims to be “sample efficient,” yet both stages require access to the entire retain set $D_r$. For instance, Eq. (1) explicitly computes activation differences between $D_f$ and $D_r$, and Eq. (5) requires gradients from both. The main concern lies in the use of $D_r$, which, according to the paper’s definition, corresponds to $D_{\text{train}} / D_u$. This potentially dominates the computational cost, contradicting the claim of sample efficiency. This issue is not sufficiently discussed in the paper.

2. The paper lacks a systematic analysis of individual components—such as $|\theta|$, incoming activations, outgoing activations—and the designed hyperparameter $\tau$. More detailed ablations would help clarify the contribution of each element to the overall performance.

3. Missing discussion and comparison with related works on corrective unlearning, particularly [1,2]. If my understanding is correct, the “SAP” mentioned in the experiments does not refer to the existing method [2], which may cause confusion. A more explicit comparison and discussion would strengthen the paper.  

[1] Wei S, Zhang M, Zha H, et al. Shared adversarial unlearning: Backdoor mitigation by unlearning shared adversarial examples. NeurIPS 2023.

[2] Kodge S, Ravikumar D, Saha G, et al. SAP: Corrective Machine Unlearning with Scaled Activation Projection for Label Noise Robustness. AAAI 2025.

### Questions
1. What is the actual wall-clock time breakdown between $D_f$ and $D_r$ processing?

2. Can the method work with a random subset of $D_r$ (e.g., 10%)? How does $|D_r|$ affect unlearning performance?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper tackles “corrective unlearning”: a setting where we discover that part of the training data is "bad" (corrupted, malicious, mislabeled, etc), and we want to remove its influence from the model without a full retrain. In this setting the unlearning algorithm is given only a fraction of 'bad' examples and the algorithm aims to remove the influence of all without damaging utility.

The proposed approach, SPARC, works in two stages: (i) it estimates which internal parameters are disproportionately important by the bad examples compared to clean ones (looking at activations along forward paths through the network). and (ii) SPARC then fine-tunes on clean examples trying to avoid reintroducing the bad behavior.

### Strengths
1. the problem studied is practical/realistic and interesting and differs compared to solutions for 'traditional' unlearning methods.

2. Their two-phase approach makes sense (removing influence and recalibrating to ensure utility).

3. The paper aims to address and evaluates against multiple tasks (classification, regression-like settings, sequential unlearning), and reports both unlearning and utility performance.

### Weaknesses
1. A major weakness is that the work makes an **implicit assumption that the bad data/behavior (and their pathways) pathways are clearly/markedly separable from acceptable data/behaviour.** It is only then that their method will not hurt model utility, right?. 
This assumption (if correct) should be:

(i) clearly **spelled-out** in their paper; 

(ii) **analyzed**, w.r.t. to which target applications this assumption makes more or less sense, eg: 

- for backdoor attacks, this probably makes sense;

- for targeted poisoning attacks, this may make less sense;

- for inter-class confusion cases, where the “bad” signal can overlap (correct) decision boundaries, one could argue that this assumption does not make sense; 

(iii) **test explicitly** the above (and maybe other) cases, quantifying possible utility loss for each.

2. There are serious misgivings about the paper as it does not discuss, analyze, and quantify performance **based on how representative is and on the coverage of the given forget set**. One would have expected that this is studied in detail, showing that things like: 

(i) the **representativeness**: the cardinality of the forget set is important, ie may be too small for being representative of the *exact* set of examples to be unlearned. Their results on varying the size of the forget set are valuable. But, is it not preferable to experiment, say with forget sets activating pathways which are used by 'normal' data?

(ii) the **coverage** of the original forget set versus that of the full (forget) set to be unlearned.

Basically, if the starting point (forget set) is not "good enough" (wrt coverage or representativeness), the method may end up either hurting 'forget performance' or model utility.

**1+2 above negate the selling of the new method** that says "Give me a forget set and I will deliver true forgetting and utility for all bad exampels, and for all applications of bad data/behavior (such as backdoor attacks, data poisoning, class confusion etc.).

3. Another key concern centers around the set of chosen **baselines**. These are rather weak and do not represent SOTA methods.

- I would strongly urge the authors to consider using the Salun method (https://arxiv.org/abs/2310.12508), especially since it is also based on defining **saliency** (albeit differently) of examples, which drives unlearning.

- The authors appear to be unaware of a new SOTA method, called RUM (https://arxiv.org/abs/2406.01257) and its companion 
(https://arxiv.org/abs/2410.16516) which actually can improve performance substantially. **RUM is especially relevant here as one may intuit that the separability of learning/activation pathways (see 1 and 2 above) depends on the degree of memorization of examples**. And this is what RUM leverages in order to improve both forgetting performance and utility. For example, for low-memorized examples in the forget set it does nothing; for medium ones and for highly-memorized ones, it uses different unlearning algorithms... 

4. One key issue that is also overlooked is that the work seems to not address the possibility that, after their recalibration step (to maintain utility), their **method may be inadvertently reintroducing "bad" behaviour.** **Your metric meant to appease us of this possibility is an average**, missing the possibility that **some bad behavior (associations) may have been reintroduced**. So, for safety and compliance, knowing whether this is a failure mode and to what extent is necessary.

5. it is not clear if baselines were afforded the same "budget" for hyper-parameter tuning as the proposed method (for which it appears that it is very large).

### Questions
Please address all above weaknesses.

The paper is lacking and should provide:
1. A study where the full set of corrupted samples, S_c is fixed. Then they could subsample from S_c forming the provided forget set, S. Then they can quantify how well unlearning "generalizes" to the rest of S_c (not just to S).

2. Measurements of accuracy/utility specifically on S_rel (set of retain samples that are closest (in their activation pathways) to the provided forget examples, before vs after unlearning.

3. The paper should study better baselines (eg SalUn and RUM) and whether (i) memorization affects their separabiloty (assumption) and (ii) how the performance of their method compares against SalUn and RUM. In my view both SalUn and RUM are especially relevant.

4. Study/analyze the possibility for "recontamination" - especially important as the work is about corrective learning.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces SPARC, a novel, architecture-agnostic algorithm for corrective machine unlearning. Unlike prior methods designed primarily for privacy-oriented unlearning, SPARC targets correcting the influence of corrupted or poisoned data. The method has two stages: a) Selective Parameter Adjustment (SPA) which identifies and attenuates parameters with high influence on the forget set relative to the retain set via an activation-based influence score, b) ReCalibration, which is an orthogonal gradient descent on low-influence parameters to restore utility while avoiding relearning of forget samples.Experiments across multiple datasets and model architectures show that SPARC achieves competitive or superior performance in both unlearning success and retained utility under various settings (classification, regression, feature/label/label+feature manipulation), with a small computational budget.

### Strengths
**S1**.  The influence computation is forward-pass only, making the approach computationally efficient. Demonstrating strong results with as little as 2% forget-set data is compelling for real-world applications.

**S2**. The two-stage forget then recalibrate process is logical (and has appeared in prior unlearning works too), and the core contributions are strong. 

**S3**. The experiments are comprehensive covering different baselines, datasets, and model architectures.

### Weaknesses
**W1**. Although the combination is novel, both components i.e., parameter influence estimation and orthogonal gradient descent, draw heavily from existing ideas. The contribution lies in their integration for unlearning rather than a new learning principle.

**W2**. The authors report that for regression unlearning, the SPA component (forgetting-only) performs best, and the full SPARC algorithm (with recalibration) underperforms it. This is a significant finding that directly contradicts the general applicability of the full SPARC method. The paper's justification, that it is "less suited for continuous targets" is vague and unsatisfying.

**W3**. Although efficiency is discussed qualitatively, the paper lacks explicit wall-clock or FLOPs comparisons vs. baselines like SCRUB or SSD to substantiate “sample-efficient” claims quantitatively.

**W4**. The paper can benefit from more ablations. For example, what happens if the influence measuring method is replaced by gradient-based influences?  What happens if the activation path is longer than just one layer before and after the parameter? What is the effect of the choice of the activation function? 

**W5**. I suggest moving section 2.1 to after section 3, as it’s more similar to the experimental setup.

### Questions
**Q1**. How sensitive is the influence measure to activation function choice or network depth?

**Q2**. In the regression setting, did you measure how far predictions are pushed outside the forget range (i.e., margin of forgetting)?

**Q3**. How would SPARC perform if the recalibration step were replaced by ordinary fine-tuning without orthogonal projection?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes a sample-efficient correction framework for machine unlearning. The author aims to reduce the reliance on the knowledge of data points to unlearn without degrading model performance. The proposed approach contains two steps: 1. a targeted parameter adjustment step to selectively forget undesired data 2. a recalibration phase that updates low-importance parameters
to recover overall model utility. The approach is evaluated on several benchmark datasets, showing strong performance while using fewer correction samples and lower computational cost.

### Strengths
1. The focus on sample-efficient correction directly addresses a key problem in existing unlearning methods.
2. The paper provides a clear problem formulation and the motivation is well-grounded.
3. The paper provides extensive experiments across multiple datasets, architecture and scenarios.
4. The paper presents experiments on multiple datasets, model architectures, and unlearning scenarios. The results are solid, demonstrating clear improvements in computational efficiency and comparable accuracy recovery relative to full retraining.

### Weaknesses
1. Experimental results are presented as plots without numerical tables. Some plots are dense and difficult to read precisely. Providing corresponding tables with numerical values would make the comparisons clearer and improve interpretability/readablity of the results.
2. The overall experimental scope remains moderate. It does not include large-scale datasets that could demonstrate scalability.
3. The design choices are described at a high level, but their rationale and potential alternatives are not thoroughly discussed.
4. While the paper clearly explains the overall framework and includes a basic ablation study on module contributions, the rationale behind key design choices is not well justified/discussed. Moreover, no detailed sensitivity analysis is conducted to examine robustness to important hyperparameters.

### Questions
1. How does the proposed perform on large-scale datasets? 
2. Would the authors consider providing numerical tables corresponding to the plots? Many figures are dense and difficult to read precisely, and tables could make the quantitative comparisons clearer and easier to interpret.
3. Could the authors elaborate on the motivation behind specific design choices in the proposed method? Were alternative designs considered, and how did they perform?
4. Beyond the current ablation, have the authors examined the effect of important hyperparameters? A more detailed sensitivity analysis could help assess the robustness and general applicability of the proposed under different conditions.

### Soundness
2

### Presentation
2

### Contribution
3

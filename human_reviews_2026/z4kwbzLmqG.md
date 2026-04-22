# Explaining to Learn: Regularization Using Contrastive Visual Explanation Pairs For Distribution Shifts

- Avg Score: 3.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 6, 2, 4

## Abstract
While a myriad of algorithms have been proposed to address distribution shifts, most algorithms are known to perform best only under specific conditions and fail to outperform the baseline empirical risk minimization (ERM) in other scenarios. Furthermore, the algorithmic complexity of some existing methods can render them less interpretable, and their approach to addressing spurious correlations, a hallmark of distribution shifts, is often indirect. To specifically address spatial confounders, we propose Explaining to Learn (ETL), an interpretable, explanation-based learning algorithm that removes spatial confounders from the primary classifier's latent representations during training. ETL achieves this by penalizing the similarity between GradCAM activation maps from a primary label classifier and a concurrently trained confounder classifier. On the more recent and difficult Spawrious Many-to-Many Hard Challenge benchmark, ETL achieves an average accuracy (AA) of 82.24% (±3.87) and a worst-group accuracy (WGA) of 66.31% (±8.73), outperforming leading state-of-the-art (SOTA) benchmarks by a significant 5% and 11%, respectively. This strong performance extends to other challenging benchmarks, where ETL also outperforms SOTA regularization methods on CMNIST (AA: 69.02% ±0.53; WGA: 67.63% ±1.39) and Waterbirds (AA: 92.12% ±0.67; WGA: 86.92% ±0.56). We complement these empirical results with theoretical analyses, demonstrating the viability of explanation-based learning for mitigating distribution shifts.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes Explaining to Learn (ETL), an explanation-based regularization method to improve robustness to distribution shifts by penalizing simialrity between heatmaps generated from GradCAM of a label classfier and a confound classifier. The authors claim that this approach explicitly removes spatial confounders and enhances interpretability. Experiments on CMNIST, Waterbirds and CelebA show superior performance over simple baseline such as GroupDRO.

### Strengths
1. The idea of incorporating GradCAM-based explanations directly into the training process to mitigate distribution shifts is novel and conceptually appealing.

2. Experimental results show consistent improvements across multiple benchmarks, including Waterbirds and CelebA, demonstrating the method’s effectiveness in diverse dataset.

3. The approach enhances interpretability by explicitly linking model decisions to visual explanations.

### Weaknesses
1. The training procedure is difficult to follow. The paper does not provide a concise, end-to-end description of the full training pipeline, and the reader have to reconstruct it from scattered equations and algorithm boxes.

2. The method depends on GradCAM, which is designed specifically for CNN-based architectures. This limits its generality and prevents application to modern architectures such as Vision Transformers (ViTs). No experiments are conducted with other attribution methods (e.g., Integrated Gradients or GradCAM++), which could help demonstrate whether the proposed idea generalizes beyond CNN settings.

3. Since GradCAM requires gradient computation with respect to intermediate feature maps, incorporating it in every training iteration can significantly increase both training time and memory cost. The paper does not quantify or analyze this additional overhead, making it unclear whether ETL remains efficient compared to simpler baselines such as GroupDRO.

4. The experimental evaluation omits several recent competitive methods yet proposed early, such as JTT proposed in 2021 [1] and DFR proposed in 2023 [2]. These methods achieve substantially higher worst-group accuracies on benchmarks like Waterbirds (e.g., DFR: 92.9%), where the performance of ETL is not competitve (ETL-MAE: 87.45%) while requiring more data annotation (confounder) and computation cost (GradCAM).

[1] Just Train Twice: Improving Group Robustness without Training Group Information, ICML 2021.

[2] Last Layer Re-Training is Sufficient for Robustness to Spurious Correlations, ICLR 2023.

### Questions
1. Would ETL still work with other attributions that can be generalize to other architectures (e.g. Integrated Gradient, Attention Rollout)?

2. Could the authors provide comparison with more advanced methods especially on spurious correlation?

3. What is the additional computation cost (FLOPs / clock-wise time) of ETL compared to vanilla baseline?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes Explaining to Learn, a novel training framework that leverages contrastive GradCAM explanations to remove spatial spurious correlations under distribution shifts. Instead of solely aligning representations or reweighting groups, ETL jointly trains a primary classifier and a confounder classifier, and penalizes similarity between their GradCAM activation maps. This encourages the model to rely on semantically meaningful regions while suppressing confounding visual cues.

Experiments on CMNIST, Waterbirds, CelebA, and the Spawrious benchmark show that ETL improves WGA and is particularly strong under complex many-to-many background shifts.

### Strengths
-Turning GradCAM into a training signal is fresh and intuitive. The paper bridges explainability and robustness research, opening a direction of explanation-guided learning.
- Explicitly tackles spatial spurious correlations: Directly regularizes where the model looks, rather than indirectly enforcing invariance.
- Significant margins on Hard Spawrious benchmark (+5% AA, +11% WGA)
- Clear motivation & narrative: Easy to follow the causal logic -- don’t look at confounder pixels.

### Weaknesses
- assumes environment/confounder annotations, which limits applicability in scenarios without such metadata.

- heavily on GradCAM accuracy. GradCAM is coarse and can be noisy or misleading in some architectures and tasks.

- computation overhead: Two models + GradCAM computation increases training cost.

- focuses only on spatial confounders: may not generalize to style, texture, frequency, temporal, or high-level semantic spurious cues.

### Questions
see Weakness

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
5

### Summary
The authors propose a twin network, trained in parallel for mitigating bias and spurious correlation effects. One network learns the spurious attribute while the other one learns the real task label. During training, they compute the grad-CAM heatmaps for both networks and enforce them to be dissimilar. This simple, yet effective strategy seems to be capable of removing bias on different datasets. The proposed method outperforms all Domain Generalization baselines that perform on environments while it is also better than GroupDRO in most settings.

### Strengths
The proposed method is intuitive, simple, yet powerful. The idea is good enough to be worth a publication but the experimental setup and baseline choice are fragile. The choice of relevant and interesting bias mitigation datasets, including using multiple of them, is a strong point. Providing the code to let the reviewers and readers understand their implementation details is highly appreciated.

### Weaknesses
While the idea is very lean and useful, the execution and contextualization of this work have room for improvement. This idea has potential, but it is not fully explored in this submission.

- Contextualization: the idea is mainly framed in the field of distribution shifts, especially in the context of domain generalization. This, however, is not only limited but also misleading or simply wrong. There is tons of work in the field of bias mitigation and shortcut removal that are not considered at all. The only really relevant baseline method is GroupDRO that operates on a group level, thus being actually capable of removing biases/shortcuts effectively.

- Baselines: Related to the previous point, we also see that the choice of baselines in their experiments is heavily biased and mismatched. Instead of comparing against true bias mitigation and shortcut removal methods, the authors compare mainly against domain generalization methods. Kaur et al. [1] have shown that explicit methods that have access to groups outperform environment (domain) based methods. They also directly use a MMD-based approach that operates on groups (but priorly, other works already proposed MMD on groups, such as [2], [3]). Instead of using DANN, the authors should have further used a method that debiases the confounding attribute instead of the domain, such as [4,5, ...]. One can continue this list with a myriad of baselines that would have been more appropriate than the environment-based methods. Thus, these experiments are not convincing, as it is expected that group-based methods outperform these significantly [1].

- Unnecessary theoretical analysis: Subsection 3.3 feels bloated. The main message is simple and clear, and making the assumptions explicit is appreciated. Apart from that, there is a limited benefit in this subsection, since there are not concrete bounds for the Lipschitz analysis presented in the manuscript: it is well-known that chaining classical deep learning activations and matrix operations together with an explicitly defined Lipschitz function (Def. 1) yields another (locally) Lipschitz function. Without giving bounds, all this analysis is unnecessary as it was clear from the beginning that the regularization penalty will be smooth enough (as it already is differentiable, operates with well-known functions, etc.).

- Confusing notation and intransparency: it is unclear how the environments and the confounding variables relate. Without the code, this reviewer would not have been able to understand how the authors defined the different training and testing envs for the domain generalization methods. This adds a layer of intransparency and made the paper hard to follow.


[1] Kaur, Jivat Neet, Emre Kiciman, and Amit Sharma. "Modeling the Data-Generating Process is Necessary for Out-of-Distribution Generalization." The Eleventh International Conference on Learning Representations.
[2] Makar, Maggie, et al. "Causally motivated shortcut removal using auxiliary labels." International Conference on Artificial Intelligence and Statistics. PMLR, 2022.
[3] Veitch, Victor, et al. "Counterfactual invariance to spurious correlations in text classification." Advances in neural information processing systems 34 (2021): 16196-16208.
[4] Mitigating Unwanted Biases with Adversarial Learning
[5] Adeli, Ehsan, et al. "Representation learning with statistical independence to mitigate bias." Proceedings of the IEEE/CVF winter conference on applications of computer vision. 2021.

### Questions
Could the authors add highly relevant bias mitigation and shortcut removal methods introduced in the weaknesses section in an extensive comparison? This would make the entire work more credible.

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a method that uses GradCAM similarity penalties between a label classifier and confounder classifier during training. The goal is to remove spatial confounders from latent representations. The method penalizes similarity between GradCAM activation maps from both classifiers. Experiments show improvements on CMNIST, Waterbirds, and Spawrious benchmarks compared to baselines like GroupDRO.

### Strengths
1. Shows improvements on multiple benchmarks. Particularly strong on Spawrious Many-to-Many Hard benchmark (66.31% WGA vs 54.40% for GroupDRO).

2. Tests multiple similarity functions. Includes GradCAM visualizations and UMAP plots showing confounder separation.

3.  Paper is well-written. GradCAM visualizations effectively show the method addresses spatial confounders.

### Weaknesses
1. Using explanation-based methods for removing confounders is not new. The paper's own related work cites Hagos et al. (2022) and Dammu & Shah (2023) who use explainability for spurious correlations. The core idea of matching/penalizing explanation maps has been explored before.

2. Only works for spatial confounders in images. Cannot handle non-spatial spurious correlations.

3. Lipschitz continuity proof is trivial. Just shows loss is stable, doesn't explain why penalizing GradCAM similarity removes confounders or improves OOD generalization.

4. Assumption 2 (different activation maps) is critical but never validated empirically. May not hold in practice.

5. Trains two full networks plus computes GradCAM at every step. Paper doesn't report actual training time.

### Questions
1. Hagos et al. (2022) already used explanation-based learning for spurious correlations. How is ETL fundamentally different?

2. Why GradCAM specifically?: Other explanation methods exist. Why is GradCAM the right choice? Did you compare to other explanation methods?

3. What's the actual wall-clock training time compared to baselines?

### Soundness
3

### Presentation
2

### Contribution
2

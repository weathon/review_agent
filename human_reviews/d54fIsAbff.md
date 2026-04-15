# GSE: Group-wise Sparse and Explainable Adversarial Attacks

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 5

## Abstract
Sparse adversarial attacks fool deep neural networks (DNNs) through minimal pixel perturbations, often regularized by the $\ell_0$ norm. Recent efforts have replaced this norm with a structural sparsity regularizer, such as the nuclear group norm, to craft group-wise sparse adversarial attacks. The resulting perturbations are thus explainable and hold significant practical relevance, shedding light on an even greater vulnerability of DNNs. However, crafting such attacks poses an optimization challenge, as it involves computing norms for groups of pixels within a non-convex objective. We address this by presenting a two-phase algorithm that generates group-wise sparse attacks within semantically meaningful areas of an image. Initially, we optimize a quasinorm adversarial loss using the $1/2$-quasinorm proximal operator tailored for non-convex programming. Subsequently, the algorithm transitions to a projected Nesterov's accelerated gradient descent with $2$-norm regularization applied to perturbation magnitudes. Rigorous evaluations on CIFAR-10 and ImageNet datasets demonstrate a remarkable increase in group-wise sparsity, e.g., $50.9\%$ on CIFAR-10 and $38.4\%$ on ImageNet (average case, targeted attack). This performance improvement is accompanied by significantly faster computation times, improved interpretability, and a $100\%$ attack success rate.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces GSE (Group-wise Sparse and Explainable Adversarial Attacks), a novel method for generating sparse adversarial attacks on Deep Neural Networks (DNNs). These attacks aim to perturb semantically meaningful regions of an image, making the attacks more explainable. The authors propose a two-phase algorithm to generate such group-wise sparse adversarial examples: The first phase utilizes 1/2-quasinorm regularization to select the most critical pixels for perturbation using a proximal operator. The second phase applies a projected Nesterov’s accelerated gradient descent to fine-tune perturbations in targeted regions, minimizing their magnitude while maintaining effectiveness. The approach is evaluated on CIFAR-10 and ImageNet datasets, demonstrating significant improvements in group-wise sparsity and explainability while achieving a high attack success rate and reduced time consumption.

### Strengths
1. The methodology is well-formulated, and the proposed two-phase approach is sound. Rigorous experiments demonstrate the efficacy of the method, comparing it with SOTA adversarial attacks.

2. The paper is well-structured, with clear explanations of the algorithms.

### Weaknesses
1. The paper relies on grid search for selecting hyperparameters, but there is no detailed analysis of how sensitive the method’s performance is to these choices.

2. The rationale for the specific update mechanism in Equation 9, where tradeoff parameters decrease slower than they increase, is not fully explained. It would be helpful to understand how this impacts the selection of critical regions for perturbation and whether a more symmetric adjustment could improve results.

### Questions
1. In Phase Two, why did you choose to use gradient computation instead of applying a proximal operator update for the $\ell_2$ norm? Would using a proximal operator provide any computational or optimization benefits, especially regarding convergence and perturbation magnitude control?

2. The paper mentions that a grid search was used to select hyperparameters. How sensitive are the results to these hyperparameters? Could slight variations in the hyperparameters significantly impact the effectiveness, sparsity, or explainability of the adversarial attacks?

3. I noticed that in Equation 9 and 10, the update for the blurred perturbation mask causes the tradeoff parameters to decrease more slowly than they increase. Could you explain the rationale behind this design choice? How does the asymmetry in the speed of adjustment affect the performance and explainability of the generated adversarial examples?

### Soundness
2

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
4

### Summary
This paper introduces a novel approach for generating sparse adversarial perturbations targeting deep neural networks (DNNs). GSE achieves higher group-wise sparsity than state-of-the-art (SOTA) methods, allowing for fewer, yet targeted pixel modifications. This results in minimal visible changes while effectively fooling the DNNs.

### Strengths
1. The paper is well-organized and written. The problem setup, objectives, and contributions are articulated in a way that is accessible to readers.
2. The GSE framework achieves high levels of sparsity and group-wise organization in perturbations, making it a valuable tool for evaluating and improving the robustness of DNNs.

### Weaknesses
1. The GSE method’s effectiveness seems sensitive to specific hyperparameters.
2. The authors search the hyperparameters only for their method. This may not be fair to other methods. I think the authors should search the hyperparameters for all methods.
3. An ablation study on $\hat{k}$ should be given since the first phase will highly affect the performance.
4. The experiments only have two datasets. The performance on more datasets is required.

### Questions
See weakness

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
4

### Summary
This paper presents a novel sparse attack method named GSE (Group-wise Sparse and Explainable), designed to generate structured adversarial attacks against deep neural networks (DNNs).

The GSE algorithm emphasizes group-wise sparse attacks, meaning the perturbations are strategically grouped within semantically meaningful regions of the image, rather than being scattered randomly.

**Limitations in Previous Work:** Prior methods primarily focus on minimizing the number of altered pixels without imposing constraints on their location or intensity. Consequently, these modified pixels often show considerable variations in color or intensity relative to their surroundings, making the alterations more noticeable. This highlights the need for structured, group-wise sparse perturbations that better target key areas of the image. However, I feel this motivation is somewhat broad, as it applies to all group-based adversarial attacks.

**Two-Phase Approach to Solving the Optimization Problem:**
- **Step 1:** This phase identifies the pixels most vulnerable to perturbation. It employs a non-convex regularization approach using the 1/2-quasinorm, allowing for individual tuning of tradeoff parameters (\(\lambda\)) for each pixel. The algorithm iteratively reduces these parameters for pixels close to those already perturbed, making them more susceptible to further modification. I believe this strategy prevents further updates on perturbed pixels, effectively addressing issues with high-intensity or conspicuous changes. However, the authors assert that this step primarily reduces the number of pixels that need alteration—a claim that seems misaligned with the method’s description, in my view.
   - This step can be achieved through a heuristic adjustment of \(\lambda\), with individualized settings for each pixel.
  
- **Step 2:** This phase constructs a low-magnitude adversarial attack by optimizing an objective function over the pixels chosen in Step 1. The objective includes the classification loss and a penalty term to control the magnitude of perturbation. The algorithm employs projected Nesterov's Accelerated Gradient (NAG) descent to efficiently solve this refined problem.

The authors claim that the proposed method achieves both effective attack performance and, importantly, enhanced explainability.

### Strengths
•	The proposed method appears to be faster than previous approaches.

•	The number of perturbed pixels is also fewer compared to prior work, which, as the authors claim, might contribute to better explainability.

### Weaknesses
- **Writing Quality:** Several areas need improvement.
  - **Flow of the Paper:** Despite multiple readings, the main limitations of previous work that this paper aims to address remain unclear. Specifically, the introduction (lines 45-53) outlines the motivation for group-wise sparse attacks in general, but it doesn't clearly articulate the specific issue that this paper seeks to tackle.
  - **Detail in Section 2.2:** This section is overly detailed but lacks sufficient explanation to help readers understand how (1) it identifies a smaller set of suitable pixels and (2) achieves lower perturbation magnitudes. Additionally, the AdjustLambda algorithm would benefit from a standalone paragraph with clear notation to help readers follow the paper more easily.
  - **Confusing Notations:** Some notations are inconsistent and confusing. For instance, in Section 3.2, `n` represents both the number of images and the size of a patch (n by n pixels), while `m` refers to both the mask and the first `m` adversarial examples (line 256).

- **Evaluation Metrics:** The evaluation metrics are somewhat questionable. Specifically, in Section 3.2, the authors propose using Attack Success Rate (ASR) = m/n, where `n` is the total number of images. However, it would be more appropriate for `n` to represent the number of correctly classified images by a classifier. The use of the IS score is also questionable. More specifically, the IS score show the alignment between the Adversarial Saliency MAP (ASM) and the perturbation w, generated by attacks. However, the ASM is calculated based on the gradient of the prediction loss w.r.t. true label. Therefore, based on insight from the Short-cut learning paper, if the model based on the imperceptible (and non-sensible feature) to make the prediction, the ASM map is align with this feature instead of the human perspective or human explainability. 
- **Interpretability Technique:** Grad-CAM is generally a more advanced method than CAM, raising questions about its suitability for comparison.

### Questions
1. Why is it necessary for adversarial attacks to be explainable? More specifically, why should the attacks follow a particular structure? Insights from the shortcut-learning paper suggest that DNNs may learn features differently than humans. Is the structure of group-based attacks intended to align more closely with human interpretation?
2. What are the benefits of explainable attacks compared to non- or less-explainable ones?
3. Could the authors explain in greater detail why the proposed method outperforms previous attacks in terms of speed (Table 1)? Were consistent parameter settings used across methods, such as the number of iterations \(\hat{k}\) and \(K\)? In my opinion, the most time-intensive steps are the gradient calculations in lines 3 and 8 of Algorithm 2, which would likely be similar in duration to other methods if using the same number of iterations.
4. The authors compare their method with four baselines (Homotopy, SAPF, StrAttack, FWnucl) but only provide results for Homotopy and SAPF in Table 1. Could the authors explain this choice?
5. In lines 149–152, the authors mention, “this approach does not guarantee that the perturbations will affect the most critical pixels in the image.” How do they define or identify these “most critical pixels”? Based on which metric?
6. I am interested in the interpretability score of Grad-CAM or CAM techniques.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
In this paper, the authors propose a two-stage approach for generating group-wise sparse and explainable adversarial examples. To enhance the attack performance, they introduce 1/2-quasinorm regularization and sparse hyperparameter tuning methods to select key group-wise sparse pixels. After that, the coordinate and magnitude of these selected pixels are further refined using Nesterov’s Accelerated Gradient Descent (NAG) to produce the sparse adversarial example. Experiments on multiple CNN and ViT networks using the CIFAR and ImageNet benchmark datasets demonstrate the effectiveness and efficiency of the proposed group-wise sparse and explainable attack compared to baseline sparse attack methods.

### Strengths
1. Compared to existing methods that use L0 or L1 regularization, this paper presents a sparse adversarial attack based on 1/2-quasinorm regularization. Building on this, an effective heuristic approach for tuning the regularization hyperparameter is introduced to select the most important pixels.

2. For the selected pixels, a projected Nesterov’s accelerated gradient descent algorithm is introduced to solve the L2-norm constrained problem, thereby ensuring low-magnitude sparse adversarial examples.

3. Experimental results conducted on multiple white-box models in both targeted and untargeted attack settings demonstrate that the proposed GSE method achieves better attack effectiveness and efficiency compared to other sparse attack methods.

### Weaknesses
1. The motivation behind the proposed 1/2-quasinorm regularization is not clearly explained. Compared to existing L0- or L1-norm based sparse attacks, what advantages does 1/2-quasinorm regularization offer for crafting sparse adversarial examples? Additionally, why not consider combining L0- or L1-norm based regularization with the proposed coefficient tuning strategy?

2. The derivation between Eq. 3 and Eq. 4 appears problematic, as it should rely on separable proximal functions. However, unlike the L1-norm, the Lp-norm is not a separable function, particularly when p=1/2. 

3. The authors claim to introduce a group-wise sparse attack. However, Sections 2.1-2.2 lack a clear mathematical description of how the selected sparse pixels are grouped, resulting in ambiguity regarding the concept of group-wise sparse attack.

4. The evaluation of attack performance lacks comprehensiveness. For example, while sparse attacks are applied to white-box models, their transferability in a black-box attack setting is not considered, which is arguably a more practical scenario. Additionally, all attacks are performed using 200 iterations, which is inconsistent with the settings used in previous sparse attack methods.

5. The evaluation metrics are not clearly explained. First, since different sparse attack methods may generate distinct sets of successful adversarial examples when ASR is below 100%, the ACP metric defined in Eq. 15 becomes difficult to compare across methods. Second, while Eq. 16 introduces the d_{2,0} metric, the definition of “overlapped patches” is not provided, adding to the ambiguity.

### Questions
1. How does the proposed coefficient tuning strategy perform in terms of attack effectiveness when combined with different sparsity-inducing regularizations, such as L0-norm, L1-norm, and 1/2-quasinorm?

2. Could the authors provide additional details on how Eq. 4 is derived from Eq. 3? Additionally, could the authors explain why the Lp-norm (p=1/2) is considered a separable function?

3. Compared to other sparse attacks (e.g., StrAttack, Homotopy-Attack) that provide a clear mathematical formulation of group sparsity, could the authors offer additional details on how sparse pixels in the adversarial example are grouped during the two-phase perturbation training process?

4. How does the performance of other sparse attacks, using their default iteration settings, compare to that of the proposed GSE attack in this paper? 

5. What is the transferability performance of different sparse attacks in the black-box setting? Does the proposed GSE attack still achieve a higher ASR?

### Soundness
2

### Presentation
2

### Contribution
2

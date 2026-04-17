# Exploring Mode Connectivity in Krylov Subspace for Domain Generalization

- Decision: Accept (Poster)
- Scores: 2, 8, 8

## Abstract
This paper explores the geometric characteristics of loss landscapes to enhance domain generalization (DG) in deep neural networks. 
Existing methods mainly leverage the local flatness around minima for improved generalization. However, recent theoretical studies indicate that flatness does not universally guarantee better generalization. Instead, this paper investigates a global geometrical property for domain generalization, i.e., \emph{mode connectivity}, the phenomenon where distinct local minima are connected by continuous low-loss pathways. Different from flatness, mode connectivity enables transitions from poor to superior generalization models without leaving low-loss regions. 
To navigate these connected pathways effectively, this paper proposes a novel Billiard Optimization Algorithm (BOA), which discovers superior models by mimicking billiard dynamics. 
During this process, BOA operates within a low-dimensional Krylov subspace, aiming to alleviate the curse of dimensionality caused by the high-dimensional parameter space of deep models. 
Furthermore, this paper reveals that oracle test gradients strongly align with the Krylov subspace constructed from training gradients across diverse datasets and architectures. 
This alignment offers a powerful tool to bridge training and test domains, enabling the efficient discovery of superior models with limited training domains.
Experiments on DomainBed demonstrate that BOA consistently outperforms existing sharpness-aware and DG methods across diverse datasets and architectures.
Impressively, BOA even surpasses the sharpness-aware minimization by 3.6\% on VLCS when using a ViT-B/16 backbone.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
To find flat minima that perform better on the test data under distribution shift, the paper considers mode connectivity. It proposes a Billiard Optimization Algorithm (BOA) that traverses the flat basin of the loss landscape analogous to the reflection of a billiard on the board. To find the search direction effectively, the paper leverages the observation that test loss gradients align with the Krylov subspace. Experiments on five datasets show the potential of BOA.

### Strengths
1. It is meaningful to use mode connectivity to find better flat minima in domain generalization. The observation that test loss gradients align with the Krylov subspace is interesting and inspiring. 

2. BOA proposes a novel strategy to search over the loss landscape. 

3. Figure 6 is very helpful in understanding how BOA works.

### Weaknesses
1. Figure 1 suggests that minima that are equally flat on the loss landscape of the training distribution can have different sharpness on the loss landscape of test data. The observation motivates finding better minima through mode connectivity. BOA relies on a validation set of the test data to find an optimal model. However, in domain generalization, it is common to assume that test data is not accessible. My biggest concern is that under this condition, how to guarantee that the minima found via BOA are better? 

2. The clarity of section 3 should be improved to aid understanding. For example, some notations in section 3 are not explained in the main paper (e.g. $\alpha, \epsilon$), and the role of $h$ and its connection with $\alpha$ are not explicitly stated.

### Questions
1. How is $\alpha^*$ in equation 4 determined?

2. BOA's reflection results in symmetric exploration of the loss landscape. The geometry of the loss landscape and initial search points both affect the efficiency of the symmetric search. How robust is the symmetric exploration in searching flat minima when the flat basin is not symmetric, and the initial search point lies at the center of the flat basin?

3. Does the search angle $\phi$ affect the search of optimal model? Why is it set to a fixed value instead of as a hyperparameter?

4. How is $\epsilon$ in equation 8 determined? 

5. Is it a fair comparison with other DG methods, given that many of them do not utilize a validation set of test data?

### Soundness
2

### Presentation
2

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
This paper proposes a new optimization framework, termed "Billiard Optimization Algorithm" (BOA), to improve domain generalization using VPT with ViT backbones. This algorithm leverages the "mode connectivity" properties of the loss landscape, instead of relying only on flatness (as done by methods like SAM).

BOA consists of a line search part, where boundaries of the loss contour are reached, and a reflecion part, where the new search direction is selected using a physics-inspired rule based on the local gradients. Moreover, BOA uses the Krylov subspace to select the initial direction and constrain the search trajectory. This choice is motivated by the observation that the test gradients seem to be aligned with the Krylov subspace generated from the training gradients.

Empirical evaluation using the DomainBed benchmark with VLCS, PACS, OfficeHome, TerraIncognita and DomainNet shows that the proposed BOA method consistently outperforms the evaluated DG baselines. The paper also provides interesting theoretical analyses showing the benefits of approximating gradients using Krylov subspaces compared to random subspaces.

### Strengths
- This paper presents an interesting and novel optimization approach towards domain generalization introducing several new (for DG) concepts, like mode connectivity, the use of a new search approach and the use of Krylov subspaces
 - The paper also offers theoretical and empirical insights on the properties of the Krylov subspace, especially regarding its alignment with the test gradients.
 - The results presented consistently outperform the compared DomainBed baselines and sharpness-aware methods
 - Although no detailed (theoretical or empirical) analysis of the computational complexity of the proposed method is provided, the method appears to be efficient since it avoids the computation of the Hessian.

### Weaknesses
- The reason / underlying mechanisms behind the observed improvements remain unclear. The paper does not convincingly explain why mode connectivity or Krylov alignment lead to better cross-domain generalization.
 - The experiments are constrained to VPT with ViTs. Although this choice may indeed lead to the best results, the paper would be much more convincing if similar findings were observed e.g., for ResNet backbones and/or with full fine-tuning. Even if performance is degraded, demonstrating consistent improvements over comparable baselines would better support the stated claims. In addition, given the increased dimensionality of the parameter space, the proposed method should benefit even more compared to the baselines in this case.

### Questions
- How sensitive is the method to the hyperparameters (e.g., K, reflection count)?
- What is the computational cost / runtime compared to SAM or GSAM?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces BOA (Billiard Optimization Algorithm) for domain generalization. The idea is to stay inside a low-loss region of the training loss, move to the loss boundary with a line search, then reflect and keep going. To avoid getting lost in high dimensions, the method searches only inside a Krylov subspace built from training gradients/HVPs. The authors also show that test gradients tend to align with this subspace, and that a train-computed path often also looks good on the test landscape. On DomainBed, BOA beats ERM/SAM and several DG methods; for example, on VLCS with a ViT-B/16 model, BOA improved accuracy by 3.6 percentage points compared to SAM.

### Strengths
- Clear formulation and intuition. BOA’s use of a training-loss sublevel set, a concrete line-search to the boundary, and a reflection update is straightforward and well motivated.

- Low-dimensional search. Constraining the trajectory to a Krylov subspace is a sensible way to capture salient curvature directions without exploring the full parameter space.

- Useful visual evidence. Overlaying the same trajectory on train and test landscapes helps illustrate why a train-computed path can still navigate good regions on test.

- Strong empirical results. Consistent improvements on DomainBed with ViT backbones (including VPT) and a notable margin over SAM on VLCS.

### Weaknesses
- Limited backbones: Experiments are mostly on ViT. Results on CNNs (e.g., ResNet) or other architectures would strengthen generality.

- Unclear compute cost: Please compare total elapsed (wall-clock) time, memory, and counts of line-search/HVP calls to ERM/SAM under the same settings.

- Heuristic initial direction: The current choice is simple; more analysis or alternatives would improve justification.

### Questions
Is it acceptable not to include CNN backbones, or can you provide BOA vs ERM/SAM results on a standard CNN such as ResNet-50 under the same budget with brief notes on hyperparameter transferability and compute cost?

### Soundness
3

### Presentation
2

### Contribution
3

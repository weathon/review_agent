# Counterfactual Visual Explanation via Causally-Guided Adversarial Steering

- Avg Score: 2.80
- Decision: Reject
- Scores: 2, 2, 4, 2, 4

## Abstract
Recent work on counterfactual visual explanations has contributed to making artificial intelligence models more explainable by providing visual perturbations to flip the prediction. However, these approaches neglect the causal relationships and the spurious correlations behind the image generation process, which often leads to unintended alterations in the counterfactual images and renders the explanations with limited quality. To address this challenge, we introduce a novel framework CECAS, which leverages a causally-guided adversarial method to generate counterfactual explanations. It innovatively integrates a causal perspective to avoid unwanted perturbations on spurious factors in the counterfactuals. Extensive experiments demonstrate that our method outperforms existing state-of-the-art approaches across multiple benchmark datasets and ultimately achieves a balanced trade-off among various aspects of validity, sparsity, proximity, and realism.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes CECAS, a framework for generating causally guided counterfactual visual explanations. The method combines (1) a PGD-based adversarial perturbation guided by a causal penalty that aims to modify only causal (not spurious) features, and (2) a post-refinement stage that uses thresholding and diffusion-based inpainting to improve image quality. The authors claim this approach achieves better balance between validity, sparsity, proximity, and realism. Experiments on CelebA, CelebA-HQ, and several ImageNet subsets show improved quantitative metrics compared to prior diffusion-based counterfactual methods (ACE, DiME). They also evaluate realism with a vision-language model (GPT-4o) acting as a human-like judge.

### Strengths
- Ambitious goal of integrating causal reasoning with counterfactual visual explanation.
- Combines diffusion models with adversarial perturbations, which is technically non-trivial.
- Proposes a post-hoc refinement step that masks only important regions for modification.
- Includes experiments on multiple datasets with qualitative and quantitative results

### Weaknesses
1. **Weak motivation.** The central premise—that counterfactuals should change only causal, not spurious, features—is not convincingly justified. In explainability, highlighting spurious dependencies is *often desirable* to audit models. The paper gives no scenario where suppressing spurious cues yields better insight.  
2. **Unclear and unsound causal mechanism.** The role of the auxiliary classifier \(g\) and the VAE-based decomposition \(R\!\to\!(C,S)\) is vague. The method trains \(g\) on “spurious” features to predict \(Y\), then penalizes the change in its logits, but this contradicts the notion of “spurious.” The causal factorization is neither theoretically grounded nor empirically validated.  
3. **Missing ablations.** The architecture is complex but not decomposed experimentally. There is no evidence separating contributions from (a) the causal projection, (b) diffusion denoising, and (c) post-refinement. Qualitative examples suggest most gains come from the simple mask + inpainting step.  
4. **Weak evaluation.**  
   - Only a few datasets and two baselines are compared on all metrics.  
   - Standard deviations are omitted (“negligible”), preventing assessment of significance.  
   - No metrics measure *causal faithfulness* despite the causal framing.  
   - Reliance on GPT-4o scoring lacks validation against human judgments.  
5. **Poor figure and writing quality.** Figure 2 is overloaded and incomprehensible; several notations (orthogonalization, R′, g) are undefined in text.  
6. **Omission of relevant literature.** Prior work on causal generative models and causal counterfactual generation (e.g., causal diffusion, SCM-based image editing) is ignored, overstating novelty.

### Questions
1. Why should we disregard spurious features in counterfactual explanations, given that exposing reliance on them is a key purpose of XAI? Provide a concrete use case where your approach yields more valuable insight.  
2. Precisely how is the VAE trained to obtain \(C\) and \(S\)? What losses, architectures, and supervision are used? How do you verify that the split corresponds to causal versus spurious factors?  
3. What is the exact role of the classifier $g$?
4. Provide ablations isolating each component: causal loss only, post-refinement only, full method. How much improvement stems from the simple inpainting step?  
5. Discuss or cite causal counterfactual generation literature and clarify how your method differs.

### Soundness
1

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
3

### Summary
This paper proposes a new framework for producing images for counterfactual explanation. This framework, CECAS, integrates causal reasoning with adversarial perturbation methods. The primary contribution lies in addressing spurious correlations during counterfactual generation by constraining modifications to causally relevant features while preserving spurious factors unchanged. The approach combines adversarial attacks (PGD), diffusion models (DDPM), and a causal disentanglement mechanism to produce counterfactuals that achieve high validity, sparsity, proximity, and realism across multiple benchmark datasets.

### Strengths
- Strong motivation: The paper clearly articulates a significant weakness in existing visual counterfactual methods: they often latch onto spurious correlations (e.g., beards and age) rather than causal features, leading to unrealistic and unhelpful explainations.
- Introducing a causal perspective to guide the adversarial generation process is a novel and meaningful contribution. It directly tackles the reasoning behind poor counterfactuals.
- The experimental design tests the method across four distinct datasets with varying attributes (CelebA, CelebA-HQ, BDD-OIA, and ImageNet) and uses a robust set of quantitative metrics. So it's thorough.
- The authors incorporate a novel, human-like evaluation using Vision-Language Models. This adds a valuable layer of semantic validation to their claims.
- Convincing qualitative results: The visual examples demonstrate that CECAS avoids the unrealistic artifacts produced by other methods, such as the spurious beard growth on female faces during an age transition.
- The parameter study on the causal constraint weight (`α`) effectively demonstrates its impact on the trade-off between flip rate and causal faithfulness, reinforcing the method's core claims.

### Weaknesses
- Causal assumption is invalid (or at the least, unvalidated). The paper assumes the independence of causal (`C`) and spurious (`S`) factors (`C⊥S`)  but never validates it. This v-structure in Figure 1 is a strong assumption.
- The VAE-based disentanglement is underspecified, which is a major weakness . The paper provides no details on the VAE architecture, its training, or any quantitative or qualitative evidence that the disentanglement was successful. Without this, there is no proof that the `Ls` loss is actually constraining spurious factors; it's just constraining an arbitrary, unvalidated latent subspace.
- The baselines are narrowly focused on other recent generative methods (STEEX, DiVE, DiME, ACE). A comparison against a wider range of non-diffusion CE methods would have strengthened the paper.
- The VLM evaluation is presented without any scientific rigor. There is no discussion of prompt bias, nor reliability (e.g., running the same prompt multiple times), and no validation against human-in-the-loop judgments.
- In the mask generation equation, the paper divides the pixel difference by M, which it defines as the maximum value after the summation. Is M calculated once per image, or is it a constant for all images, or calculated based on a local window? M is ambiguous, and hence not reasonably reproducible.
- In equation 9, if `m=0`, $ [(1-m)x_t + mx_t^0] $ becomes $ x_t $. If `m=1`, it becomes $  x_t^0 $. This tells the model to keep the counterfactual in the unchanged areas and the original image in the changed areas? This seems backwards (or I could be misunderstanding it).
-  In Table 2, the metrics contradict the claims of the paper. Correlation Difference (CD) score for CECAS is 5.33, worse than the baseline DiME (4.00). The authors don't explain satisfactorily why CD should be discounted.

### Questions
Restating or repeating some of the weaknesses:
1.  In Table 2, the metrics contradict the claims of the paper. Correlation Difference (CD) score for CECAS is 5.33, worse than the baseline DiME (4.00). The authors don't explain satisfactorily why CD should be discounted. Could you elaborate on why your causally guided method performed worse on this specific metric?
2. Can you expand on the inpainting logic? it's possible that the $ x_t $ and $  x_t^0 $ are flipped in the equation.
3. The paper's central claim rests on the VAE's ability to disentangle C and S. Are there any quantitative or qualitative evidence that this disentanglement was successful and how it was validated?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces a method for counterfactual image generation called CECAS. The core idea is to take a causal perspective on the image generation process to avoid spurious features, and instead focus only on the causally relevant ones. The paper's experiments show their method achieves better scores on their metrics of counterfactual flip rate and other plausibility metrics.

### Strengths
The paper addresses an interesting problem of causality in images, and appears to avoid some unnecessary changes in the images based on the figures.

The experimental evaluation is heavily in their method's favor, suggesting it is a worthwhile contribution.

An extra qualitative evaluation adds weight to the actual image quality.

### Weaknesses
In general, my main critique of the paper is that it appears to be quite incremental in its contributions. There are quite a few papers addressing the issue of counterfactual image generation, and the contribution of this one is not overly clear to me.

For example, in the paper's claimed contributions they state "We provide a new perspective on the problem of counterfactual visual explanation by highlighting the critical role of causality". But please see e.g. [1], this just isn't true at all, plenty of work has highlighted this.

Most of the evaluation only compares against a single method ACE, which is quite weak.

I also wonder of the validity of these UNet-based CF methods (e.g. [2]) in light of current image generation tools such as Nano Banana, which can generate far superior image edits to these approaches. I understand of course that adapting such an approach to this would be non-trivial given the whole setup, but far more interesting in my opinion (and likely far more useful).

I am quite wary of metrics such as FID, I would much rather the authors evaluate the usefulness of the method compared to baselines in downstream tasks. That would be better and more convincing.

***

[1] Melistas, T., Spyrou, N., Gkouti, N., Sanchez, P., Vlontzos, A., Panagakis, Y., Papanastasiou, G. and Tsaftaris, S.A., 2024. Benchmarking counterfactual image generation. Advances in Neural Information Processing Systems, 37, pp.133207-133230.

[2] He, Z., Zuo, W., Kan, M., Shan, S. and Chen, X., 2019. Attgan: Facial attribute editing by only changing what you want. IEEE transactions on image processing, 28(11), pp.5464-5478.

### Questions
Can you please explain how this makes a significant contribution beyond the myriad of CF image generation methods? You seem to claim you are the first to consider the separation of causal v. non-causal features, and that is the core contribution. This seems a bit of a stretch to me given work such as (but hardly limited to) these [1, 2].

Do you have any indication your method would be useful for tasks such as model improvement, HCI collaboration, teaching humans concepts learned by the model etc...

***

[1] Kocaoglu, M., Snyder, C., Dimakis, A.G. and Vishwanath, S., 2018, February. CausalGAN: Learning Causal Implicit Generative Models with Adversarial Training. In International Conference on Learning Representations.

[2] Pawlowski, N., Coelho de Castro, D. and Glocker, B., 2020. Deep structural causal models for tractable counterfactual inference. Advances in neural information processing systems, 33, pp.857-869.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper considers the problem of counterfactual explanation (CE) generation for classifiers in computer vision. The proposed approach adapts the ACE method to additionally limit the editing of spuriously correlated factors. This is achieved by adding a variational autoencoder (VAE) trained on the representations of the explained model, which are later fed into an auxiliary classifier trained to mimic the behavior of the original one. The resulting misalignment is used to regularize the optimization procedure. Empirically, the method is evaluated on a broad range of datasets and problems.

### Strengths
S1. The proposed approach is evaluated on a representative set of datasets with metrics covering the entire evaluation spectrum present in the current literature on CEs.

### Weaknesses
W1. The proposed methodology is flawed at its core. The authors propose generating CEs with an additional constraint on the spurious correlations learned by the explained model that limits their appearance in the explanations. However, the primary role of CEs is to actually reveal these correlations, since they highlight the model's unreasonable failure cases, which is desirable.

W2. The paper claims to be outperforming *recent state-of-the-art* methods for CE generation, while ignoring the actual recent papers on this topic that have appeared in top conferences in the last two years. For example, the paper is missing references to OCTET [1], DiG-IN [2] and RCSB [3]. These baselines are crucial, as they consider similar techniques for constraining the changes in the CEs. Moreover, the paper properly cites DVCE [4], which is a work concurrent to DiME and ACE, but fails to compare with it in any way.

W3. The novelty of the method is limited and the authors do not properly credit the ACE paper [5]. A quick glance at the paper's implementation shows that it is entirely based on ACE's codebase. The method is presented in a way that frames both phases (adversarial-attack-based and refinement) as contributions of the paper, while both are entirely taken from ACE. The only remaining novelty is the addition of the factor-disentangling VAE and the auxiliary classifier, a contribution that is logically flawed (W1).

W4. The paper does not provide any further details on the training procedure for the VAE and the auxiliary classifier, which seem to be the primary novelty claimed by the authors. Does the VAE require causal and spurious concept annotations? If yes, is the method limited exclusively to datasets that provide them? If not, how does its accuracy influence the effectiveness of CECAS? What is the error of the auxiliary classifier? These are just a few sample questions that are never addressed in this work.

W5. As the authors correctly point out, adversarial attacks (AA) result in imperceptible changes that influence the model's decision. Incorporating AA techniques into CE generation risks obtaining a modified image where the semantic edits are not actually responsible for the decision change, but rather the small AA-based pixel changes. The same can be said about the ACE method.

W6. Many less significant errors, some of which are pointed out below, show that the work is currently unpolished and requires crucial refinement:

1.  line 058: The actual problem with standard diffusion-based models is that they do not operate in semantically meaningful latent spaces, but rather in high-dimensional pixel spaces.
2.  "Two-colored nodes" are nowhere to be seen in Figure 1.
3.  Table 2 incorrectly bolds and emphasizes the results. It only considers the authors' method and not others.
4.  The Figure 4 caption incorrectly references the line plot of FR vs COUT. Why do the results of ACE and the proposed CECAS not differ perceptually? Is it because these methods are essentially the same on ImageNet (W3)?
5.  The $\alpha$ variable is introduced in an ambiguous way. It first appears as the step size for the PGD attack (Eq. 5), then is used as the regularization strength for the second term in Eq. 6. Moreover, figures like Figure 4 once again highlight its flawed influence - it restricts the range of edits to deviate from the spuriously correlated ones.
6.  lines 051-053: This example is inherently wrong, since the shape of a traffic sign is not a spurious factor, but rather a property used to characterize types of signs. Stating that either content or shape is *causally relevant to the model’s decision* is rather far-fetched, as such a statement requires access to the ground-truth of the model's explanation, which is almost always unavailable.

[1] Zemni et al., OCTET: Object-aware Counterfactual Explanations, CVPR, 2023

[2] Augustin et al., DiG-IN: Diffusion Guidance for Investigating Networks - Uncovering Classifier Differences Neuron Visualisations and Visual Counterfactual Explanations, CVPR, 2024

[3] Sobieski et al., Rethinking Visual Counterfactual Explanations Through Region Constraint, ICLR, 2025

[4] Augustin et al., Diffusion Visual Counterfactual Explanations, NeurIPS, 2022

[5] Jeanneret et al., Adversarial Counterfactual Visual Explanations, CVPR, 2023

### Questions
Please refer to the weaknesses above.

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes CECAS (Counterfactual Explanation via Causally Adversarial Steering), a framework for generating counterfactual visual explanations. The method combines an adversarial perturbation stage with a causal regularization term​, which aims to preserve spurious (non-causal) features while modifying causal ones, using a VAE-based disentanglement of causal and spurious representations. A second stage refines the resulting images using ℓ₁-based masks and diffusion-based inpainting to improve realism and sparsity. Experiments on CelebA, CelebA-HQ, BDD-OIA, and ImageNet subsets show that CECAS achieves higher flip rates and better perceptual quality metrics (FID, LPIPS) than existing counterfactual explanation methods such as DiVE, STEEX, DiME, and ACE.

### Strengths
- The paper tackles an important problem in counterfactual visual explanation by aiming to make generated examples causally faithful and semantically meaningful.

- The experimental evaluation is extensive and covers multiple datasets, metrics, and strong baselines, which supports the empirical effectiveness of the results.

- The paper is clearly written and easy to follow, with well-structured sections and illustrative figures that make the proposed method understandable.

### Weaknesses
My main concern with this paper is that the proposed method is overly complex and lacks conceptual elegance. The pipeline combines several loosely connected components: a PGD-based adversarial attack, a VAE for causal–spurious disentanglement, a mask extraction step, and diffusion-based inpainting. Each of them introduces its own set of hyperparameters and loss terms. While these parts collectively improve visual quality, the overall design feels more like a layered engineering solution than a coherent causal formulation. The causal guidance is implemented through an additional regularization term rather than a principled causal mechanism, and the diffusion refinement serves mainly as a post-hoc fix to clean artifacts instead of addressing their root cause within the generation process. As a result, the framework appears heavy, ad-hoc, and difficult to interpret or reproduce, which weakens the overall conceptual clarity and novelty of the work.
- The causal guidance mechanism is largely heuristic and depends on a VAE-based disentanglement that is not guaranteed to separate true causal and spurious factors.
- The method still operates in pixel space, so adversarial perturbations may flip predictions through imperceptible or non-semantic changes rather than meaningful causal edits.
- There is limited evidence that the generated counterfactuals truly preserve spurious features or modify causal ones, as no causal validation or ablation analysis is provided.
- The diffusion refinement improves realism but can also re-generate image regions, which weakens the interpretability of the counterfactual as a minimal intervention on the same instance.
- The GPT-4o–based vision-language evaluation is interesting but may introduce bias and lacks reproducibility or robustness analysis.
- Some qualitative figures are cluttered with red and green circles that distract from the visual comparison and are not essential to interpret the results.
- In Figure 5, the counterfactual edits for gender conversion appear too subtle to convincingly demonstrate a change in the intended attribute. The “Male to Female” counterfactuals still largely resemble male faces, and the “Female to Male” results also retain feminine features. This suggests that the method may struggle to produce sufficiently strong or semantically distinct attribute changes for certain categories.
- The quality and reliability of the generated counterfactuals heavily depend on the classifier used to guide them. If the classifier itself relies on spurious correlations or non-causal features, the resulting counterfactuals will likely reflect and reinforce those same biases rather than reveal true causal changes. This raises concerns about whether the method can produce meaningful explanations when the underlying model is imperfect or biased.
- The overall pipeline involves multiple ad-hoc steps, including PGD-based adversarial attack, VAE-based disentanglement, mask computation, and diffusion-based inpainting, which makes the approach complex and less principled.
- The diffusion-based refinement serves mainly as a post-hoc fix to remove artifacts, rather than addressing the underlying cause of imperfect or unrealistic counterfactual generation within the main optimization process.

### Questions
- How well does the proposed causal–spurious disentanglement generalize across datasets, and how sensitive are the results to the choice of the VAE architecture or latent dimensionality?
- Have you quantitatively verified that the causal loss truly preserves spurious features while altering causal ones?
- Since the method depends on the classifier’s gradients, how robust are the counterfactuals if the classifier itself relies on spurious correlations?
- Can the authors provide evidence that the generated counterfactuals remain perceptually and semantically distinct when evaluated by an independent or robust classifier?
- How consistent are the GPT-4o evaluation results across different prompt formulations or repeated runs?
- Could the causal guidance be implemented more elegantly, for example, directly in latent space or via a unified generative causal model, rather than through multiple post-hoc components?
- Does the causal guidance term risk over-constraining the perturbation, leading to weaker or insufficient edits as seen in the Male↔Female examples?

### Soundness
2

### Presentation
3

### Contribution
2

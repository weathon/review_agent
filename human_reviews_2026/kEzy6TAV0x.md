# Causal-Adapter: Taming Text-to-Image Diffusion for Faithful Counterfactual Generation

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 4, 8, 4

## Abstract
We present Causal-Adapter, a modular framework that adapts frozen text-to-image diffusion backbones for counterfactual image generation. Our method enables causal interventions on target attributes, consistently propagating their effects to causal dependents without altering the core identity of the image. In contrast to prior approaches that rely on prompt engineering without explicit causal structure, Causal-Adapter leverages structural causal modeling augmented with two attribute regularization strategies: prompt-aligned injection, which aligns causal attributes with textual embeddings for precise semantic control, and a conditioned token contrastive loss to disentangle attribute factors and reduce spurious correlations. Causal-Adapter achieves state-of-the-art performance on both synthetic and real-world datasets, with up to 91\% MAE reduction on Pendulum for accurate attribute control and 87\% FID reduction on ADNI for high-fidelity MRI image generation. These results show that our approach enables robust, generalizable counterfactual editing with faithful attribute modification and strong identity preservation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes to do counterfactual image editing task using a known/assumed causal graph and observed semantic attributes.
Given the causal graph and observed semantic attributes, the paper proposes to train a causal adapter that learns the causal mechanisms between attributes (though not the causal mechanism between attributes and image).
Key challenges with current T2I is *continuous* attribute control and attribute entanglement.
Because attributes can be entangled in text-to-image models, the paper proposes two techniques (PAI and CTC) to mitigate this (spurious/improper) attribute entanglement.
The paper then provides several experiments on pendulum toy dataset, CelebA and brain scan images.

### Strengths
- The paper proposes a simple adapter to enable causal attribute edits given a causal graph and observed semantic attributes.
- The paper proposes two regularizations to encourage better disentanglement of semantic attribute injection. 
- The paper gives some evidence of the qualitative benefits of their approach over certain baselines.

### Weaknesses
- Assumes known causal graph and observed causal attributes. This is a significant limitation that should not be brushed off. The authors even argue that prior causality based methods for learning representations (e.g., CausalVAE) are not good. While agreed that it is a very difficult problem, this should have a more nuanced discussion and the limitation should be acknowledged as a causal graph for all semantic attributes is not always known and neither are causal semantic attributes always observed.

- Experiments seem limited in several cases either because of different setup of baselines or non-diffusion baselines:

  - The experiments are related to CausalVAE but (at least from my understanding) CausalVAE does not assume knowledge of the graph or observed semantic attributes. Thus, I'm not sure it is a fair comparison. Do DisDiffAE and CausalDiffAE assume access to the causal graph and observed semantic attributes?

  - The experiments on CelebA only compare to VAE, HVAE and GAN. No diffusion baselines are included, which is concerning as diffusion baselines would be more comparable. VAE-based and GAN-based methods have not be the standard for a long time. Why not include the baselines you mentioned in Figure 2 that are diffusion based?

- The paper has multiple small components and modifications that make it difficult to take away generalizable insight. There does not seem to be a unifying theme per se but more like a combination of different architecture and training ideas to improve performance via engineering. This reduces the overall impact of the paper since there are not 1-2 major takeaways that could be generalized to new scenarios. 

- The abduction step assumes that the the frozen diffusion model correctly recovers the "causal" exogenous noise. This implicitly assumes that the diffusion model is the "correct" causal mechanism that maps from semantic attributes to an image (or at least counterfactually equivalent mechanism, see bijective causal models, https://arxiv.org/abs/2302.02228). Yet, this may not be the case---i.e., there may be a true mapping between causal attributes and image that is different than the frozen diffusion model. This should be carefully noted and explained.

- (Minor) Method requires DDIM inversion for editing.

### Questions
See weaknesses above for several questions.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
In this work, the authors target the problem of generating more sound visual counterfactuals by integrating causal knowledge into a frozen text-to-image generation model with diffusion backbones.
The contribution is twofold. In the first part, they propose a prompt-aligned injection, which allows the causal attributions to align with text embeddings. The second part aims to prevent spurious correlations by introducing an additional loss to disentangle attribute factors.
They tested the proposed method on both synthetic and real-world datasets, such as human faces and MRI images.

### Strengths
1. The presentation of this work is generally very good, except that most of the figures contain very small text that is not readable in printed papers.
2. I appreciate the detailed illustration of the comparison with previous literature in Figure 2, though the text is indeed too small to read. This strength also extends to the main text, where the authors describe previous studies in detail and make clear comparisons with their own work.
3. The experiments are quite thorough, involving three datasets and visualizations of the results.
4. The evaluation is comprehensive, covering four types of metrics.

### Weaknesses
1. The steps of training and inference for the causal adapter are a bit hard to follow.
2. It seems that this framework requires a pre-defined causal graph as well as annotations of the attributes in the causal graph, which is quite a constraint. If that is the case, it should be discussed more thoroughly in the paper.
3. The motivation of the work appears to encourage the correlation of attributes when generating counterfactuals (as illustrated in Fig. 1, where the proposed model seems to perform better at changing correlated attributes. that is, editing more attributes than non-causal models). At the same time, a loss is introduced to prevent spurious correlations. From the machine’s point of view, correlation is correlation; how would the model distinguish between spurious and acceptable correlations?
4. It also seems that the results on CelebA are based on a single seed, without standard deviations from multiple runs. This makes the results somewhat difficult to evaluate, especially for the Minimality part.

### Questions
1. In Figure 2, do the first and second groups (VAE or GAN, and diffusion SCM) also include conditional diffusion models without a causal graph (or, say, only with attributes but not the causal relationships between attributes)? Because in the text, the works you listed are not all with causal graphs. If that is the case, it would be better to specify at the beginning of this paragraph that the SCM could downgrade to conditional generation with only one attribute, without a causal group, when there are no parent nodes.
2. In Tables 1 and 2, is the first row the target semantic attributes Y for the training of the causal adapter, and is the second row (e.g., do(p)) the inference goal?
3. In Figure 3, it seems that the generations from your model are a bit blurry. Why is that?

### Soundness
3

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
3

### Summary
This paper proposes a lightweight, plug-in module that adapts a frozen text-to-image diffusion backbone for faithful counterfactual image generation. The core contributions are a PAI mechanism that maps each causal attribute to a learnable prompt token so cross-attention can align semantics with spatial features, and a CTC objective that disentangles attribute tokens to reduce leakage and spurious correlations. Compared with prompt-only editing, this adaptive, causally informed design yields strong, generalizable control across synthetic and real domains with large gains in effectiveness and fidelity. The paper also notes evaluation caveats in OOD cases.

### Strengths
- The paper is sharply motivated by a clear diagnosis of text-to-image editing limits, established via a targeted motivational study.

- It introduces a lightweight, plug-in causal-adapter that steers a frozen diffusion backbone with explicit causal semantics, offering a practical, modular path to faithful counterfactual generation without retraining the base model.

- The two core techniques are well-motivated and empirically shown to align attribute semantics, disentangle tokens, and curb spurious correlations while preserving identity.

- The experimental section is broad and convincing.

### Weaknesses
- The work provides no formal theoretical guarantees. 

- Its evaluation hinges on intervention classifiers and can break in OOD regimes, meaning reported effectiveness can be confounded by classifier limitations rather than true causal faithfulness.

- Code is promised only upon acceptance.

### Questions
- The evaluation relies on intervention classifiers that can mislabel valid counterfactuals. What robustness checks did you run to deconfound this?

Overall, this is a good paper. I will not argue if this paper is accepted.

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
4

### Summary
The paper presents Causal-Adapter, a framework for generating counterfactual images. It uses pretrained text-to-image diffusion models for causal interventions based on the Structural Causal Model (SCM). The motivation is that standard text-to-image editing methods lack explicit causal structure, leading to inconsistent edits and failure to handle attribute entanglement. The main contribution is claimed to be a lightweight, pluggable adapter that injects causal semantics into the diffusion backbone without full retraining. This is supported by two regularization strategies: Prompt-Aligned Injection (PAI) for semantic alignment and a Conditioned Token Contrastive (CTC) Loss to disentangle attribute representations. The framework was evaluated on synthetic (Pendulum), CelebA, and ADNI datasets, using metrics such as MAE/F1, FID, LPIPS, and the minimality of CLD.

### Strengths
The paper's claimed contributions are a study demonstrating the inadequacy of prompt-tuning for causal tasks, the Causal-Adapter framework itself, and the PAI/CTC regularization methods. The primary impact stems from its efficiency and modularity, allowing a frozen foundation model to be adapted for specialized, causally aware tasks (e.g., medical imaging) without computationally expensive retraining. The evaluation spans three different domains (physics, human faces, and medical imaging) and includes an ablation study that validates the necessity of the PAI and CTC components. The writing is well-structured and clear.

### Weaknesses
The framework's primary limitation is the lack of novelty in the method and its assumption of a known, correctly specified causal graph; it was not evaluated under conditions of a misspecified or unknown graph. 
A significant flaw exists in the evaluation protocol: the "Effectiveness" classifiers, trained on biased data, penalize the model for successfully generating valid, correlation-breaking counterfactuals (e.g., a "bearded female" is misclassified). 
The "Minimality" metric (CLD) is an indirect proxy that does not directly verify the invariance of non-descendant attributes. 
The paper also lacks a direct comparison to alternative paradigms for causal control, such as methods that use semantic-level guidance during the sampling process.

### Questions
Was there any reason you did not compare or cite the following works:

Chao, P., Blöbaum, P., Patel, S. and Kasiviswanathan, S.P., 2023. Modeling causal mechanisms with diffusion models for interventional and counterfactual queries. arXiv preprint arXiv:2302.00860.

Lyu, M., Yang, Y., Hong, H., Chen, H., Jin, X., He, Y., Xue, H., Han, J. and Ding, G., 2024. One-dimensional adapter to rule them all: Concepts diffusion models and erasing applications. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 7559-7568).

Yeganeh, Y., Farshad, A., Charisiadis, I., Hasny, M., Hartenberger, M., Ommer, B., Navab, N. and Adeli, E., 2025. Latent Drifting in Diffusion Models for Counterfactual Medical Image Synthesis. In Proceedings of the Computer Vision and Pattern Recognition Conference (pp. 7685-7695).

### Soundness
2

### Presentation
3

### Contribution
2

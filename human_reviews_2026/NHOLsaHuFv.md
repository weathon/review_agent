# Label-Free Mitigation of Spurious Correlations in VLMs using Sparse Autoencoders

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 2

## Abstract
Vision-Language Models (VLMs) have demonstrated impressive zero-shot capabilities across a wide range of tasks and domains. However, their performance is often compromised by learned spurious correlations, which can adversely affect downstream applications. Existing mitigation strategies typically depend on additional data, model retraining, labeled features or classes, domain-specific expertise, or external language models posing scalability and generalization challenges. In contrast, we introduce a fully interpretable, zero-shot method that requires no auxiliary data or external supervision named DIAL (Disentangle, Identify, And Label-free removal). Our approach begins by filtering the representations that might be disproportionately influenced by spurious features, using distributional analysis. We then apply a sparse autoencoder to disentangle the representations and identify the feature directions associated with spurious features. To mitigate their impact, we remove the subspace spanned by these spurious directions from the affected representations. 
Additionally, for cases where prior knowledge of spurious features in a dataset is unknown, we introduce DIAL+ which can detect and mitigate the spurious features. We validate our method through extensive experiments on widely used spurious correlation benchmarks. Results show that our approach consistently outperforms or matches existing baselines in terms of overall accuracy and worst-group performance, offering a scalable and interpretable solution to a persistent challenge in VLMs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper presents DIAL, the motivation is to achieve robustness of zero-shot classification of VLM like CLIP. For the disentanglement, the author proposes the use of a spare autoencoder to decompose the attribution in the column space of the decoder. For identification, the author proposes the attribution score such that it can align the feature vector with the spurious concept without using spurious labels. To remove the spurious feature, the author first finds the space spanned by the spurious vector, then projects the visual embedding to this space and removes such components. The author conducts benchmark dataset evaluation on the Group robustness and overall performance. The results show it surpasses the existing SOTA, like TIE or Orth-Cali.

### Strengths
I found this paper quite interesting.

**1** It combines the recent trend of explainability in LLM that use a sparse autoencoder to disentangle the feature vector, and align with a post-hoc explanation to find spurious feature direction. 

**2** The paper is well presented, and the logical flow of the paper is good. The overall soundness of the paper is good for me. 

**3** The author conducted the benchmark evaluation, which is at least comparable with the existing work.

### Weaknesses
1. In Figure 4, I don't quite understand why the average Acc can be even lower than WGA  in FMOW. 

2. To my knowledge, sparse autoencoders (SAE) are often used in the explanation of the transformer encoder's FFN layer. I don't know how the performance would be that migrates the SAE to explain the latent representation. 

3. Correct me if I am wrong. I found the SAE uses pre-trained weights. Would there be any distribution shift to align with the spurious vector in the CLIP models? Why don't we train on the specific dataset?

4. Equation at line 208, I think both parts could be sort of the attribution score. For the first term, it shows how the activation of the feature towards the positive spurious concept. For the second term, it shows how the feature vector aligns with the spurious text prompt. Then my question is, why do we need both terms multiplied together? Have you tried just using a single term in ablation? 

5. In line 239, why do we use the reconstructed embedding to remove the spurious vector, not the original embedding?

### Questions
(1) Related to weakness 3, I would like to see the dataset-specific SAE on the outcomes. 

(2) I don't quite get how you evaluate the explaniblity, as you mentioned this method is fully interpretable.

(3) Figure 2 is a good motivation figure. I would also want to see the heatmap after applying this method.

(4) I am also curious how the alignment between the spurious vector you found based on SAE and the spurious vector found by TIE using the spurious text prompt?

(5) Can the method find the novel spurious feature? Like in ISIC, there are multiple spurious features. Can we apply DIAL to find and mitigate such spurious features?

### Soundness
3

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
5

### Summary
The paper presents a novel, zero-shot method named DIAL to mitigate spurious correlations in Vision-Language Models. DIAL uses the VLM's own zero-shot predictions to create pseudo-labels for identifying samples likely affected by a known spurious attribute and leverages a pre-trained Sparse Autoencoder to decompose VLM embeddings to gain more disentangled, interpretable features that correspond to the spurious attribute.
The authors validate DIAL on five benchmark datasets using multiple VLM backbones. The results show that DIAL consistently improves worst-group accuracy over baselines.

### Strengths
1. **Comprehensive Empirical Evaluation:** The experimental setup is thorough and convincing.
The use of five standard and diverse benchmark datasets, including challenging medical and real-world scenarios, demonstrates the method's broad applicability.

2. **Clarity and Presentation**: The paper is well-written, logically structured, and easy to follow.

### Weaknesses
1. **Dependency on Pre-trained SAEs**: The method's effectiveness is largely affected by the quality of a pre-trained SAE for the given VLM backbone. The paper does not discuss the sensitivity of DIAL to the SAE's quality (e.g., degree of disentanglement, reconstruction error, sparsity level). If a high-quality, pre-trained SAE is not available for a particular VLM, the contribution of DIAL is limited.  Plus, I'm quite curious about which kinds of SAE models[1,2,3] are most suitable for spurious tasks. 

2. **Requirement of a Spurious Concept Description:** While label-free, the method still requires a user to provide a high-level textual description of the spurious attributes (e.g., "Male", "Female"). This assumes that the source of spurious correlation is known, which means that the method cannot discover unknown or hard-to-describe spurious features (e.g., a subtle imaging artifact without a common name). This limitation should be explicitly stated.

3. **Potential for Negative Interference:**  The orthogonal projection forcefully removes any information in the direction of the spurious subspace. If a genuinely causal feature is closely aligned with a spurious one in the embedding space, this process could inadvertently harm model performance by removing useful information. I believe the paper should give a more detailed discussion or analysis of this potential failure mode.

[1] BatchTopK SAE: Bussmann, Bart, Patrick Leask, and Neel Nanda. "Batchtopk sparse autoencoders." arXiv preprint arXiv:2412.06410 (2024).

[2] JumpReLU SAE: Rajamanoharan, Senthooran, et al. "Jumping ahead: Improving reconstruction fidelity with jumprelu sparse autoencoders." arXiv preprint arXiv:2407.14435 (2024).

[3] SAE + Contrastive loss: Wen, Tiansheng, et al. "Beyond matryoshka: Revisiting sparse coding for adaptive representation." arXiv preprint arXiv:2503.01776 (2025).

### Questions
The paper's utilization of SAE with KNN for identifying useful sparse representations for downstream tasks is a technique that has been explored in prior literature. For example, the following papers should be discussed in the related work:

[1] Tian, Zhihua, et al. "Sparse autoencoder as a zero-shot classifier for concept erasing in text-to-image diffusion models." arXiv preprint arXiv:2503.09446 (2025).

[2] Wen, Tiansheng, et al. "Beyond matryoshka: Revisiting sparse coding for adaptive representation." arXiv preprint arXiv:2503.01776 (2025).

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
3

### Summary
VLMs often rely on spurious correlations, which can affect downstream tasks. The authors introduce DIAL, a zero-shot method that finds and mitigates spurious correlations. DIAL operates by (1) filtering representations that might be disproportionately influenced by spurious features, (2) applying a sparse autoencoder to disentangle the representations and identify feature directions associated with spurious feature, and (3) removing the subspace spanned by the spurious directions from the representations. Results across several benchmarks demonstrate the utility of the method.

### Strengths
- This work addresses an important problem - finding and mitigating spurious correlations learned by vision-language models
- Results show performance improvements when compared to several baselines, suggesting utility of the approach. The method also works well across domains (i.e. general domain as well as medical domain).

### Weaknesses
- **Insufficient analysis:** Section 4.4 provides overall metrics across various datasets, but does not provide sufficient fine-grained analysis of results. Ablations are also limited. Ultimately, it is not clear to me *why* the method works better than baselines.
- **Need for attribute labels:** The proposed method requires a set of candidate spurious attributes, which may not always be known ahead of time and might limit utility of the method in real-world settings.
- **Choice of sparse autoencoder:** The authors consider one off-the-shelf pretrained sparse autoencoder for their analyses. How robust are the results to different choices of the autoencoder?

### Questions
Questions are listed above under weaknesses.

### Soundness
3

### Presentation
3

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
This paper proposes DIAL, a label-free and zero-shot method to mitigate spurious correlations in vision-language models (VLMs). The approach uses sparse autoencoders to disentangle image embeddings and identify feature directions associated with spurious attributes. These directions are then removed via orthogonal projection to produce debiased representations. The method is evaluated on several benchmark datasets and compared against existing zero-shot debiasing techniques.

### Strengths
1. The method is fully zero-shot and does not require labeled data, retraining, or external models, which improves scalability.
2. The use of sparse autoencoders for disentangling representations is well-motivated and contributes to interpretability.

### Weaknesses
1. Limited novelty: The core idea—removing spurious directions via projection—is conceptually similar to prior work. The use of sparse autoencoders is incremental and not fundamentally new in the context of representation disentanglement.

2. Low practical impact: The spurious correlation issues addressed (e.g., background bias in Waterbirds, gender bias in CelebA) are well-known and have been extensively studied. The paper does not convincingly demonstrate that these issues remain critical in modern VLMs.

3. Outdated model focus: The analysis centers on older VLMs like CLIP ViT-B. It remains unclear whether the same spurious correlation problems persist in newer models such as SigLIP, OpenCLIP, or multi-modal transformers trained with more diverse data.

4. Assumption-heavy candidate selection: The method relies on pseudo-labels and centroid-based heuristics to identify biased samples, which may be unreliable in real-world settings or for more complex tasks.

5. Lack of generalization evidence: The paper does not explore whether the proposed mitigation transfers across tasks (e.g., retrieval, captioning) or domains beyond the selected benchmarks.

### Questions
Can your method be extended to mitigate spurious correlations in text embeddings or multi-modal fusion layers?

Is there any evidence that your projection-based debiasing improves downstream task performance beyond classification?

### Soundness
2

### Presentation
2

### Contribution
2

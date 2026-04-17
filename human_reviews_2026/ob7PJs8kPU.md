# All Patches Matter, More Patches Better: Enhance AI-Generated Image Detection via Panoptic Patch Learning

- Decision: Accept (Poster)
- Scores: 6, 6, 8, 2

## Abstract
The rapid proliferation of AI-generated images (AIGIs) highlights the pressing demand for generalizable detection methods. In this paper, we establish two key principles for AIGI detection task through systematic analysis:
**(1) All Patches Matter**, since the uniform generation process ensures that each patch inherently contains synthetic artifacts, making every patch a valuable detection source; and
**(2) More Patches Better**, as leveraging distributed artifacts across more patches improves robustness by reducing over-reliance on specific regions.
However, counterfactual analysis uncovers a critical weakness: naively trained detectors display **Few-Patch Bias**, relying disproportionately on *minority patches*.
We identify this bias to **Lazy Learner** effect, where detectors to limited patch artifacts while neglecting distributed cues.
To address this, we propose **Panoptic Patch Learning** framework, which integrates: 
(1) *Randomized Patch Reconstruction*, injecting synthetic cues into randomly selected patches to diversify artifact recognition; 
(2) *Patch-wise Contrastive Learning*, enforcing consistent discriminative capability across patches to ensure their uniform utilization.
Extensive experiments demonstrate that PPL enhances generalization and robustness across datasets.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper identifies a Few-Patch Bias in current AI-generated image detectors and proposes a Panoptic Patch Learning (PPL) framework consisting of (1) Randomized Patch Reconstruction (RPR), reconstructing random real-image patches using Stable Diffusion v1.4 inpainting,  and (2) Patch-wise Contrastive Learning (PCL) applied to ViT patch tokens. The method aims to enforce equal attention to all patches and enhance artifact distribution information. Extensive experiments across several benchmarks show substantial gains.

### Strengths
1. The Few-Patch Bias is demonstrated visually (attention/TDE) and via patch-mask counterfactuals. The fix (RPR+PCL) is straightforward and well-motivated.

2. The combination of RPR and PCL provides measurable improvements over existing works.

3. The experiment is conducted on both toy and in-wild datasets, and demonstrates its SOTA performance.

4. There are rich ablation studies to examine the designs of the proposed method.

### Weaknesses
1. RPR reconstructs patches of real images using Stable Diffusion v1.4 inpainting (empty prompt). This introduces a major domain-specific bias. It can train additional variants of RPR using GAN-based reconstruction and observe the performance changes. This can further examine the generality of PPL framework. Also, since some baselines, like SAFE, train on GAN-generated images and test on all, such an experiment can make the comparison more fair.

2. In Table 9, Midjourney and ADM accuracies with dropout = 0.15 (94.3% and 87.9%, mAcc=94.2%) are much higher than at 0.1, 0.2, or 0.25 (≈70%), and close to the proposed PPL result (mAcc=97.2%). This raises several questions:

(1) Why does such a narrow dropout range produce a sudden performance surge? Is there randomness in patch masking or seed selection? 

(2) If a properly tuned dropout rate (0.15) nearly matches PPL while avoiding diffusion reconstruction and PCL fine-tuning, the claimed advantage of PPL becomes less convincing. 

3. The paper fine-tunes CLIP and DINOv2 with PPL but does not clearly report the results of directly training CLIP and DINOv2.

### Questions
1. In terms of Weakness-1, (1) have you tried using GAN-based reconstruction or other diffusion models within RPR? (2) And how does the performance change when training RPR with different reconstruction backbones?

2. In terms of Weakness-2, please see the questions in the comment.

3. Could you report the results of directly training or evaluating CLIP/DINOv2 on the same datasets to quantify the actual gain from PPL fine-tuning?

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
This paper proposed a novel patch-based AI-generated image detection method. The main motivation is the few-patch bias observed in existing detectors, i.e., they overly rely on a limited proportion of patches and neglect the diversity of artifacts across patches. To encourage the utilization of information from all patches, Randomized Patch Reconstruction (RPR) is proposed, which applies diffusion reconstruction to real images and replaces a random set of the original image patches with the corresponding reconstructed patches. Patch-wise Contrastive Learning (PCL) further encourages the learning and utilization of all patch features. Results on several benchmarks suggest the state-of-the-art generalization performance of the proposed method.

### Strengths
1. The analysis of the few-patch bias of AIGI detectors reveals a significant limitation of existing methods.
2. The proposed RPR and PCL effectively encourage the model to utilize the information across all patches.
3. The generalization of the proposed method is comprehensively evaluated, including testing on challenging datasets like Chameleon and robustness studies.

### Weaknesses
1. The motivation of the *All Patches Matter* principle requires further clarification.
- The two lines of evidence stated in Lines 45-49 only support that "some patches matter" (i.e., some of the patches contain discriminative patterns) rather than "all patches matter".
- The "Theory" in Line 146 may be an inappropriate title for the first key finding, as no theoretical results are provided. In addition, the assumptions behind the statement "Because every patch of a synthetic image is itself generated, each inherently contains artifacts" need further clarification (e.g., what "artifacts" are and why generative models produce them across every pixel).
- The details for the "Experiments" in Lines 154-157 are not specified.
- Given that "a single patch contains sufficient information for reliable discrimination" (Line 157), it seems unnecessary to emphasize the utilization of all patches. This point may not support the *All Patches Matter* principle.
2. It seems that the attention maps in Figure 3(a) can be explained by the observations in [1] that vision transformers tend to utilize patch tokens in low-informative background areas as registers for aggregating global information. Repeating the visualization experiments with vision transformers with dedicated registers proposed in [1] may eliminate this possibility. Besides, the acquisition of the attention maps needs explanation.
4. The experimental details for Figure 3(b) are not specified, including what detectors are tested and how the patch masking is implemented. This is important for supporting the generalization of the conclusion.
5. It seems that the Total Direct Effect (TDE) described in Lines 210-213 should be the Controlled Direct Effect (CDE).
6. Previous reconstruction-based methods such as [2] and [3] should be discussed and compared.


[1] Vision Transformers Need Registers. ICLR 2024.  
[2] Aligned Datasets Improve Detection of Latent Diffusion-Generated Images. ICLR 2025.  
[3] A Bias-Free Training Paradigm for More General AI-generated Image Detection. CVPR 2025.

### Questions
1. How are the reconstructed images (in Figure 2 and Section 4) produced? Is there any image processing process, such as upsampling and downsampling, that can affect the low-level details of the image or introduce artifacts?
2. Why does the proposed method generalize effectively to GAN-generated images, despite the training is solely based on diffusion models?
3. Is it possible to set $p_{rpr}$ to 1, i.e., using only the real images and the RPR images for training?
4. In Figure 8, what contributes to the difference between the blue and red bars at +LoRA, given that RPR is not used?

### Soundness
2

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
This paper tackles the challenge of generalizing AI-generated image (AIGI) detectors across different generation models. Through systematic analysis, the authors identify a key issue — “Few-Patch Bias” — where existing detectors over-rely on a small number of image patches despite artifacts being uniformly distributed across all regions of synthetic images. They propose two guiding principles, *All Patches Matter* and *More Patches Better*, and introduce the **Panoptic Patch Learning (PPL)** framework to operationalize them. PPL combines **Randomized Patch Reconstruction (RPR)**, which injects synthetic artifacts into randomly chosen patches to diversify learning, and **Patch-wise Contrastive Learning (PCL)**, which enforces consistent discriminative capability across patches. Extensive experiments on major benchmarks (GenImage, DRCT-2M, AIGCDetectionBenchmark, and Chameleon) demonstrate that PPL achieves state-of-the-art accuracy and robustness, significantly improving generalization to unseen generators and real-world data.

### Strengths
1. The paper introduces clear and insightful principles (“All Patches Matter” and “More Patches Better”) that reveal a fundamental property of AI-generated images and motivate the proposed framework.
2. The proposed Panoptic Patch Learning method is conceptually simple yet effective, combining randomized patch reconstruction and patch-wise contrastive learning to mitigate few-patch bias.
3. Extensive experiments across diverse benchmarks demonstrate strong generalization and robustness, supported by thorough analyses and clear visual evidence.

### Weaknesses
1. In the second paragraph of the introduction, the term *“patch”* appears for the first time but lacks a clear definition or motivation. Since AIGI detection includes many CNN-based detectors that do not explicitly rely on patch-level representations, introducing the patch concept without clarification may confuse readers about its relevance to this task. It is recommended that the authors explain why they adopt patch as the basic analytical unit and cite related works that have previously used patch-based approaches, which would make the motivation more convincing.
2. In line 364, the reference of SAFE is wrong.
3. In Tables 3 and 4, I notice that all baseline methods are trained on GAN-based datasets, while the proposed PPL is consistently trained on SDv1.4 (a diffusion-based model). Although it is reasonable to fix one generator for evaluating generalization, different training datasets may have biases toward different test distributions (e.g., a model trained on diffusion data may generalize better to diffusion-based test sets). Therefore, I suggest that the authors also report results trained on a GAN-based dataset for these two benchmarks. This would ensure a fair comparison and further demonstrate that the proposed method is also effective when trained on GAN-generated data.

### Questions
I have no further questions.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes two principles for AI-generated image (AIGI) detection: "All Patches Matter"  and "More Patches Better". The authors contend that existing detectors suffer from "Few-Patch Bias," relying excessively on a minimal number of highly discriminative patches. To address this issue, they propose the Panoptic Patch Learning (PPL) framework, comprising Randomized Patch Reconstruction (RPR) and Patch-wise Contrastive Learning (PCL). The method achieves superior cross-model generalization performance.

### Strengths
- The methodology is well-aligned with the motivation: RPR corresponds to the principle of "More Patches Better," while PCL reflects the idea that "All Patches Matter." Both methods are clearly motivated and logically consistent with the proposed principles.
- The experiments are comprehensive: performance is reported on benchmarks such as GenImage, DRCT-2M, and Chameleon, and robustness under various corruptions (e.g., compression and blur) is demonstrated. The study also includes ablation studies and hyperparameter analysis.

### Weaknesses
- The paper presents “All Patches Matter / More Patches Better” as a primary principle, but ideas such as “any local patch of an AI-generated image contains artifacts, and even a single patch can be sufficient for reliable discrimination” have already been explored in prior patch-based detection work discussed in the related work section. This makes the proposed patch-level principles feel incremental rather than conceptually new.
- Similarly, although the proposed PPL framework is shown to be effective, it is essentially an engineering-level refinement rather than a fundamentally new paradigm. The core idea of RPR is to move the DRCT-style diffusion reconstruction from the image level down to the patch level, while PCL is a straightforward application of supervised contrastive learning at the patch-token level.
- The paper relies on TDE for attribution. However, TDE does not appear to be a commonly adopted attribution method in deep learning. Why not use more standard interpretability approaches, such as Grad-CAM and its variants, or LRP (Layer-wise Relevance Propagation) [1], to explain what the detector is actually using?
- The paper focuses primarily on CLIP, DINOv2, and other ViT-style backbones, while largely ignoring CNN-based detectors. This raises an important question: do the core principles proposed in this paper still hold under CNN-based architectures?
- Going further, [2] uses Guided-GradCAM and LRP as attribution methods, extracts transferable forensic features from different layers of a CNN-based detector, and maps them back to the input image patches. The results indicate that color statistics are a key signal for CNN-based forgery detectors, rather than an extreme reliance on a few spatial patches. Different layers highlight different input regions. This behavior is not the same type of bias that the authors call “Few Patch Bias.” Therefore, the bias analysis in this paper may not generalize to CNNs, and it is not yet convincing that the claimed principles are architecture-agnostic.

[1] Transformer Interpretability Beyond Attention Visualization

[2] Discovering Transferable Forensic Features for CNN-generated Images Detection

### Questions
1. In Section 5, the citation of SAFE is incorrect.
2. In Table 6, the row labeled “Infonce/tau=0.5” is not aligned with the notation used in the main text.
3. In Section 3.1, the text uses “donot,” which is a typographical error. It should be “do not.”
4. In Algorithm 1, the classification loss is denoted as ( L_{ce} ) in Step 3, but it appears as ( L_{bce} ) in Step 5 when forming the total loss. 
5. The terminology for the proposed bias is inconsistent. The Introduction (in the italicized part) uses “Few Patch Bias,” while other parts of the paper use “Few-Patch Bias.”

### Soundness
3

### Presentation
3

### Contribution
2

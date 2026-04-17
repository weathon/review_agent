# Improved Object-Centric Diffusion Learning with Registers and Contrastive Alignment

- Decision: Accept (Poster)
- Scores: 4, 6, 8

## Abstract
Slot Attention (SA) with pretrained diffusion models has recently shown promise for object-centric learning (OCL), but suffers from slot entanglement and weak alignment between object slots and image content. We propose Contrastive Object-centric Diffusion Alignment (CODA), a simple extension that (i) employs register slots to absorb residual attention and reduce interference between object slots, and (ii) applies a contrastive alignment loss to explicitly encourage slot–image correspondence. The resulting training objective serves as a tractable surrogate for maximizing mutual information (MI) between slots and inputs, strengthening slot representation quality. On both synthetic (MOVi-C/E) and real-world datasets (VOC, COCO), CODA improves object discovery (e.g., +6.1% FG-ARI on COCO), property prediction, and compositional image generation over strong baselines. Register slots add negligible overhead, keeping CODA efficient and scalable. These results indicate potential applications of CODA as an effective framework for robust OCL in complex, real-world scenes. Code and pretrained models are available at https://github.com/sony/coda.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper presents a method for object-centric diffusion learning that extends Object-Centric Slot Diffusion (LSD) [1] by introducing two additional mechanisms: (i) register slots, designed to act as attention sinks, and (ii) a contrastive alignment loss to reduce slot entanglement. The model fine-tunes key and value projections in the cross-attention layers of a frozen Stable Diffusion backbone, while keeping the rest of the weights fixed.
The claimed novelty of register slots is, however, questionable. The concept was originally proposed in SlotAdapt [2], which introduced a separate learnable slot that serves exactly the same purpose. As also shown in Table 9 of the appendix, the authors acknowledge this similarity. The only difference is that in the current paper, the register slots are frozen rather than learnable, a minor technical modification rather than a conceptual contribution.
The proposed contrastive loss closely follows Learning to Compose [3], where compositional consistency is enforced through a shared slot initialization across samples. This paper adopts the same shared initialization trick, and the resulting contrastive objective is largely an adaptation rather than a new formulation.

[1] Jiang et al., LSD, NeurIPS 2023
[2] Akan & Yemez, SlotAdapt, ICLR 2025
[3] Jung et al., Learning to Compose, ICLR 2024

### Strengths
The method reports state-of-the-art results on both unsupervised object segmentation and compositional generation benchmarks, though the fairness of the compositional generation evaluation is debatable (see Weaknesses first and last point).
- The idea is conceptually simple, it does not introduce any architectural changes. (Although new register tokens can cost in terms of runtime (mostly negligible I guess, as the paper states that it only costs 0.02%) and contrastive loss is actually 2x forward (but in training))
- The experimental setup is solid, with comprehensive evaluations across multiple datasets and informative supplementary analyses in the appendix.

### Weaknesses
- The compositional generation evaluation is not entirely fair. The proposed method (CODA) is explicitly trained to reconstruct images given random slot combinations through its contrastive objective, which involves positive and negative slot pairs. This directly optimizes the model for composition-like reconstruction, so improved performance under this metric is somewhat expected. The qualitative results, however, are convincing and indicate stronger disentanglement than SlotAdapt, likely due to reduced slot entanglement from contrastive alignment.
- The paper’s claim that it "introduces" register slots is inaccurate. SlotAdapt (Akan & Yemez) already proposed a similar concept (see Table 9 in its appendix). While the current implementation with frozen register slots inspired by attention sinks in LLMs can be considered an improved version, it does not constitute a new contribution and should be framed as such.
- The contrastive loss significantly increases training cost, requiring 2× forward passes. 1x forward with 1/2 batch size is not possible because of the gradient stopping for the diffusion model. 
- The ablation results are ambiguous. The combination of register slots and contrastive alignment (Reg+CA) already achieves nearly all the reported gains on COCO and VOC. The additional contrastive loss provides only marginal quantitative improvements despite its high computational cost. My guess is that this component mainly benefits compositional generation, since the contrastive objective explicitly encourages the model to reconstruct plausible images under slot mixing. If that is the case, it would be important to show both quantitative and qualitative results for compositional generation using the Reg+CA model (without CO) to confirm whether the improvement truly stems from better compositionality or merely from training bias toward the contrastive task.

### Questions
- The paper reports diffusion generation at 512×512 resolution, while most prior work (e.g., SlotAdapt, LSD) uses 256×256. How were comparisons conducted under this mismatch? Higher resolution can directly improve image quality metrics such as FID and KID.
- The training iterations are inconsistent across methods. SlotAdapt trains for 200K (MOVi-E), 250K (COCO), and 190K (VOC), while CODA uses 500K for COCO and 250K for VOC. Comparisons would be more meaningful if conducted under the same training budget.
- The paper would benefit from additional qualitative results on compositional editing to more clearly demonstrate improvements in visual plausibility and slot-level control.
- In Figure 7 (Appendix), it is unclear whether the single-slot generations include register slots. That is, whether the output is generated from one semantic slot plus registers or solely from the selected slot.
- On Table 3 results, is MOVi-E position prediction really correct? 0.01 looks like an anomaly.
- Results are reported for single runs, but given the stochastic nature of slot initialization, multi-seed evaluations would provide stronger statistical support.
- Since the proposed method is relatively lightweight, it could be scaled to larger backbones such as SDXL or FLUX variants to test its generality. No prior object-centric work appears to have explored this yet.
- An ablation on the classifier-free guidance (CFG) parameter, similar to that reported in SlotAdapt (Table 10, Appendix), would help analyze the effect of guidance strength.

Minor typo: Fig. 3 caption, "Orginal image" -> "Original image".

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes Contrastive Object-centric Diffusion Alignment (CODA), an object-centric learning (OCL) framework with Slot-Attention and a pre-trained Stable Diffusion (SD) decoder. Prior work SlotAdapt already shows that using a powerful pre-trained SD model can greatly improve the generation quality of the OCL method. CODA further strengthens the results in three ways: 1) register slots (empty slots) to absorb attention on non-object areas to improve slot disentanglement, 2) an easier approach adapting the pre-trained text-conditioned cross-attention layer to condition on slots, and 3) contrastive alignment that maximizes the reconstruction loss when conditioned on unrelated slots. With these techniques, CODA outperforms baselines in both segmentation and generation on real-world datasets.

### Strengths
- In this era of foundation models, I always believe the community of OCL should move to more pre-trained models. This paper is a nice attempt at using DINO and SD models, and demonstrates effectiveness at large-scale real-world datasets.
- All three techniques in CODA are well-motivated and implemented. I appreciate the thorough ablations in the main paper and the Appendix. They help answer the effectiveness of each component very well.
- The analysis of mutual information (MI) is interesting and inspiring.

### Weaknesses
I don't see big weaknesses in the paper. Some minor weaknesses:
1. Each component is not very novel, as they pre-exist in other areas. Though I don't view this as a big issue -- combining them in a nice way and achieve strong results is also a good contribution.
2. Why not comparing with GLASS in the experiments? It seems that the segmentation results are similar to them (by checking tables in their paper), but I believe the generation capability of CODA must be stronger. Please include this in the paper.
3. I don't see enough qualitative generation comparison with SlotAdapt. While the quantitative results of CODA clearly outperform it, it's always beneficial to see some visual results. For example, I would like to see SlotAdapt results in Fig.9 and Fig.10 in the Appendix. Is it because they haven't released their code? There is a `Code` button on their webpage but it doesn't seem to lead to anywhere. Then how do you get SlotAdapt in Fig.7 of Appendix?

### Questions
See weaknesses. Some minor questions not affecting my decision:
1. Have you tried SD 2.1? In my experience, SD 2.1 has a similar model size / arch as SD 1.5, but often performs better in generation, likely due to its v-predicition formulation.
2. What do you think is still missing in achieving better reconstruction of input images from slots? Checking results in Fig.10, the reconstructed object's identity is still clearly not preserved. I understand this is also the case for SlotAdapt so I don't view it as a weakness of the paper.
3. In the qualitative segmentation results in Fig.7, why are there still some meaningless background pixels being segmented to a slot? E.g., column 1, 2 in the train example, they don't seem to correspond to any objects. In theory, they should be absorbed by the register slots?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper addresses the issue of binding of objects to slots in existing slot-based OCL methods. It proposes using register tokens/slots and a contrastive loss to alleviate this issue, and also fine-tunes the query/key embeddings in the diffusion model to better adapt it to the slot representation. The effectiveness of the proposed solution is evaluated on standard datasets in terms of various tasks, including object discovery, property prediction, and compositional generation. Furthermore, detailed ablations demonstrate the benefit of (fixed) slot registers and the contrastive loss. The paper discuss in detail how these components contribute to the performance gain and provides an additional mathematical justification for the contrastive loss.

### Strengths
* (S1) The paper is very well written, clearly explains the shortcomings of existing methods and motivates the proposed solutions. The methodology section is well-explained.
* (S2) Though the ideas in the paper can be seen as a combination of several ideas from existing works (register tokens [1], negative guidance [2], contrastive learning of slots [3]), the paper creatively combine these ideas in the context of slot attention methods. This enhances the technical contributions of the paper.
* (S3) The results for different tasks, from property prediction, object discovery, and compositional generation, clearly show that the method outperforms previous methods by a large margin. The improvement for categorical prediction is particularly impressive. Additionally, the paper provides detailed ablations for several configuration settings, including the number of register tokens and the scale of the CFG guidance weight.
* (S4) The results in Figure 7 are impressive, indicating that the slots actually bind to the objects and that the model is able to reconstruct the objects to a surprising degree with these slots. This highlights that the approach indeed allows to better disentangle the slot representations.

References:
* [1] Darcet et al., Vision Transformers Need Registers, ICLR 2024.
* [2] Karras et al., Guiding a Diffusion Model with a Bad Version of Itself, NeurIPS 2024.
* [3] Manasayan et al., Temporally Consistent Object-Centric Learning by Contrasting Slots, CVPR 2025.

### Weaknesses
* (W1) Fairness of evaluation — A significant issue in the comparison with other methods is the use of DINOv2 and the 512x512 image size for the training method. Other methods, such as SPOT and DINOSAUR, use a DINOv1 model with an image size much smaller than 512x512. Thus, the proposed gains in the method cannot be attributed solely to the slot attention registers and constructive loss.
* (W2) Ablations on the VOC dataset — Though I understand that performing the ablations on VOC is quicker, the VOC dataset is much simpler compared to the COCO dataset (many images consist of only a single object), and to thoroughly gain insights on the effect of the proposed changes, the paper should conduct the ablations on a more complex dataset like COCO (even if the ablations are done over a subset of the COCO dataset).
* (W3) The need for training of cross-attention (CA) layers is not clearly explained by Tab. 5. FG-ARI as a metric has been criticized by several papers (SPOT, GLASS), etc. The primary motivation for training the CA layers appears to be achieving a good reconstruction output rather than improving object discovery metrics. The paper should refer to Table 9 in the appendix to clarify the contribution of training CA layers.

Minor points:
1. Figure 2, lower branch should be $\epsilon_\bar{\theta}$ instead of $\epsilon_\{\theta}$.
2. Figure 2: DINO should be DINOv2.
3. Figure 6 is hard to understand due to the use of different input images with different layers. The figure should be supported with a more illustrative caption.
4. It would be useful if the paper were to provide more qualitative results for compositional generation.
5. In l. 236, the paper claims that $\mathbf{s}$ are DINOv2 features, but these are actually the slots computed from DINOv2 features.
6. Reference [2], see above, is missing.

### Questions
1. Can the authors discuss the effect of the method vs. the number of objects in the scene? For example, if a given image has a 7 objects for 7 slots, does adding register tokens hurt the performance in this case? What happens if the number of object tokens is greater than 7?
2. How did the authors settle on the ‘odd’ number of 77 register tokens? Where does this number come from? Also, does this large number of register tokens affect computational efficiency?
3. Can the authors discuss why the contrastive loss guidance scale needs to set to a really low value for good results?
4. For figure 7, it seems like the slots from StableLSD do not correspond/encode the object at all (we are not able to reconstruct the object from individual slot), however, the joint slot is able to reconstruct the scene fairly well. Why do you think this is case?
5. Is it really the case that fine-tuning the cross-attention mechanism (CA) does not reduce the expressive power of the model (l. 228)? How can we assure that this is indeed the case?

### Soundness
3

### Presentation
4

### Contribution
3

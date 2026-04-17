# ReFocusEraser: Refocusing for Small Object Removal with Robust Context-Shadow Repair

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 4

## Abstract
Existing diffusion-based object removal and inpainting methods often fail to recover the fine structural and textural details of small objects. This is primarily due to the VAE encoder’s downsampling, which inevitably compresses small masked regions and causes significant detail loss, while the decoder’s upsampling alone cannot fully restore the lost fine details.
However, the adverse effects of this fixed compression can be mitigated by enlarging the perspective of these regions.
To this end, we propose ReFocusEraser, a two-stage framework for small object removal that combines camera-adaptive zoom-in inpainting with robust context- and shadow-aware repair. In Stage I, a camera-adaptive refocus mechanism magnifies masked regions, and a LoRA-tuned diffusion model ensures precise semantic alignment for accurate reconstruction. However, reintegrating these magnified inpainted regions into the original image introduces challenges due to VAE asymmetry, such as color shifts and seams. Stage II addresses these issues by fine-tuning an additional decoder to create a seam- and shadow-aware module that eliminates residual artifacts while preserving background consistency. 
Extensive experiments demonstrate that our proposed RefocusEraser achieves state-of-the-art performance, outperforming existing methods across benchmark datasets. 
Related code and data are available at https://github.com/ProAirVerse/ReFocusEraser.git.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes ReFocusEraser, an innovative two-stage framework for small masked object removal with robust context and shadow repair.

### Strengths
1. This paper proposes an innovative two-stage framework for small masked object removal with robust context and shadow repair. By integrating a Camera-Adaptive Refocus Inpainter with a Seam- and Shadow-Aware Repair module, the method effectively enhances the structural and textural restoration of small masked regions.
2. Experimental results demonstrate that the proposed method achieves significant performance improvements on benchmark datasets such as RORD-val and RemovalBench, outperforming existing state-of-the-art methods.

### Weaknesses
1. Regarding the generalization of ReFocusEraser, this method is mainly evaluated on camera-captured data and scenes. It is unclear whether it remains effective on non-camera-captured images.
2. The current framework adopts a two-stage decoupled training strategy. If the two stages are trained jointly in an end-to-end manner, what are the differences in training cost and performance?
3. While training details are reported, the paper does not provide a discussion or comparison of inference efficiency. The overall training time, inference time, and computational complexity compared to mainstream methods remain unclear.

### Questions
Refer to "Weaknesses".

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
This paper proposes diffusion-based framework to handle *object removal*, capable of handling objects of various sizes. To improve the object removal of small objects with securing the consistency between inpainted region and surrounding context, the paper proposes two main components:

1. Camera-adaptive zoom-and-crop module to upscale the appearance of small objects.
2. Additional decoder to mitigate the misalignment between surrounding context and removed region.

### Strengths
* The paper points out the limitation of previous works that struggle to remove the small objects in the image due to the information loss in VAE encoder. The paper addresses aforementioned issue where the motivation is clear.
* The paper proposes *Camera-adaptive Refocus Mechanism* where it crops the target object based on the camera parameters, better preserving the object context. Furthermore, it introduces additional decoder to enhance the inconsistency between removed part and surrounding context such as boundary of the removed object. The framework makes sense with resulting in clear performance improvement compared to the baseline.

### Weaknesses
My main concern is on the efficiency of the model. 

* **Training Efficiency:** In L47-49, the paper points out that fine-tuning the VAE incurs huge computational cost.  However, ReFocusEraser also needs substantial amounts of GPUs which is 8 H200 GPUs to optimize the additional decoder. 
* **Inference/parameter Efficiency:** ReFocusEraser needs two decoder for the inference which raises the concern on the efficiency of inference time and parameter size. Can you compare the inference time of ReFocusEraser with the baselines in Tab. 1? Also, it would be also better to compare the size of parameters as well.
* In Fig. 6b and Fib. 6d, it seems that there are color shift (more brighter) on the removed part compared to the Fig. 6a and Fib. 6c. Due to this color shift, it seems that box-based stitching results in more natural output compared to the mask-based stitching which is inconsistent with the paper's conclusion on L431. 
* In Tab. 2, PSNR of row (c) is higher than row (d). However, the number on row (d) is bolded. Furthermore, not using decoder achieves better results on PSNR, SSIM and LPIPS compared to the setting with using additional decoder while opposed results can be found in FID and CMMD metrics. Can you explain more about this inconsistent results?
* In Tab. 3, it is better to add the results with fixed crop or zoom to validate the effectiveness of camera-adaptive zoom-in method.

### Questions
NA

### Soundness
3

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
2

### Summary
This paper proposes a two-stage framework for small object removal, addressing the performance limitations caused by detail loss in the VAE encoder’s downsampling process. In the first stage, the authors introduce a camera-adaptive refocus mechanism that enlarges masked regions to facilitate coarse object removal. In the second stage, they fine-tune an additional decoder to correct residual artifacts such as color shifts and seams. The method is evaluated through experiments on both small and large object removal tasks.

### Strengths
1. The paper effectively addresses the challenging and often overlooked problem of small object removal.
2. The work is supported by extensive experiments and demonstrates compelling qualitative results.
3. The paper is well-structured and clearly presented.

### Weaknesses
1. The camera-adaptive refocus mechanism in Stage-I appears highly effective for small objects, but its contribution to large object removal is less clear. An ablation study specifically isolating the contributions of Stage-I and Stage-II for large object removal would be insightful.
2. For Stage-II, it is unclear how robust the shadow-aware repair is to shadows that are cast far from the original mask region. Could such cases lead to failures or the introduction of new artifacts?
3. In Table 2, the metrics for Exp. (d) show a slight degradation in PSNR, SSIM, and LPIPS compared to Exp. (c). Could the authors provide further analysis on the reason for this performance trade-off?
4. Including a discussion or visualization of typical failure cases would further strengthen the paper by clarifying the method's limitations.

### Questions
Please refer to the weaknesses.

### Soundness
2

### Presentation
3

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
This paper proposes a two-stage framework to address the small object removal problem. First, the small object is enlarged using the Camera-Adaptive Refocus Strategy and input into the first-stage Camera-Adaptive Refocus Inpainter, where the object is removed via a fine-tuned pre-trained FLUX model. After that, the second-stage Seam- and Shadow-Aware Repair is applied. This step uses a fine-tuned FLUX VAE to maintain background consistency and eliminate shadows. Both metric validation and visual result demonstrations confirm the effectiveness of this framework.

### Strengths
1. This method can address the small object removal challenge and the color shift issue inherent in diffusion models.
2. The framework not only achieves significant metric improvement over most SOTA models but also produces high-quality visual results.

### Weaknesses
1. The first stage is reported to remove shadows, yet the shadowy background is pasted back and shadow removal is repeated in the second stage. This process seems to waste the shadow-free result obtained from the first stage.
2. In the experimental comparison with other models that lack the ability to remove unmasked shadows, providing only object-only masks may introduce unfairness to these competing methods.

### Questions
1. It is noted that RORD’s masks include objects, shadows, and an extra expanded edge, whereas the masks presented in this work only fit tightly around objects. It remains unclear whether these masks were re-labeled, and whether the same "object-only" masks were used for model training.
2. The reason for selecting VAE for the second-stage correction requires further clarification, as other architectures may exhibit better performance. Additionally, VAE typically tends to reduce image quality, and how this issue is addressed in the proposed method needs more explanation.
3. In the second stage, since only the zoomed image (with shadows but without the target object) is input, the model may struggle to distinguish which shadow originates from the target object. It is recommended to provide test results on a more challenging scenario: when other non-target-related shadows exist in the zoomed image, can the method exclusively remove the target’s shadow without affecting the others?
4. Whether the Camera-Adaptive Refocus Strategy has considered the possibility that part of the object’s shadow may be cropped outside the zoomed-in region needs to be confirmed.

### Soundness
2

### Presentation
3

### Contribution
2

# RestoreVAR: Visual Autoregressive Generation for All-in-One Image Restoration

- Decision: Accept (Poster)
- Scores: 4, 6, 2, 6

## Abstract
The use of latent diffusion models (LDMs) such as Stable Diffusion has significantly improved the perceptual quality of All-in-One image Restoration (AiOR) methods, while also enhancing their generalization capabilities. However, these LDM-based frameworks suffer from slow inference due to their iterative denoising process, rendering them impractical for time-sensitive applications. Visual autoregressive modeling (VAR), a recently introduced approach for image generation, performs scale-space autoregression and achieves comparable performance to that of state-of-the-art diffusion transformers with drastically reduced computational costs. Moreover, our analysis reveals that coarse scales in VAR primarily capture degradations while finer scales encode scene detail, simplifying the restoration process. Motivated by this, we propose RestoreVAR, a novel VAR-based generative approach for AiOR that significantly outperforms LDM-based models in restoration performance while achieving over $\mathbf{10\times}$ faster inference. To optimally exploit the advantages of VAR for AiOR, we propose architectural modifications and improvements, including intricately designed cross-attention mechanisms and a latent-space refinement module, tailored for the AiOR task. Extensive experiments show that RestoreVAR achieves state-of-the-art performance among generative AiOR methods, while also exhibiting strong generalization capabilities.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes RestoreVAR, an All-in-One Image Restoration (AiOR) model that is built on a pre-trained Visual Auto-Regressive (VAR) model. This method leverages the multi-scale prior knowledge in the VAR model and is thus well-suited for the AiOR task. To achieve this, the authors introduce a cross-attention mechanism for conditioning the pre-trained VAR model on the degradation images. Finally, to mitigate the information loss due to the vector quantization operations, a transformer-based post-processing model is introduced to refine the generated tokens. Similarly, the VAE decoder is fine-tuned to further improve the reconstruction quality. Together, RestoreVAR achieves state-of-the-art performances as the diffusion-based generative-IR models while being 10 times faster at inference.

### Strengths
- The method’s design is intuitive and reasonable. The proposed method is tailored for the VAR, utilizing its efficient architecture while offering good sample quality. 
- The motivation is well presented and analyzed. In Section 3.1 and 3.2, the authors presented their observations, where the multi-scale structure of the VAR model effectively simplifies the AiOR task to the generation of low resolution features.
- The manuscript is overall well-structured and easy reading.

### Weaknesses
- As mentioned in Section 3, the generative-IR methods are known to suffer from hallucinations that introduce artifacts in the results and affect metrics like PSNR and SSIM. Meanwhile, the generative nature of these methods offer improved image quality than their non-generative counterparts. This balance is also known as perception-distortion tradeoff. While RestoreVAR demonstrates advantages in PSNR and SSIM over other generative methods, they are not quantitatively compared on the image quality metrics like MUSIQ and CLIP-IQA. Specifically, Table 3 only presents a comparison in such metrics between RestoreVAR, a generative method, and other non-generative methods. I believe more discussions on the perception-distortion tradeoff will better demonstrate the advantages of RestoreVAR over other generative-IR methods.
- Although the efficiency advantage of RestoreVAR is well established, the advantage in terms of generation quality of a VAR backbone is unclear. I believe Table 4 shall also include image quality metrics. Without LRT, the performance of RestoreVAR on PSNR and SSIM metrics appears to be the same level as other generative-IR models (21.71/0.690 v.s. 21.28/0.738). If LRT compromised image quality, the improvements in generation quality of RestoreVAR will be limited, especially considering other generative-IR methods were not evaluated on a fine-tuned VAE decoder.

### Questions
- To help us better understand Weakness 1, could authors provide a set of ablation studies on the performance gains on the predicted higher resolution index maps? Specifically, a quantitative comparison between images with high-res index maps generated from ground-truth low-res index maps and from scratch generation. In other words, comparisons between [GT-low-res, Gen-high-res (from GT-low-res)] and [GT-low-res, Gen-high-res (from scratch)]? Additionally you could also provide another set of [Gen-low-res, Gen-high-res (from GT-low-res)] and [Gen-low-res, Gen-high-res (from scratch)], to better demonstrate the difference.
- Given the limitations in Weakness 1 and 2, could authors discuss more on the evidence that how to justify RestoreVAR to be claimed as state-of-the-art, instead of another point on the perception-distortion plane?

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
This paper introduces RestoreVAR, an all-in-one image restoration (AiOR) method based on visual autoregression (VAR). The authors improve VAR, originally designed for image generation, to enable it to handle multiple degradation types (dehazing, rain removal, snow removal, deblurring, and low-light enhancement) in a single model. Key technical contributions include: (1) a cross-attention mechanism to inject semantic information from degraded images; (2) a lightweight latent refinement transformer (LRT) to preserve fine details; and (3) a VAE decoder fine-tuned on a continuous latent vector rather than discrete labels. RestoreVAR achieves state-of-the-art performance among LDM-based generative methods, with inference speed 10 times faster than diffusion models, but significantly underperforms non-generative methods on some metrics.

### Strengths
1.RestoreVAR is the first work to apply Visual Autoregressive Modeling (VAR) to image restoration tasks in a generative setting, demonstrating the potential of autoregressive models beyond traditional generation tasks.

2.Across five degradation types, RestoreVAR achieves state-of-the-art performance among current LDM-based generative approaches, while also providing a 10× faster inference speed.

3.The ablation studies are well-designed and verify the effectiveness of each component in the proposed framework.

### Weaknesses
1.The paper compares RestoreVAR with both non-generative AiOR models and LDM-based generative methods. However, on many standard metrics, non-generative methods still outperform RestoreVAR. Although the paper acknowledges this and attributes the limitation to the VAE, it could further clarify when and why a generative approach is preferable in practice and quantify the trade-offs.

2.The fine-tuning of the VAE decoder on continuous latent variables seems like a heuristic workaround, but the paper does not sufficiently justify the correctness or theoretical soundness of this approach.

3.The user study reports average scores from 36 participants but does not provide confidence intervals or statistical tests. The evaluation lacks statistical significance analysis, confidence reporting, and detailed methodology, which are essential to support claims of perceptual quality superiority over non-generative methods.

4.The paper briefly mentions limitations but does not provide detailed failure case analysis to illustrate when and why the method may fail.

### Questions
1.According to the ablation study, continuous latent fine-tuning improves performance by around 20%. Could the authors elaborate on why the original discrete VAR struggles so significantly in restoration tasks?

2.How does the method perform on real-world mixtures of degradations that do not strictly fall into the five predefined categories?

3.What is the impact of varying the value of K (number of scales) on the trade-off between speed and restoration quality?

4.Can the authors provide a rigorous perceptual quality evaluation to justify that the gap on certain quantitative metrics between RestoreVAR and non-generative methods is acceptable in practice?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper extends VAR (Tian et al., 2024) for All-in-one Image Restoration where a single model is trained to handle multiple image degradation types. By adopting VAR as the backbone, the proposed method claims to be much faster at inference time when compared to latent diffusion model-based methods. The method incorporates cross attention with the degraded input latents to reduce the risk of hallucination. It also includes a latent refinement transformer (LRT) to further improve the final output quality. The paper describes a series of experiments to benchmark against existing generative and non-generative methods. Further ablations are presented to justify the design choices such as the LRT and use of continuous latents for conditioning and decoding.

**References:**  
Tian, K., Jiang, Y., Yuan, Z., Peng, B., & Wang, L. (2024). Visual autoregressive modeling: Scalable image generation via next-scale prediction. *Advances in neural information processing systems*, *37*, 84839-84865.

### Strengths
* The paper extends VAR for image restoration tasks, and discusses how restoration tasks can be framed within this ‘next-scale’ paradigm through the scale space analysis (Sec 3.2).  
* The proposed method performs better than other generative methods in terms of PSNR, SSIM and LPIPS (Table 1\) and is also significantly faster than these prior generative methods.  
* Experimental comparisons across numerous state of the art generative and non generative methods over standard datasets.

### Weaknesses
* In terms of PSNR, SSIM and LPIPS, the proposed method is poorer than existing non-generative baselines. The authors suggest that the generative nature of the proposed method provides better generalisation capabilities and attempt to demonstrate this with comparison on other datasets using referenceless IQA metrics. Given the differences in the datasets, methods benchmarked against (generative and non-generative in Table 1 vs only non-generative in Table 2), and metrics, it is hard to place where this proposed method fits into the wider literature. The paper discusses three different dimensions: i) Multi-task restoration performance, ii) generalizability to unseen degradation types, and iii) computational complexity. A concise discussion of the proposed method in comparison to prior generative and non-generative methods along these three dimensions could be useful.  
* The paper claims its strength for generalizability from the priors it has learnt from the VAE (line 84). It claims too that it is the first method to train a VAR directly for AiOR (line 147). Can the authors justify this claim when they have to finetune the VAE decoder and add in a latent refiner transformer? How is this different from HART (which is not compared).

### Questions
* Compare computational complexity of proposed method against non-generative baselines.  
* For the datasets in Table 2 that have clean/GT images (e.g. REVIDE, POLED, TOLED), is it possible to report PSNR, SSIM, LPIPS?  
* How do the different scales contribute to the overall restoration? An ablation study in the style of section 3.2 with {restored, restored+coarse\_deg, restored+fine\_deg, deg} would be helpful to validate that restoring the coarse scales contribute most to overall the restoration performance.  
* \[Off tangent\]What are the relative contributions of the VAR backbone and LRT to the overall restoration. Could an LRT \+ VAE decoder acting directly on the output of the VAE encoder outputs (continuous or quantized) produce reasonable results? (i.e. $\\hat{f}\_{cont} \= f^{deg}\_{cont} \+ LRM(f^{deg}\_{cont}, 0)$})  
* Can the authors clarify the results presented in Table 1\. Is RestoreVar trained specifically for each of the tasks? If so, is the comparison with Table 1 actually fair, as noted in line 374, non task-specific restoration models are not compared. If not, are the compared models trained with the same datasets (RESIDE, Snow100k, Rain13k, LOLv1, GoPro)?  
* What is the performance for other generative methods (Diff-plugin, Auto-DIR, PixWizard) on real-world unseen degradations (Table 2)? Why were they excluded?  
* For Table 2, RestoreVar is trained over what datasets?  
* What about other VAR restoration methods? VarSR and VarFormer – their exclusion from the experiments makes it hard for this reviewer to gauge the contribution of this proposed method.  
* The key claim of the paper’s approach in line 214 is that the degradations are separable from the fine details at various scales. Fig. 2’s illustration are images with synthetic degradation added → how far is this claim valid for real world degradations where the degradations from sensor and environment are mixed in a more complicated manner. Can the authors show the same results for the data used in Table 2?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces RestoreVAR, a method for image restoration based on Vision Autoregressive Modelling (VAR) framework. Compared to diffusion-based restoration methods, VAR-based image restoration offers rapid inference speed. The paper details the architecture modifications such as cross attention over the latents of degraded image, an additional latent refinement module for predicting the residual after quantization, and a fine-tuning procedure for the VAE decoder. Combining these changes, the paper shows RestoreVAR can obtain superior performance while while achieving over 10x faster inference.

### Strengths
The paper provides a nice visualization on on what is encoded in each scale and finds that VAR captures degradations predominantly in coarse scales and scene-level details in fine scales.

The paper argues that generative model offers strong generalization and convincingly demonstrates it with performance on real-world degradation as well as human preference.

Overall, it's a well executed and presented paper.

### Weaknesses
Fine-tuning VAE decoder and adding a Latent Refiner Transformer (LRT) are critical to the final performance of the model. In the meantime, these techniques seem to be transferrable to other methods. So it's unclear whether we can attribute the success solely to VAR.

### Questions
1. Could the author include some discussions on methods that speeds up inference for diffusion and how it compares with RestoreVAR? For example, consistency models and rectified flow.

2. In section 4.4, the paper provides an ablation on adversarial and pixel level loss. How about SSIM and perceptual loss?

3. In table 4, the paper reports a significant degradation in performance when the refinement network is trained without the last block output. Could you provide an explanation? Do you see degradation if removing the quantized prediction instead?

4. Have the authors tried inference tricks that boost performance for generation, such as classifier-free guidance? It seems like the degraded image can be treated as a conditioning signal?

5. The model seems to be reliant on conditioning signal after the fine-tuning. Does the model still retain any ability to generation image at all? What does $g_i$ (line 261) look like after the training?

### Soundness
3

### Presentation
4

### Contribution
2

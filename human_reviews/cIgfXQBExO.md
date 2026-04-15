# ZeroNVS: Zero-shot 360-degree View Synthesis from a Single Real Image

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 3, 6, 5

## Abstract
We introduce a 3D-aware diffusion model, ZeroNVS, for single-image novel view synthesis for in-the-wild scenes. Compared with existing methods mainly developed for single objects with masked backgrounds, we propose key improvements to address challenges introduced by in-the-wild scenes with complex backgrounds. Specifically, we train a generative prior on a mixture of data sources that capture object-centric, indoor, and outdoor scenes. As the data mixture presents various issues such as depth-scale ambiguity,  we present a novel camera parameterization and normalization scheme. Further, we observe that Score Distillation Sampling (SDS) tends to truncate the distribution of complex backgrounds during distillation of 360º scenes, and  propose ``SDS-anchoring'' to improve the diversity of synthesized novel views. Our model sets a new state-of-the-art in LPIPS on DTU in the zero-shot setting, even outperforming methods specifically trained on DTU. We further adapt the challenging  MipNeRF360 dataset as a new benchmark for single-image novel view synthesis, and demonstrate strong performance. Our code and data will be available on acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper tackles single-image novel view synthesis for in-the-wild scenes with diffusion models. Its main contributions lie in three folds.

1) The authors train a single-view-to-3D model for scene with complex backgrounds.
2) To make use of diverse datasets against scale ambiguities, the authors use off-the-shelf depth estimation to align the scales.
3) The authors adopt SDS anchoring which conditions the diffusion model on the nearest view in score distillation.

The model is trained on a mixture of CO3D, ACID, and RealEstate10K and evaluated on DTU and Mip-NeRF 360 datasets, achieving SOTA on both of them. The qualitative results and user study have also shown promising results.

### Strengths
1. The quantitative results are better than current state-of-the-art approaches.
2. The analysis of SDS anchoring is useful in scene-based single-image novel view synthesis.

### Weaknesses
1. The qualitative results are lack of comparison with other methods. This makes readers hard to decide if the qualitative results are better. Most of the results shown in the paper are still object-centric while with a complex background. It would be good to show comparisons with object-centric methods that mask out the background to see if the novel view synthesis for the object is better.
2. The authors mentioned SDS anchoring can increase diversity but did not include any diverse results to support this point, only showing a user study which is not convincing enough. It would also be good to use FID metrics in comparison.
3. The technical contribution is limited. The 2nd contribution (aligning the scale of different datasets) is weak to me. And the 3rd contribution (SDS anchoring) is not well supported.

### Questions
1. In Sec. 3.2 "be a function which scales the translation component of the extrinsic matrix $E$ by $\lambda$" and the following equation - is the L2 norm applied to the whole extrinsic matrix or only the translation part?
2. For SDS anchoring if we condition the diffusion model on a generated view, is it possible to degenerate?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper approaches the task of single image novel view synthesis for scenes. It proposes to finetune Zero 123 on a collection of three large datasets (CO3D, RE10K, ACID) and generalize to evaluation datasets (Mip-NeRF 360, DTU) zero-shot. In order to train on data with 6DoF data, it adapts the existing pose parameterization of Watson et al. To train on data with scale ambiguity, it uses a normalization scheme; to encourage diversity it proposes SDS-anchoring. It outperforms reported baselines in LPIPS on DTU and MipNeRF-360.

### Strengths
Introduces new training paradigm which yields respectable results
- Technical contributions of handling changing depth scale and intrinsics allow the model to train on several large datasets. 
- The model generalizes well to new data zero-shot
- Conditioning representation is ablated
SDS-anchoring is effective in improving sampled view diversity
- Clear from qualitative and quantitative results. Idea is intuitive and novel.
Paper is generally clear and well-motivated
- Fig 2-5 are effective along with the text, in motivating the major contributions of the paper (excluding the motivation for the choice of data)

### Weaknesses
Summary: I believe the paper has good contributions and presentation, but is currently missing several important evaluations. If the paper had more robust and convincing experiments I would be open to raising my score.

Missing comparisons
- The paper claims SSIM and PSNR are not appropriate for views the model is hallucinating. It is true these are not the best metrics, but it then only reports one metric (LPIPS), and contains only one visual comparison against prior work (Figure 7) – this includes supplemental and website. In a task of novel view synthesis, if two out of the three reported metrics are weaker than prior work, visual results are necessary to convince reviewers the proposed model improves on results. More examples would also be helpful to understand the drop in PSNR and SSIM numbers.
- The paper says “long runtime makes typical generation-based metrics such as FID cost-prohibitive”. I’m confused: in my experience, FID can be run in a minute or two over thousands of images. Given FID is an important metric for hallucinated images, it (or perhaps KID, or similar) should be reported at least on a subset of images. The alternative could be a human A/B test, which is not as ideal, since it cannot be precisely replicated, but still could give more defense to the results vs. existing methods.
- PixelNeRF is not the state of the art on single-image novel view synthesis. Table 2 is central to the paper’s argument, but unfortunately the comparisons are limited to Zero-1-to-3 and PixelNeRF. 
    - The proposed method finetunes this on scenes, so should improve over Zero-1-to-3; so it would be good to see other baselines as well.
    - These two baselines do not have mechanisms to deal with differing intrinsics. Perhaps a better comparison would be to train these only on e.g. CO3D or only on RealEstate10K
    - As a result, it is hard to determine how effective the model is vs. prior work, or of the main contribution is training on large data?
    - Perhaps some recent SOTA methods could be trained on the same data. For instance (comparison does not have to be to this baseline, just an idea:) Consistent View Synthesis with Pose-Guided Diffusion Models (Tseng et al., CVPR 23), SynSin (Wiles et al., CVPR 20)
    - RegNeRF is better than DietNeRF in 3, 6 and 9 view. It should be considered: (RegNeRF: Regularizing Neural Radiance Fields (Niemeyer et al., CVPR 22))
-	Can the proposed model be finetuned e.g. on DTU for performance gain? If not, why not?

### Questions
- Why train on ACID? I was under the impression this doesn’t have views facing 360 degrees, but is rather mostly forward trajectory. What was the motivation for these datasets?
- Ablations were trained for fewer steps (25k) for compute reasons. How much compute was used? How long did training take?
- Typo, Figure 6 caption

### Soundness
2 fair

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
**Summary:** 
This paper introduces ZeroNVS, representing a scene-level NVS approach. In addition, it proposes several conditioning representation methods.

### Strengths
**Advantages:** 
1. The paper gives a comprehensive introduction to the conditioning representation method.
2. Under the zero-shot setting, the result is good,  but there is still a significant potential for further improvement.
3. The overall writing is satisfactory but could benefit from further refinement.

### Weaknesses
**Disadvantages:** 
1. The contributions made in the paper appear nuanced, making it challenging to discern the most crucial contribution.
2. It would be beneficial to compare the proposed method with GenNVS, which demonstrates leading performance in scene-central applications.

### Questions
**Questions/Concerns:** 
1. What distinguishes ZeroNVS from Zero123 when claiming that your method is scene-central as opposed to object-central? It is unclear which elements bolster this claim. Specifically, does the distinction lie in Zero123 being trained on Objectverse while your method relies on RealEstate10K, ACID, and CO3D?
2. Regarding the zero-shot application on the DTU Dataset, how does ZeroNVS stack up against GenNVS? 
Moreover, if ZeroNVS undergoes fine-tuning on the DTU dataset, how does its performance compare to that of GenNVS? 
Notably, in a zero-shot setting, it lags behind GenNVS, which has been trained on the DTU dataset.
3. About the comparison with Zero123, can you further report the result of ZeroNVS on GSO dataset?

**Reference:** 
@misc{chan2023genvs,
  title={GeNVS: Generative novel view synthesis with 3D-aware diffusion models},
  author={Chan, Eric R and Nagano, Koki and Chan, Matthew A and Bergman, Alexander W and Park, Jeong Joon and Levy, Axel and Aittala, Miika and De Mello, Shalini and Karras, Tero and Wetzstein, Gordon},
  year={2023},
  publisher={arXiv}
}

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper introduces a method for new view synthesis (NeRF reconstruction) from a single image. They propose a two-stage method, ZeroNVS, and new techniques to elevate single-image object-centric novel view synthesis (NVS) to NVS on real-world scenes. Comparing to its backbone method DreamFusion, the main contributions are:

- Introduce focal length into poses to address the scale ambiguity
- Improve SDS in DreamFusion to SDS-anchoring
- Adopt Multi-plane model in Stereo Magnification to do a scene scale normalization for better generalizations

### Strengths
- Introduce focal length into poses to address the scale ambiguity
- Improve SDS in DreamFusion to SDS-anchoring
- Adopt a Multi-plane model in Stereo Magnification to do a scene scale normalization for better generalizations

### Weaknesses
- Technical contributions are incremental, mostly are add-ons to previous methods
- What is DDIM sampling in Figure 6?  I feel it is the main content in SDS-anchoring, and possibly the key point of why SDS can be used to scenes not just objects. But no details and descriptions are found. 
- Typos "us us to train..." in Section 3 (2nd paragraph). 
- If PSNR and SSIM are not reliable, why not change to other metrics? One example in Figure 7 cannot show the whole picture, so if we should not trust them, why evaluating them in tables? Not to mention that in Table 2, ZeroNVS has the bect PSNR and SSIM in DTU dataset, is it good or not? It is very confusing here.

### Questions
See weakness

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

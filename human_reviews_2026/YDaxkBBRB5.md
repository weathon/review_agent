# Robust Image Self-Recovery against Tampering using Watermark Generation with Pixel Shuffling

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 8, 8

## Abstract
The rapid growth of Artificial Intelligence-Generated Content (AIGC) raises concerns about the authenticity of digital media. In this context, image self-recovery, reconstructing original content from its manipulated version, offers a practical solution for understanding the attacker’s intent and restoring trustworthy data. However, existing methods often fail to accurately recover tampered regions, falling short of the primary goal of self-recovery. To address this challenge, we propose ReImage, a neural watermarking-based self-recovery framework that embeds a shuffled version of the target image into itself as a watermark. We design a generator that produces watermarks optimized for neural watermarking and introduce an image enhancement module to refine the recovered image. We further analyze and resolve key limitations of shuffled watermarking, enabling its effective use in self-recovery. We demonstrate that ReImage achieves state-of-the-art performance across diverse tampering scenarios, consistently producing high-quality recovered images.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This submission aim to embed self-recovery watermarks into images to enable tamper localization and original content reconstruction. This work advances prior work by addressing spatial alignment fragility via fine-grained pixel shuffling, introducing a learned Watermark Generator (WG) to suppress high-frequency artifacts, and adding an Image Enhancement (IE) module.

### Strengths
Results show that the recovery quality of the method is higher. In many tests, the results are much better than the existing methods.

### Weaknesses
1. It seems the applied validation dataset is different from the applied baselines. It seems how these baseline models are prepared remain unclear.

2. It seems the applied methodology lacks novelty. Like Invertible networks and mask free generation are also applied in Imuge.

3. It seems the reviewer cannot easily benchmark the advantage of this method, either via theoretical or empirical analysis, or via source code (no code or API provided. though it is completely optional, the reviewer cannot play with the model to address some concerns).

4. It seems the role of accurate attack localization is underestimated (why does the propose mechanisms also boost localization? And why the compared methods fail in the detection stage in many provided samples?) The review can only see the lower results of the compared methods via figures and tables, but barely see the performance of attack localization (except table 8, deeply hidden in the supplement). The reviewer suspects that the focus on localization training of the proposed method more contributes to the final result (especialy considering the reported large performance gap). The reviewer think in this task, authors need to report fair quantitative comparison of recovery using same localization result (either being it a good one or a poor one), in order to benchmark the contribution of modules like WG, IE, etc. In the current version, the two factors are simply interwined.

5. The reviewer thinks the visualization of the proposed method can be rather biased. The authors do show the improvement in the end-to-end result (better recovered images). However, the reviewer barely see any visualization of intermediate results, i.e., prediction mask, or any other signals that could somehow interpret the successes. The authors claim that the compared baseline methods can be less robust or show less generalizability. However, in many shown cases, the baselines simply cannot locate the attack, which can significantly lower their overall performance. So the reviewer personally is in favor of a more well-rounded result report than a biased one.

6. Figure 6 reports results under simulated attacks, which can be less convincing, since attacking images using circles and rectangles can easily leave traces for localization. Also, the. difference between table 7 and table 9 is worth inquiring: does table 7 report in-domain (train and test from a same dataset) performance and table 9 report cross-domain? If not, the reviewer cannot understand why there is a large gap of the baselne performances between the two tables. Also, the figure 10, the watermarks can be barely seen for all methods.

7. Finally, the baselines can be further improved. From the results, we see that the baselines have a gap with the proposed method. Meanwhile, imuge and imuge+ are proposed by a same team, and W-RAE is also using INN for this task. Thus, the reviewer think we need to add more stronger baselines. Remind: it seems this task can be easily achieved by many other image translation networks, such as transformers, diffusions, mambas, etc.

### Questions
Please refer to the previous section.

### Soundness
2

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
4

### Summary
This paper focuses on image self recovery, which in short is to locate the tampered areas within an image and try to recover the original area. The proposed method ReImage proposes embedding a shuffled version of the target image as a watermark via a novel framework to misalign corrupted regions and their watermarks.

### Strengths
- Authors design several modules to improve the self-recovery, and the recovered images are better.
- Authors provides many experimental results to show the performance in different aspects.

### Weaknesses
- This paper to me is more like an engineering paper. Many parts (from the pipeline: two-staged, similar to the compared imuge, to the model: inn, similar to W-RAE, again to the training details: mixed and fixed jpeg, filtering, etc.) show minor academic innovation. The proposed pixel shuffling for watermark and the two modules (IE and WG) are also trivial designs.
- Pixel shuffling improved image recovery, but also increase the entropy of the information to be hidden. After reading Section 3.3.2, I still don't understand why in the proposed method, both imperceptibility and recovery are both improved (simply via using transformer blocks in INN?)
- Experimental details seems lacking, and the compared results deviate from the ones reported in the paper.

### Questions
Major:
- How are the methods compared? I see a significant performance gap between the original reported ones and those in here. E.g., In W-RAE, the averaged reported PSNR was 32.42 (db), while in this paper mostly lower than 25db.
- Why do the authors use a novel dataset setting for comparison? Rather than accepting an existing experiment setting? Why a third party paper Editguard's criterion is applied here?
- In the tables and figures i don't see the mask result. Also in table 9 the psnr of two compared method are as low as 15- db. Authors should give mandatory test details to indicate that this was not a mistake.
- Ensuring weak data hiding and good performance is a trade-off. Just like what is mentioned in "weaknesses", still do not understand how is the method so efficient in the watermarking? Also, what block size do you use in the experiments? If i understand it corrent, the P=4 refers to the num of block in transformer? Besides, why the curve of recovered image also go down with the increase of num of blocks? WIth hidden image (ground truth) shuffled further, we should expect that the recovery capacity should increase, despite the burden on the embedding side.
- Unclear motivation: "Since we apply the shuffling algorithm, the recovered image ˆIorg exhibits globally distributed degradation" i 

Minor:
- Real-scene attacks can be more likely compound attacks than rare attacks like hue adjust and contrast change. E.g., dual compression, or rescale than JPEG.
- It is not easy for me to distinguish which table and figures are reporting results on real-scene attacks, which are on simulated conditions.
- Writing issue - you should refrain from always using the right (closing) quotation mark in the paper.

### Soundness
2

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
3

### Summary
This paper introduces TamperTok, a novel framework for image tampering detection and localization (IMDL) that leverages Multi-modal Large Language Models (MLLMs). The authors identify key weaknesses in prior MLLM-based approaches, namely their reliance on "stitched" pipelines with exogenous segmentation decoders , which leads to information bottlenecks and imprecise localization due to a mismatch between semantic MLLM embeddings and the spatial nature of the task.

### Strengths
- The paper's primary contribution, reformulating MLLM-based localization from a segmentation-based "stitched" pipeline to an end-to-end autoregressive token generation task, is highly novel and elegant.

- The problems are clearly articulated, and the proposed solutions (KSD, SwEI) are well-motivated, technically sound, and directly target the identified weaknesses. The experimental evaluation is comprehensive and rigorous.

- The fact that this MLLM-based model surpasses dedicated, SOTA forensic expert models like SparseViT in a challenging cross-domain setting, provides a new and much more effective blueprint for adapting MLLMs to fine-grained, non-semantic, and spatially-precise tasks.

### Weaknesses
- The SwEI module's success relies on injecting features from SparseViT, which is itself a SOTA forensic expert. This raises a question of how much of the performance is simply a successful distillation of the expert model.

- The proposed TamperTok architecture involves running a large MLLM and an expert model (SparseViT) in parallel to extract features, followed by an autoregressive decoding step. This is almost certainly more computationally expensive (in terms of parameters, GFLOPs, and latency) than the baselines (e.g., SparseViT alone, or FakeShield which uses a lightweight decoder).

### Questions
How crucial is the choice of SparseViT as the expert model for SwEI? Have you experimented with using a weaker or architecturally different expert (e.g., ManTra-Net, MVSS-Net)? Is the SwEI performance boost contingent on using a top-tier expert?

Other questions may be referred from Weakness part.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduces ReImage, a self-recovery framework for images based on neural watermarking. The approach embeds a shuffled, high-frequency-suppressed version of the target image into itself as a watermark using an invertible neural network (INN)-based system. The key contributions include a specialized watermark generator, pixel shuffling to disrupt spatial correlation for better tamper recovery, and an image enhancement module to further improve output fidelity. Extensive experiments on the MS-COCO2017 dataset and various tampering scenarios show state-of-the-art performance, both in visual quality and robustness to common degradations.

### Strengths
1. Novel and Effective Framework: The paper proposes ReImage, a well-designed neural watermarking-based self-recovery method that leverages pixel shuffling to spatially misalign watermark content with the image. This innovation addresses a known issue of recovery failure due to alignment between tampered and watermarked regions along with clustered tampered regions in secret image.

2. Thorough Design and Ablation Study: The architecture is modular and interpretable, consisting of components like the invertible watermarking network, a learned watermark generator, image enhancement, and tamper localization. Ablation experiments clearly demonstrate the impact of each component on recovery performance.

3. Superior Performance on Tampering Tasks: On various tampering types (e.g., SD-Inpaint, SDXL, and splicing), ReImage achieves state-of-the-art results, showing improvements over prior work like W-RAE and Imuge+.

### Weaknesses
1. Insufficient Evaluation under diverse degradations: While robustness to three types of degradations is briefly evaluated (Gaussian noise, JPEG compression, and Poisson noise), the degradation types are limited, and there is no geometric degradation included (Imuge has included cropping in its experimental evaluation). It is well known that geometric degradations, such as cropping, pose significant challenges for watermarking models, suggesting a potential trade-off between robustness and image quality.

2. Limited Real-World Validation: The limited range of tested degradations raises concerns about the method’s practicality, as real-world scenarios often involve more complex and compound distortions—such as slight cropping, minor rotations, or cases where a region of an image is cropped and spliced onto another image. These types of manipulations were not sufficiently explored, making it unclear how well the method generalizes beyond controlled experimental settings.

### Questions
Please refer to the weakness.

### Soundness
3

### Presentation
3

### Contribution
3

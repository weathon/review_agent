# Less Gaussians, Texture More: 4K Feed-Forward Textured Splatting

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 2, 6, 4, 8

## Abstract
Existing feed-forward 3D Gaussian Splatting methods predict pixel-aligned primitives, leading to a quadratic growth in primitive count as resolution increases. This fundamentally limits their scalability, making high-resolution synthesis such as 4K intractable. We introduce LGTM (Less Gaussians, Texture More), a feed-forward framework that overcomes this resolution scaling barrier. By predicting compact Gaussian primitives coupled with per-primitive textures, LGTM decouples geometric complexity from rendering resolution. This approach enables high-fidelity 4K novel view synthesis without per-scene optimization, a capability previously out of reach for feed-forward methods, all while using significantly fewer Gaussian primitives. Project page: https://yxlao.github.io/lgtm/.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper introduces LGTM (Less Gaussians, Texture More), a feed-forward framework for high-resolution (up to 4K) novel view synthesis with textured Gaussian primitives. The core idea is to decouple geometry from appearance: a primitive network consumes low-resolution inputs to predict a compact grid of 2DGS geometry (centers, scales, rotations, SH base color), while a texture network consumes high-resolution inputs to predict per-primitive color and alpha texture maps via image patchifying and learned projective texturing. They used a 2-stage training, achieving coarse-to-fine prediction. Firstly a rough geometric and then detailed textures.

### Strengths
1. The paper is well-writen and easy to understand. 
2. The results seem to be pretty good improvement on the base models. 
3. Pretty efficient algorithm and is friendly to small GPUs.

### Weaknesses
1. This work uses the few Gaussians from low-resolution images to serve as geometry probes, and paint the surfaces with higher resolution images. This is some what equivalent to first obtain all Gaussian points from the full resolution images (one may achieve this through similar way as e.g. Point3R[1]) and then uniformly **drop** most of the Gaussians. This relies on an important (but not necessarily true) assumption that real-world geometries are all smooth. This may causes the model to potentially fail when observing high frequency geometry details e.g. hairs, cloths, bumps, etc. No result has been given in that regard. Instead, a more clever way of downsampling / compressing Gaussians could be leveraged to both preserving geometry details & acceptable resources needed.
2. The baseline models are never trained on high resolution images while it can be done: e.g. augmenting the original data with different zoom-in / zoom-out scale & crop to original resolution at train time and inference with full-resolution. The current large scale evaluation doesn't make too much sense.
3. In L232-245 the authors use literally 1/4 page talking about a very elementary technique on assigning texture colors to Gaussian points, while ignoring some of the possibly more challenging topics such as how to handle view inconsistencies with less Gaussian points on high resolution images.
4. The authors may have used the term "primitive" with different meanings without firmly defining any of them. As a result, it is hard to parse some parts of the manuscript. For example, what is "per-primitive texture maps", and how does it differ from "2DGS primitives"? The term "Primitive Resolution" repeatedly appears but is never explained.
5. The use of mathematical symbols are not consistent, e.g. Eq 4. $f_{texture}$ seems like a mapping, while in L226, it becomes a "feature". The use of superscripts and subscripts are totally a mess.

[1] Wu et al. Point3R: Streaming 3D Reconstruction with Explicit Spatial Pointer Memory.  arXiv:2507.02863.

### Questions
1. Do you have results/failure cases on scenes with thin/fine structures (hair, fabric wrinkles, foliage)? 
2. Can you compare against a full-resolution primitive predictor followed by (a) uniform decimation and (b) geometry-aware decimation/compression (e.g., curvature/edge cues)? What are the trade-offs?
3. Were baselines retrained with strong multi-scale zoom/crop augmentation and full-resolution supervision, and evaluated under compute-matched budgets (memory/time)? If not, can you add this setting (and optionally test-time supersampling) to isolate the benefit of your texture mechanism?
4. Can you report a simple consistency diagnostic (e.g., reprojection error or pose-jitter stress test) at high resolution?

### Soundness
2

### Presentation
2

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
This paper proposed LGTM, a general extension for feed-forward 3D/2D Gaussian Splatting 
to enable 4K resolution novel view synthesis. Specifically, the proposed method decomposes the geometry and texture, utilizing a primitive branch to predict the geometry from low-resolution images and a texture branch to predict the high-resolution textures.

### Strengths
1. The paper is well-motivated.
2. The paper is well-written and generally easy to follow.
3. The experiments are throughout and the proposed method significantly boosts the performance of the baselines.
4. The results is visually good.

### Weaknesses
1. Necessity of integrating 4K texture to 3DGS: another solution to get 4K renderings can be a general feed-forward 3DGS followed by an image super-resolution model. A comparison should be made between it and the proposed method.
2. Evaluation: As the paper claims an immersive user experience, a non-reference perceptual metric such as Niqe [1] or Q-align [2] should be added as an evaluation metric.
3. Line space issue in L.474.

[1] Zhang, L., Zhang, L., & Bovik, A. C. (2015). A feature-enriched completely blind image quality evaluator. IEEE Transactions on Image Processing, 24(8), 2579-2591.

[2] Wu, H., Zhang, Z., Zhang, W., Chen, C., Liao, L., Li, C., ... & Lin, W. (2023). Q-align: Teaching lmms for visual scoring via discrete text-defined levels. arXiv preprint arXiv:2312.17090.

### Questions
Please see in weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces a new feed-forward framework called LGTM (Less Gaussians, Texture More) to solve a major problem: existing feed-forward 3D Gaussian Splatting (3DGS) methods cannot scale to high resolutions like 4K.
The Solution (LGTM): The method decouples geometry prediction from appearance prediction using a dual-network architecture.
- A Primitive Network takes a low-resolution image (e.g., $512 \times 288$) to predict a compact, fixed set of geometric primitives.
- A Texture Network takes the high-resolution image (e.g., 4K) to predict detailed, per-primitive texture maps that "paint" high-frequency details onto the simple geometry.

### Strengths
1. A novel topic on feed-forward 3D-GS.
2. A fair well performance compared with baseline methods.

### Weaknesses
1. High-resolution rendering requires accurate geometric prediction. However, the proposed method seems more like a trick—it projects a high-resolution image onto relatively coarse geometry. While this may work well when the input views have small viewpoint differences, it would be helpful if the authors could include additional experiments to evaluate the novel-view synthesis quality under different camera pose settings.
2. The paper claims to "decouple" geometry and appearance 20, yet the architecture (Fig. 2) and method (Sec 4.2) show that the texture network $f_{texture}$ explicitly takes the primitive network's features $F_{prim}^v$ as input. This seems to be a one-way coupling rather than a full disentanglement.
3. Although the paper provides a comprehensive analysis of the feed-forward 3DGS paradigm, recent per-scene optimization methods (e.g., Grendel-GS, CityGS-X) have already addressed high-resolution (e.g., 4K) rendering quite effectively. The reviewer understands that a direct comparison with these methods may be beyond the scope of this work, but it nonetheless raises the concern that the problem addressed here might not be as critical for the 3D vision community as implied.

Overall, the paper tackles a relatively minor problem (i.e., 4K resolution rendering) using a fairly simple approach—mainly by adding an additional downstream head.

### Questions
Perhaps the authors could include more visualizations, especially video demonstrations of the rendered views based on the predicted 3D-GS.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This work proposes LGTM, a feed-forward framework that predicts compact Gaussian primitives coupled with per-primitive textures for high-resolution novel view synthesis. By decoupling geometry and appearance with a dual-network design, the method enables 4K rendering with significantly fewer primitives and without per-scene optimization. LGTM is applicable across various feed-forward 3DGS baselines, including single-view, two-view, and multi-view setups, and demonstrates consistent quantitative and qualitative improvements on several benchmarks.

### Strengths
* The idea of combining low-resolution Gaussian primitives with high-resolution texture maps to achieve high-resolution feed-forward Gaussian Splatting is well-motivated and sound. Such an insightful finding may significantly boost the feed-forward 3DGS community to explore higher-quality synthesis.

* The introduced module is architecture-agnostic and is thoroughly evaluated across several state-of-the-art models and large-scale benchmarks. All experiments show consistent quantitative and qualitative improvements.

* The manuscript is well structured and easy to follow.

### Weaknesses
* Lack of multi-view results. It would be better to provide video results or multi-view images to better illustrate the impact of the texture modules. I am concerned that the texture module may potentially destroy multi-view consistency to some extent.


* Lack of evaluation under dense multi-view settings. Most experiments are conducted with 1, 2, or 4 views. Since the multi-view model is based on VGGT, which natively supports dense input views, it would be better to include results with denser settings, such as 32 or 64 views, similar to AnySplat [ref 1].


* Missing comparison with per-scene optimization methods. It would strengthen the work to compare with per-scene optimization approaches such as BBSplat, given that BBSplat inspired this work. Such comparisons are also common in other works  exploring high-resolution feed-forward 3DGS, such as Long-LRM [ref 2] and LVT [ref 3].

### References:

* [ref 1] Jiang, Lihan, et al. "AnySplat: Feed-forward 3D Gaussian Splatting from Unconstrained Views." arXiv preprint arXiv:2505.23716 (2025).
* [ref 2] Ziwen, Chen, et al. "Long-lrm: Long-sequence large reconstruction model for wide-coverage gaussian splats." ICCV 2025.
* [ref 3] Imtiaz, Tooba, et al. "LVT: Large-Scale Scene Reconstruction via Local View Transformers." arXiv preprint arXiv:2509.25001 (2025).

### Questions
Kindly refer to [Weaknesses] section.

### Soundness
4

### Presentation
4

### Contribution
4

# Motion-Aware Surface Smoothing for Monocular Avatar Representations

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 6, 2

## Abstract
3D Gaussian Splatting (3DGS) has become a popular representation for 3D avatar modeling due to its fast training and real-time rendering. However, the state-of-the-art methods struggle to generalize from sparse inputs and often fail to recover realistic geometry. We introduce a motion-aware surface smoothing framework to improve 3DGS for learning from monocular human videos. Our method regularizes the training of Gaussian parameters, modulates the Adaptive Density Control (ADC) for improving surface quality, and supervises Gaussian motions under unseen camera viewpoints. The enforcement of surface smoothness yielding superior geometry contours and higher-fidelity rendering. Across five public datasets, including MVHumanNet, DNA-Rendering, ActorsHQ and outdoor videos, our approach consistently outperforms prior methods in novel view synthesis, novel pose animation, and 3D shape reconstruction. Code will be published upon acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The focus of the paper is on human modeling from monocular videos, especially from sparse frames. 

The main contribution is a regularization term to encourage the smoothness of the surface from a camera view. Since the regularization is computed in 2D space after rendering the avatar articulated by a certain pose, the smoothness is disrupted at occlusion boundaries and in regions with holes. Therefore it further weights the regularization based on a distance between the corresponding points in the canonical space.

It furthers adds a smoothness-based heuristic in adaptive density control: If a Gaussian contributes more to smooth surface, it will be more likely to be densified.

It experiments on MVHumanNet, DNA-Rendering, ActorsHQ and Youtube videos. It shows superior results in sparse-view settings compared to the baselines.

### Strengths
* The paper tackles the important problem of human modeling from sparse inputs. While dense inputs may require cumbersome data collection, sparse views as inputs are more practical.
* The paper is clearly written and easy to follow. The methods are well motivated, e.g., why to weight the difference of depths based on the distance of Gaussians in canonical space.
* The idea of regularizing the surface smoothness is interesting. It identifies the cases (occlusions and holes) where smoothness regularization does not hold and proposes the fix for such scenarios. It is novel and reasonable to considering the contribution of a Gaussian to the smooth surface in adaptive density control. 
* It experiments on various dataset and challenging in-the-wild videos, showing that the method is widely applicable.

### Weaknesses
* The paper does not compare to iHuman, which also tackles the problem of human modeling from sparse inputs. iHuman [1] can model with as few as 6 views. Therefore, it is an important baseline.

[1] iHuman: Instant Animatable Digital Humans From Monocular Videos, Paudel et al., ECCV 2024.

* While the contributions lie in the smooth regularization and geometry-aware adaptive density control, the biggest improvement comes from replacing 3DGS with GoF and the normal map supervision from Sapiens.

* When applying the smoothness regularization, the authors choose the observation space where occlusions disrupt the smoothness assumption. However, such difficulties may not exist in the canonical space, where the avatar is carefully poses to avoid occlusions as much as possible. It is not clear to me the benefits of using observation space.

* The image sizes in the supplementary videos vary continuously, making it hard to identify the potential flickering and multiview consistency.

* One of the key challenges of modeling from monocular videos is the incomplete observation. The monocular videos may not depict every aspect of the avatar. The paper does not show multiview rendering, therefore it's hard to tell how it hallucinates in unseen regions.

### Questions
My main concerns are the missing baseline and limited improvement from the contributions, as listed in the weakness. Besides, the followings are minor questions and suggestions:

* When locating the nearest Gaussian along the ray, does the method also consider the value of the opacity? For example, if a Gaussian very close to the camera is almost transparent, will it be used as depth? If so, the depth may be misleading.

* I suggest to be careful with the word "sparse inputs" as sometimes it refers to even sparser input like 3 images. The settings of the paper are monocular videos which use over 100 frames for training.

Potential typos:
* In Eq. 3, the right most $x_o$ should be $x_c$.
* In Eq. 7, is it supposed to be $S_i=\sum_{j}w_j \|I_j^d - I_i^d\|$ rather than $S_i=\sum_{j}w_j (I_j^d - I_i^d)$?

### Soundness
3

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
4

### Summary
The paper addresses the task of monocular video-based human reconstruction by introducing a motion-aware surface smoothing framework. The key contribution lies in modulating the Adaptive Density Control (ADC) and supervising Gaussian motions using unseen virtual camera viewpoints. By explicitly regularizing surface smoothness through depth-based constraints, the method enhances the geometric consistency of 3D Gaussian Splatting. Evaluations across five public datasets demonstrate improved performance over recent state-of-the-art approaches in novel view synthesis, pose animation, and 3D shape reconstruction.

### Strengths
- The motion-aware surface smoothing mechanism effectively enforces spatial regularization of Gaussians using rendered depth maps. Combined with the geometry-aware Adaptive Density Control, it results in smoother and more compact surface geometry.
- The proposed method not only improves visual fidelity but also produces plausible mesh reconstructions, outperforming previous Gaussian-based avatar models in terms of geometric coherence and normal consistency.

### Weaknesses
- While the paper claims improvements in mesh reconstruction quality, no mesh-level evaluation metrics such as *Chamfer Distance* or *Point-to-Surface (P2S)* error are reported. Including these would quantitatively support the claim that the method improves geometric accuracy. A comparison against recent mesh-based human reconstruction works would further strengthen this aspect.

- In ActorsHQ novel pose evaluation, the performance drops compared to several baselines. The authors briefly acknowledge this in the appendix but do not provide an explicit analysis. An explanation regarding the causes would be valuable.

- The supplementary video exhibits minor stretching artifacts and floating points around the legs (notably at 1m 5–8s). Since the framework is designed to promote spatial smoothness and motion consistency, a discussion on why these artifacts occur would clarify the limitations.

- The comparison set is limited. Several recent monocular video-based reconstruction works with released code, such as *HUGS* (Kocabas et al., 2024), and *Expressive Gaussian Avatar* (Moon et al., 2024), should be included for a fairer comparison.

- Missing relevant recent references related to mesh-based / gaussian-based reconstruction:
    - [1] Moon, Gyeongsik, et al. *“Expressive Whole-body 3D Gaussian Avatar.”* **ECCV 2024.**

    - [2] Shin, Jisu, et al. *“CanonicalFusion: Generating Drivable 3D Human Avatars from Multiple Images.”* **ECCV 2024.**

    - [3] Shao, Zhijing, et al. *“SplattingAvatar: Realistic Real-time Human Avatars with Mesh-embedded Gaussian Splatting.”* **CVPR 2024.**

    - [4] Svitov, David, et al. *“HAHA: Highly Articulated Gaussian Human Avatars with Textured Mesh Prior.”* **ACCV 2024.**

### Questions
- Could the authors provide training time comparisons with other baselines in the main paper or appendix? Since 3DGS-based methods often emphasize efficiency, this comparison would contextualize the proposed improvements.

- What are the common failure cases (e.g., loose garments, extreme poses, occlusions, or inaccurate SMPL initialization)? Including a small visualization of typical failure cases would help readers understand the framework’s limitation.

### Soundness
3

### Presentation
2

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
This paper proposes MASSAR (Motion-Aware Surface Smoothing for Avatar Representations), a novel framework designed to enhance 3D Gaussian Splatting (3DGS) for human avatar modeling from monocular videos.
The core contribution lies in a motion-aware smoothness regularization, which promotes geometrically consistent Gaussian distributions through depth-map-based supervision.
The proposed regularization is applied in three complementary ways: (1) directly as a smoothness loss, (2) integrated with a geometry-aware Adaptive Density Control (ADC) strategy, and (3) extended to a self-supervised virtual-view training scheme.
Through this design, MASSAR achieves improved performance in both novel-pose animation and novel-view rendering.

### Strengths
1.The motion-aware surface smoothing term is simple yet effective in enforcing geometric regularization, addressing a known limitation of 3D Gaussian Splatting (3DGS) in sparse-view or monocular settings.

2.The proposed smoothness weight computation in canonical space (Eq. 7) is elegant and interpretable, as it considers the geometric relationships of nearby pixels in both canonical and observation spaces.

### Weaknesses
1.The main contribution—smoothness-based regularization—while effective, is a incremental improvment. compare to original depth smooth regularization, the improvement is limit.


2.While ablations show clear improvements, it would be more convincing to include quantitative sensitivity analysis for hyperparameters such as $\sigma_{s} $ and
$\sigma_{c}$

3.The runtime and memory usage of MASSAR are not reported compared to baseline. 

4.There is no visualization result without $\{w_{i}\}$ and with $\{w_{i}\}$.

### Questions
1.Since different scenes may exhibit varying levels of surface smoothness, how sensitive is the method to this hyperparameter? Is  $\sigma_{s}$ fixed across all datasets, or is it adaptively tuned per case?

2.Could the proposed smoothness regularization be integrated into other Gaussian-based frameworks (e.g., 2DGS), given that some of them already include surface regularization terms? Would such integration require any modification, or is it directly compatible?

### Soundness
2

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
4

### Summary
This paper tackles the task of avatar reconstruction from a monocular video. The paper follows the strategy of canonical space + deformation (linear blend skinning). To improve the quality, the authors propose to enhance surface smoothness. Specifically, the method encourages 3DGS that are close in the observation space to also be close in the canonical space. Further, it utilizes normal map alignment with priors from foundation models. Experiments on various datasets demonstrate the effectiveness of the proposed approach.

### Strengths
- originality-wise: the surface smoothing idea is interesting.
- quality-wise: the final results look promising.
- clarity-wise: the paper is generally well-written.
- significance-wise: reconstructing a human avatar from a monocular video is important for various downstream tasks, e.g., AR/VR.

### Weaknesses
1. I am not convinced that the surface smoothing design (Sec. 3.3) contributes most to the model. Actually, when it comes to LPIPS, from Tab. 1, the most important factor is the utilization of Sapiens to compute surface normal (L367).

2. Further, qualitatively, in the ablation (Fig. 5), without the Sapiens' prior, the model produces the lowest quality results, much worse than any other ablations.

3. Eq. (7) encourages close points in the observation space to be close in the canonical space, i.e., $\hat{d}_j = (\tilde{\mu}_j^c - \tilde{\mu}_i^c)$. I am curious whether this is a good signal, as close points in observation space can be really far away in the canonical space. For example, the hand will be close to the feet in the observation space if a person places their hands on their feet. However, in a T-shape, the hand will be really far from the feet.

### Questions
See "weakness".

### Soundness
2

### Presentation
3

### Contribution
2

# Pixel3DMM: Versatile Screen-Space Priors for Single-Image 3D Face Reconstruction

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 6

## Abstract
We address the 3D reconstruction of human faces from a single RGB image. To this end, we propose Pixel3DMM, a set of highly-generalized vision transformers which predict per-pixel geometric cues in order to constrain the optimization of a 3D morphable face model (3DMM). We exploit the latent features of the DINO foundation model, and introduce a tailored surface normal and uv-coordinate prediction head. We train our model by registering three high-quality 3D face datasets against the FLAME mesh topology, which results in a total of over 1,000 identities and 976K images. For 3D face reconstruction, we propose a FLAME fitting opitmization that solves for the 3DMM parameters from the uv-coordinate and normal estimates. To evaluate our method, we introduce a new benchmark for single-image face reconstruction, which features high diversity facial expressions, viewing angles, and ethnicities. Crucially, our benchmark is the first to evaluate both posed and neutral facial geometry. Ultimately, our method outperforms the state-of-the-art (SoTA) by over 15\% in terms of geometric accuracy for posed facial expressions.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces a hybrid method for single-image 3D face reconstruction, named Pixel3DMM, which leverages a powerful prior model to predict normals and UV coordinates as supervisory signals, guiding the optimization of the FLAME mesh through test-time optimization. The approach utilizes two screen-space priors—surface normals and UV correspondences—predicted via a customized ViT architecture. Despite training on a moderately sized dataset, it achieves competitive accuracy with reduced training resources compared to prior methods. The estimated priors are used to fit the FLAME model, delivering strong performance on a newly proposed benchmark, particularly excelling in expression disentanglement evaluation based on the NeRSemble dataset.

### Strengths
* Employs a lightweight ViT-based approach for face normal estimation using limited data, providing a reproducible alternative to complex methods.
* Enhances model robustness by carefully processing a large-scale multi-view dataset and applying IC-light-based data augmentation to account for lighting variations.
* Innovatively decomposes FLAME parameter recovery into an image translation problem and dense keypoint optimization, yielding strong experimental performance.

### Weaknesses
* The paper's core techniques, including the use of ViT for predicting screen-space attributes and the FLAME fitting process, show minimal novelty, as similar approaches have been extensively explored in prior research with only minor architectural adjustments.
* While the method outperforms baselines on the new benchmark, it falls short of state-of-the-art methods like FlowFace and TokenFace on established benchmarks (e.g., NoW, FaceScape), indicating a limited competitive advantage.

### Questions
My core concerns regarding this paper lie in its limited innovation and relatively minor improvements, as highlighted in the Weaknesses section. Should the authors identify any aspects overlooked in my evaluation of innovation and performance enhancements, I encourage them to point them out. Such clarifications may potentially lead to a revision of my overall assessment.

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
This paper presents Pixel3DMM, a method for 3D face reconstruction from single RGB images. The idea lies somewhere between conventional parameter regression pure 3DMM approaches and screenspace facial normal prediction. The approach trains two Vision Transformers (ViTs) built on DINOv2 features to predict per-pixel surface normals and UV coordinates, which are then used to constrain FLAME 3DMM fitting optimisation. The authors also introduce a new benchmark for evaluating both posed and neutral facial geometry reconstruction.

### Strengths
1. The overall idea is simple but effective - this is appealing.
2. The method achieves significant improvements over state-of-the-art, particularly on posed expressions.
3. The paper introduces the first benchmark that jointly evaluates both posed and neutral facial geometry, addressing an important gap in the field. The benchmark includes diverse, extreme expressions from NeRSemble.
4. Training requires only 2 GPUs for 3 days using publicly available data, making the work reproducible and accessible to the research community.
5. The paper includes extensive ablations, comparisons on multiple benchmarks (NoW, FaceScape, plus their own), and evaluations of the normal estimation component.
6. Qualitative results (Fig. 1, Fig. 8) demonstrate robust performance on challenging in-the-wild images with occlusions, lighting variations, and diverse appearances.

### Weaknesses
1. The method primarily fine-tunes DINOv2 with a simple prediction head (4 transformer blocks + 3 up-convolutions). So there is limited novelty in the architecture or set up itself.
2. Despite strong posed reconstruction, the method only marginally improves over MICA for neutral faces.
3. The paper lacks discussion of when and why the method fails. What types of expressions or conditions are most challenging? The qualitative comparisons show strong results, but no failure cases are presented.
4. The method critically depends on MICA's identity predictions (Table 3: "no MICA" ablation shows significant degradation). This is a strong assumption that limits the method's independence and could propagate MICA's biases or errors.
5. The use of IC-Light for lighting augmentation is neat but not thoroughly evaluated. How much does this contribute to robustness? An ablation would be valuable.
6. The fitting takes 30 seconds in an "unoptimised implementation." How does this compare to baselines? Real-time performance matters for many applications.
7. In the UV coordinate loss design (Eq. 6) the nearest neighbour lookup seems clumsy to me. Can't you use barycentric interpolation from the per-pixel UVs to interpolate a vertex position?
8. Simple L1 difference in normal space. Why not use cosine similarity or angular error, which are more standard for normal estimation?
9. The paper states all datasets are registered to FLAME topology using NPHM's procedure, but doesn't discuss registration quality or errors this might introduce.

### Questions
Besides responding to the above listed weaknesses, some additional questions are:

Can you provide quantitative analysis of where the method fails? What percentage of benchmark images have errors above certain thresholds?
Have you explored learning identity/expression disentanglement more explicitly, rather than relying on MICA?
What is the actual runtime comparison with baselines in a fair setting (same hardware)?

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
The paper proposes Pixel3DMM for 3D face reconstruction from a single image. It first predicts pixel-aligned geometric priors by training two foundation models to predict per-pixel surface normals and UV coordinates separately. Then it uses them as supervision in an optimization-based FLAME fitting process for reconstruction. To get the training data, the authors register three public 3D facial datasets (FaceScape, NPHM, Ava256) to NPHM along with lighting augmentation to get the image with corresponding normal and UV groundtruth. Moreover, they introduce a new benchmark from the NeRSemble dataset for both posed and neutral face reconstruction evaluation. Experiment results show that the model outperforms SOTA methods on reconstruction and normal prediction, and demonstrates solid generalization to in-the-wild images.

### Strengths
The idea of using pixel-level geometry prior for 3D reconstruction supervision is sound to me. This paper combines the geometric prior of foundation model with FLAME optimization, showing an interesting direction to improve 3DMM-fitting robustness. Another strength is that it only requires 2 48G GPUs to train, making it computationally accessible. Quantitative and qualitative comparisons with previous methods show better performance across multiple benchmarks with large expression and poses. Moreover, the model is robust to in-the-wild examples and video tracking. The paper is overall easy to read and provides enough details for data processing, model architecture and training, which contributes to reproducibility.

### Weaknesses
1. The proposed 3D reconstruction method takes two steps and requires two networks to predict normal and UV coordinates separately, which is relatively complex compared with previous feed-forward or optimization-only methods. 
2. The reconstruction still relies on FLAME parameters, which have limited representation capacity, so the method cannot reconstruct fine-grained details beyond 3DMM space. Also the paper uses NPHM to get uniform topology for supervision, which brings error.
3. The ablation study shows that MICA identity initialization plays a significant role in performance, which brings questions about the importance of the predicted priors versus inherited identity cues from MICA.
4. The lack of qualitative ablation results, which would help the understanding of the benefit for each component.

### Questions
Beyond the concerns listed in the weakness section, I have a few questions. First is the choice of the normal map and UV-coordinate for geometry cues. Have you considered other 3D representations like depth map or point map? Another question is why not performing FLAME fitting for all the training data and then train a feedforward prediction network to regress the parameters? Would it lead to degraded generalization or accuracy?

### Soundness
3

### Presentation
3

### Contribution
3

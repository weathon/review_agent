# LVSPM: Long Sequence View Synthesis and Pose Estimation Model

- Decision: Reject
- Scores: 4, 4, 2, 6

## Abstract
We present LVSPM, a generalizable model that jointly estimates camera poses and synthesizes novel views from uncalibrated image collections. Unlike prior approaches that rely on dense geometric supervision, LVSPM is trained only with RGB images and pose supervision, avoiding the need for dense 3D ground truth. LVSPM employs test-time training (TTT) layers, enabling efficient compression of tokens into fixed-size hidden states and scaling seamlessly to hundreds of input views. 
Experiments on  RealEstate10k, Co3Dv2 and DL3DV, LVSPM surpasses VGGT in pose estimation across 10–256 input views. For novel view synthesis, LVSPM achieves state-of-the-art results in pose-free long-sequence rendering of the large baseline dataset DL3DV, and even exceeds pose-dependent models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents LVSPM, a transformer-based model for long-sequence pose-free novel view synthesis (NVS) and camera pose estimation.
Unlike prior approaches that require dense 3D supervision (e.g., depth or point maps), LVSPM only uses RGB and pose supervision.
It extends the LaCT (Large-Chunk Test-Time Training) architecture to handle hundreds of unposed input views efficiently.
The model achieves state-of-the-art performance on several datasets such as DL3DV-10K, RealEstate10k, and Co3Dv2, outperforming VGGT in pose estimation and AnySplat in NVS.

### Strengths
A unified framework that jointly performs pose estimation and pose-free NVS at scale.

Can handle up to 256 unposed input images efficiently via the LaCT backbone.

Outperforms VGGT in pose estimation and AnySplat/DepthSplat in view synthesis across multiple datasets.

### Weaknesses
- Limited Generalization Evaluation


The experiments are mostly conducted on in-domain datasets, especially DL3DV-10K.

At this scale of training (64×H100 GPUs), more zero-shot evaluations are expected to demonstrate generalization.


- Sparse-View Unexplored


The paper evaluates primarily on semi-dense inputs (6–128 views).

It remains unclear how LVSPM performs under very sparse conditions (1–6 views), which are critical for real-world deployment and fairness against other baselines, e.g. the sparse views are supported in VGGT.


- Inference Efficiency Analysis Missing

Since LaCT is the core contribution, the paper should include inference runtime scaling with the number of views (e.g., 16→256).



- RGB v.s. Geometry Scaling

Without need of geometry labels, e.g. depth, the propsoed method could scale more than baselines VGGT, how much benefit will gain from this? How's the scaling curve of the proposed method w.r.t the data scale.

### Questions
Please see the weakness section.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work extends Test-Time Training layers to do novel view synthesis without known input camera parameters. The plucker map of the conditional input views is removed. Instead, like VGGT, a camera token is attached to finally predict the camera pose of the corresponding views. To model is trained to map a given plucker map to their corresponding novel-view views. The ground truth camera poses are normalized in VGGT manner so the target-view poses can be determined for training. The effectiveness is verified on pose estimation and novel-view synthesis tasks.

### Strengths
The proposed method is a straightforward extension to the recent strong TTT models for unposed scenarios. Every module makes sense. The improvement is good comparing to the other recent works under the same unposed setup.

### Weaknesses
Technical contribution is limited. It's a simple modification to the original TTT by removing input pose condition and attaching VGGT-like camera token to predicting camera parameters. The new knowledge it brings is limited.

The reason why a synthetic dataset pertaining is needed unclear to me. The ground-truth for training is poses and images where the poses for real-world data are quite accurate already. However, the ablation says synthetic pertaining is needed. Discussion and analysis for this is needed.

### Questions
Minor comment. Some of the subscript/superscript in the equations can be skip for brevity. The current equations is somewhat hard to read.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper presents LVSPM, a feedforward model that simultaneously performs novel view synthesis and camera pose estimation. It extends the implicit novel view synthesis models LVSM and LaCT by removing the pose embedding from the input and adding an additional pose estimation head to predict camera poses. LVSPM supports a variable number of input views. Experimental results demonstrate that the proposed method achieves state-of-the-art performance in both pose estimation and novel view synthesis.

### Strengths
- The proposed method achieves SOTA performance in both pose estimation and novel view synthesis.

- The paper is well written, clear, and easy to follow.

### Weaknesses
- The model uses an input resolution of 512 × 448, which differs from that used in previous pose estimation and novel view synthesis models such as VGGT, AnySplat, DepthSplat, and NoPoSplat. How does the proposed method compare against these methods?

- The paper lacks details about the evaluation setup. For novel view synthesis on DL3DV and Tanks and Temples, how are the input views sampled?

- The novelty of the paper is limited. It mainly combines the LaCT version of LVSM with VGGT. Since LaCT has already shown strong performance in novel view synthesis and VGGT in pose estimation, the paper provides little new insight.

- The claim that the proposed method outperforms DepthSplat is not well supported. DepthSplat was trained using a fixed 6-view input, which explains why the proposed method performs better when more views are provided. A comparison with DepthSplat under the same 6-view setting is missing.

- The paper does not evaluate novel view synthesis on the RealEstate10K dataset, which is widely used in this field.

### Questions
- How does the pose estimation performance compare with recent SOTA methods such as $\pi^3$ (not affect the rating, but it will be good to provide a comparison)?

- How does the model perform when the number of input views is small, e.g., only two views?

- How is the inference time compared with baseline methods?

### Soundness
1

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces LVSPM, a feed-forward model that jointly estimates camera poses and synthesizes novel views from unposed, long multi-view sequences using only RGB and pose supervision. It frames both tasks as sequence-to-sequence token prediction with minimal 3D inductive bias, employing a LaCT backbone with large-chunk test-time training to scale to 100–256+ input views, lightweight per-view camera tokens for pose regression, and Plücker-ray tokens for target views; simple MLP heads decode camera parameters and RGB. Empirically, LVSPM delivers strong pose estimation (often surpassing VGGT, especially as views increase) and state-of-the-art pose-free long-sequence NVS on DL3DV-10K, outperforming AnySplat and even pose-dependent DepthSplat in many settings; ablations indicate the RGB loss aids pose accuracy, supporting the claim that dense 3D labels are unnecessary.

### Strengths
- The work is original in unifying pose-free long-sequence NVS and camera estimation within a minimal-bias token-based model, creatively combining LaCT (large-chunk TTT), lightweight camera tokens, and Plücker-conditioned targets to scale to 100–256+ views without dense 3D labels.  
- The empirical quality is strong: evaluations across RE10k, Co3Dv2, and DL3DV-10K show competitive-to-better pose AUC than VGGT as views increase, and clear SOTA pose-free NVS on DL3DV-10K; ablations substantiate design choices (e.g., RGB loss aiding pose, synthetic pretraining benefits).  
- The paper is generally clear, with a coherent sequence-to-sequence formulation, explicit tokenization/prediction heads, and transparent training/evaluation details that support reproducibility; figures/tables aid understanding of pose alignment and rendering fidelity.

### Weaknesses
- Limited analysis of TTT dynamics and stability  
The LaCT/TTT layer is central to scaling, yet the paper lacks ablations on chunk size, number of TTT layers updated, adaptation step size $\eta$, and update frequency. Providing sensitivity curves (pose AUC and PSNR/LPIPS vs. chunk length and $\eta$) and a runtime/memory breakdown could help clarify the adaptation cost-benefit trade-offs.

- Ambiguity in supervision claims and fairness vs. VGGT  
The paper states “only RGB and pose supervision” yet compares pose accuracy to VGGT trained with dense 3D labels. Clarify whether LVSPM uses any geometric priors (e.g., Plücker rays for targets, canonicalization, scale normalization) and align the comparison by adding a version of LVSPM with depth/point-map auxiliary losses to quantify the benefit gap. This will make the “cheaper supervision” claim more rigorous.

- Missing NVS results on RealEstate10K  
What's the NVS performance on RealEstate10K? Only pose estimation results are reported on this dataset, making the zero-shot generalization capability of LVSPM on the NVS task unclear.

- Limited diagnosis of failure modes  
Figures focus on successes; please include analyses where LVSPM fails (e.g., large specularities, low-texture walls, extreme baselines), which help readers understand its limitations.

### Questions
- What is the precise role of Plücker conditioning in a pose-free pipeline? Target views are encoded via Plücker rays computed from (E_t, K_t). At inference, are these target cameras given or sampled? If given, does this effectively assume known target intrinsics/poses for rendering, and how are they obtained in real scenarios? If sampled, what sampling strategy is used, and how sensitive is NVS quality to errors in target K/E?  
- The LaCT TTT mechanism is pivotal. Could you report sensitivity to chunk size, learning rate η, number of TTT-updated layers, and update frequency?
- Please add a detailed breakdown of inference time, peak memory, and token counts vs. number of input views and target views, with and without TTT. This will help assess practicality for real deployments.

### Soundness
3

### Presentation
3

### Contribution
3

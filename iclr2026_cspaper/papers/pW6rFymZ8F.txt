000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053

# Embodiedmae: A Unified 3D Multi-Modal Rep- Resentation For Robot Manipulation

Anonymous authors Paper under double-blind review

## Abstract

We present EmbodiedMAE, a unified 3D multi-modal representation for robot manipulation. Current approaches suffer from significant domain gaps between training datasets and robot manipulation tasks, while also lacking model architectures that can effectively incorporate 3D information. To overcome these limitations, we enhance the DROID dataset with high-quality depth maps and point clouds, constructing DROID-3D as a valuable supplement for 3D embodied vision research. Then we develop EmbodiedMAE, a multi-modal masked autoencoder that simultaneously learns representations across RGB, depth, and point cloud modalities through stochastic masking and cross-modal fusion. Trained on DROID-3D, EmbodiedMAE consistently outperforms state-of-the-art vision foundation models (VFMs) in both training efficiency and final performance across 70 simulation tasks and 20 real-world robot manipulation tasks on two robot platforms. The model exhibits strong scaling behavior with size and promotes effective policy learning from 3D inputs. Experimental results establish EmbodiedMAE as a reliable unified 3D multi-modal VFM for embodied AI systems, particularly in precise tabletop manipulation settings where spatial perception is critical.

## 1 Introduction

Pre-trained Vision Foundation Models (VFMs) have made remarkable progress in visual understanding (Caron et al., 2021; Oquab et al., 2024; He et al., 2022; Zhai et al., 2023; Nair et al., 2022; Majumdar et al., 2023; Bachmann et al., 2022; Zhu et al., 2025), becoming essential components for embodied AI systems (Octo Model Team et al., 2024; Kim et al., 2024; Black et al., 2024; Liu et al., 2025; Ze et al., 2024; Chi et al., 2023; Li et al., 2025b). As research demonstrates that 3D spatial understanding can improve robot manipulation capabilities (Ze et al., 2024; Ke et al., 2024; Li et al., 2025a; Zhen et al., 2024), the demand for effective 3D VFMs has grown. 3D information provides critical spatial context, enabling robots to localize targets, avoid collisions, and execute complex manipulations. Despite this increasing need, existing models fall short of meeting requirements. There are two primary reasons behind the lack of suitable 3D VFMs for embodied AI. First, a significant domain gap exists in training data. Mainstream 3D VFMs are trained on outdoor or indoor static scenario datasets (Huang et al., 2023; Zhu et al., 2023; Qian et al., 2022; Yang et al., 2024a;b). These models operate at spatial scales incompatible with tabletop manipulation, resulting in a weak understanding of robot-object interactions (Ze et al., 2024). While training 3D embodied-specific VFMs from scratch on robot manipulation datasets seems promising, these efforts are hampered by extremely limited training data (Zhu et al., 2025; Qu et al., 2025; Vuong et al., 2023). *Second,*
there is a lack of efficient and scalable model architectures for 3D perception. Simply integrating 3D information without careful design often degrades robot operation capabilities rather than enhancing. For example, many advanced 3D VFM architectures demonstrate unexpectedly poor performance in policy learning, sometimes even underperforming simple MLPs (Ze et al., 2024; Zhu et al., 2024).

To address these challenges, we propose EmbodiedMAE, a unified 3D multi-modal representation learning framework specifically designed for embodied AI. We first enhance the original DROID
dataset (Khazatsky et al., 2024) by extracting high-quality metric depth maps and point clouds for each frame using ZED SDK temporal fusion and AI-augmented enhancement. This creates DROID-3D, a large-scale 3D robot manipulation dataset containing 76K trajectories (350 hours) of high-fidelity interaction data. This dataset provides the scale and quality needed for effective pre-training while maintaining domain compatibility with manipulation tasks. We then develop a 1 054

![1_image_0.png](1_image_0.png) 055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107 Figure 1: **Overview of EmbodiedMAE Pre-training.** We pre-train a ViT-Giant scale multi-modal MAE on the large-scale DROID-3D robot manipulation dataset. We fix the total number of unmasked patches across RGB, depth, and point cloud modalities. The mask ratio allocated to each modality is stochastically sampled. After the Giant model pre-training, we distill it to obtain our Small/Base/Large scale models.

multi-modal masked autoencoder that simultaneously learns representations across RGB images, depth maps, and point clouds through stochastic masking and cross-modal fusion. By masking different proportions of each modality and using explicit modal fusion in the decoder, our model learns to infer across modalities, developing powerful spatial perception capabilities and object-level semantic understanding (Figure 3). To thoroughly validate our representation model, we conduct extensive evaluations across diverse settings: 40 tasks from the LIBERO benchmark (Liu et al., 2023) and 30 from the MetaWorld benchmark (Yu et al., 2019), 10 real-world tasks on the low-cost open-source SO100 robot (Cadene et al., 2024), and 10 on the high-performance xArm robot. We use a scaled-down RDT (Liu et al., 2025) model as the policy backbone to simulate the performance of VFMs in advanced VLA training, and compare EmbodiedMAE against various categories of state-of-the-art (SOTA) VFMs, including vision-centric models, language-augmented models, embodied-specific models, and 3D-aware models.

Our experiments demonstrate that EmbodiedMAE consistently outperforms all baseline VFMs in both training efficiency and final performance, exhibits strong scaling behavior with model size, and effectively promotes policy from 3D input. These findings establish EmbodiedMAE as a reliable foundation model for embodied AI applications requiring robust 3D visual understanding.

Our contributions can be summarized as follows:
- We present EmbodiedMAE, a unified 3D multi-modal representation learning framework for embodied AI that effectively integrates RGB, depth, and point cloud modalities. It achieves SOTA performance in both RGB-only and multi-modal settings while maintaining computational efficiency and scaling properties.

- We introduce DROID-3D, a high-quality, large-scale DROID supplement containing 76K
trajectories (350 hours) of robot data with synchronized RGB, depth maps, and point clouds.

Unlike previous works that process subsets or use low-quality estimated depth, we provide temporally consistent depth by ZED SDK, creating a valuable resource for 3D robot learning.

- We establish comprehensive evaluation benchmarks for embodied representation learning across diverse settings: simulation tasks from LIBERO and MetaWorld, real-world tasks on a low-cost open-source robot (SO100), and tasks on a high-performance robot (xArm). Our results demonstrate consistent performance improvements across these varied platforms, validating the model's generalization capabilities.

## 2 Methodology 2.1 3D Data Collection

Effective pre-training of our model necessitates a large-scale 3D robot manipulation dataset. We conduct a systematic evaluation of depth data quality across several mainstream large-scale embodied AI datasets, primarily including BridgeDataV2 (Walke et al., 2023), RH20T (Fang et al., 2023), and DROID (Khazatsky et al., 2024), as illustrated in Figure 2. We find significant limitations in existing datasets: BridgeDataV2 contains only 13% data with 3D information, with available depth maps being of insufficient quality; RH20T exhibits similar issues with unreliable and noisy depth data; while DROID includes stereo image recordings but lacks readily usable 3D annotations. Several previous approaches attempted to address this by estimating depth from 2D images using AI models. For instance, SPA (Zhu et al., 2025) employs CrocoV2-Stereo (Weinzaepfel et al., 2023) to estimate depth for approximately 1/15 of the DROID dataset. We observe that such methods lack precision and temporal consistency, making them unable to accurately capture fine-grained details during robot-object interactions, which are essential for manipulation tasks. To overcome these challenges, we use ZED SDK to extract the recording files in the raw DROID dataset. The ZED SDK integrates multiple techniques that significantly improve depth quality, including temporal fusion to reduce noise and increase consistency, AI-augmented enhancement to refine stereo matching in textureless regions, and hardware-calibrated metric depth to provide accurate absolute distance measurements. With these high-quality depth maps, we further extract point clouds with the camera's intrinsic matrix. We apply farthest point sampling (FPS) to downsample them to 8,192 points, striking a balance between computational efficiency and geometric fidelity. Unlike SPA's approach of processing only a subset of the DROID dataset, we process the complete collection of 76K trajectories (350 hours of interaction data), requiring nearly 500 hours of processing time. Due to these significant improvements in data quality and coverage, we construct DROID-3D as a supplementary resource to the original DROID dataset. We believe it will serve as a valuable resource for pre-training 3D VLA models and foster innovative research in embodied AI, particularly for applications requiring precise spatial understanding for manipulation tasks.

108

![2_image_0.png](2_image_0.png) 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161

## 2.2 Multi-Modal Encoder

EmbodiedMAE processes three modalities commonly used in robot perception: RGB images, depth maps, and point clouds. Given the robot observation of RGB image I ∈ R
3×H×W , depth D ∈ R
1×H×W , and point cloud P ∈ RM×3, we first use modal-specific patchifiers to project them into patches ¯I, D, ¯ P¯ ∈ R
L×C . Then we draw a random binary mask for each modality mI , mD, mP ∈ {0, 1}
L, and obtain two complementary masked views I1 = ¯I[mI ], I2 = ¯I[1 − mI ],
similar for D and P. We use a Vision Transformer (ViT) f to process the unmasked patches and obtain the joint representation h = f(I1, D1, P1). Masking Strategies. Effective masked autoencoding requires masking a large portion of input tokens during training, and the specific masking strategy has a significant impact on learned representations (Bachmann et al., 2022; He et al., 2022). Following Bachmann et al. (2022), we fix the total number of unmasked patches across all modalities, i.e., the number of ones in (mI , mD, mP ) is fixed, and allocate them according to a symmetric Dirichlet distribution: (λI , λD, λP ) ∼ Dir(α), where λI + λD + λP = 1 and each λ ≥ 0. The concentration parameter α controls the diversity of masking proportions. When α = 1, the distribution is uniform over the simplex, assigning equal likelihood to all valid combinations. Lower values (α ≪ 1) tend to concentrate sampling on a single modality, while higher values (α ≫ 1) produce more balanced allocations across modalities. We intentionally avoid introducing any modality bias by keeping the distribution symmetric, aiming to maintain flexibility for a variety of downstream tasks and input configurations.

162

![3_image_0.png](3_image_0.png) 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 Modal Patchifiers. For RGB and depth maps, we break them into 16×16-size patches, i.e., L =
H·W
162 ,
and we incorporate 2D sine-cosine positional embeddings after a linear projection (Dosovitskiy et al.,
2021; Touvron et al., 2021). For point clouds, we apply Farthest Point Sampling (FPS) to select N cluster centers, and then use K-Nearest Neighbors (KNN) to group each center with its K nearest neighbors, forming N point groups of K + 1 points each, i.e., L = N. Each group is normalized and encoded using a DP3 encoder (Ze et al., 2024) to generate token embeddings, while each group center is processed by an MLP to create positional embeddings (Pang et al., 2022). We omit explicit modality-type embeddings, as the bias term in each projection layer implicitly encodes modality-specific information. These tokens are masked, concatenated, and passed to the ViT encoder. Transformer Encoder. We implement the same ViT structure as DINOv2 (Oquab et al., 2024), with the exception of removing the [CLS] token. This design choice allows us to initialize the ViT directly from DINOv2 pre-trained weights, thereby enhancing its general capabilities.

## 2.3 Multi-Modal Decoder

The decoder is only used during EmbodiedMAE training, where it reconstructs the masked portions of each modality based on the visible tokens and learned [MASK] tokens.

Specifically, the decoder employs cross-attention to enable explicit fusion across modalities. Visible tokens from each modality are projected, concatenated with [MASK] tokens, and then augmented with positional embeddings to form the query sequence. Meanwhile, all visible patches are projected and enhanced with modality encodings to form the key and value sequences. The fused features are then fed into a smaller, modality-shared ViT decoder to produce the final hidden states. Modalityspecific MLP heads generate the reconstruction outputs: masked RGB and depth patches, and normalized point coordinates for point cloud groups. Suppose that (hI , hD, hP ) = f(I1, D1, P1) are modality representations, the decoder outputs can be expressed as gI (hI , h), gD(hD, h), and gP (hP , h), corresponding to each modality. Notably, our design shares transformer components across modalities, reducing computational cost by approximately a factor of three. We adopt a simple mean square error (MSE) loss:

$$\mathcal{L}_{\text{MAE}}=\mathbb{E}_{(I,D,P)\sim\mathcal{D},\text{Dst}(\alpha)}\left[\underbrace{\left\|g(h_{I},h)-I_{2}\right\|^{2}}_{\text{RGB}}+\underbrace{\left\|g(h_{D},h)-D_{2}\right\|^{2}}_{\text{Dst}}+\underbrace{\left\|g(h_{P},h)-P_{2}\right\|^{2}}_{\text{RdB}}\right],\tag{1}$$

where the decoder outputs gI (hI , h), gD(hD, h) are l2-normalized, and gP (hP , h) is group centernormalized, following that normalized targets yield better performance (He et al., 2022).

## 2.4 Model Distillation

Following Oquab et al. (2024), we first train a ViT-Giant EmbodiedMAE model from scratch on the DROID-3D dataset, then distill it into Small, Base, and Large variants. Both teacher and student models receive identical masked inputs (I1, D1, P1), with the teacher model kept entirely frozen.

Rather than simply copy the final outputs, we apply feature-level supervision at strategically selected 216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 network depths to ensure comprehensive knowledge transfer. Specifically, we align features at three critical positions in the network hierarchy: (Bottom) immediately after the modal patchifiers to capture low-level perceptual features, (Top) at the final hidden layer to preserve high-level semantic understanding, and (Middle) at a middle layer positioned at 3/4 of the encoder depth to transfer intermediate representations (Bai et al., 2023) (For example, when distilling from a 24-layer ViT-L teacher to a 12-layer ViT-B student, the 9th layer of the student aligns with the 18th layer of the teacher.). We adopt trainable linear projections before computing alignment losses to accommodate dimensional differences between teacher and student features. Formally, we denote the feature alignment pairs (y j, hj) ∈ A, where y jand h jrepresent the j-th pair of hidden states from teacher and student models, respectively, and l jis the linear projector. The feature alignment loss is:

$${\mathcal{L}}_{\mathrm{Align}}=\sum_{(y^{j},h^{j})\in A}\mathrm{SmoothL1}\left(y^{j},l^{j}(h^{j})\right).$$
$${\mathcal{L}}_{\mathrm{Disstill}}={\mathcal{L}}_{\mathrm{MAE}}+\beta\cdot{\mathcal{L}}_{\mathrm{Align}},$$
j). (2)
We train student models by jointly optimizing the standard multi-modal MAE reconstruction loss and the feature alignment loss (Figure 1, Distillation part):
LDistill = LMAE + β · LAlign, (3)
where β > 0 controls the balance between mask autoencoding and feature alignment. This approach enables our smaller models to achieve performance closer to the Giant model while maintaining computational efficiency, making them practical in resource-constrained robotics applications.

## 2.5 Put All Together

Building on our architectural design described above, we first pre-train the Giant-scale model and subsequently distill it into more computationally efficient Small, Base, and Large variants on the DROID-3D dataset. We employ AdamW optimizer with a weight decay of 0.01. The base learning rate is set at 1.5e-4, incorporating an initial warmup period followed by a cosine schedule decay. We apply a 0.1 gradient norm clip to stabilize training. All computational workflows utilize bfloat16 precision, which substantially reduces memory requirements and computational costs while maintaining numerical stability. During the pre-training phase, we maintain 96 unmasked patches across all modalities, representing approximately 1/6 of the total patch count. For the distillation phase, we further reduce the number of unmasked patches to 60, approximately 1/10 of the total. This extremely aggressive masking approach significantly decreases training costs without compromising representational quality, as the student models benefit from the teacher's already robust understanding of multi-modal relationships. Our codebase follows Huggingface Transformers (Wolf et al., 2020) convention, making Embodied- MAE highly user-friendly. It ensures that researchers can easily incorporate our models into existing robotics pipelines with minimal adaptation effort. A simple usage example is illustrated in Figure 4.

$$(2)^{\frac{1}{2}}$$
$$({\mathfrak{I}})$$

Figure 4: **Usage Example.** We follow the Huggingface Transformers convention to make EmbodiedMAE highly user-friendly and easy to integrate.

## 3 Experiments

In this section, we present evaluation results of EmbodiedMAE across both simulation and real-world robotic manipulation tasks. Our experiments are designed to address three key research questions: (RQ1) Does EmbodiedMAE learn features that integrate information across different modalities? (RQ2) How does EmbodiedMAE perform compared to SOTA VFMs in robot manipulation tasks? (RQ3) Can EmbodiedMAE enable efficient robot learning in real-world environments for both low-cost and high-performance robot platforms? 3.1 EXPERIMENTAL SETUP Policy Network. To evaluate how effectively different VFMs support advanced VLA models, we adopt a compact RDT (Liu et al., 2025) (approximately 40M parameters) as our policy network. This architecture has demonstrated excellent scalability and strong performance in diffusion-based

| the average across all tasks rather than the three difficulty levels. Highest scores are emphasized with bold. MetaWorld R3M SigLIP DINOv2 SPA EmboodiedMAE DINOv2 EmbodiedMAE DP3 EmbodiedMAE Difficulty Level -RGB -RGB -RGB -RGB -RGB -RGBD -RGBD -PointCloud -PointCloud Easy (18) 74.1 76.4 79.8 80.9 81.8 61.9 85.2 79.2 79.8 Medium (9) 28.1 32.7 57.1 62.8 60.4 35.6 63.2 48.0 76.7 Very Hard (3) 49.8 14.0 56.4 55.8 57.8 65.6 61.6 38.7 68.7 Average 57.9 57.0 70.7 73.0 73.0 54.4 76.2 65.8 77.7   |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

policy learning. As shown in Figure 5, all baselines and EmbodiedMAE share the same architecture, ensuring fair comparison by isolating the visual representation component. See Section A.1 for more details. Baselines. For a comprehensive comparison, we benchmark against several SOTA VFMs with diverse design principles: DINOv2-Large (Oquab et al., 2024) (visioncentric), SigLIP-Large (Zhai et al., 2023) (languagecontrastive), R3M-Resnet50 (Nair et al., 2022), VC-1 (Majumdar et al., 2023), and SPA (Zhu et al., 2025) (embodiedspecific). SPA incorporates implicit 3D priors during training, making it particularly relevant for comparison with our multi-modal approach. Benchmarks. Our simulation evaluations are based on the LIBERO and MetaWorld benchmarks. LIBERO includes 40 tasks in four task suites: Goal, Spatial, *Object*, and Long. MetaWorld includes 30 tasks from various difficulty levels. For real-world experiments, we deploy the models on two robot platforms: The SO100 robot (low-cost, open-sourced, equipped with dual RGB cameras)
evaluated on 10 tasks in suites: Pick&Place, *MoveTo*, Wipe, and *Unfold*; The xArm robot (higherprecision, equipped with one Intel RealSense L515 LiDAR camera) evaluated on 10 tasks in suites:
Pick&Place, Pot, Pour, and *Moka*. We show detailed task configurations in Section A.2.

Figure 5: **Policy Network for All VFMs.**

![5_image_0.png](5_image_0.png)

We adopt a compact RDT as the policy network, in which only VFMs are modular.

## 3.2 Mae Predictions (Rq1)

270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323

## 3.3 Overall Comparison (Rq2)

In this section, we evaluate SOTA VFM baselines, EmbodiedMAE, and several its variants (in terms of model scale and input modality) on the LIBERO and MetaWorld benchmark. We report learning To assess the ability of EmbodiedMAE to integrate information across modalities, we design a series of controlled experiments probing its cross-modal fusion capabilities. Our evaluation focuses on three settings: (a) Extreme modality inference: We mask most patches from two modalities, leaving primarily one modality as the inference source (Figure 3, columns 1-9). (b) Cross-modal translation: We test the model's ability to predict one entire modality from another, specifically RGB from depth (column 10) and depth from RGB (column 11). (c) Re-coloring: We allow the model to see a deliberately altered RGB patch during depth-to-RGB prediction (column 12), where the color of the visible patch is modified to assess semantic understanding. Our results demonstrate that EmbodiedMAE effectively leverages available modalities to reconstruct missing information, suggesting strong cross-modal alignment. In column 10, the predicted RGB from depth lacks precise color information but maintains structural fidelity, indicating the model has learned to separate geometric and appearance features. Similarly, in column 11, depth predictions from RGB show smoothed object boundaries compared to ground truth, revealing a learned prior for depth continuity. Most notably, in the re-coloring setting (column 12), when injecting an altered RGB patch during depth-to-RGB reconstruction, only the corresponding object (table) adopts the modified color while surrounding elements (background, robot, cup) maintain their original appearance. This suggests EmbodiedMAE has implicitly learned object-level semantic segmentation and can propagate semantic information based on contextual cues, despite never being explicitly trained for segmentation.

These visualizations collectively demonstrate that EmbodiedMAE possesses strong multi-modal fusion capabilities, enabling it to enhance spatial understanding in 3D embodied perception tasks.

324

![6_image_0.png](6_image_0.png) 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377 curves on LIBERO in Figure 6 and success rate on MetaWorld in Section 3.3. Unless otherwise specified, "EmbodiedMAE" refers to the Large-scale, RGB-only variant.

Finding 1: **EmbodiedMAE consistently outperforms all baseline VFMs in terms of both training**
efficiency and final performance. Among the baselines, SPA and DINOv2 are the most competitive ones. SPA shows score gains on tasks where spatial understanding is crucial, e.g., LIBERO-Spatial and MetaWolrd, and performs comparably to DINOv2. The language-contrastive model, SigLIP, performs poorly across all embodied tasks, consistent with findings from Zhu et al. (2025). R3M and VC-1, although specifically designed for robot learning, do not demonstrate clear advantages.

Finding 2: **EmbodiedMAE exhibits strong scaling behavior with model size.** Performance improves monotonically as model capacity increases. Among all the variants, only the Small variant shows unstable performance on LIBERO-Goal and LIBERO-Object suites. The Base and Large models achieve similar performances, with the Large model slightly ahead. The Giant model consistently delivers superior performance, particularly in training efficiency. These results suggest EmbodiedMAE to be an effective training paradigm for scaling multi-modal representation learning.

Finding 3: **EmbodiedMAE promotes policy learning from 3D input.** When provided with RGBD inputs, EmbodiedMAE establishes a substantial performance gap over other baselines on both LIBERO and MetaWorld benchmarks. Remarkably, our Large-scale RGBD model even outperforms the Giant-scale RGB-only model on LIBERO-Goal and LIBERO-Object suites, and performs comparably on average across the LIBERO benchmark. In contrast, adding a trainable depth branch for DINOv2 (See Section A.3 for details of this variant) can degrade performance relative to RGB-only input, consistent with observations in Zhu et al. (2024). These findings establish EmbodiedMAE as a reliable VFM for scenarios requiring 3D visual understanding.

## 3.4 Real-World Experiments (Rq3)

To further assess generalization in practical settings, we conduct real-world evaluations on two robot platforms: the low-cost, open-source SO100 (Cadene et al., 2024) and the high-performance xArm. We show quantitative results in Figure 8, and rollout visualizations in Figure 7.

Finding 1: **EmbodiedMAE maintains SOTA performance in real-world robot manipulation.**
EmbodiedMAE consistently achieves SOTA performance across real-world manipulation tasks, particularly those requiring strong spatial understanding. With multi-modal inputs, EmbodiedMAE
further improves policy learning performance: EmbodiedMAE-RGBD and EmbodiedMAE-PC both surpass na¨ıve fusion baselines such as DINOv2-RGBD (Section A.3) and DP3 (Ze et al., 2024),
highlighting the effectiveness of our design in promoting robust 3D perception for real-world robotics.

Finding 2: **3D information plays a critical role in robot manipulation.** Incorporating 3D inputs significantly improves task success rates. We observe that most failures in baseline models stem from localization errors, causing grasp failures or collisions. EmbodiedMAE-RGBD, benefiting from enhanced spatial understanding, avoids these issues more reliably (see Figure 7). The choice of 3D
modality also matters. Although prior works (Li et al., 2025a; Ze et al., 2024; Zhu et al., 2024) have highlighted the compactness and training efficiency of point cloud (PC) representations, we find their practical effectiveness is hindered by sensor noise from object reflectivity and lighting variations. Consequently, PC-based policies even underperform RGB-only inputs. In contrast, the RGBD setting, where depth serves as an auxiliary cue, yields better performance and is more robust to noise. This suggests that effective post-processing of PCs is essential for leveraging them reliably.

## 3.5 Ablation Studies

378

![7_image_0.png](7_image_0.png) 379 380 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431 Due to the prohibitive cost of ViT-Giant pre-training, our ablation studies focus on model distillation insights. We evaluate masking ratio, feature alignment, and loss ratio on the LIBERO benchmark, reporting average success rates in Table 4, with default settings underlined. **(1) Masking Ratio**: Our default configuration sets 60 unmasked patches, approximately masking ratio of 90%. We test 70%, 80%, and 100% ratios (100% representing training with only feature alignment loss). Results indicate performance insensitivity to masking ratio, though ratios ¡100% perform better, suggesting feature alignment's predominant role while mask autoencoding provides additional benefits. (2) Feature Alignment: By default, we implement feature alignment at three positions (see Section 2.4).

Sequential removal of alignment points reveals diminishing impact from Top to Bottom, with each component contributing positively to model performance. **(3) Loss Ratio**: With default β = 1, we test β = 0.5/2.0/4.0. Results show performance robustness across β values, with slight degradation at β < 1.0, confirming feature alignment necessity, consistent with findings in (Bai et al., 2023). (4) Policy Model: While our primary policy focus is on diffusion-based models, we recognize the popularity of transformer-based models like the Action Chunking Transformer (ACT) (Fu et al.,

![7_image_1.png](7_image_1.png)

432 433 434 435 436 437 438 439 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485 2024). To confirm the generalizability of EmbodiedMAE's representations, we expand our evaluation to include the ACT on LIBERO-Goal (RGB and RGBD) and MetaWorld (PC) benchmarks.

| Table 2: Ablation study with ACT policy on LIBERO-Goal.                                                                                                                                                                                                                                     |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Policy Model EmbodiedMAE DINOv2 SPA EmbodiedMAE-RGBD DINOv2-RGBD ACT Policy 83.7 76.3 82.5 90.8 82.2 Table 3: Ablation study with ACT policy on MetaWorld. Policy Model + VFM Easy (18) Medium (9) Very Hard (3) ACT Policy + EmbodiedMAE-PC 80.0 64.4 56.2 ACT Policy + DP3 78.8 42.7 33.1 |

## 4 Related Works

Masking Ratio 0.7 0.8 0.9 1.0 92.2 91.2 92.4 90.1 Feature Alignment w/o Bottom w/o Middle w/o Top All 91.4 88.5 74.4 92.4 Loss Ratio β 0.5 1 2 4 90.8 92.4 91.1 92.2 Table 4: **Ablation study on LIBERO.** We conduct ablation experiments on masking ratio, feature alignment, and loss ratio on the LIBERO benchmark and report the average success rate.

Vision Foundation Models are models trained on large-scale data in a self-supervised or semisupervised manner that can be adapted for several other downstream tasks (Bommasani et al., 2022). Beyond conventional image classification, these models have shown strong transfer capabilities to tasks such as depth estimation (Yang et al., 2024a;b; Weinzaepfel et al., 2023), semantic segmentation, and robot control (Octo Model Team et al., 2024; Kim et al., 2024; Liu et al., 2025; Kim et al., 2025). Common pre-training techniques include contrastive learning (He et al., 2019; Chen et al., 2020; Chen* et al., 2021), masked autoencoding (Bai et al., 2023; Tong et al., 2022; Wang et al., 2023; Feichtenhofer et al., 2022; He et al., 2022), self-distillation (Oquab et al., 2024; Caron et al., 2021), and CLIP-style language-image contrastive learning (Zhai et al., 2023; Radford et al., 2021). VFMs greatly improve AI systems' visual understanding. Visual Representations for Embodied AI are crucial for enabling agents to perceive and interact with the physical world. Embodied perception must model robot-object interactions in dynamic environments, which general-purpose VFMs trained on static images often lack. Several recent methods have attempted to bridge this gap by training models directly on robot datasets. However, the limited scale and quality of embodied data hinder their generalization. These embodied-specific models often fail to generalize as well as VFMs trained on diverse in-the-wild datasets. As a result, many VLA models still rely on general-purpose VFMs like DINOv2 (Oquab et al., 2024; Kim et al., 2024; 2025) and SigLIP (Zhai et al., 2023; Liu et al., 2025; Kim et al., 2024) for better generalization, prompting the need for dedicated large-scale embodied VFM pretraining. 3D Robot Learning has proven effective in improving both embodied agents' training efficiency and policy performance (Ze et al., 2024; Li et al., 2025a; Zhu et al., 2024). Properly introducing 3D visual inputs often leads to better spatial understanding compared to RGB-only inputs. However, na¨ıvely incorporating 3D information, e.g., adding an extra depth channel, may severely degenerate the model's performance. Scalable native 3D multi-modal models remain largely absent in the current research landscape. EmbodiedMAE aims to address this gap by pre-training VFMs on large-scale, embodied-specific datasets to facilitate the development of scalable and effective 3D VLA models.

## 5 Conclusion, Limitations, And Future Works

In this work, we introduce EmbodiedMAE, a unified 3D multi-modal representation learning framework designed for embodied AI. We first construct DROID-3D, a high-quality, large-scale DROID supplement. Then we propose a multi-modal masked autoencoder architecture that fuses RGB, depth, and point cloud inputs through stochastic masking and cross-modal decoding. Trained on DROID-3D, our model, EmbodiedMAE, demonstrates superior spatial understanding, strong multi-modal fusion ability, and effective scaling behavior. It outperforms strong VFM baselines across 70 simulation tasks and 20 real-world tasks on two robot platforms (SO100 and xArm). We believe both the DROID-3D
dataset and EmbodiedMAE provide a valuable resource for 3D robot learning research. Despite the strong performance, EmbodiedMAE remains solely a vision backbone and does not natively support language instruction as input. A promising future direction is to fully leverage the language and action annotations available in the DROID-3D dataset to train a vision-language backbone, or even develop a multi-modal VLA model for instruction-following general embodied agents.

## 6 Ethics Statement

486 487 488 489 490 491 492 493 494 495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539 The authors have adhered to the ICLR Code of Ethics. This work does not involve human subjects, sensitive data, or raise any direct ethical concerns. All datasets used are publicly available.

## 7 Reprodicibility Statement

We are committed to ensuring the reproducibility of our research. Our source code will be made public upon publication. Detailed descriptions of our methods and model architectures are available in the section Section 2.5. All experimental settings, including datasets, training hyperparameters, and evaluation settings, are specified in the section Section 3 and Appendix.

## References

Roman Bachmann, David Mizrahi, Andrei Atanov, and Amir Zamir. MultiMAE: Multi-modal multi-task masked autoencoders. In *European Conference on Computer Vision, ECCV*, 2022.

Yutong Bai, Zeyu Wang, Junfei Xiao, Chen Wei, Huiyu Wang, Alan Yuille, Yuyin Zhou, and Cihang Xie. Masked autoencoders enable efficient knowledge distillers. In Conference on Computer Vision and Pattern Recognition, CVPR, 2023.

Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, Szymon Jakubczak, Tim Jones, Liyiming Ke, Sergey Levine, Adrian Li-Bell, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Lucy Xiaoyang Shi, James Tanner, Quan Vuong, Anna Walling, Haohuan Wang, and Ury Zhilinsky. π0: A
vision-language-action flow model for general robot control. *arXiv preprint arXiv:2410.24164*, 2024.

Rishi Bommasani, Drew A. Hudson, Ehsan Adeli, Russ Altman, Simran Arora, Sydney von Arx, Michael S. Bernstein, Jeannette Bohg, Antoine Bosselut, Emma Brunskill, Erik Brynjolfsson, Shyamal Buch, Dallas Card, Rodrigo Castellon, Niladri Chatterji, Annie Chen, Kathleen Creel, Jared Quincy Davis, Dora Demszky, Chris Donahue, Moussa Doumbouya, Esin Durmus, Stefano Ermon, John Etchemendy, Kawin Ethayarajh, Li Fei-Fei, Chelsea Finn, Trevor Gale, Lauren Gillespie, Karan Goel, Noah Goodman, Shelby Grossman, Neel Guha, Tatsunori Hashimoto, Peter Henderson, John Hewitt, Daniel E. Ho, Jenny Hong, Kyle Hsu, Jing Huang, Thomas Icard, Saahil Jain, Dan Jurafsky, Pratyusha Kalluri, Siddharth Karamcheti, Geoff Keeling, Fereshte Khani, Omar Khattab, Pang Wei Koh, Mark Krass, Ranjay Krishna, Rohith Kuditipudi, Ananya Kumar, Faisal Ladhak, Mina Lee, Tony Lee, Jure Leskovec, Isabelle Levent, Xiang Lisa Li, Xuechen Li, Tengyu Ma, Ali Malik, Christopher D. Manning, Suvir Mirchandani, Eric Mitchell, Zanele Munyikwa, Suraj Nair, Avanika Narayan, Deepak Narayanan, Ben Newman, Allen Nie, Juan Carlos Niebles, Hamed Nilforoshan, Julian Nyarko, Giray Ogut, Laurel Orr, Isabel Papadimitriou, Joon Sung Park, Chris Piech, Eva Portelance, Christopher Potts, Aditi Raghunathan, Rob Reich, Hongyu Ren, Frieda Rong, Yusuf Roohani, Camilo Ruiz, Jack Ryan, Christopher Re, Dorsa Sadigh, ´ Shiori Sagawa, Keshav Santhanam, Andy Shih, Krishnan Srinivasan, Alex Tamkin, Rohan Taori, Armin W. Thomas, Florian Tramer, Rose E. Wang, William Wang, Bohan Wu, Jiajun Wu, Yuhuai `
Wu, Sang Michael Xie, Michihiro Yasunaga, Jiaxuan You, Matei Zaharia, Michael Zhang, Tianyi Zhang, Xikun Zhang, Yuhui Zhang, Lucia Zheng, Kaitlyn Zhou, and Percy Liang. On the opportunities and risks of foundation models. In *arXiv preprint arXiv:2108.07258*, 2022.

Remi Cadene, Simon Alibert, Alexander Soare, Quentin Gallouedec, Adil Zouitine, and Thomas Wolf. Lerobot: State-of-the-art machine learning for real-world robotics in pytorch. https: //github.com/huggingface/lerobot, 2024.

Mathilde Caron, Hugo Touvron, Ishan Misra, Herve J ´ egou, Julien Mairal, Piotr Bojanowski, and ´
Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the International Conference on Computer Vision, ICCV, 2021.

Xinlei Chen, Haoqi Fan, Ross Girshick, and Kaiming He. Improved baselines with momentum contrastive learning. In *arXiv preprint arXiv:2003.04297*, 2020.

Xinlei Chen*, Saining Xie*, and Kaiming He. An empirical study of training self-supervised vision transformers. In *arXiv preprint arXiv:2104.02057*, 2021.

Cheng Chi, Siyuan Feng, Yilun Du, Zhenjia Xu, Eric Cousineau, Benjamin Burchfiel, and Shuran Song. Diffusion policy: Visuomotor policy learning via action diffusion. In *Proceedings of* Robotics: Science and Systems, RSS, 2023.

Spconv Contributors. Spconv: Spatially sparse convolution library. https://github.com/
traveller59/spconv, 2022.

Zibin Dong, Yifu Yuan, Jianye HAO, Fei Ni, Yi Ma, Pengyi Li, and YAN ZHENG. Cleandiffuser:
An easy-to-use modularized library for diffusion models in decision making. In The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track, NIPS, 2024.

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In *International Conference on Learning Representations, ICLR*, 2021.

Hao-Shu Fang, Hongjie Fang, Zhenyu Tang, Jirong Liu, Junbo Wang, Haoyi Zhu, and Cewu Lu.

Rh20t: A robotic dataset for learning diverse skills in one-shot. In RSS 2023 Workshop on Learning for Task and Motion Planning, RSS, 2023.

Christoph Feichtenhofer, Haoqi Fan, Yanghao Li, and Kaiming He. Masked autoencoders as spatiotemporal learners. In *Advances in Neural Information Processing Systems, NIPS*, 2022.

Zipeng Fu, Tony Z. Zhao, and Chelsea Finn. Mobile ALOHA: Learning bimanual mobile manipulation using low-cost whole-body teleoperation. In 8th Annual Conference on Robot Learning, CoRL, 2024.

Shijia Ge, Yinxin Zhang, Shuzhao Xie, Weixiang Zhang, Mingcai Zhou, and Zhi Wang. Vggt-dp:
Generalizable robot control via vision foundation models. In *arXiv preprint arXiv:2509.18778*, 2025.

540 541 542 543 544 545 546 547 548 549 550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. In *arXiv preprint arXiv:1911.05722*, 2019.

Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollar, and Ross B. Girshick. Masked ´
autoencoders are scalable vision learners. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR, 2022.

Di Huang, Sida Peng, Tong He, Honghui Yang, Xiaowei Zhou, and Wanli Ouyang. Ponder: Point cloud pre-training via neural rendering. In Proceedings of the IEEE/CVF International Conference on Computer Vision, ICCV, 2023.

Tsung-Wei Ke, Nikolaos Gkanatsios, and Katerina Fragkiadaki. 3d diffuser actor: Policy diffusion with 3d scene representations. In *8th Annual Conference on Robot Learning, CoRL*, 2024.

Alexander Khazatsky, Karl Pertsch, Suraj Nair, Ashwin Balakrishna, Sudeep Dasari, Siddharth Karamcheti, Soroush Nasiriany, Mohan Kumar Srirama, Lawrence Yunliang Chen, Kirsty Ellis, Peter David Fagan, Joey Hejna, Masha Itkina, Marion Lepert, Yecheng Jason Ma, Patrick Tree Miller, Jimmy Wu, Suneel Belkhale, Shivin Dass, Huy Ha, Arhan Jain, Abraham Lee, Youngwoon Lee, Marius Memmel, Sungjae Park, Ilija Radosavovic, Kaiyuan Wang, Albert Zhan, Kevin Black, Cheng Chi, Kyle Beltran Hatch, Shan Lin, Jingpei Lu, Jean Mercat, Abdul Rehman, Pannag R Sanketi, Archit Sharma, Cody Simpson, Quan Vuong, Homer Rich Walke, Blake Wulfe, Ted Xiao, Jonathan Heewon Yang, Arefeh Yavary, Tony Z. Zhao, Christopher Agia, Rohan Baijal, Mateo Guaman Castro, Daphne Chen, Qiuyu Chen, Trinity Chung, Jaimyn Drake, Ethan Paul Foster, Jensen Gao, David Antonio Herrera, Minho Heo, Kyle Hsu, Jiaheng Hu, Donovon Jackson, Charlotte Le, Yunshuang Li, Roy Lin, Zehan Ma, Abhiram Maddukuri, Suvir Mirchandani, Daniel Morton, Tony Khuong Nguyen, Abigail O'Neill, Rosario Scalise, Derick Seale, Victor Son, Stephen Tian, Emi Tran, Andrew E. Wang, Yilin Wu, Annie Xie, Jingyun Yang, Patrick Yin, Yunchu Zhang, Osbert Bastani, Glen Berseth, Jeannette Bohg, Ken Goldberg, Abhinav Gupta, Abhishek 594 595 596 597 598 599 600 601 602 603 604 605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 Gupta, Dinesh Jayaraman, Joseph J Lim, Jitendra Malik, Roberto Mart´ın-Mart´ın, Subramanian Ramamoorthy, Dorsa Sadigh, Shuran Song, Jiajun Wu, Michael C. Yip, Yuke Zhu, Thomas Kollar, Sergey Levine, and Chelsea Finn. DROID: A large-scale in-the-wild robot manipulation dataset. In *RSS 2024 Workshop: Data Generation for Robotics, RSS*, 2024.

Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan P Foster, Pannag R Sanketi, Quan Vuong, Thomas Kollar, Benjamin Burchfiel, Russ Tedrake, Dorsa Sadigh, Sergey Levine, Percy Liang, and Chelsea Finn. OpenVLA: An open-source vision-language-action model. In *8th Annual Conference on Robot Learning, CoRL*, 2024.

Moo Jin Kim, Chelsea Finn, and Percy Liang. Fine-tuning vision-language-action models: Optimizing speed and success. In *arXiv preprint arXiv:2502.19645*, 2025.

Chengmeng Li, Junjie Wen, Yan Peng, Yaxin Peng, Feifei Feng, and Yichen Zhu. Pointvla: Injecting the 3d world into vision-language-action models. In *arXiv preprint arXiv:2503.07511*, 2025a.

Yinchuan Li, Xinyu Shao, Jianping Zhang, Haozhi Wang, Leo Maxime Brunswic, Kaiwen Zhou, Jiqian Dong, Kaiyang Guo, Xiu Li, Zhitang Chen, Jun Wang, and Jianye Hao. Generative models in decision making: A survey. In *arXiv preprint arXiv:2502.17100*, 2025b.

Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, qiang liu, Yuke Zhu, and Peter Stone. LIBERO:
Benchmarking knowledge transfer for lifelong robot learning. In *Thirty-seventh Conference on* Neural Information Processing Systems Datasets and Benchmarks Track, NIPS, 2023.

Songming Liu, Lingxuan Wu, Bangguo Li, Hengkai Tan, Huayu Chen, Zhengyi Wang, Ke Xu, Hang Su, and Jun Zhu. RDT-1b: a diffusion foundation model for bimanual manipulation. In The Thirteenth International Conference on Learning Representations, ICLR, 2025.

Arjun Majumdar, Karmesh Yadav, Sergio Arnaud, Yecheng Jason Ma, Claire Chen, Sneha Silwal, Aryan Jain, Vincent-Pierre Berges, Tingfan Wu, Jay Vakil, Pieter Abbeel, Jitendra Malik, Dhruv Batra, Yixin Lin, Oleksandr Maksymets, Aravind Rajeswaran, and Franziska Meier. Where are we in the search for an artificial visual cortex for embodied intelligence? In Thirty-seventh Conference on Neural Information Processing Systems, NIPS, 2023.

Suraj Nair, Aravind Rajeswaran, Vikash Kumar, Chelsea Finn, and Abhinav Gupta. R3m: A universal visual representation for robot manipulation. In *6th Annual Conference on Robot Learning, CoRL*, 2022.

Octo Model Team, Dibya Ghosh, Homer Walke, Karl Pertsch, Kevin Black, Oier Mees, Sudeep Dasari, Joey Hejna, Charles Xu, Jianlan Luo, Tobias Kreiman, You Liang Tan, Pannag Sanketi, Quan Vuong, Ted Xiao, Dorsa Sadigh, Chelsea Finn, and Sergey Levine. Octo: An open-source generalist robot policy. In *Proceedings of Robotics: Science and Systems, RSS*, 2024.

Maxime Oquab, Timothee Darcet, Th ´ eo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khalidov, ´
Pierre Fernandez, Daniel HAZIZA, Francisco Massa, Alaaeldin El-Nouby, Mido Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Herve Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. DINOv2: Learning robust visual features without supervision.

Transactions on Machine Learning Research, TMLR, 2024.

Yatian Pang, Wenxiao Wang, Francis EH Tay, Wei Liu, Yonghong Tian, and Li Yuan. Masked autoencoders for point cloud self-supervised learning. In *European Conference on Computer* Vision, ECCV, 2022.

Guocheng Qian, Yuchen Li, Houwen Peng, Jinjie Mai, Hasan Abed Al Kader Hammoud, Mohamed Elhoseiny, and Bernard Ghanem. Pointnext: Revisiting pointnet++ with improved training and scaling strategies. In *Advances in Neural Information Processing Systems, NIPS*, 2022.

William Peebles and Saining Xie. Scalable diffusion models with transformers. In arXiv preprint arXiv:2212.09748, 2022.
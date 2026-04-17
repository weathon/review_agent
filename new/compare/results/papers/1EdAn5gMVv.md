000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053

# Spatialboost: Enhancing Visual Representa- Tion Through Language-Guided Reasoning

Anonymous authors Paper under double-blind review

## Abstract

Despite the remarkable success of large-scale pre-trained image representation models (i.e., vision encoders) across various vision tasks, they often fail to learn 3D spatial relationships between objects and backgrounds in the real world, constraining their effectiveness in various downstream applications. We attribute this to the limited availability of large-scale 3D training data, which makes it difficult for current image representation learning approaches to learn spatial relationships. This motivates the need for learning paradigms that rely on strong supervision while requiring less data. To address this, we propose a novel learning framework that enhances the spatial awareness of existing pre-trained vision encoders by injecting dense 3D spatial knowledge expressed in linguistic forms. To be specific, the core idea involves converting dense 3D spatial information from 2D images into linguistic expressions, which is then used to inject such spatial knowledge into vision encoders through a Large Language Model (LLM). To this end, we adopt a multi-turn Chain-of-Thought (CoT) reasoning process that progressively incorporates dense spatial knowledge and builds hierarchical spatial understanding. To validate effectiveness, we adapt SpatialBoost to state-of-the-art vision encoders such as DINOv3, and evaluate its performance gains on a wide range of benchmarks requiring both 3D perception and general vision abilities.

## 1 Introduction

Pre-trained image representation models (He et al., 2020; Donahue & Simonyan, 2019; Chen et al., 2020b; Dosovitskiy et al., 2021; Li et al., 2023b; Assran et al., 2023) have shown remarkable success in various downstream tasks, such as image classification (Krizhevsky et al., 2009; Cui et al., 2018), semantic segmentation (Lin et al., 2014; Zhou et al., 2019), monocular depth prediction (Silberman et al., 2012; Geiger et al., 2012), and vision-language understanding (Antol et al., 2015; Hudson & Manning, 2019). The core idea behind these successes is extracting transferrable representation from large-scale image datasets such as ImageNet (Deng et al., 2009), enabling the model to understand semantic information within images that is significantly useful for various downstream tasks. Despite their success, these models are predominantly trained on 2D images and hence face a fundamental challenge in acquiring 3D spatial awareness capabilities. Consequently, large vision language models struggle to discern 3D spatial relationships between objects in images (Liu et al., 2023a; Fu et al., 2024b; Wang et al., 2025b; Cheng et al., 2024), and demonstrate sub-optimal performance in vision-based robotic control tasks compared to approaches that directly utilize 3D information (Ze et al., 2024; Ke et al., 2024; Zhen et al., 2024). To address these limitations, several works train vision models on multi-view images that naturally encode spatial information (Zhang et al., 2024; Wang et al., 2024b; Charatan et al., 2024). While these approaches have shown promise in robot control tasks (Seo et al., 2023; Sermanet et al., 2018), their broader applicability remains constrained by the need to use carefully curated data (Yu et al., 2023) or obtain multi-view datasets from simulation environments (Savva et al., 2019), creating significant limitations for scaling up these approaches. These challenges highlight the need for a novel framework that enables effective learning of 3D information with substantially less data. However, we note that vision models specialized for individual tasks are able to infer object positions and point depths from standard 2D images. These extracted cues make it possible to extend spatial information by modeling geometric relationships between objects in a scene. We hypothesize that 1

![1_image_0.png](1_image_0.png)

054 055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107 such spatial information can be systematically converted into explicit representations by leveraging language. Moreover, since language naturally composes information in a sequential and structured form, this property allows the construction of labels that capture dense spatial relationships within a scene. Based on these insights, we introduce SpatialBoost, a training framework that enhances the spatial understanding of pre-trained vision encoders by leveraging language-guided reasoning (see Figure 1). We inject linguistically described spatial knowledge through decoder-based fine-tuning with Large Language Models (LLM), where the model takes single or multi-view images as input and generates descriptions. In particular, to leverage this knowledge without forgetting the existing knowledge, we incorporate additional learnable parameters (*i.e.*, dual-channel attention module) into the vision encoder and train only them while freezing the existing parameters. Furthermore, to incorporate dense spatial information in a structured manner, we present a multi-turn visual spatial reasoning approach that builds hierarchical spatial understanding through pixel-level, object-level, and scene-level sub-questions and answers. To validate the effectiveness of our method, we apply SpatialBoost to state-of-the-art image encoders, including DINOv3 (Simeoni et al. ´ , 2025) and SigLIPv2 (Tschannen et al., 2025), and evaluate them across a diverse set of vision tasks: monocular depth estimation, semantic segmentation, 3D scene understanding, vision-based robotic control, image classification, image retrieval, spatial reasoning, and general VQA.1 Our experiment first shows that SpatialBoost consistently improves performance on tasks requiring 3D spatial knowledge. For example, on the 3D scene understanding task, SpatialBoost improves DINOv3 by 3.5% (51.4% → 54.9%) on the SQA3D task from Lexicon3D Benchmark (Man et al., 2024). In addition, on depth estimation tasks, SpatialBoost improves SigLIPv2 from an RMSE score of 0.51 to 0.39 on NYUd linear probing. Moreover, we show that SpatialBoost even improves the performance of the vision encoders across all benchmarks, notably in image classification: SpatialBoost improves ImageNet linear probing performance of DINOv3 from 88.4% to 90.2%.

## 2 Related Work

Self-supervised Learning for Image Representation. In earlier years, most approaches relied on supervised learning with large-scale labeled datasets to train models (Deng et al., 2009; Simonyan & Zisserman, 2014; Szegedy et al., 2014; He et al., 2016). However, the dependence on annotated data introduced scalability challenges due to label expense. To address this, self-supervised learning (SSL) has emerged as a dominant paradigm, leveraging unlabeled data to learn image representations. Contrastive learning methods, including SimCLRv2 (Chen et al., 2020c), MoCov3 (Chen et al., 2021), DINOv2 (Oquab et al., 2023), and iBOT (Zhou et al., 2021), are trained to distinguish between representations of augmented views of the same image and those of different images. Concurrently, mask prediction approaches such as BEiT (Bao et al., 2021) and MAE (He et al., 2022), learn representations by reconstructing masked portions of input images. While these methods excel at capturing rich semantic features within 2D images, they lack mechanisms to effectively encode 3D spatial knowledge. On the other hand, we overcome this limitation by enhancing image representations through a novel method that injects 3D spatial knowledge by utilizing language decoding.

108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140

## 141

142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 Multi-modal Learning for Image Representation. The increasing prominence of multi-modal tasks has catalyzed the development of vision-language models that jointly represent visual and textual information. These models typically employ weakly supervised learning by leveraging text caption. Contrastive learning schemes, e.g., CLIP (Radford et al., 2021), SigLIP (Zhai et al., 2023)
and OpenCLIP (Cherti et al., 2023), consist of vision and text encoders and are trained to align their representations in a shared embedding space. Alternative methodologies like M3AE (Geng et al., 2022), jointly encode image patches and text tokens, employing masked prediction objectives to reconstruct both modalities. More recently, autoregressive formulations such as iGPT (Chen et al., 2020b), have emerged, treating image patches and text tokens as sequential elements for predictive modeling. These approaches successfully enrich visial representations with semantic context derived from natural language descriptions. However, existing models necessitate joint pre-training of both modalities from scratch, imposing significant computational demands and preventing efficient adaptation of existing pre-trained models. Our method eliminates the need for joint text-image representation learning by using LLM, thereby enhancing pre-trained models with relevant linguistic information efficiently.

## 3 Method 3.1 Training Pipeline

Multi-View Learning for Image Representation. Recent advances in vision tasks that require 3D spatial understanding and generation have increased the demand for effective 3D spatial representations (Chen et al., 2024b; Wu et al., 2024; Goyal et al., 2023; Shridhar et al., 2023). Multi-view images from different camera viewpoints or video sequences serve as input for these tasks. Our focus is specifically on augmenting image representations with useful 3D information. Typically, following approaches similar to single-view image representation learning, multi-view data has been processed by converting images into patches for masked prediction such as MV-MWM (Seo et al., 2023) or through contrastive learning methods (Sermanet et al., 2018). Additionally, to learn 3D- related information more explicitly, approaches that predict 3D features from image representation (Ke et al., 2024; Gervet et al., 2023; Ze et al., 2024) have been proposed. These approaches have led to significant performance improvements in vision-based robot control. However, such methods are limited by multi-view data, making it difficult to develop them into pre-trained models for general 3D understanding. Our approach proposes a method to learn 3D spatial representations from both single-view and multi-view images, avoiding these limitations. In this section, we introduce SpatialBoost, a visual representation learning framework designed to improve vision encoders by injecting 3D spatial information expressed in natural language. We first present a multi-modal architecture that incorporates linguistically expressed visual information into the vision encoder through a dual-channel attention layer, ensuring that original visual features are preserved while 3D spatial information is fully exploited (see Section 3.1). On top of this architecture, we design a Visual-Question-Answering (VQA) dataset that hierarchically disentangles 3D
spatial relations from both single/multi-view images, enabling the vision encoder to learn spatial information more effectively (see Figure 1). To train a vision encoder from rich spatial information encoded in large-scale linguistic expressions, our key idea is to utilize Large-Language Models (LLM) by constructing a multi-modal architecture composed of a vision encoder fV , a trainable projection module gP , and the LLM fL. However, without proper alignment between visual and textual representations, the training signals from the LLM cannot effectively propagate back to the vision encoder, making the learning process ineffective. To fully exploit language supervision, we begin by aligning the visual encoder with the textual embedding space of the LLM. Specifically, we adopt LLaVA (Liu et al., 2023b), a two-stage training for the alignment: feature alignment (Stage 1) and visual instruction tuning (Stage 2). After the alignment, we introduce a training framework that uses a language-guided reasoning dataset to

![3_image_0.png](3_image_0.png)

fine-tune the vision encoder (Stage 3). Notably, direct full fine-tuning in this final stage would lead to catastrophic forgetting of the pre-trained knowledge embedded in the vision encoder. To address this challenge, we introduce *dual-channel attention* layers that enable the model to acquire spatial understanding while preserving its original representational capabilities. Formally, given an input image x and multi-turn conversation data (x 1q, x 1a, *· · ·* , x T
q, x T
a) from question-answering (QA) pairs (Qx, Ax), we first encode x to obtain visual features zv = fV (x), which are mapped into the token embedding space via gP (zv). These visual tokens are then concatenated with text tokens and fed into the LLM. Given the multi-turn conversation data and input image, we optimize the model through autoregressive loss. Our training pipeline consists of three stages and all stages are trained with supervised fine-tuning (SFT) loss. We describe each stage in the following paragraphs.

162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 Stage 1: Feature alignment. In this stage, we train a projector gP that maps image features into the textual embedding space of the LLM. This projector pre-training contributes to the stable vision-language alignment. Following the training setup in multi-modal large language models (Liu et al., 2023a; 2024a), we freeze the parameters of both the visual encoder fV and the language model fL, and optimize only the projector gP .

Stage 2: Visual instruction tuning. Following the projector alignment in Stage 1, this stage extends the alignment to the LLM. We freeze the visual encoder fV and fine-tune the projector gP and the language model fL using our multi-view VQA data, combined with the singleview visual instruction data from LLaVA (Liu et al.,
2023a). This step enables fL and gP to handle multiview visual questions. We provide details of proposed multi-view VQA data in Section 3.2.

![3_image_1.png](3_image_1.png)

Stage 3: Vision encoder fine-tuning with dual-channel attention. Finally, we fine-tune the vision encoder fV to have the capability of spatial understanding. To effectively inject dense spatial knowledge into the vision encoder, we use multi-turn visual spatial reasoning dataset (see Section 3.2),
which is carefully designed for hierarchical spatial reasoning. We train the vision encoder fV and 216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 Table 1: Results on monocular depth estimation from NYUd (Silberman et al., 2012) and KITTI (Geiger et al., 2013) benchmarks. We report the RMSE score between ground truth and predicted depth values. Lower is better. For all results, we freeze the encoder backbone and train a linear head (lin.) or DPT head (Ranftl et al., 2021) on top of the image features of the last layer.

Table 2: **Results on semantic segmentation** from ADE20K (Zhou et al., 2017) and Pascal VOC (Everingham et al., 2010) benchmarks. We report mIoU score. Higher is better. For all results, we freeze the encoder backbone and report results of linear probing (lin.) or multiscale evaluation (+ms), where the multi-scale approach uses features from the last four layers of the visual encoder to perform segmentation.

ADE20K Pascal VOC
Method lin. +ms lin. +ms OpenCLIP 39.5 46.0 71.7 79.3 +SpatialBoost (Ours) 40.5 47.3 75.1 **80.9** SigLIPv2 42.8 48.7 72.6 79.1 +SpatialBoost (Ours) 45.1 50.8 79.0 **82.2** DINOv2 49.3 53.0 83.0 86.2 +SpatialBoost (Ours) 52.0 54.9 84.5 **87.6** DINOv3 55.9 60.3 86.6 89.8 +SpatialBoost (Ours) 59.7 63.1 88.5 **90.9**

| Method               | lin.   | +ms   | lin.   | +ms   |
|----------------------|--------|-------|--------|-------|
| OpenCLIP             | 39.5   | 46.0  | 71.7   | 79.3  |
| +SpatialBoost (Ours) | 40.5   | 47.3  | 75.1   | 80.9  |
| SigLIPv2             | 42.8   | 48.7  | 72.6   | 79.1  |
| +SpatialBoost (Ours) | 45.1   | 50.8  | 79.0   | 82.2  |
| DINOv2               | 49.3   | 53.0  | 83.0   | 86.2  |
| +SpatialBoost (Ours) | 52.0   | 54.9  | 84.5   | 87.6  |
| DINOv3               | 55.9   | 60.3  | 86.6   | 89.8  |
| +SpatialBoost (Ours) | 59.7   | 63.1  | 88.5   | 90.9  |
| NYUd                 | KITTI  |       |        |       |
| Method               | lin.   | DPT   | lin.   | DPT   |
| OpenCLIP             | 0.53   | 0.41  | 3.54   | 2.70  |
| +SpatialBoost (Ours) | 0.40   | 0.38  | 2.79   | 2.54  |
| SigLIPv2             | 0.51   | 0.40  | 3.32   | 2.64  |
| +SpatialBoost (Ours) | 0.39   | 0.34  | 2.71   | 2.50  |
| DINOv2               | 0.37   | 0.29  | 2.60   | 2.11  |
| +SpatialBoost (Ours) | 0.30   | 0.25  | 2.53   | 2.07  |
| DINOv3               | 0.31   | 0.25  | 2.33   | 2.02  |
| +SpatialBoost (Ours) | 0.25   | 0.21  | 2.20   | 1.84  |

the projection module gP while keeping the parameters of the LLM fL frozen, allowing only the vision encoder to benefit from language-driven spatial information. We employ SFT loss, and through this training process, the vision encoder learns to extract meaningful representations necessary for producing answers. However, direct full fine-tuning risks forgetting of the pre-trained knowledge embedded in the vision encoder. To address this challenge, we introduce a dual-channel attention mechanism (see Figure 3). Specifically, for each attention layer Attn(·) in the visual encoder fV , we introduce an additional attention layer Attn+(·), whose weight parameters are initialized to the same values as those of Attn(·). Given an input x to each attention layer, we merge the outputs of Attn(·) and Attn+(·) by introducing a trainable mixture factor α = sigmoid(a) ∈ (0, 1)d with zero-initialized parameter a ∈ R
d, where d is the hidden dimension of x, as follows:

## Attnfinal(X) = Α · Attn(X) + (1 − Α) · Attn+(X). (1)

During fine-tuning, we only update the parameters of Attn+ and α while keeping all other parameters frozen. This approach allows the vision encoder to initially rely on pre-trained attention weights and gradually incorporate new attention weights, smoothly enhancing spatial awareness without discarding existing knowledge (see classification result in Figure 6).

## 3.2 Enhancing Vision Encoder With Spatial Cot

To effectively inject dense spatial information into vision encoders, we address the fundamental limitations of existing spatial datasets. Current spatial VQA data consist of simple single-turn QA pairs with limited information content, insufficient for transferring comprehensive 3D understanding. To overcome this limitation, We introduce Multi-view VQA, which helps align the vision encoder with the LLM to effectively handle multi-view data and a multi-turn Chain-of-Thought (CoT) framework (Wei et al., 2022) for both single-view and multi-view images that enables the injection of substantially richer spatial information in a single training instance. Multi-view VQA Dataset. To enhance multi-view VQA capabilities during the visual instruction tuning (Stage 2), we construct multi-view VQA dataset. We first apply LPIPS (Zhang et al., 2018) metric to the 3D or video dataset to obtain a pair of images. Given the pair of images, we employ GPT-4o (Achiam et al., 2023) to generate visual questions targeting general multi-view knowledge. We provide more details in Section C. Multi-Turn Visual Spatial Reasoning Dataset. To enhance spatial reasoning capabilities of the vision encoder (Stage 3), we construct multi-turn visual spatial reasoning dataset for single-view and multi-view. Additionally, to enhance general knowledge of the vision encoder, we append GPT-generated scene captions after spatial reasoning turn. For single-view image, we first extract a 3D point cloud from given an image x by applying diverse vision models (e.g., depth estimation 270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323

## 4 Experiments

Through extensive experiments, we validate the performance of SpatialBoost and ablate its key components, focusing on following questions: - Can SpatialBoost improve spatial knowledge of the vision encoder? (Tables 1 to 4)
- Isn't SpatialBoost overfitted to spatial knowledge? (Table 5)
- Which components contribute to SpatialBoost performance? (Table 6 and Figure 6)

## 4.1 Experimental Setup

VQA Dataset Construction. For single-view image, we use randomly sampled 100K images from the SA1B dataset (Kirillov et al., 2023) to construct the single-view VQA dataset specialized in chain-of-thought spatial reasoning. For multi-view images, we use filtered 200K samples from the ego-centric video dataset (Grauman et al., 2022) and 3D dataset (Jensen et al., 2014; Dai et al., 2017; Mildenhall et al., 2021; Barron et al., 2022) to construct multi-view VQA dataset niche in multi-view reasoning or alignment. More details in Section D. Baselines. For all experiments, we compare our methods with the recent widely-used pre-trained image representation models. To be specific, we first consider OpenCLIP (Cherti et al., 2023) ViT- G/14 and SigLIPv2 (Tschannen et al., 2025) ViT-g/16, known for language-aligned vision encoder. We also consider DINOv2 (Oquab et al., 2023) ViT-g/14 and DINOv3 (Simeoni et al. ´ , 2025) ViT- 7B/16, which is a recent state-of-the-art vision encoder.

Table 3: **Results on 3D-centric tasks.** We evaluate unified probing on diverse 3D-related tasks from ScanNet (Dai et al., 2017) scenes. We report BLEU-1 score for Vision-Language Reasoning (VLR) on ScanQA (Azuma et al., 2022) and SQA3D (Ma et al., 2023). For Visual Grounding (VG), we report accuracy on overall category of ScanRefer (Chen et al., 2020a) dataset. For Geometric Understanding (GU), we report Registration Recall (RR) at 0.05m RMSE threshold and Relative Translation Error (RTE). For 3D Semantic Understanding (3D SU), we report accuracy and mIoU. Lower is better for RTE and higher is better for all other metrics.

VLR VG GU 3D SU

Method ScanQA ↑ SQA3D ↑ ScanRefer-Overall ↑ RR@0.05m (%) ↑ RTE (m) ↓ Acc ↑ mIoU ↑ OpenCLIP 36.9 48.0 50.1 22.6 0.40 39.8 6.9 +SpatialBoost (Ours) 39.2 49.9 56.6 78.8 0.17 76.9 **54.9** SigLIPv2 38.1 48.5 51.4 47.8 0.28 47.7 9.2 +SpatialBoost (Ours) 40.8 50.1 56.8 86.4 0.15 81.0 **55.5** DINOv2 39.5 49.8 52.7 82.4 0.15 83.0 64.1 +SpatialBoost (Ours) 40.3 50.4 57.0 92.4 0.13 89.8 **68.3** DINOv3 40.6 51.4 56.2 86.9 0.10 91.1 69.1 +SpatialBoost (Ours) 43.3 54.9 61.1 97.5 0.06 91.9 **70.6**

model (Bochkovskii et al., 2024) and image segmentation model (Ravi et al., 2024)). For multiview images {x1, *· · ·* , xN}, we use 3D reconstruction model (Wang et al., 2025a) to extract a 3D
point cloud from given images. Using the point cloud, we synthesize QA pairs specialized in spatial reasoning about x or {x1, *· · ·* , xN}.

We then design spatial reasoning QA pairs at three hierarchical levels: pixel, object, and scene, enabling LLM to perform CoT reasoning from narrow to broad view. Specifically, at the pixellevel, the QA task is designed to capture the overall geometry in the image by querying the absolute or relative 3D position of a point, e.g., "What is the depth value at coordinate (*x, y*)?". At the object-level, the QA task tackles the semantic spatial information of objects inside the image using a bounding cube of the object in 3D space, e.g., "Is [A] on the left side of [B]?", where [A] and [B] is the descriptions about the object in image. We note that this level uses the pixel-level spatial information as a rationale, enabling LLM to reason about the geometry of objects in 3D space. Lastly, at the scene-level, the QA task is designed to predict the exact distance between multiple objects that requires coherent 3D spatial understanding, e.g., "How far is [A] from [B]?".

324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377 Table 4: **Results on vision-based robot learning.** We report the performance of imitation learning agents on 4 domains from CortexBench (Majumdar et al., 2023), which are trained upon the image representations. In particular, we report the normalized score for DMControl and success rates (%) for other tasks.

Method Adroit MetaWorld DMControl Trifinger Avg. OpenCLIP 52.6 ± 4.9 83.0 ± 2.7 58.5 ± 1.9 67.7 ± 0.5 65.5 +SpatialBoost (Ours) **61.1** ± 3.4 **87.0** ± 3.3 **61.0** ± 1.6 **72.9** ± 0.3 **70.5** SigLIPv2 56.5 ± 3.0 84.7 ± 2.9 69.4 ± 2.1 68.3 ± 0.8 69.7 +SpatialBoost (Ours) **66.5** ± 1.9 **89.1** ± 0.9 **73.5** ± 1.8 **73.9** ± 0.7 **75.8** DINOv2 55.4 ± 2.7 82.4 ± 4.0 67.9 ± 1.0 66.8 ± 0.2 68.1 +SpatialBoost (Ours) **68.1** ± 2.9 **88.5** ± 3.1 **75.0** ± 1.1 **71.4** ± 0.8 **75.8**

DINOv3 63.9 ± 1.5 83.8 ± 1.6 70.8 ± 1.8 72.8 ± 0.5 72.8

+SpatialBoost (Ours) **71.8** ± 3.4 **92.0** ± 1.9 **80.4** ± 2.4 **79.0** ± 0.6 **80.8**

Figure 4: **Examples of vi-**

![6_image_0.png](6_image_0.png)

sual observations from CortexBench. We train imitation learning agents to learn a mapping from these visual observations to expert actions.

Implementation Details. We choose Qwen-2.0-7B (Yang et al., 2024) as the LLM backbone and 2-layer MLP as the projector, following the architecture of LLaVA-1.5 (Liu et al., 2024a). Further details are provided in Section A.

## 4.2 Dense Prediction Tasks

Setup. We evaluate SpatialBoost on dense prediction tasks requiring geometric and semantic spatial understanding. For geometric understanding, we perform monocular depth estimation on NYUd (Silberman et al., 2012) and KITTI (Geiger et al., 2013) using linear or DPT (Ranftl et al., 2021) heads. For semantic understanding, we evaluate on ADE20K (Zhou et al., 2017) and Pascal VOC (Everingham et al., 2010) segmentation benchmarks using linear or multi-scale heads. All experiments freeze the visual backbone during training (see Section A for details). Results. As shown in Table 1 and 2, SpatialBoost consistently improves both geometric and semantic spatial understanding across various encoders. For instance, OpenCLIP's RMSE on NYUd decreases from 0.53 to 0.40 with a linear head, while DINOv3's mIoU on ADE20K increases from 55.9% to 59.7%. These consistent gains demonstrate that language-based spatial knowledge transfer effectively enhances visual encoders' spatial understanding capabilities.

## 4.3 Complex 3D-Centric Tasks

Setup. We evaluate SpatialBoost on Lexicon3D (Man et al., 2024), a unified benchmark for 3D scene understanding covering vision-language reasoning, visual grounding, semantic understanding, and geometric understanding. Following Lexicon3D protocols, we freeze visual backbones and train task-specific heads (see Section A for details). Results. As shown in Table 3, SpatialBoost shows comprehensive improvements across diverse 3D tasks. OpenCLIP's BLEU-1 improves from 36.9 to 39.2 on ScanQA (Azuma et al., 2022), while DINOv3 increases from 51.4 to 54.9 on SQA3D (Ma et al., 2023), demonstrating that SpatialBoost improves spatial understanding without compromising language capabilities. Notably, SigLIPv2's 3D semantic segmentation dramatically improves from 6.9 to 54.9 mIoU, highlighting SpatialBoost can inject robust spatial knowledge into encoders with initially limited spatial awareness.

## 4.4 Vision-Based Robot Learning

Setup. We evaluate SpatialBoost on vision-based robot control using 4 domains from CortexBench (Majumdar et al., 2023) spanning locomotion and manipulation tasks (Rajeswaran et al., 2017; Yu et al., 2020; Tassa et al., 2018; Wuthrich et al. ¨ , 2020). Following CortexBench protocols, we train behavior cloning agents using [CLS] representations to predict expert actions from visual observations. We report the mean of best performance across 5 evaluation runs (see Section A for details).

378 379 380 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431

Table 5: **Results on image classification and retrieval tasks.** We report Top-1 accuracy of kNN performance and linear probing (lin.) for image classification on validation set of ImageNet-1K (Russakovsky et al., 2015). For image retrieval, we report global average precision (GAP) on Met (Ypsilantis et al., 2021) and mean average precision (mAP) on Oxford-Hard (Oxford-H) (Radenovic´ et al., 2018), Paris-Hard (Paris-H) (Radenovic et al. ´ , 2018), and AmsterTime dataset (Yildiz et al., 2022). For all results, we freeze the encoder backbone.

Image classification Image retrieval

Method ImageNet (kNN) ImageNet (lin.) Oxford-H Paris-H Met (GAP) AmsterTime OpenCLIP 84.0 86.8 23.4 59.7 7.4 24.4 +SpatialBoost (Ours) 86.1 87.9 32.8 69.4 19.7 **30.3** SigLIPv2 86.3 89.1 25.1 60.9 13.9 15.5 +SpatialBoost (Ours) 87.6 90.0 36.0 69.1 24.0 **27.2** DINOv2 84.5 87.3 58.2 84.6 44.6 48.9 +SpatialBoost (Ours) 86.4 88.6 61.3 85.2 45.1 **50.8** DINOv3 85.8 88.4 60.7 87.1 55.4 56.5 +SpatialBoost (Ours) 87.7 90.2 64.1 88.6 57.0 **56.9**

![7_image_0.png](7_image_0.png)

![7_image_1.png](7_image_1.png)

oU

Figure 5: **Effect of dataset scalability.** We investigate the effect of the size of analysis of data scalability effects on (a) depth estimation results (AbsRel, RMSE) on NYUd benchmark for SigLIPv2, (b) depth estimation results (AbsRel, RMSE) on NYUd benchmark for DINOv3, and (c) semantic segmentation results (mIoU) on ADE20K benchmark for SigLIPv2 and DINOv3. The results show scalable performance improvements with increased data size. Results. As shown in Table 4, SpatialBoost significantly improves robot task performance across all vision encoders. For example, DINOv2 + SpatialBoost achieves 68.1% on Adroit versus 55.4% for DINOv2 alone, demonstrating that enhanced spatial representations directly benefit robot control.

## 4.5 Image Classification And Retrieval Tasks

Setup. We evaluate SpatialBoost's impact on instance recognition using ImageNet-1K (Russakovsky et al., 2015) classification and retrieval benchmarks (Oxford, Paris (Radenovic et al. ´ , 2018), Met (Ypsilantis et al., 2021), AmsterTime (Yildiz et al., 2022)). Following DINOv3 protocols, we use linear probing on [CLS] representations for classification and similarity-based ranking for retrieval (see Section A for details). Results. As shown in Table 5, SpatialBoost improves both classification and retrieval despite these tasks not explicitly requiring spatial understanding. DINOv3's ImageNet accuracy increases from 88.4% to 90.2%, while Oxford-Hard mAP improves from 60.7 to 64.1. These results demonstrate that SpatialBoost enhances general vision capabilities without overfitting to spatial features, likely due to our dual-channel attention preserving pre-trained knowledge and the inclusion of general scene captions alongside spatial reasoning.

## 4.6 Ablation Study And Analysis

Effect of LLM-based Fine-tuning. In Table 6, we investigate whether LLM-based decoders provide superior supervision compared to pixel-level alternatives. We fine-tune the vision encoder with linear layer, SAM (Kirillov et al., 2023) decoder, VGGT (Wang et al., 2025a) decoder, and LLM (Yang et al., 2024). We then evaluate encoders on ImageNet-1K classification, ADE20K segmentation, and NYUd depth estimation. The results show that LLM consistently outperform pixel-level supervision methods, validating that language provides superior dense information transfer for vision encoders (see Section E for details).

432 433 434 435 436 437 438 439 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485 Table 6: **Effect of LLM-based fine-tuning.** We fine-tune

![8_image_0.png](8_image_0.png) the vision encoder with different headers. We report accuracy (%) for classification (Cls) on ImageNet-1K, mIoU for segmentation (Seg) on ADE20K, RMSE for depth estimation on NYUd, and BLEU-1 score for vision-language reasoning (VLR) on ScanQA. We use ViT-L/14 as the backbone architecture of the encoder.

Method Cls ↑ Seg ↑ Depth ↓ VLR ↑ DINOv2 86.3 47.7 0.38 39.2 +Linear (depth) 85.7 (-1.39%) 47.9 (+0.42%) 0.35 (-7.89%) 36.9 (-5.87%)

+Linear (seg.) 86.6 (+0.35%) 48.8 (+2.31%) 0.45 (+18.42%) 37.1 (-5.36%) +SAM decoder 86.3 (+0.0%) 50.1 (+5.03%) 0.42 (+10.53%) 37.6 (-4.08%)

+VGGT decoder 84.8 (-1.74%) 45.6 (-4.40%) 0.35 (-7.89%) 37.3 (-4.85%)

+LLM (Ours) 88.3 (+2.32%) 51.5 (+7.97%) 0.32 (-15.79%) 40.0 (+2.04%)

Effect of Multi-turn Visual Reasoning. In Table 7, we investigate how the hierarchical structure of reasoning affects representation learning. We compare dataset construction strategies: (a) shuffled multi-turn, (b) reversed order (scene→object→pixel), and (c) forward order (pixel→object→scene). The forward hierarchical ordering shows optimal performance, demonstrating that reasoning order significantly impacts the quality of representation. Effect of Single-view and Multi-view Data. In Table 7, we investigate the effect of single-view and multi-view reasoning data. With fixed total samples, we compare single-view only, multi-view only, and combined training. While both data types independently improve performance, the combination achieves the highest results, confirming their complementary nature.

| Method        | Multi-turn order   | Single-view data   | Multi-view data   | Cls ↑   | Seg ↑   | Depth ↓   |
|---------------|--------------------|--------------------|-------------------|---------|---------|-----------|
| DINOv2        | ✗                  | -                  | -                 | 86.3    | 47.7    | 0.38      |
| +SpatialBoost | Reverse            | +100K              | -                 | 87.4    | 48.4    | 0.35      |

Comparison with Naive Post-training. In Table 8, we investigate the effect of post-training. With fixed total samples (i.e., 300K data in multi-turn reasoning data), we compare the naive post-training scheme and SpatialBoost. We evaluate the performance of the vision encoder across five tasks: depth estimation, segmentation, vision-language reasoning, robot learning, and classification. The results show that naive post-training does not yield effective representations for downstream tasks. Effect of Dual-channel Attention Layer. In Figure 6, we investigate whether our dual-channel attention mechanism preserves pre-trained knowledge during fine-tuning. We evaluate several approaches for fine-tuning the vision encoder including full fine-tuning, LoRA (Hu et al., 2021), and dual-channel (Hong et al., 2023a) on ImageNet (Russakovsky et al., 2015) and ADE20K (Zhou et al., 2017). Dual-channel attention uniquely preserves and even enhances pre-trained knowledge, while other approaches cause degradation. Dataset Scalability. We analyze the impact of dataset sizes on depth estimation results from NYUd (Silberman et al., 2012) benchmark and semantic segmentation results from ADE20K (Zhou et al., 2017) benchmark. With matched training iterations (i.e., one epoch for 300K data), larger datasets yield consistent improvements, indicating robust scalability potential.

## 5 Conclusion

In this paper, we have presented SpatialBoost, a framework to enhance the vision encoders by leveraging linguistic expressions of geometric and semantic information within images. SpatialBoost

## References

| Method               | Depth Estimation ↓   | Segmentation ↑   | Vision-Language Reasoning ↑   | Robot Learning ↑   | Classification ↑   |
|----------------------|----------------------|------------------|-------------------------------|--------------------|--------------------|
| OpenCLIP             | 0.53                 | 39.5             | 36.9                          | 65.5               | 84.0               |
| +Simple FT           | 0.56                 | 39.6             | 37.7                          | 63.7               | 84.3               |
| +SpatialBoost (Ours) | 0.40                 | 40.5             | 39.2                          | 72.9               | 86.1               |
| SigLIPv2             | 0.51                 | 42.8             | 38.1                          | 69.7               | 86.3               |
| +Simple FT           | 0.53                 | 43.0             | 38.4                          | 67.9               | 86.4               |
| +SpatialBoost (Ours) | 0.39                 | 45.1             | 40.8                          | 75.8               | 87.6               |
| DINOv2               | 0.37                 | 49.3             | 39.5                          | 68.1               | 84.5               |
| +Simple FT           | 0.36                 | 49.6             | 39.4                          | 69.4               | 84.7               |
| +SpatialBoost (Ours) | 0.30                 | 52.0             | 40.3                          | 75.8               | 86.4               |
| DINOv3               | 0.31                 | 55.9             | 40.6                          | 72.8               | 85.8               |
| +Simple FT           | 0.31                 | 56.4             | 40.2                          | 75.5               | 86.1               |
| +SpatialBoost (Ours) | 0.25                 | 59.7             | 43.3                          | 80.8               | 87.7               |

uses LLM and dual-channel attention layers to exploit linguistic information into image representations, generates a multi-turn visual spatial reasoning dataset, and leverages them to improve the image representations. Our experiments show that SpatialBoost consistently enhances the vision encoders on various downstream tasks that require a spatial understanding of images. We hope that our work further facilitates future research on designing and enhancing vision encoders.

486 487 488 489 490 491 492 493 494 495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539 Saminda Abeyruwan, Joshua Ainslie, Jean-Baptiste Alayrac, Montserrat Gonzalez Arenas, Travis Armstrong, Ashwin Balakrishna, Robert Baruch, Maria Bauza, Michiel Blokzijl, et al. Gemini robotics: Bringing ai into the physical world. *arXiv preprint arXiv:2503.20020*, 2025.

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. *arXiv preprint arXiv:2303.08774*, 2023.

Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C Lawrence Zitnick, and Devi Parikh. Vqa: Visual question answering. In *ICCV*, 2015.

Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, and Nicolas Ballas. Self-supervised learning from images with a joint-embedding predictive architecture. In *CVPR*, 2023.

Daichi Azuma, Taiki Miyanishi, Shuhei Kurita, and Motoaki Kawanabe. Scanqa: 3d question answering for spatial scene understanding. In *CVPR*, pp. 19129–19139, 2022.

Hangbo Bao, Li Dong, Songhao Piao, and Furu Wei. Beit: Bert pre-training of image transformers.

arXiv preprint arXiv:2106.08254, 2021.

Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P Srinivasan, and Peter Hedman. Mip-nerf 360: Unbounded anti-aliased neural radiance fields. In *CVPR*, pp. 5470–5479, 2022.

Aleksei Bochkovskii, Amael Delaunoy, Hugo Germain, Marcel Santos, Yichao Zhou, Stephan R ¨
Richter, and Vladlen Koltun. Depth pro: Sharp monocular metric depth in less than a second. arXiv preprint arXiv:2410.02073, 2024.

David Charatan, Sizhe Lester Li, Andrea Tagliasacchi, and Vincent Sitzmann. pixelsplat: 3d gaussian splats from image pairs for scalable generalizable 3d reconstruction. In *CVPR*, 2024.

Boyuan Chen, Zhuo Xu, Sean Kirmani, Brain Ichter, Dorsa Sadigh, Leonidas Guibas, and Fei Xia.

Spatialvlm: Endowing vision-language models with spatial reasoning capabilities. In *CVPR*,
2024a.

Dave Zhenyu Chen, Angel X Chang, and Matthias Nießner. Scanrefer: 3d object localization in rgb-d scans using natural language. In *ECCV*, pp. 202–221. Springer, 2020a.

Mark Chen, Alec Radford, Rewon Child, Jeffrey Wu, Heewoo Jun, David Luan, and Ilya Sutskever.

Generative pretraining from pixels. In *ICML*, 2020b.

Ting Chen, Simon Kornblith, Kevin Swersky, Mohammad Norouzi, and Geoffrey E Hinton. Big self-supervised models are strong semi-supervised learners. In *NeurIPS*, 2020c.

Xinlei Chen, Saining Xie, and Kaiming He. An empirical study of training self-supervised vision transformers. In *ICCV*, 2021.

Yuedong Chen, Haofei Xu, Chuanxia Zheng, Bohan Zhuang, Marc Pollefeys, Andreas Geiger, Tat-
Jen Cham, and Jianfei Cai. Mvsplat: Efficient 3d gaussian splatting from sparse multi-view images. In *ECCV*, 2024b.

An-Chieh Cheng, Hongxu Yin, Yang Fu, Qiushan Guo, Ruihan Yang, Jan Kautz, Xiaolong Wang, and Sifei Liu. Spatialrgpt: Grounded spatial reasoning in vision-language models. In *NeurIPS*, 2024.

Mehdi Cherti, Romain Beaumont, Ross Wightman, Mitchell Wortsman, Gabriel Ilharco, Cade Gordon, Christoph Schuhmann, Ludwig Schmidt, and Jenia Jitsev. Reproducible scaling laws for contrastive language-image learning. In *CVPR*, 2023.

Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al. Scaling instruction-finetuned language models. *Journal of Machine Learning Research*, 25(70):1–53, 2024.

Yin Cui, Yang Song, Chen Sun, Andrew Howard, and Serge Belongie. Large scale fine-grained categorization and domain-specific transfer learning. In *CVPR*, 2018.

Angela Dai, Angel X Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. Scannet: Richly-annotated 3d reconstructions of indoor scenes. In *CVPR*, pp. 5828– 5839, 2017.

Timothee Darcet, Maxime Oquab, Julien Mairal, and Piotr Bojanowski. Vision transformers need ´
registers. In *ICLR*, 2023.

540 541 542 543 544 545 546 547 548 549 550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 Google DeepMind. Gemini 2.0 model updates: 2.0 flash, flash-lite, pro experimental. February 2025.

Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In *CVPR*, 2009.

Jeff Donahue and Karen Simonyan. Large scale adversarial representation learning. In *NeurIPS*,
2019.

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. In *ICLR*, 2021.

M. Everingham, L. Van Gool, C. K. I. Williams, J. Winn, and A. Zisserman. The pascal visual object classes (voc) challenge. *IJCV*, 2010.

Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, Yunsheng Wu, and Rongrong Ji. Mme: A comprehensive evaluation benchmark for multimodal large language models. *arXiv preprint arXiv:2306.13394*, 2024a.

Xingyu Fu, Yushi Hu, Bangzheng Li, Yu Feng, Haoyu Wang, Xudong Lin, Dan Roth, Noah A
Smith, Wei-Chiu Ma, and Ranjay Krishna. Blink: Multimodal large language models can see but not perceive. In *ECCV*, 2024b.

Andreas Geiger, Philip Lenz, and Raquel Urtasun. Are we ready for autonomous driving? the kitti vision benchmark suite. In *CVPR*, 2012.

Andreas Geiger, Philip Lenz, Christoph Stiller, and Raquel Urtasun. Vision meets robotics: The kitti dataset. *The international journal of robotics research*, 2013.

Xinyang Geng, Hao Liu, Lisa Lee, Dale Schuurmans, Sergey Levine, and Pieter Abbeel. Multimodal masked autoencoders learn transferable representations. *arXiv preprint arXiv:2205.14204*, 2022.

Theophile Gervet, Zhou Xian, Nikolaos Gkanatsios, and Katerina Fragkiadaki. Act3d: 3d feature field transformers for multi-task robotic manipulation. *arXiv preprint arXiv:2306.17817*, 2023.

Ankit Goyal, Jie Xu, Yijie Guo, Valts Blukis, Yu-Wei Chao, and Dieter Fox. Rvt: Robotic view transformer for 3d object manipulation. In *CoRL*, 2023.

594 595 596 597 598 599 600 601 602 603 604 605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In *CVPR*,
pp. 6904–6913, 2017.

Kristen Grauman, Andrew Westbury, Eugene Byrne, Zachary Chavis, Antonino Furnari, Rohit Girdhar, Jackson Hamburger, Hao Jiang, Miao Liu, Xingyu Liu, et al. Ego4d: Around the world in 3,000 hours of egocentric video. In *CVPR*, pp. 18995–19012, 2022.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In *CVPR*, 2016.

Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. In *CVPR*, 2020.

Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollar, and Ross Girshick. Masked ´
autoencoders are scalable vision learners. In CVPR, 2022.

Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. Cogvideo: Large-scale pretraining for text-to-video generation via transformers. In *ICLR*, 2023a.

Yining Hong, Haoyu Zhen, Peihao Chen, Shuhong Zheng, Yilun Du, Zhenfang Chen, and Chuang Gan. 3d-llm: Injecting the 3d world into large language models. In *NeurIPS*, 2023b.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021.

Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In *CVPR*, 2019.

Stephen James and Andrew J Davison. Q-attention: Enabling efficient learning for vision-based robotic manipulation. *IEEE Robotics and Automation Letters*, 2022.

Rasmus Jensen, Anders Dahl, George Vogiatzis, Engin Tola, and Henrik Aanæs. Large scale multiview stereopsis evaluation. In *CVPR*, pp. 406–413, 2014.

Wolfgang Kabsch. A solution for the best rotation to relate two sets of vectors. Foundations of Crystallography, 32(5):922–923, 1976.

Tsung-Wei Ke, Nikolaos Gkanatsios, and Katerina Fragkiadaki. 3d diffuser actor: Policy diffusion with 3d scene representations. *arXiv preprint arXiv:2402.10885*, 2024.

Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In *ICCV*, 2023.

Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images.

2009.

Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In *International conference* on machine learning, pp. 19730–19742. PMLR, 2023a.

Tianhong Li, Huiwen Chang, Shlok Mishra, Han Zhang, Dina Katabi, and Dilip Krishnan. Mage:
Masked generative encoder to unify representation learning and image synthesis. In *CVPR*, 2023b.
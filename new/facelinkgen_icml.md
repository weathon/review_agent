

# FaceLinkGen: Rethinking Identity Leakage in Privacy-Preserving Face Recognition with Identity Extraction

Anonymous Authors<sup>1</sup>

## Abstract

Transformation-based privacy-preserving face recognition (PPFR) aims to verify identities while hiding facial data from attackers and malicious service providers. Existing evaluations mostly treat privacy as resistance to pixel-level reconstruction, measured by PSNR and SSIM. We show that this reconstruction-centric view fails. We present FaceLinkGen, an identity extraction attack that performs linkage/matching and face regeneration directly from protected templates without recovering original pixels. On three recent PPFR systems, FaceLinkGen reaches over 98.5% matching accuracy and above 96% regeneration success, and still exceeds 92% matching and 94% regeneration in a near zero knowledge setting. These results expose a structural gap between pixel distortion metrics, which are widely used in PPFR evaluation, and real privacy. We show that visual obfuscation leaves identity information broadly exposed to both external intruders and untrusted service providers.

## 1. Introduction and Related Works

The fundamental promise of transformation-based Privacy Preserving Face Recognition (PPFR) systems is compelling: verify a user’s identity without ever exposing their raw facial data to potential attackers (Dai et al., 2025; Mi et al., 2023; 2024; 2022; Jin et al., 2024). Originally, the threat model was for a curious or malicious recognition service provider (Erkin et al., 2009; Ji et al., 2022), but it now also includes, or shifted towards, external attackers by wiretapping (Mi et al., 2024) or leaked databases. We argue that a robust leakage analysis must address both the service provider who inherently accesses the templates and the external intruder who obtains the templates by wiretapping or database leak-

![Figure 1: Regeneration attack results. A 3x6 grid of face images. Each row shows examples from one PPFR method. In each subplot, the left image is the original registration image and the right image is the regenerated image from the protected template. The methods shown are PartialFace, MinusFace, and FracFace.](0538daaa5583c23e17db3a12f2281a55_img.jpg)

Figure 1: Regeneration attack results. A 3x6 grid of face images. Each row shows examples from one PPFR method. In each subplot, the left image is the original registration image and the right image is the regenerated image from the protected template. The methods shown are PartialFace, MinusFace, and FracFace.

Figure 1. Regeneration attack results. In each subplot, the left is the original image and the right is the regenerated image from the protected template. Each row shows examples from one PPFR method, in the order of PartialFace, MinusFace, and FracFace.

age.

The prevailing evaluation paradigm for these systems is currently significantly limited. Historically, the dominant objective has been to prevent the reconstruction of the original registration image. This is typically measured through pixel-level or local similarity metrics such as peak signal-to-noise ratio (PSNR) and structural similarity index measure (SSIM) between the original and reconstructed images, a legacy inherited from image privacy and compression literature. Consequently, a large body of prior work adopts this paradigm, using resistance to pixel-level recovery as evidence of privacy protection and optimizing attack objectives accordingly. Representative systems, including DuetFace (Mi et al., 2022), MinusFace (Mi et al., 2023), PartialFace (Mi et al., 2024), FaceObfuscator (Jin et al., 2024), and FracFace (Dai et al., 2025), explicitly rely on these metrics to argue robustness against recovery attacks.

This evaluation paradigm, however, rests on a critical implicit assumption: that preventing pixel-level reconstruction is both necessary and sufficient to prevent identity leakage. In this paper, we show that this assumption does not hold. Crucially, compromising privacy does not require recovering the original registered image, nor does pixel-level similarity reliably correspond to identity consistency. In the facial domain, two images that are visually or pixel-wise similar may represent different identities, while identity-revealing information can remain accessible even when

<sup>1</sup>Anonymous Institution, Anonymous City, Anonymous Region, Anonymous Country. Correspondence to: Anonymous Author <anon.email@domain.com>.

Preliminary work. Under review by the International Conference on Machine Learning (ICML). Do not distribute.

![Figure 2: Comparison of SSIM and PSNR metrics for two cases. Top row: High Pixel Similarity, Low Identity Similarity (SSIM: 0.86, PSNR: 29.2, FS: 0.140). Bottom row: Low Pixel Similarity, High Identity Similarity (SSIM: 0.35, PSNR: 12.8, FS: 0.728).](c803f6f6e2c49429d2951832bd0f208d_img.jpg)

Figure 2: Comparison of SSIM and PSNR metrics for two cases. Top row: High Pixel Similarity, Low Identity Similarity (SSIM: 0.86, PSNR: 29.2, FS: 0.140). Bottom row: Low Pixel Similarity, High Identity Similarity (SSIM: 0.35, PSNR: 12.8, FS: 0.728).

Figure 2. SSIM and PSNR are not always correlated with identity correlation.

Table 1. Comparison between pixel-level metrics and identity-level metrics on two cases: a protected face generated by CanFG with its original face, and two images of the same person. We can see that higher pixel-level similarity does not mean higher identity-level similarity.

| Compared With | SSIM  | PSNR  | MSE  | FS    |
|---------------|-------|-------|------|-------|
| Another Face  | 0.235 | 10.44 | 6699 | 0.586 |
| CanFG Face    | 0.841 | 26.81 | 143  | 0.008 |

pixel-level reconstruction is infeasible. CanFG (Wang et al., 2024a) can generate two images with very high pixel-level similarity yet in completely different identities; conversely, in daily life, any two arbitrary photos of the same person (one ID photo and one social media photo) would have very high identity similarity yet likely very low pixel-level or structure-level similarity. An example is provided in Figure 2.

These misconceptions mislead not only the evaluation but also the simulated attack design. By employing pixel-level loss functions, simulated attackers (red-team researchers) are inadvertently trapped into pursuing the specific registration image as ground truth. This objective is often mathematically impossible due to the information loss in protection, causing the generator to produce a blurry image, likely an average of all images in the dataset. The failure is illustrated in Figure 3. Even though FracFace (Dai et al., 2025) measured ID-similarities, they mainly focus on pixel-level or local metrics. The StyleGAN (Karras et al., 2019) is also likely guided by pixel-level loss, such that even though it generated a realistic face, it failed to generate the face with the same identity as the original image.

Recent work has begun to challenge this reconstruction-centric perspective in non-facial domains by shifting attention toward semantic-level inversion. In such settings, the attacker aims not to recover the original image itself, but to regenerate information that is semantically consistent with the original identity. This goal is both easier to achieve and more aligned with realistic attack objectives.

![Figure 3: Using pixel-level loss or StyleGAN will yield unsuccessful reconstruction compared to our ID-guided method. Top row: Original Face and U-Net Attack. Bottom row: StyleGAN Attack (Farrar et al., 2023) and Our ID-based Attack (Farrar et al., 2023).](b6cb8677b4ffb35c6468fa5c24091bff_img.jpg)

Figure 3: Using pixel-level loss or StyleGAN will yield unsuccessful reconstruction compared to our ID-guided method. Top row: Original Face and U-Net Attack. Bottom row: StyleGAN Attack (Farrar et al., 2023) and Our ID-based Attack (Farrar et al., 2023).

Figure 3. Using pixel-level loss or StyleGAN will yield unsuccessful reconstruction compared to our ID-guided method.

For instance, Yue et al. (2023) proposes a semantic recovery framework that leverages generative models to synthesize semantic-consistent images without pixel-level similarity in federated learning attacks. Complementary findings further suggest that traditional reconstruction metrics fail to capture how humans perceive privacy leakage (Sun et al., 2023).

Our approach differs from standard Model Inversion Attacks (MIAs). One category of MIA, represented by Wang et al. (2025), recovers original images from embeddings. For many deep facial embedding models, such processes resemble image generation tasks rather than adversarial attacks, as they exploit the inherent invertibility of learned representations. Similar works exist in the ID-controlled image generation domain, like Arc2Face (Papantoniou et al., 2024), FullID (Guo et al., 2024), and FaceID IP-Adapter (Ye et al., 2023). In contrast, our method targets structural vulnerabilities in the template generation process itself. Since this conversion often utilizes rule-based transformations independent of specific deep models, the attack surface differs from embedding-based reconstruction. Another category of MIA, which is closer to the original definition (Fredrikson et al., 2015), such as the ones prevented by Wang et al. (2024b), aims to reconstruct the training dataset to compromise identity privacy in the training set. This focus on training data deviates from our objective of protecting individual user templates. Existing solutions for training-level privacy include FaceMAE (Wang et al., 2022) and the use of synthetic data (Bae et al., 2022).

Additionally, it is also important to distinguish PPFR systems from face anonymization (or De-Identity) systems. PPFR aims to verify an individual’s identity without exposing private information, whereas face anonymization seeks to completely replace a true identity with a synthetic or virtual one, typically for dataset sharing or social media posting. Ideally, the de-IDed face should still look like a face, but with a completely different identity, and cannot be used for recognition. PPFR also differs from invertible anonymization, as PPFR explicitly requires templates to be non-invertible. Even during normal usage, such as verification, inversion should never occur, as this would expose the user’s identity to the curious or malicious service provider. Certain works (Wang et al., 2024a; Alam et al., 2025) fail to distinguish between these two objectives, leading to secondary complications which are addressed in the Section 9 section. This fundamental difference in objectives justifies the exclusion of anonymization-based methods from our comparative analysis.

**Our Contribution:** This paper proposed a new attack to expose a fundamental misalignment between prevailing reconstruction-based evaluation paradigms and the realistic objectives of an attacker. Moving beyond the outdated assumption that preventing pixel-level reconstruction equates

to security, we introduce and validate an “identity-centric” leakage evaluation standard. We demonstrate that identity security in PPFR is far more fragile than suggested by pixel-level metrics; identity information can be systematically exploited for linkage and regeneration even when the original image remains “unreconstructable” under legacy metrics like PSNR or SSIM. Our primary contribution lies in shifting the realm of privacy analysis from visual metrics to identity metrics. By showing that even recent SOTA methods like FracFace (NeurIPS 2025) are vulnerable under our FaceLinkGen analysis, we argue that the current path of “protection by visual distortion” is inherently insufficient. Thus, this work is not a mere failure analysis, but a call for a new, identity-centric security standard that defines the next generation of privacy-preserving face recognition.

## 2. Threat Model

We assume the attacker, which may be an external attacker or a curious or malicious recognition service provider <sup>1</sup>, has oracle access to the conversion process: the ability to query it with arbitrary inputs and observe outputs, but no knowledge of its internal architecture or parameters. This assumption is obviously realistic to malicious or curious service providers (which have complete knowledge of the system) but also realistic to outsiders because the conversion process runs locally on the user’s device in all evaluated systems (Mi et al., 2023; 2024; Dai et al., 2025). An external attacker operating in their own environment can directly invoke this function or intercept protected templates via packet capture without reverse engineering the application. From a regulatory perspective, requiring raw face images to be uploaded to a remote server is considered even more problematic because one of the purposes of PPFR is to protect the user against curious or malicious recognition service providers.

This threat model is strictly more constrained than prior PPFR evaluations. Mi et al. (2024) assumes the attacker knows the conversion architecture but not the random channel selection parameters, and in the black box setting of Mi et al. (2023), the attacker has access to the conversion process but not the selected channel IDs. By contrast, we assume no knowledge of the architecture, parameters, or hyperparameters, treating the conversion process purely as a black box oracle.

Although our model assumes oracle access, it imposes stricter constraints than some prior black box attacks. For example, Zhang et al. (2024) requires approximately 6,900 online verification queries per identity and depends on the server returning continuous similarity scores for optimization.

<sup>1</sup>Recent work has shifted the focus more towards external attackers, while we argue that the curious service provider is the main adversary in the PPFR threat model.

tion. This approach is highly susceptible to rate limiting and fraud detection systems, which typically throttle or lock access after a small number of failed attempts. Many deployed FR systems return only binary accept or reject decisions or quantized similarity scores, rendering gradient-based optimization infeasible (National Cryptologic Centre, 2011; The BioAPI Consortium, 2001). This is widely known as a common, no-cost approach to prevent “hill climbing” attacks. By contrast, our method exploits the locally available conversion process, which can be queried offline without constraints and without reliance on server-side behavior.

In Section 6, we also test our method in a different, minimal assumption setting, where the attacker cannot query the conversion process in batches but only has a few (30) known pairs and does not have access to the server for similarity metrics.

## 3. Methods

The simplicity of our method is intentional. We show that even strong protection methods fail with a simple, standard distillation process, proving that the vulnerability resides in the representation itself. To formulate this, we consider a face image  $X$  as a combination of identity information  $z_I$  and non-identity (nuisance) information  $z_N$ , such that  $X \sim p(\cdot | z_I, z_N)$ .

In transformation-based PPFR systems, a protected template  $T$  is generated to hide the visual data of  $X$  while retaining identity utility. This process can be viewed as a lossy mapping that suppresses the information quantity of  $z_N$  while preserving  $z_I$ :

$$T \sim p(\cdot | z_I). \quad (1)$$

Existing evaluations often equate privacy with the failure of pixel-level reconstruction. However, since  $z_N$  is largely discarded, recovering the original pixels  $X$  is a severely ill-posed problem. Conventional attacks fail because they attempt to optimize for specific nuisance factors (e.g., exact lighting or pose) that no longer exist in  $T$ , resulting in blurry or identity-inconsistent outputs.

Our approach, FaceLinkGen, instead focuses on extracting the remaining identity information. We use a distillation-style procedure to align the template domain with a standard identity embedding space. Given a public dataset, we train a student model  $f_s$  to recover an identity representation  $z'_I$  from  $T$ . The training objective is to maximize the cosine similarity between the student’s output and the embeddings  $z_I$  extracted by a frozen teacher model  $f_t$  from original images  $X$ :

$$\mathcal{L} = 1 - \frac{1}{N} \sum_{k=1}^N s(f_s(t_k), f_t(i_k)), \quad (2)$$

where  $z'_I = f_s(T)$  is the identity feature recovered by the

165 attacker.

166 Once  $z'_J$  is extracted, the attack bypasses the need for pixel  
 167 reconstruction by leveraging a diffusion-based generative  
 168 model  $g_{\text{diff}}$ . Rather than trying to find the original  $z_N$ , we  
 169 substitute the missing information by sampling from the  
 170 model's stochastic noise  $\epsilon$ :

$$\epsilon \sim \mathcal{N}(0, I), \quad Y = g_{\text{diff}}(z'_J, \epsilon). \quad (3)$$

174 In this process, we do not attempt to recover the discarded  
 175 original nuisance factor  $z_N$ . Instead, we extract the identity  
 176 representation  $z'_J = f_s(T)$  from the template and introduce  
 177 a stochastic noise vector  $\epsilon \sim \mathcal{N}(0, I)$  as a proxy for a newly  
 178 sampled set of non-identity factors  $z'_N$ . This enables the  
 179 model to bypass the ill-posed reconstruction of the origi-  
 180 nal  $z_N$  by providing the necessary information density to  
 181 synthesize a realistic face  $Y$  with the same identity but dif-  
 182 ferent attributes. ( $z_N \not\approx z'_N$ ) Our formulation demonstrates  
 183 that as long as the information quantity of  $z_J$  persists in  
 184  $T$ , an attacker can recover  $z'_J$  and combine it with a ran-  
 185 dom  $\epsilon$  to regenerate a high-fidelity, identity-consistent face.  
 186 This structural vulnerability implies that visual distortion  
 187 does not provide meaningful identity privacy, as the specific  
 188 recovery of  $z_N$  is not a prerequisite for a successful attack.

189 Note that our attack method is independent of the specific  
 190 face recognition model employed by the PPFR server. We  
 191 use ArcFace as the teacher and student network solely be-  
 192 cause it is a widely-adopted, publicly available facial embed-  
 193 ding model, and its compatibility with the Arc2Face genera-  
 194 tive model simplifies our demonstration. Other embedding  
 195 models like FaceNet (Schroff et al., 2015) can also be used  
 196 as long as there is a compatible generative model. The  
 197 server's actual recognition backbone could be any commer-  
 198 cial or proprietary model. Our attack only requires that the  
 199 protected template retains identity-discriminative features  
 200 that are learnably aligned with some facial embedding space  
 201 accessible to the attacker, a condition implicitly assumed by  
 202 any PPFR system that aims to preserve recognition utility.

## 4. Attack Vectors

### 4.1. Linkage Attack

208 A linkage attack aims to associate a real-world identity (e.g.,  
 209 a public face image) with a protected identity, or to link  
 210 two protected templates belonging to the same individual  
 211 across different leaked databases. The first case is referred  
 212 to as face-to-template linkage, while the second is template-  
 213 to-template linkage. ISO/IEC 24745 explicitly requires  
 214 resistance against template-to-template search, but does not  
 215 address face-to-template search. This is likely a utility trade  
 216 off for the verification needs of the service provider. This  
 217 attack vector is similar to an attack vector for hashing: when  
 218 the input space is known, an attacker can enumerate all  
 219

possible inputs and map each hashed output back to its  
 original input.

In both attack scenarios, the attacker first obtains a query  
 embedding  $e_q$ . This embedding can be extracted using ei-  
 ther the student model  $f_s$  or the teacher model  $f_t$ , depending  
 on the domain of the query data. The adversary then com-  
 putes embeddings for all protected templates in the leaked  
 database using  $f_s$  and performs a nearest-neighbor search.  
 This process can be written as

$$\arg \max_{t \in T} s(e_q, f_s(t)), \quad (4)$$

where  $s(\cdot, \cdot)$  denotes cosine similarity.

### 4.2. Regeneration Attack

As discussed earlier, reconstructing the original enrollment  
 image is unnecessary and, in many cases, impossible. How-  
 ever, once a universal face embedding (e.g., ArcFace) can be  
 extracted from a protected template, modern face generation  
 models can be leveraged. In this work, we use Arc2Face,  
 which takes a facial embedding as input and generates a face  
 image whose embedding matches the input. This allows us  
 to synthesize a realistic face corresponding to the protected  
 template without reconstructing the original image.

## 5. Experiments and Results

We selected three SOTA or near-SOTA work that has their  
 accessible source code, PartialFace (Mi et al., 2023) from  
 ICCV 2023, MinusFace (Mi et al., 2024) from CVPR 2024,  
 and FracFace (Dai et al., 2025) from NeurIPS 2025. For  
 distillation, we used a subset of CASIAWebFace (Yi et al.,  
 2014) with around 10K identities and 90K images. The  
 facial embedding model is Antelopev2 with one additional  
 3x3 Conv2D layer added before to be compatible with dif-  
 ferent template formats (channel numbers) if needed. The  
 Antelopev2 is chosen because it is what Arc2Face accepts.  
 The dataset is split into a training and a validation set in an  
 80-20 ratio. For regeneration testing, we used three datasets.  
 The validation hold-out set of CASIA-WebFace, "this per-  
 son does not exist" (TPDNE) dataset<sup>2</sup>, and Labelled Face  
 in the Wild (LFW) dataset. The hold-out set is used to  
 test the ability of our method in real images while ensur-  
 ing no ID duplications. The LFW dataset is used to test  
 the cross-dataset performance of our method with distribu-  
 tion shift, and the TPDNE is used as a synthetic dataset  
 to avoid data cross-contamination from Stable Diffusion  
 1.5 and Arc2Face training data. Compared to the hold-out  
 set and LFW, photos in the TPDNE dataset are also closer  
 to a headshot, which is what is usually used to create the  
 protected templates.

<sup>2</sup>TLonoidas/this-person-does-not-exist

![Figure 4: Diagram illustrating the training, linkage, and re-generation processes of FaceLinkGen. Training: An Original Face and a Protected Template are processed by a Teacher Model (Frozen) and a Student Model (Training) respectively. The Teacher Model's output is used to initialize the Student Model via Cosine Similarity Loss. Linkage: A Query Face and a Template Database are processed by a Teacher Model and a Student Model respectively. The Teacher Model's output is used to query the Student Model's output, which is stored in an Embedding Bank. Re-generation: A Protected Template is processed by a Student Model to generate an Embedding. This Embedding is then used by a Stable Diffusion Model with an IP-Adapter to generate a Re-Generated Face from an Original Face.](690fce4fb5c9cbb8beb560cb2a3fcbeb_img.jpg)

Figure 4: Diagram illustrating the training, linkage, and re-generation processes of FaceLinkGen. Training: An Original Face and a Protected Template are processed by a Teacher Model (Frozen) and a Student Model (Training) respectively. The Teacher Model's output is used to initialize the Student Model via Cosine Similarity Loss. Linkage: A Query Face and a Template Database are processed by a Teacher Model and a Student Model respectively. The Teacher Model's output is used to query the Student Model's output, which is stored in an Embedding Bank. Re-generation: A Protected Template is processed by a Student Model to generate an Embedding. This Embedding is then used by a Stable Diffusion Model with an IP-Adapter to generate a Re-Generated Face from an Original Face.

Figure 4. After our model is trained, two main attack vectors can be performed: linkage and re-generation.

The distillation process was completed in under two hours on a single NVIDIA A6000 GPU for each of the three evaluated methods, at an estimated cost of approximately USD 0.80 to 1.60. The same process can also be executed on consumer GPUs such as the RTX 4090 or RTX 5090 with comparable wall-clock time; in fact, it is theoretically faster on the RTX 5090 due to the newer architecture of the RTX 5090. This extreme low costs are deliberate: they serve as a lower-bound analysis demonstrating that current protection mechanisms succumb to a lightweight, generic distillation without requiring complex adversarial optimization. Consequently, the attack can be carried out with minimal computational resources, undermining any assumptions that identity-level extraction is impractical or lacks a realistic threat.

### 5.1. Linkage Attack

Since all templates are converted to the standard ArcFace domain, we can not only link between original images and protected templates, but also link between two templates from the same or different protection methods. In any case, we are linking two different images (or templates) of the same person, not a face image and its corresponding template. We used the CASIA-Webface hold-out set for the linkage attack to ensure no identity overlap between training and testing identities. The hold-out dataset size is 2115.

The closed-set 1-to-N linkage results are in Table 4. The original-image-to-original-image linkage (0.88) establishes a performance upper bound. With the WebFace dataset containing 9.3%-13.0% noise (Wang et al., 2018), perfect linkage is impossible regardless of method. Our attack achieves linkage success rates consistently above 70%, frequently exceeding 80%, essentially reaching the dataset’s theoretical maximum performance. This confirms that the extracted embeddings function as effective identity descriptors for cross-domain matching. Additionally, the 1-to-1 verification accuracy used in the traditional face recognition benchmark (Table 3) remains near 100% and comparable to the original ArcFace performance, demonstrating that

Table 2. Comparison of protection claims. FracFace measures the proportion of distorted frequency channels; our metric measures identity recovery success via commercial verification. We demonstrate that high channel disruption, as reported in prior work, does not prevent identity extraction.

|                                        | Success@5 | Pass@1e-5 | Pass@1e-4 | Pass@1e-3 |
|----------------------------------------|-----------|-----------|-----------|-----------|
| <i>Dataset: TPDNE</i>                  |           |           |           |           |
| PartialFace                            | 1.000     | 0.993     | 0.996     | 0.998     |
| MinusFace                              | 0.996     | 0.936     | 0.970     | 0.989     |
| FracFace                               | 0.992     | 0.904     | 0.957     | 0.985     |
| <i>Dataset: CASIA-WebFace Hold-Out</i> |           |           |           |           |
| PartialFace                            | 0.992     | 0.957     | 0.970     | 0.982     |
| MinusFace                              | 0.989     | 0.930     | 0.958     | 0.978     |
| FracFace                               | 0.991     | 0.920     | 0.950     | 0.977     |
| <i>Dataset: LFW</i>                    |           |           |           |           |
| PartialFace                            | 0.988     | 0.980     | 0.983     | 0.986     |
| MinusFace                              | 0.987     | 0.974     | 0.981     | 0.983     |
| FracFace                               | 0.979     | 0.943     | 0.961     | 0.970     |

Table 3. 1-to-1 verification accuracy between template-to-face and face-to-face on LFW.

|                     | Accuracy | AUROC |
|---------------------|----------|-------|
| MinusFace-to-Face   | 0.992    | 0.995 |
| FracFace-to-Face    | 0.988    | 0.993 |
| PartialFace-to-Face | 0.992    | 0.996 |
| Face-to-Face        | 0.998    | 0.998 |

the protection systems fail to meaningfully impede identity matching.

### 5.2. Regeneration Attack

For the regeneration attack, we evaluate identity recovery on the first 1,000 images from each of the following datasets: the TPDNE dataset, the hold-out set of CASIA-WebFace, and the LFW dataset. Each image is converted into a protected template and mapped to a facial embedding using the student model. For each embedding, five images are generated using Arc2Face to account for stochasticity. We report both per-image success rate and Success@5. Face genera-

Table 4. Linkage Results between MinusFace, PartialFace, FracFace, and Original Image Embeddings on CASIA-Webface dataset. The numbers reported are top-1 recall at a closed set settings.

| Query       | Key      |           |             |          |
|-------------|----------|-----------|-------------|----------|
|             | FracFace | Minusface | Partialface | Original |
| FracFace    | 0.7863   | 0.7537    | 0.8137      | 0.8478   |
| Minusface   | 0.7305   | 0.7206    | 0.7754      | 0.8132   |
| Partialface | 0.8028   | 0.7868    | 0.8270      | 0.8572   |
| Original    | 0.8444   | 0.8241    | 0.8563      | 0.8823   |

Table 5. Cross-validation results (pass rate) using the Amazon API

| PartialFace | MinusFace | FracFace |
|-------------|-----------|----------|
| 0.99        | 0.98      | 0.92     |

tion is highly efficient due to the small backbone of SD1.5: generating a batch of five images for a single embedding takes approximately three seconds on an NVIDIA A6000 GPU, corresponding to a throughput of roughly 1,200 identities per hour and an estimated cost of \$0.0005 per identity generation. Visual examples are shown in Figure 1.

Following the evaluation protocol of CanFG, we employ a commercial face verification system, Face++, to assess identity consistency between the original dataset image and each generated face. We used Face++, marketed to have “financial-grade security standards,” which is usually a higher standard (e.g., more challenging for us) than many open-source methods. This also avoids using the same model (e.g., ArcFace) or models trained on the same datasets (most open source ones) for both embedding extraction and verification. Face++ outputs a confidence score together with three operating thresholds corresponding to error rate of  $1 \times 10^{-3}$ ,  $1 \times 10^{-4}$ , and  $1 \times 10^{-5}$ . For each generated image, we record the strictest threshold at which the identity match is accepted and use this as the evaluation outcome. If no face is detected in the original image, it is excluded from the data, while if no face is detected in the generated image, it counts as a failure.

The results are summarized in Table 2. On all three datasets, the success rate at the first attempt is all higher than 97%, and the success rate for five attempts ranges from 97.9% to 100%. Even at the strictest threshold, the success rate is still above 90%. We directly compare our regeneration attack with the original reconstruction attack protocol used by FracFace’s authors. In Table 6, we reported the protection rates claimed in FracFace under its own evaluation and the corresponding rates under our attack on the TPDNE dataset.

The evaluation of attack success in FracFace (Dai et al., 2025) is based on a Protection (%) metric, defined as the proportion of frequency-domain channels that are filtered or structurally disrupted. This formulation establishes a

lower barrier for defensive claims than our identity-centric standard, which requires successful regeneration of images passing commercial-grade verification. By our metric, the protection rate of most recent PPFR methods in U-Net/StyleGAN attack is almost always 100%. Despite our stricter criterion being unfavorable to reported attack success, we show that high channel protection does not prevent identity leakage: even when FracFace claims high protection under its frequency-domain metric, FaceLinkGen achieves near-total identity recovery. This also shows that channel distubution does not mean identity protection.

To cross-verify this result, we used another commercial facial comparison API from Amazon through EdenAi on 700 selected images on the LFW dataset. The Amazon API only provides a single pass/fail decision with a confidence score; the results are shown in Table 5. The values are close to the Face++ results, validating our claims.

To rule out the dependence on models like Arc2Face or third-party verification services like Face++ or Amazon, we compared the similarity of the extracted embeddings with the original face. As detailed in Section 7, the embedding extracted from a protected template shows higher cosine similarity to its source image than to another image of the same person.

## 6. What If the Attacker Knows Almost Nothing?

Our main results demonstrate that identity information can be reliably extracted when the attacker has access to the conversion process, following the threat models in prior PPFR evaluations (Mi et al., 2024; 2023; Dai et al., 2025). However, we further pose a more provocative question: To what extent does this vulnerability remain when the attacker, whether an external intruder or a malicious insider with restricted system access, has neither access to the conversion process nor any knowledge about it?

To investigate this, we consider an extreme, minimal-assumption scenario. In this setting, the attacker possesses only 30 paired image-template samples for validation (not for training) and has zero knowledge of the underlying protection mechanisms. This is actually stricter than the “black-box” scenario in Mi et al. (2023), which assumes the attacker knows the conversion process but not the channel parameters, and more realistic than Zhang et al. (2024), which relies on thousands of server queries per identity. In real attacks, these 30 pairs can be obtained by a small number of leaked samples, known identities, or attacker-controlled accounts. It can also be simulated with low-frequency queries to the authentication server.

We observe that despite their claimed algorithmic complexity, the output templates of these systems share a common

**Table 6.** Comparison with FracFace defensive claims. While FracFace (Dai et al., 2025) measures protection by frequency channel disruption, we evaluate actual identity leakage. High disruption rates fail to prevent extraction, as FaceLinkGen achieves near-total recovery through commercial-grade verification.

|                               | Venue        | Protection Tested in FracFace (Dai et al., 2025) | Protection Tested Using Our Method | Protection Tested Using Our Method (5 trials) |
|-------------------------------|--------------|--------------------------------------------------|------------------------------------|-----------------------------------------------|
| PartialFace (Mi et al., 2023) | ICCV 2023    | 0.680                                            | 0.002                              | 0.000                                         |
| MinusFace (Mi et al., 2024)   | CVPR 2024    | 0.850                                            | 0.011                              | 0.004                                         |
| FracFace (Dai et al., 2025)   | NeurIPS 2025 | 1.000                                            | 0.015                              | 0.008                                         |

**Table 7.** Regeneration Success@5 on Face++ and Amazon API and 1-to-1 Linkage Success Rate In Assumption-Constrained Settings.

| Method      | Face++ | Amazon API | Matching |
|-------------|--------|------------|----------|
| FracFace    | 0.946  | 0.473      | 0.949    |
| PartialFace | 0.946  | 0.447      | 0.925    |
| MinusFace   | 0.963  | 0.570      | 0.962    |

visual essence: they all preserve high-frequency information while obfuscating low-frequency information. Based on this intuition, the attacker can bypass any system-specific modeling and instead use a generic Gaussian-blur-based high-pass filter as a universal proxy task. We avoided using DCT or DWT to decouple from the methods used in the tested systems. By subtracting a slightly blurred version from the original image and applying simple data augmentations (e.g., varying kernels and strengths), the attacker trains a student model to align this simple high-pass domain with the identity embedding space. The attacker does not need to know any details about the system; instead, the high-pass characteristic is easily observable visually.

The training process is the same as our main text, with simply the known PPFR conversion process replaced with a high-pass filter. During inference, the templates are directly fed into the student model except for MinusFace, for which it is passed through a Gaussian-blur-based high-pass filter to remove low-frequency noise.

We trained only one model to attack all three methods. As shown in Table 7, identity leakage remains strikingly persistent. For all three systems, we achieved over 92% 1-to-1 matching success rate and over 94% re-generation success@5 on Face++ and about 44-57% on the Amazon API. The Amazon API is likely more strict or sensitive to AI-generated images. Notably, the 1-to-1 linkage success rate on LFW remains around 92-96% (Table 7), close to our main experiments. Selected re-generated results are shown in Figure ??. Even though they are slightly worse than Figure 1, they are still very close to the original face.

These results indicate that identity-consistent regeneration and reliable linkage remain feasible even under extremely constrained attacker assumptions. This suggests that the evaluated PPFR methods, despite their claimed algorithmic complexity, share a common vulnerability: their output

representations exhibit strong coupling with simple high-pass filtering operations. Whether this generalizes to future methods that do not rely on frequency-domain obfuscation remains an open question.

Overall, these findings reveal a deeply concerning reality: even under an extreme near-zero-knowledge regime, identity information remains robustly and systematically extractable. This demonstrates that identity leakage is not a byproduct of specific model designs, training strategies, or attacker assumptions, but rather a structural property of existing PPFR representations themselves. Consequently, restricting access to the conversion process offers little meaningful protection. As long as the released templates preserve recognition utility, they inevitably encode recoverable identity cues, rendering current conversion-based defenses fundamentally insufficient.

## 7. Similarity Distribution

To directly quantify identity leakage from protected templates, independent of the downstream face generation process (Arc2Face) or the specific behavior of commercial verification APIs (Face++, Amazon), we analyze the cosine similarity in the standard ArcFace embedding space: the similarity between two normal images of the same person, and the similarity between one image and its protected template. The histogram is shown in Figure 6 in the Appendix. Due to the dataset noise, some real photos and their similarity is near 0 (See (Wang et al., 2018)); this also appears in the ArcFace paper (Deng et al., 2022), but we focus on the main cluster here. We tested this on our testing set. We observed that the similarity between the original image and its template is higher than the similarity between another image of the same person in all three methods. This means that the template is a better identity descriptor for this specific image than another image of the same person. Note that this does not imply that protected templates are universally closer to the underlying identity than real images. Instead, the template remains most similar to its corresponding source image. This indicates that the template is an image-conditioned projection that preserves identity while retaining instance-specific bias, rather than a global identity prototype.

## 8. Soft Identity Leakage: Beyond Unique Identifiers

Beyond hard identity recognition, the exposure of soft biometric attributes presents a significant privacy risk. Characteristics such as skin color, age, and gender are sensitive personal data that enable unauthorized profiling and algorithmic discrimination. Privacy frameworks like the Canadian Privacy Act (Branch Legislative Services, 2025) explicitly protect race and age. A robust privacy-preserving system must therefore prevent the recovery of these attributes from its templates.

In the public rebuttal of FracFace (Dai et al., 2025) on OpenReview (noa), the authors claimed successful obfuscation of age, gender, and ethnicity. They cited a human perception study where participants reported less than 13% usage of these biometrics for identity inference, with over 76% of participants relying on guesswork. However, our empirical results in Figure 1 demonstrate that these soft biometrics remain visible in the re-generated faces. This matches with previous research, which indicates that ArcFace embeddings retain such information (Melzi et al., 2023; Osorio-Roig et al., 2023). Because our extracted embeddings closely resemble the original facial embeddings, we hypothesize that a model can learn a direct mapping from the embedding to these attributes without facial reconstruction.

To test this, we trained MLP models on the FairFace dataset (Kärkkäinen & Joo, 2019) to predict age, gender, and race (7 classes) across 500 test images. Table 8 shows that gender is identified with at least 82% accuracy, and the Age MAE ranges from 6.1 to 7.5 years. The race accuracy is lower, ranging from 0.50 to 0.60, but considering the number of classes in race, it is still remarkable leakage, as there are about 50% of the time the race could be recovered. Given that FairFace labels use 10-year intervals, an MAE below 10.0 suggests leakage that matches the inherent precision of the reference model. Our attack reaches comparable metrics to the FairFace model on some datasets like LFWA+ (Liu et al., 2015).

Table 8. Soft Biometrics Leakage

| Method      | Race Acc $\uparrow$ | Gender Acc $\uparrow$ | Age MAE $\downarrow$ |
|-------------|---------------------|-----------------------|----------------------|
| FracFace    | 0.50                | 0.82                  | 7.5                  |
| MinusFace   | 0.56                | 0.86                  | 6.4                  |
| PartialFace | 0.60                | 0.88                  | 6.1                  |

Some methods, such as CanFG (Wang et al., 2024a) and FaceAnonyMixer (Alam et al., 2025), intentionally preserve soft biometrics for auxiliary tasks. We contend that this design is problematic. PPFR systems should rely only on identity-discriminative features. Retaining soft biometrics offers attackers more data for reconstruction and profiling without improving recognition performance. These

attributes are central to privacy frameworks and require stricter protection (Osorio-Roig et al., 2022).

## 9. Future Directions

We suggest several potential pathways for future PPFR designs and evaluations, primarily focusing on stronger defensive mechanisms and broader vulnerability assessments.

One rigorous approach is to incorporate secret keys into the conversion process, similar to Yuan et al. (2022). This serves as a multi-factor authentication system (requiring both biometrics and a key), preventing attackers—and our method—from converting a face without the secret. Alternatively, systems may revert to formal cryptographic methods like (Ao & Boddeti, 2025). While traditionally viewed as computationally expensive, modern resources make this trade-off acceptable; for instance, Jindal et al. (2020) reports only 2.83ms processing time per face pair. Importantly, these computational costs can act as an effective client-side constraint against brute-force attacks (conceptually similar to slow hashing), enhancing privacy while remaining imperceptible to regular users. Again, we want to emphasize that de-ID or reversible face encryption methods *cannot* be used for PPFR tasks, as they either make the face completely useless for recognition or the reversibility compromised the privacy-preserving nature.

## 10. Conclusion

This paper demonstrates that current frequency-based obfuscation methods fail to meet the fundamental security requirements of PPFR systems. We prove that identity-discriminative information remains accessible at the representation level, allowing for high-accuracy linkage and facial regeneration. Crucially, our findings reveal that such systems provide negligible protection against malicious service providers who, despite having legitimate authorization for verification, can systematically revert protected templates to identifiable facial data. The success of FaceLinkGen—even under minimal knowledge assumptions—exposes a structural collapse of the visual distortion paradigm for external attackers. Future PPFR research must move beyond human-centric visual metrics toward mathematically asymmetric protection mechanisms that effectively prevent unauthorized semantic extraction by all potential adversaries, including the service providers themselves.

## 11. Impact Statements

This work studies privacy-preserving face recognition (PPFR) systems, a domain that inherently involves sensitive biometric technologies. All the datasets used in the paper are common academic datasets or synthetic datasets. By identifying weaknesses in prevailing PPFR designs and evaluation practices, our goal is to highlight limitations of current security assumptions and to motivate the development of stronger protection mechanisms and evaluation paradigms that better safeguard user identity privacy. The systems analyzed in this paper are primarily academic methods proposed in the literature, rather than deployed industrial systems. As such, our findings are intended to inform research directions and evaluation standards, rather than to characterize the security posture of specific real-world deployments.

We informed the authors of the three targeted works prior to submission. To mitigate the risk of weaponization, we withhold full attack code, trained models, and specific implementation details from public repositories. This decision follows established precedents in both high-stakes AI releases (Siméoni et al., 2025; Grattafiori et al., 2024) and offensive security research (Saharia et al., 2022; Bagwe et al., 2025; Zhao et al., 2025; Kumar et al., 2025; Zhang et al., 2025; Crețu et al., 2022), where the potential for misuse outweighs unconditional transparency.

We decline to disclose specific technical details due to significant safety and security concerns. While scientific reproducibility is a core principle, the right to privacy and life must take precedence over the reproducibility of high-risk protocols. Sharing detailed instructions for attack poses a severe threat to global privacy and safety if misused. We believe that safeguarding human privacy outweighs the requirement for full technical transparency in this instance. This decision ensures that scientific progress does not create unnecessary vulnerabilities for society. The information already disclosed provide enough information for scientific validation without revealing sensitive data that could facilitate unauthorized replication.

To balance verification with harm control, we provide high-level algorithmic descriptions sufficient for conceptual understanding while restricting access to the operational tooling. We offer two channels for rigorous scientific validation: (1) direct sharing of models and code with qualified researchers upon verified request, and (2) a planned security testing server. This server will allow researchers to submit templates for identity leakage assessment under a controlled protocol without exposing the underlying attack payloads. This measured disclosure strategy ensures our findings can be peer-validated and used to harden future systems without providing a turnkey weapon to malicious actors.

Our work also demonstrates identity-consistent face regeneration, which may raise concerns regarding the ethical risks of controllable deepfake generation. We emphasize that controllable deepfake synthesis is not a novel contribution of this paper. Our pipeline builds upon Arc2Face-generated images and existing image generation models, and the ethical implications of such techniques have been extensively discussed in prior literature. Our contribution lies in showing that such generation can be triggered from protected templates, rather than in advancing deepfake generation capabilities themselves.

## References

- 495 **References**
- 496
- 497 Wayback Machine. URL [https://web.archive.](https://web.archive.org/web/20260000000000/*)
- 498 [org/web/20260000000000/\\*](http://web/20260000000000/*); [https://](https://openreview.net/forum?id=JSSvYZKvL8)
- 499 [//openreview.net/forum?id=JSSvYZKvL8](http://openreview.net/forum?id=JSSvYZKvL8).
- 500 Alam, M. T., Shamshad, F., Karray, F., and Nandakumar, K.
- 501 FaceAnonyMixer: Cancelable Faces via Identity Consis-
- 502 tent Latent Space Mixing, August 2025. URL [http://](http://arxiv.org/abs/2508.05636)
- 503 [arxiv.org/abs/2508.05636](http://arxiv.org/abs/2508.05636). arXiv:2508.05636
- 504 [cs].
- 505
- 506 Ao, W. and Boddeti, V. N. CryptoFace: End-to-End
- 507 Encrypted Face Recognition. In *Proceedings of the*
- 508 *Computer Vision and Pattern Recognition Conference*, pp.
- 509 19197–19206, 2025. URL [https://openaccess.](https://openaccess.thecvf.com/content/CVPR2025/html/Ao_CryptoFace_End-to-End_Encrypted_Face_Recognition_CVPR_2025_paper.html)
- 510 [thecvf.com/content/CVPR2025/html/Ao\\_](https://openaccess.thecvf.com/content/CVPR2025/html/Ao_CryptoFace_End-to-End_Encrypted_Face_Recognition_CVPR_2025_paper.html)
- 511 [CryptoFace\\_End-to-End\\_Encrypted\\_Face\\_](https://openaccess.thecvf.com/content/CVPR2025/html/Ao_CryptoFace_End-to-End_Encrypted_Face_Recognition_CVPR_2025_paper.html)
- 512 [Recognition\\_CVPR\\_2025\\_paper.html](https://openaccess.thecvf.com/content/CVPR2025/html/Ao_CryptoFace_End-to-End_Encrypted_Face_Recognition_CVPR_2025_paper.html).
- 513
- 514 Bae, G., Gorce, M. d. L., Baltrusaitis, T., Hewitt, C., Chen,
- 515 D., Valentin, J., Cipolla, R., and Shen, J. DigiFace-1M:
- 516 1 Million Digital Face Images for Face Recognition, Oc-
- 517 tober 2022. URL [http://arxiv.org/abs/2210.](http://arxiv.org/abs/2210.02579)
- 518 [02579](http://arxiv.org/abs/2210.02579). arXiv:2210.02579 [cs].
- 519
- 520 Bagwe, G., Chaturvedi, S. S., Ma, X., Yuan, X., Wang,
- 521 K.-C., and Zhang, L. E. Your RAG is Unfair: Ex-
- 522 posing Fairness Vulnerabilities in Retrieval-Augmented
- 523 Generation via Backdoor Attacks. In Christodoulopou-
- 524 los, C., Chakraborty, T., Rose, C., and Peng, V. (eds.), *Proceedings of the 2025 Conference on Em-*
- 525 *pirical Methods in Natural Language Processing*, pp.
- 526 15930–15948, Suzhou, China, November 2025. As-
- 527 sociation for Computational Linguistics. ISBN 979-
- 528 8-89176-332-6. doi: 10.18653/v1/2025.emnlp-main.
- 529 804. URL [https://aclanthology.org/2025.](https://aclanthology.org/2025.emnlp-main.804/)
- 530 [emnlp-main.804/](https://aclanthology.org/2025.emnlp-main.804/).
- 531
- 532
- 533 Branch Legislative Services. Consolidated federal
- 534 laws of Canada, Privacy Act, June 2025. URL
- 535 [https://laws-lois.justice.gc.ca/eng/](https://laws-lois.justice.gc.ca/eng/ACTS/P-21/page-1.html#h-397182)
- 536 [ACTS/P-21/page-1.html#h-397182](https://laws-lois.justice.gc.ca/eng/ACTS/P-21/page-1.html#h-397182).
- 537
- 538 Cai, F., Guo, Y., Li, J., Li, W., Fang, X., and Chen, J. Fast-
- 539 FLUX: Pruning FLUX with Block-wise Replacement and
- 540 Sandwich Training, June 2025. URL [http://arxiv.](http://arxiv.org/abs/2506.10035)
- 541 [org/abs/2506.10035](http://arxiv.org/abs/2506.10035). arXiv:2506.10035 [cs] ver-
- 542 sion: 1.
- 543
- 544 Crețu, A.-M., Monti, F., Marrone, S., Dong, X., Bronstein,
- 545 M., and de Montjoye, Y.-A. Interaction data are identi-
- 546 fiable even across long periods of time. *Nature Commu-*
- 547 *nications*, 13:313, January 2022. ISSN 2041-1723. doi:
- 548 10.1038/s41467-021-27714-6. URL [https://pmc.](https://pmc.ncbi.nlm.nih.gov/articles/PMC8789822/)
- 549 [ncbi.nlm.nih.gov/articles/PMC8789822/](https://pmc.ncbi.nlm.nih.gov/articles/PMC8789822/).
- 550
- Dai, W., Li, B., Dong, N., Bai, G., and Dong,
- J. S. FracFace: Breaking The Visual Clues—Fractal-
- Based Privacy-Preserving Face Recognition. October
2025. URL [https://openreview.net/forum?](https://openreview.net/forum?id=JSSvYZKvL8)
- [id=JSSvYZKvL8](https://openreview.net/forum?id=JSSvYZKvL8).
- Deng, J., Guo, J., Yang, J., Xue, N., Kotsia, I., and Zafeiriou,
- S. ArcFace: Additive Angular Margin Loss for Deep Face
- Recognition. *IEEE Transactions on Pattern Analysis*
- and Machine Intelligence*, 44(10):5962–5979, October
2022. ISSN 0162-8828, 2160-9292, 1939-3539. doi:
- 10.1109/TPAMI.2021.3087709. URL [http://arxiv.](http://arxiv.org/abs/1801.07698)
- [org/abs/1801.07698](http://arxiv.org/abs/1801.07698). arXiv:1801.07698 [cs].
- Erkin, Z., Franz, M., Guajardo, J., Katzenbeisser, S., La-
- gendijk, I., and Toft, T. Privacy-Preserving Face Recog-
- nition. In Goldberg, I. and Atallah, M. J. (eds.), *Privacy*
- Enhancing Technologies*, pp. 235–253, Berlin, Heidel-
- berg, 2009. Springer. ISBN 978-3-642-03168-7. doi:
- 10.1007/978-3-642-03168-7\_14.
- Fredrikson, M., Jha, S., and Ristenpart, T. Model Inver-
- sion Attacks that Exploit Confidence Information and
- Basic Countermeasures. In *Proceedings of the 22nd*
- ACM SIGSAC Conference on Computer and Communi-*
- cations Security*, pp. 1322–1333, Denver Colorado USA,
- October 2015. ACM. ISBN 978-1-4503-3832-5. doi:
- 10.1145/2810103.2813677. URL [https://dl.acm.](https://dl.acm.org/doi/10.1145/2810103.2813677)
- [org/doi/10.1145/2810103.2813677](https://dl.acm.org/doi/10.1145/2810103.2813677).
- Grattafiori et al. The Llama 3 Herd of Models, Novem-
- ber 2024. URL [http://arxiv.org/abs/2407.](http://arxiv.org/abs/2407.21783)
- [21783](http://arxiv.org/abs/2407.21783). arXiv:2407.21783 [cs].
- Guo, W. M., Qian, Q., Hasan, K., and Du, S. Aesthetic
- Alignment Risks Assimilation: How Image Generation
- and Reward Models Reinforce Beauty Bias and Ideo-
- logical “Censorship”, December 2025. URL [http://](http://arxiv.org/abs/2512.11883)
- [arxiv.org/abs/2512.11883](http://arxiv.org/abs/2512.11883). arXiv:2512.11883
- [cs].
- Guo, Z., Wu, Y., Chen, Z., Chen, L., Zhang, P., and He,
- Q. PuLID: Pure and Lightning ID Customization via
- Contrastive Alignment, October 2024. URL [http://](http://arxiv.org/abs/2404.16022)
- [arxiv.org/abs/2404.16022](http://arxiv.org/abs/2404.16022). arXiv:2404.16022
- [cs].
- Ji, J., Wang, H., Huang, Y., Wu, J., Xu, X., Ding, S., Zhang,
- S., Cao, L., and Ji, R. Privacy-Preserving Face Recog-
- nition with Learnable Privacy Budgets in Frequency Do-
- main, July 2022. URL [http://arxiv.org/abs/](http://arxiv.org/abs/2207.07316)
- [2207.07316](http://arxiv.org/abs/2207.07316). arXiv:2207.07316 [cs].
- Jin, S., Wang, H., Wang, Z., Xiao, F., Hu, J., He, Y.,
- Zhang, W., Ba, Z., Fang, W., Yuan, S., and Ren, K.
- FaceObfuscator: Defending Deep Learning-based Pri-
- vacy Attacks with Gradient Descent-resistant Features in

- 550 Face Recognition. In *33rd USENIX Security Symposium*  
551 (*USENIX Security 24*), pp. 6849–6866, Philadelphia,  
552 PA, August 2024. USENIX Association. ISBN 978-  
553 1-939133-44-1. URL [https://www.usenix.](https://www.usenix.org/conference/usenixsecurity24/presentation/jin-shuaifan)  
554 [org/conference/usenixsecurity24/](https://www.usenix.org/conference/usenixsecurity24/presentation/jin-shuaifan)  
555 [presentation/jin-shuaifan](https://www.usenix.org/conference/usenixsecurity24/presentation/jin-shuaifan).
- 556 Jindal, A. K., Shaik, I., Vasudha, V., Chalamala, S. R., Ma,  
557 R., and Lodha, S. Secure and Privacy Preserving Method  
558 for Biometric Template Protection using Fully Homo-  
559 morphic Encryption. In *2020 IEEE 19th International*  
560 *Conference on Trust, Security and Privacy in Comput-*  
561 *ing and Communications (TrustCom)*, pp. 1127–1134,  
562 December 2020. doi: 10.1109/TrustCom50675.2020.  
563 00149. URL [https://ieeexplore.ieee.org/](https://ieeexplore.ieee.org/document/9343021)  
564 [document/9343021](https://ieeexplore.ieee.org/document/9343021). ISSN: 2324-9013.
- 565
- 566 Karras, T., Laine, S., and Aila, T. A Style-Based  
567 Generator Architecture for Generative Adversarial Net-  
568 works, March 2019. URL [http://arxiv.org/](http://arxiv.org/abs/1812.04948)  
569 [abs/1812.04948](http://arxiv.org/abs/1812.04948). arXiv:1812.04948 [cs].
- 570
- 571 Kumar, D., Birur, N. A., Baswa, T., Agarwal, S., and Har-  
572 shangi, P. Quantifying CBRN Risk in Frontier Mod-  
573 els, October 2025. URL [http://arxiv.org/abs/](http://arxiv.org/abs/2510.21133)  
574 [2510.21133](http://arxiv.org/abs/2510.21133). arXiv:2510.21133 [cs].
- 575
- 576 Kärkkäinen, K. and Joo, J. FairFace: Face Attribute  
577 Dataset for Balanced Race, Gender, and Age, Au-  
578 gust 2019. URL [http://arxiv.org/abs/1908.](http://arxiv.org/abs/1908.04913)  
579 [04913](http://arxiv.org/abs/1908.04913). arXiv:1908.04913 [cs].
- 580
- 581 Liu, Z., Luo, P., Wang, X., and Tang, X. Deep learning face  
582 attributes in the wild. In *Proceedings of International*  
583 *Conference on Computer Vision (ICCV)*, 2015.
- 584
- 585 Ma, Y., Shui, Y., Wu, X., Sun, K., and Li, H. HPSv3:  
586 Towards Wide-Spectrum Human Preference Score, Au-  
587 gust 2025. URL [http://arxiv.org/abs/2508.](http://arxiv.org/abs/2508.03789)  
588 [03789](http://arxiv.org/abs/2508.03789). arXiv:2508.03789 [cs].
- 589
- 590 Melzi, P., Shahreza, H. O., Rathgeb, C., Tolosana,  
591 R., Vera-Rodriguez, R., Fierrez, J., Marcel, S.,  
592 and Busch, C. Multi-ive: Privacy enhancement of  
593 multiple soft-biometrics in face embeddings. In  
594 *Proceedings of the IEEE/CVF Winter Conference*  
595 *on Applications of Computer Vision*, pp. 323–331,  
596 2023. URL [https://openaccess.thecvf.](https://openaccess.thecvf.com/content/WACV2023W/DVPBA/html/Melzi_Multi-IVE_Privacy_Enhancement_of_Multiple_Soft-Biometrics_in_Face_Embeddings_WACVW_2023_paper.html)  
597 [com/content/WACV2023W/DVPBA/html/](https://openaccess.thecvf.com/content/WACV2023W/DVPBA/html/Melzi_Multi-IVE_Privacy_Enhancement_of_Multiple_Soft-Biometrics_in_Face_Embeddings_WACVW_2023_paper.html)  
598 [Melzi\\_Multi-IVE\\_Privacy\\_Enhancement\\_](https://openaccess.thecvf.com/content/WACV2023W/DVPBA/html/Melzi_Multi-IVE_Privacy_Enhancement_of_Multiple_Soft-Biometrics_in_Face_Embeddings_WACVW_2023_paper.html)  
599 [of\\_Multiple\\_Soft-Biometrics\\_in\\_Face\\_](https://openaccess.thecvf.com/content/WACV2023W/DVPBA/html/Melzi_Multi-IVE_Privacy_Enhancement_of_Multiple_Soft-Biometrics_in_Face_Embeddings_WACVW_2023_paper.html)  
600 [Embeddings\\_WACVW\\_2023\\_paper.html](https://openaccess.thecvf.com/content/WACV2023W/DVPBA/html/Melzi_Multi-IVE_Privacy_Enhancement_of_Multiple_Soft-Biometrics_in_Face_Embeddings_WACVW_2023_paper.html).
- 601
- 602 Mi, Y., Huang, Y., Ji, J., Liu, H., Xu, X., Ding, S., and  
603 Zhou, S. DuetFace: Collaborative Privacy-Preserving  
604 Face Recognition via Channel Splitting in the Frequency  
605 Domain. In *Proceedings of the 30th ACM International*
- Conference on Multimedia, pp. 6755–6764, October 2022.  
doi: 10.1145/3503161.3548303. URL [http://arxiv.](http://arxiv.org/abs/2207.07340)  
[org/abs/2207.07340](http://arxiv.org/abs/2207.07340). arXiv:2207.07340 [cs].
- Mi, Y., Huang, Y., Ji, J., Zhao, M., Wu, J., Xu, X., Ding, S.,  
and Zhou, S. Privacy-preserving face recognition using  
random frequency components. In *Proceedings of the*  
*IEEE/CVF International Conference on Computer Vision*,  
pp. 19673–19684, 2023.
- Mi, Y., Zhong, Z., Huang, Y., Ji, J., Xu, J., Wang, J., Wang,  
S., Ding, S., and Zhou, S. Privacy-preserving face recog-  
nition using trainable feature subtraction. In *Proceedings*  
*of the IEEE/CVF Conference on Computer Vision and*  
*Pattern Recognition*, pp. 297–307, 2024.
- Mou, C., Wu, Y., Wu, W., Guo, Z., Zhang, P., Cheng, Y.,  
Luo, Y., Ding, F., Zhang, S., Li, X., Li, M., Liu, M.,  
Zhang, Y., Wu, S., Zhao, S., Zhang, J., He, Q., and Wu, X.  
DreamO: A Unified Framework for Image Customization,  
November 2025. URL [http://arxiv.org/abs/](http://arxiv.org/abs/2504.16915)  
[2504.16915](http://arxiv.org/abs/2504.16915). arXiv:2504.16915 [cs].
- National Cryptologic Centre. *Characterizing Attacks to*  
*Fingerprint Verification Mechanisms*. 2011.
- Osorio-Roig, D., Rathgeb, C., Drozdowski, P., Terhörst,  
J., Struc, V., and Busch, C. An Attack on Facial Soft-  
Biometric Privacy Enhancement. *IEEE Transactions on*  
*Biometrics, Behavior, and Identity Science*, 4(2):263–275,  
April 2022. ISSN 2637-6407. doi: 10.1109/TBIOM.  
2022.3172724. URL [https://ieeexplore.ieee.](https://ieeexplore.ieee.org/document/9770950)  
[org/document/9770950](https://ieeexplore.ieee.org/document/9770950).
- Osorio-Roig, D., Gerlitz, P. A., Rathgeb, C., and Busch, C.  
Reversing Deep Face Embeddings with Probable Privacy  
Protection, October 2023. URL [http://arxiv.org/](http://arxiv.org/abs/2310.03005)  
[abs/2310.03005](http://arxiv.org/abs/2310.03005). arXiv:2310.03005 [cs].
- Papantoniou, F. P., Lattas, A., Moschoglou, S., Deng,  
J., Kainz, B., and Zafeiriou, S. Arc2Face: A Foun-  
dation Model for ID-Consistent Human Faces, Au-  
gust 2024. URL [http://arxiv.org/abs/2403.](http://arxiv.org/abs/2403.11641)  
[11641](http://arxiv.org/abs/2403.11641). arXiv:2403.11641 [cs].
- Saharia, C., Chan, W., Saxena, S., Li, L., Whang, J., Den-  
ton, E., Ghasemipour, S. K. S., Gontijo-Lopes, R., Ayan,  
B. K., Salimans, T., Ho, J., Fleet, D. J., and Norouzi, M.  
Photorealistic Text-to-Image Diffusion Models with Deep  
Language Understanding. October 2022. URL [https:](https://openreview.net/forum?id=08Yk-n512Al)  
[//openreview.net/forum?id=08Yk-n512Al](https://openreview.net/forum?id=08Yk-n512Al).
- Schroff, F., Kalenichenko, D., and Philbin, J. FaceNet: A  
Unified Embedding for Face Recognition and Clustering.  
In *2015 IEEE Conference on Computer Vision and Pat-*  
*tern Recognition (CVPR)*, pp. 815–823, June 2015. doi:  
10.1109/CVPR.2015.7298682. URL [http://arxiv.](http://arxiv.org/abs/1503.03832)  
[org/abs/1503.03832](http://arxiv.org/abs/1503.03832). arXiv:1503.03832 [cs].

- 605 Siméoni, O., Vo, H. V., Seitzer, M., Baldassarre, F., Oquab,  
606 M., Jose, C., Khalidov, V., Szafraniec, M., Yi, S., Rama-  
607 monijisoa, M., Massa, F., Haziza, D., Wehrstedt, L., Wang,  
608 J., Darcet, T., Moutakanni, T., Sentana, L., Roberts, C.,  
609 Vedaldi, A., Tolan, J., Brandt, J., Couprie, C., Mairal, J.,  
610 Jégou, H., Labatut, P., and Bojanowski, P. DINOv3, Au-  
611 gust 2025. URL [http://arxiv.org/abs/2508.](http://arxiv.org/abs/2508.10104)  
612 [10104](http://arxiv.org/abs/2508.10104). arXiv:2508.10104 [cs].
- 613  
614 Sun, X., Gazagnadou, N., Sharma, V., Lyu, L., Li, H., and  
615 Zheng, L. Privacy Assessment on Reconstructed Images:  
616 Are Existing Evaluation Metrics Faithful to Human Per-  
617 ception?, October 2023. URL [http://arxiv.org/](http://arxiv.org/abs/2309.13038)  
618 [abs/2309.13038](http://arxiv.org/abs/2309.13038). arXiv:2309.13038 [cs].
- 619  
620 The BioAPI Consortium. *BioAPI Specification Version 1.1*.  
621 March 2001.
- 622  
623 Wang, F., Chen, L., Li, C., Huang, S., Chen, Y., Qian, C.,  
624 and Loy, C. C. The Devil of Face Recognition is in the  
625 Noise, July 2018. URL [http://arxiv.org/abs/](http://arxiv.org/abs/1807.11649)  
626 [1807.11649](http://arxiv.org/abs/1807.11649). arXiv:1807.11649 [cs].
- 627  
628 Wang, H., Wang, S., Lu, C.-S., and Echizen, I. Diffusion-  
629 Driven Universal Model Inversion Attack for Face Recog-  
630 nition, April 2025. URL [http://arxiv.org/abs/](http://arxiv.org/abs/2504.18015)  
631 [2504.18015](http://arxiv.org/abs/2504.18015). arXiv:2504.18015 [cs] version: 1.
- 632  
633 Wang, K., Zhao, B., Peng, X., Zhu, Z., Deng, J.,  
634 Wang, X., Bilen, H., and You, Y. FaceMAE: Privacy-  
635 Preserving Face Recognition via Masked Autoencoders,  
636 May 2022. URL [http://arxiv.org/abs/2205.](http://arxiv.org/abs/2205.11090)  
637 [11090](http://arxiv.org/abs/2205.11090). arXiv:2205.11090 [cs].
- 638  
639 Wang, T., Zhang, Y., Xiao, X., Yuan, L., Xia, Z., and  
640 Weng, J. Make Privacy Renewable! Generating  
641 Privacy-Preserving Faces Supporting Cancelable Bio-  
642 metric Recognition. In *Proceedings of the 32nd ACM*  
643 *International Conference on Multimedia*, pp. 10268–  
644 10276, Melbourne VIC Australia, October 2024a. ACM.  
645 ISBN 979-8-4007-0686-8. doi: 10.1145/3664647.  
646 3680704. URL [https://dl.acm.org/doi/10.](https://dl.acm.org/doi/10.1145/3664647.3680704)  
647 [1145/3664647.3680704](https://dl.acm.org/doi/10.1145/3664647.3680704).
- 648  
649 Wang, Y., Huang, Y., Li, J., Yang, L., Song, K., and  
650 Wang, L. Adaptive Hybrid Masking Strategy for Privacy-  
651 Preserving Face Recognition Against Model Inversion At-  
652 tack, April 2024b. URL [http://arxiv.org/abs/](http://arxiv.org/abs/2403.10558)  
653 [2403.10558](http://arxiv.org/abs/2403.10558). arXiv:2403.10558 [cs].
- 654  
655 Wu, X., Hao, Y., Sun, K., Chen, Y., Zhu, F., Zhao, R., and  
656 Li, H. Human Preference Score v2: A Solid Benchmark  
657 for Evaluating Human Preferences of Text-to-Image Syn-  
658 thesis, September 2023. URL [http://arxiv.org/](http://arxiv.org/abs/2306.09341)  
659 [abs/2306.09341](http://arxiv.org/abs/2306.09341). arXiv:2306.09341 [cs].
- Ye, H., Zhang, J., Liu, S., Han, X., and Yang, W. IP-Adapter:  
Text Compatible Image Prompt Adapter for Text-to-  
Image Diffusion Models, August 2023. URL [http://](http://arxiv.org/abs/2308.06721)  
[arxiv.org/abs/2308.06721](http://arxiv.org/abs/2308.06721). arXiv:2308.06721  
[cs].
- Yi, D., Lei, Z., Liao, S., and Li, S. Z. Learning Face Rep-  
resentation from Scratch, November 2014. URL [http://](http://arxiv.org/abs/1411.7923)  
[/arxiv.org/abs/1411.7923](http://arxiv.org/abs/1411.7923). arXiv:1411.7923  
[cs].
- Yuan, Z., You, Z., Li, S., Qian, Z., Zhang, X., and Kot,  
A. On Generating Identifiable Virtual Faces. In *Pro-*  
*ceedings of the 30th ACM International Conference*  
*on Multimedia*, pp. 1465–1473, Lisboa Portugal, Oc-  
tober 2022. ACM. ISBN 978-1-4503-9203-7. doi:  
10.1145/3503161.3548110. URL [https://dl.acm.](https://dl.acm.org/doi/10.1145/3503161.3548110)  
[org/doi/10.1145/3503161.3548110](https://dl.acm.org/doi/10.1145/3503161.3548110).
- Yue, K., Jin, R., Wong, C.-W., Baron, D., and  
Dai, H. Gradient Obfuscation Gives a False  
Sense of Security in Federated Learning. In  
*32nd USENIX Security Symposium (USENIX Secu-*  
*rity 23)*, pp. 6381–6398, Anaheim, CA, August 2023.  
USENIX Association. ISBN 978-1-939133-37-3.  
URL [https://www.usenix.org/conference/](https://www.usenix.org/conference/usenixsecurity23/presentation/yue)  
[usenixsecurity23/presentation/yue](https://www.usenix.org/conference/usenixsecurity23/presentation/yue).
- Zhang, H., Dong, X., Lai, Y., Zhou, Y., Zhang, X., Lv,  
X., Jin, Z., and Li, X. Validating Privacy-Preserving  
Face Recognition Under a Minimum Assumption. In  
*2024 IEEE/CVF Conference on Computer Vision and Pat-*  
*tern Recognition (CVPR)*, pp. 12205–12214, June 2024.  
doi: 10.1109/CVPR52733.2024.01160. URL [https://](https://ieeexplore.ieee.org/document/10655270)  
[ieeexplore.ieee.org/document/10655270](https://ieeexplore.ieee.org/document/10655270).
- Zhang, Y., Wang, L., Zhao, J., Zhao, W., Zhou, F., Dang,  
Y., and Yin, J. 3DGAA: Realistic and Robust 3D  
Gaussian-based Adversarial Attack for Autonomous Driv-  
ing, July 2025. URL [http://arxiv.org/abs/](http://arxiv.org/abs/2507.09993)  
[2507.09993](http://arxiv.org/abs/2507.09993). arXiv:2507.09993 [cs] version: 1.
- Zhao, J., Fu, T., Schaeffer, R., Sharma, M., and  
Barez, F. Chain-of-Thought Hijacking, Novem-  
ber 2025. URL [http://arxiv.org/abs/2510.](http://arxiv.org/abs/2510.26418)  
[26418](http://arxiv.org/abs/2510.26418). arXiv:2510.26418 [cs].

Table 9. Controllable Deepfake Results

| Pass@1e-5 | Pass@1e-3 | HPSv2.1 | Face++ Conf |
|-----------|-----------|---------|-------------|
| 0.892     | 0.958     | 0.283   | 82.33       |

![Figure 5: Controlled generation examples. The figure shows three columns of images labeled 'PurrrFace', 'MinusFace', and 'FracFace'. Each column contains three images: a top-left register image, a top-right generated image, and a bottom generated image. All images show a person playing a guitar in a street setting. The PurrrFace column shows a person with glasses, the MinusFace column shows a person with a hood, and the FracFace column shows a person with a scarf.](0cc86fe8fc37b0edc9581f2af9459a52_img.jpg)

Figure 5: Controlled generation examples. The figure shows three columns of images labeled 'PurrrFace', 'MinusFace', and 'FracFace'. Each column contains three images: a top-left register image, a top-right generated image, and a bottom generated image. All images show a person playing a guitar in a street setting. The PurrrFace column shows a person with glasses, the MinusFace column shows a person with a hood, and the FracFace column shows a person with a scarf.

Figure 5. Controlled generation examples. The top left image is the image used to register, the top right image is produced by Arc2Face, and the bottom image is generated by DreamO based on the top right image. All images are prompted with “a person playing guitar in the street” and all are verified by Face++ with a FAR of 1e-5. Even with blurry register images, our method still works.

## A. Controllable Deepfake

This enables direct deepfake creation from the “protected” templates. Arc2Face can only generate a face image from facial embeddings without textual guidance. FaceID IP-Adapter (Ye et al., 2023) allows the user to generate a face with both text and face prompts; however, its quality and face consistency are worse. Recently, models do not use facial embeddings, but raw image input directly (or both) has achieved high quality and ID consistency, such as DreamO (Mou et al., 2025) and PulID (Guo et al., 2024). However, aligning another student encoder for them might be more challenging. Thus, we use a simple pipeline to feed the face(s) generated by Arc2Face as prompts for other downstream models. If the attacker is willing to trade some regeneration accuracy for controllability, they can use this pipeline to generate the person’s face in any setting by using a textual prompt. In Figure 5, we have shown examples of the results of this pipeline. We can see that even after 2 transformations, the generated face in a controlled setting is still highly similar to the original image.

We selected FracFace as the large-scale attack target because it is the most recent model and yields our lowest attack success rate, allowing it to serve as a conservative lower bound. We generated 100 textual prompts using ChatGPT-5.2 and used each prompt to condition DreamO together with a randomly selected identity from LFW. In total, we sampled 500 identities and produced 500 images. We evaluated performance using the image-text matching BLIP score, facial consistency rate at FAR = 1e-3, and image quality measured by HPSv2 (Wu et al., 2023). We did not use the latter version HPSv3 (Ma et al., 2025) because Guo et al. (2025) found that HPSv3 is heavily biased toward a narrow

![Figure 6: Similarity Distribution. This is a line graph showing the frequency of similarity scores for four different methods. The x-axis is 'Similarity' ranging from 0.0 to 1.0. The y-axis is 'Frequency' ranging from 0.0 to 0.4. The four curves are: 'Another Photo' (blue), 'MinusFace Template' (orange), 'PartialFace Template' (green), and 'FracFace Template' (red). The 'Another Photo' curve peaks at a similarity of approximately 0.6. The 'MinusFace Template' curve peaks at approximately 0.75. The 'FracFace Template' curve peaks at approximately 0.85. The 'PartialFace Template' curve peaks at approximately 0.9.](303b94716b6713757d1fdf940a6b345f_img.jpg)

Figure 6: Similarity Distribution. This is a line graph showing the frequency of similarity scores for four different methods. The x-axis is 'Similarity' ranging from 0.0 to 1.0. The y-axis is 'Frequency' ranging from 0.0 to 0.4. The four curves are: 'Another Photo' (blue), 'MinusFace Template' (orange), 'PartialFace Template' (green), and 'FracFace Template' (red). The 'Another Photo' curve peaks at a similarity of approximately 0.6. The 'MinusFace Template' curve peaks at approximately 0.75. The 'FracFace Template' curve peaks at approximately 0.85. The 'PartialFace Template' curve peaks at approximately 0.9.

Figure 6. Similarity Distribution

mainstream standard of beauty, which may systematically disadvantage faces that deviate from this norm. This process is more expensive than Arc2Face generation due to the larger backbone (Flux). It took about 50 seconds to generate one image on an A6000 GPU, which is still very cheap for a high-quality deepfake generation. The results are shown in Table 9.

Even after 2 conversions, the pass rate is still near 90%. The HPSv2 score (0.28) is close to the original Flux baseline, and the average score of AI-generated images tested in Cai et al. (2025); Guo et al. (2025). It means the generated images have both high quality and prompt-following.
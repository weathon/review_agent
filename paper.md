# Facelinkgen: Rethinking Identity Leakage In Privacy-Preserving Face Recognition

Anonymous Author(s)
62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 31 32 33 34 35 37 38 39 40

## 1 Introduction And Related Works

41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58

## Abstract

Frequency-domain transformation-based privacy-preserving face recognition (PPFR) systems protect user identity by converting raw face images into protected templates in the frequency domain before recognition. Existing evaluations treat privacy as resistance to pixellevel reconstruction, measuring attack success with PSNR and SSIM.

We show that this evaluation paradigm may not adequately capture identity-level privacy: preventing pixel-level reconstruction does not necessarily prevent identity leakage. We present FaceLinkGen, an identity-centric attack that performs identity linkage and face regeneration directly from protected templates without recovering original pixels. Treating the conversion process as a black-box oracle, FaceLinkGen achieves over 98.5% matching accuracy and above 96% regeneration success on three recent frequency-domain PPFR systems. These results reveal a gap between pixel-distortion metrics and identity-level privacy in the evaluated methods, and motivate complementing pixel-level metrics with identity-centric evaluation in frequency-domain PPFR research.

Figure 1: Regeneration attack results. In each subplot, the left is the original image and the right is the regenerated image from the protected template. Each row shows examples from one PPFR method, in the order of PartialFace, MinusFace, and FracFace. image privacy and compression literature. While some recent approaches, such as FracFace [6], also report identity similarity and employ StyleGAN for non–pixel-level reconstruction, they still have a strong reconstruction-centric focus. A large body of prior work adopts this paradigm, using resistance to pixel-level recovery as evidence of privacy protection and optimizing attack objectives accordingly. Representative systems, including DuetFace [19], PartialFace [20], MinusFace [21], FaceObfuscator [14], and FracFace, explicitly rely on this reconstruction-based evaluation to argue robustness against recovery attacks.

This evaluation paradigm, however, rests on a critical implicit assumption: that preventing pixel-level reconstruction is both necessary and sufficient to prevent identity leakage. In this paper, we show that this assumption does not hold. Crucially, compromising privacy does not require recovering the original registered image, nor does pixel-level similarity reliably correspond to identity consistency. In the facial domain, two images that are visually or pixel-wise similar may represent different identities and images that represent the same identity may share no pixel or local similarity. CanFG [35] can generate two images with very high pixel-level similarity yet in completely different identities; conversely, in daily life, any two arbitrary photos of the same person (one ID photo and one social media photo) would have very high identity similarity yet likely very low pixel-level or structure-level similarity. An example is provided in Figure 2.

This misconception misled not only the evaluation but also the simulated attack design because of the overlooked fact that identity-revealing information can remain accessible even when pixel-level reconstruction is infeasible. By employing pixel-level loss functions, simulated attackers (red-team researchers) are inadvertently trapped into pursuing the specific registration image as ground truth. This objective is often mathematically impossible 1 privacy-preserving face recognition, identity leakage, face regeneration, linkage attack ACM Reference Format: Anonymous Author(s). 2026. FaceLinkGen: Rethinking Identity Leakage in Privacy-Preserving Face Recognition. In Proceedings of ACM Multimedia
(ACM MM '26). ACM, New York, NY, USA, 10 pages. https://doi.org/10.1145/
nnnnnnn.nnnnnnn The fundamental promise of transformation-based Privacy-Preserving Face Recognition (PPFR) systems is compelling: verify a user's identity without ever exposing their raw facial data [6, 14, 19–21, 37]. A user's face is converted into a protected template that a recognition server can use for matching, but that, ideally, cannot be reversed to recover the original image or used to infer private attributes. The primary adversary in this setting is the curious or malicious service provider who receives the template [8, 13].

The prevailing evaluation paradigm for these systems is currently significantly limited. Historically, the dominant objective has been to prevent the reconstruction of the original registration image, with attack success measured through pixel-level or local similarity metrics such as peak signal-to-noise ratio (PSNR) and structural similarity index measure (SSIM), a legacy inherited from ACM MM '26, 2026. ACM ISBN 978-x-xxxx-xxxx-x/YYYY/MM https://doi.org/10.1145/nnnnnnn.nnnnnnn

## Keywords

93

## Ccs Concepts

- Security and privacy → Biometrics; - Computing methodologies → *Computer vision*.

28 36 27 117 118 119 120 121 122 123 124 125 126 127 128 Figure 2: SSIM and PSNR
are not always correlated with identity correlation.

129 130 131 132 133 135 136 137 138 139 140 143 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 165 166 167 168 169 170 171 172 174 Table 1: Comparison between pixel-level metrics and identity-level metrics on two cases: a protected face generated by CanFG with its original face, and two images of the same person. We can see that higher pixel-level similarity does not mean higher identity-level similarity.

| Compared With   | SSIM   | PSNR   | MSE   | FS    |
|-----------------|--------|--------|-------|-------|
| Another Face    | 0.235  | 10.44  | 6699  | 0.586 |
| CanFG Face      | 0.841  | 26.81  | 143   | 0.008 |

due to the information loss in protection, causing the generator to produce a blurry image, likely an average of all images in the dataset. The failure is illustrated in Figure 3. Although FracFace [6] also reports identity similarity, the simulated U-Net attack remains reconstruction-based, aiming to recover the original pixels. The StyleGAN [16] attack also failed to generate a face with the same identity as the original image, as shown in Figure 3.

Recent work has begun to challenge the reconstruction-centric perspective in non-facial domains by shifting attention toward semantic-level inversion. In such settings, the attacker aims not to recover the original image itself, but to regenerate information that is semantically consistent with the original identity. This goal is both easier to achieve and more aligned with realistic attack objectives. For instance, Yue et al. [42] proposes a semantic recovery framework that leverages generative models to synthesize semantically consistent images without pixel-level similarity in federated learning attacks. Complementary findings further suggest that traditional reconstruction metrics fail to capture how humans perceive privacy leakage [30].

Our approach differs from standard Model Inversion Attacks
(MIAs). One category of MIA, represented by Wang et al. [33], recovers original images from embeddings. For many deep facial embedding models, such processes resemble image generation tasks rather than adversarial attacks, as they exploit the inherent invertibility of learned representations. Similar works exist in the ID-controlled image generation domain, like Arc2Face [25], PuLID [11], and FaceID IP-Adapter [39]. In contrast, our method targets structural vulnerabilities in the template generation process itself. Since this conversion often utilizes rule-based transformations independent of specific deep models, the attack surface differs from embedding-based reconstruction. Another category of MIA, which is closer to the original definition [9], such as the ones prevented by Wang et al. [36], aims to reconstruct the training dataset to compromise identity privacy in *the training set*. This focus on training data deviates from our objective of protecting individual user templates. Existing solutions for training-level privacy include FaceMAE [34] and the use of synthetic data [4].

181 182 183 184 185 186 187 188 189 190 191 192 193 194 196 197 198 199 200 201 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 175 176 177 178 179 180 Our Contribution. This paper argues that the prevailing evaluation paradigm for frequency-domain PPFR, which measures privacy as resistance to pixel-level reconstruction, may not adequately capture identity-level leakage. We introduce and validate an identitycentric evaluation approach, and present FaceLinkGen, an attack that performs identity linkage and face regeneration directly from protected templates without recovering original pixels. We show that pixel-level metrics can overestimate the privacy provided by the evaluated frequency-domain PPFR systems: identity information can be extracted via a straightforward distillation pipeline even when the original image remains "unreconstructable" under PSNR and SSIM. We additionally provide pilot evaluations on an adversarial de-identification method (TIP-IM) and a non-frequencydomain method (CanFG), with preliminary results suggesting that similar vulnerabilities may extend beyond the frequency-domain PPFR family. The simplicity of the pipeline is intentional: if a standard procedure suffices, the vulnerability is in the representation. This work encourages identity-centric evaluation alongside existing metrics in future PPFR and de-identification research.

## 2 Threat Model

The PPFR paradigm was originally motivated by a specific and well-defined concern: a user must submit face data to a remote recognition server they do not fully trust [8]. Since the raw face image is deeply personal, the user has reason to want the server to perform recognition without gaining access to the original biometric. PPFR addresses this by transforming the face into a protected representation before transmission, one that the server can use for matching but, ideally, cannot revert to recover the original face, infer soft biometrics such as age, gender, or ethnicity, or link the identity to records outside the service. The primary adversary in this setting is therefore the *curious or malicious service provider*, an insider who legitimately receives the protected template and seeks to exploit it.

Protection against network-level eavesdroppers is a separate concern and was not the original intent of PPFR. Interception of data in transit is well addressed by general-purpose secure communication protocols such as TLS, which are application-agnostic and do not require altering the face representation itself. Nevertheless, some recent PPFR work has adopted an external-attacker framing: Mi et al. [21] describes this attacker as "typically envisioned as a malicious third-party wiretapping the transmission," departing from the original insider-centric design intent. One might argue that the conversion process and the recognition service could be controlled by separate entities, with the conversion model running locally or under a trusted third party server, thereby preventing the service 2 173 202 203 145 144 142 141 134 195 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 265 266 267 268 269 270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 provider from accessing the conversion process. However, a locally running model can be reverse-engineered by the service provider to obtain oracle access, and the provider can submit arbitrary faces through the local client to collect paired (original, protected) data on their server. If instead the conversion process runs on the thirdparty remote server, the trust problem is merely shifted rather than solved: the user must now trust the third-party server with their raw face data, which contradicts the original motivation for PPFR.

We return to first principles and evaluate PPFR against its intended adversary: a service provider with oracle access to the conversion process (the ability to query it with arbitrary inputs and observe outputs), but no knowledge of its internal architecture or parameters. Even under this insider-focused framing, our attacker model assumes less than prior work: Mi et al. [21] assumes knowledge of the conversion architecture (but not the random channel selection parameters), and Mi et al. [20] assumes access to the conversion process but not the selected channel IDs. We assume no knowledge of the architecture, parameters, or hyperparameters. CanFG [35] places all of its security in the secrecy of the conversion model, which is an unrealistic assumption when the service provider itself is considered the adversary. They themselves pointed out that if the paired face data can be obtained, a reverse CanFG model could be trained to reconstruct the original faces.

For completeness, we also note that Zhang et al. [43] requires approximately 6,900 online verification queries per identity and depends on the server returning continuous similarity scores for optimization, a setup highly susceptible to rate limiting and fraud detection. Many deployed FR systems return only binary accept/reject decisions or quantized similarity scores, rendering such approaches infeasible [22, 31]. Our method does not rely on server-side behavior at all.

## 3 Methods

The simplicity of our method is intentional. We show that even strong protection methods fail with a simple, standard distillation process, proving that the vulnerability resides in the representation itself. To formulate this, we consider a face image  as a combination of identity information  and non-identity (nuisance) information  , such that  ∼ (· | 
,  ).

In transformation-based PPFR systems, a protected template  is generated to hide the visual data of while retaining identity utility.

This process can be viewed as a lossy mapping that suppresses the information quantity of  while preserving 
:
 ∼ (· | ). (1)
Existing evaluations often equate privacy with the failure of pixellevel reconstruction. However, since  is largely discarded, recovering the original pixels  is a severely ill-posed problem. Conventional attacks fail because they attempt to optimize for specific nuisance factors (e.g., exact lighting or pose) that no longer exist in
 , resulting in blurry or identity-inconsistent outputs.

Our approach, FaceLinkGen, instead focuses on extracting the remaining identity information. We use a distillation-style procedure to align the template domain with a standard identity embedding space. Given a public dataset, we train a student model  to recover an identity representation 
′
from  . The training objective is to maximize the cosine similarity between the student's output and 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323 324 325 326 327 328 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 the embeddings  extracted by a frozen teacher model  from original images :

$$\mathcal{L}=1-\frac{1}{N}\sum_{k=1}^{N}s(f_{s}(t_{k}),f_{t}(i_{k})),\tag{2}$$

where  and  are the protected template and original image derived from the same source image of identity , and 
′

=  ( ) is
the identity feature recovered by the attacker.
Once 
′

is extracted, the attack bypasses the need for pixel reconstruction by leveraging a diffusion-based generative model diff .
Rather than trying to find the original  , we substitute the missing information by sampling from the model's stochastic noise :
 ∼ N (0, ),  = diff (
$$Y=g_{\mathrm{diff}}(z_{I}^{\prime},\epsilon).$$
, ). (3)
Rather than recovering the discarded nuisance factor  , we sample a new 
′via , bypassing the ill-posed reconstruction problem
( 0 
′
). As long as  persists in  , an attacker can recover 
′

and combine it with random  to regenerate an identity-consistent face. This implies that visual distortion of  does not prevent identity extraction, as recovering  is not a prerequisite for a successful attack.

Note that our attack method is independent of the specific face recognition model employed by the PPFR server. We use ArcFace as the teacher and student network solely because it is a widelyadopted, publicly available facial embedding model, and its compatibility with the Arc2Face generative model simplifies our demonstration. Other embedding models like FaceNet [28] can also be used as long as there is a compatible generative model. The server's actual recognition backbone could be any commercial or proprietary model. Our attack only requires that the protected template retains identity-discriminative features that are learnably aligned with some facial embedding space accessible to the attacker, a condition implicitly assumed by any PPFR system that aims to preserve recognition utility.

## 4 Attack Vectors

The identity extraction framework described above enables two distinct attack vectors. The linkage attack requires only the extracted embedding 
′ 

and operates entirely in the embedding space, while the regeneration attack additionally employs the diffusion-based generator diff to synthesize identity-consistent face images.

## 4.1 Linkage Attack

A linkage attack aims to associate a real-world identity (e.g., a public face image) with a protected identity, or to link two protected templates belonging to the same individual across different leaked databases. The first case is referred to as face-to-template linkage, while the second is template-to-template linkage. ISO/IEC 24745 explicitly requires resistance against template-to-template search, but does not address face-to-template search. This is likely a utility trade-off for the verification needs of the service provider. This attack vector is similar to an attack vector for hashing: when the input space is known, an attacker can enumerate all possible inputs and map each hashed output back to its original input.

In both attack scenarios, the attacker first obtains a query embedding . This embedding can be extracted using either the student 3 329 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377 378 379 380 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 405 404 402 403 406

model  or the teacher model , depending on the domain of the
query data. The adversary then computes embeddings for all protected templates in the leaked database using  and performs a
nearest-neighbor search. This process can be written as
$$\arg\operatorname*{max}_{t\in T}\;s(e_{q},f_{s}(t)),$$
(,  ()), (4)
where (·, ·) denotes cosine similarity.

## 4.2 Regeneration Attack

As discussed earlier, reconstructing the original enrollment image is unnecessary and, in many cases, impossible. However, once a universal face embedding (e.g., ArcFace) can be extracted from a protected template, modern face generation models can be leveraged. In this work, we use Arc2Face, which takes a facial embedding as input and generates a face image whose embedding matches the input. This allows us to synthesize a realistic face corresponding to the protected template without reconstructing the original image.

## 5 Experiments And Results

We selected three frequency-domain PPFR methods with accessible source code: PartialFace [20] (ICCV 2023), MinusFace [21]
(CVPR 2024), and FracFace [6] (NeurIPS 2025). These represent the mainstream of currently open-sourced transformation-based PPFR
work. For distillation, we used a subset of CASIA-WebFace [40] with around 10K identities and 90K images. Importantly, there is no dataset or architecture overlap between the attacked methods and our student model: PartialFace and FracFace are training-free hardcoded transformations, and MinusFace is trained on MS1Mv2 [10],
while our attack model is trained on CASIA-WebFace. The facial embedding model is Antelopev2 with one additional 3x3 Conv2D layer prepended to ensure compatibility with different template formats (channel numbers) if needed. Antelopev2 was chosen because it is the backbone that Arc2Face accepts. The dataset is split into a training and a validation set in an 80-20 ratio. For regeneration testing, we used three datasets: the validation hold-out set of CASIA-WebFace, "this person does not exist" (TPDNE) dataset 1, and Labelled Faces in the Wild (LFW) dataset. The hold-out set is used to test the ability of our method in real images while ensuring no ID duplications. The LFW dataset is used to test the cross-dataset performance of our method with distribution shift, and the TPDNE is used as a synthetic dataset to avoid data cross-contamination from Stable Diffusion 1.5 and Arc2Face training data. Compared to the hold-out set and LFW, photos in the TPDNE dataset are also closer to a headshot, which is what is usually used to create the protected templates.

The distillation process was completed in under two hours on a single NVIDIA A6000 GPU for each of the three evaluated methods, at an estimated cost of approximately USD 0.80 to 1.60. The same process can also be executed on consumer GPUs such as the RTX 4090 or RTX 5090 with comparable wall-clock time; in fact, it is theoretically faster on the RTX 5090 due to its newer architecture. These extreme low costs are deliberate: they serve as a lower-bound analysis demonstrating that current protection mechanisms succumb to a lightweight, generic distillation without requiring complex adversarial optimization.

1TLeonidas/this-person-does-not-exist Table 2: Regeneration attack results on three PPFR methods across three datasets. Success@5: at least one of five generated images passes Face++ verification. Pass@**: per-image**
pass rate at the Face++ threshold corresponding to false acceptance rate .

| Success@5                                               | Pass@1e-5   | Pass@1e-4   | Pass@1e-3   |       |
|---------------------------------------------------------|-------------|-------------|-------------|-------|
| Dataset: TPDNE PartialFace 1.000                        | 0.993       | 0.996       | 0.998       |       |
| MinusFace                                               | 0.996       | 0.936       | 0.970       | 0.989 |
| FracFace                                                | 0.992       | 0.904       | 0.957       | 0.985 |
| Dataset: CASIA-WebFace Hold-Out PartialFace 0.992 0.957 | 0.970       | 0.982       |             |       |
| MinusFace                                               | 0.989       | 0.930       | 0.958       | 0.978 |
| FracFace                                                | 0.991       | 0.920       | 0.950       | 0.977 |
| Dataset: LFW PartialFace                                | 0.988       | 0.980       | 0.983       | 0.986 |
| MinusFace                                               | 0.987       | 0.974       | 0.981       | 0.983 |
| FracFace                                                | 0.979       | 0.943       | 0.961       | 0.970 |

## 5.1 Minimal-Resource Attack

To stress-test the data requirements, we conducted an experiment on FracFace with only approximately 800 images. With higher weight decay and lower batch size to mitigate overfitting, training completed in under 50 seconds, yet the attack still achieved a 97.0%
generation pass rate at FAR 1 × 10−5and 99.5% linkage accuracy.

We further reduced the training set to 256 images, obtaining 98.7% linkage accuracy and 90.5% regeneration success. The consistently high performance across these resource-constrained settings suggests that identity information is not meaningfully disrupted by the transformation, but rather preserved in a form that remains easily extractable.

## 5.2 Linkage Attack

Since all templates are converted to the standard ArcFace domain, we can not only link between original images and protected templates, but also link between two templates from the same or different protection methods. In any case, we are linking two different images (or templates) of the same person, not a face image and its corresponding template. We used the CASIA-WebFace hold-out set for the linkage attack to ensure no identity overlap between training and testing identities. The hold-out dataset size is 2115.

The closed-set 1-to-N linkage results are in Table 4. The originalimage-to-original-image linkage (0.88) establishes a performance upper bound. With the WebFace dataset containing 9.3%-13.0% noise [32], perfect linkage is impossible regardless of method. Our attack achieves linkage success rates consistently above 70%, frequently exceeding 80%, essentially reaching the dataset's theoretical maximum performance. This confirms that the extracted embeddings function as effective identity descriptors for cross-domain matching. Additionally, the 1-to-1 verification accuracy used in the traditional face recognition benchmark (Table 3) remains near 100%
4 407 408 409 410 411 413 412 414 415 416 417 418 419 421 420 422 423 424 425 426 427 429 430 431 432 433 434 435 436 437 438 439 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 459 460 461 463 464 462 428 458

Training Linkage Re-generation Emb eddin g Initialize Cosine Similarity Loss Teacher Model
(Frozen)
Student Model Teacher Model Query Protected Template Original Face Query Face Student Model Embedding Bank Student Model
(Training)
Arc2Face Template Database Original Face Re-Generated Face Protected Template
523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539 540 541 542 543 544 545 546 547 548 549 550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 465 466 467 468 469 470 471 473 474 475 476 477 478 Table 3: 1-to-1 verification accuracy between template-toface and face-to-face on LFW.

480 481 482 483 484 485 486 487 489 490 491 492 493 494 495 497 499 500 503 504 505 506 507 508 509 510 511 512 513 514 515 516 518 519 520 521

| PartialFace   | MinusFace   | FracFace   |
|---------------|-------------|------------|
| 0.99          | 0.98        | 0.92       |

522 and comparable to the original ArcFace performance, demonstrating that the protection systems fail to meaningfully impede identity matching.

## 5.3 Regeneration Attack

For the regeneration attack, we evaluate identity recovery on the first 1,000 images from each of the following datasets: the TPDNE dataset, the hold-out set of CASIA-WebFace, and the LFW dataset. Each image is converted into a protected template and mapped to 517 a facial embedding using the student model. For each embedding, five images are generated using Arc2Face to account for stochasticity. We report both per-image success rate and Success@5. Face generation is highly efficient due to the small backbone of SD1.5: generating a batch of five images for a single embedding takes approximately three seconds on an NVIDIA A6000 GPU, corresponding to a throughput of roughly 1,200 identities per hour and an estimated cost of $0.0005 per identity generation. Visual examples are shown in Figure 1.

Following the evaluation protocol of CanFG, we employ a commercial face verification system, Face++, to assess identity consistency between the original dataset image and each generated face. We used Face++, marketed to have "financial-grade security standards," which is usually a higher standard (e.g., more challenging for us) than many open-source methods. This also avoids using the same model (e.g., ArcFace) or models trained on the same datasets (most open source ones) for both embedding extraction and verification. Face++ outputs a confidence score together with three operating thresholds corresponding to error rate of 1 × 10−3, 1 × 10−4, and 1 × 10−5. For each generated image, we record the strictest threshold at which the identity match is accepted and use this as the evaluation outcome. If no face is detected in the original image, it is excluded from the data, while if no face is detected in the generated image, it counts as a failure.

The results are summarized in Table 2. On all three datasets, the success rate at the first attempt is all higher than 97%, and the success rate for five attempts ranges from 97.9% to 100%. Even at the strictest threshold, the success rate is still above 90%. We directly compare our regeneration attack with the original reconstruction attack protocol used by FracFace's authors. In Table 6, we reported the protection rates claimed in FracFace under its own evaluation and the corresponding rates under our attack on the TPDNE dataset.

The evaluation of attack success in FracFace [6] is based on a Protection (%) metric, defined as the proportion of frequency-domain channels that are filtered or structurally disrupted. This formulation establishes a lower barrier for defensive claims than our identitycentric standard, which requires successful regeneration of images passing commercial-grade verification. By our metric, the protection rate of most recent PPFR methods in U-Net/StyleGAN attack 5 Table 5: Regeneration pass rate cross-verified with the Amazon face comparison API on 700 LFW images. Table 4: Linkage Results between MinusFace, PartialFace, FracFace, and Original Image Embeddings on CASIA-
WebFace dataset. The query and key for each pair are generated from different images of the same identity. The numbers reported are top-1 recall at a closed set setting.

Accuracy AUROC

501

| Query       | Key       |             |          |        |
|-------------|-----------|-------------|----------|--------|
| FracFace    | MinusFace | PartialFace | Original |        |
| FracFace    | 0.7863    | 0.7537      | 0.8137   | 0.8478 |
| MinusFace   | 0.7305    | 0.7206      | 0.7754   | 0.8132 |
| PartialFace | 0.8028    | 0.7868      | 0.8270   | 0.8572 |
| Original    | 0.8444    | 0.8241      | 0.8563   | 0.8823 |

502 498 472 479

| MinusFace-to-Face   | 0.992   | 0.995   |
|---------------------|---------|---------|
| FracFace-to-Face    | 0.988   | 0.993   |
| PartialFace-to-Face | 0.992   | 0.996   |
| Face-to-Face        | 0.998   | 0.998   |

488 496 is almost always 100%. Despite our stricter criterion being unfavorable to reported attack success, we show that high channel protection does not prevent identity leakage: even when FracFace claims high protection under its frequency-domain metric, FaceLinkGen achieves near-total identity recovery. This also shows that channel disruption does not mean identity protection.

To cross-verify this result, we used another commercial facial comparison API from Amazon through EdenAi on 700 selected images on the LFW dataset. The Amazon API only provides a single pass/fail decision with a confidence score; the results are shown in Table 5. The values are close to the Face++ results, validating our claims.

To rule out the dependence on models like Arc2Face or thirdparty verification services like Face++ or Amazon, we compared the similarity of the extracted embeddings with the original face.

As detailed in Section 7, the embedding extracted from a protected template shows higher cosine similarity to its source image than to another image of the same person.

599 602 603 604 605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 581 582 583 584 585 586 587 588 589 590 591 592 593 594 595 596 597 598

## 6 Attack Under Constrained Assumptions

Our main results demonstrate that identity information can be reliably extracted by the primary adversary in the PPFR threat model, namely the service provider or developer who has full access to the conversion process [8]. This also follows the threat models in prior PPFR evaluations [6, 20, 21]. We additionally ask how far this vulnerability extends to more constrained adversaries: an external attacker or a knowledge-constrained insider who has neither access to the conversion process nor any knowledge of its internals.

To investigate this, we consider a constrained scenario. In this setting, the attacker possesses only 30 paired image–template samples for validation (not for training) and has limited knowledge of the underlying protection mechanisms. This is actually stricter than the "black-box" scenario in Mi et al. [20], which assumes the attacker knows the conversion process but not the channel parameters, and more realistic than Zhang et al. [43], which relies on thousands of server queries per identity. In real attacks, these 30 pairs can be obtained by a small number of leaked samples, known identities, or attacker-controlled accounts. It can also be simulated with low-frequency queries to the authentication server.

We observe that despite their claimed algorithmic complexity, the output templates of these systems share a common visual essence: they all preserve high-frequency information while obfuscating low-frequency information. Based on this intuition, the attacker can bypass any system-specific modeling and instead use a generic Gaussian-blur-based high-pass filter as a universal proxy task. We avoided using DCT or DWT to decouple from the methods used in the tested systems. By subtracting a slightly blurred version from the original image and applying simple data augmentations (e.g., varying kernels and strengths), the attacker trains a student model to align this simple high-pass domain with the identity embedding space. The attacker does not need to know any details about the system; instead, the high-pass characteristic is easily observable visually.

The training process is the same as our main text, with simply the known PPFR conversion process replaced with a high-pass 637 636 638 661 662 663 664 665 666 667 668 669 670 671 672 673 674 675 676 677 678 679 680 681 682 683 684 685 686 687 688 689 690 691 692 To quantify identity leakage independent of Arc2Face or commercial APIs, we analyze cosine similarity in ArcFace embedding space between (1) two images of the same person and (2) an image and 6 693 694 695 696 639 640 641 642 643 644 645 646 647 648 649 650 651 652 653 654 655 656 657 659 660 filter. During inference, the templates are directly fed into the student model except for MinusFace, for which it is passed through a Gaussian-blur-based high-pass filter to remove low-frequency noise.

We trained only one model to attack all three methods. As shown in Table 7, identity leakage remains strikingly persistent. For all three systems, we achieved over 92% 1-to-1 matching success rate and over 94% re-generation success@5 on Face++ and about 4457% on the Amazon API. The Amazon API is likely more strict or sensitive to AI-generated images. Notably, the 1-to-1 linkage success rate on LFW remains around 92-96% (Table 7), close to our main experiments.

These results show that, for the frequency-domain PPFR methods evaluated, identity extraction remains feasible even under constrained attacker assumptions. The three systems share a common characteristic: their output representations retain strong coupling with simple high-pass filtering operations, which is what makes this proxy-based attack possible. We note that this particular attack is specific to the frequency-domain family and does not apply to non-frequency methods such as CanFG. However, CanFG remains vulnerable under the oracle-access threat model (Section 5), indicating that identity leakage is not unique to frequency-domain design. Across both settings, the results suggest that as long as templates preserve recognition utility, they tend to retain recoverable identity information, and current frequency-domain defenses do not adequately address this.

658

## 7 Similarity Distribution

| Venue            | Protection Tested in   | Protection Tested           | Protection Tested   |       |
|------------------|------------------------|-----------------------------|---------------------|-------|
| FracFace [6]     | Using Our Attack       | Using Our Attack (5 trials) |                     |       |
| PartialFace [20] | ICCV 2023              | 0.680                       | 0.002               | 0.000 |
| MinusFace [21]   | CVPR 2024              | 0.850                       | 0.011               | 0.004 |
| FracFace [6]     | NeurIPS 2025           | 1.000                       | 0.015               | 0.008 |

| Method      | Face++   | Amazon API   | Matching   |
|-------------|----------|--------------|------------|
| FracFace    | 0.946    | 0.473        | 0.949      |
| PartialFace | 0.946    | 0.447        | 0.925      |
| MinusFace   | 0.963    | 0.570        | 0.962      |

| TIP-IM                         | CanFG      |            |            |       |
|--------------------------------|------------|------------|------------|-------|
| 𝑓 (𝑥𝑝 )                     | 𝑑 (𝑥𝑝 ) | 𝑓 (𝑥𝑝 ) | 𝑑 (𝑥𝑝 ) |       |
| Cosine Similarity to 𝑓 (𝑥)   | −0.081     | 0.648      | 0.042      | 0.520 |
| Linkage Accuracy (same image)  | 0.000      | 0.997      | 0.10       | 0.997 |
| Linkage Accuracy (cross image) | 0.000      | 0.939      | 0.04       | 0.825 |

Table 6: Comparison with FracFace defensive claims. While FracFace [6] measures protection by frequency channel disruption, we evaluate actual identity leakage. High disruption rates fail to prevent extraction, as FaceLinkGen achieves near-total recovery through commercial-grade verification.

697 698 699 700 701 703 761 704 762 705 763 706 764 707 765 766 767 768 769 770 771 772 714 715 773 774 716 708 709 710 711 712 713 Table 8: Identity leakage evaluation on TIP-IM deidentification. We report cosine similarity and linkage accuracy using the standard FR model  **and our student** model  **on protected images**  .

Table 7: Assumption-constrained attack results. Regeneration Success@5 is reported for Face++ and Amazon API; matching accuracy is the maximum 1-to-1 linkage accuracy at the optimal threshold.

721 722 723 724 725 726 727 728 729 730 731 732 733 734 735 736 737 738 739 740 741 742 743 744 745 746 747 748 749 750 751 752 753 its protected template (Figure 5). Some near-zero similarities are due to dataset noise [7, 32]; we focus on the main cluster. Across all three methods, the similarity between an image and its template exceeds that between two different images of the same person.

781 782 783 784 785 an FR system. We evaluate on a small subset of CASIA-WebFace
(2,082 images from 408 identities).

We report sim( (),  ( )), where  is the original image and is the protected image, reflecting how well the de-identification method deceives the original FR model, and sim( (),  ( )), measuring whether our distillation pipeline can recover the identity information that the protection attempts to suppress. The results in Table 8 show that although TIP-IM provides perfect protection against ArcFace, this is almost completely undermined by our distillation pipeline. This confirms that adversarial perturbation suppresses identity in one model's embedding space while leaving it recoverable by a student aligned to a different space, suggesting a structural limitation of adversarial de-identification as a privacy mechanism.

We additionally evaluate on CanFG [35], which can be viewed as both a non-frequency-domain PPFR method and a non-adversarial de-identification method. As discussed in our threat model, CanFG conflates PPFR with facial de-identification and places its security entirely in the secrecy of the conversion model: an assumption that does not hold when the service provider is the adversary. Its deterministic transformation also leaves it vulnerable to linkage attacks. Under our pipeline, the results are also shown in Table 8, confirming that the same distillation framework used for frequencydomain PPFR and adversarial de-identification generalizes to nonadversarial, non-frequency-based settings as well.

786 787 788

## 8 Transfer To Attacking De-Identification Systems

Our primary focus is on PPFR systems, which retain machinereadable identity information while rendering the original face unreconstructible from the template by removing some visual information. However, a complementary line of work pursues the opposite objective: preserving visual recognizability to humans while disrupting machine-based face recognition [27, 29, 38]. These systems apply adversarial noise or synthetic makeup so that the protected face remains human-recognizable but causes FR systems to fail.

This design introduces a vulnerability symmetric to the one we identify in PPFR. PPFR systems leak identity by preserving machine-readable mutual information; these perception-consistent de-identification systems leak identity by preserving human-perceivable information. In both cases, sufficient identity signal is retained for extraction. Notably, the same learning-based pipeline suffices to recover this information in both settings. As long as this mutual information can be aligned to a publicly accessible embedding space, a student model can recover it.

We selected TIP-IM [38] as a representative dodging method. We note that many methods [12, 27] conflate dodging with impersonation, which are not equivalent: an FR system can simultaneously associate a face with both the original identity and the impersonation target. In privacy-purposed de-identification scenarios, users care about dodging, not impersonation, which is closer to attacking 789 790 791 792 793 794 795 796 797 798 799 800 801

## 9 Soft Identity Leakage: Beyond Unique Identifiers

Beyond hard identity recognition, the exposure of soft biometric attributes presents a significant privacy risk. Characteristics such 7 810 811 754 755 756 757 758 759 702 760 775 717 718 776 719 777 720 778 779 780 802 803 804 805 806 807 808 809 812 813 814 815 816 817 818 819 820 821 822 823 824 825 826 827 828 829 830 831 832 833 834 835 836 837 838 839 840 841 842 843 844 845 846 847 848 849 850 851 as skin color, age, and gender are sensitive personal data that enable unauthorized profiling and algorithmic discrimination. Privacy frameworks like the Canadian Privacy Act [5] explicitly protect race and age. A robust privacy-preserving system must therefore prevent the recovery of these attributes from its templates.

In the public rebuttal of FracFace [6] on OpenReview [1], the authors claimed successful obfuscation of age, gender, and ethnicity. They cited a human perception study where participants reported less than 13% usage of these biometrics for identity inference, with over 76% of participants relying on guesswork. However, our empirical results in Figure 1 demonstrate that these soft biometrics remain visible in the re-generated faces. This matches with previous research, which indicates that ArcFace embeddings retain such information [18, 23]. Because our extracted embeddings closely resemble the original facial embeddings, we hypothesize that a model can learn a direct mapping from the template to these attributes without facial reconstruction.

To assess this, we appended a multi-head prediction MLP to the distilled student model and trained it for one epoch on the FairFace dataset [17] to infer soft biometrics, predicting age, gender, and race (7 categories). Evaluation was performed on 10k images. As reported in Table 9, for all these protected templates, soft-biometrics can be inferred at nearly the same accuracy as from the original images for FracFace and PartialFace. For MinusFace, the reconstruction is less accurate, though a substantial amount of information is still recovered.

For completeness, it is worth noting that some methods, such as CanFG [35] and FaceAnonyMixer [2], intentionally retain soft biometric traits to facilitate auxiliary tasks. We contend that this design decision should be scrutinized more carefully. In PPFR systems, preserving soft biometric details furnishes adversaries with highly sensitive information while offering no advantage for recognition accuracy. These attributes are a focal point of privacy regulations and may necessitate stricter protections [24]. We did not assess softbiometric leakage on faces protected by CanFG, as such leakage is immediately apparent: as illustrated in Figure 2, CanFG-generated faces, by design, retain nearly all soft biometric characteristics by design.

| Method          | Race Acc ↑   | Gender Acc ↑   | Age MAE ↓   |
|-----------------|--------------|----------------|-------------|
| FracFace        | 0.673        | 0.925          | 4.737       |
| MinusFace       | 0.569        | 0.893          | 5.883       |
| PartialFace     | 0.663        | 0.932          | 4.952       |
| Random Baseline | 0.146        | 0.493          | 21.51       |
| Original Image  | 0.700        | 0.949          | 4.584       |

852 853 854 856 858 859 860 861 862 863 864 866 867 869 870

## 10 Future Directions

We suggest several potential pathways for future PPFR designs and evaluations, primarily focusing on stronger defensive mechanisms and broader vulnerability assessments.

871 872 873 874 875 876 877 878 879 880 881 882 883 884 885 886 887 888 889 890 891 892 893 894 895 896 897 898 899 900 901 902 903 904 905 906 907 908 909 910 911 912 913 914 915 917 918 919 921 922 923 924 925 926 927 928

## 11 Conclusion

This paper shows that pixel-level reconstruction metrics may not adequately capture identity-level leakage in the frequency-domain PPFR systems we evaluated. Identity information can be extracted from protected templates via a straightforward distillation pipeline and used for linkage and face regeneration without recovering original pixels. FaceLinkGen achieves over 98.5% matching accuracy and above 96% regeneration success on three frequency-domain PPFR systems, suggesting that these methods do not provide the level of identity privacy their pixel-level evaluation results imply.

We hope these findings encourage the adoption of identitycentric evaluation alongside existing metrics in frequency-domain PPFR research, and motivate exploration of stronger protection mechanisms, such as cryptographic approaches, that offer more formal privacy guarantees.

Cryptographic and Key-Based Hardening. One rigorous approach is to incorporate secret keys into the conversion process, similar to Yuan et al. [41]. This serves as a multi-factor authentication system (requiring both biometrics and a key), preventing attackers (including our method) from converting a face without the secret. Alternatively, systems may revert to formal cryptographic methods like [3]. While traditionally viewed as computationally expensive, modern resources make this trade-off acceptable; for instance, Jindal et al. [15] reports only 2.83ms processing time per face pair. Importantly, these computational costs can act as an effective client-side constraint against brute-force attacks (conceptually similar to slow hashing), enhancing privacy while remaining imperceptible to regular users. Again, we want to emphasize that de-ID or reversible face encryption methods *cannot* be used for PPFR tasks, as they either make the face completely useless for recognition or the reversibility compromises the privacy-preserving nature.

De-identification: Fooling Human Perception Instead of Machines.

Current de-identification methods aim to fool FR systems while keeping the identity visually intact to human observers. However, this is almost guaranteed to be attacked due to the information still being intact. However, research suggests that human and machine perception of facial similarity diverges [26]. We propose an alternative direction: rather than hiding identity from machines while preserving it for humans, one could actively remove identity information from the image while optimizing a human perceptual loss to make the result appear identity-consistent to human observers. This inverts the typical de-identification objective and may offer stronger privacy guarantees if real identity information can be completely removed. One possible approach is to reconstruct the face from identity-agnostic geometric cues such as depth maps or Canny edge maps, which preserve appearance structure without encoding biometric identity. Alternatively, we can adopt an inverted approach similar to these frequency-based methods, where we retain only the low-frequency components that are important for human perception and discard the high-frequency components that carry identity information. This mapping could also be N-to-1 such that a human can still *recognize* the face, but neither a human nor a machine can re-identify the face due to lost information.

920 8 868 865 857 855 916 930 929 931 932 933 934 935 936 937 938 939 940 941 942 943 944 945 946 947 948 949 950 951 952 953 954 955 956 957 958 959 960 961 962 963 964 965 966 967 968 969 970 971 972 973 974 975 976 977 978 979 980 981 982 983 984

## References

[1] [n. d.]. Wayback Machine. https://web.archive.org/web/20260000000000*/https:
//openreview.net/forum?id=JSSvYZKvL8
[2] Mohammed Talha Alam, Fahad Shamshad, Fakhri Karray, and Karthik Nandakumar. 2025. FaceAnonyMixer: Cancelable Faces via Identity Consistent Latent Space Mixing. doi:10.48550/arXiv.2508.05636 arXiv:2508.05636 [cs].

[3] Wei Ao and Vishnu Naresh Boddeti. 2025. CryptoFace: End-to-
End Encrypted Face Recognition. In *Proceedings of the Computer* Vision and Pattern Recognition Conference. 19197–19206. https: //openaccess.thecvf.com/content/CVPR2025/html/Ao_CryptoFace_Endto-End_Encrypted_Face_Recognition_CVPR_2025_paper.html
[4] Gwangbin Bae, Martin de La Gorce, Tadas Baltrusaitis, Charlie Hewitt, Dong Chen, Julien Valentin, Roberto Cipolla, and Jingjing Shen. 2022. DigiFace-1M: 1 Million Digital Face Images for Face Recognition. doi:10.48550/arXiv.2210.02579 arXiv:2210.02579 [cs].

[5] Branch Legislative Services. 2025. Consolidated federal laws of Canada, Privacy Act. https://laws-lois.justice.gc.ca/eng/ACTS/P-21/page-1.html\#h-397182
[6] Wanying Dai, Beibei Li, Naipeng Dong, Guangdong Bai, and Jin Song Dong.

2025. FracFace: Breaking The Visual Clues—Fractal-Based Privacy-Preserving Face Recognition. https://openreview.net/forum?id=JSSvYZKvL8
[7] Jiankang Deng, Jia Guo, Jing Yang, Niannan Xue, Irene Kotsia, and Stefanos Zafeiriou. 2022. ArcFace: Additive Angular Margin Loss for Deep Face Recognition. *IEEE Transactions on Pattern Analysis and Machine Intelligence* 44, 10 (Oct. 2022), 5962–5979. doi:10.1109/TPAMI.2021.3087709 arXiv:1801.07698 [cs].

[8] Zekeriya Erkin, Martin Franz, Jorge Guajardo, Stefan Katzenbeisser, Inald Lagendijk, and Tomas Toft. 2009. Privacy-Preserving Face Recognition. In *Privacy* Enhancing Technologies, Ian Goldberg and Mikhail J. Atallah (Eds.). Springer, Berlin, Heidelberg, 235–253. doi:10.1007/978-3-642-03168-7_14
[9] Matt Fredrikson, Somesh Jha, and Thomas Ristenpart. 2015. Model Inversion Attacks that Exploit Confidence Information and Basic Countermeasures. In Proceedings of the 22nd ACM SIGSAC Conference on Computer and Communications Security. ACM, Denver Colorado USA, 1322–1333. doi:10.1145/2810103.2813677
[10] Yandong Guo, Lei Zhang, Yuxiao Hu, Xiaodong He, and Jianfeng Gao. 2016. MS-
Celeb-1M: A Dataset and Benchmark for Large-Scale Face Recognition. https: //arxiv.org/abs/1607.08221v1
[11] Zinan Guo, Yanze Wu, Zhuowei Chen, Lang Chen, Peng Zhang, and Qian He.

2024. PuLID: Pure and Lightning ID Customization via Contrastive Alignment. doi:10.48550/arXiv.2404.16022 arXiv:2404.16022 [cs].

[12] Shengshan Hu, Xiaogeng Liu, Yechao Zhang, Minghui Li, Leo Yu Zhang, Hai Jin, and Libing Wu. 2022. Protecting Facial Privacy: Generating Adversarial Identity Masks via Style-robust Makeup Transfer. doi:10.48550/arXiv.2203.03121 arXiv:2203.03121 [cs].

[13] Jiazhen Ji, Huan Wang, Yuge Huang, Jiaxiang Wu, Xingkun Xu, Shouhong Ding, ShengChuan Zhang, Liujuan Cao, and Rongrong Ji. 2022. Privacy-Preserving Face Recognition with Learnable Privacy Budgets in Frequency Domain. doi:10. 48550/arXiv.2207.07316 arXiv:2207.07316 [cs].

[14] Shuaifan Jin, He Wang, Zhibo Wang, Feng Xiao, Jiahui Hu, Yuan He, Wenwen Zhang, Zhongjie Ba, Weijie Fang, Shuhong Yuan, and Kui Ren. 2024. FaceObfuscator: Defending Deep Learning-based Privacy Attacks with Gradient Descentresistant Features in Face Recognition. In 33rd USENIX Security Symposium (USENIX Security 24). USENIX Association, Philadelphia, PA, 6849–6866. https: //www.usenix.org/conference/usenixsecurity24/presentation/jin-shuaifan
[15] Arun Kumar Jindal, Imtiyazuddin Shaik, Vasudha Vasudha, Srinivasa Rao Chalamala, Rajan Ma, and Sachin Lodha. 2020. Secure and Privacy Preserving Method for Biometric Template Protection using Fully Homomorphic Encryption. In 2020 IEEE 19th International Conference on Trust, Security and Privacy in Computing and Communications (TrustCom). 1127–1134. doi:10.1109/TrustCom50675.2020.00149 ISSN: 2324-9013.

[16] Tero Karras, Samuli Laine, and Timo Aila. 2019. A Style-Based Generator Architecture for Generative Adversarial Networks. doi:10.48550/arXiv.1812.04948 arXiv:1812.04948 [cs].

[17] Kimmo Kärkkäinen and Jungseock Joo. 2019. FairFace: Face Attribute Dataset for Balanced Race, Gender, and Age. doi:10.48550/arXiv.1908.04913 arXiv:1908.04913 [cs].

[18] Pietro Melzi, Hatef Otroshi Shahreza, Christian Rathgeb, Ruben Tolosana, Ruben Vera-Rodriguez, Julian Fierrez, Sébastien Marcel, and Christoph Busch. 2023. Multi-ive: Privacy enhancement of multiple soft-biometrics in face embeddings. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. 323–331. https://openaccess.thecvf.com/content/WACV2023W/ DVPBA/html/Melzi_Multi-IVE_Privacy_Enhancement_of_Multiple_Soft- Biometrics_in_Face_Embeddings_WACVW_2023_paper.html
[19] Yuxi Mi, Yuge Huang, Jiazhen Ji, Hongquan Liu, Xingkun Xu, Shouhong Ding, and Shuigeng Zhou. 2022. DuetFace: Collaborative Privacy-Preserving Face Recognition via Channel Splitting in the Frequency Domain. In *Proceedings of* the 30th ACM International Conference on Multimedia. 6755–6764. doi:10.1145/ 3503161.3548303 arXiv:2207.07340 [cs].

986 985 987 988 989 990 991 992 993 994 995 996 997 998 999 1000 1001 1002 1003 1004 1005 1006 1007 1008 1009 1010 1011 1012 1013 1014 1015 1016 1017 1018 1019 1020 1021 1022 1023 1024 1025 1026 1027 1028 1029 1030 1031 1032 1033 1034 1035 1036 1037 1038 1039 1040 1041 1042 1043 9 1044
[20] Yuxi Mi, Yuge Huang, Jiazhen Ji, Minyi Zhao, Jiaxiang Wu, Xingkun Xu, Shouhong Ding, and Shuigeng Zhou. 2023. Privacy-preserving face recognition using random frequency components. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 19673–19684.

[21] Yuxi Mi, Zhizhou Zhong, Yuge Huang, Jiazhen Ji, Jianqing Xu, Jun Wang, Shaoming Wang, Shouhong Ding, and Shuigeng Zhou. 2024. Privacy-preserving face recognition using trainable feature subtraction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 297–307.

[22] National Cryptologic Centre. 2011. Characterizing Attacks to Fingerprint Verification Mechanisms.

[23] Daile Osorio-Roig, Paul A. Gerlitz, Christian Rathgeb, and Christoph Busch.

2023. Reversing Deep Face Embeddings with Probable Privacy Protection. doi:10. 48550/arXiv.2310.03005 arXiv:2310.03005 [cs].

[24] Dailé Osorio-Roig, Christian Rathgeb, Pawel Drozdowski, Philipp Terhörst, Vitomir Štruc, and Christoph Busch. 2022. An Attack on Facial Soft-Biometric Privacy Enhancement. IEEE Transactions on Biometrics, Behavior, and Identity Science 4, 2 (April 2022), 263–275. doi:10.1109/TBIOM.2022.3172724
[25] Foivos Paraperas Papantoniou, Alexandros Lattas, Stylianos Moschoglou, Jiankang Deng, Bernhard Kainz, and Stefanos Zafeiriou. 2024. Arc2Face: A Foundation Model for ID-Consistent Human Faces. doi:10.48550/arXiv.2403.11641 arXiv:2403.11641 [cs].

[26] Amir Sadovnik, Wassim Gharbi, Thanh Vu, and Andrew Gallagher. 2018. Finding Your Lookalike: Measuring Face Similarity Rather Than Face Identity. 2345– 2353. https://openaccess.thecvf.com/content_cvpr_2018_workshops/w48/html/ Sadovnik_Finding_Your_Lookalike_CVPR_2018_paper.html
[27] Ali Salar, Qing Liu, Yingli Tian, and Guoying Zhao. 2025. Enhancing facial privacy protection via weakening diffusion purification. In *Proceedings of the Computer* Vision and Pattern Recognition Conference. 8235–8244. http://openaccess.thecvf. com/content/CVPR2025/html/Salar_Enhancing_Facial_Privacy_Protection_ via_Weakening_Diffusion_Purification_CVPR_2025_paper.html
[28] Florian Schroff, Dmitry Kalenichenko, and James Philbin. 2015. FaceNet: A
Unified Embedding for Face Recognition and Clustering. In 2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 815–823. doi:10.1109/CVPR. 2015.7298682 arXiv:1503.03832 [cs].

[29] Fahad Shamshad, Muzammal Naseer, and Karthik Nandakumar. 2023.

Clip2protect: Protecting facial privacy using text-guided makeup via adversarial latent search. In *Proceedings of the IEEE/CVF Conference on Computer Vision* and Pattern Recognition. 20595–20605. http://openaccess.thecvf.com/content/
CVPR2023/html/Shamshad_CLIP2Protect_Protecting_Facial_Privacy_Using_
Text-Guided_Makeup_via_Adversarial_Latent_CVPR_2023_paper.html
[30] Xiaoxiao Sun, Nidham Gazagnadou, Vivek Sharma, Lingjuan Lyu, Hongdong Li, and Liang Zheng. 2023. Privacy Assessment on Reconstructed Images: Are Existing Evaluation Metrics Faithful to Human Perception? doi:10.48550/arXiv. 2309.13038 arXiv:2309.13038 [cs].

[31] The BioAPI Consortium. 2001. *BioAPI Specification Version 1.1*. [32] Fei Wang, Liren Chen, Cheng Li, Shiyao Huang, Yanjie Chen, Chen Qian, and Chen Change Loy. 2018. The Devil of Face Recognition is in the Noise. doi:10. 48550/arXiv.1807.11649 arXiv:1807.11649 [cs].

[33] Hanrui Wang, Shuo Wang, Chun-Shien Lu, and Isao Echizen. 2025. Diffusion-
Driven Universal Model Inversion Attack for Face Recognition. doi:10.48550/ arXiv.2504.18015 arXiv:2504.18015 [cs] version: 1.

[34] Kai Wang, Bo Zhao, Xiangyu Peng, Zheng Zhu, Jiankang Deng, Xinchao Wang, Hakan Bilen, and Yang You. 2022. FaceMAE: Privacy-Preserving Face Recognition via Masked Autoencoders. doi:10.48550/arXiv.2205.11090 arXiv:2205.11090 [cs].

[35] Tao Wang, Yushu Zhang, Xiangli Xiao, Lin Yuan, Zhihua Xia, and Jian Weng.

2024. Make Privacy Renewable! Generating Privacy-Preserving Faces Supporting Cancelable Biometric Recognition. In *Proceedings of the 32nd ACM International* Conference on Multimedia. ACM, Melbourne VIC Australia, 10268–10276. doi:10. 1145/3664647.3680704
[36] Yinggui Wang, Yuanqing Huang, Jianshu Li, Le Yang, Kai Song, and Lei Wang.

2024. Adaptive Hybrid Masking Strategy for Privacy-Preserving Face Recognition Against Model Inversion Attack. doi:10.48550/arXiv.2403.10558 arXiv:2403.10558 [cs].

[37] Yinggui Wang, Jian Liu, Man Luo, Le Yang, and Li Wang. 2022. Privacy-Preserving Face Recognition in the Frequency Domain. Proceedings of the AAAI Conference on Artificial Intelligence 36, 3 (June 2022), 2558–2566. doi:10.1609/aaai.v36i3.20157
[38] Xiao Yang, Yinpeng Dong, Tianyu Pang, Hang Su, Jun Zhu, Yuefeng Chen, and Hui Xue. 2021. Towards face encryption by generating adversarial identity masks. In Proceedings of the IEEE/CVF international conference on computer vision. 3897–3907. http://openaccess.thecvf.com/content/ICCV2021/html/Yang_ Towards_Face_Encryption_by_Generating_Adversarial_Identity_Masks_ ICCV_2021_paper.html
[39] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. 2023. IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models. doi:10.

48550/arXiv.2308.06721 arXiv:2308.06721 [cs].

[40] Dong Yi, Zhen Lei, Shengcai Liao, and Stan Z. Li. 2014. Learning Face Representation from Scratch. doi:10.48550/arXiv.1411.7923 arXiv:1411.7923 [cs].

| [41] Zhuowen Yuan, Zhengxin You, Sheng Li, Zhenxing Qian, Xinpeng Zhang, and Alex Kot. 2022. On Generating Identifiable Virtual Faces. In Proceedings of the 30th ACM International Conference on Multimedia. ACM, Lisboa Portugal, 1465–1473. doi:10.1145/3503161.3548110 [42] Kai Yue, Richeng Jin, Chau-Wai Wong, Dror Baron, and Huaiyu Dai. 2023. Gradient Obfuscation Gives a False Sense of Security in Federated Learning. In 32nd USENIX Security Symposium (USENIX Security 23). USENIX Association, Anaheim, CA, 6381–6398. https://www.usenix.org/conference/usenixsecurity23/ presentation/yue [43] Hui Zhang, Xingbo Dong, YenLung Lai, Ying Zhou, Xiaoyan Zhang, Xingguo Lv, Zhe Jin, and Xuejun Li. 2024. Validating Privacy-Preserving Face Recognition Under a Minimum Assumption. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 12205–12214. doi:10.1109/CVPR52733.2024. 01160   |
|---|

| 1045 1046 1047 1048 1049 1050 1051 1052 1053 1054 1055   |
|----------------------------------------------------------|

| 1056   |
|--------|

| 1057   |
|--------|

| 1058   |
|--------|

| 1059   |
|--------|

| 1061   |
|--------| | 1062   |
|--------|

| 1063   |
|--------|

| 1064   |
|--------|

| 1065   |
|--------|

| 1069   |
|--------|

| 1070   |
|--------|

| 1071   |
|--------|

| 1074   |
|--------|

ACM MM '26, 2026, Anon.

1103 1104 1105 1106 1107 1108 1109 1110 1111 1112 1113 1114 1115 1116 1117 1060 1118 1119 1120 1121 1122 1123 1124 1066 1125 1067 1068 1126 1127 1128 1129 1130 1072 1131 1073 1132 1075 1133 1134 1076 1135 1077 1136 1078 1137 1079 1138 1080 1081 1139 1140 1082 1141 1083 1142 1084 1143 1085 1144 1086 1087 1145 1088 1146 1147 1089 1090 1148 1149 1091 1150 1151 1092 1093 1094 1152 1153 1095 1096 1154 1155 1097 1156 1098 1157 1099 1100 1158 1101 1159 1102 1160 10

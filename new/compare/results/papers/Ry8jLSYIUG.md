000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053

# We Can Hide More Bits:

THE UNUSED WATERMARKING CAPACITY IN THEORY AND IN PRACTICE
Anonymous authors Paper under double-blind review

## Abstract

![0_image_0.png](0_image_0.png)

Figure 1: **Existing image watermarking models have capacities well under what this paper** suggests to be possible. Shown are theoretical bounds on watermarking capacity under a PSNR constraint alone (thick line) and in combination with robustness requirements (thin lines). Recent methods operate far below the achievable bounds, often by orders of magnitude, as seen in the log-scale inset. Our proposed **Chunky Seal (1024 bits)** pushes capacity higher than prior work, but is still very far from the theoretical limits, indicating a large potential for future development.

## 1 Introduction

Invisible image watermarking embeds an *imperceptible* secret message of a *certain capacity* recoverable *under a variety of perturbations*, leading to an inherent capacity-quality-robustness trade-off.

Classic methods used hand-crafted tools, such as the mid-frequencies of the discrete cosine transform (Al-Haj, 2007; Navas et al., 2008), discrete wavelet transform (Xia et al., 1998; Barni et al., 2001) or a combination of them (Navas et al., 2008; Feng et al., 2010; Zear et al., 2018). Deep learning led to significant improvements in all three dimensions via attacking fixed decoders (Vukotic et al. ´ , 2018; Fernandez et al., 2022), or via end-to-end training of the embedder and decoder (Mun et al., 2017; Zhu et al., 2018; Tancik et al., 2020; Bui et al., 2023a; Xu et al., 2025; Sander et al., 2025). Yet, despite these techniques, it seems that progress has stagnated. State-of-the-art methods successfully embed around 100−200 bits in a relatively imperceptible way (i.e., Peak Signal-to-Noise Ratio, PSNR, above 40 dB) while robust to perturbations. Improvements in quality and robustness continue, but they are only marginal, leading many to believe we are nearing the limits of what is possible.

Despite rapid progress in deep learning–based image watermarking, the capacity of current robust methods remains limited to the scale of only a few hundred bits. Such plateauing progress raises the question: How far are we from the fundamental limits of image watermarking? To this end, we present an analysis that establishes upper bounds on the message-carrying capacity of images under PSNR and linear robustness constraints. Our results indicate theoretical capacities are orders of magnitude larger than what current models achieve. Our experiments show this gap between theoretical and empirical performance persists, even in minimal, easily analysable setups. This suggests a fundamental problem. As proof that larger capacities are indeed possible, we train Chunky Seal, a scaled-up version of Video Seal, which increases capacity 4× to 1024 bits, all while preserving image quality and robustness. These findings demonstrate modern methods have not yet saturated watermarking capacity, and that significant opportunities for architectural innovation and training strategies remain.

1 054 055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107 Image watermarking indeed may already be a solved problem. Unlike generative or discriminative models that can improve as data and parameters are scaled, watermarking has an inherent performance ceiling. Given an image resolution and a set of robustness constraints, there is a finite amount of information that can be embedded imperceptibly. The existence of this limit and the converging empirical performance of recent models naturally leads to a critical question: **Have we already** reached the theoretical ceiling of watermarking performance? To answer this question, we need to know what this limit actually is and to measure how close our models are to it. We address these challenges in the current paper and offer the following findings:
i. We propose bounds on the capacity of watermarking under a PSNR constraint and robustness to linear augmentations, indicating capacities orders of magnitude larger than seen in practice.

ii. Watermarking models are trained with constraints we cannot directly analyse so we retrain Video Seal (Fernandez et al., 2024), a SOTA image and video watermarking model, to match our simplest theoretical setup: watermarking a single gray image under only a PSNR constraint. Yet, Video Seal fails to encode even 1024 bits, when we successfully encode 2048 bits with a linear model, 32,768 bits by tiling lower-resolution watermarks, and 456,509 bits with a handcrafted model. This indicates severe structural limitations.

iii. With the standard quality and robustness constraints, we train Chunky Seal, a simple scale-up of Video Seal, which embeds 1024 bits while maintaining similar robustness and image quality.1 Therefore, our theory and experiments show that **it is possible to achieve much higher capacities** than we currently have, although that might require innovation in architectures and training.

## 2 Bounds On Watermarking Capacity

We first discuss previous approaches to watermarking capacity in Section 2.1. We then model images as points on a high-dimensional grid, where capacity is determined by the number of unique points that satisfy imperceptibility and robustness constraints. Using this model, we first establish the absolute maximum information capacity (Section 2.2), then apply a PSNR constraint (Sections 2.3 and 2.4), and subsequently incorporate robustness to transformations like cropping, rescaling, and rotation (Section 2.5). We conclude by exploring the impact of data distribution on capacity (Section 2.6). 2.1 RELATED WORK ON THEORETICAL MODELS OF WATERMARKING CAPACITY Previous work on watermarking capacity largely relied on unrealistic assumptions like Gaussian noise (Costa, 1983; Cohen and Lapidoth, 2002; Chen and Wornell, 2002). More practical approaches were limited to small-magnitude perturbations (Moulin and O'Sullivan, 2003; Moulin and Koetter, 2005; Somekh-Baruch and Merhav, 2004) or specific geometric transformations (Merhav, 2005). Rather than these information-theoretic methods, which view the problem as power-limited communication over a super-channel with a state that is known to the encoder, our work takes a geometric approach allowing us to study more realistic conditions. Extended related works on image watermarking methods and theoretical approaches are discussed in App. A.

## 2.2 Absolute Capacity Of The Image Space

Watermarking embeds a message m into an image x. Since each message must correspond to a distinct encoded image, the number of unique messages, that is, the watermarking *capacity* is limited by the number of distinct images. An l-bit message requires at least 2 lsuch images. We represent an image as a vector of length cwh, where c is the number of channels, w is the width and h the height, with each element having 2 k discrete levels when using k-bit colour depth. The tuple (*c, w, h, k*)
defines an *image format*. The set of all possible images in this format is I = {0, 1*, . . . , ρ*}
cwh with ρ = 2k − 1, which can be thought of as a finite grid2 of points in R
cwh. This immediately gives us a trivial upper bound on watermarking capacity: since each message must correspond to a distinct watermarked image, it is not possible to embed more messages than there are distinct images. Bound 1: Absolute capacity of the image. The capacity of images in the format (*c, w, h, k*) is capacity[in bits] = log2 |I| = log2
(2k)
cwh= *cwhk* bits.

![2_image_2.png](2_image_2.png)

![2_image_3.png](2_image_3.png)

Figure 2: **The box-ball configurations of the PSNR-only constraint.** The cube CI is the set of all images and the sphere is the PSNR ball centred at the cover. Their intersection determines the set of feasible watermarked images, with the cardinality being the watermarking capacity. (a), (b) and (c) are the cases with the cover image x at the centre of the cube CI (gray image, resulting in highest capacity, Section 2.3). (d) is the case of the worst-case cover x, i.e., at the corner of CI (Section 2.4). Bound 1 simply states that the maximum number of embeddable bits is the uncompressed size of the image in bits. We next introduce imperceptibility and robustness to measure their effect on capacity.

## 2.3 Capacity Under A Psnr Constraint

A standard way to quantify distortion is the *peak signal-to-noise ratio (PSNR)*, measured in dB. Requiring a minimum PSNR τ between the cover x and the watermarked image x˜ is equivalent to bounding their ℓ2 distance (see App. B for the full derivation):

$\left(\mathrm{l}\right)_{\mathrm{f}}$

![2_image_1.png](2_image_1.png)

PSNR(x, x˜) ≥ τ *⇐⇒ ∥*x − x˜∥2 ≤ ϵ(τ ), with ϵ(τ ) = ρ
√cwh 10−τ/20. (1)
108

![2_image_0.png](2_image_0.png) 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 When τ is low, ϵ(τ ) is large and the ball contains the entire cube CI = [0, ρ]
cwh. The PSNR
constraint does not rule out any images, so the capacity is just the absolute maximum (Bound 1):
Bound 2: Gray image, PSNR constraint (low PSNR). The capacity of a gray image xg under a very low minimum PSNR threshold τ is capacity[in bits] = *cwhk.* Bound validity: When ϵ(τ ) ≥ ρ/2
√cwh, or equivalently when τ ≤ 20 log10 2 ≈ 6.02 dB.

## 2.3.2 Ball In Cube (High Psnr)

For small radii the volume approximation becomes inaccurate. However, then there are relatively few integer points in the ball and we can explicitly count them, as long as the dimension cwh of the ambient space is not too high. Instead of brute-force enumeration (which scales poorly), we use a method introduced by Mitchell (1966) leveraging symmetries for efficient counting (see Algorithm 2). Interpreting PSNR as an ℓ2-ball constraint gives us an avenue for measuring the message-carrying capacity under it by considering the amount of integer points inside both the cube and this ball. Counting how many such points exist is not trivial, and we analyse the three possible cases (see Figure 2): i. the ball is so large that it contains the entire cube (very low τ ); ii. the ball is small enough to lie fully inside the cube (high τ ); *iii.* the ball and cube partially overlap (medium τ ). We begin by assuming the cover image x lies at the centre of the admissible range, i.e., xg = 2k−1 1 as then the volume of the intersection (and thus the capacity) is maximized. In Section 2.4 we will extend the analysis to arbitrary images.

## 2.3.1 Cube In Ball (Low Psnr)

When τ is high, the PSNR ball is fully inside the cube. The capacity is the number of integer points inside the ball, a problem with no general closed form. In high dimensions cwh and sufficiently large radii ϵ(τ ), this is well approximated by the ball volume Vol Bcwh [·, ϵ(τ )] (see Appendix C).

Bound 3: Gray image, PSNR constraint (high PSNR, volume approximation). The capacity of a gray image xg under a high minimum PSNR threshold τ is approximately log2 Vol Bcwh [·, ϵ(τ )].

Bound validity: When the ball is fully inside the cube, i.e.ϵ(τ ) ≤ ρ/2 (i.e., τ ≥ 20 log10(2√cwh))
and ϵ(τ ) large enough for accurate volume approximation (see Bound 8 for small ϵ). Bound 4: Gray image, PSNR constraint (high PSNR, exact count). For small ϵ(τ ) the capacity is Bound validity: When ϵ(τ ) ≤ ρ/2 (i.e., τ ≥ 20 log10(2√cwh)) and ϵ(τ ) small enough that exact counting is computationally feasible.

## 2.3.3 Non-Trivial Intersection (Medium Psnr)

For intermediate PSNR values τ , Bcwh [xg, ϵ(τ )] and CI intersect non-trivially. We can approximate this count by the volume of the intersection, using the same volume-based method as in Bound 3. One can use exact volume computation (see Bound 5 in Appendix E), though this tends to be numerically unstable. In practice, a simpler upper bound approximates it well: Bound 6: Gray image, PSNR constraint (medium PSNR, approximation). The capacity of a gray image under minimum PSNR τ is upper-bounded by min[Bound 2, Bound 3].

Bound validity: When ρ/2 ≤ ϵ(τ ) ≤ ρ/2
√cwh, or equivalently 20 log10 2 ≤ τ ≤ 20 log10(2√cwh).

As shown in Fig. 3 left, this simple upper Bound 6 closely tracks the exact Bound 5. Thus Bound 6 is the practical choice going forward, while Bound 5 is provided in the appendix for completeness. Figure 3 left illustrates all the bounds from this section for a 16×16px image. At 45 dB these bounds give us roughly 2000 bits of capacity (more than 2.5 bpp): orders of magnitude more than the 0.001 bpp we see in practice (Figure 1).

## 2.4 From Central Gray Image To Arbitrary Cover Images

162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 In Section 2.3 we assumed the cover lies at the centre of the pixel range, thereby maximizing the volume of the intersection between the PSNR ball and the cube CI. Real images, however, may be anywhere in CI. Being at the corner of CI minimizes overlap with the ball and thus provides a lower bound valid for any image. When ϵ is not too large, exactly 1/2 cwh of the PSNR ball centred at a corner of CI remains inside CI. Although this may seem drastic, the penalty is in fact modest: at most cwh bits, i.e., one bit per pixel. In Appendix F we provide the formal bounds for this corner setting. Bound 7 adapts Bound 3, the volume approximation when the ball is fully in the cube. Bound 8 is the analogue of Bound 4, i.e., exact counting for small ϵ(τ ). Bound 9 parallels Bound 5 for the case when numerical integration is needed. As shown in Figure 3, the gap from the gray-only image bounds is at most 1 bpp, thus: **Watermarking with a PSNR constraint should allow for** capacity upwards of 2 bpp and does not explain the low capacities we observe in practice. We use Bound 4 whenever we can evaluate Algorithm 2 in reasonable time, and otherwise Bound 3. As shown in Figure 7, the transition between the two regimes is smooth.

![3_image_0.png](3_image_0.png)

capacity[in bits] = log2 PointsInHypersphereMitchell(dim = *cwh,* radius = ϵ(τ )).

216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269

![4_image_0.png](4_image_0.png)

## 2.5 Adding Robustness Constraints

In practice, watermarking must balance imperceptibility with robustness: the message should survive common processing, like compression, resizing, cropping, rotation, etc. In our model, we consider linear transformations, which encompass most transformations used in practice. We also develop LinJPEG, a linearized version of JPEG, allowing us to study the effects of compression in the same setting (see Appendix G.4 for the construction). Take a linear transformation M ∈ R
cwh×cwh that maps an image x to a transformed Mx and a quantization operation Q (element-wise rounding or floor operation) to map the pixel values of Mx to the valid images I. Hence, we have the final transformed image x
′ = Q[Mx]. We need to find the subset of the possible watermarked images under only the PSNR constraint that map to unique valid images after applying M and Q to them. The main complication in this setup is that Q is non-linear. Heuristic bounds. A simple approach is to take a volumetric approach akin to Bounds 3, 5, 7 and 9.

We factor in how M changes the volume and account for directions compressed by the transformation which destroy capacity as different watermarked images get collapsed together. We also account for directions fully collapsed by M when it is singular. Finally, the stretched directions might result in some watermarked images being outside CI after the transformation, leading to them being clipped. Bounds 10 to 12 use a heuristic based on the singular values of M to account for the effect on capacity. Refer to Appendix G.2 for details. In Figure 4 we plot these bounds for robustness to rotation, cropping followed by rescaling and LinJPEG, showing that even under the most aggressive cropping, we should expect around 0.5 bpp or almost 100,000 bits for 256×256px images. Conservative bounds. We can show cases where these heuristic bounds under-approximate and cases where they over-approximate the true capacity, e.g., Figures 8 and 9. Thus, the true capacity under linear transformation could be much lower than these bounds predict. To ensure that this is not the case, we develop an actual lower bound: Bound 13. While we reserve the details for Appendix G.3, this bound is based on over-approximating the set of images that can be quantized by Q to the same image after M is applied to them. As a result, Bound 13 is extremely conservative and unrealistic. We believe that despite Bounds 10 to 12 not being valid lower bounds, they are much closer to the true capacity. Still, we report the conservative bound in Table 2: the most aggressive crop still leaves at least 904 bits for 256×256px images. For the other augmentations, the conservative capacity is much higher. Therefore, **robustness to geometric transformations and compression significantly** reduces the capacitybut cannot fully explain the low watermarking capacity of current models.

## 2.6 From Single Cover Images To Datasets And Data Distributions

In a blind watermarking setup, the decoder must operate without access to the original cover image, creating potential collisions: if multiple natural images (i.e., potential covers) are very close to each other in pixel space, a watermarked version of one cover could be identical to a watermarked version of another. To prevent such ambiguity, the total set of watermarked images within a given region 270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323

![5_image_0.png](5_image_0.png) 

(like the PSNR ball) must be partitioned among all the potential covers it contains. If there are N possible covers, the capacity for each is reduced by log2(N) bits. We estimate N using neural compression models like VQ-VAE (Van Den Oord et al., 2017) and VQGAN (Esser et al., 2021), which upper-bound the number of perceptually distinct images. For instance, a 256×256px image can be compressed into a 32×32 latent with a 1024-entry codebook (Muckley et al., 2023). This representation can express at most 102432×32 = 210240 distinct images. Conservatively assuming all could fall in the PSNR ball of the considered image, capacity is reduced by 10, 240 bits, or about 0.05 bpp, on top of the 1 bpp loss from Section 2.4. Thus, from this perspective, **the data distribution** has only a negligible effect on watermarking capacity and cannot explain the low performance of current models. This aligns with prior findings for Gaussian channels that decoder knowledge of the cover does not affect capacity (Costa, 1983; Chen and Wornell, 2002; Moulin and O'Sullivan, 2003).

## 3 Empirical Performance Is Much Lower Than Predicted

Section 2 showed that capacities of over 2 bpp at PSNR of 40 dB without robustness constraints, and of 0.5 bpp with robustness, are possible. Even under the very conservative Bound 13 we still would expect capacities of at least 0.01 bpp. However, in practice, the models reported in the literature have significantly lower capacities (less than 0.001 bpp, Figure 1). To understand the cause of this gap, this section asks: **are existing models significantly under-performing relative to what is possible** in practice,or are our bounds too unrealistic? There are five possible explanations of the large discrepancy between the performance we see in practice (Figure 1) and the bounds in Section 2:
A. **Real models might be near-optimal if we consider advanced robustness constraints.** B. **Real models might be near-optimal if we consider advanced perceptual constraints.** C. **Real models might be near-optimal if we consider real-world image distributions.** D. **Our bounds overestimate capacity and cannot be approached empirically.** E. **We can do much better and push the Pareto front well beyond the current state-of-the-art.**
To understand the cause of the gap between theoretical and real-world performance in image watermarking, we need to find out which of these hypotheses is the underlying cause. If it is A., B., C., D., or a combination of them, then it is possible that, indeed, the best current models are close to what is ultimately possible and we can expect only marginal further improvements. On the other hand, if the cause is E., then that means that there is plenty of space for significant improvements.

## 3.1 The Real-World Complexity Does Not Explain The Performance Gap

Let's first address cases A., B., C., i.e., that our bounds cannot capture the complexity of the robustness, quality and data constraint with which real models are trained. While we cannot bring the real-world complexity to our analytical bounds, we can bring the models to the simplified theoretical setup.

324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377

Message size Message size if

tiled to 256x256px PSNR Bit acc. λi lr

| Message size                                                                                    | Message size if   |          |         |      |      |
|-------------------------------------------------------------------------------------------------|-------------------|----------|---------|------|------|
| VideoSeal (256x256px, 600 epochs) VideoSeal (32x32px, 600 epochs) Linear (256x256px, 50 epochs) | 1024 bits         | 44.28 dB | 100.00% | 20.0 | 5e-4 |
| 623232 bits                                                                                     | 36.00 dB          | 100.00%  |         |      |      |
| 551948 bits                                                                                     | 38.00 dB          | 100.00%  |         |      |      |
| Handcrafted                                                                                     | 456509 bits       | 42.00 dB | 100.00% |      |      |
| 311616 bits                                                                                     | 48.00 dB          | 100.00%  |         |      |      |

128 bits 8192 bits 51.02 dB 100.00% 1.0 5e-4 256 bits 16384 bits 48.98 dB 100.00% 1.0 5e-4 512 bits 32768 bits 41.66 dB 100.00% 1.0 5e-5

1024 bits 65536 bits 29.66 dB 84.39% 0.1 5e-5 1024 bits 65536 bits 33.20 dB 83.86% 0.5 5e-5 1024 bits 65536 bits 34.63 dB 83.78% 1.0 5e-5 1024 bits 65536 bits 50.83 dB 50.60% 0.5 5e-4

More concretely, we take the simplest of setups: a single gray image with a PSNR constraint, as in Section 2.3. We will use Video Seal as the base for our experiments (Fernandez et al., 2024), originally introduced as an image watermarking model with frame copying that generalizes to video. It was first demonstrated with a 96-bit capacity and was recently extended to a 256-bit open-source version, which we use as the strongest available baseline. To match the setup of Section 2.3, we replace the dataset with a single solid gray image, remove all perceptual constraints but the MSE loss and remove all augmentations. We first retrain it for nbits = 128, 256, 512, and 1024 bits. We have hereby reduced the task to simply find a way to encode nbits into a single fixed image. From Figure 3 we expect capacities of around 600,000 bits at 40 dB in this setup. Thus, the model should easily learn these much lower nbits. We train with AdamW (Loshchilov and Hutter, 2019) with batch size 256 for 600 epochs, 1000 batches per epoch, cosine learning rate schedule with a 20-epoch warm-up, similarly to Video Seal. We sweep over the learning rate (5e-4, 5e-5, 5e-6) and λi, the MSE loss weight (0.1, 0.5, 1.0), with LR=5e-5 and λi = 0.5 being the values used for training Video Seal. The results of training Video Seal on a single gray image can be seen in Figure 5 left and Table 1. There are runs for the 128, 256 and 512 bit models that do achieve 100% bit accuracy and PSNR values above 42dB. However, Video Seal cannot even get to 1024 bits, far from what we expect from the bounds. This is surprising: the model cannot approach the theoretical bounds even after removing the complexities that supposedly make watermarking difficult. This means that neither A., B. nor C. can explain why we see such a gap between the theoretical and real-world performance. 3.2 OUR SIMPLEST BOUNDS ARE ACHIEVABLE, YET MODELS STRUGGLE TO GET NEAR THEM Section 3.1 showed that Video Seal cannot match the capacity predicted by the bounds in Section 2.3 even when trained only on a single gray image and with no augmentations. Thus, the complexity of real world watermarking cannot explain the gap between the theoretical and real-world performance. This leaves us with two options: D. our bounds are wrong and unachievable, or E. our models are under-performing. There are a couple simple experiments that can demonstrate that we can get much closer to the bounds in Section 2.3 and hence D. also does not explain the gap. Linear embedder and extractor. We trained a simple linear embedder and extractor. The embedder gets the 1024 bit message (shifted and scaled to −1 and +1 values) and produces a 256×256×3 watermark residual which gets added to the original gray image. Similarly, the decoder is a linear layer from the flattened 256×256×3 image to 1024 outputs, which are thresholded to recover the message. We train only for 50 epochs, with the same learning rate values and λi ∈ {4, 8, 12, 20}.

Table 2: **Conservative capacity** bounds under robustness constraints for PSNR 42 dB. These values are calculated via Bound 13 and are strongly conservative lower bounds on the capacity that is achievable while maintaining robustness to the respective transformations and PSNR under 42 dB.

![7_image_0.png](7_image_0.png)

| Conservative capacity   |       |             |               |
|-------------------------|-------|-------------|---------------|
| Augmentation            | bpp   | for 16×16px | for 256×256px |
| Horizontal Flip         | 3.064 | 2,352 bits  | 602,353 bits  |
| Crop&Rescale 50%        | 0.015 | 11 bits     | 3,013 bits    |
| Crop&Rescale 75%        | 0.005 | 3 bits      | 904 bits      |
| LinJPEG q=10            | 0.136 | 104 bits    | 26,757 bits   |
| LinJPEG q=15            | 0.137 | 105 bits    | 27,020 bits   |
| Rotation 30deg          | 0.075 | 57 bits     | 14,676 bits   |
| Rotation 45deg          | 0.083 | 64 bits     | 16,401 bits   |

Figure 6: **Simple models outperform**
Video Seal on a gray image with only a PSNR constraint. Experiments from Section 3 compare our theoretical bounds (Section 2.3) against trained models. Video Seal falls well below the predictions, while a linear model performs slightly better and a tiled 32×32px Video Seal is even better. Our handcrafted model nearly matches the bound.

The results in Figure 5 right and Table 1 show the linear layer learns what Video Seal could not: 100% bit accuracy for 1024 bits with PSNR of 44 dB. We also trained a linear model for 2048 bits which achieved 100% bit accuracy. This shows that capacities beyond 512 bits are possible in practice (at least for a gray image and no robustness) and are learnable via gradient descent. All one needs is the right architecture. Lower-resolution training and tiling. Our experiments reveal that Video Seal does not exploit the additional degrees of freedom available at higher image resolutions. When trained at 256×256px, the model achieves essentially the same capacity and PSNR as when trained at 32×32px (see Section 3.1 and Table 1). To verify this, we train Video Seal in the setup of Section 3.1 at 32×32px using the same learning-rate and λi sweeps for 600 epochs. As shown in Figure 5 (centre) and Table 1, the performance at 32×32px is nearly identical to that at 256×256px: the 512-bit model reaches 100% bit accuracy with 41.7 dB, despite operating on 64× fewer pixels. In other words, the effective capacity we observe at 256×256px is comparable to what one would expect around 20×20px, confirming that the architecture fails to utilize the available resolution. Because this setup does not require robustness to geometric or valuemetric transformations and we consider only gray images, we can use the 32×32px model to demonstrate that higher capacities are possible. A simple tiling strategy suffices: each tile is embedded with an independent secret using the same model. The decoder similarly is applied per patch with the individual decoded messages concatenated to obtain the final combined message. Using 256×256px as the reference size, tiling yields 64× the capacity of the base 32×32px model. Thus, tiling the 512-bit model—which already achieves 100% accuracy at 41.7 dB—produces a watermark with 32,768 bits total capacity while maintaining the same PSNR (which is resolution-independent). This effective capacity of 32,768 bits is already much closer to our bound of roughly 600,000 bits, though still only about 0.167 bpp. It is interesting that the model could not learn at the 256×256px resolution even for 1024 bits when it is clear that it is possible to embed 32,768 bits as seen here. More importantly, this shows that our bounds are not that far off and capacities of at least 32,768 bits are indeed possible.

378 379 380 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431 Handcrafted embedder and extractor. We can do even better by manually crafting an embedder and extractor. The key observation is that mapping a hypercube to binary messages is easy. Take the ball of radius ϵ(τ ) = ρ
√cwh 10−τ/20 from Equation (1). The half-side of the largest cube that can fit in this ball is d = ϵ(τ )/
√cwh = ρ 10−τ/20. We have that each edge of the box contains a cwh-dimensional grid of q = 2⌊d⌋ + 1 = 2⌊2 k 10−τ/20⌋ + 1 points per side. Hence, that gives us 432 433 434 435 436 437 438 439 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485 Table 3: **Chunky Seal performance**
on images from SA-1B (Kirillov et al., 2023) at their original resolution. Chunky Seal has much higher capacity (1024 bits) than Video Seal while preserving its image quality and robustness on a wide variety of transformations. The improvement is driven by scaling the model size and its training. Extended results on SA-1B (Kirillov et al., 2023) and COCO (Lin et al., 2014) as well as qualitative results, are reported in Appendix J.1 total capacity in bits of

$$\log_{2}\left[\left(2\left[2^{k}\,10^{-\tau/20}\right]+1\right)^{cwh}\right]=cwh\log_{2}\left[2\left[2^{k}\,10^{-\tau/20}\right]+1\right]=cwh\log_{2}q,\tag{2}$$  where $c$ is the $\tau$-function of $\tau$ and $\tau$ is the $\tau$-function of $\tau$. The $\tau$-function is given by 
or log2q bits per pixel. See Figure 6 for a plot of that for different PSNR values. For 42 dB, and images of 256×256px that gives us a capacity of 456,509 bits (see Table 1) almost 14× what we could embed with the 32×32px tiling approach. Moreover, it gets us close to the theoretical bound. Therefore, we can get much closer to the boundary, at least in the solid gray image case with PSNR constraint and no robustness requirements. Thus, case D., that our bounds are wrong and impossible to achieve, is unlikely. This leaves us with one possible explanation as to why models in practice do not exhibit performance anywhere near what our theory predicts. That would be option E: Our models are likely significantly underperforming relative to what is possible in practice. We likely can do much better and push the Pareto front well beyond the current state-of-the-art.

## 4 Better Performance In Practice Is Possible: Chunky Seal

While it remains possible that current models approach a theoretical limit under robustness and quality constraints, training a watermarking model with comparable quality and robustness but with substantially higher capacity would decisively rule this out. We take Video Seal (Fernandez et al., 2024) as the base model and train it for 1024 bits. We increased the embedding dimension to 2048, the U-Net channel multipliers from [1, 2, 4, 8] to [4, 8, 16, 32], and enabled watermarking in all three channels, not just the luma (Y) channel. This results in an embedder 90× larger than the original Video Seal embedder. The ConvNeXt (Liu et al., 2022) extractor was similarly scaled: we increased the depths for each stage from [3, 3, 9, 3] (as in ConvNeXt-tiny) to [3, 3, 27, 3] (as in ConvNeXt-base), with their dimensions increased from [96, 192, 384, 768] to [256, 512, 1024, 2048]. The stride of the first layer was reduced from 4 to 2. This results in an extractor that is 23× larger than the original Video Seal extractor. Due to its significantly increased size, we name this model Chunky Seal. We train it at the original 256×256px resolution. We apply gradient clipping with a maximum norm of 0.01, which proved critical for stabilizing training. As shown in Table 3, Chunky Seal shows image quality and robustness comparable to Video Seal across a wide range of distortions, while providing a 4× **higher message capacity** (1024 vs. 256 bits). Despite its much larger capacity, Chunky Seal maintains nearly identical image quality across all metrics, and only slightly higher LPIPS. The robustness results further confirm that Chunky Seal sustains high bit-accuracy across transformations such as rotation, resizing, cropping, brightness and contrast changes, JPEG compression, and blurring, closely matching Video Seal. We emphasize that these results were achieved *without hyperparameter tuning*, whereas Video Seal was extensively optimized for quality and robustness. Achieving 4× **the capacity per pixel with comparable** robustness and quality through simple scaling strongly suggests that substantially higher capacities are within reach using improved architectures and training strategies.

| Chunky Seal (ours)             | Video Seal 256bits   |               |
|--------------------------------|----------------------|---------------|
| Capacity                       | 1024 bits            | 256 bits      |
| 0.0052 bpp                     | 0.0013 bpp           |               |
| Embedder size                  | 1022.7M              | 11.0M         |
| Extractor size                 | 773.7M               | 33.0M         |
| PSNR ↑                         | 45.32±2.16           | 44.42±2.21    |
| SSIM ↑                         | 0.995±0.006          | 0.996±0.003   |
| MS-SSIM ↑                      | 0.997±0.002          | 0.997±0.001   |
| LPIPS ↓                        | 0.0085±0.0067        | 0.0019±0.0011 |
| Bit acc. Identity              | 99.74±0.28%          | 99.90±0.21%   |
| Bit acc. Flip                  | 99.65±0.34%          | 99.89±0.24%   |
| Bit acc. Rotate (≤10°)         | 98.27±2.10%          | 98.84±1.10%   |
| Bit acc. Resize (71–95%)       | 99.74±0.28%          | 99.90±0.21%   |
| Bit acc. Crop (77–95%)         | 98.25±1.75%          | 98.04±1.57%   |
| Bit acc. Brightness (0.5–1.5×) | 98.99±1.87%          | 98.67±2.67%   |
| Bit acc. Contrast (0.5–1.5×)   | 99.54±0.51%          | 99.56±0.45%   |
| Bit acc. JPEG (Q 50–80)        | 98.79±0.75%          | 99.74±0.47%   |
| Bit acc. Gaussian Blur (k≤9)   | 99.74±0.28%          | 99.90±0.22%   |
| Bit acc. Overall               | 99.15±0.63%          | 99.31±0.60%   |

## 5 Discussion And Conclusions

486 487 488 489 490 491 492 493 494 495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539 Higher watermarking capacities open up new avenues for content provenance. Instead of using a watermark to retrieve a C2PA manifest from a third-party database (Collomosse and Parsons, 2024), we could embed the entire manifest, eliminating the need for a registry. Beyond this, improvements in capacity can be traded for greater robustness or higher image quality, depending on the application. The fact that our theoretical capacity bounds are an order of magnitude higher than even the best existing models also helps explain why applying and detecting multiple watermarks on a single image is feasible, as demonstrated by (Petrov et al., 2025). Despite achieving substantially higher capacities than prior models, Chunky Seal still remains far from the theoretical bounds established in this work. Our controlled experiments show that this gap cannot be attributed to factors such as data distribution, resolution, or augmentations. Instead, the evidence consistently points to limitations in the model architecture itself. Learning an identity map is notoriously difficult for neural networks (He et al., 2016; Hardt and Ma, 2017), a point underscored by the fact that simple linear models outperform Video Seal in settings where the architecture should, in principle, excel. Importantly, we do not suggest that na¨ıvely scaling Chunky Seal is a practical path forward. The purpose of this scaling exercise was to explore feasibility, not to advocate for large models in deployment. These results simply illustrate that current architectures fall well short of saturating watermarking capacity, even under generous scaling. Looking ahead, we argue that substantial progress will require new architectural designs, improved losses, and revised training procedures that better encode the inductive biases inherent to watermarking, rather than further scaling of existing models.

We therefore propose a set of sanity checks for the next generation of watermarking methods. A principled approach should scale capacity linearly with image size, decrease capacity linearly with higher PSNR, outperform simple linear or handcrafted baselines, and show predictable drops under stronger augmentations (e.g., 4× lower capacity for a 25% crop). These are necessary for Pareto-optimality and can steer the community toward watermarks with far higher capacity or quality. Our analysis is not without limitations. We restricted our study to image watermarking, though the insights likely carry over to video. Theoretical bounds are derived only for analytically tractable setups, with some cases relying on numerical integration that becomes impractical at higher resolutions. Our robustness bounds are heuristic rather than formal, leaving ample room for sharper theoretical advances. Finally, while Chunky Seal delivers clear performance gains, its size and latency highlight the need for future architectures that deliver both higher capacities and efficiency.

## References

Ali Al-Haj. 2007. Combined DWT-DCT digital image watermarking. *Journal of Computer Science*,
3(9):740–746.

Matthias Althoff, Olaf Stursberg, and Martin Buss. 2010. Computing reachable sets of hybrid systems using a combination of zonotopes and polytopes. *Nonlinear analysis: Hybrid systems*, 4(2):233–249.

Yoshinori Aono and Phong Q Nguyen. 2017. Random sampling revisited: Lattice enumeration with discrete pruning. In *Annual International Conference on the Theory and Applications of* Cryptographic Techniques, pages 65–102.

Mauro Barni, Franco Bartolini, and Alessandro Piva. 2001. Improved wavelet-based watermarking through pixel-wise masking. *IEEE transactions on image processing*, 10(5):783–791.

Patrick Bas, J-M Chassery, and Benoit Macq. 2002. Geometrically invariant watermarking using feature points. *IEEE transactions on image Processing*, 11(9):1014–1028.

Adrian G Bors and Ioannis Pitas. 1996. Image watermarking using DCT domain constraints. In *ICIP*.

Tu Bui, Shruti Agarwal, and John Collomosse. 2023a. Trustmark: Universal watermarking for arbitrary resolution images. *arXiv preprint arXiv:2311.18297*.

Tu Bui, Shruti Agarwal, Ning Yu, and John Collomosse. 2023b. RoSteALS: Robust steganography using autoencoder latent space. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops.

Brian Chen and Gregory W Wornell. 2002. Quantization index modulation: A class of provably good methods for digital watermarking and information embedding. IEEE Transactions on Information theory, 47(4):1423–1443.

Xiangyu Chen, Varsha Kishore, and Kilian Q Weinberger. 2023. Learning iterative neural optimizers for image steganography. In *International Conference on Learning Representations*.

540 541 542 543 544 545 546 547 548 549 550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 Aaron S Cohen and Amos Lapidoth. 2002. The Gaussian watermarking game. IEEE Transactions on Information Theory, 48(6):1639–1667.

Seongmin Hong, Kyeonghyun Lee, Suh Yoon Jeon, Hyewon Bae, and Se Young Chun. 2024. On exact inversion of DPM-solvers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.

John Collomosse and Andy Parsons. 2024. To authenticity, and beyond! Building safe and fair generative AI upon the three pillars of provenance. *IEEE Computer Graphics and Applications*.

Denis Constales. 1997. Solution to "The volume of the intersection of a cube and a ball in N-space" posed by Liqun Xu. *SIAM Review (Problems and Solutions)*, 39.4:779–786.

Max Costa. 1983. Writing on dirty paper. *IEEE Transactions on Information Theory*, 29(3):439–441. I.J. Cox, J. Kilian, F.T. Leighton, and T. Shamoon. 1997. Secure spread spectrum watermarking for multimedia. *IEEE Transactions on Image Processing*, 6(12):1673–1687.

Patrick Esser, Robin Rombach, and Bjorn Ommer. 2021. Taming transformers for high-resolution image synthesis. In *Proceedings of the IEEE/CVF conference on computer vision and pattern* recognition.

Liu Ping Feng, Liang Bin Zheng, and Peng Cao. 2010. A DWT-DCT based blind watermarking algorithm for copyright protection. In 2010 3rd International Conference on Computer Science and Information Technology, volume 7, pages 455–458.

Pierre Fernandez, Guillaume Couairon, Herve J ´ egou, Matthijs Douze, and Teddy Furon. 2023. ´ The stable signature: Rooting watermarks in latent diffusion models. In International Conference on Computer Vision.

Pierre Fernandez, Hady Elsahar, I Zeki Yalniz, and Alexandre Mourachko. 2024. Video Seal: Open and efficient video watermarking. *arXiv preprint arXiv:2412.09492*.

Pierre Fernandez, Alexandre Sablayrolles, Teddy Furon, Herve J ´ egou, and Matthijs Douze. 2022. ´
Watermarking images in self-supervised latent spaces. In ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP).

Carl Friedrich Gauss. 1837. De nexu inter multitudinem classium, in quas formae binariae secundi gradus distribuuntur, earumque determinantem. In *Werke: Band 2*, pages 269–291.

S. I. Gel'fand and M. S. Pinsker. 1980. Coding for channel with random parameters. Problems of Control and Information Theory, 9(1):19–31.

Antoine Girard. 2005. Reachability of uncertain linear systems using zonotopes. In Proceedings of the 8th International Conference on Hybrid Systems: Computation and Control.

Moritz Hardt and Tengyu Ma. 2017. Identity matters in deep learning. In *International Conference* on Learning Representations.

G. H. Hardy. 1915. On the expression of a number as the sum of two squares. Quarterly Journal of Mathematics, 46:263–283.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. 2016. Deep residual learning for image recognition. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*.

Hai Ci, Pei Yang, Yiren Song, and Mike Zheng Shou. 2024. RingID: Rethinking tree-ring watermarking for enhanced multi-key identification. *arXiv preprint arXiv:2404.14055*.

Zhaoyang Jia, Han Fang, and Weiming Zhang. 2021. MBRS: Enhancing robustness of DNN-based watermarking by mini-batch of real and simulated JPEG compression. In Proceedings of the 29th ACM international conference on multimedia.

Changhoon Kim, Kyle Min, Maitreya Patel, Sheng Cheng, and Yezhou Yang. 2023. Wouaf: Weight modulation for user attribution and fingerprinting in text-to-image diffusion models. arXiv preprint arXiv:2306.04744.

Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. 2023. Segment anything. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4015–4026.

Varsha Kishore, Xiangyu Chen, Yan Wang, Boyi Li, and Kilian Q Weinberger. 2022. Fixed neural network steganography: Train the images, not the network. In International Conference on Learning Representations.

594 595 596 597 598 599 600 601 602 603 604 605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 Minzhou Pan, Yi Zeng, Xue Lin, Ning Yu, Cho-Jui Hsieh, Peter Henderson, and Ruoxi Jia. 2024.

JIGMARK: A black-box approach for enhancing image watermarks against diffusion model edits. arXiv preprint arXiv:2406.03720.

Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollar, and C Lawrence Zitnick. 2014. ´ Microsoft COCO: Common objects in context. In Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13.

Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie.

2022. A Convnet for the 2020s. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11976–11986.

Ilya Loshchilov and Frank Hutter. 2019. Decoupled weight decay regularization. In International Conference on Learning Representations.

Xiyang Luo, Ruohan Zhan, Huiwen Chang, Feng Yang, and Peyman Milanfar. 2020. Distortion agnostic deep watermarking. In *CVPR*.

Neri Merhav. 2005. An information-theoretic view of watermark embedding-detection and geometric attacks. *Proceedings of WaCha, 2005, First Wavila Challenge*.

WC Mitchell. 1966. The number of lattice points in a k-dimensional hypersphere. *Mathematics of* Computation, 20(94):300–310.

Pierre Moulin and Ralf Koetter. 2005. Data-hiding codes. *Proceedings of the IEEE*, 93(12):2083–
2126.

Pierre Moulin and Joseph A O'Sullivan. 2003. Information-theoretic analysis of information hiding.

IEEE Transactions on information theory, 49(3):563–593.

Matthew J. Muckley, Alaaeldin El-Nouby, Karen Ullrich, Herve Jegou, and Jakob Verbeek. 2023.

Improving statistical fidelity for neural image compression with implicit local likelihood models. In *Proceedings of the 40th International Conference on Machine Learning*, pages 25426–25443.

Seung-Min Mun, Seung-Hun Nam, Han-Ul Jang, Dongkyu Kim, and Heung-Kyu Lee. 2017. A
robust blind watermarking using convolutional neural network. *arXiv preprint arXiv:1704.03248*.

K. A. Navas, Mathews Cheriyan Ajay, M. Lekshmi, Tampy S. Archana, and M. Sasikumar. 2008.

DWT-DCT-SVD based watermarking. In 2008 3rd International Conference on Communication Systems Software and Middleware and Workshops (COMSWARE '08), pages 271–274.

Zhicheng Ni, Yun-Qing Shi, N. Ansari, and Wei Su. 2006. Reversible data hiding. IEEE Transactions on Circuits and Systems for Video Technology, 16(3):354–362.

Aleksandar Petrov, Shruti Agarwal, Philip HS Torr, Adel Bibi, and John Collomosse. 2025. On the coexistence and ensembling of watermarks. *arXiv preprint arXiv:2501.17356*.
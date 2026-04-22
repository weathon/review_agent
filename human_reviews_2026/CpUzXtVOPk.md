# Training-Free Rate-Distortion-Perception Traversal With Diffusion

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 6, 4, 4, 2

## Abstract
The rate-distortion-perception (RDP) tradeoff captures the fundamental limits of lossy compression by jointly considering bitrate, reconstruction fidelity, and perceptual quality. While recent neural compression methods have improved perceptual performance, they typically operate at a fixed point on the RDP surface, requiring retraining to target different tradeoffs. In this work, we propose a training-free framework for traversing the full RDP surface, utilizing pretrained diffusion models. Our approach integrates a reverse channel coding (RCC) encoder with a novel score-scaled probability flow ODE decoder. We theoretically prove that the proposed decoder is optimal for the distortion-perception tradeoff under AWGN observations and that the overall framework with the RCC encoder is optimal for the RDP function in the Gaussian case. Empirical results across multiple datasets demonstrate the framework's flexibility and effectiveness in navigating the ternary RDP tradeoff using pre-trained diffusion models. Our results establish a practical and theoretically grounded approach to adaptive, perception-aware compression.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper propose a lossy image compression scheme based on the DiffC algorithm to traverse the rate-distortion-perception frontier. While DiffC achieves perfect realism for reconstructions, the current work modifies the decoder by simulating a score-scaled PF-ODE (eq. 5). The authors further prove optimality under Wasserstein-2 perception measure of the proposed scheme for scalar Gaussian sources. In practice, the conditional distributions are approximated by a pre-trained diffusion model. Shared seed is required for encoder and decoder to implement the algorithm. Experiments are implemented on CIFAR-10, Kodak, and Div2k under LPIPS perception measure.

### Strengths
The paper provides a practical image lossy compression scheme using pre-trained diffusion model to traverse the RDP frontier. The main theoretical contribution is Eq. (5), which traverses two extremes: reverse SDE in Eq. (3) and PF-ODE in Eq. (4). For Gaussian cases, the proposed scheme is shown to achieve optimality. 

The proposed scheme is training-free, and practical algorithms are given and implemented on CIFAR-10, Kodak, and Div2k datasets to demonstrate effectiveness.

### Weaknesses
I am most concerned with the motivation for traversing the RDP frontier. As is mentioned in the paper, a GAN-based scheme is available in Zhang et al. (2021). Comparisons between the proposed scheme and the GAN-based scheme regarding practical constraints such as latency, compute, and memory usage are not provided.

### Questions
1. In Eq. (5), when $\rho=0$, should $Z_t$ follow $p_{T_c}$? What are the conditions for Eq. (5) to hold at $\rho=0$?

2. Do the optimality results and the results in Lemma 1 hold for Gaussian cases because of the coincident $p_{T_c}=W_\tau$ therein? Could you provide some insights into more complex sources such as mixture Gaussian? 

3. Does the shared seed need to be entropy coded in Alg 1 and Alg 2?

4. In line 320, should $z_k = \bar{z}_k^{(c_k)}$ be $z_t =...$?

5. In the experiment, LPIPS is used as the perception measure, while in section 3 and section 4, W2 is used. Why?

6. Could you compare with the GAN-based scheme (Zhang et al. 2021) in terms of latency, compute, and memory?

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a training-free compression framework that allows traversal of the Rate-Distortion-Perception tradeoff using pre-trained diffusion models. It uses a reverser-channel-coding encoder and a probability-flow ODE based decoder. The decoder allows for he distortion-perception tradeoff whereas the encoder can vary the compression rate. Along with theoretical proofs the paper also shows empirical results on multiple datasets using a pre-trained diffusion model.

### Strengths
The paper is well-written and easy to follow.

1. The idea to re-purpose a trained diffusion model for adaptive compression is well-motivated and allows to overcome the expensive retraining of decoders.

2. The factors controlling the RDP tradeoff - sampling step and scaling are intuitive and easy to follow.

Overall, the paper is exploring a useful idea of training-free RDP traversal. Both the theoretical and empirical justifications look promising.

### Weaknesses
1. Although the method described in the paper is training-free, it appears to be expensive during inference-time as it needs multiple ODE iterations to traverse the RDP  curve and the pre-trained model itself might be much more expensive than other GAN/hybrid codecs.

2. An ablation study disentangling the effect of RCC encoder and the PF Decoder could be useful here.  That is, how much contribution towards rate-distortion comes from RCC encoder vs the PF-ODE decoder. For example, fixing one parameter (say the scale) and varying the encoder parameter, would the distortion perception curve move around much?

Looking forward to authors' responses and will adjust my score accordingly.

### Questions
Same as weaknesses.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a training-free framework for navigating the rate-distortion-perception (RDP) tradeoff in lossy compression using pre-trained diffusion models. The method combines a reverse channel coding (RCC) encoder with a novel score-scaled probability flow ODE decoder, controlled by two parameters: the noise level t (rate) and the score-scaling factor ρ (distortion-perception tradeoff). Theoretically grounded and empirically validated, this approach enables flexible traversal of the entire RDP surface using a single model, achieving a versatile balance between bitrate, reconstruction fidelity, and perceptual quality without any retraining.

### Strengths
Training-Free and Generalizable: It directly leverages pre-trained diffusion models without requiring retraining for different rates or trade-offs. A single model, controlled by the two knobs t and ρ, covers a wide R-D-P region, enabling a perception-distortion trade-off.
Rigorous Theoretical Proofs: The paper provides extensive theoretical proofs, significantly enhancing its readability and credibility.

### Weaknesses
1. Perceived Lack of Innovation: Overall, the main innovation arguably lies in introducing the parameter ρ on top of the DiffC[1] framework to control the perception-distortion trade-off. While it successfully achieves this trade-off, the core concept may not be considered highly novel, as it does not address the fundamental limitations of this class of methods.
2. High Computational Complexity and Slow Inference: Similar to DiffC[1], this method inherently suffers from significant encoding and decoding latency. This delay increases further at higher bitrates. Compared to some existing, more efficient image compression methods[2,3,4], the advantages of this approach are less pronounced.
3. Engineering Burden of RCC/PFR: While RCC (PFR) is information-theoretically elegant, it requires shared randomness and careful index coding. The paper provides the one-shot vs. asymptotic rate bounds but does not deeply quantify the runtime or bpp overhead resulting from practical PFR implementation aspects like truncation or batching on large images.
[1] Vonderfecht J, Liu F. Lossy Compression with Pretrained Diffusion Models[C]//The Thirteenth International Conference on Learning Representations.
[2]Jia Z, Li J, Li B, et al. Generative latent coding for ultra-low bitrate image compression[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024: 26088-26098.
[3] A. Ke, X. Zhang, T. Chen, M. Lu, C. Zhou, J. Gu, and Z. Ma, “Ultra Lowrate Image Compression with Semantic Residual Coding and Compression-aware Diffusion,” in Proc. Int. Conf. Mach. Learn. (ICML), 2025.
[4] Zhang, Tianyu, et al. "StableCodec: Taming One-Step Diffusion for Extreme Image Compression." arXiv preprint arXiv:2506.21977 (2025).

### Questions
1. On the performance of generative models: In your article, I see that both Flux and SDXL demonstrate better performance than SD 2.1. Why would stronger generative models lead to worse performance metrics?
2. Regarding the issue of latency: I'm very curious about potential feasible directions for optimization. Otherwise, it's difficult to assess whether this type of method can be applied in practice.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work proposes a training-free framework for traversing the rate-distortion-perception (RDP) tradeoff using pre-trained diffusion models. The authors largely build on DiffC - a work that proposed to transmit the (diffusion) noisy sample at some timestep $t$, $x_t$, via reverse-channel-coding efficiently, and then showed that it's better on the decoder side to continue to the reconstruction $x_0$ via an ODE (DiffC-F) instead of via ancestral sampling (the SDE, termed DiffC-A).
The key idea in the current paper is to modify the decoder side of DiffC-F by introducing a "score-scaled probability flow ODE (PF-ODE)", 
which is a combination of the original DiffC-F and another method which gives the MMSE solution.
Specifically, the PF-ODE includes a parameter $\rho$ which balances between perception (closer to DiffC) and distortion (closer to the MMSE solution), thereby enabling traversal of the distortion-perception tradeoff for a given bitrate. Compression rate is controlled on the encoder side, as in DiffC, by selecting which timestep is transmitted, yielding full traversal of the RDP surface.  

The authors provide theoretical results establishing DPR optimality for the scalar Gaussian case of their proposed PF-ODE, though parts of this analysis build on directions already explored in related literature. 
Experimentally, they test their method in pixel space on CIFAR-10, and on latent space with SD2.1 and Flux on Kodak and DIV2K demonstrating traversal, against traditional codecs and relatively recent neural methods.

### Strengths
- The paper motivation of traversing the RDP tradeoff[1] is clear and, in my opinion, important.
- This paper continues the recent trend (DiffC[2], PSC[3], DDCM[4]) of using a single, pre-trained model to support multiple bitrates in a flexible manner - which is also very important in this reviewer's opinion. Other methods such as HiFiC that need differently trained models for every new bitrate make neural compression less practical in the long run for edge devices.
- The idea of basically combining 2 solutions, one adapted for distortion and one for perception, has not been done recently in neural compression, which is good. *That said, the novelty in the context of RDP is limited, as mentioned in the weaknesses section.*
- The paper gives theoretical guaranties for the full algorithm's RDP optimality in the scalar Gaussian case (Thm 4), and for multivariate sources under an AWGN channel (Thm 3). Theoretical guarantees, although simplified, of algorithms in such a practical field as compression are good and paper should strive more towards that. However, *the novelty of the proofs is again limited, as mentioned in the weaknesses section.*
- Relatively detailed proofs are provided for all claims in the paper. 
- The provided demo helps with visualization.
- The paper was generally well-written, with intuition into the core idea.

[1] Yochai Blau and Tomer Michaeli. "Rethinking lossy compression: The rate-distortion-perception
tradeoff". ICML 2019.

[2] Lucas Theis, Tim Salimans, Matthew D. Hoffman, and Fabian Mentzer. "Lossy compression with gaussian diffusion". ArXiv preprint, 2022.

[3] Noam Elata, Tomer Michaeli, and Michael Elad. "PSC: Posterior sampling-based compression". TMLR 2025.

[4] Guy Ohayon, Hila Manor, Tomer Michaeli, and Michael Elad. "Compressed Image Generation with Denoising Diffusion Codebook Models." ICML 2025.

### Weaknesses
- The key concept - traversing the RDP curve by using a combined solution of the MMSE (distortion-optimal) and perfect-perception reconstructions - is the main idea in [5]. That work formalized the interpolation mechanism and proved its optimality. While here the combination is of the score (and not the estimator), this manuscript does not acknowledge them at all. Specifically, as the optimality in this paper is concerned with the scalar Gaussian case - are their results (and therefore the optimality) not transferable to your case by a linear transformation? In any case, this should be explicitly discussed to accurately position the actual contribution of this paper (which is more technical in that regard).
- LPIPS is used throughout the paper as a “perception” metric, but LPIPS is a feature-space distortion measure. This is well-documented in the literature, including the original PD tradeoff paper [6] (it's before LPIPS' time, so it appears as MSE between VGG features in Secs. 5-6), and in the RPD paper [1] (See e.g. Sec. 4.2, they even cite LPIPS there directly). Therefore, Figs 3,4,5 and in the App 7,10,11 all show a **distortion-vs-distortion** curve, and **not the RDP curve** as is claimed. The only FID evidence is for CIFAR-10 in Figs 2+8, where the resolution is too low to visually assess realism. Specifically, Fig 8 - the only actual RDP curve - does not clearly demonstrate the canonical tradeoff shape (e.g., at high BPPs the same FID is achieved for both high and low $\rho$ values). This, in total, amounts to a misrepresentation of the tradeoff. 
- Proposition 2 (Eq. 8) appears redundant and insufficiently attributed. The tradeoff in this setting, and even in more general cases, was already derived in [5] (e.g., Sec. 3.3; see their Eq. 9). This should be cited accordingly and properly discussed. Does plugging in your assumptions on the inputs into the general case aligns with the results? As written, it looks like re-deriving a special case without acknowledging the general theory.
- The writing occasionally feels overclaiming. For example, the statement "PF-ODE yields the best perception with relatively low distortion" is somewhat misleading. What is *relatively low*? The algorithm of Theis et al. in the general case is not optimal. Specifically, in the scalar Gaussian case, they calculate the distortion for DiffC-F as $2(1-\sqrt{1-\sigma^2}$ (their eq11), which they did not recognize to be optimal (and you did). However, in the multivariate case and the more general case this is not optimal. This distinction on the conditions for each optimality are not clear throughout the paper, and imply more than what is actually proven. Please rephrase to avoid implying global MSE optimality and to clearly separate cases.
- Several important baselines are missing. Beyond not discussing [5], the experiments omit recent diffusion-based compression works such as [4,7]. In particular, I think [4] is especially theoretically relevant, as it is similar to DiffC but positions itself as predetemined-rate alternative. Including these baselines is crucial for a fair performance evaluation. 
- High-resolution images for higher bitrates should be included in the appendix in large figures to allow careful visual inspection. For example, in Fig. 6 the entire right-most column looks identical at the available zoom, even though theoretically it shouldn't. 
- Samples figures (e.g. 6, 12, 13) should include PSNR values per image, to allow for evaluation of distortion.
- Flux visual samples are only reported for a single BPP of 0.41. This doesn't really allow examining the **R** in the RDP tradeoff. Moreover, the provided zoom shows visually identical results across different $\rho$ values, making the DP traversal unconvincing for Flux

Overall, the paper would benefit from a more measured positioning of its novelty and a clearer acknowledgment of how it extends, rather than restates, previous results. It is important for me to mention that under the current phrasing, I would give the paper **a score of 2**, with the main reason being the missing discussion of [5]. Still, I am currently giving the paper a score of 4 under the assumption that this is easily fixable during the rebuttal phase, leaving the rest of the weaknesses as the reason for the score of 4. 

Minor:
- The proof appendices are sometimes difficult to follow. For example, in Appendix A the discretization proof and the proof of Lemma 1 are not well separated. Please split the appendix into more subsections to improve readability. Also, A.4 introduces results without clear cross-references in the main text, leaving some proofs "hanging"
- For pixel-space experiments, I suggest ImageNet would be more appropriate than CIFAR. Several works (e.g., [3,4]) have already used ImageNet, and this would allow for more meaningful visual comparisons
- Please provide in the appendix full tables with numerical results for the metrics per BPP to allow for better reproducibility.
- Generally speaking, theoretical guarantees are limited to very special cases: Gaussian data, and scalar Gaussian data for full RDP optimality. The authors acknowledge this as out of scope, but it still limits the generality of the claims.
- The authors cite Tweedie's formula for "Herbert E. Robbins. An Empirical Bayes Approach to Statistics, 1992". As far as I'm aware, the proceedings are from 1956 (Proceedings of the Third Berkeley Symposium on Mathematical Statistics and Probability).

[5] Freirich, Dror, Tomer Michaeli, and Ron Meir. "A theory of the distortion-perception tradeoff in Wasserstein space." NeurIPS 2021.

[6] Blau, Yochai, and Tomer Michaeli. "The perception-distortion tradeoff." CVPR 2018.

[7] Jinpei Guo, Yifei Ji, Zheng Chen, Kai Liu, Min Liu, Wang Rao, Wenbo Li, Yong Guo, and Yulun Zhang. "OSCAR: One-Step Diffusion Codec Across Multiple Bit-rates." ArXiv preprint 2025.

### Questions
- Lemma 1: The authors seem to claim (L200) that the reconstruction is the MMSE itself, however the proof suggests that the mean of $Z_0$ converges to the MMSE. Can the authors clarify?
- Theorem 3 - this seems to imply a different $\rho$ parameter is required for each pixel (or PC direction?). Is this interpretation correct?
- Related - for the achievability proof - in Theis et al. work eq 13,14,15 (which are similar here) have different variance per dimension, which does not seem the case for the proof as stated here. However, the decomposition still seems similar. Don't you still need the orthogonality constraint (as in Theis' Thm 3) in the achievability proof?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper considers the rate-distortion-perception tradeoff using pre-trained diffusion-based models. It provides some results for Gaussian case and implements some experiments on a few datasets to explore the RDP tradeoff.

### Strengths
The paper studies an interesting problem of rate-distortion-perception tradeoff.

### Weaknesses
I believe that the paper has several major technical issues, and its contributions are not sufficiently compelling when compared to prior work such as Blau & Michaeli, Salehkalaibar et al., and Zhang et al.

1) The theoretical results lack sufficient rigor. For example, in the proof of Theorem 4, a reverse channel coding encoder is paired with a score-based decoder, where the central challenge is to establish a one-shot result for this encoder-decoder combination. However, the proof simply assumes that such a one-shot result holds, without justification. This assumption is non-trivial, and its validity is unclear for this specific encoder-decoder pair. Consequently, the subsequent simplification to the Gaussian case also lacks foundation in the absence of a rigorous one-shot argument.

2) Proposition 2 is not technically sound. Even at a fixed rate, the distortion-perception tradeoff does not admit the simple closed-form expression claimed in Eq. (8) (see Qian et al. for reference). Moreover, Proposition 2 entirely omits the compression rate, which contradicts the paper’s stated goal of analyzing the rate-distortion-perception (RDP) tradeoff. A result that ignores rate is not meaningful in this context.

3) The paper does not provide compelling theoretical or experimental insights. In particular, the experimental section only plots RDP curves without offering deeper analysis or qualitative insight into reconstruction characteristics, perceptual quality, or practical behavior. There is no clear experimental contribution beyond reproducing expected tradeoff plots.

4) The proofs in the appendix are difficult to follow and, based on the issues noted above, appear to contain technical gaps. The overall presentation of the theoretical arguments would benefit from significant revision for correctness and clarity.

5) The title of the paper claims a “training-free” method. However, while the encoder may be pretrained, the decoder still requires training. Therefore, the “training-free” characterization is misleading and should be reconsidered.

### Questions
what are the main contributions of the paper when diffusion-based models are used?

### Soundness
2

### Presentation
2

### Contribution
2

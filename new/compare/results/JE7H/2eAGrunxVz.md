---
job_id: 4dc61cf7-570e-4d10-9305-23f2a1c14012
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: 2eAGrunxVz.pdf
paper: Spherical Watermark: Encryption-Free, Lossless Watermarking for Diffusion Models
main_score_norm: 0.8
desk_reject: false
---
# Desk Rejection Assessment:
## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The work is squarely about watermarking for diffusion-based generative models, with theoretical and empirical analysis of mappings from bits to Gaussian latents, which fits ICLR’s scope on generative models, representation learning, and robustness.

## Minimum Quality
Pass ✅.  
All required sections (Abstract, Introduction, Related Work, Method, Experiments, Discussion/Limitations, Conclusion) are present and reasonably complete. The method is technically non‑trivial, proofs are provided (main ideas in the paper, details in the appendix), and the experimental section is extensive with appropriate baselines and metrics. I see no glaring methodological error or misuse of evaluation protocol.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I did not find any hidden prompts, instructions to reviewers, or other attempts to manipulate an automated review process.

---

# Expected Review Outcome:

## Summary

The paper proposes **Spherical Watermark**, a lossless, encryption‑free watermarking framework for diffusion models that embeds user‑specific bitstrings directly into the initial Gaussian noise. The method uses a binary embedding matrix to mix repeated watermark bits with random padding, then a spherical mapping that projects the resulting bit vector onto the unit sphere, applies an orthogonal rotation, and scales by a chi‑square–distributed radius to approximate standard multivariate Gaussian noise. The authors prove that the resulting distribution matches a spherical 3‑design and converges to a standard normal in the limit, and empirically show that the watermarked latents/images are statistically indistinguishable from clean ones while preserving high watermark extraction accuracy and robustness under a wide range of attacks.

## Strengths

1. **Conceptually clean, model‑agnostic design with strong theoretical underpinnings.**  
   The method decomposes into three invertible modules: binary embedding, spherical mapping, and diffusion integration (Section 3.2, Eq. (9)–(13), Figure 1). Theoretical analysis in Section 3.3 (Theorem 3.1, Theorem 3.2, Lemmas 3.3–3.4) carefully tracks the distribution through each transformation and shows that the final latent is an approximate standard Gaussian: the binary transform yields 3‑wise independent Bernoulli bits, the normalized vector forms a spherical 3‑design, and orthogonal rotation plus chi‑square scaling recovers Gaussianity. This is a much clearer and more principled story than many prior heuristics for “Gaussian‑like” watermarking.

2. **Losslessness is not just claimed but argued as a central robustness mechanism.**  
   The paper goes beyond saying “lossless is good”: Appendix E gives a detector‑driven optimization analysis where the adversary’s ability to craft WEvade‑style attacks is tied to the KL divergence between watermarked and clean distributions (Eq. (69)–(77)). This connection between distributional indistinguishability and adversarial robustness is, to my knowledge, largely absent from previous diffusion watermark papers and is a meaningful conceptual contribution.

3. **Strong, diverse empirical evaluation with well‑chosen baselines.**  
   Experiments (Section 4) are extensive: two SD backbones (v1.5, v2.1), two prompt datasets, a wide suite of baselines (traditional: DwtDct, DwtDctSvd, RivaGAN; latent: Tree‑Ring, Gaussian Shading, PRC Watermark), and thorough metrics (FID, IS, classifiers for detectability, ACC/TPR across post‑processing and adversarial attacks, computational costs, and multiple ablations).  
   * Table 1 shows that only PRC Watermark and the proposed method have FID essentially identical to the original generator, while Gaussian Shading and Tree‑Ring noticeably degrade FID.  
   * Table 2 and Figure 5 demonstrate that under both “clean” and post‑processing perturbations, the proposed method maintains ACC and TPR ≳95% and is much more robust than lossy baselines under adversarial attacks.  
   * Figure 4’s log‑scale timing comparison makes the computational advantage obvious: extraction is ~4 orders of magnitude faster than PRC Watermark and far faster than Gaussian Shading.

4. **Undetectability is validated both theoretically and empirically.**  
   The use of spherical 3‑designs and Stein’s method (Lemma 3.3) gives a nontrivial argument for indistinguishability up to third‑order moments. Figure 2 (and the expanded Figure 11 in the appendix) is particularly convincing: latent‑level and image‑level classifiers utterly fail on the proposed method and PRC (test accuracy ≈50%), while they reach ~100% accuracy for Tree‑Ring and ~97–100% for Gaussian Shading. This directly supports the claimed advantage of the proposed construction in hiding statistical artifacts in both latent and image domains.

5. **Clear benefit over the main lossless competitor (PRC Watermark).**  
   The comparison with PRC Watermark is unusually careful. Beyond FID and detectability parity, the paper highlights three concrete advantages:
   * **Robustness vs distortions:** Figure 5 and Table 2 show consistently higher ACC/TPR under stronger post‑processing and adversarial attacks; Figure 6(a) and Figure 16 show stable extraction even at high watermark capacities where PRC fails.  
   * **Computational efficiency:** Figure 4 quantifies orders‑of‑magnitude speedups, especially in extraction, by avoiding belief‑propagation decoding.  
   * **Parameter robustness:** Table 3 / 15–17 and Figure 6(d), 14 show that a wide range of sparsity and repetition settings keep undetectability and robustness near‑optimal, in contrast to PRC’s delicate code‑design trade‑offs.

6. **The rotation analysis vs Gaussian Shading is mathematically insightful.**  
   Appendix D compares the AWGN‑channel performance of Gaussian Shading’s coordinate‑wise |z|\*s embedding (Eq. (55)–(57)) with the proposed rotated embedding (Eq. (60)–(64)), showing that for equal energy constraints the orthogonal rotation maximizes pairwise symbol distance and asymptotically yields larger per‑bit extraction probability. This is a sharp, concrete explanation for the empirical robustness gains, not just a qualitative claim.

7. **Careful ablations that test the necessity of each module.**  
   Figure 6(b–c) and Figure 15 present ablations where spherical mapping is replaced by Gaussian Shading or binary embedding is removed. Without binary embedding, latent‑level classifiers easily detect watermarks; without spherical mapping, robustness against brightness and other distortions drops substantially. These figures effectively validate that both modules contribute essentially different aspects: independence/entropy vs robustness.

8. **Good generalization across architectures and tasks.**  
   Appendix F.1 applies the method to SD v3, FLUX.1‑DEV, a pixel‑space diffusion (G‑Diffusion), and a flow‑based model (Glow), with Figure 8 and Table 6–7 showing high extraction accuracy and good perceptual quality. This supports the claim that the approach is not tied to a specific SD variant, but to the general “Gaussian prior + invertible mapping” pattern.

## Weaknesses

1. **Security/indistinguishability claims are stronger in wording than in formal justification.**  
   In Section 3.1, Equations (2)–(3) state *computational indistinguishability* against any probabilistic polynomial‑time adversary with negligible advantage. However, Section 3.3 and Appendix C only show matching low‑order moments and asymptotic convergence to Gaussian marginals via spherical 3‑designs and Stein’s method (Lemma 3.3), not a cryptographic reduction from distinguishing z\_w to breaking some hardness assumption. There is no argument that a PPT adversary cannot exploit, say, subtle higher‑order dependencies in finite dimension. The empirical classifier tests (Figure 2) are valuable but limited to MLP/ResNet‑18. To be precise, the paper should substantially tone down the “computationally indistinguishable” language or provide a concrete hardness‑based argument.

2. **Reliance on high‑dimensional asymptotics is not quantified for realistic latent dimensions.**  
   The normal approximation in Lemma 3.3 yields a Wasserstein distance bound of order O(l\_x^{-1/2}), but the constants are swept into big‑O notation and not instantiated for the actual l\_x=16,384 used in the experiments. Given the dependence on the embedding matrix’s dependency degree D and moments of Haar entries C\_{ij}, a more explicit finite‑sample bound or at least numerical estimates would help justify that any residual deviation is practically undetectable. Without that, the theoretical guarantee is somewhat qualitative.

3. **The threat model and robustness scope are narrower than the rhetoric suggests.**  
   The paper emphasizes robustness under “common attacks” and WEvade, but the security model is effectively: (i) watermark key (T,C) is secret, (ii) attacker can apply standard image perturbations and WEvade crafted on a surrogate classifier. There is no serious discussion of an adaptive attacker who learns an approximate inversion or estimates C statistically from many watermarked samples; given T and C are fixed across all users, an adversary with oracle access to many images could in principle try to infer the spherical codebook or detect deviations in higher‑order statistics. Section 5 briefly notes that “extremely strong inversion‑breaking attacks” can compromise recovery, but the implications for security (e.g., collection of many marked images from the same API) are not explored.

4. **Assumptions on inversion quality are strong and not fully stress‑tested.**  
   The method fundamentally depends on accurate inversion of the diffusion process and VAE (Eq. (11)–(13)), which is known to be noisy and model‑/prompt‑dependent. While Table 4 and Table 5 nicely examine different ODE solvers and timestep schedules, these are fairly mild variations. More adversarially tuned or mis‑configured inversion (e.g., different guidance scales, wrong sigma schedule, editing‑driven inversion) could significantly distort z\_T and the radial r, increasing rounding errors in Eq. (13). Figures 17 and related ablations look at synthetic Gaussian noise on latents, but not genuine “inversion gone wrong” scenarios beyond those in Appendix F.2. The limitations section acknowledges inversion‑breaking attacks but the main text slightly underplays how central they are to traceability.

5. **No explicit capacity–robustness tradeoff analysis for the binary embedding design.**  
   The binary embedding matrix T is controlled by sparsity s, repetition N, and padding length l\_r (Eq. (5)–(6), Algorithm 1). Qualitatively, the paper notes that larger s amplifies error propagation and smaller N reduces redundancy; Tables 3, 15–17 provide some sample TPR values under a few attacks. However, there is no more systematic analysis or rule‑of‑thumb for choosing (N,s,l\_r) given target capacity and noise level. For instance, the security parameter ρ introduced in Section 3.1 vanishes from the concrete design choices. Practitioners looking to deploy this at different scales are left to guess how close to the limits they can go without collapse.

6. **Some cryptographic comparisons are somewhat hand‑wavy or incomplete.**  
   The critique of Gaussian Shading and PRC in Section 2 and 4.2 is partially qualitative: “nontrivial latency”, “careful tuning”, “irreducible error floor”. While the empirical timing plot (Figure 4) and the capacity failures of PRC in Figure 6(a) are good, there is no similar quantitative characterization of, e.g., Spherical Watermark’s key storage and secrecy implications: the signature (T,C) is reused across all images and users, so unlike per‑image key schemes there is a global secret whose compromise could break traceability. That tradeoff is not really discussed.

7. **The “lossless vs lossy” dichotomy is oversimplified relative to the space of methods.**  
   Appendix E and the main text frame non‑Gaussian schemes as fundamentally vulnerable because they induce a nonzero KL divergence between watermarked and clean distributions. While true in the strict sense, many practical systems might settle for “very small but nonzero” KL, or aim for distributional shifts only in subspaces that are difficult to detect. The paper does not compare against such “approximate lossless” designs (e.g., constrained Tree‑Ring variants or more modern latent modulation methods) and somewhat overgeneralizes from a few lossy baselines. This weakens the claim that losslessness is *the* crucial ingredient rather than a particularly strong but perhaps unnecessary condition.

8. **Notation and exposition could be tightened in several technical parts.**  
   While overall readable, some math passages are unnecessarily convoluted or slightly sloppy:
   * In Section 3.3, Theorem 3.2’s statement conflates properties of a random vector z^(2) with the “finite set” on the sphere; the discrete support size is huge (2^{l\_x}), but the notation X={z^(2)} is never made precise (support vs draws).  
   * Lemma 3.1 in Appendix C appears to restate Theorem 3.1 with slightly inconsistent equation numbering (Eq. 11 vs Eq. 9 of the main text), which is confusing.  
   * In Eq. (41) / (23) of Appendix C, ⟨q\_j,e⟩\_{F2} is not clearly defined in the main text; readers not familiar with binary inner products must infer this.  
   These are minor but detract from the clarity needed for a work that leans heavily on formal properties.

9. **Image‑level perceptual analysis could be more nuanced.**  
   Figure 3 and Figure 7 showcase qualitative examples where differences between methods are quite subtle, and Table 9–11 report PSNR/SSIM/IS. However, these metrics are known to be imperfect for generative outputs, and no user study or stronger perceptual metric (LPIPS, CLIP‑based measures, etc.) is used. Since a key selling point is “distribution‑preserving” watermarking, it would be helpful to convincingly show that even careful human scrutiny or perceptual metrics cannot distinguish watermarked vs clean images, especially compared against the most competitive latent schemes.

10. **Higher‑order dependencies and multibit correlation are not empirically probed.**  
    The theory guarantees matching up to third‑order moments, but a determined adversary might exploit higher‑order correlations among coordinates or between z\_T and downstream features. The only empirical detectability tests (Figure 2, 11, 12) use relatively shallow discriminators and do not attempt, for example, kernel two‑sample tests, higher‑order cumulants, or larger transformers over latents. While such tests are not standard today, they would substantiate the claim that the distributional differences are negligible beyond low‑order moments.

## Potentially Missing Related Work

The following papers appear directly relevant to latent‑diffusion watermarking or Gaussian‑noise‑based schemes and are not cited or discussed:

1. **Wu, Lin, Tan, “Spread Spectrum Image Watermarking Through Latent Diffusion Model” (2025).**  
   Proposes spread‑spectrum watermarking in the latent space of diffusion models, focusing on robustness to regeneration. This is conceptually close to modifying Gaussian noise and should be compared against in Section 2 (Related Works) and experimentally, at least qualitatively, in Section 4, particularly in terms of robustness vs. undetectability.

2. **Li, Zhang, Qu, “Shallow Diffuse: Robust and Invisible Watermarking through Low‑Dimensional Subspaces in Diffusion Models” (2024).**  
   Uses low‑dimensional latent subspaces to embed invisible, robust watermarks. This is highly relevant to the claimed advantages in Section 1 and 4.2 around robustness and undetectability, and should be cited and contrasted, especially with respect to capacity and distribution preservation.

3. **Lei, Gai, Yu, “Secure and Efficient Watermarking for Latent Diffusion Models in Model Distribution Scenarios” (2025).**  
   Addresses watermarking in scenarios where diffusion models are distributed to multiple users and emphasizes security and efficiency. Given this paper’s discussion of per‑image key management vs a fixed signature (Section 2–3.2), it should be integrated into the security/efficiency comparison and related‑work narrative.

4. **Hu, Zhang, Al‑Dossari, “Robust Watermarking for Diffusion Models Using Error‑Correcting Codes and Post‑Quantum Key Encapsulation” (2026).**  
   Extends error‑correcting‑code‑based watermarking (similar to PRC) with post‑quantum cryptography. Since Spherical Watermark positions itself partly as an alternative to heavy cryptographic constructs, this paper should be cited near the discussion of PRC Watermark (Section 2, 4.2) and the tradeoffs between cryptographic strength and computational burden.

5. **Tang, Zhang, Lai, “Fixed Neural Network Image Steganography Based on Secure Diffusion Models” (2025).**  
   While framed as steganography rather than watermarking, it uses secure diffusion models to hide information without altering the distribution, closely related in spirit to this paper’s lossless mapping. It would be useful to mention in Section 2 when situating the work among diffusion‑based hiding schemes.

6. **Hur, Kang, Seo, “Latent Diffusion Models for Image Watermarking: A Review of Recent Trends and Future Directions” (2025).**  
   A survey of diffusion‑based watermarking. Including it in Related Works would help contextualize Spherical Watermark within the broader landscape and ensure coverage of the most recent trends.

7. **Li, Zhang, Qu, “Watermarking via Gaussian Noise Modulation in Diffusion Models” (2025).**  
   Appears particularly close conceptually, since it uses Gaussian noise modulation as a plug‑and‑play watermarking strategy. It should be directly compared in Section 2 and possibly Section 4.2, especially around how Spherical Watermark’s spherical design and orthogonal rotation relate to or improve upon their Gaussian modulation scheme.

8. **Zhang, Li, Qu, “Robust Watermarking for Diffusion Model Generated Images” (2025).**  
   Proposes robust watermarking for diffusion‑generated images, likely in latent space. This seems like another strong baseline candidate and should be cited and conceptually compared in Related Work and, if feasible, in the robustness experiments.

Incorporating and contrasting against these works would strengthen the positioning and novelty claims.

## Questions

1. **Clarification on the “computational indistinguishability” claim.**  
   How do you justify Equations (2)–(3) beyond the spherical 3‑design and CLT‑style marginal arguments? Are you willing to weaken this to “indistinguishable up to third‑order moments and empirically for the considered classifiers,” or do you envision a more formal hardness‑based argument (e.g., that distinguishing z\_w from Gaussian violates some assumption about Haar random rotations)?

2. **Security under multi‑image observation and adaptive attackers.**  
   The paper mainly considers WEvade attacks using a trained classifier but not an attacker who has access to many watermarked images (for potentially different users) and can attempt to estimate C or detect non‑Gaussian structure. Could you discuss how security degrades in such a scenario and whether you can bound information leakage about C or T from many samples?

3. **Guidelines for parameter selection (N, s, l\_r, l\_m).**  
   Could you provide more explicit guidance on how to choose these parameters given target watermark length and anticipated perturbation strength? For example, is there a simple analytical heuristic relating N and s to an effective error‑correction rate or equivalent Hamming distance on the code?

4. **Inversion robustness beyond the tested solvers.**  
   How sensitive is extraction when inversion is deliberately mis‑matched, e.g., using a different guidance scale, an intentionally wrong sigma schedule, or a different sampler from the one used during generation? Can you provide empirical results or justification on how far from the “correct” inversion one can go before recovery breaks down?

5. **Potential to support variable per‑user signatures.**  
   Currently, (T,C) is fixed per model developer and only m and r change per user/image. Would it be feasible and beneficial to also randomize or derive user‑specific rotations from a master key (e.g., via PRFs) to reduce risk in case of partial key compromise, while maintaining losslessness?

6. **Higher‑order detectability tests.**  
   Have you tried nonparametric two‑sample tests (e.g., MMD with RBF kernels over latents) or higher‑capacity discriminators (transformer‑based) to detect differences between z and z\_w? If so, what were the results, and if not, could you speculate based on your theoretical approximations?

Author responses that clarify these points and, in particular, address the mismatch between the cryptographic indistinguishability language and the current theory would increase my confidence.

## Flag For Ethics Review

No ethics review needed.

## Details Of Ethics Concerns

N/A.

## Soundness Rating

3: good.  
The core construction is mathematically sensible and supported by solid, if asymptotic, analysis and extensive experiments. Some security claims are a bit stronger than what is rigorously proven, but the empirical evidence for indistinguishability and robustness is convincing.

## Presentation Rating

3: good.  
Overall the paper is clearly written with helpful figures (e.g., Figure 1 pipeline, Figure 2/11 classifier performance, Figure 4 timing, Figure 5 robustness) and tables (Table 1 FID, Table 2 tracing accuracy). A few notational inconsistencies and dense proofs in the appendix could be cleaned up, but they do not block understanding.

## Contribution Rating

3: good.  
The combination of spherical 3‑design–based mapping, orthogonal rotation analysis, detector‑driven robustness argument, and thorough empirical comparison against both lossy and lossless baselines is a meaningful step forward in diffusion watermarking. It is not a completely new problem setting, but the execution and analysis are strong.

## Overall Rating

8: Accept, good paper (poster).  
The paper presents a well‑designed, theoretically motivated, and empirically validated watermarking scheme that improves on both robustness and practicality compared to prior lossless approaches, while keeping undetectability and image quality at essentially optimal levels. Some cryptographic claims should be toned down and the threat model expanded, but these do not undermine the core technical contribution.

## Reviewer Confidence

4: confident.  
I am familiar with diffusion watermarking, Gaussian latent mappings, and robustness evaluation, and I carefully checked the main derivations and experimental methodology. Some cryptographic subtleties are outside my deepest expertise, hence not a 5.
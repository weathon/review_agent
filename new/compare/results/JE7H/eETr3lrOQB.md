---
job_id: 71639172-62fb-4ece-b530-217ee7f64113
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: eETr3lrOQB.pdf
paper: VQ-Transplant: Efficient VQ-Module Integration for Pre-trained Visual Tokenizers
main_score_norm: 0.6
desk_reject: false
---
# Desk Rejection Assessment:
## Paper Length  
Pass ✅.

## Topic Compatibility  
Pass ✅.  
The paper focuses on vector-quantized visual tokenizers, representation learning, and efficient training for generative models, which fits squarely within ICLR’s core topics (representation learning, generative models, optimization, and vision applications).

## Minimum Quality  
Pass ✅.  
The paper is complete (Abstract, Introduction, Related Work, Method/preliminaries, Experiments, Conclusion). It is written in clear English, proposes a concrete framework (VQ‑Transplant) and a new VQ variant (MMD‑VQ), and backs its claims with substantial experiments and quantitative/qualitative results. I do not see fatal methodological or theoretical flaws that would justify desk rejection.

## Prompt Injection and Hidden Manipulation Detection  
Pass ✅.  
I see no signs of hidden prompts, steganographic text, or instructions targeting automated reviewers.

---

# Expected Review Outcome:

## Summary

The paper introduces **VQ‑Transplant**, a two‑stage framework for efficiently replacing the vector quantization (VQ) module inside a pre‑trained discrete visual tokenizer while keeping the encoder and decoder largely frozen. Stage I trains a new VQ module on frozen encoder features, and Stage II performs a short decoder adaptation phase to mitigate distributional mismatch between the new codebook and the original decoder. Within this framework, the authors further propose **MMD‑VQ**, which uses Maximum Mean Discrepancy (MMD) to align feature and codebook distributions, and show that transplanting MMD‑based VQ into a strong pre‑trained VAR tokenizer yields reconstruction quality close to or better than full VAR training at around 5% of its computational cost.

## Strengths

1. **Clear and practically meaningful problem formulation (decoupling VQ from tokenizer training).**  
   The paper articulates a very practical bottleneck: exploring new VQ algorithms currently requires full adversarial training of large encoder–decoder tokenizers on datasets like ImageNet or OpenImages. The idea of *surgically swapping* the VQ module in a frozen tokenizer and doing only a short decoder adaptation (Section 4.1) is simple yet addresses a real pain point for researchers with limited compute. Figure 1 effectively clarifies the workflow: Block 1 shows the original encoder–VQ–decoder pipeline, Block 2 shows Stage I (VQ module substitution with frozen encoder/decoder), and Block 3 shows Stage II (decoder adaptation with the new VQ frozen). This diagram makes the overall methodology very easy to follow.

2. **Solid empirical evidence that VQ‑Transplant works and is compute‑efficient.**  
   The experimental section is extensive and largely well‑designed. Using the pre‑trained VAR tokenizer as the main host model is sensible, since it is a strong industry‑level baseline with adversarial training.  
   - Table 1 quantifies the training cost comparison against several tokenizers (Llama‑GEN, ImageFolder, VAR, UniTok). While the “Speedup” column is somewhat apples‑to‑oranges, it still conveys the magnitude: VAR requires 16×A100 for 60 hours on OpenImages vs VQ‑Transplant needing 2×A100 for 22 hours on ImageNet‑1k.  
   - Table 3 (multi‑scale VAR case) and Table 7 (fixed‑scale case) show that after decoder adaptation, **MMD‑VAR** achieves r‑FID 0.81 vs 0.92 for original VAR (both on ImageNet‑1k) with full codebook utilization and slightly improved PSNR/SSIM. Table 4 and Figure 3 show consistent r‑FID improvement over adaptation epochs, confirming that Stage II is not a one‑off hack but a stable training phase that translates reduced quantization error from Stage I into perceptual gains.  
   - Table 6 directly contrasts VQ‑Transplant with limited “from‑scratch” MMD‑VAR training under similar wall‑clock budgets: the transplanted models have much better quantization error and r‑FID despite *less* total training time, which supports the claim that leveraging a good pre‑trained encoder–decoder is far more efficient than re‑training everything.

3. **Decoder adaptation insight is well‑supported and important.**  
   A key conceptual contribution is the explicit observation that minimizing quantization error alone is not sufficient once the decoder’s prior is tuned to a particular latent distribution.  
   - In Table 3 (Substitution rows), MMD‑VAR and Wasserstein‑VAR have smaller quantization error than the original VAR tokenizer (e.g., \( \mathcal{E}=0.234 \) vs 0.283 at K=8192) but substantially *worse* r‑FID (1.49 vs 0.92), highlighting a decoder–latent space mismatch.  
   - After running Stage II (Adaptation rows in Table 3), r‑FID drops sharply (e.g., MMD‑VAR from 1.49 to 0.81 at K=8192) and even surpasses original VAR.  
   - Figure 2 offers a convincing qualitative view: the top row (after Stage I only) shows noticeable blur and missing high‑frequency details; the bottom row (after adaptation) restores sharp textures and edges, qualitatively matching or beating the original VAR reconstructions.  
   This is a useful takeaway for the community: VQ replacements in pre‑trained models *must* be coupled with at least some decoder adaptation; otherwise, low distortion at the code level does not automatically yield good images.

4. **MMD‑VQ is logically motivated and mathematically well‑defined.**  
   The extension from Wasserstein‑based distribution matching to MMD‑based matching is coherent and clearly presented. Equation (5) rigorously defines the empirical MMD objective between features \( X=\{z_i\} \) and codebook vectors \( Y=\{e_j\} \), with a multi‑Gaussian kernel to provide a characteristic RKHS. The paper correctly notes that MMD with a characteristic kernel satisfies \( \mathcal{D}^2_\text{MMD}=0 \) iff the distributions match.  
   Section B in the appendix carefully revisits the approximate Wasserstein‑VQ formulation: starting from the general Wasserstein distance (Equation (6)) then specializing it to Gaussian distributions (Equations (7)–(8)). The authors correctly point out that this leads to matching only first and second moments, which becomes suboptimal for highly non‑Gaussian or multimodal features. The synthetic experiments in Tables 12 and 13 and Figure 7 convincingly support this: as the bimodal separation parameter \( \zeta \) increases, Wasserstein‑VQ’s quantization error and utilization degrade substantially, while MMD‑VQ maintains significantly better error (e.g., at \( \zeta=4.0 \), 1.502 vs 1.240) and utilization (34.8% vs 75.6%). This is a nice, sharp illustration of the distribution‑matching argument.

5. **Cross‑dataset and cross‑tokenizer analysis shows decent robustness and scope.**  
   The main experiments focus on VAR, but the authors also:
   - Evaluate fixed‑scale VQ transplantation across VAR on ImageNet‑1k (Table 7), showing that the same observations (distribution‑aligned VQs reduce quantization error, decoder adaptation is crucial for recon quality) hold across codebook sizes up to 65,536.  
   - Test cross‑dataset generalization by transplanting onto FFHQ, CelebA‑HQ, and LSUN‑Churches (Tables 8–10 and Figures 4–6). On FFHQ, Wasserstein‑VQ reaches r‑FID=1.21 and MMD‑VQ=1.37 with very high PSNR/SSIM, outperforming several baselines trained from scratch (RQVAE, VQGAN variants, MQVAE, VQ‑WAE, VQGAN‑LC). Qualitatively, Figures 4–6 show that both Wasserstein‑VQ and MMD‑VQ reconstructions preserve identity and fine details across faces and churches.  
   - Explore compatibility with a *continuous* tokenizer, LDM‑16 (Table 16). While the transplanted discretizers underperform VAR‑based ones in r‑FID/r‑IS, the authors diagnose and discuss this: LDM’s decoder is trained on continuous latents and has smaller capacity, making adaptation harder. This adds nuance to the claims and avoids overselling universality of the approach.

6. **Ablations on training strategies and adaptation length are thorough.**  
   The paper does not just present a single training recipe.  
   - Table 5 and Figure 3 examine 5/10/15/20 adaptation epochs, showing r‑FID continues to marginally improve (e.g., MMD‑VAR at K=8192 improves from 0.81 at 5 epochs to 0.74 at 20 epochs) with relatively stable behaviour, giving practitioners a sense of the trade‑off.  
   - Appendix C and Table 14 compare decoder‑only adaptation vs joint optimization of encoder, VQ and decoder. Joint optimization gives slight extra gains (e.g., MMD‑VAR, K=8192 drops r‑FID from 0.81 to 0.79) at the cost of additional training time (Table 15). This is a nice exploration of design space and shows the framework is not brittle.

7. **Overall clarity and structure are good.**  
   The exposition is generally clear and well‑structured. Preliminaries (Section 3) correctly formalize the discrete tokenizer and VQGAN objective (Equation (2)). The VQ‑Transplant stages are described in a compact but understandable way, and notation is largely consistent. Figures 8–9 in the appendix provide ample qualitative reconstructions for multiple VQ methods, which helps readers visually sanity‑check the reported metrics.

## Weaknesses

1. **Conceptual novelty of VQ‑Transplant itself is limited and somewhat under‑theorized.**  
   The core idea of freezing a pre‑trained encoder, plugging in a new quantizer, and fine‑tuning the decoder with reconstruction + perceptual + adversarial loss is intuitively straightforward, essentially a form of partial fine‑tuning or layer replacement. The paper does not provide a deeper analysis of *why* training only the VQ module + decoder is sufficient beyond empirical evidence. For example, there is no formal argument relating the encoder’s representation geometry to the attainable quantization error or how the decoder adaptation finds a new optimum under the frozen encoder. As a result, the main algorithmic novelty lies more in packaging a reasonable engineering strategy than in fundamentally new theory or architecture.

2. **Evaluation is almost entirely restricted to reconstruction metrics, with no downstream generative or VLM experiments.**  
   While the paper is about tokenizers, their primary practical use is as front‑ends for generative models (autoregressive, diffusion, or VLMs). All reported metrics (r‑FID, r‑IS, LPIPS, PSNR, SSIM) are reconstruction‑focused. There is no evaluation of how VQ‑Transplant affects:  
   - sample generation quality when plugged into an autoregressive generator (e.g., training a small AR model on the transplanted tokens and comparing FID vs VAR tokens), or  
   - performance in text‑image models or multimodal transformers.  
   This is a big gap given claims about “democratizing quantization research” for practical tokenizers. It remains unclear whether improvements in reconstruction r‑FID from 0.92 to 0.81 (Table 3) or 0.86 to 0.74 (Table 5) translate into any meaningful gain or even parity for generative modeling, especially since many prior works report that reconstruction metrics and downstream generative quality can be misaligned.

3. **Comparison of computational cost in Table 1 is not fully fair or transparent.**  
   Table 1 compares training hours and speedups across Llama‑GEN, ImageFolder, VAR, UniTok, and VQ‑Transplant on *different* datasets and hardware configurations (e.g., VAR on OpenImages with 16×A100 vs VQ‑Transplant on ImageNet‑1k with 2×A100). The “Speedup” column is unclear: for example, it lists 21.8× speedup for VAR in the VAR row (which is confusing) and “–” for VQ‑Transplant. There is no normalization by FLOPs, total images seen, or GPUs×hours; nor is there discussion of the fact that VQ‑Transplant benefits from starting from an already pre‑trained, expensive VAR model, whose cost is not credited. While the qualitative claim that transplanting is much cheaper than training VAR from scratch is valid, the current quantitative framing risks overstating actual end‑to‑end efficiency gains if one includes the cost of training the original host tokenizer.

4. **MMD‑VQ improvements over Wasserstein‑VQ on *real* data are modest, raising questions about added complexity.**  
   On ImageNet and the real datasets, gains of MMD‑VQ compared to Wasserstein‑VQ are often minor or even slightly worse.  
   - In Table 3 after adaptation, MMD‑VAR and Wasserstein‑VAR at K=8192 have identical PSNR and LPIPS (24.37 vs 24.40, 0.104 vs 0.104) and very close r‑FID/r‑IS (0.81 vs 0.83 and 201.0 vs 198.8).  
   - In Table 7 (fixed‑scale ImageNet‑1k), MMD‑VQ and Wasserstein‑VQ are again nearly tied (differences in PSNR/SSIM/LPIPS within 0.02 at most, and r‑FID 0.86 vs 0.92 at K=65536).  
   - On FFHQ and CelebA‑HQ (Tables 8–9), Wasserstein‑VQ sometimes has lower r‑FID, sometimes MMD‑VQ wins slightly; neither is clearly dominant.  
   The non‑Gaussian synthetic experiments (Tables 12–13, Figure 7) nicely justify MMD‑VQ *in principle*, but the paper does not show any *real* regime where the feature distribution is actually sufficiently non‑Gaussian for MMD‑VQ to matter. Given that MMD is more expensive than a simple Gaussian moment‑matching term, the practical trade‑off is unclear. At minimum, there should be a more explicit discussion that on current tokenizers, benefits are marginal and MMD is a forward‑looking choice for potential future settings.

5. **Decoding and loss formulations omit some details that matter for reproducibility and understanding.**  
   - Equation (4) gives the decoder loss \(\mathcal{L}_\text{Decoder}\), but the precise form of \(\mathcal{L}_\text{Per}\) and \(\mathcal{L}_\text{GAN}\) is only vaguely referenced as “we follow Tian et al. (2024), Chen et al. (2025a), and Li et al. (2025) and employ an identical frozen DINO‑S discriminator… with DiffAug, consistency regularization, and LeCAM regularization”. For a central training phase, this is rather shorthand; for example, the hinge loss form, consistency regularization coefficient, and the definition of real vs reconstructed scores in the DINO discriminator are not spelled out. One has to rely heavily on external code.  
   - In Equation (3), \(\mathcal{L}_\text{VQ}(\phi)=\|\text{sg}(z_e)-z_q(\phi)\|_2^2 + \gamma \mathcal{L}_\text{unique}(\mathcal{Q}_\phi^\text{new})\), the usual commitment loss term \(\beta\|z_e - \text{sg}(z_q)\|^2\) (present in Equation (2)) is intentionally dropped, but the paper does not justify this design choice; dropping that term modifies the gradient flow to the encoder (here frozen) vs the codebook. Given the focus on quantization quality, it would be useful to explain whether including the commitment term hurt performance or was simply unnecessary for frozen encoders.  
   - Similarly, in Appendix C, the joint loss \(\mathcal{L}_\text{Joint}\) appears to merge VQ loss and decoder loss but is written in a slightly compressed form where indexing and sg operations could be made clearer. Precisely specifying which terms backpropagate to which modules would enhance clarity.

6. **Scope is largely limited to reconstructing *images* at 256×256 with a single backbone tokenizer.**  
   Despite some cross‑dataset variation (FFHQ, CelebA‑HQ, LSUN‑Churches), all experiments are on single images at 256×256. Given VAR and related tokenizers also target higher resolutions and video domains, it would be valuable to see at least one experiment at a higher resolution, or on a different backbone (e.g., OmniTokenizer or a BEiT‑style tokenizer). The LDM‑16 experiments in Table 16 are a step, but the authors themselves acknowledge that adaptation is weaker there due to architectural differences. As it stands, strong claims like “VQ‑Transplant democratizes quantization research” may be overextended relative to the evidence, which focuses almost exclusively on the VAR architecture.

7. **Missing and under‑discussed related work in visual tokenizers and efficient training.**  
   While the related work section covers major VQ/VQGAN and VAR‑style models, several directly linked recent works on tokenizer design and scalable training are absent. Specifically:
   - Works on *semantic‑rich tokenizers* built for masked image modeling and generative pre‑training, such as BEiT v2’s vector‑quantized visual tokenizer, are not discussed. That line of work explicitly considers how to design VQ codebooks for large‑scale vision pre‑training.  
   - Recent methods that use DINO or other self‑supervised backbones to build tokenizers (e.g., leveraging hierarchical representations or efficient quantization during pre‑training) are omitted, even though this paper heavily relies on a DINO‑based discriminator and pre‑trained VAR.  
   - Recent papers focusing on *scalable training of visual tokenizers* and efficient quantization procedures (e.g., new quantization rules like index backpropagation) are closely related to the goal of making VQ research less compute‑intensive but are not cited or compared.  
   Including and contrasting with these works would better ground the claimed gap that “current approaches treat VQ and encoder‑decoder as monolithic” and show more precisely where VQ‑Transplant fits within the evolving landscape of plug‑and‑play tokenizers versus end‑to‑end training.

8. **Terminology and table notation occasionally inconsistent or confusing.**  
   There are a few minor but noticeable issues in tables and text that can hinder readability:  
   - In Table 5, columns labeled “I(↓)” and “D(↑)” appear to correspond to quantization error and utilization, but the notation differs from earlier tables (\(\mathcal{E}\), \(\mathcal{U}\)) and is never defined in the caption.  
   - In Table 6, columns “l(↓)” and “l4(↑)” are likely typos for \(\mathcal{E}\) and \(\mathcal{U}\) (or similar). This should be corrected.  
   - Some hyperparameters are mismatched: Appendix A states AdamW uses \(\beta_1=0.9\) and “\(\beta_1=0.95\)” (clearly a typo for \(\beta_2\)).  
   These do not undermine the core scientific claims but do suggest that the paper would benefit from a careful pass for consistency.

## Potentially Missing Related Work

These works appear directly relevant and are not cited in the submission:

1. **Peng et al., “BEiT v2: Masked Image Modeling with Vector‑Quantized Visual Tokenizers”, 2022.**  
   BEiT v2 introduces a vector‑quantized semantic tokenizer tailored for masked image modeling and large‑scale pre‑training. It is directly relevant since it also focuses on designing VQ modules that can be integrated into powerful vision backbones. It should be discussed in Section 2 (“Visual Tokenizer for Generative Models”) and possibly compared in the main ImageNet reconstruction table (Table 2) or at least qualitatively positioned versus VQ‑Transplant’s use of pre‑trained tokenizers.

2. **Jia et al., “DINO‑Tok: Adapting DINO for Visual Tokenizers”, 2025.**  
   DINO‑Tok adapts DINO representations for visual tokenization, using a pre‑trained self‑supervised backbone to define a tokenizer. This shares the idea of leveraging an existing strong model to build tokenizers efficiently. It should be mentioned in the related work on visual tokenizers and adversarial training (Section 2), and contrasted with VQ‑Transplant’s approach of transplanting quantizers into pre‑trained tokenizers rather than building tokenizers directly from DINO.

3. **Shi et al., “Scalable Image Tokenization with Index Backpropagation Quantization”, 2025.**  
   This paper proposes an efficient quantization mechanism (index backpropagation) for scalable training of visual tokenizers. It directly targets the same overarching problem of making tokenizer training less expensive and more scalable. It should be added to Section 2 in the discussion of vector quantization methods and, ideally, included as part of the baseline set in Table 2 or at least discussed relative to the computational savings reported in Table 1.

4. **Yao et al., “Towards Scalable Pre‑training of Visual Tokenizers for Generation”, 2025.**  
   This work addresses large‑scale pre‑training of tokenizers for generative models, exploring data and compute scaling trends. It is highly relevant context for the claim that adversarially trained tokenizers like VAR are costly and that more scalable solutions are needed. It should be cited in Section 1 and Section 2 and compared when discussing VQ‑Transplant’s 95% training‑cost reduction claim.

5. **Li et al., “XQ‑GAN: An Open‑source Image Tokenization Framework for Autoregressive Generation”, 2024.**  
   XQ‑GAN provides an open‑source tokenizer framework that supports multiple quantization techniques, enabling researchers to swap quantization components within a unified system. Conceptually, this resembles VQ‑Transplant’s goal of plug‑and‑play VQ integration, and should be cited in Section 2 and discussed more explicitly when motivating the need for a transplant framework. A short comparison in Section 5 (perhaps near Table 2) noting differences in whether encoders/decoders are retrained vs frozen would clarify the distinction.

## Questions

1. **Downstream generative modeling impact.**  
   Can the authors provide at least a preliminary experiment where a small autoregressive generator (or diffusion model) is trained on the transplanted tokens (e.g., MMD‑VAR vs original VAR) to compare sample quality (FID/IS) and training stability? Even a reduced‑scale model on ImageNet‑64 or a subset of ImageNet‑1k would significantly strengthen the argument that improved reconstruction metrics translate into better or comparable generative performance.

2. **Why omit the commitment loss term in Equation (3)?**  
   In classical VQ‑VAE and VQGAN objectives, the commitment term \(\beta\|z_e - \text{sg}(z_q)\|^2\) is important to prevent encoder outputs from drifting. In your Stage I formulation \(\mathcal{L}_\text{VQ}(\phi)=\|\text{sg}(z_e)-z_q(\phi)\|_2^2 + \gamma \mathcal{L}_\text{unique}\), you remove it. Was this a deliberate design decision based on experiments, or simply because the encoder is frozen? Have you tried including a small commitment term and, if so, how did it affect quantization error and reconstruction?

3. **Practical cost of MMD vs Wasserstein losses.**  
   For realistic configurations (e.g., codebook size 65,536, feature dimension 32, mini‑batch size 32×16×16 tokens), what is the relative training‑time overhead of computing the MMD loss in Equation (5) versus the Gaussian Wasserstein‑based loss? Do you use any approximations (e.g., random feature MMD) beyond the multi‑Gaussian kernel stated? A brief complexity analysis or wall‑clock comparison of Stage I training with Wasserstein‑VQ vs MMD‑VQ would clarify the real trade‑off.

4. **Distributional diagnostics on real encoder features.**  
   You convincingly show in Appendix B that MMD‑VQ shines on non‑Gaussian synthetic data. For real encoder features from the VAR tokenizer, have you computed any diagnostics (e.g., kurtosis, skewness, multimodality tests, or projections along principal components) that justify the claim that these are close to Gaussian, explaining why MMD‑VQ and Wasserstein‑VQ perform similarly? Such analysis would turn a speculative explanation into evidence and might guide where MMD‑VQ will give larger benefits.

5. **Fairness of the compute comparison with VAR.**  
   For Table 1 and the “95% cost reduction” statement, could you explicitly clarify:  
   - Whether the reported 22 hours for VQ‑Transplant includes both Stage I and Stage II, and on which dataset (ImageNet‑1k vs OpenImages)?  
   - How you amortize or account for the cost of pre‑training the original VAR tokenizer, since VQ‑Transplant leverages its encoder–decoder?  
   Providing a more apples‑to‑apples comparison (e.g., cost to go from a *given* pre‑trained VAR to a *new quantization scheme* vs cost to train an entirely new VAR tokenizer from scratch) would make the efficiency claim more precise.

6. **Extending to higher resolutions or video.**  
   Do you foresee any obstacles in applying VQ‑Transplant to higher‑resolution VAR variants (e.g., 512×512) or video tokenizers such as OmniTokenizer or MagViT‑style models? The current framework scales in principle, but decoder adaptation with adversarial training can be tricky at higher resolutions. Any insights or preliminary results here would help assess the broader applicability.

## Flag For Ethics Review

- No ethics review needed.

## Details Of Ethics Concerns

N/A. The work uses standard image datasets (ImageNet‑1k, FFHQ, CelebA‑HQ, LSUN‑Churches) with widely used protocols, and focuses on tokenizer training. I do not see particular issues around sensitive populations, privacy, safety, or misuse that go beyond usual concerns for generative models.

## Soundness Rating

3: good.  
The methodology is conceptually straightforward, training objectives (Equations (2)–(5)) are standard and consistent, and the empirical evaluation is extensive. There are some gaps (no downstream generative experiments, partial details on GAN/perceptual losses), but nothing fundamentally undermines the main claims.

## Presentation Rating

3: good.  
The paper is generally well‑written and structured, with helpful figures (especially Figures 1–3 and 8–9) and detailed tables (2–3, 5–7, 8–10, 16). Minor notation inconsistencies and typos in tables and hyperparameter descriptions should be cleaned up but do not seriously impede understanding.

## Contribution Rating

3: good.  
The VQ‑Transplant framework is a practically useful way to decouple VQ research from full tokenizer training; MMD‑VQ is a theoretically sound extension of Wasserstein‑VQ. Conceptual novelty is moderate and MMD‑VQ’s empirical edge over Wasserstein‑VQ on real data is small, but the combination of a clear problem, efficient solution, and thorough empirical study makes this a worthwhile contribution for ICLR.

## Overall Rating

6: marginally above the acceptance threshold. But would not mind if paper is rejected.  
The paper offers a sensible and well‑validated framework that can make VQ research more accessible by reusing strong pre‑trained tokenizers and optimizing only a new VQ module plus a short decoder adaptation. The idea is not particularly deep, and there are missing pieces (no downstream generative results, somewhat optimistic compute comparison, modest MMD‑VQ gains over Wasserstein‑VQ), but the empirical work is solid and the insights about decoder–quantizer alignment are valuable. On balance, I lean to a positive recommendation, though I see reasonable room for debate.

## Reviewer Confidence

4: confident.  
I am familiar with VQ/VQGAN/VAR literature, have carefully checked the main equations and tables, and feel confident in my assessment, though I would welcome author clarifications on the points raised above.
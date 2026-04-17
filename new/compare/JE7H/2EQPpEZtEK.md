---
job_id: f8893250-55fe-4c5e-a9b4-97fa28673033
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: 2EQPpEZtEK.pdf
paper: DISTAR: Diffusion Over a Scalable Token Autoregressive Representation for Speech Generation
main_score_norm: 0.6
desk_reject: false
---
# Desk Rejection Assessment:
## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper addresses zero-shot text-to-speech using discrete RVQ token modeling with an autoregressive LM and masked diffusion, which falls squarely within ICLR’s core areas of generative modeling, representation learning for audio, and large-scale sequence modeling.

## Minimum Quality
Pass ✅.  
The paper is in English and has all key sections (Abstract, Introduction, Related Work, Method, Experiments, Results including ablations, and Conclusion). The method is technically coherent, experiments are substantial on standard benchmarks with multiple baselines, and there are no obvious fatal methodological or theoretical errors.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I do not see any hidden prompts, steganographic text, or attempts to manipulate automated reviewing systems in the provided content.

---

# Expected Review Outcome:

## Summary

The paper proposes DiSTAR, a zero-shot text-to-speech framework that operates entirely in an RVQ (residual vector quantization) discrete code space and couples a causal autoregressive (AR) Transformer with a masked diffusion Transformer. The AR component advances generation at the patch level by producing a long-context sketch, while the masked diffusion model infills RVQ tokens within each patch in parallel, jointly modeling temporal and RVQ-layer dependencies. Experiments on LibriSpeech-PC and SeedTTS show that DiSTAR achieves strong or state-of-the-art objective and subjective performance compared to both continuous-latent and discrete-token TTS baselines, while enabling controllable bitrate and compute via RVQ-layer pruning and decoding heuristics.

## Strengths

1. **Well-motivated hybrid AR + discrete masked diffusion design in RVQ space.**  
   The core architectural idea, described in Sections 3.1–3.3, is conceptually clean: factorization over patches (Equation (1)) handled by an AR Transformer, with intra-patch RVQ token dependencies handled via a masked diffusion LM objective (Equation (2)). This directly attacks two known pain points: (i) long-range instability and exposure bias of purely AR RVQ LMs, and (ii) optimization fragility of continuous next-patch diffusion. The paper provides a coherent narrative explaining why patch-level AR plus within-patch diffusion is a natural fit for multi-layer RVQ.

2. **Strong empirical performance with careful baseline selection and head-to-head comparisons.**  
   Table 1 compares DiSTAR-base/medium against strong zero-shot systems including IndexTTS (discrete AR), E2TTS, F5TTS, and DiTAR (continuous next-patch), on LibriSpeech-PC and SeedTTS. DiSTAR-medium achieves the best or second-best WER and UTMOS across both benchmarks (e.g., LibriSpeech WER 1.66% vs 2.39% for DiTAR and 2.02% for F5TTS; SeedTTS WER 1.32%, best overall). It also matches or approaches the best SIM. Importantly, DiSTAR-base reaches these numbers with only 0.15B parameters, smaller than many baselines (e.g., DiTAR 0.6B, IndexTTS 0.5B). This suggests the architecture provides a genuine modeling advantage per parameter.

3. **Subjective results support the claims of improved robustness and naturalness.**  
   Table 2 presents human CMOS/SMOS on SeedTTS. DiSTAR attains the highest SMOS (3.31) and the highest CMOS (+0.22) among listed systems, surpassing FireRedTTS, CosyVoice 2, E2TTS, and F5TTS. This is in line with the objective UTMOS scores in Table 1 and supports the claim that the discrete RVQ + masked diffusion combination yields perceptually higher-quality speech, not just better ASR metrics.

4. **Clear and useful analysis of controllability and efficiency via RVQ-layer pruning.**  
   Figure 2 provides a concise and informative view of the trade-off between the number of retained RVQ layers, speaker similarity (SPK), and WER. The plot shows that WER is relatively flat beyond ~6 layers while SPK improves steadily, supporting the interpretation that deeper RVQ layers encode mainly acoustic detail rather than linguistic content. Together with the stochastic layer truncation mechanism in Section 3.4, this convincingly demonstrates that DiSTAR can provide variable bitrate / compute control at inference without retraining.

5. **Non-trivial ablations on decoding strategies and patch size.**  
   Table 3 benchmarks different decoding heuristics (temperature shaping and greedy vs sampling), showing a sensible diversity–determinism trade-off: greedy decoding with shaped temperatures yields the lowest WER (1.91%) at slightly reduced SIM compared to sampling. Table 6 in Appendix D explores patch size, clearly illustrating that both too-small (P=2) and too-large (P=8 vs P=4) patches hurt performance, which supports the central modeling decision around patching. These ablations materially increase confidence that the design choices are not arbitrary.

6. **Good utilization and adaptation of recent discrete diffusion theory and practice.**  
   The masked diffusion objective in Equation (2) is reasonably aligned with recent discrete diffusion / masked-LM connections (LLaDA-style) and uses a cosine mask schedule. The training and inference masking schedules, classifier-free guidance variants (Equations (3)–(7)), and re-masking scheme show that the authors are aware of technical subtleties in discrete diffusion and adapt them carefully to RVQ TTS.

7. **Clarity of overall system diagram and modular decomposition.**  
   Figure 1 effectively communicates the full pipeline: RVQ codes aggregated into patch embeddings, fed into a causal AR backbone, with the masked diffusion module operating on the next patch under text + historical code conditioning. The right-hand schematic showing RVQ-layer embedding and pooling clarifies how multiple RVQ layers per frame are processed. This figure ties together many textual details that would otherwise be easy to misinterpret.

## Weaknesses

1. **Limited empirical scope and dataset diversity for a system claiming zero-shot robustness.**  
   Despite strong results on LibriSpeech-PC and SeedTTS, the evaluation is confined to English speech from two benchmarks and a single training corpus (Emilia-English, ~50k hours). There is no evaluation on more challenging or diverse test conditions such as noisy in-the-wild recordings, code-switching, emotional or conversational speech, or non-English languages. Given that DiSTAR is proposed as a general zero-shot framework “for scalable speech generation”, the current evidence is somewhat narrow. This matters because one of the main motivations for going fully discrete-RVQ and patchwise AR+diffusion is supposed robustness under domain shift; without stress tests beyond LibriSpeech/SeedTTS, it is hard to assess generalization.

2. **Incomplete disentangling of the benefits of discreteness vs architecture vs training recipe.**  
   The main comparison to DiTAR (Table 1) pits DiSTAR against a continuous latent next-patch system, but several confounding factors remain: DiTAR uses a different codec (continuous), larger model size (0.6B), and different NFE (10 vs DiSTAR’s 24). The paper claims “comparable or lower computational cost”, but FLOPs, wall-clock latency, or GPU-seconds per second of audio are not reported. Without matched NFE budgets or explicit complexity analysis, it is unclear whether DiSTAR’s better WER comes from the discrete RVQ design, the AR+MD coupling, or simply more refinement steps. A more controlled ablation, e.g., varying NFE for both DiTAR and DiSTAR at equal param count and plotting quality vs cost, would significantly strengthen the argument that DiSTAR is Pareto-superior rather than just slower and more refined.

3. **Mathematical formulation of the training objective is under-specified and somewhat inconsistent.**  
   Equation (2) is presented as minimizing an expectation over \(t\), \(\hat{\mathbf{C}}_0^{(k)}\), and \(\hat{\mathbf{C}}_t^{(k)}\) with a weight \(1/t\) to “recover an upper bound on the sequence negative log-likelihood,” but the derivation of this bound is not provided or even sketched. Crucially:
   - The sampling distribution of \(t\) is only described later (Section 3.3) as \(t \sim \mathcal{U}(0,1]\) with \(\lambda(t)=\cos((1-t)\pi/2)\); plugging this into the 1/t weighting is non-trivial, and it is unclear how it yields a correct bound.  
   - The forward process masks each token independently with probability \(\lambda(t)\), but the resulting transition kernel is not written explicitly, so it is difficult to verify the connection to the likelihood bound results in the cited works *(Ou et al., 2024; Shi et al., 2024; Shih et al., 2022)*.  
   - The notation for \(\hat{\mathbf{C}}_t^{(k)}\) vs \(\dot{\mathbf{C}}_{\rho_n}^{(k)}\) and \(\widehat{\mathbf{C}}_t^{(k)}\) on Page 5 introduces multiple symbols for essentially the same object in training and inference, with no precise mapping between \(\lambda(t)\), \(\rho_n\), and the discrete decoding steps \(n = 0,\ldots,N-1\).  
   This makes it hard to judge how closely inference matches the training corruption process, and whether any formal likelihood bound still holds. A more concrete algebraic link between Equation (2), the schedule \(\lambda\), and the iterative decoding mask budgets \(\rho_n\) would reduce this ambiguity.

4. **Patchification and aggregator design are insufficiently ablated and partially ad hoc.**  
   While Table 6 in Appendix D varies patch size \(P\), several important degrees of freedom in the aggregator remain mostly untested:
   - Stride \(S\) is fixed to \(S=P\) in most experiments, yet the method emphasizes the overlapped setting \(S<P\) (Pages 4–6). The claim that overlapping windows “smooth boundaries and provide more information” is plausible, but there is no quantitative evidence; all main results seem to be with stride 8 for patch size 8. If overlapped contexts are a key ingredient for mitigating exposure bias and boundary artifacts, a simple WER/SIM table comparing S=P vs S<P would be expected.
   - The mixing of RVQ layer embeddings via a learnable scalar per layer (Section 3.2) is appealing, but not evaluated against simpler alternatives such as concatenation followed by a shallow MLP, or fixed averaging. Without such comparison, it is difficult to tell whether the reported gains are specific to the DiSTAR factorization or depend crucially on this embedding trick.

5. **Decoder heuristics are complex and under-justified empirically.**  
   Section 3.4 introduces several non-trivial decoding heuristics: layer-wise and position-wise temperature shaping, hybrid sample/greedy schedules, CFG with rescaling and nested variants, repetition penalties, etc. Yet Table 3 provides only a very limited comparison between three configurations, without isolating the impact of each trick. For example, we are told that “tail-first bias” motivates position-wise temperature shaping, but there is no figure showing error concentration near the end of patches or how the bias changes before/after shaping. From a deployment and reproducibility perspective, this level of heuristic stacking without more granular ablation makes the method feel fragile and tuned, even if the final WER/SIM are strong.

6. **Limited analysis of failure modes and robustness beyond average scores.**  
   The evaluations focus primarily on global averages of WER, SIM, UTMOS, SMOS, and CMOS. We do not see any analysis of where DiSTAR fails: e.g., very long sequence synthesis, prompts with atypical prosody, or extremely short/noisy prompts. Given the stated goal of reducing brittleness under distribution shift, some qualitative or quantitative investigation (e.g., WER vs utterance length, speaker similarity vs prompt duration, or comparison on out-of-domain accents) would be valuable. The current evidence suffices to show average gains on standard benchmarks, but not robustness in the broader sense claimed in the abstract.

7. **Related work positioning around AR–diffusion hybrids and non-VQ TTS is not fully complete.**  
   The related work focuses on zero-shot TTS in discrete vs continuous spaces and on masked diffusion, but omits several works that are directly relevant to the conceptual framing:
   - Recent continuous-token AR TTS models that explicitly avoid VQ while retaining an LM flavor (see missing works section). Their existence suggests that some advantages claimed for discrete RVQ might also be achievable in continuous token spaces, which should be discussed.
   - Work on reinterpreting diffusion as autoregressive modeling in discrete domains (again, see missing works), which is conceptually close to DiSTAR’s AR-coupled diffusion setup.  
   Without discussing these, the paper may give the impression that the proposed factorization is more unique than it actually is.

8. **Subjective evaluation setup is under-detailed.**  
   Table 2 reports SMOS and CMOS with error bars but provides no information on the number of listeners, number of evaluated utterances per system, rating protocol (pairwise vs single-ended), or randomization. Given that improvements are relatively modest (e.g., DiSTAR’s CMOS 0.22±0.13 vs F5TTS 0.01±0.12), it is important to know whether effects are statistically significant or how listener fatigue and system order were controlled. Without this, it is hard to evaluate how strong the subjective preference is.

## Potentially Missing Related Work

1. **Meng, L., Zhou, L., Liu, S. (2025): “Autoregressive Speech Synthesis without Vector Quantization.”**  
   This work proposes MELLE, a continuous-valued token-based LM for TTS that bypasses vector quantization. It is directly relevant because it addresses some of the same motivations (stability and controllability of LM-style decoding) while avoiding discrete codebooks altogether. The authors should:  
   - Discuss it in Section 2.1 as a third option in addition to discrete VQ and continuous latents,  
   - Clarify how DiSTAR’s discrete-RVQ design compares against continuous-token AR in terms of robustness and controllability, and  
   - Possibly include it as a baseline or at least a qualitative comparison in the main experiments.

2. **Gao, Z., Shou, M. Z. (2025): “D-AR: Diffusion via Autoregressive Models.”**  
   D-AR recasts diffusion as standard next-token prediction on discrete tokens, effectively unifying diffusion and AR. Conceptually, this is close to DiSTAR’s idea of coupling an AR Transformer with a masked diffusion process in the discrete RVQ space. It should be cited and discussed in Section 2.2 and/or Section 3.1 when motivating the AR–diffusion interplay. A short comparison could clarify similarities and differences: DiSTAR uses an explicit AR backbone plus a separate masked diffusion head over RVQ tokens, whereas D-AR embeds diffusion steps into an AR view over discrete noise levels.

If the authors judge these works not directly comparable for some reason (e.g., non-zero-shot, different modality), they should still justify that explicitly.

## Questions

1. **Training objective and likelihood bound.**  
   Could the authors provide a more explicit derivation connecting Equation (2) to an upper bound on \(-\log p_{\theta}(\mathbf{C}|\mathbf{X})\)? In particular:  
   - What is the exact joint distribution of \(t\) and the mask pattern used in the expectation,  
   - How does the 1/t weighting arise, and  
   - Under what assumptions does this bound hold for the overlapping patch factorization in Equation (1)?

2. **Compute and latency comparison to DiTAR and other baselines.**  
   Can you report per-second-of-audio FLOPs or wall-clock latency (including codec) for DiSTAR-base/medium versus DiTAR and F5TTS at comparable quality? Since DiSTAR uses 24 NFE vs DiTAR’s 10 in Table 1, it would be helpful to see quality vs NFE curves for both systems and indicate where DiSTAR sits on that trade-off.

3. **Impact of stride and overlapping patches.**  
   Most text emphasizes the benefit of overlapping windows (\(S<P\)), but experiments appear to default to stride equal to patch size. Do you have results showing WER/SIM/UTMOS for \(S<P\) versus \(S=P\)? This would clarify how crucial the overlap is for boundary smoothness and exposure-bias mitigation.

4. **Decoder heuristic ablations.**  
   Among layer-wise temperature shaping, position-wise shaping, hybrid sample/greedy scheduling, repetition penalties, and CFG variants, which contribute most to the performance gains? It would be valuable to see a small ablation table showing WER/SIM when each heuristic is ablated (starting from a minimal baseline), to assess robustness and simplify possible future implementations.

5. **Behavior under long utterances and noisy prompts.**  
   Have you evaluated DiSTAR on very long passages (e.g., >30 seconds) or with short/noisy prompts? Any qualitative or quantitative observations on stability (speaker drift, prosodic collapse) compared with DiTAR or F5TTS would help validate the claimed robustness.

6. **Scope of RVQ-layer pruning in real-time or streaming scenarios.**  
   Figure 2 suggests a trade-off between SPK and compute via layer pruning. Could DiSTAR be adapted for streaming / low-latency synthesis by dynamically adjusting the number of layers per segment based on bandwidth or device constraints? If so, how would this interact with the AR and diffusion modules?

## Flag For Ethics Review

- Yes, Potentially harmful insights, methodologies and applications  

## Details Of Ethics Concerns

The system is a high-fidelity zero-shot TTS model capable of closely matching speaker timbre from short prompts, which raises standard risks around impersonation, social engineering, spoofing of voice biometrics, and non-consensual voice cloning. While Appendix G acknowledges these concerns and suggests mitigations (watermarking, consent-first deployment, access revocation), the paper does not elaborate on concrete mechanisms or empirical evaluation of watermark robustness. Given the increasing misuse potential of such systems, it is appropriate to flag this for ethics review to ensure that recommended safeguards are adequate and realistic for prospective deployment.

## Soundness Rating

3: good.  
The method is technically coherent and experimentally supported on two benchmarks with strong baselines and meaningful ablations. Some mathematical aspects (likelihood bound, alignment between training and inference schedules) and empirical gaps (compute comparison, broader robustness) remain under-specified, but there are no obvious fatal flaws.

## Presentation Rating

3: good.  
The paper is generally well written and structured, with clear diagrams (Figure 1) and reasonably explained algorithms. Some notation is overloaded or inconsistent (mask schedules, t vs n, different \(\hat{\mathbf{C}}\) symbols), and the exposition of the objective and decoding could be more rigorous, but overall the work is understandable.

## Contribution Rating

3: good.  
The combination of patch-level AR with masked diffusion in discrete RVQ code space for zero-shot TTS, together with RVQ-layer controllability and strong empirical performance, constitutes a solid contribution likely of interest to the ICLR community. It is not radically new conceptually, but it refines and substantiates a promising direction with convincing evidence.

## Overall Rating

6: Marginally above the acceptance threshold. But would not mind if paper is rejected.  
The paper presents a well-motivated and reasonably thorough study of a discrete RVQ AR+masked-diffusion TTS system that achieves strong empirical results and provides useful insights into RVQ-layer trade-offs and decoding strategies. Weaknesses mainly concern limited evaluation scope, underdeveloped mathematical connections, and somewhat heuristic-heavy decoding, but these do not undermine the main claims. With some clarification and broader experiments, this work would be a solid addition to ICLR.

## Reviewer Confidence

4: confident.  
I am familiar with neural codec LMs, diffusion-based TTS, and discrete diffusion literature, and I carefully checked the core equations and experimental setup. Some implementation details (exact training dynamics, code) are not fully verifiable from the paper alone, but the overall assessment is unlikely to change drastically.
---
job_id: 5a519400-719e-4892-9dc2-5e5a23a032ae
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: IdJakw2jta.pdf
paper: Towards Long-Form Spatio-Temporal Video Grounding
main_score_norm: 0.4
desk_reject: false
---
# Desk Rejection Assessment:
## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper proposes a new autoregressive transformer architecture with memory mechanisms for long-form spatio-temporal video grounding, squarely within representation learning and multimodal video understanding, which fits ICLR’s scope.

## Minimum Quality
Pass ✅.  
All essential sections (Abstract, Introduction, Related Work, Method, Experiments, Results, Conclusion) are present and written in English. The method is non-trivial, experiments are reasonably extensive, and no obvious fatal methodological or theoretical errors are apparent from the text.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I do not observe any instructions aimed at influencing automated reviewing, no hidden prompts, and no suspicious formatting indicative of manipulation.

---

# Expected Review Outcome:

## Summary

The paper studies *Long-Form Spatio-Temporal Video Grounding* (LF-STVG), where the goal is to temporally and spatially localize an object/event described by text in videos lasting 1–5 minutes, substantially longer than traditional STVG benchmarks. The authors propose ART-STVG, an autoregressive transformer that processes video frames sequentially, augmented with spatial and temporal memory banks and heuristic memory-selection mechanisms, plus a cascaded spatial→temporal decoder connection. They extend the HCSTVG-v2 validation set to longer durations (1–5 minutes) and show ART-STVG strongly outperforms existing STVG models on these extended benchmarks, while remaining competitive on the original short-form STVG setting.

## Strengths

1. **Clear problem motivation and relevance (LF-STVG)**  
   The paper addresses a real and important gap: existing STVG work is almost entirely on short clips (<1 minute), while many practical scenarios involve multi-minute or hour-long videos (Introduction, Pages 1–2). Pushing STVG into the long-form regime is scientifically and practically meaningful. The failure of existing methods as video length grows is convincingly illustrated in **Figure 2**, where performance curves of TubeDETR, STCAT, CG-STVG, TA-STVG drop rapidly with video length, whereas ART-STVG degrades more gracefully.

2. **Reasonable and coherent architectural idea (autoregressive + memory)**  
   Treating video as a stream and decoding per-frame outputs autoregressively is a natural way to avoid quadratic cost in the number of frames. The architecture in **Figure 3** clearly shows how the multimodal encoder, spatial memory bank, temporal memory bank, and cascaded decoders fit together. The method design is coherent: spatial memory captures instance-centric cues, temporal memory captures event-level information, and both are used via cross-attention in Equations (8)–(11).

3. **Memory selection mechanisms supported by visual evidence**  
   The paper goes beyond just “having memory” and proposes two simple selection strategies: (i) text-guided top‑$N_s$ spatial memory selection via similarity, and (ii) a TextTiling-inspired event segmentation over temporal memories. While heuristic, the effect is made plausible via qualitative visualizations:  
   - **Figure 5** shows attention maps for the spatial query with and without selective spatial memory; with memory, attention is much more focused on the described target, whereas without it, attention spills onto background persons.  
   - **Figure 6** illustrates temporal memories grouped into events via cosine similarity dips and shows start/end probability peaks aligned to the correct event. These figures substantively support the claim that memory selection helps disambiguate long videos.

4. **Strong empirical improvements on long-form benchmarks**  
   **Table 1** is the central quantitative result. Across LF-STVG-{1,2,3,4,5}min, ART-STVG consistently and substantially outperforms TubeDETR, STCAT, CG-STVG, and TA-STVG on m_tIoU, m_vIoU, and vIoU@0.3/0.5. For example, on LF-STVG-3min, ART-STVG improves m_tIoU from 13.9 (TA-STVG) to 23.0 and m_vIoU from 8.5 to 15.3, and vIoU@0.5 from essentially 0 to 9.5. The gains are even more striking at longer durations (e.g., LF-STVG-5min), supporting the central claim that the proposed design is particularly effective for long videos.

5. **Ablations reasonably support the design choices**  
   The ablations on LF-STVG-3min are quite informative:  
   - **Table 2** shows that naïvely using *all* temporal memories actually harms performance (m_tIoU 9.6 vs 16.7 without memory), while selective memory boosts to 23.0, strongly indicating that the selection heuristic matters.  
   - **Table 3** similarly shows incremental benefit of selective spatial memory over no memory and over all memory.  
   - **Table 4** compares the cascaded vs parallel decoder design, with small but consistent gains (e.g., m_tIoU 23.0 vs 21.5).  
   - **Table 5** explores $N_s$, finding a clear optimum around 32.  
   - **Table 6** shows that training on moderately longer videos (40s vs 20s) helps all models, but ART-STVG retains a large margin.  
   This experimental structure is a positive aspect: design elements are not just asserted but empirically dissected.

6. **Efficiency analysis with realistic memory footprint discussion**  
   **Table 8** in the supplementary material compares model size, inference time, and GPU memory usage across STCAT, CG-STVG, TA-STVG, and ART-STVG. While ART-STVG is slower (1.09s vs 0.47–0.71s for 64 frames), it uses much less memory (7.9G vs ~25G), which is a meaningful advantage for longer videos and resource-constrained settings. This is a relevant tradeoff analysis for a “long-form” paper.

7. **Qualitative results and failure analysis show some introspection**  
   **Figure 9** contrasts ART-STVG with the baseline without memory on long videos, showing ART-STVG maintains consistent target tubes while the baseline flips between persons. **Figure 8** provides failure modes (ambiguous event boundaries, distractor objects, very short events), which demonstrates the authors have examined limitations rather than just cherry-picking successes.

## Weaknesses

1. **Dataset construction and evaluation protocol are under-specified and potentially fragile**  
   The core empirical claim rests on extended HCSTVG-v2 validation videos (LF-STVG-1–5min), but the creation process is only very briefly described on Page 7: “based on original YouTube videos, not concatenated clips, and we manually review the extended videos to ensure their quality.” There are several unresolved issues:  
   - It is unclear how the original 20s annotated segments are embedded within the longer videos. Are they always contiguous prefixes extended further, or are 1–5 minute segments centered differently?  
   - How is it guaranteed that the original spatio-temporal annotation remains valid and *unique* in the extended clip, i.e., that the query does not also describe another person or another similar event appearing later? Manual review is mentioned but no criteria, inter-annotator agreement, or statistics are provided.  
   - The training set remains 20-second clips, and only the validation split is extended. This makes the “benchmark” slightly non-standard: models train on short clips and evaluate on synthetic long clips derived from the same source videos. There is no discussion of potential distribution shift between training and extended validation (e.g., extended backgrounds, more events per video) and how severe it is.  
   These issues matter because **Table 1** and much of the contribution hinge on these extended sets. Without a more rigorous dataset description and sanity checks, it is hard to interpret the magnitude of improvements or whether they generalize to *other* long-form data.

2. **Limited diversity of datasets and absence of external generalization tests**  
   All experiments, both long-form and short-form, are on HCSTVG-v2 (Page 7, Page 9). There is no evaluation on other established STVG datasets such as VidSTG or HCSTVG-v1, nor on any genuinely long-video dataset with independent annotations. This is a substantial limitation: the method is positioned as generally solving LF-STVG, but evidence is confined to a single dataset family and to extended versions created by the authors. Given that many STVG works report results on multiple datasets, the single-dataset evaluation weakens the case for broad impact.

3. **Memory mechanisms are heuristic and under-theorized**  
   The memory selection schemes in Sections 3.3–3.4 are plausible but somewhat ad-hoc:  
   - Spatial memory selection first inserts every spatial query into the memory bank without any expiration or compression and then takes top‑$N_s$ memories (based on similarity to text). There is no discussion of how memory size scales over very long videos or whether older, irrelevant memories (e.g., from different scenes) could crowd out more recent yet less text-similar ones.  
   - Temporal memory selection borrows the intuition of TextTiling, but uses cosine similarity between adjacent memories and a simplistic segmentation rule: “points with lower similarities are considered as event boundaries” (Page 7). Precise algorithmic details are missing (e.g., threshold choice, window size, handling of noisy similarities), and there is no complexity analysis regarding how many events/memory segments are maintained.  
   The lack of more formal analysis or even a clear pseudo-code makes it hard to assess robustness. For example, in **Figure 6**, the segmentation appears neat, but this could be cherry-picked; there is no quantitative measure of how well the temporal segmentation correlates with ground-truth event boundaries.

4. **Mathematical formulation and notation issues, especially around the encoder and losses**  
   Several equations are imprecise or opaque:  
   - **Equation (1)** for $f_i'$ is almost unreadable in its current form, with repeated underbraces and what appear to be typos (`f_{i_1}^a` etc., and repeated “textual feature $f^t$” labels). The indexing $i_1, \dots, i_{H \times W}$ is never defined, and the typesetting suggests there may have been copy-paste or LaTeX encoding issues. A clearer formulation would be  
     \[
       f_i' = [ \text{vec}(f_i^a), \, \text{vec}(f_i^m), \, f^t ],
     \]
     with explicit attention to dimensionality. Currently, it is ambiguous whether features are flattened, how they are concatenated, and in what axis.  
   - The self-attention encoder in **Equation (2)** operates on $f_i' + \mathcal{E}_{pos} + \mathcal{E}_{typ}$, but the shapes of these embeddings and how they are broadcast to the different modalities are not specified. In multimodal transformers, positional and type embeddings are usually per-token; here, positional structure in $H \times W$ grids and in text is non-trivial and should be clearer.  
   - The loss function is relegated to the supplementary, but **Equation (12)** shows that temporal localization uses KL divergence between predicted and ground-truth start/end “distributions” $\mathcal{H}_s, \mathcal{H}_e$. The main text never explains how these are normalized or constructed (e.g., Gaussian around boundary frames, uniform over event, etc.). Without this, it is hard to understand what the temporal head $h_i \in \mathbb{R}^2$ (Eq. (7)) is actually predicting beyond being “probabilities”. Precise definition of these distributions is essential for reproducibility.

5. **Autoregressive modeling and training procedure are insufficiently detailed**  
   The paper emphasizes autoregressive decoding (Section 3.2), but key aspects are under-specified:  
   - During training, is teacher forcing used? Are ground-truth boxes and timestamps fed into subsequent steps, or are predictions fed back? How are errors propagated across frames?  
   - The training is said to use a frame length $N_f = 64$ (Page 7), but the video lengths in validation reach up to 5 minutes with FPS 3.2 (≈960 frames). How exactly is the model trained to handle sequences longer than 64 frames? Sliding windows? Truncation? Curriculum? How many frames’ worth of memory does the model actually see in practice during training?  
   - There is no discussion of how the autoregressive state (memory banks) is initialized and reset across clips during training, nor details on batching strategy. These details are important in an autoregressive, memory-based design and affect both performance and reproducibility.

6. **Baseline and comparison fairness are somewhat underdeveloped**  
   - The “Baseline (ours)” in **Table 1** and **Table 7** is only explained in the supplementary (Figure 7) and is basically ART-STVG without memory. That is fine, but the main text (Page 8) gives only a vague description (“similar architecture ... but without memory and memory selection modules”). Given how central the memory claim is, more explicit description in the main paper would be helpful.  
   - It is not completely clear that all existing baselines are fairly adapted for long-form evaluation. For example, are TubeDETR, STCAT, CG-STVG, TA-STVG run on the full 1–5 minute videos at 3.2 FPS with their default hyperparameters and no truncation? Section 4.1 only says “all methods are trained exclusively on the HCSTVG-v2 training set (average video length 20 seconds) for fair comparison”. But some of these models may have internal maximum length assumptions or memory limits; we see huge performance collapse in **Table 1** for 3–5min (vIoU@0.5 ~0 for most baselines), raising the question whether they are simply failing numerically (e.g., due to memory) rather than semantically. This matters because part of the claimed advantage is computational tractability; explicit description of how baselines are run on long videos is needed.

7. **Scope of “long-form” is modest, and claims are a bit overstated**  
   Although the paper speaks about videos of “minutes or even hours” (Abstract, Introduction), the actual experiments only reach 5 minutes, with frames heavily downsampled to 3.2 FPS. 5 minutes at 3.2 FPS is ≈960 frames, which is indeed longer than classic STVG benchmarks, but still short compared to the hour-level video understanding literature cited in Section 2. Moreover, ART-STVG’s inference time scales linearly with frames, and **Table 8** already shows >1s for 64 frames; scaling to hour-long videos seems non-trivial even with smaller memory. The writing could be more cautious about generalizing to “hours” without empirical or analytic evidence.

8. **Related work on long-video grounding and STVG is incomplete**  
   The Related Work section focuses on standard STVG and long-video understanding in other tasks, but misses several directly relevant works on long video temporal grounding and advanced STVG models (see “Potentially Missing Related Work” below). In particular, there is existing work on long video temporal grounding (e.g., coarse-to-fine alignment for long videos) and very recent STVG variants that could overlap with or complement this work. This weak positioning slightly undermines the novelty narrative.

9. **Short-form STVG performance slightly lags state of the art**  
   **Table 7** shows ART-STVG is competitive but not state-of-the-art on short-form HCSTVG-v2: TA-STVG achieves 60.4/40.2 (m_tIoU/m_vIoU) vs 59.2/39.2 for ART-STVG. Given that ART-STVG consumes more computation per frame due to autoregressive processing (Table 8), the lack of clear gains in the standard setting somewhat tempers the impact; the benefit is essentially limited to longer videos, which rely on an extended benchmark under the authors’ control.

10. **No robustness or ablation on memory length and forgetting mechanisms**  
    The memory banks grow monotonically over time as queries are inserted (Section 3.3: “without removing any existing memories”). Beyond the $N_s$ selection in the spatial side and event-based cropping on the temporal side, there is no analysis of what happens for very long streams:  
    - Is there a cap on the total number of memories?  
    - How sensitive is performance to the density of memory insertion (every frame vs every $k$ frames)?  
    - Could a simple recency-based forgetting scheme perform similarly?  
    Such analysis would strengthen the argument that the proposed memory design is fundamentally necessary rather than just one reasonable instantiation.

Overall, the work is technically sound at a high level and presents strong empirical results on an extended benchmark, but several aspects of dataset construction, method specification, and evaluation breadth fall short of ICLR’s usual bar.

## Potentially Missing Related Work

1. **Liu et al., “Single-Frame Supervision for Spatio-Temporal Video Grounding”, 2025**  
   This work tackles STVG with weak supervision and may include longer temporal contexts or alternative training strategies relevant for scaling to long videos. It should be discussed in Section 2 as part of recent STVG models and compared in terms of how it handles temporal reasoning and supervision efficiency.

2. **Luo et al., “Spatial–Temporal Video Grounding with Cross-Modal Understanding and Enhancement”, 2025**  
   This is a transformer-based STVG method focusing on advanced cross-modal interactions. It is directly comparable to ART-STVG as another one-stage, transformer STVG model and should be cited in Related Work (Section 2) and ideally added as a baseline in **Table 7** for SF-STVG, or at least discussed qualitatively.

3. **Hou et al., “CONE: An Efficient COarse-to-fiNE Alignment Framework for Long Video Temporal Grounding”, 2023**  
   Although CONE focuses on temporal sentence grounding rather than full spatio-temporal tubes, it specifically addresses *long video temporal grounding* with efficient alignment. Its treatment of long temporal context and computational efficiency is directly relevant and should be cited in the “Long-term video understanding” paragraph of Section 2, with a short discussion contrasting CONE’s coarse-to-fine temporal strategy to ART-STVG’s sequential memory-based design.

4. **Zhang et al., “Temporal Sentence Grounding in Videos: A Survey and Future Directions”, 2023**  
   This survey summarizes advances and challenges in temporal sentence grounding, including for long videos. It could help situate the LF-STVG problem more clearly and should be at least briefly referenced in the Related Work section to acknowledge broader trends in long-form grounding.

(If any of these are actually in the authors’ reference list under different abbreviations or names, I may have missed them, but I did not see them cited in the main text.)

## Questions

1. **Details of temporal distribution modeling and loss**  
   Please give a precise definition, in the main paper, of how $\mathcal{H}_s, \mathcal{H}_e$ and their ground truth counterparts are constructed for the KL losses in Equation (12). Are these soft distributions over frame indices normalized to sum to 1, or per-frame Bernoulli probabilities? If the latter, why KL rather than, say, binary cross-entropy?

2. **Exact algorithm for temporal memory selection**  
   In Section 3.4 and **Figure 6**, could you provide pseudo-code or a more formal description: how are cosine similarities between adjacent memories smoothed or aggregated, what threshold or heuristic defines event boundaries, and how is the “event closest to current frame” determined? This would help assess complexity and reproducibility.

3. **Dataset extension protocol and quality checks**  
   Can you clarify:  
   - How exactly are the 1–5 minute segments selected from YouTube videos? Are they always extensions of the original 20s clip, or can they be taken from anywhere around it?  
   - Did you check for multiple occurrences of visually similar events corresponding to the same text, and how did you handle ambiguous cases?  
   - Could you provide statistics such as the average number of “events” per extended video, or the fraction of videos where the target event occupies less than 5% of total frames?

4. **Handling of sequences longer than 64 frames during training**  
   With $N_f = 64$ frames per training sample but evaluation up to ~960 frames, how do you ensure that the model learns to operate with long autoregressive memories? Do you use overlapping windows during training, or do you randomly sample 64-frame chunks across the full video?

5. **Baseline configuration on long videos**  
   Please detail how TubeDETR, STCAT, CG-STVG, and TA-STVG are executed on the 1–5 minute videos:  
   - Are they run on a single pass over the entire clip, or on windowed segments with some fusion?  
   - Do any of them fail due to GPU memory and require downsampling/truncation?  
   - Are their temporal maximum lengths or positional embeddings adapted?

6. **Memory growth and control**  
   Do you impose any maximum size on the spatial and temporal memory banks? If not, have you profiled memory growth for 5-minute videos and beyond? What happens in even longer settings (e.g., 10 or 20 minutes)?

Clear answers to these questions, plus any additional experiments on at least one external dataset or more rigorous dataset description, could substantially increase my confidence.

## Flag For Ethics Review

- No ethics review needed.

## Details Of Ethics Concerns

N/A.

## Soundness Rating

2: fair.  
The overall architecture and experiments appear broadly correct and convincing on the provided benchmarks, but important methodological details (dataset extension, temporal loss distributions, autoregressive training regime, precise memory selection algorithms) are under-specified, and baselines on long videos are not fully clarified.

## Presentation Rating

3: good.  
The high-level ideas, figures (especially Figures 1–6), and main results tables (1–7) are clear and informative, but some equations (notably Eq. (1)), training details, and dataset descriptions are imprecise or relegated to the supplement, which detracts from clarity.

## Contribution Rating

2: fair.  
The paper proposes a reasonable autoregressive + memory architecture for long-form STVG and shows significant gains on an extended benchmark, but the novelty is incremental relative to existing transformer and memory-based designs, evaluation is limited to one dataset family, and the long-video benchmark itself is not sufficiently documented.

## Overall Rating

4: Marginally below the acceptance threshold. But would not mind if paper is accepted.  
The work tackles an important and under-explored problem (LF-STVG) and presents a well-motivated architecture with strong gains on extended benchmarks. However, concerns about dataset construction, limited evaluation breadth, heuristic and under-specified memory mechanisms, and missing methodological details prevent me from fully endorsing it for ICLR. With clearer dataset documentation, more rigorous algorithmic specification, and at least one additional dataset, this could become a solid contribution.

## Reviewer Confidence

4: confident.  
I am familiar with STVG and long-video grounding literature, carefully checked the method and experiments, and I am reasonably confident in the above assessment, though some missing details (especially about dataset construction and training regime) leave room for clarification during rebuttal.
## Summary
This paper introduces Long-Form Spatio-Temporal Video Grounding (LF-STVG), a new task aiming to localize targets in videos of 1-5 minutes, addressing a gap between current research on short clips and real-world applications. It proposes ART-STVG, an autoregressive transformer framework that processes frames sequentially, equipped with selective spatial and temporal memory banks and a cascaded spatio-temporal decoder. Experiments on newly extended datasets show significant improvements over state-of-the-art methods on long videos, while maintaining competitive performance on short-form benchmarks.

## Strengths
- **Novel and well-motivated problem formulation**: The paper is the first to explicitly define and tackle LF-STVG, clearly articulating the limitations of existing methods that process all frames at once and the need for scalable solutions. Evidence: The abstract and introduction state this, and related work distinguishes it from other long-term video tasks.
- **Coherent and effective architecture**: ART-STVG integrates autoregressive processing, memory banks with simple yet effective selection strategies, and a cascaded decoder where spatial localization informs temporal localization. Ablations validate each component: selective memories improve performance (Tables 2-3), and the cascaded design outperforms a parallel one (Table 4).
- **Strong empirical results**: On five extended long-form benchmarks (1-5 minutes), ART-STVG substantially outperforms existing STVG methods, with gains generally increasing for longer videos (Table 1, Figure 2). It also remains competitive on the standard short-form HCSTVG-v2 benchmark (Table 7).
- **Practical efficiency advantage**: While inference is slower due to sequential processing, ART-STVG uses significantly less GPU memory (7.9G vs. ~25G for others), making it more scalable for long videos—a key claim of the work (Supplementary Table 8).

## Weaknesses
- **Underspecified memory management**: The memory bank update is described as appending queries without removing existing memories, which could lead to unbounded growth in streaming scenarios. This matters because it affects the practicality for truly long videos and reproducibility, yet no capacity management or forgetting mechanism is discussed.
- **Lack of quantitative validation for memory selection**: The spatial (top-N similarity) and temporal (TextTiling-inspired) selection strategies are motivated heuristically, but no quantitative metrics (e.g., relevance scores or boundary accuracy) are provided to verify their effectiveness. This matters because it leaves the selection mechanisms as black boxes, reducing interpretability and confidence in the design.
- **Training-inference mismatch**: Training uses a fixed frame length (N_f=64), but inference is claimed to handle streaming, arbitrary-length videos. The paper does not clarify how this transition is managed (e.g., via truncation or sliding windows), which matters for assessing true streaming capability and generalization to very long sequences.
- **Limited ablation across video lengths**: Ablation studies (e.g., on memory selection and cascaded design) are conducted only on the 3-minute dataset, not across all five lengths. This matters because the core claim is effectiveness for long-form videos of varying durations, and the impact of components might differ with length.
- **Dataset constraints and reproducibility**: The long-form evaluation relies solely on an extended validation set of HCSTVG-v2, which may not capture full diversity, and the extended datasets are not released. This matters because it limits validation of generalizability and hinders community benchmarking for this new problem.

## Nice-to-Haves
- Comparison to adapted baselines, such as applying state-of-the-art STVG methods to video chunks with overlap, to better isolate the benefits of the autoregressive and memory design.
- More detailed failure analysis with quantitative categorization of error types (e.g., spatial vs. temporal, due to ambiguous boundaries or distractions) to guide future improvements.
- Visualization of memory selection over time, showing which frames or memories are selected during grounding, to enhance interpretability.

## Removed Points
- **Claim of being "first" overstated**: The paper correctly positions itself as the first for LF-STVG specifically, not for long-video understanding in general, as clarified in related work.
- **Demand for statistical significance**: While error bars would strengthen claims, single-run evaluation is common in this field, and the improvements are substantial and consistent.
- **Formatting issues with figures**: The garbled tables/figures are parser artifacts, not paper problems.
- **Criticism about insufficient comparison to long-form methods from other tasks**: The paper's scope is STVG; demanding comparisons to action detection or VQA methods is scope creep.

## Novel Insights
The paper identifies that current STVG methods fail on long videos due to computational bottlenecks and irrelevant information, and proposes an autoregressive approach with selective memory to address this. The cascaded decoder design, where spatial localization informs temporal localization, is a novel insight for leveraging fine-grained cues in complex long sequences. Beyond the paper's own contributions, the reviews suggest that the autoregressive framework itself might be a major factor in the gains, hinting at future work to explore simpler autoregressive baselines.

## Suggestions
- Clarify the memory bank update mechanism, including any capacity limits or forgetting strategies for streaming inference, in the main paper or supplement.
- Provide quantitative metrics for memory selection, such as calculating similarity scores for spatial memories or evaluating temporal boundary detection accuracy.
- Explicitly describe how inference handles videos longer than the training frame length, e.g., through sequential processing of segments or truncation.
- Conduct ablations on multiple long-form datasets (e.g., 1min, 5min) to show consistency of component contributions across video lengths.
- Release the extended validation sets or provide detailed documentation on the extension process (e.g., source video handling, annotation propagation) to facilitate reproducibility and benchmarking.
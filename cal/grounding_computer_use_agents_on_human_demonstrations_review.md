=== CALIBRATION EXAMPLE 70 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- **Title Accuracy:** The title accurately reflects the core contribution: a dataset grounded in human demonstrations and models optimized for computer-use agent (CUA) grounding.
- **Abstract Clarity:** Clearly states the problem (lack of high-quality desktop grounding resources), the method (GROUNDCUA dataset + GROUNDNEXT SFT/RL pipeline), and the results (SOTA on 5 benchmarks with <1/10th the data of prior works, strong agentic performance with o3 planner).
- **Supported Claims:** The claims are largely supported by Tables 2, 3, and 4. However, the phrase *“attains comparable or superior results to models trained with substantially more data”* in an agentic setting should be lightly tempered: Table 4 evaluates a hybrid system (o3 planner + GROUNDNEXT grounder) against end-to-end proprietary agents. While valid for system-level comparison, it conflates planner capability with grounding accuracy. The abstract could more precisely frame this as “grounding-augmented agentic performance.”

### Introduction & Motivation
- **Problem Motivation & Gap:** Well-motivated. The authors correctly identify that desktop environments pose unique grounding challenges (density, resolution variability, proprietary/custom workflows) and that existing datasets rely on automation (DOM/AX trees) or synthetic generation, missing dense, real-world visual cues.
- **Contributions Accuracy:** Clearly stated and accurate. The three contributions (dataset, model family, analysis/release) map directly to the sections.
- **Over/Under-selling:** The introduction appropriately emphasizes data efficiency. The comparison to JEDI’s 9M synthetic samples highlights a meaningful research direction (quality over quantity). No major over-claims, though readers should be aware that synthetic data scales trivially while human annotation does not, framing the trade-off differently than pure algorithmic efficiency.

### Method / Approach
- **Clarity & Reproducibility:** The dataset construction pipeline (Sec 3) and training setup (Sec 4, Appendix C) are detailed. Key hyperparameters (learning rate, batch size, epochs, RLOO group size $n=8$) are provided. The 700K SFT subset composition is explicitly defined.
- **Assumptions & Justifications:** The paper assumes MLLM-generated instructions (Qwen2.5-VL-72B) preserve grounding fidelity. A 4% error rate on 100 human-eval samples is reported, which is acceptable but relies on a very small human validation set given the 700K scale. A larger stratified sample or automated consistency metric would strengthen confidence in instruction reliability.
- **Logical Gaps / Ambiguities:** 
  - In Sec 4.1, the normalized distance reward uses $D_{\text{ref}} = D_{\text{max}}(B, I)$ for exterior predictions. The exact computation of $D_{\text{max}}$ is not explicitly defined (e.g., is it the image diagonal, the maximum possible distance from $B$ to any pixel in $I$, or an empirical maximum?). This ambiguity impedes exact reproducibility of the reward function.
  - The reward design encourages centering ($D_{\text{norm}} \in [0.1, 0.5) \to +0.5$), but it does not explicitly handle topological ambiguity (e.g., overlapping bounding boxes or visually identical icons), which is common in dense desktop toolbars.
- **Edge Cases/Failure Modes:** Discussed in Sec E and F (near-misses, domain shift, spatial reasoning limits). The analysis is honest and aligns with empirical observations.

### Experiments & Results
- **Testing Claims:** The experiments directly test the central claim. The controlled ablation in Sec 5.1/Fig 3 (sampling 100K from each dataset under identical training) is a strong design that validates the data-quality hypothesis.
- **Appropriate Baselines:** Comparisons span recent SOTA (JEDI, UI-TARS, InfiGUI-G1, OS-Atlas, UGround, etc.) at matched scales. The baselines are fair and well-chosen for ICLR standards.
- **Missing Ablations / Potential Confounders:** A notable confounder in the “data quality vs quantity” claim is *instruction formulation*. GROUNDCUA uses a custom mix of Direct/Functional/Spatial prompts. It is unclear if the competing datasets (Aguvis, UGround, OS-Atlas) were converted to the same instruction templates for the 100K ablation. If not, part of the performance delta may stem from superior prompt engineering rather than annotation quality alone. An ablation controlling instruction templates across datasets would isolate the variable.
- **Statistical Significance:** No error bars, confidence intervals, or multi-seed results are reported for benchmark scores. Given the stochastic nature of RL post-training and SFT initialization, ICLR conventions increasingly expect reporting of variance or at least acknowledgment of seed sensitivity. For the RL runs (Section 5.2, Table 3), showing gains across 3 seeds would significantly strengthen the “consistent lift” claim.
- **Dataset/Metric Appropriateness:** Standard bounding-box containment accuracy is used. Benchmarks are appropriate and cover desktop, web, and mobile splits. The cross-platform generalization claim (Sec 5.3) is well-supported, though the drop on web/mobile (Table 13/14) is appropriately acknowledged.

### Writing & Clarity
- **Clarity of Sections:** The paper is well-organized. The progression from dataset construction → instruction generation → SFT → RL → evaluation is logical. Section 5.1 cleanly separates data efficiency analysis from RL post-training effects.
- **Figures & Tables:** Tables 2–4 and Table 7 are highly informative and directly support the narrative. Figure 8 effectively motivates the discrete reward design by showing the distribution of near-miss errors. Figure 9 provides excellent intuition for the reward tiers. The instruction type ablation (Table 7) is particularly clear and adds interpretability.

### Limitations & Broader Impact
- **Acknowledged Limitations:** The authors adequately cover static keyframe limitation, open-source app bias, annotation cost, and lack of end-to-end agentic testing. Section E thoroughly categorizes error modes (domain shift, near-misses, app-specific semantics, spatial reasoning).
- **Missed Fundamental Limitations:** 
  - The dataset exclusively uses open-source applications. While the authors draw parallels to commercial software (LibreOffice vs MS Office), enterprise/desktop environments often contain heavily customized, theme-altered, or proprietary UI frameworks (e.g., Qt vs WinForms vs Electron) with distinct interaction paradigms. The dataset’s coverage may not capture enterprise-specific workflows or heavily skinned applications.
  - The evaluation focuses on single-step grounding accuracy. Real desktop workflows involve state tracking across multi-window contexts, menus, and transient dialogs. The paper notes this in the Limitations section but could more explicitly frame how grounding errors compound in long-horizon agentic trajectories beyond the OSWorld benchmark.
- **Broader Impact:** The ethics statement is responsible. It addresses privacy (open-source, no PII), annotator compensation, and deployment risks. Appropriate for a dataset/model release paper.

### Overall Assessment
This is a strong, well-executed contribution that addresses a concrete bottleneck in computer-use agents: the lack of dense, human-verified grounding data for desktop environments. The introduction of GROUNDCUA fills a meaningful gap, and the controlled ablations (Sec 5.1, Appendix D.6) provide compelling evidence that high-quality supervision can outperform larger, noisier corpora. The GROUNDNEXT models achieve competitive or SOTA results with remarkably efficient training, and the analysis of RL post-training behavior offers valuable empirical insights for the community. 

The primary concerns are methodological clarifications rather than fundamental flaws: (1) the RL reward normalization term $D_{\text{max}}(B,I)$ requires a precise mathematical definition for full reproducibility; (2) the “data quality” claim would benefit from controlling instruction templates when comparing against external datasets to rule out prompt-engineering confounders; (3) reporting variance/multi-seed results, especially for RL gains, would meet ICLR’s increasing rigor standards. Despite these points, the core contribution stands solidly. The dataset release, comprehensive analysis, and clear demonstration of efficient grounding make this paper a valuable addition to the agent/grounding literature, and I recommend it for acceptance pending these clarifications.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces GROUNDCUA, a large-scale, human-annotated desktop UI grounding dataset comprising 56K screenshots and 3.56M element annotations across 87 open-source applications. Leveraging this resource, the authors train the GROUNDNEXT family of vision-language models (3B and 7B) via supervised fine-tuning (SFT) on 700K samples followed by reinforcement learning (RL) with a custom distance-based reward. The resulting models achieve state-of-the-art performance across multiple desktop grounding and multi-step agentic benchmarks, demonstrating that expert-curated, high-density supervision enables superior grounding accuracy with significantly fewer training samples than prior data-scaling approaches.

### Strengths
1. **Rigorous Dataset Curation Filling a Critical Gap:** Desktop GUI grounding has historically lagged behind web/mobile due to the difficulty of acquiring high-resolution, dense, and accurate annotations. GROUNDCUA addresses this by combining real-world human demonstrations with exhaustive keyframe annotation, yielding 64 elements/screenshot on average (vs. ~8-12 in prior datasets) and resolving small, icon-heavy UI components that automated parsing misses (Section 3, Table 1).
2. **Compelling Data Efficiency Empirical Results:** The paper robustly demonstrates that quality and instruction design can outperform raw data scale. Table 2 shows GROUNDNEXT-3B (SFT) surpasses JEDI-3B (trained on ~9M samples) using only 700K instructions, achieving a +5.4 absolute point average improvement (excluding UI-V). The controlled cross-dataset SFT evaluation (Figure 3) cleanly isolates dataset quality as the primary driver of gains.
3. **Transparent Experimental Design & Thorough Ablations:** The paper provides comprehensive analyses that align with ICLR standards for empirical rigor. Ablations on instruction type mix (Table 7), RLOO reward granularity (Table 8), group size (Table 9), and SFT scaling (Table 18) demonstrate systematic exploration. The commitment to release the dataset, model weights, code, and detailed datasheets strongly supports open science and reproducibility.

### Weaknesses
1. **Incremental Methodological Novelty & Modest RL Gains:** While the dataset is novel, the training pipeline (SFT + RLOO RL) follows established multimodal VLM fine-tuning patterns. The RL stage yields only marginal absolute gains (~1.5-2.0%), which the authors attribute to a "strong SFT ceiling." However, the paper lacks a deeper mechanistic analysis of the RL dynamics (e.g., reward variance, policy collapse, or error correction patterns), making it difficult to assess whether the reward design is optimal or if alternative RL objectives (e.g., advantage normalization, DPO/ORPO variants) would yield stronger lifts.
2. **Limited Cross-Domain Generalization Analysis:** GROUNDNEXT is trained exclusively on desktop screenshots but evaluated on mobile and web splits (SSv2, MMBench-GUI). While performance is competitive, it consistently trails models trained natively on multi-platform data, particularly on web tasks (e.g., Table 2: SSv2 87.3-90.4 vs. InfiGUI-G1-7B's 91.1-93.5). The paper acknowledges the distribution shift but does not quantify feature misalignment (e.g., layout density, aspect ratios, touch targets) or propose a lightweight adaptation strategy, weakening the claim of universal grounding transfer.
3. **Agentic Evaluation Relies on Proprietary Planners:** Table 4 evaluates task completion using OpenAI `o3` as the external planner. While this aligns with recent CUA literature, it obscures the actual contribution of the grounding module versus the planner's reasoning capabilities, and limits reproducibility for researchers without API access. Without an open-source planner control (e.g., Qwen2.5-32B or Llama-3.1-70B), it remains unclear whether GROUNDNEXT's grounding precision directly translates to robust end-to-end autonomy in practice.
4. **Instruction Generation Pipeline Lacks Granular Transparency:** The 700K SFT dataset relies on MLLM-generated instructions (Qwen2.5-VL-72B) using templates and bounding box prompts. While a 4% human error rate is reported, the exact prompt structures, template distributions, deduplication thresholds (pHash + text matching), and potential filtering biases are not fully disclosed. Subtle overfitting to LLM-generated phrasing or benchmark-style prompts could inflate in-distribution scores, warranting stricter contamination checks.

### Novelty & Significance
**Novelty:** The primary novelty lies in the dataset construction paradigm rather than model architecture. Moving beyond synthetic pipelines or accessibility-tree scraping to expert human demonstrations for desktop GUIs is a meaningful methodological contribution to multimodal grounding. The custom discrete distance-based reward for RL is practical and well-motivated, though incremental.
**Clarity:** The manuscript is well-structured, logically sequenced, and accessible. Figures, tables, and appendices effectively support the claims, despite minor parser-induced formatting glitches in equations.
**Reproducibility:** Strong. The authors provide explicit hyperparameters, training configurations, data curation protocols, and a firm commitment to open-source all artifacts (dataset, weights, code, evaluation scripts). This meets ICLR's high bar for reproducibility.
**Significance:** High. Desktop computer-use agents are rapidly gaining industry and academic traction, and precise grounding remains a critical bottleneck. By demonstrating that curated, high-density supervision dramatically reduces data/compute requirements while improving accuracy, the work challenges the prevailing "scale-at-all-costs" paradigm. GROUNDCUA will likely become a foundational resource for the GUI agent community, driving more efficient and robust multimodal agent development.

### Suggestions for Improvement
1. **Deepen RL Mechanism Analysis:** Instead of attributing modest RL gains solely to an "SFT ceiling," analyze the RL trajectory: quantify how many near-misses convert to hits, examine reward distribution shifts, and test at least one alternative post-training paradigm (e.g., DPO or preference optimization on generated pairs) to clarify whether RLOO is inherently limited for dense coordinate regression.
2. **Isolate Grounding Impact in Agentic Evaluation:** Add a controlled agentic experiment using an open-source planner (e.g., a standard ReAct or ToT implementation with an open-weight LLM). Report the same OSWorld-Verified suite to demonstrate that grounding improvements directly translate to task success independent of proprietary planning APIs, strengthening reproducibility and impact.
3. **Quantify Cross-Domain Adaptation:** Conduct a lightweight fine-tuning experiment where GROUNDNEXT is further trained on 5K-10K publicly available mobile/web grounding samples. Report performance deltas to show how easily desktop-centric grounding transfers, providing actionable guidance for practitioners seeking cross-platform models.
4. **Strengthen Data Transparency & Contamination Safeguards:** Release the exact MLLM prompts, template libraries, and pHash deduplication thresholds in the appendix. Additionally, run and report a strict n-gram/semantic overlap analysis between training instructions and test prompts across all five benchmarks to definitively rule out benchmark contamination, which is critical given the high in-distribution performance.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Add a strict UI-Vision and benchmark contamination audit. Without application-level or layout-level overlap verification and performance on a rigorously isolated holdout, the out-of-domain SOTA claims are unverified.
2. Include the 7B model in the OSWorld-Verified agentic table. Omitting the larger variant breaks the efficiency scaling narrative and leaves readers unable to assess how model size impacts end-to-end task completion.
3. Report multi-seed runs (≥3) with standard deviations for all main benchmark tables. ICLR reviewers will treat 1–3 point SOTA margins as statistical noise without variance reporting, directly undermining the reliability of your performance claims.
4. Add a cross-domain SFT baseline trained on mixed desktop/mobile/web data. The claim that desktop-only training yields "strong cross-domain generalization" is unconvincing without a head-to-head comparison proving a mixed-domain model doesn't generalize better.

### Deeper Analysis Needed (top 3-5 only)
1. Disentangle the "SFT ceiling" hypothesis from RL design limitations. The modest RL gains could stem from the small 10K RL set or simple distance reward rather than a true SFT saturation point; ablate RL buffer size and reward complexity to prove your hypothesis.
2. Stratify benchmark accuracy by bounding box area and image resolution. You claim to solve fine-grained desktop grounding, but without reporting performance specifically on small (<0.1% area) vs. large elements, the claim remains unsubstantiated.
3. Quantify the impact of LLM-generated instruction errors on ambiguous/refusal benchmarks. A 4% human error rate on 100 samples ignores how instruction hallucinations compound on complex or multi-intent prompts, which directly affects robustness in UI-Vision and OSWorld-G.

### Visualizations & Case Studies
1. Plot prediction distributions before and after RL. Overlay predicted coordinates against ground-truth bounding boxes to verify RL actually refines precision rather than merely shifting spatial bias.
2. Provide a failure matrix comparing spatial vs. functional prompts. Show side-by-side prediction overlays for complex relative instructions versus direct functional ones to expose where the model's reasoning truly breaks down.
3. Visualize systematic cross-domain error patterns. Map GROUNDNEXT's mobile/web failures with ground-truth highlights to demonstrate whether errors stem from aspect-ratio shifts, missing UI conventions, or genuine localization failure.

### Obvious Next Steps
1. Provide a rigorous train/test split protocol in the methodology. Document application-level, OS-level, and UI-template separation to guarantee no information leakage influenced the reported benchmarks.
2. Integrate confidence intervals and seed variance into all result tables. Replace single-run point estimates with mean ± std to meet ICLR’s reproducibility and statistical significance standards.
3. Complete the agentic evaluation with identical planner settings for both 3B and 7B, plus a planner-ablation. Isolate the grounding model's contribution from o3's planning capability to prove the reported gains aren't confounded by the external LLM.

# Final Consolidated Review
## Summary
This paper introduces GROUNDCUA, a large-scale, expert-annotated desktop UI grounding dataset comprising 56K screenshots and 3.56M element annotations across 87 applications. Leveraging this resource, the authors train the GROUNDNEXT family of vision-language models (3B and 7B) via supervised fine-tuning on 700K curated instructions, followed by reinforcement learning post-training with a custom distance-based reward. The resulting models achieve state-of-the-art performance across five desktop and cross-platform benchmarks using a fraction of the training data required by prior work, and demonstrate strong downstream agentic capabilities when integrated with an external planner.

## Strengths
- **Fills a critical gap with rigorously curated desktop data:** Desktop GUI grounding has historically relied on automated accessibility parsing or synthetic generation, missing dense, real-world visual cues. GROUNDCUA's human-verified annotations average 64 elements per screenshot with high resolution, successfully capturing fine-grained icons and complex toolbars that automated pipelines routinely drop (Table 1, Sec 2).
- **Compelling empirical demonstration of data efficiency:** The paper robustly shows that annotation quality and instruction design can outperform raw data scale. The controlled 100K-sample cross-dataset ablation (Fig 3, Sec 5.1) cleanly isolates dataset quality as the performance driver, demonstrating that GROUNDNEXT outperforms SFT baselines trained on multi-million corpora (e.g., JEDI's 9M samples) using only 700K instructions.
- **Systematic ablations and transparent empirical analysis:** The experimental design meets high empirical standards, with thorough investigations into instruction formulation (Table 7), reward granularity (Table 8), RL group sizing (Table 9), and SFT scaling (Table 18). The analysis of how strong SFT baselines limit marginal RL gains provides a practical, community-relevant insight, and the commitment to full open-source release strongly supports reproducibility.

## Weaknesses
- **Instruction template confounding in the cross-dataset comparison:** The core claim that GROUNDCUA's superior performance stems purely from annotation quality is partially undermined by the experimental setup in Sec 5.1. While the authors retrained baselines on 100K samples, it is unclear whether competing datasets (Aguvis, UGround, OS-Atlas) were converted to the same Direct/Functional/Spatial instruction templates used for GROUNDCUA. If each dataset retained its native prompt format, the performance delta may conflate annotation density with prompt engineering advantages. An ablation holding instruction templates constant is needed to strictly validate the "data quality over quantity" hypothesis.
- **Missing 7B variant in agentic evaluation breaks the scaling narrative:** Table 4 reports OSWorld-Verified task completion only for GROUNDNEXT-3B paired with OpenAI's o3 planner, entirely omitting the 7B model. Since the paper positions model size vs. data efficiency as a key practical contribution, failing to report how the 7B's superior grounding accuracy translates to multi-step agentic success leaves a significant gap in the end-to-end validation pipeline.
- **Lack of performance stratification by target size undermines the fine-grained grounding claim:** The dataset's primary selling point is its coverage of small, dense desktop elements (avg. element area 0.13% of image). However, all benchmark results are reported as aggregate accuracy without stratification by bounding box size or resolution. Without demonstrating whether the model specifically succeeds on the smallest targets (<0.1% area) versus larger buttons/text fields, the claim that GROUNDCUA effectively solves the fine-grained desktop grounding bottleneck remains insufficiently substantiated.

## Nice-to-Haves
- While single-run evaluation is still common in large-scale VLM grounding due to H100 compute constraints, reporting multi-seed variance (mean ± std) for Tables 2 and 3 would strengthen confidence in the 1–3 point SOTA margins, particularly for the RL post-training lifts.
- Releasing the exact MLLM prompt templates, pHash deduplication thresholds, and a brief n-gram/semantic overlap check between training instructions and benchmark test sets would enhance transparency and rule out LLM-prompt overfitting or contamination.
- A brief visualization of coordinate prediction distributions before and after RL (e.g., showing the reduction in "near-miss" cases in the 0–0.1 $D_{\text{norm}}$ bucket) would empirically validate that the RL stage refines localization precision rather than shifting spatial bias.

## Novel Insights
The paper empirically challenges the prevailing "scale-at-all-costs" paradigm in GUI agent training by demonstrating that densely grounded, expert-curated desktop data saturates SFT performance to a point where standard RL post-training yields only marginal returns (~1.5–2%). Crucially, the cross-dataset RL transfer analysis (Appendix D.6.2) reveals that weaker SFT models benefit substantially more from GROUNDCUA-based RL than stronger ones, establishing a practical guideline for the community: high-quality, instruction-diverse supervision should be prioritized during initial fine-tuning, as RL serves optimally as a targeted refinement mechanism rather than a primary performance driver for well-initialized grounding models.

## Suggestions
- In the Appendix or a revised Sec 5.1, add a controlled arm for the 100K cross-dataset experiment where a unified instruction template is applied to all datasets to perfectly isolate annotation quality from prompt formulation.
- Add GROUNDNEXT-7B (RL) to the OSWorld-Verified agentic evaluation table to complete the model scaling analysis and show how grounding improvements compound in multi-step planning.
- Stratify key benchmark results (particularly ScreenSpot-Pro and UI-Vision) by bounding box area percentiles (e.g., <0.05%, 0.05–0.15%, >0.15%) to provide direct quantitative evidence that the model succeeds on the smallest, most challenging desktop UI elements.

# Actual Human Scores
Individual reviewer scores: [6.0, 4.0, 6.0, 6.0]
Average score: 5.5
Binary outcome: Accept

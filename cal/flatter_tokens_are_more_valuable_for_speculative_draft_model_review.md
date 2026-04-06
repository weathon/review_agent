=== CALIBRATION EXAMPLE 67 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
The title clearly reflects the core insight: flatter token distributions are more valuable for training draft models in speculative decoding. The abstract succinctly states the problem, the proposed data-centric solution, the key metric (flatness), the method (SFDD), and the main result (over 2x training speedup with 50% data, inference speedup within 4% of full-dataset baseline). All claims are supported in the paper.

### Introduction & Motivation
The introduction effectively motivates the problem by highlighting the discrepancy between standard knowledge distillation (minimizing KL) and the speculative decoding objective (maximizing acceptance rate, tied to L1-norm). It clearly explains why simply changing the loss function is insufficient and why a data-centric perspective is needed. The contributions are explicitly listed and align with the paper’s content.

### Method / Approach
- **Theoretical Analysis (Section 3.2):** The analysis using a Gaussian toy model to derive the relationship between target distribution variance and the reduction in L1 distance (∆L1) is insightful and well-presented. The assumption of Gaussianity is a simplification but acknowledged as a tractable approximation. The link between variance and the proposed flatness metric (cosine similarity to uniform) is justified via an asymptotic argument (Appendix B), which is reasonable given large vocabularies. However, the empirical validation in Figure 2 relies on only 10 randomly selected samples, which is a very small set for drawing general conclusions. While the later large-scale experiments support the claim, this initial validation should be strengthened or the limitation noted.
- **Flatness Definition and Empirical Validation (Section 4.1):** The definition of flatness as cosine similarity to uniform is simple and computationally cheap. The empirical plots (Figure 2) show convincing trends when tokens are sorted by target flatness, demonstrating that high-flatness tokens yield larger ∆L1. The comparison with entropy (Figure 2d) indicates flatness is a better filter, though the theoretical advantage over entropy (e.g., linear vs. logarithmic scaling with variance) could be elaborated. The aggregation from token-level to sample-level via averaging is straightforward; the authors show robustness to using the median (Appendix G.2).
- **SFDD Pipeline (Section 4.3):** The method is simple and practical: one offline pass over the target model to compute sample-level flatness, then retain top-k% samples. The workflow is clear and easily reproducible.

### Experiments & Results
- **Setup (Section 5.1):** The experimental setup is appropriate, using the EAGLE-2 framework with LLaMA3-8B-Instruct and ShareGPT data. The baselines include a comprehensive set of data importance metrics. Evaluation across five diverse tasks is thorough.
- **Main Results (Section 5.2, Table 1):** SFDD consistently outperforms all baselines at 50% retain ratio, achieving an average speedup within 4% of the full-dataset baseline. The results are convincing, but a major weakness is the lack of statistical significance reporting (e.g., standard deviations over multiple runs or seeds). Given the computational cost, this is understandable but should be explicitly acknowledged as a limitation.
- **Ablation Studies (Section 5.3, Tables 2-3):** The ablation across retain ratios (70% down to 5%) shows SFDD’s robustness and superiority over random filtering and top-1 probability. The performance at extreme low ratios (5-20%) is particularly impressive.
- **Training Efficiency (Section 5.4, Figure 4):** The training time analysis shows that SFDD not only reduces data but also slightly improves training time compared to random filtering at the same retain ratio, possibly due to better batching efficiency. The reported 2.02x training speedup at 50% retain ratio is substantial.
- **Additional Experiments (Appendices C, G.1, G.2):** Results at temperature=0, on Vicuna-7B, and on GSM8K data further demonstrate generality. The median aggregation experiment shows robustness.

### Writing & Clarity
The paper is well-structured and clearly written. The flow from motivation to theory, empirical validation, method, and experiments is logical. Figures and tables are helpful, though some formatting artifacts from PDF extraction exist (e.g., misaligned tables, broken lines in figures). These do not impede understanding.

### Limitations & Broader Impact
The paper includes an ethics statement and reproducibility statement. Limitations are discussed in the appendix (e.g., token-level filtering is not currently practical due to framework constraints, the Gaussian assumption, and the small sample for token-level validation). However, the lack of statistical significance analysis and the potential sensitivity of the flatness metric to changes in the target model (e.g., if the target model is updated) are not addressed. The broader impact is positive, aiming to reduce computational costs for training.

### Overall Assessment
This paper introduces a novel, data-centric approach to improve the training efficiency of draft models in speculative decoding. The core insight—that tokens with flatter target distributions provide more training value—is supported by a thoughtful theoretical analysis (albeit with simplifying assumptions) and substantial empirical evidence across multiple tasks and models. The proposed SFDD method is simple, effective, and achieves significant training speedups (over 2x) with minimal loss in inference performance. The main weaknesses are the lack of statistical significance reporting and the limited sample size for the initial token-level validation. Nevertheless, the contribution is solid, the methodology is sound, and the work is well-aligned with ICLR’s focus on efficient and scalable machine learning. The paper merits acceptance, provided the authors address the minor concerns.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes a data-centric method to improve the training efficiency of draft models in speculative decoding (SD). The key insight is that tokens for which the target model's output distribution is flatter (more uniform) are more valuable for improving the acceptance rate. The authors introduce a "flatness" metric (cosine similarity to uniform) and a Sample-level-flatness-based Dataset Distillation (SFDD) approach that filters the training set to retain only high-flatness samples. Experiments within the EAGLE framework show SFDD can achieve over 2x training speedup using only 50% of data while maintaining inference speedup within 4% of the full-dataset baseline.

### Strengths
1. **Novel and well-motivated insight:** The paper provides a clear, data-centric perspective on SD training, identifying that token importance correlates with the flatness of the target model's distribution. This is backed by a theoretical toy model (Gaussian analysis) and thorough empirical validation (Section 4.1, Figure 2).
2. **Simple, practical, and effective method:** SFDD is straightforward to implement—requiring only one offline forward pass of the target model to compute sample-level flatness—and integrates seamlessly into existing SD training pipelines. The method consistently outperforms common baselines (entropy, top-1 probability, etc.) across multiple tasks and retain ratios (Tables 1, 2).
3. **Comprehensive experimentation:** The paper evaluates SFDD on five diverse benchmarks (GSM8K, Alpaca, etc.) with detailed ablation studies (retain ratios from 5% to 70%), analysis of training efficiency (Figure 4), and robustness checks (different temperatures, models, and datasets in Appendices). The results are convincing and show significant training speedups with minimal performance degradation.

### Weaknesses
1. **Limited comparison with advanced data selection techniques:** The baselines are standard heuristics (entropy, margin, etc.). A comparison with more recent or sophisticated data selection methods (e.g., gradient-based, coreset selection, or active learning techniques) would better contextualize the contribution and might reveal relative strengths/weaknesses.
2. **Theoretical analysis rests on strong assumptions:** The Gaussian model used to derive the flatness-variance relationship (Section 3.2) is a simplification of discrete LLM output distributions. While the authors bridge this via cosine similarity and provide empirical validation, the theoretical justification for why this proxy works so well in practice could be deeper (e.g., connection to other divergence measures).
3. **Explanation of flatness vs. entropy could be expanded:** Although Figure 2d shows flatness filters out more saturated tokens than entropy, a more detailed analysis (e.g., visualizing the distributions of selected tokens) would help clarify why flatness is a more precise indicator for SD specifically.

### Novelty & Significance
**Novelty:** The work is the first to systematically investigate data importance for speculative decoding from a flatness perspective. While data selection for efficient training is a well-studied area, applying it to SD with a metric derived from the acceptance-rate objective is novel.
**Significance:** The proposed method offers a practical way to drastically reduce the computational cost of training draft models—a growing concern as SD methods incorporate more complex training procedures (multi-step SFT, RL). The gains (2x training speedup with 50% data) are substantial and could make SD more accessible. The insight that flat tokens are more valuable may also inspire future work in efficient alignment and distillation.

### Suggestions for Improvement
1. **Compare with state-of-the-art data selection methods:** Incorporate comparisons with advanced techniques like gradient matching, forgetting events, or loss-driven selection to better demonstrate where SFDD stands in the landscape.
2. **Deepen the analysis of the flatness metric:** Provide more intuition on why cosine similarity to uniform works better than entropy or other uncertainty measures in this context. A qualitative analysis of the types of tokens/samples selected by each method could be informative.
3. **Discuss limitations and failure modes:** Explicitly discuss scenarios where flatness-based selection might underperform (e.g., when the target model is extremely confident, or on out-of-domain data). A brief experiment on a dataset with very peaked distributions could test robustness.
4. **Explore dynamic selection strategies:** The current method uses a static, offline selection. Investigating an adaptive strategy that updates the selected set during training (based on the draft model's progress) could further boost efficiency.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Compare with state-of-the-art train-based SD methods that already optimize training.** The paper only benchmarks against random filtering and generic importance metrics. Without comparison to methods like Medusa, Lookahead, or other EAGLE variants that may have efficient training procedures, the claim that SFDD provides a novel and necessary improvement is unconvincing.
2. **Evaluate the impact on generation quality (e.g., perplexity, task accuracy).** The paper only reports speedup and acceptance length. Speculative decoding must preserve the target model's output distribution. Without showing that filtered training does not degrade output quality, the method's practical utility is questionable.
3. **Test on a wider range of target/draft model pairs (architectures and sizes).** Experiments are limited to LLaMA3-8B-Instruct and a single-layer draft. To claim generality, results on other model families (e.g., GPT, Mistral) and draft depths are necessary to verify the flatness principle holds broadly.
4. **Ablation on the choice of flatness metric versus other similarity measures.** The paper uses cosine similarity to uniform but does not justify why it is better than L2 distance, Jensen-Shannon divergence, or other metrics. An ablation would confirm this specific choice is optimal.

### Deeper Analysis Needed (top 3-5 only)
1. **Provide a rigorous theoretical link between flatness and acceptance rate improvement for discrete distributions.** The Gaussian model is a continuous approximation; the derivation connecting cosine similarity to variance relies on a large-window assumption. A discrete analysis or tighter justification is needed to trust the metric for real token distributions.
2. **Analyze how token importance evolves as the draft model trains.** The paper claims flatness is independent of the draft model, but the value of a token likely changes as the draft improves. A study of dynamic importance would validate the static selection strategy or reveal its limitations.
3. **Justify sample-level aggregation over token-level filtering.** The paper averages token flatness per sample without analyzing the distribution of flatness within samples. This aggregation may lose information; an analysis showing why this simple average is effective is missing.

### Visualizations & Case Studies
1. **Visualize target distributions for high vs. low flatness tokens.** Show concrete examples (e.g., bar plots) of token distributions to illustrate what "flat" and "peaked" mean in practice, helping readers understand the selection criterion intuitively.
2. **Show examples of selected and rejected training samples.** Provide snippets of text that are kept or filtered out, revealing what kind of data SFDD prioritizes (e.g., ambiguous vs. clear-cut contexts).

### Obvious Next Steps
1. **Implement dynamic data selection that re-scores tokens during training.** Static selection may become suboptimal as the draft model learns; a dynamic strategy could adapt and further improve efficiency.
2. **Combine flatness with other data selection criteria (e.g., draft model confidence, token frequency).** Flatness alone might not capture all aspects of importance; a hybrid approach could yield better results.
3. **Apply SFDD to other train-based SD frameworks (e.g., Medusa, Lookahead).** Demonstrating effectiveness in multiple SD settings would broaden the impact and show the method is not framework-specific.

# Final Consolidated Review
## Summary
This paper proposes a data-centric method to improve training efficiency for draft models in speculative decoding. The key idea is to select training samples where the target model's token distributions are flatter (more uniform), as these provide more headroom for improving acceptance rates. The resulting SFDD approach achieves over 2× training speedup with 50% data while maintaining inference speedup within 4% of the full-data baseline.

## Strengths
- **Novel and well-motivated insight:** The paper identifies that token importance for SD training correlates with the flatness of the target model’s distribution—a new perspective backed by a theoretical toy model and empirical validation.
- **Simple and effective method:** SFDD is easy to implement (one offline pass) and integrates seamlessly into existing SD pipelines, achieving significant training speedups (over 2× with 50% data) with minimal loss in inference performance (within 4% of full-data baseline).
- **Comprehensive evaluation:** The method is tested on five diverse tasks, multiple retain ratios (5% to 70%), and different models (LLaMA3, Vicuna) and datasets, demonstrating robustness and generality.

## Weaknesses
- **Lack of output quality evaluation:** The paper only reports speedup and acceptance length, but does not evaluate the quality of generated text (e.g., perplexity, task accuracy) to ensure filtered training does not harm the draft model’s ability to approximate the target distribution beyond acceptance rate.
- **Theoretical analysis relies on continuous approximation:** The justification uses a Gaussian model and links to discrete distributions via an asymptotic argument; while empirical results are strong, the theoretical connection to real token distributions is not fully rigorous.
- **Comparison with advanced data selection methods is limited:** The paper compares with standard heuristics (entropy, top‑1, etc.) but not with more advanced data selection techniques (e.g., gradient‑based, coreset selection), making it hard to assess SFDD’s standing relative to state‑of‑the‑art in data selection.

## Nice-to-Haves
- Statistical significance analysis (multiple runs) would strengthen the empirical results.
- More detailed analysis of why flatness works better than entropy (beyond the provided gap plot).
- Visualization of example tokens/samples to illustrate what flatness captures in practice.

## Novel Insights
The paper’s core insight is that for speculative decoding, tokens with flatter target distributions (higher uncertainty) are more valuable for training the draft model, because they offer more headroom for reducing the L1 discrepancy that directly governs acceptance rate. This insight—novel in the SD context—leads to a simple, effective data selection method that substantially improves training efficiency without compromising inference speed.

## Suggestions
- Evaluate the impact on output quality (e.g., perplexity or task accuracy) to ensure no degradation beyond acceptance rate.
- Consider comparing with more advanced data selection baselines to better contextualize the contribution.
- Provide a more detailed analysis of the flatness metric and its relationship to other uncertainty measures.

# Actual Human Scores
Individual reviewer scores: [8.0, 6.0, 4.0, 4.0]
Average score: 5.5
Binary outcome: Accept

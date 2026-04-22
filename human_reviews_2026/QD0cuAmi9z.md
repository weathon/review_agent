# Mapping Overlaps in Benchmarks through Perplexity in the Wild

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 4, 8, 6, 8

## Abstract
We introduce benchmark signatures to characterize the capacity demands of LLM benchmarks and their overlaps. Signatures are sets of salient tokens from *in-the-wild* corpora whose model token perplexity, reflecting training exposure, predicts benchmark performance. We extract them via stepwise forward selection with linear regression in a meta-evaluation spanning 32 LLMs and 89 benchmarks across diverse domains. We then analyze how these signatures relate to both the semantic similarity of benchmark questions and the correlation structure of model performance. While performance correlations are uniformly high and semantic overlaps stay in a narrow mid-range, benchmark signatures reveal more nuanced structure. For instance, they uncover substantial overlap between benchmarks in knowledge and reasoning tasks, whereas benchmarks in culture- and humanity-oriented domains show low similarity with each other. Unlike raw performance correlations, which are influenced by benchmark-*orthogonal* factors such as question formats, signatures are robust to such confounds. We further identify cross-functional overlaps between logic, math, language, instruction following, and cultural/world modeling, with coding emerging as the most isolated function, interacting only moderately with the ability of detecting missing information. Qualitative analysis shows that only the knowledge signature aligns with actual knowledge, suggesting that LLM semantic organization may differ from human conceptual structure. Together, these findings offer insights into benchmark validity, LLM sensitivities, and the landscape of interconnected LLM capacities. We have open-sourced the code and data in [this GitHub repository](https://github.com/siyangwu1/Benchmark-Signature-Repository).

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces the concept of benchmark signatures that can be used to characterize and compare LLM benchmarks. The signatures are defined as a set of salient tokens drawn from large corpora such that the perplexity of various LMs on those tokens predict their performance on a benchmark. The paper’s contributions are (1) a novel two-stage pipeline to derive “fingerprints” of a benchmark. The pipeline involves perplexity-based screening to identify candidate tokens, and then applying stepwise regression to select a spare set of tokens more predictive of performance. (2) A framework for multi-level benchmark comparison. (3) Empirical insights into how current LLM benchmarks may be redundant or divergent.

### Strengths
- The introduction of benchmark signatures is a new way to quantify what a benchmark tests. It introduces a principled multi-level framework for analyzing benchmark redundancy. 
- The idea of a misaligned benchmark intent – e.g., logical reasoning benchmarks ending up testing a model’s instruction-following capability – is a notable discovery worthy of further investigation.

### Weaknesses
- The paper’s general hypothesis that pretraining familiarity affects performance has been noted in works on benchmark saturation. For example, Humanity’s Last Exam (HLE) was created specifically to address the saturation of MMLU by providing expert-written questions. The paper doesn’t mention HLE or similar broad evaluations that address saturation or overlapping knowledge issues.

      Phan, L., Gatti, A., Han, Z., Li, N., Hu, J., Zhang, H., ... & Wykowski, J. (2025). Humanity's last exam. arXiv preprint arXiv:2501.14249

- It appears the evaluation focuses on academic QA-style tasks (e.g., MMLU, Big-Bench Hard), many of which are multiple-choice or short-answer. This omits classes of benchmarks, such as open-ended generation tasks like summarization or dialogue, where the signature approach may be harder to apply because it inherently works better with well-defined quantitative performance metrics. This focus should be acknowledged as a limitation. 

- Some choices in the signature extraction pipeline could use further justification. The authors use a fixed threshold (top/bottom 1% by thrush correlation) for pre-filtering, and then a greedy forward selection up to 1500 tokens. Why 1% and how was 1500 chosen as the max features? There’s no comparison to regularization methods like Lasso or Elastic-Net, which the authors mention as a possible extension but do not report results for. 

- A conceptual weakness is the paper’s implicit conflation of model capability and pretraining familiarity. By design, the signature tokens are those that models find easier (lower perplexity) and that correlate with better benchmark scores. This risks a tautology since models may perform well on a benchmark because they have seen similar content before. For benchmarks aiming to test reasoning beyond memorized knowledge, a high signature overlap might indicate the benchmark is inadvertently easier for models that have seen certain clues. In other words, the approach might not distinguish generalization ability from training-set overlap.

### Questions
-  If classical regression methods are ill-posed for this high-dimensional problem, did the authors consider regularization techniques or matrix factorization/filtering methods? The use of Lasso or Elastic Net regression is mentioned as an extension but is not explored in this paper.
- For the THRUSHPREFILTER step, the paper mentions using "approximately the top 1% of tokens with the strongest signal” but doesn't specify the exact threshold. Is there a more principled data-driven procedure for selecting this threshold; for example, using cross-validation or analyzing the correlation distribution?
- Relatedly, how sensitive are the results to the fixed variable choices, such as 1% thrush filtering and up to 1500 tokens used in forward selection?
- While code is provided, certain details of the experimental setup are not explicit in the paper. For example, data processing details for extracting signatures from the RedPajama corpus would be helpful; e.g., how the corpus was tokenized, how large $d$ was, and how contexts for tokens were chosen.
- The paper would be stronger if it included concrete examples of signature tokens or token clusters for a few benchmarks. Including a table of signature tokens for different benchmarks and their regression coefficients would enhance interpretability.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces benchmark signatures to quantify overlaps among LLM benchmarks by mining salient tokens from in-the-wild corpora where token-level perplexity predicts benchmark performance. Using 32 LLMs and 88 benchmarks, the authors apply Thrush correlation filtering and stepwise forward selection to extract signatures. The work examines three overlap levels: semantic similarity, performance correlation, and signatures. Key findings show signatures discriminate better than semantic or performance measures, performance correlations are biased by format and benchmark family, and coding is the least overlapping domain while logic, math, language, and instruction-following are interconnected.

Despite the writing issues, I highly recognize the motivation of this work. I believe the problem this paper addresses is extremely important and has troubled many researchers working on large language model pretraining. The perspective that in-the-wild corpora implicitly encode the benchmark signature is quite interesting. The discovery that performance-based benchmark agreement is contaminated by format and family effects is valuable for the community. The community should encourage bold papers that tackle important problems. Soundness can be gradually supplemented during the rebuttal phase. This represents ambitious thinking we need for fundamental issues in LLM evaluation. With revisions addressing scalability and adding qualitative analysis of signatures, this will be a strong contribution.

### Strengths
The motivation is strong. Benchmark submissions have grown sevenfold and we need methods to understand if new benchmarks measure distinct capabilities or just repackage existing ones. The insight that in-the-wild corpora implicitly encode benchmark signatures through perplexity patterns is interesting and provides a theoretical connection between pretraining exposure and benchmark performance.

The experimental scope with 32 models and 88 benchmarks is substantial. The finding that performance correlations are more influenced by question format and benchmark family than actual function (Figure 1) exposes a real limitation in current benchmark agreement studies. The observation that coding benchmarks are relatively clean and distinct is useful.

The statistical framework using Thrush correlation screening and AIC-based forward selection is reasonable for the ultra-high dimensional setting.

### Weaknesses
The writing has issues. It feels like the authors have limited writing experience. The abstract is not very clear and could benefit from referencing how other papers structure theirs. Also, the last line of the last page is not aligned to the bottom. This directly affects the clarity of presentation.

My main concern is computational cost and scalability. The authors use only 1B tokens from RedPajama, then downsample by 50x to ~16.9M token contexts. Our large model in-the-wild datasets are on the scale of tens of trillions of tokens. It is unclear how this work would handle larger scales. From Section A.5.1:
- Initial scale: ~8.45 × 10^9 tokens (RedPajama 1B variant)  
- Downsampling: 1/50
- Final scale: ~1.69 × 10^7 tokens
- Feature matrix: P ∈ R^(32 × 1.69×10^7)

The complexity of generalizing this needs discussion. A previous work addressing similar scale is LESS [1]. Getting scaling behavior from 1B → 10B → 100B tokens seems quite difficult, but the authors should discuss this.

The paper never shows what the actual salient tokens are. They extract ~30 tokens per benchmark but provide no examples. What tokens predict math performance? What about coding? Without this, the signatures remain black-box features. This is a significant missed opportunity.

Methodological choices lack justification. The authors mention Lasso or Elastic Net but choose forward selection for "interpretability" without demonstrating this. No comparison with Spearman correlation or mutual information for screening. More ablations would help.

The paper discusses two interpretations of overlap - genuine cognitive interdependence versus "leaky" benchmarks - but never resolves this. When instruction-following and logic overlap more than within-logic benchmarks, the authors could use signature tokens to distinguish these interpretations but do not.

[1] Chen et al., "LESS: Selecting Influential Data for Targeted Instruction Tuning," ICML 2024.

### Questions
Can you provide analysis of how signatures change with corpus size? Even experiments with 100M, 500M, and 1B token subsets would show whether signatures stabilize. What is the computational cost and how does it scale to 10B or 100B tokens?

Please show examples of actual salient tokens. What are the top tokens for MMLU math versus MBPP coding? Do they make intuitive sense?

When instruction-following and logic show high signature overlap, what do the shared tokens look like? This could distinguish between genuine interdependence versus contamination.

Have you validated signatures on held-out models not used in extraction? This would show signatures capture fundamental properties rather than overfitting.

### Soundness
3

### Presentation
2

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces benchmark signatures: sets of salient tokens from in-the-wild corpora where LLM perplexity predicts benchmark performance. Using 32 LLMs and 88 benchmarks, the authors compare three overlap measurement approaches: semantic (question similarity), performance (score correlations), and signature-based (perplexity patterns). Key findings: (1) signature analysis better distinguishes benchmarks than semantic/performance approaches, (2) performance-level results are strongly influenced by benchmark-orthogonal factors, (3) coding is the most distinct domain while math/logic/language show cross-functional overlap.

### Strengths
1. I like the idea of signature. The signature approach elegantly unifies semantic and performance perspectives: signatures inherently contain semantic information (salient tokens from meaningful contexts) while perplexity directly hints at performance. This is a creative and well-motivated synthesis.
2. The experiments are done in Impressive scale: 32 models × 88 benchmarks with multi-level analysis provides comprehensive coverage.
3. There are good theory grounding. Good justification via SIS theory for ultra-high dimensional feature selection and two-stage filtering approach appropriately handles the d >> m problem.

### Weaknesses
1. All signatures are extracted from RedPajama only, yet the 32 models were trained on diverse corpora. It remains unclear whether RedPajama signatures are representative of these varied training distributions. Furthermore, proprietary models may rely on private training data containing unconsidered signatures. If signatures are biased, can they still meaningfully reflect model capabilities? Robustness experiments are needed to validate the approach.
2. The claim that signature correlations reveal "model capabilities" is questionable. Two alternative explanations exist: (a) High within-dataset vs. between-dataset correlations may simply reflect training data contamination, as authors acknowledge; (b) Same dataset may inherently represent a single capability independent of intended subject divisions—for instance, if all MMLU subjects source from Wikipedia, signatures may capture "Wikipedia-style capability" (or even spurious correlation) rather than genuine subject-specific skills (history vs. chemistry). The perplexity-performance link could be coincidental. Can you do more causality analysis on that?

### Questions
See weakness.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper exposes a critical flaw in benchmark categorization that benchmark overlap is dominated by superficial factors like quention format and benchmark family, rather than the actual capabilities, and proposes "Benchmark signatures" to reliably describe benchmark features. Key findings from this analysis include:
- Logic, math, IF and language form an interconnected cluster
- Coding stands distinctly alone from other domains
- Many "logic" benchmarks actually measure instruction-following
- Multilingual / cultural benchmarks show low within-cateogry similarity
The takeaway is the current evaluation methods conflate test design with capability. Signature-based analysis reveals what LLMs actually learn.

### Strengths
1. The benchmark signature concept is novel and elegantly bridges pretraining data characteristics and benchmark performance through perplexity. The mechanistic view (rather than correlational view) reveals intrinsic capabilities that benchmarks measure rather than superficial artifacts.
2. The paper repurposes proven perplexity-correlation methods from data selection literature for benchmark characterization, and provide rigorous theoretical justification through SIS framework. It explains why the proposed method work through rigorous theoretic analysis.
3. The paper reveals that cross-function overlap (instruction following vs. logic) exceeds within-function overlap in some cases, shows interesting observation that some benchmarks designed to test "logic" acutually measure instruction following, and this further demonstrates the power of the methodology which leads to non trivial findings.

### Weaknesses
1. Table 1 shows token level has the greatest standard deviation and interquartile range, and the paper claims this indidcates token level is the most informative. However, this is not necessarily true: the higher vairance could also originates from larger noise. It is better to show signal-to-noise ratio analysis (like R^2) for token vs. chunk vs. doc level.
2. All analysis uses the same 32 models for extraction and comparison. Cross-validation and held-out test results are absent.
3. Across 16 targets, token-level values achieved 12 wins, which indicates 25% of loses. Please provide analysis of when and why token level loses.
4. AIC is a greedy algorithm known to be suboptimal, why not use LASSO, elastic net, etc.? Could you provide stability analysis of feature selection?

### Questions
1. Can signatures from 24 models predict the performance ranking of the held-out 8 models on benchmarks (held-out test for predictive validation)?
2. What are the examples of actual salient tokens for specific benchmarks? Do they align with human intuition?
3. Please add confidence intervals to Fig. 4 and conduct pairwise tests for within category vs. cross category differences.

### Soundness
3

### Presentation
3

### Contribution
4

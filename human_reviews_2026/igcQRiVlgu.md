# Don't Throw Away Your Beams: Improving Consistency-based Uncertainties in LLMs via Beam Search

- Decision: Accept (Poster)
- Scores: 8, 8, 4, 4, 0

## Abstract
Consistency-based methods have emerged as an effective approach to uncertainty quantification (UQ) in large language models. These methods typically rely on several generations obtained via multinomial sampling, measuring their agreement level. However, in short-form QA, multinomial sampling is prone to producing duplicates due to peaked distributions, and its stochasticity introduces considerable variance in uncertainty estimates across runs. We introduce a new family of methods that employ beam search to generate candidates for consistency-based UQ, yielding improved performance and reduced variance compared to multinomial sampling. We also provide a theoretical lower bound on the beam set probability mass under which beam search achieves a smaller error than multinomial sampling. We empirically evaluate our approach on six QA datasets and find that its consistent improvements over multinomial sampling lead to state-of-the-art UQ performance.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The authors propose replacing multinomial sampling with beam search sampling for consistency-based UQ methods. They show that it's even theoretically less erroneous in low-sampling regimes. They verify the proposed method's effectiveness on various datasets and various methods.

### Strengths
- The paper focuses on an important topic with a neat, focused contribution. 
- The paper is well written.
- The proposed replacement of multinomial sampling with beam search definitely makes sense.
- The idea is supported by a clean, small, but necessary theory.
- Very good ablations and additional experiments with different temperatures and sampling ideas.
- Clear demonstration of the effectiveness of the idea on various consistency-based methods.

### Weaknesses
- I don't see any major weaknesses, but the impact of the work is probably limited to the UQ community.

### Questions
- Do you have any results for when you combine your idea with other methods that do sampling but use probabilities, such as SE and SAR?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper targets uncertainty quantification (UQ) for short-form QA where consistency-based methods aggregate multiple generations to estimate confidence. It argues that the standard practice to generate samples via multinomial sampling under a small budget produces many duplicates in peaky distributions. This results in high variance and unstable UQ. The authors propose to use beam-search with probability weighting to obtain samples for the consistency estimator. Experiments across several QA datasets report consistent gains in UQ metrics.

### Strengths
The paper clearly motivates the problem and proposes a simple drop-in replacement for multinomial sampling in consistency-based UQ via probability-weighted beam search. It provides a bias–variance analysis with an interpretable beam-mass condition. Experiments on multiple short-form QA datasets show consistent gains, including extensive ablation studies. Moreover, the probability-weighted beams consistently improve other UQ baselines (e.g., semantic entropy), suggesting broader applicability.

### Weaknesses
The improvements appear strongest for short answers, the coverage of probability mass by top-M beams degrades with length (Fig. 3). It’s unclear whether the advantages vanish or even reverse for longer outputs. An additional discussion or experiments on benchmarks with longer outputs could help to clarify.

### Questions
* For what answer lengths and budget M does beam-weighted consistency cease to outperform sampling? Any failure cases? 
* Could you discuss whether there are use cases in which beam search produces suboptimal predictions which may lead to unreliable uncertainty estimates?
* Figure 4 indicates that even for M=1, the PRR of the beam search is comparable with the multinomial sampling with up to M=15 (dissimilarity) and M=5 (eccentricity). Could you explain why the beam search with such small beam width is performing so well? 
* The related works section is relatively short. Work of Hashimoto et al. (2025) is mentioned where different decoding strategies are compared in terms of UQ for different tasks. How is the beam search approach performing in those evaluations? Is the main difference and gain in this paper the combination of beam search with weighted consistency scores?

### Soundness
3

### Presentation
3

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
The paper presents a new uncertainty quantification scheme for LLMs that leverages the sequence probabilities of the multiple beams obtained during the generation phase of the response. This is in contrast to existing consistency-based approaches that typically rely on generating multiple sample responses from the distribution represented by the LLM. The basic idea behind the proposed uncertainty estimator is to weight the disimilarity measure of each sequence by it' s corresponding normalised probability. Furthermore, the paper also shows that several consistency based estimators such as those based on eccentricity or eigenvectors dissimilarity can be expressed in terms of the probabilities of the beams. The experimental evaluation is restricted to QA benchmark datasets and 3 open models from the Llama, Qwen and Gemma families. The results show indeed improved performance over the corresponding UQ schemes that rely on multinomial sampling.

### Strengths
- Uncertainty quantification for LLM is currently a hot topic and therefore advances in this area are warranted.

- The quality of the presentation is overall quite good and therefore the paper is relatively easy to follow even by readers outside this research area. Most of the technical details presented in the paper are discussed in a relatively clear manner.

### Weaknesses
- The experimental evaluation considers only one generative task, namely QA, where responses are typically fairly short. LLMs are applied to a much wider range of tasks, and therefore it would be very interesting to see how the proposed approach performs in other tasks, such as for example summarisation or text-to-SQL translation.

- The performance improvements appear to be just marginal over the standard multinomial sampling e.g., 0.543 vs 0.505 in Table 3. Standard deviations appear to be missing from the table, and these need to be included to account for the noise. Also, for such mediocre performance bumps, statistical significance tests are mandatory.

- The proposed method assumes access to sequence probabilities and therefore it is not applicable to closed models such as GPT or Gemini. I think this limitation should be made clearer in the presentation.

### Questions
- I would be interested to get your perspective if I am to apply your method in summarisation tasks where sequences are longer. How would that affect the sequence probabilities?

### Soundness
3

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
3

### Summary
They propose a new family of UQ methods based on beam search, using importance-weighted estimators to produce distinct candidate outputs. The authors provide a theoretical analysis alongwith an empirical analysis across multiple datasets and models.

### Strengths
- The paper is generally well written, well structured and easy to understand.

- It proposes a new way of leveraging beam search for UQ, going beyond routine sampling or decoding tweaks.

### Weaknesses
- The focus is primarily on short QA tasks. It’s not clear how well the approach would generalize to long-form or structured generation, or to tasks with less peaked probability distributions.

- An deeper study on beam width and similarity function for semantic entropy is missing.

- A more intuitive summary and visualization of the main theorem’s impact would increase accessibility.

### Questions
- How does the method scale to tasks involving longer generations, where the probability mass may be less concentrated on a few beams?

- How does it compare to multinomial sampling technique with averaging of confidence instead of just doing maximal voting?

- How to set beam width for different tasks or LLMs?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 5

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
Uncertainty quantification is an important topic for LLMs. This paper focuses on the consistency-based methods that measure agreement among multiple generations. The authors argue that multinomial sampling often produces duplicate short answers and high run‑to‑run variance, especially in short‑form QA. They propose replacing samples with beam‑search candidates and computing importance‑weighted consistency scores over the beam set (top‑M). The goals were to reduce duplication, lower estimator variance and achieve efficiency and effectiveness with 'no extra cost', since beam search is already used for decoding.

### Strengths
**Interesting problem and good motivation**

> The paper pinpoints a concrete weakness of consistency‑based UQ in short QA: duplicates and variance from multinomial sampling, with evidence (e.g., duplicate rates 30–50% for 2–4 token outputs in TriviaQA with 10 samples) and intuitive illustrations (Fig. 1 and Fig. 2). The proposed beam‑weighted estimator is simple and broadly applicable across prior consistency‑based methods. 

**Theoretical analysis**

> The paper provided comparison analysis, covering the MSE of the multinomial MC estimator (unbiased, variance) against the deterministic beam‑weighted estimator (bias from top‑M truncation, but no sampling variance). The resulting condition, for example beam mass $m_B$​ above a threshold (e.g., >0.842 for M=10), is interpretable and aligns with short‑form QA, where top few beams often capture most probability mass. This work shows this condition holds for a sizable subset and more often for short outputs. 

**Empirical findings**

> Six datasets spanning closed‑book, open‑book, and multiple‑choice QA, six popular LLMs (base and instruct), and comparisons to a large set of information‑based and consistency‑based baselines implemented via LM‑Polygraph. The principal metric PRR (normalized AURC with AlignScore quality) follows recent UQ benchmarking recommendations.

### Weaknesses
**Limitation in theory**

> While the paper has provided comparison condition, it heavily rely on unknown “inside‑outside” gap and assumed similarity. The estimator is deterministic given a fixed beam set, but it still inherits the bias from top‑M truncation. 

**Limitation in scope**

> Most gains are for short answers; the paper itself shows the advantage shrinks as outputs lengthen (Fig. 5). It is unclear whether the approach still helps long‑form generation (summarization, step‑by‑step reasoning), where beam search can become less diverse and costlier.

**Further report on cost claims**

> The paper states UQ is “essentially free” when beam search is already run; however, many UQ pipelines today do not decode with beam for generation (often nucleus/temperature sampling). The paper reports total GPU‑days, but not per‑query latency nor a direct throughput comparison between beam(M) and sampling(M) under matched compute.

**Sensitivity analysis**

> Although STS vs. NLI ablations are shown, results do shift on some datasets (Table 8). The paper also introduces a mass floor $\epsilon$ for stability, but the “best $\epsilon$” is case‑dependent (Table 5).

**Baselines selection**

> While diverse beam and temperature sampling are ablated, a natural question is how nucleus/temperature sampling with semantic deduplication (e.g., cluster‑then‑subsample) fares as a competing “low‑variance” sampler for short QA. The hybrid beam+sampling table suggests potential, but a semantic‑dedup sampling baseline would make the case stronger.

### Questions
Further on the questions raised in Weakness, please also answer my following questions.

1. Can you report per‑query latency/throughput comparisons for beam vs. multinomial sampling at the same MMM, both when (a) beam is not used for generation and when (b) it is used (to support the “free” claim)?

2. Beyond the sufficient bound, can you measure mBm_BmB​ and provide scatter plots of PRR gain vs.  $m_B$​? Any proxy to estimate the truncation bias on real data?

3. Do the gains persist for longer generations (e.g., multi‑sentence answers, summarization)? If not, why?

4. Can you include exact‑match (or normalized string match) for TriviaQA/WebQ and choice accuracy for MC to corroborate PRR conclusions that rely on AlignScore?

5. Given the sensitivity in Table 8, do you recommend NLI vs. STS depending on task type (factoid vs. conversational vs. MC)? How stable are results across different NLI models?

7. One claimed benefit is reduced run‑to‑run variance. Can you report std/CI of PRR over multiple runs for sampling‑ vs. beam‑based estimators at fixed M?

### Soundness
2

### Presentation
2

### Contribution
2

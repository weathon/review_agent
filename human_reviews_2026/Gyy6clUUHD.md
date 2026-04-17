# Trustworthy Retrosynthesis: Eliminating Hallucinations with a Diverse Ensemble of Reaction Scorers

- Decision: Reject
- Scores: 2, 2, 8, 6

## Abstract
Retrosynthesis is one of the domains transformed by the rise of generative models, and it is one where the problem of nonsensical or erroneous outputs (hallucinations) is particularly insidious: reliable assessment of synthetic plans is time-consuming, with automatic methods lacking. In this work, we present RetroTrim, a retrosynthesis system that successfully avoids nonsensical plans on a set of challenging drug-like targets. Compared to common baselines in the field, our system is not only the sole method that succeeds in filtering out hallucinated reactions, but it also results in the highest number of high-quality paths overall. The key insight behind RetroTrim is the combination of diverse reaction scoring strategies, based on machine learning models and existing chemical databases. We show that our scoring strategies capture different classes of hallucinations by analyzing them on a dataset of labeled retrosynthetic intermediates. To measure the performance of retrosynthesis systems, we propose a novel evaluation protocol for reactions and synthetic paths based on a structured review by expert chemists. Using this protocol, we compare systems on a set of 32 novel targets, curated to reflect recent trends in drug structures. While the insights behind our methodology are broadly applicable to retrosynthesis, our focus is on targets in the drug-like domain. By releasing our benchmark targets and the details of our evaluation protocol, we hope to inspire further research into reliable retrosynthesis.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors propose a post-processing approach for retrosynthesis model outputs that employs three filters, which are then integrated into a meta-filter to refine and eliminate implausible reactions.

### Strengths
- The method is well-motivated and logically structured.
- The approach effectively incorporates substantial human chemical expertise.

### Weaknesses
### **Major Comments**
1. **Scope of Hallucination Types**  
   The abstract claims that the method can “capture different classes of hallucinations.” However, in the main paper, only one type—*Nonsense* hallucinations—is addressed. The authors should clarify whether other classes were considered or provide justification for focusing only on one type.

2. **Claim of Avoiding All Hallucinations**  
   The claim that RetroTrim “avoids all hallucinated reactions” appears overstated. Figure 4 suggests that the outcome heavily depends on the threshold and likely other hyperparameters. The authors should discuss how recall and precision are balanced to produce the results in Table 3—for example, how the threshold value is selected, and what trade-offs exist between filtering false positives and removing valid reactions.

3. **Design and Normalization of RP Scores**  
   The rationale behind the design choices for the reaction plausibility (RP) metrics requires clarification. Why is \( S_{RP} \) normalized by \(\sqrt{T}\), while \( S_{RC}\) is normalized by \(T\)? How were these normalization schemes determined? Furthermore, practical guidance on how to choose the parameters \(\alpha\), \(\beta\), and \(\gamma\) would improve the applicability of the method.

4. **Dataset Reproducibility**  
   Although the Pistachio dataset offers advantages over USPTO, it is proprietary and not publicly accessible, which limits reproducibility. The authors are encouraged to also evaluate their method on the USPTO dataset to enhance comparison and ensure open benchmarking.

5. **Missing Reference in Figure 3**  
   The caption of Figure 3 mentions “Our generator w/o scorer,” but this variant does not appear in the figure. Please verify and correct the figure or the caption.

---

### **Minor Comments**
- Some reference formatting issues exist (e.g., missing brackets in lines 73, 318, 319).  
- Text in Figures 5 and 6 is too small to read and should be enlarged for clarity.

### Questions
see weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
Single-step predictions in multi-step retrosynthesis often accumulate false positives because single-step models remain imperfect (typical top-1 ≈60%). To assess true multi-step accuracy, this paper presents RetroTrim, an evaluation protocol that scores each proposed reaction by combining machine-learning models with reaction databases. At its core is a Reaction Prior scorer, complemented by two additional scorers tailored to distinct error modes. These three signals are then aggregated into a Meta-Scorer that delivers a robust, final judgment of reaction correctness for each step—and, by extension, the overall route.

### Strengths
1. The paper tackles a real deficit in multi-step retrosynthesis evaluation. It highlights that single-step models are not oracles (typical top-1 ≈60%) and that search success rate is an unreliable metric. The work introduces a new evaluation metric specifically designed to address these shortcomings.

2. The study validates the proposed metric(s) with expert assessment, reducing reliance on potentially biased or error-prone model signals.

### Weaknesses
1. The conceptual rationale behind the proposed metrics (Reaction Prior Score etc.) is not fully articulated, making it hard to understand why they should correlate with true reaction correctness.

2. The sample set used for assessment is small, weakening the statistical confidence in the method’s reported accuracy and robustness. 4500 reactions and 32 samples are not enough. 

3. The work does not compare against standard molecule synthesizability indicators (e.g., SA score), leaving open whether the metric adds value beyond established proxies.

### Questions
N/A

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper proposes a practical method to reduce hallucination rate and increase robustness of a retrosynthesis system by employing a combination of three complementary filters. The authors design each filter carefully and in a way that makes sense for the domain, show they are all helpful but complementary, and finally demonstrate that on their test set the combined filter removes all hallucinated predictions while maintaining a high solve rate.

### Strengths
**(S1)**: Hallucinated single-step predictions are a big issue in retrosynthesis systems. The approach designed by the authors is highly practical, and precisely targets the largest obstacle to real-world adoption for many existing retrosynthesis frameworks. Each of the three filters is designed in a way that builds on top of what worked in prior work, but extending those substantially.

**(S2)**: Authors set out to collect a dataset of chemist-annotated single-step predictions, which they then use to calibrate and evaluate their filters; this is precisely the right approach in my opinion. The dataset they generate is also fairly large given the context (and would be highly useful if released to the community).

**(S3)**: Experiments convincingly show that the filters are complementary, and using a combination of them is highly effective in increasing the robustness of the pipeline.

### Weaknesses
**(W1)**: There are some aspects of this work that are not clear to me:

- **(W1a)**: Authors explain that synthetic negative reactions are generated by applying random templates in both forward and backward direction. For the forward direction, this is standard; applying a random template to reactants from the dataset can be assumed to produce a negative reaction as long as the product differs from the one recorded in the data (this hinges on the assumption that a given set of reactants can only react to produce one potential product, which is not necessarily true due to missing conditions and reagents, but approximately this assumption would hold in most cases). However, for the backward direction, how to guarantee that the proposals are not valid ways of synthesizing the given product? Are the backward-generated synthetic negatives only accepted if they then fail round-trip with a forward model?

- **(W1b)**: Every score that is part of RP uses a different normalization (e.g. dividing by number of summed probabilities vs by the square root of that number). Is there an intuition behind this, or was this simply determined empirically to maximize the predictive power of each score?

- **(W1c)**: What does "reaction count within the candidate reaction’s coarse-grained cluster and fine-grained cluster" mean? Are the authors referring to former or latter? I suppose one could also read this as intersection of coarse-grained and fine-grained, but I assume the latter is a subset of the former.

- **(W1d)**: In Section 3.5, authors mention a "BART generator" is used as the single-step model, yet later it seems RootAligned was their default choice?

- **(W1e)**: Authors mention a subset of the chemist-annotated dataset of good/bad single-step model generations will be "released as a benchmark for the community". How many reactions/annotations are being released?

- **(W1f)**: In Section 5.2, authors mention the retrosynthetic paths were assigned a confidence score determined by the lowest-scoring reaction within them. However, doesn't this mean a human would have to score the routes generated in this experiment to obtain the confidence score?

---

**Other comments**

**(O1)**: It would be good to relate the results from this work to those from the recent RetroChimera paper [1], which also shows strong results based on ensembling. That work employs an ensemble of complementary single-step models, and shows this significantly improves both coverage and ranking of the proposed reactions. However, although hallucinated predictions are pushed down to lower ranks, and may potentially get truncated out when limiting the number of predictions to include in search, some incorrect reactions still remain (see Extended Data Figures 12-14 in [1]). RetroTrim mirrors these findings, showing ensembling is highly successful also for the case of reaction scorers.

**Nitpicks**

- The use of parentheses around citations is not consistent with common practice. Whenever citation appears as part of the sentence it should not be parenthesized, but if it appears as a remark outside of sentence it should be. There are cases of this throughout the manuscript, e.g. at the beginning of introduction or beginning of Section 2.1.

- Would be nice to also include which version of Pistachio was used (i.e. from which quarter).

- Some parts of equations, e.g. Equation 1, use long words without special formatting, which can make them look a bit unprofessional. I would consider using the `\texttt` command around words like `reaction` or `ref`.

**References**

[1] "Chemist-aligned retrosynthesis by ensembling diverse inductive bias models"

### Questions
See the "Weaknesses" section above for specific questions.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces RetroTrim, a trustworthy retrosynthesis framework designed to eliminate hallucinated reactions. The method integrates three complementary reaction scorers to prune invalid reactions dynamically during multi-step retrosynthetic search. The authors further propose a human expert annotation protocol, defining seven hallucination types and three severity levels. On 32 drug-like benchmark targets, RetroTrim is reported to completely remove hallucinated reactions, outperforming all baselines both in accuracy and the number of valid synthetic routes found.

### Strengths
* The paper focuses on the trustworthiness of synthetic route prediction in retrosynthesis and clearly illustrates the negative impact of hallucinated reactions on the reliability of computational synthesis planning.

* The proposed method mimics the multi-dimensional reasoning process of human chemists when evaluating reaction feasibility, combining both novelty and practical utility.

* The authors establish a well-structured expert annotation protocol, defining explicit error categories and severity levels, which are consistently recognized among annotators.

* The proposed method outperforms all existing baselines across every major metric and shows strong potential for direct integration into existing retrosynthesis platforms.

### Weaknesses
* The proposed method increases inference time by approximately 2.3× compared to baseline systems, which may limit its scalability in high-throughput drug discovery pipelines.

* The RRS scorer's reliance on existing reaction precedents could introduce bias against novel yet chemically valid reactions, thereby potentially suppressing creative synthetic pathways.

* The ensemble aggregation is based on a fixed weighted average, lacking adaptive or learnable optimization mechanisms that could further improve robustness.

* While the paper reports pruning efficiency, it does not provide an analysis of the trade-off between hallucination filtering strength and route diversity or completeness.

### Questions
* How sensitive is RetroTrim's performance to the choice of the ensemble threshold?

* Among the few reactions that were incorrectly filtered out ("false negatives"), what is their chemical nature or common pattern?

* Could the scoring ensemble be integrated end-to-end with the single-step retrosynthesis generator?

### Soundness
3

### Presentation
3

### Contribution
3

# LOGOS: LLM-Driven End-to-End Grounded Theory Development and Schema Induction For Qualitative Reseach

- Decision: Reject
- Scores: 4, 2, 4, 8

## Abstract
Grounded theory offers deep insights from qualitative data, but its reliance on
expert-intensive manual coding presents a major scalability bottleneck. Existing
computational tools either fail on full automation or lack flexible schema con-
struction. We introduce LOGOS, a novel, end-to-end framework that fully au-
tomates the grounded theory workflow, transforming raw text into a structured,
hierarchical theory. LOGOS integrates LLM-driven coding, semantic cluster-
ing, graph reasoning, and a novel iterative refinement process to build highly
reusable codebooks. To ensure fair comparison, we also introduce a principled
5-dimensional metric and a train-test split protocol for standardized, unbiased
evaluation. Across five diverse corpora, LOGOS consistently outperforms strong
baselines and achieves a remarkable average 80.4% alignment with an expert-
developed schema on complex datasets. LOGOS demonstrates a potential to de-
mocratize and scale qualitative research without sacrificing theoretical nuance.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposed the LLM powered coding system to analyze qualitative datasets (e.g., interviews, discussions, dialogues) based on ground theory. The authors have compared with several baseline methods on five different datasets. This paper also discusses the broad impact of the proposed method.

### Strengths
- this paper tries to address a big bottleneck for ground theory based coding process for qualitative data, which is currently manually done by domain experts, and is time consuming.
- the authors provided an end-to-end workflow using LLM to automate the inductive coding system, and can develop and further improve codebooks based on ground theory. The workflow includes LLM coding, semantic clustering, graph forming, and reiteration, which is rich and interesting. 
- the authors have tested the proposed method with several baseline methods on five interesting datasets.
- it is a good interdisciplinary research to address an important real world problem.

### Weaknesses
- The authors are talking about inductive coding, but it also includes deductive coding once the codebooks have been established which will be used to guide further coding. If it is pure inductive coding, no codebook is required. Maybe the authors want to clarify this.
- It is unclear about the final outputs, whether they are summarized themes or just a short summary of a longer content? it will be good to show some examples of inputs and outputs. 
- The evaluation is more focusing on generated codes, but codes are not the final output for ground theory based qualitative data analysis. The evaluation should also focus on the generated outputs, not the codes and codebooks.
- this paper lacks algorithm novelty, rather than just utilizing the existing methods to form workflow.

### Questions
- can you please highlight the algorithm novelty of the proposed workflow.
- what are some major challenges you are facing when you use your method to test on these five datasets. Do you face some issues that some codes appear very often, and dominate most of the codes. Do you find the long-tail distribution of the generated codes? Wil LLMs tend to pick up the most dominant codes.
- how can you ensure the diversity and balanced coding using your method?
- can you provide evaluations on your final outputs (not the codebook) but the generated themes or summaries using your methods.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The paper proposes an automated framework to generate a codebook summary of a large corpus of text without the need for human annotators. The resulting codebook consists of a set of descriptive labels, called “codes,” and a set of relations on the codes. The proposed method is intended to alleviate the need for time-consuming human annotation. To generate the codebook, the paper proposes a method in which codes are generated from subsections of the text and organized into high-level clusters; next, code-pair relations are determined. 

To assess the codebook, the paper proposes five performance metrics based on desirable characteristics rooted in grounded theory and defines their sum to be a composite metric. This composite metric was used to compare their method to other LLM-based methods. In addition to the comparison, the paper included two additional validations for their proposed method. First, the paper shows a case study in which their proposed method is used to analyze root causes for errors in LLM mathematical reasoning. Their method produces a more useful summary than the other considered methods. Second, the paper compares the output of their method to a ground truth generated by human annotators on one dataset. This experiment shows alignment between the human-annotated codebook and the codebook generated through their method, although the comparison is limited to one dataset. 

The paper’s overall contribution appears to be a proposed method for automated codebook generation and five novel codebook assessment metrics. However, empirical justification is limited because (1) the composite metric is not well-justified and (2) results on only one dataset are presented in the comparison to human-annotated ground truth.

### Strengths
+ The case study in Section 4.4 provides a persuasive example of the usefulness of the proposed method. Understanding failure patterns in LLM math reasoning is an important task which is nontrivial to solve. The proposed method handles the task well.

+ The best method for codebook assessment is an important question without an obvious solution. The authors propose five metrics for codebook assessment and comparison; this is a useful contribution to help answer this important question. 

+ The metrics proposed by the authors cover a wide range of concerns. For example, the parsimoniousness metric is helpful to go beyond accuracy of the codebook to also consider conciseness.

### Weaknesses
- The abstract claims that LOGOS “achieves a remarkable 88.2% alignment with an expert-developed schema on a complex dataset.” However, when the experiment supporting this claim is more fully described in Section 4.5 and Figure 5, the reviewer found that the reported alignment is the best case across different LLMs used for aggregation and judging. A summary statistic (e.g., a sample mean) should be used instead of the best case to avoid being perceived as cherrypicking. 

- The experimental results are not persuasive. Section 4.2 compares the proposed method, LOGOS, to other methods. It is unclear whether summing up all metrics with the same weights can lead to a fair comparison. The manuscript suggests that “Metrics reflecting semantic adequacy, such as descriptive fitness and coverage, may be given higher weight relative to structural measures like parsimony and consistency.” However, their primary results did not use such weighting. The difference is very small, but it appears that OpenCode outperforms LOGOS on fitness and coverage, which are arguably the most important metrics. 

- Because LOGOS is proposed to replace human annotators, comparison to human annotation in Section 4.5 is important and comparison to other automated LLM-based methods in Section 4.2 is less persuasive and. However, only one dataset was used for comparison in Section 4.5. 

- The notation for the set of code-pair relations is incorrect. The current notation, $\mathcal{R} \subset \mathcal{C} \times \mathcal{C}$ implies that elements in $\mathcal{R}$ are ordered pairs of codes. However, the paper defines the elements of $\mathcal{R}$ as subsumption, equivalence, or orthogonality. 

- There are several misspelled words, including “developemt” on line 76, “extreemly” on line 337, “standardiable” on line 479, and “essentailly” on line 88.

### Questions
- In Section 3.2 under Candidate Generation, how is $S_{semantic}$ calculated? The range for $S_{semantic}$ values are given, but it does not seem to be formally defined. 

- In Section 3.3 under Reusability, how are used codes defined? Do used codes refer to codes that are kept after some codes are discarded or replaced in steps 4-6? The writing leaves this definition ambiguous. 

- In Section 4.5, is it fair to aggregate the codebook simply because it has a larger size than the ground truth? If no ground truth were available, would a developer know to perform that aggregation, and if so, what size to aggregate to? Are results comparable if a different number of clusters (e.g., 17 or 25 instead of 30) are used?

- Could results on comparison to human-annotated ground truth (as in Section 4.5) be provided in the Appendix for additional datasets?

- The authors state that “we strives to automate the grounded theory develop[men]t processes” in contrast to previous works which “fall short of delivering a fully autonomous, domain-agnostic pipeline.” Have the authors considered the potential dangers of fully autonomous approaches? If their framework is designed to require no human oversight, how probable is it that a large body of text will be misinterpreted by an automated framework without anyone knowing that misinterpretation has occurred?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposed a framework for LLM-based grounded theory development.

### Strengths
Address grounded theory computationally is a novel direction.

### Weaknesses
1. Lack of justification on the 5-D evaluation of code quality. Why are these chosen? Some recent work has leverage computational version of principle of trustworthiness to evaluate qualitative results, what is the merit of the proposed evaluation method compared to existing ones? 
2. Lack of discussion and comparison with existing computational qualitative analysis methods with LLMs, in related work and experiment.
3. Lack of details and experimental analysis on embedding methods, clustering methods and graph construction, as different choices within each method will affect results. This also hinders reproducibility.

### Questions
See weakness.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
Authors introduce a new method to automate grounded theory qualitative coding. Their method outperforms other baselines according to a variety of metrics. In addition, it is validated on a dataset of expert-identified problems in multi-agent reasoning.

### Strengths
- Well-motivated and theoretically grounded study. The method makes sense, is clearly described, and every part has grounding in how coding is done in qualitative studies. The paper also explains the background very well, clearly outlining steps for the grounded theory annotation. Finally, the paper explains how these methods are actually pervasive (unknowingly) in our field. If it is easily reproducible, I am sure many AI researchers could greatly benefit from using this method for error analysis (instead of the current approaches that are much more ad-hoc).

- Thorough evaluation: sound statistical metrics to measure the proposed method's performance (though I have some questions about LLM as a judge for some components) and its advantage over baselines. Representative and strong baselines were selected for comparison. However, there are issues with not reporting statistical significance.

- Case studies provide examples of how the method can be successfully used in such pertinent applications as math reasoning failures identification, multi-agent system debugging.

### Weaknesses
- Some evaluation is done with LLM judges with no validation of their accuracy (specifically, descriptive fitness and coverage).

- Summed metrics are reported, but the metrics are on different scale (eg 1-10 vs 0-1). That would give some metrics unequal weight, so they should be normalized before summing.

- Table 1 does not report statistical significance. At the very least standard deviation should be reported. Ideally some tests should be performed to get p values or report confidence intervals. 

- Case study in experiment 3 was quite confusing. Why was the result from LOGOS acceptable but not from the other two systems? What was the ideal result supposed to look like?

- Lack of a limitation/ethics section. There could be ethical issues with LLMs misinterpreting / interpreting in a biased way the qualitative data in a way experts would not, especially in qualitative data from human subjects. The authors should identify and discuss such limitations.

### Questions
> In some sense, even though there are admittedly many abundant back-and-forth iterations, conceptual and theoretical framework are usually developed first, then computer scientists embody and animate the static-metaphysical conceptualization by programming workflow.

This sentence is very dense and unclear. What did the author actually mean to say?

line 76: strives -> strive

>because flexible aggregation and multi-view clustering (e.g. via logic operations) relies on high code-sharing rate across datapoints

this justification is very unclear

- Section 3.3 is unclear. What is the code prediction step "similar to the codebook update in iterative refinement procedure"?

### Soundness
2

### Presentation
3

### Contribution
3

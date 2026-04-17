# Review

## Summary
This paper revisits a high-profile paper from ICLR 2025, "Turning Up the Heat: Min-P Sampling for Creative and Coherent LLM Outputs," which introduced a new sampling method called min-p sampling. The original paper claimed that min-p sampling achieved superior quality and diversity compared to existing methods. However, through a detailed case study, the authors of this paper reveal that the conclusions of the original paper are not supported by its own data. Specifically, they found that:
1. The human evaluations in the original paper omitted one-third of the collected data, applied statistical tests incorrectly, and inaccurately described qualitative feedback. A correct analysis shows that min-p did not outperform baselines.
2. Extensive hyperparameter sweeps on NLP benchmarks show that min-p's claimed superiority vanishes when controlling for the volume of hyperparameter tuning.
3. The LLM-as-a-Judge evaluations in the original paper were methodologically ambiguous and appeared to have reported results inconsistently, favoring min-p.
4. Claims of widespread community adoption were found to be unsubstantiated and were retracted.

From this case study, the authors derive a blueprint for more rigorous research, highlighting the importance of fair comparison, transparent statistical analysis, full data transparency, and methodological clarity.

## Soundness
4

## Presentation
4

## Contribution
4

## Strengths
- The paper is well-written and easy to follow.
- The case study is meticulously conducted, with thorough analysis and convincing results that effectively debunk the original paper's claims.
- The lessons learned from this case study are valuable and can guide future research in the field.

## Weaknesses
- The paper primarily focuses on a single target paper (Nguyen et al., 2024) and may not be applicable to other works.
- The paper mainly points out problems in the original paper and does not propose new methods or solutions.

## Questions
- Do you think the problems you identified are common in other papers? If so, what are some ways to improve the scientific rigor of empirical machine learning research?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
8

## Confidence
4
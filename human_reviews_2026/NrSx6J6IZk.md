# Benchmark of Benchmarks: Unpacking Influence and Code Repository Quality in LLM Safety Benchmarks

- Decision: Reject
- Scores: 4, 2, 6, 8

## Abstract
The rapid growth of research in LLM safety makes it hard to track all advances. 
Benchmarks are therefore crucial for capturing key trends and enabling systematic comparisons. 
Yet, it remains unclear why certain benchmarks gain prominence, and no systematic assessment has been conducted on their academic influence or code quality.
This paper fills this gap by presenting the first multi-dimensional evaluation of the influence (based on five metrics) and code quality (based on both automated and human assessment) on LLM safety benchmarks, analyzing 31 benchmarks and 382 non-benchmarks across prompt injection, jailbreak, and hallucination.
We find that benchmark papers show no significant advantage in academic influence (e.g., citation count and density) over non-benchmark papers. 
We uncover a key misalignment: while author prominence correlates with paper influence, neither author prominence nor paper influence shows a significant correlation with code quality.
Our results also indicate substantial room for improvement in code and supplementary materials: only 39\% of repositories are ready‑to‑use, 16\% include flawless installation guides, and a mere 6\% address ethical considerations.
Given that the work of prominent researchers tends to attract greater attention, they need to lead the effort in setting higher standards.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper „Benchmark of Benchmarks: Unpacking Influence and Code Repository Quality in LLM Safety Benchmarks” presents the results of an analysis of 31 benchmarks on safety issues (jailbreaks, hallucinations, prompt injection). The analysis is uses 382 papers on safety issues that are not benchmarks as control group to measure difference between benchmark papers and other papers. Moreover, the authors study how several factors influence the success of benchmark papers measured in citations. The study finds that author prominence is related to success of the benchmarks, but that code quality is not. Moreover, having executable code is important, while having perfect documentation has no strong influence.

### Strengths
The paper „Benchmark of Benchmarks: Unpacking Influence and Code Repository Quality in LLM Safety Benchmarks” presents the results of an analysis of 31 benchmarks on safety issues (jailbreaks, hallucinations, prompt injection). The analysis is uses 382 papers on safety issues that are not benchmarks as control group to measure difference between benchmark papers and other papers. Moreover, the authors study how several factors influence the success of benchmark papers measured in citations. The study finds that author prominence is related to success of the benchmarks, but that code quality is not. Moreover, having executable code is important, while having perfect documentation has no strong influence.

### Weaknesses
While most of the paper seems sound, there is a general weakness in the setup of the statistics (comment 1) and, crucially, an important confounding factor that is not considered at all that could also explain one core conclusion, i.e., the relationship between benchmark success and author prominence (comment 1). For me, this second point is the most crucial aspect and the main reason for my judgment. The comments from 3 onwards are relatively minor. 
1) I wonder why the analysis is not based on a linear model. The coefficients of a linear model effectively measure linear correlations, same as Pearson’s correlation coefficient. However, a linear model has the additional advantage that correlation between factors can be taken into account, typically leading to better results when multiple factors are analyzed in parallel. 
2) While I intuitively also believe that author prominence is related to success, I do not believe that the study design is sufficient to conclude this. Notably, nothing in the study design controls *benchmark quality* (e.g., size of benchmark, novelty of benchmark, quality of benchmark data, or similar). Since this is not controlled for, there is a simple, possible alternative explanation for the observed effect that is not ruled out: prominent authors create author higher-quality benchmarks. The lack of control for this confounder is the key issue I have with this study that greatly limits the value of this conclusion. 
3) The execution time until the example scripts were run is a strange and biased metric. If a benchmark is more comprehensive, this might have longer examples, which would not be a bad thing. Still, this would show up as higher execution time. A cleaner measurement would be to measure the time between cloning the repository and starting the successful run, since this would only quantify the researcher effort to get to this point, excluding the confounding effect of example runtime. 
4) The ethics section is not an ethics section at all. Instead, the section reports limitations of the study. This needs to be changed and format requirements must be adhered to. 
5) Figure 2 is very hard to read and I strong suggest to avoid this strange pie-bar like visualization method. A grouped (stacked?) bar chart would likely be a lot easier to read. This is actually what is used in Figure 3, which is easier to read. However, Figure 3 suffers from bad legend placement, which partially hides the information. 
6) There seems to be a mismatch between table 2 and the textual description of the contents (Page 4, L181). The text mentions the geolocation, which I cannot find in the table. The table instead mentions area, which I assume is the research area – though I may misinterpret this. Anyways, this should be harmonized. 
7) All references are broken (missing brackets), likely because the latex stile was changed without ever checking this.

### Questions
1) What is the impact of not using a linear model for the analysis, especially given that there seem to be strong correlations between the independent variables?
2) Why are the results valid, even though there is no control for actual benchmark quality?

### Soundness
2

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
4

### Summary
This paper evaluates the academic influence and code quality of 31 LLM safety benchmarks compared to 382 non-benchmark papers. The authors claim that benchmark papers show no clear advantage in citations, and neither author prominence nor paper influence correlates with code quality. Many repositories have usability and ethical shortcomings, with only 39% of repositories ready-to-use and 6% addressing ethical considerations. The authors suggest that prominent researchers should take the lead in improving standards.

### Strengths
The paper presents several interesting and valuable findings. 

Notably, some of the ethical and reproducibility-related metrics—such as only 39% of repositories being ready-to-use, 16% including flawless installation guides, and a mere 6% addressing ethical considerations—highlight the need for researchers to pay more attention to open-sourcing and maintaining their code alongside their research contributions. 

Additionally, the observation that author's h-index does not show a strong correlation with code quality is intriguing and provides an important perspective on the relationship between academic influence and research artifacts.

### Weaknesses
Much of the experimental design seems to rely on somewhat imprecise metrics and a fair amount of manual inspection, which makes the paper’s motivation a bit hard to follow.

Additionally, the conclusions, experimental design, and motivation appear somewhat subjective, reflecting the authors’ own perspective rather than broader community evidence. It might be more appropriate to frame this part as an initial motivation supported by larger-scale community surveys rather than just the authors’ judgment. For example, the statement in the introduction, “counterintuitively, we find that benchmark papers show no significant advantage in academic influence over non-benchmark papers”, carries some subjective interpretation; collecting feedback from a larger set of participants could make this claim more robust and less reliant on individual judgment.

### Questions
1.	Relevance of conclusions in RQ1: The authors conclude that benchmark papers do not show a statistically significant difference in citation metrics compared to non-benchmark papers, based on GitHub Citation Count and Citation Density. However, benchmark and non-benchmark papers inherently serve different roles in research, so it is not clear that a “higher or lower” comparison is meaningful. Additionally, measuring influence through GitHub stars may be misleading: benchmarks often serve as tools that are widely used but not necessarily “starred,” whereas other papers may receive stars more readily. Therefore, using GitHub data alone to assess influence may introduce bias.

2.	Experimental design and evaluation metrics: Although the authors acknowledge limitations in the “Imperfect Metrics” section of the ethics statement, the choice of metrics seems questionable. For instance, in RQ2, the primary criterion for evaluating benchmark quality is code quality and reproducibility. This seems unusual, because benchmarks are meant to provide measurement standards in a field, and the underlying evaluation ideas may be more important than the engineering quality of the code. Many researchers are not professional software engineers, and their code is often written for research purposes rather than for production-grade usage. While such criteria may make sense for toolboxes, it is unclear why code quality should be the main standard for benchmarks. This seems closer to evaluating whether research code undergoes proper code review rather than assessing the scientific quality of the benchmark itself.

3.	Subjectivity of conclusions: While the paper’s conclusions may have value, the path to reaching them appears questionable. Determining how to evaluate research work, what metrics best reflect quality, and the relative importance of code within a research contribution are highly subjective issues. More justification or broader evidence may be needed to support the chosen evaluation design.

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper presents an study to evaluate the quality of benchmarks of large language models (LLMs) and safety.  The authors investigated three research questions (RQs): RQ1:the influence of current benchmark RQ2: the quality associated code repositories of the benchmark and factors to assess the quality; and RQ3: the relationship between influence of benchmark papers and the code quality.

### Strengths
Strengths
- Timely topic focusing on the safety of LLM benchmark
- Broader impact towards the research community and well as industry practioners who deploy or develop LLM

### Weaknesses
Weakness.
- Any insights of security researchers would be insightful

### Questions
N/A

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper presents an analysis of LLM safety benchmarks, analyzing 31 benchmarks and 382 non-benchmarks. The analyses what causes benchmark papers to get cited. Is it code quality or author prominence? It also analyses wheter the benchmark papers are cited more than non-benchmark papers. 

Data is colleced in a structured and transparent way.  The analysis is well-motivated and rigorous relying on statistical analyses.

Conclusions are prominent resarchers are cited more, code quality is not important, benchmark papers are not more cited than non-benchmark papers (at least in this subfield), and becnhmark with functional code that can be used without modification are more cited than those offering code that requires modifications

### Strengths
* The paper is well written and structured well. 
* The methodology is rigorous, and I trust the conclusions.  
* The conclusions are interesting and, I would guess, probably valid in general and not only for LLM safety benchmarks, although none that have ever looked at resarch code will be surprised to learn that resarch code often is not of high quality (the incentives are not there). Everyone that publishes a benchmark should take note that making it easy to run increasesits scientific impact. 
* The ethical statement is very nice and an example to follow with its detailed discussion of limitations of the method.

### Weaknesses
I do not find many weaknesses in this study. On the contrary, I find it very rigorous and trustworthy.

My main concern is whether the topic is too narrow, as it is a benchmark of LLM safety benchmarks. It is a bit on the side of representation learning, so I am not sure the community would value this study despite its many strong qualities. 

My strong belief is that this type of meta studies that informs us, the AI community, what constitutes good resarch and what influences impact are important. However, I am well aware that many do not.

### Questions
Did you consider to follow the PRISMA methodology [1] for structured literature reviews when conducting your structured search for benchmarks? 

I would have liked to see a flow diagram, such as the one proposed in the PRISMA methodology, for understanding how many papers were retrieved and excluded at different steps.

[1] https://www.prisma-statement.org/

### Soundness
4

### Presentation
4

### Contribution
3

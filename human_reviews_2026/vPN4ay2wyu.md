# The Cost of Reproducibility in Artificial Intelligence

- Decision: Reject
- Scores: 6, 4, 6, 2, 2

## Abstract
**Background.** The reproducibility crisis has not left artificial intelligence untouched.
Lack of documentation in published research can make independent replication an
unnecessarily laborious task. We propose the cost of reproducibility as the labour
required to reproduce a method and its results due to lacking documentation.

**Objectives.** We aim to quantify the cost of reproducibility to determine significant
variation between venues. We hypothesise that studies published in venues with
strict reproducibility requirements in the review process are less costly to reproduce.

**Methods.** We propose five dimensions of the cost of reproducibility and evaluate
them on a scale of 1 to 10, using objective characteristics e.g., availability of code,
data, parameter values and experiment setup. We reviewed 1061 papers published
between 2022-2024 from AAAI, ICLR, ICML, IJCAI, JAIR, JMLR and NeurIPS.

**Results.** Machine learning conferences are up to 16.52% less costly to reproduce
than artificial intelligence conferences and 12.91% than journals. Award-winning
papers are not less costly to reproduce than average papers at the same venue.

**Conclusions.** By quantifying the reproducibility cost, we find that the effectiveness
of reproducibility standards depends on community support and strict enforcement
in the review process, to significantly lower cost. We encourage the publication
of appendices and reproducibility checklists, and a low cost as a key criterion for
paper awards to drive community changes with examples of best practices.
awards to drive community changes with examples of best practices.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a rubric-based quantification of “cost of reproducibility” that operationalizes documentation quality across five dimensions—implementation, data, configuration, experimental procedure, and expertise—each scored from 1 to 10 via additive penalties. The authors sample about a thousand papers from seven leading ML/AI venues, restrict evidence to materials directly linked from the publication, and analyze venue-level differences using nonparametric tests. A small reliability study with 46 double-annotated papers reports excellent ICCs for implementation, data, and configuration; good for experimental procedure; and only moderate for expertise, after which the expertise dimension is excluded from downstream analyses. Key results include that ML conferences exhibit significantly lower implementation cost than AI conferences and journals, that configuration cost is lowest at ICLR/ICML despite NeurIPS’ stricter checklist enforcement, and that award-winning papers are not systematically less costly to reproduce. The paper argues that community norms and enforcement during review are more predictive of documentation quality than the mere presence of checklists or appendices.

### Strengths
## Novelty and contribution
The paper provides a concrete and transparent operationalization of reproducibility “cost” as documentation shortfalls rather than computational resource requirements. The five-dimensional rubric in Table 3 is detailed, public, and anchored in practical indicators. By sampling across top venues and both conferences and journals over multiple years, the work delivers a cross-sectional audit of current practice that goes beyond prior smaller-scope studies and moves from binary reproducible/irreproducible labels to a graded notion of documentation burden.

Interesting findings are e.g. the differences between major ML and AI conferences, and the fact that award papers are not easier to reproduce on average.

## Correctness
The methodological stance is clear and internally consistent: evidence is limited to materials directly linked by the publication, thereby standardizing what a diligent reader would find. Reliability is explicitly addressed; while the double-annotation is small, the ICCs for the retained dimensions are excellent to good, supporting use of those scores in venue comparisons.

It is especially commendable that the authors report distributional diagnostics and select nonparametric tests accordingly. Apart from reproducibility, parts of the field are also struggling with measures of statistical significance. It is pleasant to see that a paper analyzing the reproducibility crisis does so with due diligence on the statistical side.

## Presentation and clarity
The manuscript is carefully organized, with figures and tables that map directly to claims in the text. The contrast between venues is repeatedly tied to concrete quantities, and deviations (e.g. the slightly reduced sample size through a sampling mistake) are openly discussed.

### Weaknesses
## Novelty
While the cost is novel, it is also mostly a heuristic without too much justification. In addition, the paper's main findings do not directly result in novel recommendations for measures that venues could implement to strengthen reproducibility. The recommendations in the conclusion are fairly high-level and could have been proposed independently of the study.

## Cost design
The primary limitation is the rater design. A single rater annotated all papers, and only 46/918 were double-annotated, selected by volunteers from the authors’ network and by interest. This introduces potential selection bias and may overestimate reliability. The moderate ICC for expertise led to dropping that dimension, but it also means that the rubric initially included a dimension that was not reliably operationalized.

The score also oversimplifies things. Depending on the paper, it is not clear how important a dataset is compared to the code. In addition, the implementation may often cover the configuration and/or data. Thus, treating these categories as separate dimensions may be problematic. A single score will always contain an implicit weighting of categories. This becomes especially problematic since the penalties from Table 3. are pretty arbitrary in their weighting.

Within the rubric, equal weighting of datasets in the data dimension can penalize papers that include many secondary benchmarks while emphasizing a primary dataset. The decision not to treat widely used benchmarks differently may improve objectivity, but may also disadvantage subfields with many commonly used datasets.

## Presentation and clarity
Several central elements are relegated to the appendix, which makes readability a bit harder. Table 3 is essential for understanding scores and should be summarized more prominently in the main text. This is most likely a space concern and could be fixed with the extended 10 page limit for the rebuttal version. Minor inconsistencies (93.95% vs 93.85% public data rate), phrasing (line 376, “implementation links occurs”), and formatting glitches (citation formatting around line 881) should be corrected. 

Scope and granularity
The study aggregates across subfields and years despite policy changes within the window (e.g., NeurIPS 2024 checklist mandate versus ICLR’s optional statement). Year-wise trends could strengthen causal narratives around policy impacts. Similarly, stratification by subfield (e.g., vision, NLP, RL) could reveal domain-specific documentation norms that confound venue-level comparisons. The SCOPUS enrichment in Appendix D.1 is peripheral and thinly analyzed; if kept, it would benefit from reporting effect sizes and uncertainty or else be moved fully to supplementary material.

### Questions
- Though usually (rightfully!) discouraged for review, it would be interesting to compute the correlation of the authors judgment with several of the strongest LLMs. Have you checked your score against one of those? If successfully cross-checked on the main data from the paper, this could be used to check thousands of papers without much effort.

- Have you considered the interplay of your categories, e.g. implementation with configuration? They are presumably not independent from each other.

- How sensitive is your cost against changes to the penalties in Table 3?

- Have you made experiments on how strongly the cost is correlated with the success rate of actually reimplementing the papers? E.g. in the spirit of Gundersen et al. from 2025, have you tried to implement some of the papers and correlate that with the cost? Can you maybe reuse the results of Gundersen by also rating those papers according to your metric?

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper explores seven top machine learning and AI conferences/journals to assess the cost of reproducing approximately 50 papers per year between the years of 2022-2024, resulting in an analysis of 918 applicable reviews out of 1061 sampled papers. To conduct this cost-based analysis, the authors initially suggest five dimensions of Implementation, Data, Configuration, Experimental and Expertise; however, they omit Expertise due to their identification of it as a 'noisy' dimension during the second review. Overall, their results highlight that ML conferences typically have the lowest cost of reproducibility compared to AI conferences and Journals. The topic of reproducibility is timely and well chosen due to its importance in verifying scientific contribution and the paper provides a few recommendations alongside its cost framework, for improving reproducibility standards across ML, AI and journal publications.

### Strengths
1. The paper identifies cost as an overlooked and understudied dimension of reproducibility studies, offering a novel framework for modelling the five dimensions of cost of published papers.
2. The survey ambitiously reviews 918 applicable papers across seven of the top venues in the AI/ML domain and provides insights into their cost, which provides a large-scale overview of reproducibility cost. 
3. Practical insights are offered into how to reduce reproducibility costs, which include using visible reproducibility checklists, appendices and the introductions of reproducibility reviewers at conferences. I believe that these could be impactful in helping to curb reproducibility issues in publications; however, I wonder if they go far enough with regard to the importance of the issue as highlighted in the paper.

### Weaknesses
1. **Reducing overstatements**: 918 applicable papers are analysed in the body of the paper, and this should be represented over the number of 1061, which is highlighted in the abstract. 
2. **Fixed cost scale of 1-10**: Placing an upper bound on the cost feels as though it could be misrepresentative of how frequently we see extremely high cost papers. Especially in dimensions such as implementation, where a score of 10 is quite frequent. Having looked at the cost scale in Appendix A Table 3, I see no clear rationale for this threshold. Could you provide further reasoning for this?
3. **Balanced representation across venues**: How do you arrive at the statement L461) '150 per venue is statistically robust' across all venues, as each venue has a large variance in the number of publications. For example, NeurIPS has far more papers per year than JMLR, as you recognise in (L996) ' JMLR 2024 only had 49 papers', so it does not seem fair that each venue was not represented proportionally, which could greatly impact the findings. 
4. **Second Review**: The use of first authors to review their own work initially strikes me as an odd decision due to potential bias towards providing a lower cost for their own paper. I think the second review is a very important element of this study; however, it does not seem clear why 5.01% is a sufficient amount of second reviews. I worry that trends may change if more second reviews were conducted. Also, the second reviews were not uniformly sampled from all venues, so some venues could be better/worse represented than others in the second review stage, which can greatly impact findings.
5. **Omission of Expertise dimension from Table 3**: It is stated that this dimension was removed from analysis due to disagreements between first and second reviews which appears fair, however, it is important to see how the rubric for this dimension differs from others to observe if the dimension was removed as a result of subjectivity of the particular dimension's evaluations or a property of the sample collected for the first and second review. 
6.  **Background Section**: The background feels overly verbose, while it is important to focus on the distinction of definitions for reproducibility, it feels that the content of the section does not add to the overall content of the paper and that this could be made more concise with a fuller discussion in the appendix. Furthermore, it feels more apt to bring forward discussions in the appendix, such as that by Poldrack (2019), which directly discusses reproducibility cost. 
7. **Minor points** Table 1 caption appears incorrect, it states that AAAI/ICML have more than 150 publications reviewed, but AAAI only has 142. It might be helpful to highlight that the rubric is an indirect measure of cost earlier in the paper. Line 897 typo of 'rpresnt' which should be represent.

### Questions
1. The statement (L996) ' JMLR 2024 only had 49 papers' seems odd to me, as Volume 25 (January 2024 - December 2024)  has far more papers than the 49 discussed in the paper, please can you provide how you obtained this number?
2. How would a larger proportion of second reviews would lead to different results on the agreement between ratings?
3. Can you add a statement on whether you accept or reject your main hypothesis in the main paper?

### Soundness
2

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
The paper presents a comprehensive study on the reproducibility factors of papers published in top ML conferences. The paper introduces a notion of "cost" - different from monetary cost, this cost tracks the amount of roadblocks one could face reproducing a paper by solely relying on the paper and associated released artefacts. The authors construct an annotation guideline for cost on various metrics, and annotate a large amount of papers published in the conferences. The authors present several interesting statistics on the state of reproducibility for multiple conferences.

### Strengths
- Experimental methodology is well designed and executed
- Sound analysis and results on the reproducibility issue in multiple conferences and journals
- This work highlights multiple aspects of the reproducibility issues - what I found most interesting is the high % of broken links, and how ICLR, ICML and NeurIPS (flagship ML conferences) have lower implementation cost than other venues and journals. As the authors point out, this is likely a byproduct of strong reproducibility checklists and guidelines adopted by these conferences.
- The fact that award-winning papers also have similar costs as other papers is an interesting outcome.

### Weaknesses
- My biggest gripe about the empirical data presented in this paper is the analysis is primarily done by a single reviewer (the first author). Only 5% of the papers were reviewed by different annotators (although the paper does a good job identifying the challenges in such evaluation). While the volume of papers reviewed is much higher than Raff et al or Gundersen et al, the high volume is also a significant source of bias by being annotated by a single reviewer. However, the complexity of annotating this kind of work is very high, so getting more annotations might also be prohibitively costly. 
- The definition of cost doesn't consider software deprecation, which is an important aspect to keep in mind while reproducing prior work and is often ill-documented in the repositories.
- The cost estimates perhaps could have been done according to the subcategory of research. Certain popular domains (RL) can likely have lower cost compared to papers introducing frontier models.
- These days the difference in quality of papers in different conferences is slowly becoming less, as its increasingly the same set of people submitting (or resubmitting) the papers. The more interesting metric of tracking cost could have been on time - do we see an increasing/decreasing or mostly constant rate of cost for papers published over the years? This could have helped us understand the state of reproducibility better.

### Questions
- Does data cost change with time? I'd assume older papers have lower data cost given the datasets they used to train on are readily available now
- L359-360: the data cost is less costly, so how this is "in conflict" with higher public data rate? shouldn't it be "in agreement"?
- In the same vein, how is JMLR data cost higher even if the public data rate is high? I cannot understand how these two metrics wont be correlated with each other
- In L423, "4.22% of all empirical studies the implementation link is either empty or broken", however in L375 authors note "6.59% of all implementation urls are either empty.. " why are these two values different?

*Suggestions*

- Regarding the conclusion (L482), there is a dedicated venue for publishing reproducibility issues and improvements in ML, the Machine Learning Reproducibility Challenge (MLRC), which could be added here.
- Add links to the ML Reproducibility Checklists 
- Re L441: another reason could be why ICLR has similar cost as NeurIPS despite not having stricter reproducibility requirements is the historic use of open reviewing and rebuttals. Curious what the authors think about this.

*Typos*

- L445: ~"do no allow"~ -> "do not allow"

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper attempts to quantify the cost of reproducibility by reviewing the available documentation  of 1061 papers from a number of the most prominent AI and machine learning conferences. Cost is measured across multiple dimensions, namely Implementation, Data, Configuration and Experimental procedure. A list of guidelines are used to score different works along these dimensions, with 46 papers recieving two reviews. Machine learning conferences were found to require lower reproducibility costs than other venues. Best paper awards were not any more or less reproducible than other papers.

### Strengths
* This is an extremely important topic within the scientific community and therefore equally important to evaluate within the field of AI.  
* This work evaluates a large number of papers and does reveal some interesting insights into reproducibility patterns within the AI and ML conferences evaluated. For example, the paper illustrates the lack of detailed documentation in the absense of public code. 
* The paper is also generally well written.

### Weaknesses
The current procedure to determine cost is heavily reliant on a number of variables that rely on the subjective estimation of the effort required by researchers to reproduce research and is not measured against a more objective measure (such as hours needed by researchers to reproduce experiments). More specifically, for implementation cost, the research question is posed 

*"Given the documentation shared by the authors on a new method, how much effort would it be to re-implement the method from scratch?".*

I'm not sure that this question is answered in this paper since the gap between  *reading*   experimental documentation and *reproducing* it can be significant, as some challenges only become clear upon attempting to run code/implement experimental proceedures. A more convincing motivation for the scoring procedure proposed in this work would be to determine the correlation the proposed cost with the hours taken to reproduce work, perhaps for a random subsample of data. Though this is clearly a more challenging task, I think this is quite important as, in the same way that the authors have observed links that are available in a paper but lead to nowhere, there could be similar dark patterns in documentation, partially missing code or a large amount of well formatted content that obvisgates important parameters for reproducibility. 

I also find it challenging to justify decoupling the hardware cost from the reproducibility cost as I am inclined to disagree with the assertion that accounting for variation in computation is more closely aligned with outcome reproducibiliy, espectially in an era where large-scale models are presented as superior due to their sheer size. Such assertions cannot be tested in a tractable mannor without access to similarly large-scale computational resources. I also do have some questions on the sampling of papers for review as there does appear to be some level of sampling bias, which I have detailed in my questions below.

### Questions
1. p. 5 *"We evaluate each dimension based on guidelines that allow for the expression of subjective variation between studies."* => Can you elaborate on what this means?


2. p. 6. *"Due to the difficulty of the task, the effort required for reviewing and the challenge of finding proficient reviewers, initially, only a single review was acquired for each paper by the first author"*  => Was the first review for each paper conducted by the first author themselves? 


3. p. 6 *"To mitigate this limitation, we acquired a second review for 46 (i.e. 5.01%) out of the 918 papers from independent researchers in our network of PhD candidates and post-doctoral researchers. We let our 14 volunteers select up to five papersfor secondary review based on their expertise and personal interest"* => Could there be a selection bias here in terms of both the composition of reviews (within a single network) and the papers which are selected by them? Isn't it likely that members of a similar network want to read papers based on similar topics?

### Soundness
1

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper conducts a study of the cost of reproducibility of 918 (empirical) papers from AAAI, IJCAI, ICLR, ICML, NeurIPS, JAIR, and JMLR from the years 2022-2024. The cost of replicability was assessed on a scale from 1-10 and along 5 axes: expertise cost, experimental cost, configuration cost, data cost, and implementation cost. Implementation cost is claimed to be the most expensive for all venues. Configuration cost was the next most expensive overall, followed by data cost, followed by experimental procedure. Expertise appears to not have been reported due to issues with grounding/objectivity of the evaluation. ML conferences were found to have lower costs of replicability compared to more general AI venues.

### Strengths
This study represents a significant effort in assessing reproducibility costs of publications in major ML venues. The sample size is large and appears representative. The cost categories are largely reasonable, though I’m a bit uncertain that configuration cost represents a true reproducibility cost distinct from implementation cost, since parameter setting ought to follow from the implementation.

### Weaknesses
The paper would benefit from being restructured. A large chunk of the introduction is spent on historical anecdotes, apparently at the expense of  including the scoring methodology in the body of the paper. Since this scoring table significantly affects the interpretation of results, I would expect it to be included in the main body. 

I am also not sure what the take away from this paper ought to be. The costs of reproducibility across the 5 dimensions do not share any particular unit of measurement, and have varying maximum values, but they’re compared to one another as though their scores are comparable metrics. While scores within a category and total scores across venues could certainly be compared, I don’t think the observation that implementation reproducibility is most costly can be supported by this methodology.  

Comment on presentation - 
“We agree with Goodman et al. (2016) on the intuition behind what they refer to a search for the ‘truth’.” but then go on to say, “We also disagree with the reasoning that the search for ‘truth’ is the main motive for reproducibility. We find our answer in Popper (1934) instead: we argue it is the credibility of a method.” 
I’m generally not sure what the take away from this paragraph is meant to be, and this point in particular is unclear.

### Questions
Please see comments in weaknesses.

### Soundness
1

### Presentation
2

### Contribution
1

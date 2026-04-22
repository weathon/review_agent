# Unpacking Human Preference for LLMs: Demographically Aware Evaluation with the HUMAINE Framework

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 10, 2, 6, 6

## Abstract
The evaluation of large language models faces significant challenges. Technical benchmarks often lack real-world relevance, while existing human preference evaluations suffer from unrepresentative sampling, superficial assessment depth, and single-metric reductionism. To address these issues, we introduce HUMAINE, a framework for multidimensional, demographically aware measurement of human-AI interaction. We collected multi-turn, naturalistic conversations from 23,404 participants that were stratified across 22 demographic groups, both in the US and UK, to evaluate 28 state-of-the-art models across five human-centric dimensions. We use a hierarchical Bayesian Bradley-Terry-Davidson (BTD) model, with post-stratification to census data, and our analysis reveals three key insights. $\textbf{(1)}$ We establish a clear performance hierarchy where $\texttt{google/gemini-2.5-pro}$ ranks first overall, with a 95.6\% posterior probability of being the top-ranked model. $\textbf{(2)}$ We uncover significant preference heterogeneity, with user age emerging as the primary demographic axis of disagreement; a model's perceived rank can shift substantially across age groups, exposing failures in generalisation that unrepresentative samples typically mask. $\textbf{(3)}$ We quantify the vast difference in discriminative power across evaluation dimensions, with ambiguous qualities like Trust, Ethics and Safety showing a 65\% tie rate, in stark contrast to the decisive 10\% tie rate for Overall Winner. Our work emphasises the need for a more multidimensional, demographically aware perspective in LLM evaluation. We release our complete dataset, interactive leaderboard, and open-source framework.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes DIVERSE, a demographically aware, multidimensional framework for human evaluation of LLMs. The authors collect large-scale, multi-turn pairwise comparisons from stratified US/UK participants and analyze results with a hierarchical Bayesian Bradley–Terry–Davidson model combined with census post-stratification. They report a clear performance hierarchy among contemporary models, with one model consistently leading, but show that preferences vary meaningfully across demographic groups—especially by age—altering perceived ranks. They also find that evaluation dimensions differ markedly in how decisively users can discriminate between models, with holistic judgments being more decisive than safety-oriented ones. The work advocates moving beyond single-metric leaderboards toward nuanced, population-aware assessment.

### Strengths
1. This paper surveys a large, demographically stratified US/UK population across many groups, collects extensive multi-turn data, and commits to releasing it—providing a high-quality community resource.

2. The authors use parallel pairwise dialogues and a hierarchical Bayesian BTD model with tie handling and post-stratification, explicitly modeling demographic heterogeneity to produce uncertainty-aware, multidimensional rankings.

3. The paper also provides actionable, empirically grounded insights. It shows that “overall” judgments align closely with core task/reasoning while rankings shift on communication and safety; reveals systematic age-driven preference differences with older users more tie-prone; and demonstrates large disparities in metric discriminability. These discoveries can guide audience- and task-specific model selection and alignment.

### Weaknesses
1. The participant pool, though large, is limited to US/UK and underrepresents broader cultural and socioeconomic contexts (e.g., non‑Anglophone regions, Global South populations, rural communities, low digital‑literacy users, non‑binary identities, and diverse education/income strata).

2. The paper reports five dimensions but offers little analysis linking the four sub-dimensions to the overall score or unpacking how conversational features (topic, complexity, safety triggers) map to those judgments; the chosen sub-dimensions may also miss facets like creativity, empathy, humor, and long-horizon reliability.

3. Data arise from a study interface rather than organic, in-the-wild use, so prompts and stakes may diverge from real workflows, creating ecological bias; in addition, compared with standard automated benchmarks, this framework is harder to deploy and slower to collect, and the paper could better justify the return on that extra effort.

4. The related work section could be broadened to cover more papers on human preference variability and pitfalls in human feedback (e.g., “Human Feedback is Not a Gold Standard,” ICLR 2024; “Towards Understanding Sycophancy in Language Models,” ICLR 2024; “Dissecting Human and LLM Preferences,” ACL 2024).

### Questions
See Weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
10

### Rating Number
10

### Confidence
3

### Summary
This paper presents an exciting ongoing effort to collect real-world preference data from people across diverse demographics. Much of this information was not available in other public datasets. Having a dataset with detailed breakdowns of diverse human preferences will be very valuable to the scientific community in the long run.

### Strengths
* The pairwise preference dataset collection is very well executed with adaptive sampling to select the most uncertain pair and quality controls with GPT-4o-mini as the LLM judge.

* The dataset is collected in large-scale with 106K pairwise data from 21K participants across 27 LLMs and will be made publicly available.

* This paper also introduces a novel evaluation framework based on the hierarchical BTD model. This methodology contribution is just as important as the dataset contribution.

### Weaknesses
* It seems that participants are instructed to have at least 3 turns in the study. It will be great to have a deeper analysis/discussion on how the depth of a conversation impact quality/model performance.

* Another dimension that authors could consider is break down by tasks. One may hypothesize that preferences may vary across both tasks and demographics.

* I highly suggest the authors to make this living benchmark and ongoing dataset release.

### Questions
N/A

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper presents a large-scale dataset capturing human preferences over pairs of language model outputs along five evaluation dimensions, stratified by three demographic attributes (age, ethnicity, and politics), covering 27 models. The authors analyze these preferences using a hierarchical Bradley–Terry–Davidson model to learn demographic-specific adjustments and quantify preference variation. Their analysis reveals (1) the overall ranking of models, (2) age as the most divergent demographic axis, (3) variation in rankings across evaluation dimensions, and (4) differences in human tie rates across dimensions, indicating varying decisiveness among annotators.

### Strengths
- Well-motivated and timely research direction. The paper tackles an increasingly critical issue in LLM evaluation: how human preferences differ across demographic groups and qualitative dimensions. It contributes a more nuanced understanding of model performance evaluation.
- Valuable dataset contribution. The dataset could be a useful addition to the community: it is large-scale, multi-turn, demographically stratified, and spans a broad set of 27 LLMs. Such a dataset can serve as an important benchmark for studying model preference heterogeneity and subjective quality dimensions beyond conventional accuracy-based benchmark metrics. 
- Methodological rigor in modeling human preferences: The paper goes beyond simple aggregation of crowd judgments by employing a hierarchical Bradley–Terry–Davidson model. This allows the authors to systematically estimate demographic-specific adjustments and quantify preference variation.

### Weaknesses
- Insufficient alignment between the stated problem and the proposed solution. The paper’s introduction highlights two major issues: (1) the dominance of single-metric evaluation and (2) the neglect of subjectivity. However, the proposed solution, evaluating models on five dimensions across demographic strata, resembles running multiple benchmarks, each focusing on a different aspect. While this is a meaningful improvement, it does not fully capture the “subjectivity” aspect claimed in the introduction, since subjectivity extends beyond demographic factors and fixed evaluation axes.
- Brief and unclear data description. Section 3.2 should provide a more comprehensive account of the dataset. For example, it mentions that “each message sent by the participant was delivered to both models simultaneously” and that “a minimum of 3 conversational turns were required,” but it is unclear how the same message could be reused across multiple turns, given the dependency on previous model responses. The paper also states that participants were free to select topics—what kinds of topics did they typically choose? Furthermore, while gpt-4o-mini was used to flag “low-effort inputs,” the definition and examples of such inputs are not provided. Given that the dataset is the paper’s primary contribution, the absence of even a single example conversation or basic statistics (e.g., number of evaluations per stratum, average conversation length) makes it difficult to grasp the dataset’s structure and richness.
- Overgeneralization and overstatement of findings. The paper emphasizes that “age is the most significant demographic factor driving preference heterogeneity,” but this conclusion holds only among the three studied axes (age, ethnicity, politics). The claim should be scoped accordingly. Similarly, while the paper asserts that “a model’s competitive standing can change dramatically depending on the evaluation lens,” Figure 3 shows high correlation across dimensions, implying that rankings are fairly consistent rather than dramatically shifting. These inconsistencies weaken the strength of the argument.
- Weak overall contribution and presentation quality. The paper feels stretched for a full-length submission. For example, Section 3.4.2 describes an “LLM judge for conversational analysis,” but this analysis is absent from the results. The discussion section repeats much of the results content rather than extending it, which makes it redundant. Some metrics, such as “average rank shifts” (Section 4.2) and “tie percentage” (Section 4.4), are insufficiently defined. These issues collectively reduce the paper’s overall impact and quality.

### Questions
- The compensation rate of £9/hr is described as "recommended" (line 157), yet this falls below the UK minimum wage of £12.21/hr for workers aged 21 and over. Could you clarify how this rate was determined and whether it meets ethical guidelines for research compensation?
- The paper repeatedly refers to DIVERSE as a "living benchmark" with "continuous evaluation" and claims to "continuously add new models and update rankings." Is there an active platform where ongoing evaluation is happening? If not, these claims seem misleading and should be clarified or removed.
- The four evaluation dimensions are stated to derive from a pilot study using factor analysis (Section 3.3), but no details about this study are provided. Given these dimensions are central to your analysis, could you describe the pilot study methodology, sample size, and how factor analysis led to these specific dimensions?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces DIVERSE, a large-scale framework for LLM evaluation that is demographically aware, multi-dimensional, and human-AI interaction focused. The study collects ~110K pairwise comparisons of 27 models from ~21K participants across 22 different demographic groups in the US/UK. Using a hierarchical bayesian bradley-terry model with tie handling and demographic effect adjustments, they found that Google's gemini-2.5-pro achieved the top performance in most dimensional analyses. Lastly the authors emphasized that the context aware design of LLM evaluation is highly essential to accurately measure a model's capacity with user needs.

### Strengths
- The presentation of all methodology and results is very clear. 
- A large-scale data collection with a thoroughly curated design of DIVERSE (many participants, many data points, and multi-turn interaction logs).

### Weaknesses
- Allowing participants to choose their own topic of conversation can enhance validity of experiment setup (data collection especially), but this is likely to inject some heterogeneity in the task type and difficulty that can affect the pairwise evaluation settings. Although the paper collects LLM-as-judge annotations over several aspects, these variables are not put to the hierarchical TBD model. This can risk that the model assumes all conversations are equally treated, even though some task contexts may inherently require multi-step reasoning of problem solving (than just freely writing creative text). Ignoring this may not reveal quality differences between LLMs in a pairwise setup. If a LLM happens to be matched frequently on tasks that the model is good at, the estimated skill parameter of this LLM may be inflated independent of true capacity it has. The current manuscript does not provide those level of analysis. 

- The hierarchical BTD model treats age, ethnicity, and political affiliation as additive effects (scaled by 1/ $\sqrt{3}$). This assumes that each demographic axe contributes independently to preference. However, human preferennce often involves interaction effects. For example, political tone or language can differ between younger and older generations (e.g., younger conservatives vs. older conservatives). The political identity of an individual may differ across racial and ethnic groups. The current manuscript does not consider this interactions of several demographic dimensions in the analysis.

### Questions
- Could you provide a breakdown of task types and complexity (from the collected LLM-judge annotations) that each model was exposed to? Do win-rates change substantially when comparing models within the same task type (e.g., reasoning tasks like math, or causal conversation?) Is the overall leaderboard remaining stable when grouping by task complexity or task topic? Some stratification analysis would be highly valuable to strengthen the paper. 

- Did you check whether preference patterns differ across combinations of demographics, such as age x politics? A simple heatmap or table showing win/tie rates for age x politics groups can help assess whether those interaction effects are present.

### Soundness
2

### Presentation
3

### Contribution
4

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes DIVERSE, a large-scale, demographically aware framework for evaluating large language models through human preference data rather than technical benchmarks.

Strengths

- Dataset scale:

The study is based on 106,760 pairwise comparisons from 21,352 participants across 27 language models, which provides substantial empirical depth.

- Methodological rigor:

The hierarchical Bayesian BTD model is statistically sound and appropriate for modeling heterogeneous human preferences.

- Insightful analysis:

The examination of demographic heterogeneity and metric discriminability yields novel and meaningful findings for human-centered LLM evaluation.

Weakness

- Data Availability

The paper briefly mentions data and framework availability in the conclusion but does not provide access at review time. Given the paper’s emphasis on dataset, it would be important to release at least a partial dataset or representative samples during the review process. If the paper is accepted and made public, the authors should clearly commit to releasing the full dataset for research use.


- Representativeness

While the paper makes a valuable contribution, its claims about representativeness and the mitigation of sampling bias are overstated. The participant pool includes only users from the US and UK—two English-speaking, Western countries, failing to capture the broader global, multilingual, and cultural diversity of LLM users. The claim of being “stratified across 22 demographic groups” creates an impression of global inclusiveness, though the scope is in fact regionally constrained. The authors should moderate such claims and clearly state that their findings are representative only within certain contexts. A more cautious framing would enhance the paper.

### Strengths
- Dataset scale:

The study is based on 106,760 pairwise comparisons from 21,352 participants across 27 language models, which provides substantial empirical depth.

- Methodological rigor:

The hierarchical Bayesian BTD model is statistically sound and appropriate for modeling heterogeneous human preferences.

- Insightful analysis:

The examination of demographic heterogeneity and metric discriminability yields novel and meaningful findings for human-centered LLM evaluation.

### Weaknesses
- Data Availability

The paper briefly mentions data and framework availability in the conclusion but does not provide access at review time. Given the paper’s emphasis on dataset, it would be important to release at least a partial dataset or representative samples during the review process. If the paper is accepted and made public, the authors should clearly commit to releasing the full dataset for research use.


- Representativeness

While the paper makes a valuable contribution, its claims about representativeness and the mitigation of sampling bias are overstated. The participant pool includes only users from the US and UK—two English-speaking, Western countries, failing to capture the broader global, multilingual, and cultural diversity of LLM users. The claim of being “stratified across 22 demographic groups” creates an impression of global inclusiveness, though the scope is in fact regionally constrained. The authors should moderate such claims and clearly state that their findings are representative only within certain contexts. A more cautious framing would enhance the paper.

### Questions
1. Dataset release: Can the authors provide example cases or partial data during review to improve transparency? Will the complete dataset be released for research use if the paper is accepted?
2. Demographic scope: Do the authors plan to extend data collection beyond the US and UK to include more diverse populations in future iterations of DIVERSE? Otherwise, it would be better to limit the contexts in the introduction.

### Soundness
2

### Presentation
3

### Contribution
3

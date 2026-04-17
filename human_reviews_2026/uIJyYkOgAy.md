# Can Large Language Models Match the Conclusions of Systematic Reviews?

- Decision: Accept (Poster)
- Scores: 6, 4, 4

## Abstract
Systematic reviews (SR), in which experts summarize and analyze evidence across individual studies to provide insights on a specialized topic, are a cornerstone for evidence-based clinical decision-making, research, and policy. Given the exponential growth of scientific articles, there is growing interest in using large language models (LLMs) to automate SR generation. However, the ability of LLMs to critically assess evidence and reason across multiple documents to provide recommendations at the same proficiency as domain experts remains poorly characterized. We therefore ask: **Can LLMs match the conclusions of systematic reviews written by clinical experts when given access to the same studies?** To explore this question, we present MedEvidence, a benchmark pairing findings from 100 medical SRs with the studies they are based on. We benchmark 25 LLMs on MedEvidence, including reasoning, non-reasoning, medical specialists, and models across varying sizes (from 7B-700B). Through our systematic evaluation, we find that reasoning does not necessarily improve performance, larger models do not consistently yield greater gains, and knowledge-based fine-tuning degrades accuracy on MedEvidence. Instead, most models exhibit similar behavior: performance tends to degrade as token length increases, their responses show overconfidence, and, contrary to human experts, all models show a lack of scientific skepticism toward low-quality findings. These results suggest that more work is still required before LLMs can reliably match the observations from expert-conducted SRs, even though these systems are already deployed and being used by clinicians.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduced MedEvidence, a benchmark based on human-annotated systematic reviews, to test the LLMs’ ability in question answering based on medical context. Various LLMs are then evaluated and compared on the benchmark.

### Strengths
* All test cases are manually curated based on existing Cochran meta-analyses.
* A large number of LMs are evaluated on the benchmark.

### Weaknesses
* The soundness of the benchmark is heavily based on the annotation quality, but there is no discussion about the annotators’ background.
* The only task provided by MedEvidence so far is a multiple-choice/classification task, as the model has to answer one of five given treatment outcome effects. Please see the questions for details.
* There are no numerical experimental statistics in the main paper. All results are presented in plots.

### Questions
* Line 252-257: The categorization is directly based on DeepSeek, which is not rigorous enough. Also, in the remaining part of the paper, the authors did not utilize the categorization in the evaluation/analysis.
* Figure 6: It is hard to distinguish the medically finetuned models from the others via the thickness of the margin. Making notes on the y-axis labels can be helpful (for example, adding * for medical models)
* The results include a few surprising results. For example, reasoning models, larger models, and domain-adapted models do not improve performance. Case studies discussing potential causes for these results can be helpful.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This work presents two contributions: (1) a dataset of systematic reviews from Cochrane, and (2) an assessment of LLM ability to generate a conclusion over the SR evidence strength.

### Strengths
This is a well written paper with a scoped contribution that explicitly tests modern LLMs (incl. reasoning models) ability to synthesize evidence across systematic reviews; the analysis is solid and has high relevance for clinical settings where these models may be deployed.

### Weaknesses
My main concern with this work is the lack of adequate comparison with prior work. It’s unclear whether the dataset itself is a novel contribution; there is a lack of interaction with prior work on evaluating LLM evidence synthesis and existing cleaned SR datasets:  

[1] Three datasets built similarly: TrialReviewBench (https://arxiv.org/html/2407.00631v2), https://arxiv.org/abs/2008.11293, and https://aclanthology.org/2024.acl-srw.42/. 

[2] There is some prior work that finds similar conclusions regarding overconfidence in LLM responses, and lack of ability to synthesize evidence (https://pmc.ncbi.nlm.nih.gov/articles/PMC11613457/; https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2838106) 

I do believe this paper has a good set of experiments over models that benefits the community, especially since it’s clear that these modern models still have the same issues as prior iterations. I am happy to increase my score given the better grounding against prior work and datasets, and justification for why new labels/questions were generated for this task (see questions below).

### Questions
1. There is also a difference between general systematic reviews and randomized control trials (RCTs). Since this is sourced from Cochrane, I assume they are all RCTs? Could you confirm this? 
    (a) Often there are established research questions that are specific and provided in these reviews. Why are these questions not being used directly? 

2. For the dataset generation, there does not appear to be any assessment of annotator agreement or the quality of the question conversions, apart from the source concordance analysis. However, this measure alone is insufficient, it primarily reflects whether a question can be answered using a single source, which does not capture true evidence synthesis. How is quality is assessed among annotators? 

3. Could you clarify the motivation for mapping to *new* labels for the model to synthesize? I am concerned that *“X increases Y”* or *“X may reduce Y”* don’t imply “higher” or “lower” labels, since direction alone doesn’t indicate desirability or certainty of effect.

4. Could you clarify the difference and novelty introduced by your dataset relative to the listed prior works? 

(Clarity) L246 typo: should be “use an LLM” not “use an LLMs”

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces MedEvidence, a benchmark of closed-form QA items distilled from Cochrane systematic reviews, pairing each conclusion with the same source studies experts used. LLMs (reasoning/non-reasoning; medical-finetuned/generalist) are evaluated; key findings: reasoning/scale/medical-finetuning don’t reliably help, performance drops with long inputs, models are overconfident and insufficiently skeptical of low-quality evidence. Even frontier models trail a time-constrained expert baseline.

### Strengths
S1) This is a well-posed task and this paper makes targeted contributions. The paper removes retrieval and long-summary grading by converting conclusions into closed-QA and evaluating exact match answers. 

S2) Evaluation is reasonbly transparent with limited uncertainty. Metrics (e.g. per-class recall, accuracy, evidence uncertainty w/ source concordance) reliably support key findings. Work is methodically sound. 

S3) Clear empirical takeaways -- I am largely satisfied with their takeaways of frontier models underperforming time-constrained experts, etc.

### Weaknesses
W1) I think perhaps a reasonable weakness/questions here is re conceptual novelty and whether this paper is suited for ICLR. Essentially authors do a dataset+evaluation work with closed-class answers. While useful, it advances prior factuality/evidence-reasoning datasets incrementally (i.e. Table 1) and centers on mapping mapping SR conclusions to a QA rather than introducing new modeling/eval methods. 

W2) LLM-derived source concordance. While I think this is a reasonable thing to do for eval/metric purposes, I think there is a potential circularity/model bias element to it. And we're introducing those limitations into a key explanatory variable. 

W3) Evaluation pipeline (maybe) order-sensitive. When context overflows, and answers are refined over seqs of artile chunks; I am not sure how randomization effects play into the analysis of main text.

### Questions
Q1) On source concordance -- Beyond using DeepSeek, did you validate concordance against human judgments on a subset to ensure this variable itself isn't model induced? 

Q2) For time-constrained expert line (Fig 4a), how many experts participated, how were specialities matched to questions, and what agreement (e.g., κ/ICC) did you observe?

### Soundness
3

### Presentation
4

### Contribution
2

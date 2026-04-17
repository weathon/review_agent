# Aligning Large Language Model Behavior with Human Citation Preferences

- Decision: Reject
- Scores: 2, 4, 2, 4, 4

## Abstract
Most services built on powerful large-scale language models (LLMs) add citations to their output to enhance credibility. Recent research has paid increasing attention to the question of what reference documents to link to outputs. However, how LLMs recognize cite-worthiness and how this process should be controlled remains insufficiently explored. In this study, we focus on what kinds of content LLMs currently tend to cite and how well that behavior aligns with human preferences. We construct a dataset to characterize the relationship between human citation preferences and LLM behavior. Web-derived texts are categorized into eight citation-motivation types, and pairwise citation preferences are exhaustively evaluated across all type combinations to capture fine-grained contrasts. Our results show that humans most frequently seek citations for medical text, and stronger models display a similar tendency. We also find that current models are as much as 27% more likely than humans to add citations to text that is explicitly marked as needing citations on sources such as Wikipedia, and this overemphasis reduces alignment accuracy. Conversely, models systematically underselect numeric sentences (by -22.6% relative to humans) and sentences containing personal names (by -20.1%), categories for which humans typically demand citations. Furthermore, experiments with fine-tuning and Direct Preference Optimization (DPO) demonstrate that model behavior can be calibrated to better match human citation preferences. We expect this study to provide a foundation for more fine-grained investigations into LLM citation preferences. Our dataset and code will be released upon publication.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper studies how LLMs decide what to cite and how well this aligns with human preferences. The authors create a dataset categorizing texts into eight citation-motivation types and analyze pairwise citation choices. Results show LLMs over-cite obvious citation markers and under-cite numeric or personal-name sentences, diverging from human behavior. Further clues show that Fine-tuning and DPO can better align models with human citation preferences.

### Strengths
The authors create a dataset categorizing texts into eight citation-motivation types and analyze pairwise citation choices. Results show LLMs over-cite obvious citation markers and under-cite numeric or personal-name sentences, diverging from human behavior. The key strength of the paper is its systematic dataset and analysis, providing a detailed, fine-grained understanding of LLM citation behavior that can guide future improvements.

### Weaknesses
1. Missing details and unclear motivation: The paper lacks sufficient detail about the dataset construction process. It remains unclear why the authors chose to create this dataset, what specific research questions or gaps it aims to address, and how the base data sources were selected.

2. Uncertain data quality: The quality and reliability of the dataset are not properly evaluated. No assessment, validation, or inter-annotator agreement analysis is provided to ensure the soundness of the data.

3. Limited novelty: The paper does not present a new methodological contribution. It reads more like a technical report or dataset summary rather than a research paper with conceptual or algorithmic innovation.

4. Poor organization and writing issues: The motivation, contributions, and overall structure of the paper are unclear. The manuscript contains several typographical and formatting errors, which further reduce readability and weaken the presentation.

### Questions
1. It seems unusual that the authors recruited 402 participants to annotate only 3,000 sentence pairs. Such a large number of annotators could undermine the consistency of the dataset and introduce significant bias.

2. The problem setup in Section 3 is somewhat confusing. What is the underlying motivation for this task, and how exactly is it connected to citation preference?

3. The authors note that "some models refused to answer certain items due to safety or political restrictions." This raises ethical concerns about the dataset. Even though it is built from open-source data, it may still contain politically sensitive questions. Why were potentially unsafe items not removed?

4. Line 405 states, "Furthermore, our experiments demonstrate that LLM citation preferences can be controlled via DPO." This claim seems premature based solely on Table 6. Additional analysis is needed to support such a conclusion.

Typos:
Line 215, 305, broken links.

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper investigates how LLMs decide what content should receive citations and how well this aligns with human preferences. The authors construct a dataset of 6,000 Wikipedia sentences categorized into 8 types based on quality labels (Missing Information, Sic, Doubt, Vague, POV, Medical Content, Jargon, Unclear), with pairwise human annotations indicating which sentences in each pair should receive citations. They evaluate 11 LLMs (both open and closed) on this task and find that models only weakly align with human preferences (~60% agreement), with systematic biases: models overselect sentences explicitly marked "Citation needed" (up to +27.4%) while underselecting numeric sentences (-22.6%) and sentences with person names (-20.1%). The authors demonstrate that Direct Preference Optimization (DPO) can improve alignment by up to 11.8%, while standard fine-tuning fails. The work provides evidence that training data strongly influence citation behavior and that this influence can be controlled through preference-based training.

### Strengths
(A) This paper addresses a critical but underexplored aspect of LLM citation behavior - determining what content needs citations rather than which documents to link. This is particularly important because modern LLM services increasingly let the model itself control citation decisions in agentic workflows. The distinction between cite-worthiness (content-conditioned importance) and citation recommendation (document matching) is well-motivated and fills a genuine gap in the literature, as most prior work focuses on attribution and retrieval rather than the fundamental question of what merits verification.

(B) The dataset construction is thoughtful and rigorous. Using Wikipedia's inline templates provides a reliable signal from experienced editors, and the reorganization into 8 meaningful categories with balanced pairwise comparisons (28 category pairs) enables fine-grained analysis. The quality control measures (duplicate pairs for consistency checks, removal of malformed sentences) and annotation setup with 402 US-based participants is reasonable. 

(C) he paper provides compelling evidence of how training data shapes citation behavior through three well-chosen analysis dimensions: (1) sentences with "Citation needed" tags show models are over-influenced by Wikipedia markup (+19.5% to +27.4% vs humans), (2) numeric sentences reveal systematic underselection (-22.6%) despite human preference for citing quantitative claims, and (3) person names show similar underselection (-20.1%) despite established norms around biographical verification. 

(D) The experimental validation that DPO (but not standard fine-tuning) can improve alignment by 5.76% on average (11.8% for smaller models) provides actionable evidence that citation preferences can be learned and controlled. The negative results for fine-tuning corroborate prior findings about its limitations for knowledge injection, while the DPO success suggests a path forward for deployment.

### Weaknesses
(A) Minor: The final dataset contains only 2,596 pairwise comparisons, which becomes just 1,206 training pairs after splitting. This is quite small for training modern LLMs and limits the reliability of conclusions. More critically, the paper provides no inter-annotator agreement metrics (Cohen's kappa, Fleiss' kappa, or even pairwise agreement rates), making it impossible to assess whether the annotation task is well-defined and whether human preferences are consistent. The paper mentions removing inconsistent annotations but doesn't report what percentage of annotations were inconsistent, raising concerns about task difficulty and label reliability.

(B) The paper lacks essential methodological details that undermine reproducibility and credibility: (1) no prompts are shown for LLM evaluation, despite prompt sensitivity being well-known, (2) no statistical significance tests are reported - differences like 62.7% vs 61.2% could be within noise, (3) no confidence intervals or standard errors across evaluation runs, (4) no analysis of potential train/test contamination given that models were pretrained on Wikipedia and the dataset uses Wikipedia sentences. 

(C) 
The claim that models overselecting "Citation needed" sentences demonstrates "training data bias" (rather than appropriate learned behavior) assumes human annotations are ground truth without justification. However, Wikipedia editors explicitly marked these sentences as needing citations, so models selecting them more frequently could indicate they correctly learned Wikipedia's citation standards - which might be more rigorous than the crowdworkers' judgments. The paper doesn't validate that crowdworker preferences align with Wikipedia editor expertise or provide evidence that the model behavior is genuinely problematic rather than reflecting legitimate editorial standards that prioritize verifiability.

(D) The artificial balancing of category pairs (28 combinations with ~107 examples each) doesn't reflect natural distributions of content types requiring citations. Additionally, the eight categories seem somewhat arbitrary - the reorganization from 153 labels into 8 groups lacks clear principled justification, and some categories appear to overlap conceptually (e.g., "Vague" vs "Unclear," or why "Medical Content" is separate from domain-specific "Jargon").

### Questions
(A) What was the inter-annotator agreement rate (Cohen's kappa or Fleiss' kappa) for the pairwise comparisons? How many annotations were flagged as inconsistent and removed? Given that Table 2 shows many category pairs with win rates close to 50% (e.g., Info vs Sic: 50.5%, Doubt vs Vague: 48.9%), does this suggest the task may be ambiguous or poorly defined for certain category pairs? Could you provide evidence that crowdworkers' citation preferences align with those of experienced Wikipedia editors, given that Wikipedia's editorial standards may be more rigorous?

(B) Did you check whether the Wikipedia sentences in your dataset appear in the pretraining corpora of the evaluated models? This is particularly important since you argue models are influenced by seeing "Citation needed" tags during training - but this conclusion requires that models actually encountered these specific sentences or similar patterns. Additionally, could you provide the exact prompts used for LLM evaluation and report statistical significance tests (with confidence intervals) for the performance differences between models? Without these details, it's difficult to assess whether observed differences are meaningful or within experimental noise.

(C) our analysis shows models overselect "Citation needed" sentences (+19.5% to +27.4%) and underselect numeric/person-name sentences. However, couldn't the former indicate that models correctly learned Wikipedia's rigorous citation standards (which crowdworkers may underestimate), while the latter might reflect that numbers and names often appear in contexts where surrounding text provides attribution? Have you conducted ablation studies or controlled experiments to isolate whether these patterns are truly "biases" to correct, or whether they reflect reasonable learned heuristics? What evidence supports the assumption that crowdworker preferences should be preferred over model behavior that might better align with Wikipedia's actual editorial standards?

(D) How do you expect these findings to generalize beyond Wikipedia-style encyclopedic writing to other domains like scientific papers, journalism, or conversational AI responses where citation norms differ substantially? The category taxonomy seems Wikipedia-specific (e.g., "POV" reflects Wikipedia's neutrality policy) - have you considered how citation preferences might differ for medical literature (where citation standards are extremely rigorous), news articles (where attribution norms differ), or casual information-seeking (where users may prefer fewer citations for readability)? Could you discuss how the artificial balancing of category pairs in your dataset might affect model performance on natural distributions of content?

(E) The paper references missing appendices (e.g., "Appendix ??") multiple times for critical details about model training, hyperparameters, and DPO implementation. Could you provide complete details about: (1) DPO hyperparameters (learning rate, beta, number of epochs), (2) why standard fine-tuning failed so dramatically (Table 5 shows -14.4% for Llama 3B) - is this a hyperparameter issue or fundamental limitation, (3) how you constructed the preference pairs for DPO from pairwise human judgments, and (4) computational costs and training time? Additionally, given the small training set (1,206 pairs), did you observe overfitting, and how did you validate the models aren't simply memorizing the training data?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates what kind of content Large Language Models (LLMs) tend to cite and how well this behavior aligns with human preferences for "cite-worthiness." While many LLM services add citations to enhance credibility, this study explores a misalignment between LLM and human citation preferences.

The author created a new dataset by categorizing 6,000 Wikipedia sentences into eight "citation-motivation" types (e.g., Medical Content, Doubt, Vague, Unclear). They then conducted a large-scale study to capture human citation preferences by asking participants to choose which of two sentences from different categories most needed a citation.

At the end, they demonstrate a potential solution with DPO to train the models on human preference data resulted in an average 5.76% improvement in alignment, demonstrating that LLM citation behavior can be effectively calibrated to better match user needs.

### Strengths
- The paper clearly articulates the problem it aims to solve the gap between LLM citation behavior and human "cite-worthiness" preferences and outlines a straightforward methodology to address it
- The conclusions are supported. The paper evaluates 11 different models to identify the problem broadly, then trains and optimizes 5 open-source models to test its proposed solution
- A core contribution is the large, human-annotated preference dataset. The successful results from DPO (Direct Preference Optimization) training validate the feasibility of this approach, proving that model citation preferences can be effectively improved

### Weaknesses
- The writing of the paper need to be further crafted and polished. In line 214 and 205, there are missing appendix references. Redundancy is another issues exists, for example "fine-tuning hamrs alignment" are brought out repeatedly. I strongly suggest the author organise the logical flow and go through the paper carefully
- In the data collection phase, the description of the noise filtering process is insufficient. Additional information is required regarding "which noise filtering method was applied" and "how many instances/sentences remained at each stage of the filtering process."
- It is appreciable that the author has gathered a large group of participants for large-scale annotation work. However, this raises a concern regarding how to control the quality of human annotation, and the current paper may lack such a description
- The author categorizes the labels into eight categories, and this classification seems to lack justification for why the proposed definition is appropriate for decomposing human preferences

### Questions
Q1: Other than the three defined reference factors, what else could be essential for quantitatively assessing citation preferences?
Q2: What could be the underlying reason that the fine-tuning harm the citation alignment with human
Q3: When collecting the human annotation from such large group, how to make sure the human preferences are aligned within the group?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper studies how large language models (LLMs) decide when to attach citations and how well those decisions line up with human expectations. Towards this goal, the authors obtained 6,000 sentences from Wikipedia that contain inline quality templates (e.g., “citation needed,” “clarification needed,” “medical citation needed”) and turned them into a preference dataset with human annotations. The authors evaluated most open-source large models on the curated preference dataset and used a train split to do SFT / DPO training.

### Strengths
1. The paper is centered “given two statements, which one most needs a citation?” which isolates cite-worthiness as a preference judgment. This is a novel task and different from most prior work on attribution, which tends to assume you already know a claim needs support and then focuses on finding or attaching the right source

2. The dataset curation is thoughtful and high quality. The use of Wikipedia inline templates is a very clever design, and this is accompanied by a large scale human annotation process with 400+ annotators. The result is a dataset that is both high volume and high quality, with interpretable labels.

### Weaknesses
My main concern is how useful this alignment goal itself is. The supervision signal is strictly relative (“which of two sentences needs a citation more?”). Real assistants, however, must make absolute, independent decisions about each span (“does this claim require a citation at all?”). Because the dataset never captures ‘both’, ‘neither’, or graded severity, it’s unclear whether models trained on this signal will learn a properly calibrated trigger for citation in deployment. Therefore even the human annotators may not often agree with each other (there's no inter-annotator agreement statistic), except for on clearly more important issues such as medical documents. There is a fundamental ambiguity with respect to the preference goal studied here.

The alignment experiments optimize directly on pairwise citation-preference agreement and then re-evaluate on that same objective. We don’t see any downstream, task-level validation (e.g., does the tuned model actually insert more citations for medically sensitive claims and fewer for benign filler, without just spamming citations everywhere?). Since the paper’s motivation is user trust and cognitive load in real assistant answers, an end-to-end generation study would be important to show that DPO improves practical citation behavior rather than overfitting to this pairwise benchmark.

### Questions
1. You describe performing annotator QC and filtering pairs for consistency before arriving at 2,596 comparisons. Could you report human–human agreement statistics on the unfiltered pool (e.g., raw pairwise agreement rate, κ / α), and also after filtering?

2. The best models achieve ~60% agreement with human preferences. How does this compare to (i) average agreement of a held-out group of annotators with the majority choice and (ii) majority-vs-majority agreement across annotator splits?

3. Can you share which of your eight categories have the most annotator disagreement? For example, are categories like “Vague / POV / Unclear” substantially noisier than “Medical Content”?

4. Can you show the performance of finetuned models on this preference dataset on some downstream citation tasks? 

5. There are some frequently "preferred" topics such as medical claims. After DPO, do models become more likely to attach a citation to, e.g., medical claims? Do those citations actually support the claim any better?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies alignment between LLM citation preferences and human preferences, constructing a dataset with 8 categories from Wikipedia. While the research direction is novel, the single-source dataset (Wikipedia only) \textbf{severely limits the generalizability and validity of conclusions.}

They found that current models diverge significantly from human preferences—models over-select sentences with [Citation needed] tags (27% higher than humans) but systematically under-select sentences with numbers and person names (22.6% and 20.1% lower, respectively). 

Finally, they show that DPO training can improve alignment with human preferences.

### Strengths
**1. Novel and Timely Research Direction**  
This paper focuses on an under-explored question that what content in LLM outputs deserves citations. Existing research centers on RAG retrieval, citation validation, or not cite-worthiness itself. Closest prior work (CiteWorth ACL 2021, Redi et al. 2019) is limited to narrow domains and excludes LLM behavior.  

Its contributions fill this gap: (1) first use of preference learning for cite-worthiness; (2) cross-category comparison (8 types); (3) analysis of LLM-human alignment; (4) discovery of training data effects; (5) DPO demonstration for preference alignment.


**2. Well-Motivated Methodological Design**  
The pairwise comparison framework captures relative preferences, avoiding absolute rating subjectivity. The 8-category taxonomy covers citation motivations, with balanced sampling across 28 pairs. Using Wikipedia templates is reasonable (due to editorial standards), though generalizability needs consideration.


**3. Comprehensive Model Evaluation Across Scales**  
Evaluation includes 11 models (1B-70B+), spanning open-source (Llama, Mistral, DeepSeek) and closed-source (GPT-5, Claude, Gemini). Clear scale-performance correlation (Llama: 50.0%→56.3%→61.6%) shows how cite-worthiness capabilities emerge. Honest negative result reporting (Llama 1B at baseline) boosts credibility.


**4. Interesting Empirical Findings**  
- Models over-select "Citation needed" sentences (up to +27.4%), revealing training data surface pattern influence.  
- Consistent under-selection of numeric (-22.6%) and person name (-20.1%) sentences identifies systematic gaps.  
- Medical content has higher agreement, suggesting domain-specific pretraining patterns.


**5. Practical Exploration of Alignment Methods**  
A systematic comparison shows standard fine-tuning degrades performance, while DPO improves it by ~5.76%—aligning with recent preference optimization findings. Strong gains for small models (Llama 1B: +11.8%, 3B: +9.1%) add value for resource-constrained settings.


**6. Clear Presentation**  
The paper is well-organized, with clear problem formulation and transparent reporting of results. Tables communicate findings effectively, and sufficient implementation details are provided. Committing to data/code release benefits future research.

### Weaknesses
The paper claims to study 'alignment between LLM and human citation preferences,' but its dataset has critical limitations that undermine this core goal:  

1. **Single-domain bias**  
All 6,000 sentences are sourced from Wikipedia, a specialized text type with unique editorial standards (e.g., prioritizing verifiability over practical utility). This differs fundamentally from ordinary users’ citation needs—for example, a user seeking 'insomnia medication advice' has distinct expectations vs. reading Wikipedia’s 'insomnia' entry. The study thus measures 'Wikipedia editor preferences' rather than general 'human preferences,' calling the validity of its research question into question.  

2. **Circular reasoning in key findings**  
A central claim (models over-select 'Citation needed' tags) is attributed to training data bias. However, annotation data is reorganized from Wikipedia inline templates (Table 1), creating tautology: Wikipedia labels define ground truth → models trained on Wikipedia are tested → models prefer Wikipedia labels → conclusion blames 'training bias.' This finding risks being a dataset artifact rather than a meaningful scientific discovery.  

3. **Missing high-stakes real-world scenarios**  
Critical application scenarios requiring citation control are absent, such as:  
- High-stakes medical advice ('take XX medication') vs. medical knowledge statements ('XX drug mechanism is...')  
- Legal opinions ('you have rights under XX law') vs. statute descriptions  
Citation logic in these scenarios differs sharply from Wikipedia, yet the paper ignores them entirely.  

4. **Inadequate sample size**  
After cleaning, only 2,596 pairs remain (1,206 training, 100 validation, 1,288 test), with ~43 training pairs per category combination. This is too small to support learning 'general citation preferences' and instead risks model memorization of domain-specific patterns.  

**Severity**: This is a fundamental flaw. The paper’s core claim ('LLMs misalign with human preferences') is compromised if it measures domain-specific preferences, relies on circular reasoning, and excludes critical scenarios. It reads more like a 'Wikipedia citation preference study' than a general analysis.

5. **Typo**  
There are two '=Appendix ??' which are not correctly linked to the reference in the paper.

### Questions
**Question**
1. The study uses only Wikipedia data. Do you plan to add cross-domain datasets to test generalization? If not, how do you justify claiming the findings reflect 'human preferences' rather than Wikipedia-specific ones?  
2. Concerning ablation studies: The paper does not test how prompts, training subset sizes, or sampling strategies affect results. Do you have unpublished ablation data to clarify these variables’ impact?  


**Suggestions**

1. Add at least 2-3 datasets from other domains in humanities and social sciences, or conduct cross-domain validation demonstrating model performance across different domains. Maybe also increase the sample size accordingly.

If cross-domain data cannot be supplemented, I would suggest:
- Consider modifying the title to explicitly specify the research domain, such as 'Aligning LLM Behavior with Human Citation Preferences in Wikipedia Context' or similar.
- Clearly state in the Abstract and Introduction that this is a domain-specific study, and discuss generalization limitations in detail in the Conclusion and Limitations sections.
- Either remove the findings regarding 'Citation needed' (due to circular reasoning) or provide a reinterpretation of these results.

2. And, more ablation studies (different prompts, training data sizes, etc.) should be discussed.

### Soundness
3

### Presentation
3

### Contribution
3

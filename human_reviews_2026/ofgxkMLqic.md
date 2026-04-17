# Skywork-Reward-V2: Scaling Preference Data Curation via Human-AI Synergy

- Decision: Accept (Poster)
- Scores: 8, 6, 2, 6

## Abstract
Despite the critical role of reward models (RMs) in Reinforcement Learning from Human Feedback (RLHF), current state-of-the-art open RMs perform poorly on most existing evaluation benchmarks, failing to capture the spectrum of nuanced and sophisticated human preferences. Even approaches incorporating advanced training techniques have failed to yield meaningful performance improvements. We hypothesize that this brittleness stems primarily from limitations in preference datasets, which are often narrowly scoped, synthetically labeled, or lack rigorous quality control. To address these challenges, we present a large-scale preference dataset comprising 40 million preference pairs, named SynPref-40M. To enable data curation at scale, we design a human-AI synergistic two-stage pipeline that leverages the complementary strengths of human annotation quality and AI scalability. In this pipeline, humans provide verified annotations, while Large Language Models~(LLMs) perform automatic curation based on human guidance. Training on this preference mixture, we introduce Skywork-Reward-V2, a suite of eight reward models ranging from 0.6B to 8B parameters, trained on a carefully curated subset of 26 million preference pairs from SynPref-40M. We demonstrate that Skywork-Reward-V2 is versatile across a wide range of capabilities, including alignment with human preferences, objective correctness, safety, resistance to stylistic biases, and best-of-N scaling. These reward models achieve state-of-the-art performance across seven major reward model benchmarks, outperform the latest paradigm of generative reward models, and demonstrate strong downstream performance. Ablation studies confirm that the effectiveness of our approach stems not only from data scale but also from high-quality curation. The Skywork-Reward-V2 series represents substantial progress in open reward models, highlighting the untapped potential of existing preference datasets and demonstrating how human-AI curation synergy can unlock significantly higher data quality.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduces SynergyPref-40M, a preference dataset of 40M pairs (with 26M curated) and a two-stage human–AI curation pipeline for training Bradley–Terry reward models (0.6B–8B params). Stage 1 uses human-verified “gold” labels plus LLM-assisted “silver” labels with error-driven adaptive retrieval; Stage 2 scales automatic curation by enforcing consistency with a gold reward model and the current best RM. Trained RMs achieve state-of-the-art results on seven benchmarks (RewardBench v1/v2, PPE Preference/Correctness, RMB, RM-Bench, JudgeBench), with strong best-of-N scaling, resistance to style bias, and competitive correctness; ablations attribute gains to curation quality in addition to scale.

### Strengths
1. Two-stage, human-guided curation with adaptive retrieval is thoughtfully engineered and demonstrates measurable impact over raw LLM curation; Stage-2 consistency filtering is a pragmatic way to scale beyond the human budget.
2. Seven strong benchmarks, including RewardBench v2 and JudgeBench, with detailed breakdowns (style bias on RM-Bench, BoN scaling on PPE Correctness). Results are consistently SOTA for open RMs.
3. Training details are explicit; the authors plan to release data/models and provide scripts for reproduction.

### Weaknesses
1. While authors recognize raw LLM curation is weak and mitigate with human guidance, final labels in large parts of the mixture still depend on LLM aggregation; a more thorough error analysis comparing human vs LLM-judged segments would strengthen soundness claims.
2. The “in-the-wild” pool composition, de-duplication against evaluation sets, and licensing/PII filters are not described in depth.
3. Stage-2 selection relies on agreement with a gold RM trained on human data; without stringent de-duplication and contamination checks, this risks overfitting to the gold RM’s inductive biases and to benchmarks that overlap with the mined pool. Stronger leakage analysis (e.g., prompt/response near-duplicates) might be needed.

### Questions
1. How did you ensure that mined preference pairs (and their paraphrases) do not overlap with RewardBench v2, JudgeBench, etc.?
2. Can you report how often flips contradict original human labels vs synthetic labels, and whether flipped pairs increase spurious correlations (e.g., penalizing certain styles)?
3. Any results on downstream RLHF or human studies confirming that the SOTA RM ranking translates into policy improvements and user-perceived quality?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper presents a large-scale preference dataset (SynergyPref-40M) and a series of reward models trained on the dataset. It describes an iterative data curation pipeline merging human and LLM annotations. They show that trained reward models outperform existing models even at smaller model sizes across a wide range of benchmarks. 

Overall, this is a solid paper overall, with a clear contribution. The results are very impressive, and given that the dataset and reward models are released as promised, it is a very nice resource for other researchers. The reward model results and ablations are nice. However, the paper lacks more details on the dataset composition, choice of prompts, and generations. The methodology itself is described at a relatively abstract level. The description of evaluation protocols, instructions given to annotators, etc., is limited. Also, no actual examples from the dataset are provided. I expect some improved transparency for a revised version, to ensure reproducibility of results.

### Strengths
+ I assume that the dataset itself will be made available as promised. Obviously, a dataset at this scale and level of curation is a very strong contribution to the field
+ The reported benchmark scores are very impressive, clearly outperforming existing reward models
+ The evaluation of the trained reward models is thorough and very comprehensive
+ I really appreciate the provided ablation studies. Evaluating the trade-offs of data curation and dataset scaling is insightful. In particular, sections 4.3 and 4.4 make a strong case for continued human data curation
+ The paper presents a nice discussion and comparison of recent reward modeling benchmarks

### Weaknesses
- The actual dataset generation/curation process is missing details, in particular: What is the composition of the dataset (i.e., which categories does it contain? How are the prompts+responses formulated? What is the origin of prompts + responses? Does it contain multi-lingual samples?) Also, there are just no samples or insights given about the contents of the collected dataset. I think maybe it would also be possible to provide some more details in the main paper, important information like annotator information is only given in the appendices, a summary in the main text would be appreciated
- For many of the lessons learned, I would appreciate more detailed insights. It was noted that access to tools was crucial for high-quality annotations. How did the annotators actually use them? What kind of prompts are most relevant for?

### Questions
- I do not follow the conclusion from Figure 1 (or at least I feel it’s too strongly worded). RewardBench still shows positive (and for most benchmarks) decently strong correlation. For example, in Figure 1-Left, I only see a single derivative model that outperforms RewardBench but is noticeably worse on the other benchmarks, so it seems more like an outlier. The general point that there is over-optimization on the reward bench is a totally valid concern, and I believe this to be the case, but i do not draw this conclusion from Figure 1. Could you give your oppinion?

### Soundness
2

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper addresses the perceived brittleness and poor performance of existing open-source reward models (RMs), hypothesizing that the root cause is low-quality preference data rather than flawed modeling techniques. To solve this, the authors propose a two-stage, human-AI synergistic pipeline for large-scale data curation. This pipeline leverages a small set of human-verified "gold" data to guide iterative, error-driven LLM annotation and then uses a "gold RM" trained on this data and LLM annotations to automatically filter and curate millions of "in-the-wild" preference pairs. Training on this data, they produce the Skywork-Reward-V2 series of models (0.6B to 8B parameters), which are shown to achieve new state-of-the-art performance across seven major RM benchmarks, outperforming all existing open RMs.

### Strengths
1. The paper demonstrates a clear and significant performance improvement
2. The release of a new series of top-performing RMs and the massive underlying dataset is a valuable contribution to the open-source ecosystem

### Weaknesses
The paper's primary weaknesses are twofold: (1) a pervasive lack of clarity and the omission of essential methodological details, and (2) as a result, it is difficult to determine the true source of the claimed performance improvements.

**Lack of Clarity and Omission of Essential Details**

The paper is extremely difficult to follow. The authors have clearly performed a massive amount of work, but the execution is not explained clearly, hindering reproducibility and full comprehension. Many terms are used without definition, and no single subsection seems to provide all the necessary details.

a. Ambiguity in Stage 1 Data Curation:

Initial Data: The method for annotating the initial $y_w > y_l$ pairs is not described in the main paper. (The reviewer guesses this might be in Appendix E.1, but it must be stated).

Human vs. LLM Verification: Line 191 states human annotators follow protocols, but Line 198 mentions an "LLM-verified" portion. It is unclear how this LLM verification is performed. Is it the same as the “Preference-aware labeling”?

Missing Examples: The paper provides no example of the "LLM-generated preference attributes $a$" (Line 184) or the full 5-tuple. An example and an analysis of the quality of these attributes are needed.

Retrieval Step: In Step 2 (Line 208), it's unclear what data is used for for the retrieval to calculate the similarity. Is it from $D_{gold}$?. This makes Line 215 confusing: how do "retrieved examples" from $D_{un}$ come with "human labels"?

Line 191, it is said that human annotators perform verification following the protocols. How is the LLM-verified portion at line 198 annotated? Is it annotated by LLM using the designed protocols as described in Appendix E.3?

LLM Annotation Process: The "preference-aware" labeling (Lines 239-243) is opaque. The paper must specify which LLMs were used for annotation and provide the prompts. Without the prompt, this core step is irreproducible.

b. Ambiguity in Stage 2 Data Curation:

Data Sources: The origins of $D_{un}$ and $D_{wild}$ are never explained.

"Best Reward Model": The paper repeatedly refers to "the best reward model" used for filtering in Stage 2, but it is not specified which model this is.

Final Data Composition: It appears that in Stage 2, all pairs with $p>0.5$ are discarded, and all pairs with $p≤0.5$ are re-annotated by an LLM. This critical detail needs confirmation, as it would imply all data in the final dataset is LLM-annotated.

c. Omissions for Ablations:

Figure 6: The ablation on filtering/correction is confusing. It's unclear how these steps are performed during "iter1-8" or what the exact training set is. Stage 1 does not involve filtering $D_{wild}$ and correction.

Figure 7: This figure shows five bars, but the text (Line 416) provides only four descriptions. The descriptions are not clear. What is “LLM curation only” and what is “both human and LLM curation”? Does (4) use more data? The exact training set for each of the five bars must be specified.

d. Omissions in Figures and Tables:

Figure 1 (Right): The method for calculating "correlation" is not described.

Table 3: The caption should clarify that this metric is "accuracy on the Best-of-N split," not the more common "Best-of-N accuracy."

**Ambiguity Regarding the True Source of Improvement**

Because the methodology is so poorly described, it is impossible to determine why the model works so well. The paper's core claim is that its human-AI synergistic pipeline is key, but the details suggest that "a group of strong LLMs" with self-consistency may be doing most of the heavy lifting. D_{silver} annotations are generated by LLMs. At Stage 2, all the data included after “Preference consistency with the best reward model” are again annotated by LLMs.

Missing Baselines: The paper fails to provide the most critical baseline: what happens if you only use the LLM annotation: LLM-as-a-Judge with self-consistency? A crucial missing experiment is: (1) Use the "best RM" to filter out $p>0.5$ pairs, then (2) apply only LLM self-consistency annotations to the remaining data. Comparing this simple baseline to the full Skywork-Reward-V2 would clarify how much the complex, human-guided pipeline actually contributes. As an additional experiment: If you ensemble all the strong LLMs used in your annotation system to act as a single judge (similar to the GPT-4o judge in Table 1) with self-consistency, how does its performance compare to your final trained RMs?

In summary, the paper presents a complex pipeline. While the results are strong, the lack of clear explanation and proper baselines makes it impossible to validate the authors' claims about why it is strong. With all the missing details, it is contradictory to the reproducibility statement that “We have made extensive efforts to ensure the reproducibility of our work across all components of our research pipeline”.

### Questions
Please address all the points in weakness.

Figure 1 (Left): This chart is not fully convincing because model sizes are omitted. It is impossible to tell if a modification "fails to yield consistent gain" or if it is simply being compared to a larger model.

Please provide the prompts used for the "preference-aware" LLM annotation (L239-243) and specify which LLM(s) were used.

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
SynergyPref-40M introduces a 40M-pair preference dataset and eight reward models (0.6B–8B params) trained via a human-AI curation pipeline. Models achieve SOTA across seven benchmarks, demonstrating strong alignment with human preferences, safety, and bias resistance.

### Strengths
The open contributions and deliverables include a new, large-scale, high-quality preference dataset, a valuable asset for the research community. 

It verified that the brittleness is a root cause of RM underperformance and proposed a solution to it, which is human-AI curation synergy, that contains an elegant hybrid of human and LLM curation, balancing quality and scalability.

Empirically, Models (1.7B, 8B) achieve SOTA across seven benchmarks, outperforming much larger closed-source RMs.

### Weaknesses
The conclusion of this paper is favorable. However, for pairwise preference, it follows transitive rules. The quality of pairwise preferences can be compromised by intransitivity observed in human annotations. The paper below highlights the existence of such 'intransitivity':

- https://arxiv.org/abs/2409.19325 (Duan et al, 2017) 

In a realistic world where an 'intransitive' relationship accumulates, quality control of the curated dataset is critical, but was not clarified in the proposed pipeline.

### Questions
Given the budgeting for targeted performance, the paper still provides limited guidance on how to plan, evaluate, and achieve it.
Can you provide some thoughts on how to achieve a target performance under a given cost budget in dollar terms? 

For your reference, to overcome the cost constraints and to avoid high reliance on the availability of pairwise annotation, mechanism design has been explored as follows:
https://arxiv.org/abs/2409.18417 (Zhang, 2024), leveraging a mechanism design mindset to construct a dataset for RLHF in a cost-efficient manner.

However, the existing approaches are in different lines of techniques, more algorithmic than data-driven.

### Soundness
3

### Presentation
3

### Contribution
3

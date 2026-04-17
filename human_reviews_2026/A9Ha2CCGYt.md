# DocReward: A Document Reward Model for Structuring and Stylizing

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 4, 6

## Abstract
Recent advances in agentic workflows have enabled the automation of tasks such as professional document generation. However, they primarily focus on textual quality, neglecting visual structure and style, which are crucial for readability and engagement. This gap arises mainly from the absence of suitable reward models to guide agentic workflows toward producing documents with stronger structural and stylistic quality. To address this, we propose DocReward, a document reward model that evaluates documents based on their structure and style. We construct a multi-domain dataset DocPair of 117K paired documents, covering 32 domains and 267 document types, each including a high- and low-professionalism document with identical content but different structure and style. This enables the model to evaluate professionalism comprehensively, and in a textual-quality- agnostic way. DOCREWARD is trained using the Bradley-Terry loss to score documents, penalizing predictions that contradict the annotated ranking. To assess the performance of reward models, we create a test dataset containing document bundles ranked by well-educated human evaluators. Notably, DocReward outperforms GPT-4o and GPT-5 in accuracy by 30.6 and 19.4 percentage points, respectively, demonstrating its superiority over baselines. In extrinsic evaluations, both re-ranking and RL experiments demonstrating its utility in guiding generation agents toward producing human-preferred documents.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper describes work on building a reward model for optimizing professional aesthetics of documents for document-generation/document-assessment agentic workflows. The authors define the notion of “professionalism” based on two factors 1) structure pertaining to proper use of spacing, indention, alignment, breaks, etc and 2) style pertaining proper use of fonts, headings, emphasis, numbering, formatting, etc. The authors propose a three-step agent-based data construction pipeline to augment existing professional document datasets (GovDocs, NapierOne, collected docs) and rank based on GPT-5 judgments of professionalism. The final dataset result contains 117K paired documents, covering 32 domains and 267 document types. For the experiments, the author use Qwen-2.5-VL as the model of choice to be optimized and compared its performance exclusively with commercial models (GPT,-4o and 5 and Claude Sonnet) across pairwise and pointwise setups. Results show that DocReward-7B outperforms all other models by some margin but no tests were conducted whether the performance advantage is significant. The authors also miss comparing with strong baselines (vanilla Qwen) as well as strong openweight VL models that could have increased the paper’s empirical findings and technical rigor. There are also concerns on proper capturing of rewards from human preferences conducted by the authors. At this current stage, I’m not confident in accepting the paper at its current state. More information on issues and points for improvement are discussed below.

### Strengths
The paper is fairly well written and easy to read. The problem being tackled on notions of professionalism in documents is important but requires proper modelling of desired outcomes or aesthetics from diverse human preferences. The augmented dataset from the paper DocPairs may be of use for the community provided that proper licensing and terms of agreements are clearly-defined.

### Weaknesses
The experimented models are restrictive and exclusive to commercial LLMs and lack strong baseline comparisons. Why did you not include vanilla unoptimized Qwen-2.5-VL (3B, 7B, and 32B) for pairwise and pointwise settings? Since you’re using this as your baseline model, it’s perfectly fair to also compare it without the additional task-based optimization for document rewarding. For additional rigor and empirical evaluations to strengthen the paper’s findings, the authors can also add more updated or openweight VLMs such as DeepSeek-VL, CogVLM, etc.

There is very limited information in the human-related preference ranking which makes the paper confusing. How many (skilled) human annotators were asked to rank preferred documents? How are the pairs distributed? If only one person is employed, the reward model might be shortsighted and only favors one person’s notion of professionality. Please clarity/justify this part.

Similar to the previous question, what are the domain backgrounds of the annotators? Human annotators have inherent biases such as an annotator working on education domain might prefer formats from education documents and treat it as the gold-standard professional document which still qualifies for the “professional” definition provided by the authors. How does the study handle this possible bias? What is the specific instruction provided to the annotators when ranking? 

I strongly suggest using a more robust and recognized metric like Cohen’s kappa for calculating annotator reliability across all experiments than percentage scoring as this is misleading.

The paper lacks error analysis of the reward models. Instead of the case study, I would prefer learning about cases of instances where say a human rates a certain document layout as high score but the reward model rates it with a low score (and vice versa). These misjudgments are important to diagnoses certain weaknesses of the reward model which can be attributed to factors such as limited diversity of preference ranking, limited variation in document sources (since the domain distribution is not balanced), etc. 

There are parts using the phrase “well-educated human evaluators” - Please a more specific wording such as “qualified” or “skilled” (with n number of domain-specific experience if applicable) instead of “well-educated” as this is vague.

### Questions
Did you measure for data overlap from the existing public collections of document datasets (GovDocs1 and NapierOne) against the ones you collective via Common Crawl? Both are from public sources so there may be duplicated documents.

What would be the license and terms of agreement that the authors will associate with the new DocPairs dataset? Please mention this explicitly in the paper.

How many (skilled) human annotators were asked to rank preferred documents? How are the pairs distributed?

Are the improvements for models optimized DocReward statistically significant?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces a large-scale document preference dataset and proposes DocReward to evaluate the professionalism of documents. Experiments show that the trained model outperforms existing baselines.

### Strengths
1. The paper introduces a large-scale document preference dataset.
2. Experiments demonstrate the effectiveness of the proposed dataset and model.

### Weaknesses
1. The proposed dataset and model are limited to Microsoft Word only.

The reason I give a score of 4 and a confidence of 3 is that I have many questions (refer to the questions). Once these questions are addressed, I’m willing to raise my score.

### Questions
1.  In Figure 7, why is (c) scored higher than (b)? (I’m not entirely sure about the definition of professionalism. After checking the rules mentioned in Section 4.5, I still prefer (b).)
2. The ranking and prompt do not consider semantic accuracy. How can you ensure that the generated document does not contain incorrect content yet still achieves a high professionalism score due to its layout?
3. The ranking suggests that human-generated documents are always better. However, shouldn’t model-generated documents sometimes outperform human ones? In lines 195–198, the filtering process shows that some low-quality documents are also human-generated. In this case, the reward model may simply learn human style rather than true quality of style. (This is also reflected in the accuracy gap between Synth vs. Synth and Real & Synth.)
4. What if the versions of Microsoft Word are different?

### Soundness
2

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
The paper addresses automatic assessment of document professionalism, emphasizing structural and stylistic quality beyond textual content. The authors propose DocReward, a pointwise document reward model that operates on rendered page images and is trained with a Bradley–Terry  preference-learning objective over pairwise comparisons. To enforce textual-quality agnosticism, they construct the DOCPAIR dataset, in which each pair shares identical textual content but differs in structure and style. The model is built on Qwen-2.5-VL with multi-image inputs; a regression head outputs a single scalar score, optimized with the BT loss to separate preferred from non-preferred samples.

### Strengths
1. Clear problem specification: The textual content is held fixed and only structure/style are evaluated, thereby avoiding contamination from writing quality or factual correctness; the formal objective is consistent with the annotation protocol.
2. Training data scale and diversity: DOCPAIR spans 32 domains and 267 document types, comprising 117K paired samples, and includes both Real-vs-Synth and Synth-vs-Synth comparisons.
3. Alignment with preference learning: Pairwise supervision with the Bradley–Terry (BT) loss aligns with human-preference data and is consistent with the preference-learning paradigm underlying RLHF/DPO.

### Weaknesses
1. The work reads primarily as an engineering integration: dataset construction and a reward-modeling pipeline built on existing multimodal backbones dominate the contribution, while methodological innovations and fundamental advances over existing paradigms (preference learning/layout understanding/aesthetic evaluation) remain unclear.
2. Although the training set covers many document types, it cannot exhaust the long-tail of real-world distributions; the paper does not provide consistent evaluation on unseen types or out-of-domain settings. It is recommended to conduct explicit out-of-domain experiments (ensuring certain types/domains are entirely unseen during training) and report performance on the held-out sets; additionally, include robustness tests and error analyses for cross-lingual cases and extreme layouts (e.g., scanned documents, multi-column pages, complex tables).
3. The data pipeline relies on GPT-5 for heuristic filtering and ternary comparisons (Synth-vs-Synth). Even with a small-scale human audit, this may distill upstream model preferences in layout/style into the training signal. Moreover, the paper does not disclose expert grading rubrics/annotation guidelines or quantify inter-annotator consistency.
4. The current “extrinsic evaluation” is primarily an offline candidate generation → reward-based reranking setup and does not demonstrate that the reward can serve as an effective training signal to substantively improve generation quality. It is recommended to add downstream tasks that optimize layout/page generation with this reward (e.g., as a signal for fine-tuning or RL), and report the resulting gains.

### Questions
1. Are the DOCPAIR dataset and its construction methodology open-sourced? If so, under what license?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces DocReward, a multimodal reward model based on Qwen2.5-VL for evaluating document professionalism in structure (e.g., spacing, alignment) and style (e.g., fonts, headings), ignoring textual quality. It builds DocPair, a 117K paired dataset across 32 domains/267 types with identical content but varying visuals, using agentic expansion and GPT-5 ranking. Trained with Bradley-Terry loss on rendered images. Overall, advances agentic workflows for visually polished documents.

Strength:
- Rigorous multi-phase dataset pipeline with diverse domains provides a scalable benchmark for multimodal reward modeling.

Weaknesses:
- Heavy dependence on closed-source LLMs for agentic expansion and ranking risks propagating their stylistic biases, potentially undermining the model's independence.
- Human evaluation emphasis on win-rates overlooks potential over-optimization to annotator preferences, which may not generalize to broader cultural or accessibility standards.

Overall, the paper contributes a new dataset and a new model that may benefit the community. The method is not technically flawed per se. However, the paper lacks technical depth and novelty. Therefore, I recommend weak acceptance.

### Strengths
- Rigorous multi-phase dataset pipeline with diverse domains provides a scalable benchmark for multimodal reward modeling.

### Weaknesses
- Heavy dependence on closed-source LLMs for agentic expansion and ranking risks propagating their stylistic biases, potentially undermining the model's independence.
- Human evaluation emphasis on win-rates overlooks potential over-optimization to annotator preferences, which may not generalize to broader cultural or accessibility standards.

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
2

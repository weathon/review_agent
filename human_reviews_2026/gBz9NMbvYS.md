# ChessQA: Evaluating Large Language Models for Chess Understanding

- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Chess has played an important historical role in AI development, as it has well-defined structure and objective ground truth while admitting a wide spectrum of skill levels. However, existing evaluations of LLM ability in chess are ad hoc and narrow in scope, making it difficult to accurately measure LLM chess understanding and how it varies with scale, post-training methodologies, or architecture choices. We present ChessQA, a comprehensive benchmark that assesses LLM chess understanding across five task categories (Structural, Motifs, Short Tactics, Position Judgment, and Semantic), which approximately correspond to the ascending abstractions that players master as they accumulate chess knowledge, from understanding basic rules and learning tactical motifs to correctly calculating tactics, evaluating positions, and semantically describing high-level concepts. In this way, ChessQA captures a more comprehensive picture of chess ability and understanding, going significantly beyond the simple move quality evaluations done previously, and offers a controlled, consistent setting for diagnosis and comparison. Furthermore, ChessQA is inherently dynamic, with prompts, answer keys, and construction scripts that can evolve as models improve. Evaluating a range of contemporary LLMs, we find persistent weaknesses across all five categories and provide results and error analyses by category. We will release the code, periodically refreshed datasets, and a public leaderboard to support further research.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces ChessQA, a new benchmark for measuring chess understanding and ability. The motivation for a new chess benchmark is that existing works only cover parts of the spectrum of skills or are gameplay evaluations. ChessQA has five levels of tasks: structural, motifs, short tactics, position judgment, and semantic. Evaluation is done either as exact match with the correct answer or through multiple choice. Many of the leading LLMs are evaluated, and scores are roughly consistent with behaviour on general benchmarks: GPT-5 leads the benchmark, while weaker, open-source models struggle.

### Strengths
- The benchmark is an improvement over existing knowledge-based chess benchmarks, which only cover specific areas. ChessQA is significantly broader in scope covering five topics in detail. The set-up of the benchmark seems rigerous. 
- Since the benchmark is broken into five components, performance can be analysed in detail, and areas of weakness located. 
- There is significant interest in tracking LLMs' chess abilities from the wider AI community, though this interest is mainly tokenistic due to historical reasoning and the popularity of the game. Performance here is not necessarily the best indicator of performance on reasoning tasks more broadly.

### Weaknesses
### General points
- **Motivations:** The paper could benefit from making the necessity of this benchmark clearer. In particular, further discussion and comparison with [1], the head-to-head game evaluation, would be valuable. My thoughts are that gameplay is a better testbed for chess understanding and ability since the metric (game success) implicitly incorporates all of the skills and knowledge we care about measuring. It would be good to see an argument for why this benchmark is relevant in light of this, and how the results of these experiments compare to those from [1]. For example, do the models follow the same order or are there some models which perform well in knowledge-based tasks but poorly in gameplay, etc. 
- **Data contamination:** In addition, many of the problems are taken from existing datasets. Were these problems released before the knowledge cut-off in model training? What is the risk that models have been exposed to these specific situations before? I note that this might be a sliding scale since models may have seen the board states in a different context, but not seen the exact question. This seems like another reason why evaluating models in gameplay situations would be better, since you can almost guarantee that a board state is novel. The paper should acknowledge the limitations of reusing existing datasets
- **Future contamination:** What steps have you taken to prevent the datasets from being included in training?
- **Measuring token use:** I am unsure why token use is the most interesting metric to focus on given that some models are smaller but reason for longer while others are larger but do less reasoning. The appendix presents cost analysis, and this seems a more comprehensive metric of resources required to get the result. [This is a relatively minor point compared to the first three].

### Smaller points
- **Statistics:** Reporting standard errors over scores, etc would aid model comparison
- **Examples:** A couple of example questions in the main paper would help the user understand the tasks. The discussion of the 5 levels is fairly long and somewhat repetitive, so could be cut in place of examples. 
- **Thinking models:** The paper reports that they consider GPT-5 in a non-thinking state, which is not an option. Perhaps they mean with minimal reasoning? This is still quite different to non-thinking models. The use of the * to denote thinking is okay, but should be explained before it is used. 
- **Limitations:** Add a limitations section or discuss in one of the existing sections.

### Typos
- Section 3.1 standardisation of “First” and (i)
- The error analysis section links to the appendix. Is this deliberate?
- L452 has an incorrect quotation mark

[1] Lee et al. 2025:

### Questions
-  There is a large interest in chess since it is a strong historical benchmark, and many people play chess. However, the claims that this is an ideal testbed for reasoning more generally perhaps exceed the results. Do the authors intend to present this as a chess-specific benchmark, or are they making broader claims about the comparative reasoning abilities of LLMs? I couldn't quite establish this from the paper. Depending on the answer, the language of the paper could be better calibrated. If they are making broader reasoning claims, then comparisons with other reasoning benchmarks would be appreciated.
- GPT-5 appears to be a significant jump over existing models. Why do the authors think this is? Is this a large jump over previous OpenAI models? If so, do the authors think the benchmark will soon be saturated as frontier labs start doing more RL training? (I think this is just generally interesting rather than a crituqe of the benchmark!)

### Soundness
3

### Presentation
2

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
The paper introduces ChessQA, a 3,500-item, 50-task benchmark that evaluates LLM chess understanding across five categories (Structural, Motifs, Short Tactics, Position Judgment, and Semantic), intended to span a curriculum from rules to high-level explanation. The authors evaluate 13 contemporary LLMs in text-to-text settings (some with “thinking mode”), report overall low performance with best run 79.3% (thinking enabled GPT-5), analyze performance across categories, study token/cost efficiency, and present qualitative error typologies.

### Strengths
* Comprehensive task coverage in chess understanding. ChessQA covers 50 diverse tasks, offering a thorough and fine-grained evaluation of chess-related reasoning capabilities.
* Extensive model evaluation. The benchmark is evaluated across 13 different models, and explicitly accounts for thinking modes, providing a broad perspective on model behavior.
* In-Depth Analysis of Results. The paper conducts detailed analyses, including token efficiency and performance scaling, offering insights into both model capability and cost-effectiveness.

### Weaknesses
* Limited per-task coverage. The dataset includes 3,500 examples spanning 50 task types, resulting in an average of only 70 examples per task. This relatively small number may limit reliable evaluation for individual task categories.
* No human baseline. The paper does not report human performance, either from laypeople or domain experts, which makes it difficult to contextualize the difficulty of the tasks and assess how far current models are from human-level understanding.
* Lack of discussion on real-world relevance. The paper does not explore whether chess understanding can transfer to real-world reasoning tasks. 
* Position Judgment grid selection lacks justification. The design choices behind the bucket grid used for position judgment tasks do not provide enough support. 
* No access to the dataset.

### Questions
1. Does strong performance on ChessQA correlate with improved reasoning capabilities in real-world scenarios?
2. What is the rationale behind the choice of bucket grid boundaries for the Position Judgment task? Specifically, why were the thresholds set to {−400, −200, 0, 200, 400}? 
3. Does the use of exact-match accuracy, as stated in Section C.1, unfairly penalize models with weaker instruction-following abilities?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces ChessQA, a testing and benchmarking suite for evaluating the chess understanding of LLMs across five skill categories: Structural, Motifs, Short Tactics, Position Judgment, and Semantic. The benchmark encompasses a diverse 3,500-item, 50-task set targeting chess knowledge from rules comprehension to high-level semantic commentary. The authors systematically analyze and compare the performance of 13 recent LLMs  on this benchmark, presenting results by task and model family, as well as error analyses and insights into reasoning and token efficiency.

### Strengths
1. The construction of ChessQA is described clearly, with detailed specifications, prompt formatting, canonicalization rules, and task generation algorithms. The authors also provide reproducibility details and plan for public release.
2. The authors conducted extensive and comprehensive evaluations and experiments.
3. The authors provided detailed error analysis, which brings more insights beyond just accuracy.

### Weaknesses
1. As there are not enough examples provided in the paper this raises further concerns regarding the format and quality of the data. 
2.  The literature review misses several recent relevant papers that are directly pertinent to the goals and context of ChessQA, especially those benchmarking LLMs in chess and grid-based games, evaluating state tracking [1,2]



[1] Kuo, Mu-Tien et al. “Large Language Models on the Chessboard: A Study on ChatGPT's Formal Language Comprehension and Complex Reasoning Skills.” ArXiv abs/2308.15118 (2023)

[2] Topsakal, Oguzhan et al. “Evaluating Large Language Models with Grid-Based Game Competitions: An Extensible LLM Benchmark and Leaderboard.” ArXiv abs/2407.07796 (2024)

### Questions
1. Please check the Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper introduces ChessQA, a comprehensive QA-style benchmark to evaluate LLMs’ chess understanding across five progressively abstract categories: Structural (rules/state), Motifs (pattern recognition), Short Tactics (single-move puzzle solving), Position Judgment (centipawn evaluation classification), and Semantic (selecting the best human commentary for a given position+move among various distractors). The benchmark comprises 50 tasks with 3,500 items. Several contemporary LLMs are evaluated finding that: explicit reasoning markedly improves accuracy (~+14.7 points) but consumes many more tokens; models excel at Structural tasks (best ~97%) yet struggle on Short Tactics (mean ~17%) and Position Judgment (mostly near chance). Frequent failure modes are reported as well which are as follows: board-state hallucination, legality errors, sound analysis but wrong final action, and false “no answer”, and propose ChessQA as a dynamic benchmark because of task difficulty calibration with a public leaderboard and periodic refreshes.

### Strengths
Strengths:

1. Clear task categorization mirroring human learning trajectories: Standardized notations (FEN/PGN/UCI) along with the five-category curriculum mirrors typical human learning trajectories-from rules to explanation, yielding a richer, multi-dimensional assessment than single “best-move” tasks helping assess where modern llms lack in terms of capabilities
2. Deterministic scoring: The benchmark leans on exact-match outputs, canonicalization (e.g., sorted sets for legal moves), and engine-backed centipawn labels; motif detectors use ray scans/legality simulation; Semantic tasks enforce single correct comment with structured distractors. All these enforce determinism in scoring mechanism eliminating any randomness that could be induced because of the scoring mechanisms
3. Dynamic, parameterized construction: Ensuring difficulty is calibrated via data selection (puzzle ratings, cp buckets) and option/distractor design (tight cp grids; keyword/piece-stage/embedding retrieval) makes the dataset to evolve ensuring it won't become obsolete
4. Actionable error analysis: The four failure modes are concrete helping in understanding where the current models lack and focus on those tasks

### Weaknesses
Weaknesses:

1. No proper backing for fixed five-option mapping: Mapping engine cp to the fixed five-option grid {−400, −200, 0, 200, 400} is coarse and unclear why these boundaries serve well compared against any other setting. For example, a seven-option grid or a three-option grid or different values in the five-option grid itself
2. No Depth-Based Tactical Reasoning Analysis: The benchmark does not assess whether models can think/think how many moves ahead and make tactical sacrifices. These deeper calculations and thinking several moves ahead are essential for real-world chess strength, where short-term material loss can lead to long-term positional or strategic gains and models thinking several moves ahead get the advantage. Without this dimension, the evaluation overlooks a critical component of advanced chess reasoning
3. Accuracy vs token analysis is not very comprehensive: The thinking and non-thinking modes reveal stark differences but not clear of how much amount of thinking (low v medium v high) and controlled temperature affects accuracy 
4. Potential training-data leakage: LLM pretraining data might almost certainly includes FEN/PGN/UCI and popular puzzle/game corpora. The paper does not quantify against potential data leakage which could be the reason why it scored high on some tasks

### Questions
Check the weakness section

### Soundness
3

### Presentation
2

### Contribution
2

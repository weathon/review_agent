# EarthSE: A Benchmark Evaluating Earth Scientific Exploration Capability for Large Language Models

- Decision: Accept (Poster)
- Scores: 4, 4, 8, 6

## Abstract
Advancements in Large Language Models (LLMs) drive interest in scientific applications, necessitating specialized benchmarks such as Earth science. Existing benchmarks either present a general science focus devoid of Earth science specificity or cover isolated subdomains, lacking holistic evaluation. Furthermore, current benchmarks typically neglect the assessment of LLMs' capabilities in open-ended scientific exploration. In this paper, we present a comprehensive and professional benchmark for the Earth sciences, designed to evaluate the capabilities of LLMs in scientific exploration within this domain, spanning from fundamental to advanced levels. Leveraging a corpus of 100,000 research papers, we first construct two Question Answering (QA) datasets: Earth-Iron, which offers extensive question coverage for broad assessment, and Earth-Silver, which features a higher level of difficulty to evaluate professional depth. These datasets encompass five Earth spheres, 114 disciplines, and 11 task categories, assessing foundational knowledge crucial for scientific exploration. Most notably, we introduce Earth-Gold with new metrics, a dataset comprising open-ended multi-turn dialogues specifically designed to evaluate the depth and diversity of LLMs in scientific exploration, including methodology induction, limitation analysis, and concept proposal. Extensive experiments reveal limitations in 11 leading LLMs across different domains and tasks, highlighting considerable room for improvement in their scientific exploration capabilities. The data is available on https://huggingface.co/ai-earth.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a new set of benchmarks that deal with Earth Science, a topic that is missed by current LLM benchmarks. The contribution is two QA datasets, an easier Earth-Bronze and a more challenging Earth-Silver, with an additional scientific exploration Earth-Gold dataset. All in all, over 100,000 papers were considered for the benchmark construction. The paper additionally proposes a new metric to be used to judge LLM performance on scientific exploration, which is used on Earth-Gold. Lastly, 11 LLMs are evaluated on the 3 proposed benchmarks.

### Strengths
- Extending LLMs to topics unexplored by current benchmarks, like Earth Science is important
- The benchmarks are comprehensive, taking into account a pool of more than 100,000 papers. The end result features a large set of 114 disciplines, which is relatively large compared to other similar benchmarks
- The paper is clear and easy to understand

### Weaknesses
- LLMs are used in both benchmark construction and in evaluation (retention, win rate). Given that LLMs have been shown to be biased in various ways [1], using them to deconstruct papers into parts and then come up with questions, and then judge proposed answers seems prone to noise, biases and errors. Even with expert curation, the sheer volume of the base set of papers raises concerns with the regards about the amount of errors that can be caught. 
- The nature of the tasks means they're hard to define well. For example, the example tasks given in Table 2's research section are open ended and I could see human experts giving varied and even contradictory answers. 
- Looking at Earth-Iron/Silver: these feature more well defined answers, but on the other hand, are close to being saturated. Apart from fill in the blank, the best models get 60/70/80% on the other categories. This is approaching saturation, which raises questions about the benchmark’s continued relevance and value to the community.

Minor things:
- Figure 2 should probably be a table
- In line 238 in the caption of Table 2, $P_{hj}$ appears twice
- I think that [2] is very relevant and should be included in the discussion or in Table 1, or both. 

[1] Ye, Jiayi, et al. "Justice or prejudice? quantifying biases in llm-as-a-judge." arXiv preprint arXiv:2410.02736 (2024).

[2] Skarlinski, Michael D., et al. "Language agents achieve superhuman synthesis of scientific knowledge." arXiv preprint arXiv:2409.13740 (2024).

### Questions
1. How are biases and noise from using LLMs for both question generation and evaluation and metrics controlled? Is it possible that even that the data cleaning step, which uses LLMs, also exhibits biased/erroneous behavior?
2. In open ended tasks, what happens when more than one answer is reasonable? Are there quantitative metrics that show this not to be an issue? 
3. Given the near-saturated results on Earth-Bronze/Silver, why are these benchmarks still worth further exploration?

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
This paper presents EarthSE, a QA dataset benchmarking the earth science exploration capability of LLMs. These questions are sourced from 100k Earth Science papers, and are constructed into three subsets (Iron, Silver, and Gold). The authors decomposed each paper and then used GPT-4 to generate dialogues, followed by using human expert validation to ensure the dialogue quality. On several LLMs, the authors found that CoT guidance enhances the performances on challenging questions.

### Strengths
- This is the first science QA dataset for earth science that is derived from this magnitude of academic papers.  
- This paper presents a principled approach to construct benchmark datasets from earth science publications.

### Weaknesses
- Using one metric (SES) to assess advanced capabilities of scientific exploration (e.g., methodology induction, limitation analysis, and concept proposal) seems a bit too on reductionist side to me. Looking at the two components of the SES metric (retention rate and diversity), I became less convinced about the utility of this metric to the intended capabilities to evaluate.  
- How are the Iron and the Silver subsets divided? The boundary seems a bit arbitrary to me.  
- Similarly, the decision to make Iron & Silver in QA formats while making Gold in dialogue format also appears very arbitrary to me.

### Questions
- How is the paper decomposed into the structured components? Is it also using GPT-4?  
- The experts scored the dialogue, but did they score a subset of the QA questions in the benchmark?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
EarthSE introduces a benchmark for evaluating LLMs’ Earth science exploration capabilities. It includes three datasets: (1) Earth-Iron, (2) Earth-Silver, and (3) Earth-Gold. These datasets target different levels of knowledge, from broad assessment (Earth-Iron) to higher difficulty requiring professional understanding (Earth-Silver), and finally open-ended scientific exploration through dialogue (Earth-Gold). The authors then analyze these capabilities in leading LLMs and find that they perform reasonably well on Earth-Iron but struggle with Earth-Silver and show low retention and diversity on Earth-Gold tasks.

### Strengths
EarthSE introduces a benchmark with various difficulty levels and broad coverage across subfields (114 disciplines and 11 tasks), as well as multi-turn, open-ended dialogues for scientific exploration and discovery. To build the QA datasets, the authors leverage around 100,000 research papers and categorize them by journal impact, citation count, and topical focus. They prioritize high-quality sources and exploit paper structure aligned with the scientific discovery process, followed by extensive data cleaning and expert validation. They also introduce additional metrics for assessing Earth-Gold, evaluate multiple LLMs, and report interesting, important insights.

### Weaknesses
(1) Earth-Gold uses a fixed two-turn format. Real scientific exploration often needs longer iterative chains. It would be interesting to see the results with more turns.

(2) The inference-time “initial CoT steps” taken from question construction seem to boost FIB accuracy, but may leak answer-related cues a model wouldn’t get at test time. It would be interesting to see the self-generated CoT results as well.

### Questions
(1) In lines 455-460, it is mentioned that the initial CoT steps from the question construction step are provided during inference to see whether that helps with performance on more challenging question types. However, it is not clear to me why you didn’t instead ask the model to generate CoT reasoning while answering the question. Providing the CoT steps from the question construction step might leak knowledge about the answer that it shouldn’t, and that the model wouldn’t normally have access to.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces a suite of 3 new LLM benchmarks, two QA benchmarks around Earth science. 
There are two QA style datasets and a multi-turn conversation benchmark. 

The benchmark is based on 100k papers. Papers are categorized for field/sub-field.
The authors select subsets of size 100k, 10k, 1k by impact/popularity.


The QA pairs are constructed using LLMs. Each question/answer pair is built around a task category. The tasks include understanding but also reasoning and research tasks, such as experimental design and code generation.
For each paper tasks are selected to match the abstract.
Too simple examples are filtered.
Experts reviewed the remaining questions, keeping 


For the multi-turn dialog they model papers as a transition from( existing methods and their limitations) to (novel method and its constraints). They extract these from the top 1k most cited papers. 
Then they auto-generate two turn dialogues of summary / proposal with the respective limitations.
Again, the examples are reviewed by experts.

To evaluate the multi-turn task, the paper proposes a novel metric based on ranking several samples from an LLM against the gold example along with a diversity metric.

A slightly dated set of models is used in experiments showing significant differences between models, particularly for the multi-step task.

There are some issues with this paper that I am hopeful can be fixed.

### Strengths
Addressed a variety of scientific tasks, particularly beyond reasoning/extraction

Large corpus of papers used.

Paper is well written and has very little fluff.

Headroom for evaluation

Expert verification of at least parts of the data

### Weaknesses
GPT-4o was used in several steps of the generation process which might lead to bias. It doesn't look like a problem from the results but this should at least be explicitly acknowledged.

There is little detail on "human experts determined their retention based on the question's value". Can you tell a bit more about this process?

Evaluation of the Free Response QA tasks could be extended beyond embedding similarity, e.g. by using a judge model or using rubrics.

It is unclear if diversity should be evaluated by temperature sampling. Without calibration, that parameter might do different things with different models, as evidenced by results in Table 5

Evaluation of some more recent models with thinking enabled

### Questions
Is there an expert review of only the questions or also of the answers? How were the experts recruited / compensated? What is the acceptance / rejection rate? What is the agreement for these decisions?


Smaller issues / improvements

Please check references for the models, e.g.
* Not sure Islam & Moushi is the right reference for GPT-4o
* Gemini is cited as "Team et al" ?

The model names need to be more explicit, e.g. Gemini 2.0 Flash or Claude 3.7 Sonnet

In Figure 3, "a task that best suits your construction…" did you want to say qualification?

### Soundness
3

### Presentation
4

### Contribution
3

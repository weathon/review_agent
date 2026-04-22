# MIP-Bench: Can LLMs Implicitly Personalize Responses Using Long-Term Memory?

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 6, 4, 6

## Abstract
Implicit Personalization (IP) is the task of tailoring responses to individual users by implicitly inferring their personal context. Prior studies in IP typically infer context from a single prompt. However, as interactions accumulate, users expect Memory-driven Implicit Personalization (MIP), where models implicitly leverage contexts from users' long-term interaction histories to provide more helpful responses. MIP introduces two unique challenges: (i) identifying sparse yet relevant personal contexts within extensive historical interactions, and (ii) understanding how varying personal contexts influence preferences among plausible answers to differentiate responses between multiple users. To navigate these challenges, we introduce MIP-Bench, the first benchmark to evaluate MIP in large language models (LLMs). Our experiments reveal that recent LLMs struggle with MIP, primarily due to difficulties in identifying and retrieving relevant personal context from memory. Furthermore, our new distribution-level evaluation framework shows that even models with strong instance-level performance often fail to differentiate responses across users, defaulting to generic or overly broad outputs rather than personalized ones.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
In order to maximize their benefits for each user, LLM systems should recall relevant historical interactions with that user, and infer the optimal response based on this context.  While there has been some work in this direction, it has largely been focused on explicit personalization (i.e., adding persona information to the context) and/or short interaction timelines.  However, this explicit information will often not be available in or sufficient to inform many situations, and memory in personalization should stretch across many interactions over time.  To measure our progress in long-term personalization without explicit persona prompting, the authors of this work propose MIP-Bench, where MIP stands for Memory-driven Implicit Personalization.  MIP-Bench consists of 379 synthetic users with over 74,000 conversation sessions and 514 personalization graphs spanning different domains. Each graph links user histories to rubric-based preference vectors, and two novel metrics, Instance Personalization Score (IPS) and Distribution Personalization Score (DPS), are proposed to quantify user-level accuracy and cross-user differentiation.  Experiments reveal that current LLMs struggle with implicit personalization, mainly because retrieving the right personal context remains a bottleneck.

### Strengths
This work is in an area of high interest to the community, as LLM personalization is a widely held goal with lots of headroom for improvement.  In my opinion the biggest strength of this work is the idea of grounding examples in personalization graphs and personalization paths, which hold the potential to enable detailed analysis and interpretation of a model’s personalization performance.

### Weaknesses
While I appreciate the goals of this work, I find that it has some key weaknesses in terms of the construction, analysis, and validation of the proposed benchmark dataset:

- What role does the distinction between declarative vs. non-declarative memory play in this paper, beyond some references in related work?  It is not clear why this is a useful lens.
- I find that the paper lacks enough examples or clear explanation to understand what exactly is in the dataset.  I count 3 query examples in the entire paper including appendix (malaria/flu example, and then the two in the Qualitative Analysis section).  Since the potential contribution here is a dataset, I think much more illustration and analysis of what’s in the dataset is needed, including at least one full end to end example of one data point.  Otherwise, how am I supposed to judge if these tasks are interesting, realistic, challenging, etc.?
- This problem setup feels pretty unusual/contrived, which has several knock-on effects.  First, IPS and DPS are not general metrics that can be adopted in most personalization setups; instead, they are specific to this formulation of binary rubric outcomes.  Second, given that I am not familiar with other work with a similar formulation, I struggled to understand the problem setup after one read of Sections 3 and 4.  In 3.1, I had to read it multiple times and circle back later to totally get it.  Interlacing the notation with some complete worked example might be helpful here.  Also, in line 183, it says “an identical question Q is posed to M ≤ N users”.  Is the question posed “to” users, or “by” users?  I found line 241 to be quite jarring, where it says that “MIP-Bench contains 514 Personalization Graphs with a total of 3,285 personalization paths.”  After reading this, I was left wondering what a personalization graph/path is, as those terms had not been used yet.  I also do not understand the paragraph beginning at line 292.  Once again, describing this along with a worked example would be useful.  
- More analysis and illustration of IPS and DPS are needed to validate that they measure something useful and are helpful in making important performance distinctions.  The paper does not tell the reader what the practical difference is between, say, IPS of 0.56 and 0.66, nor does it calibrate DPS against human judgments or task difficulty, so it is unclear whether reported gains reflect meaningful personalization or, e.g., noise coming from rubric phrasing.
- Experiments do not use SoTA LLM models which makes me worry that this benchmark could already be saturated.  
- Lack of discussion of confounding between generating with GPT4o and then evaluating it on the dataset.
- Human validation is limited in scope and depth. While the paper mentions reviewer checks in expert domains, there is no presentation of annotation error rates, inter-annotator agreement on rubric applicability or scoring, or checks of WildChat insertions for semantic compatibility with the synthetic profiles. Absent rigorous human validation, spurious correlations introduced by the generation pipeline may bias scores without reflecting genuine personalization performance. 
- Isn’t the “retrieval bottleneck” finding trivial given the experiment setup?
- No explanation of retrieval methods
- Based on the qualitative example given around lease breaking, there may be some examples where the model has the relevant memory capabilities, but may want to not give specific feedback because of safety, liability, etc.  This seems very possible for this example, and makes me worry how many other examples could have similar confounding issues (once again hard to tell because very little information is given about the contents).

### Questions
Please see weaknesses

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces a realistic personalization benchmark, where models have to retrieve and reason through past interaction histories to generate a personalized answer. They showed that a graph based retrieval method beats long context and rag methods for personalization with memories.

### Strengths
1. Using interaction history as memory bank and text personalization is the most natural and realistic setting, which is something that I haven't seen from previous papers.

2. The design of the graph based retrieval is also a compelling and sensible method for tackling personalization problems with long memory.

### Weaknesses
1. My major concern is that it seems that this benchmark is highly correlated with rag benchmarks, as the authors written in 5.1 I am wondering if the correlation are so high that the benchmark is simply evaluating if the model can retrieve relevant context rather than personalization. I would like to see if sth like a covariant matrix about this.

2. User profiles are synthetically generated. It would be good to have real users and real interaction histories. Currently everything is synthetic.

3. Benchmark looks pretty easy: it seems that the last generation's frontier model can already have decent enough results, such as gpt 4o, gemini 1.5, and claude 3.5. One would wonder what might be the resutls for gpt-5, gemini 2.5, and claude 4.5. A good benchmark should be challenging enough to provide signals for people to develop new methods and algorithms. But the benchmark looks like saturated already.

### Questions
1. Can you show the correlation between some rag benchmark and your benchmark? i.e if a model performs well on rag then it performs good on your benchmark.

2. What are the performances of gpt-5, gemini 2.5, and claude 4.5?

### Soundness
3

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
3

### Summary
The paper introduces MIP-Bench, a benchmark and evaluation protocol for memory-driven implicit personalization (MIP). It builds synthetic users with long interaction histories and measures both instance-level targeting (IPS) and distribution-level differentiation (DPS). Results suggest retrieval, not generation, is the main bottleneck; a Personalization Graph retrieval scheme outperforms long-context and vanilla RAG. Strong idea with clear metrics; main concerns are synthetic-data bias, rubric oversimplification, metric incentives, and potential construction–evaluation entanglement.

### Strengths
The personalization-graph idea plus incompatibility constraints to assemble user histories is transparent and reproducible; they even inject real chat snippets to avoid fully synthetic staleness.
Defining both IPS (instance-level personalization score) and DPS (distribution-level personalization score) is neat: IPS checks “did you personalize for this user,” DPS checks “did you differentiate across users”. this directly penalizes catch-all answers.

### Weaknesses
A big part of the user histories and “decisive” signals is LLM-generated. That risks baking in the creator-model’s priors into the evaluation.
The intro argues memory helps attribution/editing, but there’s no metric like “did the model actually ground to the retrieved memory?”, so one stated motivation isn’t fully closed.
DPS rewards differentiation from the population; a model that sprinkles idiosyncratic outputs might look better, even if it's not more accurately personalized.

### Questions
Did you try a baseline that explicitly maximizes DPS by making user-id-conditioned random choices? How much DPS can you get without real personalization?

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
This paper introduces MIP-Bench, a benchmark for testing whether LLMs can implicitly personalize responses by leverage relevant details from long-term user history, rather than relying on explicit persona prompts. It consists of 514 questions across an average of 6.4 users, each having ~196 sessions per user. The scoring is based on two metrics: IPS (correct personalization per user) and DPS (whether the model outputs different answers for different users). Experimental results show that existing LLMs struggle mainly because they fail to retrieve relevant prior from memory.

### Strengths
1. Clearly identifies and isolates an important problem: personalization from memory. The evaluation metrics also identifies the importance of both correctness dimension (IPS) and differentiation dimension (DPS).
2. Dataset spans both casual and expert domains (legal, medical).
3. Shows empirical evidence that retrieval is the key failure point, which is useful takeaway for future methods.

### Weaknesses
1. Writing is very hard to follow. For example, the description in Section 3.1 is not clear whatsoever, an example could make this a lot better.

2. A central design choice in MIP is the use of rubrics to convert free-form model outputs into one-hot preference vectors. This implicitly defines personalization as discrete answer selection, so it seems like the collection of rubrics practically defines a discrete clustering of all plausible correct answers. 

Because of this (rubrics are defined at the query level with facts selection), multiple users can actually share the same ground-truth label even when their historical contexts differ substantially. In such cases, users differ, but their y-labels do not. This means personalization signal is bottlenecked by rubric granularity. If the rubric partitions the answer space coarsely or in a meaningless way, then DPS might not be sensitive to meaningful distinctions in how a model reasons about or contextualizes a response. A model may successfully retrieve and incorporate user-specific information, yet still receive low DPS.

Overall it seems like the benchmark now entangles “personalization” with answer-class selection. This limits the benchmark’s ability to capture richer or more subtle forms of personalization.

### Questions
See weakness above.

### Soundness
3

### Presentation
2

### Contribution
2

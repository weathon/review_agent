# PERSONAFEEDBACK: A Large-scale Human-annotated Benchmark For Personalization

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 6, 6, 2

## Abstract
With the rapid improvement in the general capabilities of LLMs, LLM personalization, i.e., how to build LLM systems that can generate personalized responses or services that are tailored to distinct user personas, has become an increasingly important research and engineering problem. However, unlike many new challenging benchmarks being released for evaluating the general/reasoning capabilities, the lack of high-quality benchmarks for evaluating LLM personalization greatly hinders progress in this field. To address this, we introduce PersonaFeedback, a new benchmark that directly evaluates LLMs' ability to provide personalized responses given pre-defined user personas and queries. Unlike existing benchmarks that require models to infer implicit user personas from historical interactions, PersonaFeedback decouples persona inference from personalization, focusing on evaluating the model’s ability to generate responses tailored to explicit personas. PersonaFeedback consists of 8298 \textbf{human-annotated} test cases, which are categorized into easy, medium, and hard tiers based on the contextual complexity of the user personas and the difficulty in distinguishing subtle differences between two personalized responses. We conduct comprehensive evaluations across a wide range of models. The empirical results reveal that even state-of-the-art LLMs that can solve complex real-world reasoning tasks could fall short on the hard tier of PersonaFeedback where even human evaluators may find the distinctions challenging. Furthermore, we conduct an in-depth analysis of failure modes across various types of systems, demonstrating that the current retrieval-augmented framework should not be seen as a \textit{de facto} solution for personalization tasks. All benchmark data, annotation protocols, and the evaluation pipeline will be publicly available to facilitate future research on LLM personalization.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces the PersonaFeedback dataset for evaluating personalization. It consistens of 8298 test cases that are grouped in easy, medium, and hard. The authors find that, with this dataset, enhances reasoning does not improve personalization and RAG also does not help. However, larger model do perform better

### Strengths
The authors are addressing an important gap in the field of needing high quality dataset to evaluate personalization. 

The authors have interesting results showing that RAG is not helpful for personalization but that larger models do tend to do better.

### Weaknesses
Each section of the methods could be better motivated. It wasnt always clear to me what the point of each section was.

The paper could benefit from a more detailed description how the annotators were selected and the instructions they were given as this can have a large influence on the resulting dataset.

The paper could also benefit from a limitations section describing the limitations of the dataset and what it would not be as useful for evaluating

It is not clear what the reward training model is?

My biggest concern is that the paper is over-claiming what the dataset enables. The authors state that PersonaFeedback is: "a new benchmark that directly evaluates LLMs’ ability to provide personalized responses." However, from my understanding the dataset is evaluating where the LLM is able to choose the most personalized response from a set of options, not whether it can actually generate a personalized response. these are two very different things.

### Questions
How were the 20 intial seeds collected?

how well does this generalize across cultures? Do the personas tend to be specific to the Chinese culture or is there a mix?

What is the difference between specific and general questions?

Is it possible that for the difficult level there could just be several answers that could be considered correct?

### Soundness
2

### Presentation
3

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
This paper proposes PersonaFeedback, an LLM personalization benchmark containing 8298 manually annotated samples. The main difference from existing work is the decoupling of persona inference and personalization generation tasks. The benchmark explicitly provides user personas, specifically evaluating the model's ability to generate personalized responses based on this information. The evaluation employs a binary choice task (given a persona and a query, choose the more personalized one from two candidates), with difficulty levels of easy/medium/hard based on annotator consistency. Testing on 25+ models reveals several interesting findings: augmented inference does not help personalization; larger models perform better; RAG performs poorly; and explicit personas outperform inference from dialogue history.

Overall, this is a valuable benchmark work, but it has limited methodological innovation and some experimental designs have problems.

### Strengths
1. **Important and Practical Issue:** Personalization is an important direction in LLM but lacks high-quality benchmarks; the angle of decoupling persona inference and personalization generation is quite novel and indeed an overlooked problem.

2. **Solid Data Collection:** All 8298 samples were manually labeled, with a majority of 9 labelers voting, and consistency was quantified using Fleiss's Kappa; difficulty stratification was statistically based; multiple rounds of filtering ensured quality.

3. **The method design is reasonable:** Binary selection is more objective than LLM-as-a-judge; the three-level difficulty rating (easy/medium/hard) is based on annotator consistency and has a statistical basis.

4. **Comprehensive Experimentation:** 25+ models covering 4 categories (reasoning/chat/open-source/reward), with comparisons across multiple settings (Persona Profile/RAG/No Persona).

5. **Several valuable findings:** In particular, the fact that "inference does not improve personalization" and "RAG fails" challenges some common assumptions ; these insights provide guidance for model development.

6. **Clear Writing:** Figure 1 presents an intuitive data construction process, Table 2 clearly outlines the results, and page 8 highlights five key insights for easy reader access; the appendix is detailed, providing all the prompts.

7. **High reproducibility:** Committed to open-source data and evaluation pipeline.

### Weaknesses
1. **Limited Methodological Innovation**
The theoretical contributions are insufficient, mainly relying on empirical studies. Binary choice is not a new method, and the benchmark itself lacks methodological innovation.

2. **Limitations of the Evaluation Method**
Binary choices are oversimplified. Personalization is often not black and white; the two answers may simply differ in degree or perspective. Forcing a choice between them is unreasonable. Moreover, it only uses accuracy as an indicator, lacking other dimensions for evaluation such as diversity and consistency.

3. **Soundness-related issues**
- The Persona expansion (from 20 real users to 1700) relies on LLM generation and random combination, raising questions about its authenticity and potentially creating stereotypes.
- Both the questions and answers are generated by LLM, which may introduce bias.
- Lack of statistical significance test
- It's very imprecise to conclude that "RAG failed" without discussing almost all the implementation details of RAG (retrieval strategy? top-k? memory organization?).

4. **The experimental analysis was not thorough enough.**
Without ablation studies: How are answer pairs sampled? Kappa threshold sensitivity? Importance of different Persona fields (Demographics/Personality/Preferences)? More detailed analysis is lacking: Which type of persona is more difficult? Which type of question is more difficult? How do humans perform on hard cases?

5. **Lack of theoretical analysis**
Why do inference models fail? Why does RAG fail? Why are easy problems easier to solve than hard ones? Most papers only describe the phenomena without in-depth analysis of the underlying causes. This limits a deeper understanding of the problem.

6. **Generalizability and long-term value are questionable.**
- Only Chinese data (ShareGPT-Chinese) was evaluated; cross-language/cross-cultural generalization was not verified.
- Persona primarily has Chinese users, which leads to cultural differences.
- The data size is not large (200 persons, each with ~40 questions).
- May be solved quickly (Easy is 90%+, Medium is ~80%, Hard is only ~70%)
- Static persona evaluation; in real-world scenarios, user preferences will evolve.

7. **Minor Writing Issues**
The terminology is not consistent (the terms Persona Profile, Persona, and User Persona are used interchangeably), and while Table 2 contains a lot of information, its readability could be further improved.

### Questions
1. What is the specific algorithm for Persona expansion? How is "random combinations" handled? How is realism guaranteed?

2. Complete data on the inter-annotator agreement among the 9 annotators? What is the unanimous vs. split vote ratio?

3. **RAG Implementation Details (This is what I'm most concerned about)**: What retrieval method? BM25 or dense? What is the top-k value? How is it integrated into the prompt? Figure 3 shows that RAG is even worse than No Persona; could this be an implementation issue?

4. Why is the inference model useless for personalization? Have you tried fine-tuning the inference model for personalized data? Or is the training data itself lacking in this aspect?

5. How were the answer pairs sampled? What is the size of the answer pool? Were the differences controlled?

6. How many tokens does a Persona Profile have? Will it exceed the context window of some models?

7. Which is more important: Demographics, Personality, or Preferences? Have you done any ablation?

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
The paper introduces PERSONAFEEDBACK, a human-annotated benchmark (8,298 test cases) to evaluate whether an LLM can pick the more personalized answer when it is given an explicit persona and a user query. It deliberately decouples persona inference from personalization so it can measure “can the model actually tailor to a persona?” rather than “can the model guess who the user is?” It also runs a big model sweep (reasoning models, chat models, reward models, RAG) and draws several takeaways, especially that RAG doesn’t solve personalization, and that even top models dip on the hard tier.

### Strengths
- 8,298 test instances across 200 high-quality personas, with 9 annotators and agreement-based difficulty tiers (easy/medium/hard). This is better than many LLM-only synthetic personalization benchmarks.
- They show that just retrieving user info and stuffing it in the prompt doesn’t automatically yield better personalization than giving the model a structured persona. This is an important message for practitioners. 
- The related-work section actually explains what earlier persona datasets don’t measure (mixed-inference, no difficulty levels, not fully human-annotated), so the motivation is visible.

### Weaknesses
The task is discriminative (“which answer is more persona-consistent?”), not generative (“write a persona-consistent answer”). That means you can do well with a good reranker without proving you can produce personalized outputs. This narrows what “success” means here.
The result “RAG doesn’t help” will be controversial unless they show stronger memory structuring or persona-summarization RAG. Right now it mostly rules out naive RAG.

### Questions
For RAG, What exactly was retrieved (raw user facts, past turns, structured profile), and how long was the retrieved context?

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
This paper introduces PersonaFeedback, a human-annotated benchmark designed to evaluate the model's ability to produce personalized responses when provided with a persona. This decouples personalization into persona inference and persona conditioned generation. The dataset consists of 8298 binary comparison tasks across three difficulty tiers. The authors evaluated a broad set of SOTA models and find that 1. improvements in reasoning ability do not translate to stronger personalization abilities, 2. model performance benefits primarily from scale, and 3. RAG-based personalization is ineffective compared to simply providing persona profiles directly.

### Strengths
1. This paper decomposes the personalization problem into persona inference and persona conditioned generation, which the reviewer thought was an interesting proposal.

### Weaknesses
1. The main concern is over the design of the benchmark, i.e., binary choice evaluation. In essence, the benchmark is measuring the model's ability to recognize personalization (which one of the two responses better reflects a persona), instead of measuring how good the model is at independently generating a high-quality personalized response. Binary discrimination is cognitively and computationally much easier than fluent personalization in open-ended dialogue. This seems a critical distinction in the current stage of LLM personalization research: personalization has moved beyond Alpaca-style preference tuning toward direct generation on user attributes, and the paper never addresses this. 

While such comparison-based supervision played a historical role in preference alignment (e.g., Alpaca, DPO), it is hard to view it as a final personalization objective at this stage of time.

2. For these binary preference recognition type of benchmark / work, there has already been couple in the literature (i.e., Alpaca, Zollo et al 2024), and it is unclear to the reviewer how this benchmark is materially different from the existing ones beyond the human annotation (necessarily at a cost of scale decrease compared to synthetic ones).

That being said, even these prior binary comparison frameworks themselves seem less central to how personalization is conceptualized today. Evaluating how models are able to generate personalized content as interaction continues with more context over the user seems like a more relevant topic. Given this shift, this work feels both insufficiently novel relative to prior binary-comparison work and misaligned with current research priorities.

3. The evaluation setup implicitly assumes that human annotators, when shown a persona and two candidate responses, can reliably “imagine themselves as the persona” and choose the better-aligned answer. However, this assumption is nontrivial: annotators may rely on superficial heuristics (e.g., keyword or style matching) that do not reflect the underlying user’s actual preference model. It is therefore unclear whether a higher score on this benchmark corresponds to being meaningfully better at real personalization, where user preferences are often inconsistent, evolving, or only partially observable.

### Questions
See above weakness

### Soundness
2

### Presentation
2

### Contribution
2

# A Stitch in Time Saves Nine: Detecting and Mitigating Hallucinations of LLMs by Actively Validating Low-Confidence Generation

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 5, 5, 3

## Abstract
Recently developed large language models (LLMs) have achieved remarkable success in generating fluent and coherent text. However, these models often tend to 'hallucinate' which critically hampers their reliability. In this work, we address this crucial problem and propose an approach that actively detects and mitigates hallucinations during the generation process. Specifically, we first identify the candidates of potential hallucination leveraging the model's 'logit output values', check their correctness through a 'validation' procedure, mitigate the detected hallucinations via 'prompting', and then continue with the generation process. This active intervention also facilitates in preventing the propagation of hallucinations in the LLM's output. Through extensive experiments with GPT-3.5 (text-davinci-003) on the 'article generation task', we first demonstrate the individual efficacy of our detection and mitigation techniques. Specifically, we achieve a detection recall of ~88% and successfully mitigate 57.6% of the correctly detected hallucinations. Importantly, our mitigation technique does not introduce new hallucinations even in the case of incorrectly detected hallucinations, i.e., false positives. Then, we show that the proposed active detection and mitigation approach successfully reduces GPT-3.5's hallucinations from 47.5% to 14.5%. We further demonstrate the effectiveness and wide applicability of our approach through additional experiments with different types of questions (multi-hop and false premise) and with another LLM from a different model family (Vicuna). In summary, our work contributes to improving the reliability and trustworthiness of LLMs, a crucial step en route to enabling their widespread adoption in real-world applications.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposes a multi-step procedure to reduce hallucinations in the output of LLMs.
The procedure first identifies key concepts in a sentence, filters them to uncertain ones using the model output logits, retrieves information using web search, validates the concepts by prompting, and finally corrects the output sentence by prompting with the retrieved context as support. The main experiments are conducted on article generation using a closed dataset (promised to be released after publication) using GPT-3.5 and Vicuna 1.5.

### Strengths
- The multi-step approach presented is generally sound. The approach can use black-box models hidden behind an API, and several possible solutions for each individual step are presented and evaluated to some extent.
- The experiments are primarily based on GPT-3.5, but there are also experiments with Vicuna-1.5 to validate the results. In addition, the use of an open model supports easier reproduction and improves the overall accessibility of the presented methodology.
- Hallucinations are a relevant problem with current LLMs and are a limitation to their general applicability.

### Weaknesses
- The proposed multi-step approach is likely to increase generation latency significantly. While this is noted superficially, and an improvement for one of the many steps is roughly sketched out, an in-depth discussion is missing - in particular, there are no experiments or theoretical discussions about the overall latency. I am not of the opinion that high latency is a problem for all use cases, but it would be important to have a proper discussion about this limitation and where it is a problem.
- The overall experimental design is not described in sufficient detail. In particular, it is not clear how the data used for Section 3.1 relate to those used for Sections 3.2 and 3.3. If they were the same data, I would be concerned about the reliability of the results in the later sections, since the hyperparameters of each step, such as the aggregation method used to obtain concept uncertainty, are chosen to maximize the metrics in a data set.
- It is not clear to what extent retrieval alone explains the reduction in hallucinations. Given that the proposed method uses (multiple) web search queries, a natural baseline would be to consider the article generation task based on retrieved facts about the article topic, which would have some favorable properties (e.g., lower latency, less technical complexity) compared to the proposed multi-step approach. A proper ablation/evaluation against this baseline could help to delineate this effect.
- Some of the design decisions seem to be taken quite ad-hoc; for instance, the choice of a method for key concept identification seems to be based on qualitatively looking at a few examples (Table 4, Section B.1)

### Questions
- Table 1: How can sentence-level recall (85.96) be smaller than concept-level recall (87.68) if a sentence is considered hallucinated as soon as a single concept is hallucinated?
- Section 3 describes the data selection process. In particular, the topics for article generation are selected based on the longest articles in WikiBio or Wikipedia. I would expect this selection strategy to select topics with high prevalence in most LLMs training data: either because they train directly on Wikipedia, but also because long Wikipedia articles are likely to be about a topic of general interest with high coverage in web data as well. How does this affect hallucination, or in other words, how representative are the results for hallucination detection and mitigation based on these topics?
- Regarding labeling hallucinations, how do you handle sentences that are correct given the content of previously hallucinated sentences?
- Your methodology works on the unit of a sentence. The (initial) output of the model would be a paragraph/full text. How do you segment it into sentences? What is your motivation for the sentence unit? Since you are processing the key concepts sequentially, do you need the sentence separation? Or could the approach work directly on the whole paragraph?
- Since the uncertainty calculation serves as a filter on the concepts sent for verification, I wonder about the relative importance of precision and recall. Intuitively, I would expect this to be a recall-oriented scenario, but the decision seems to be based on the area under the ROC.
- Section 2.1.5.
  > However, running this in parallel would require starting multiple threads which may not be supported by all machines.

  shouldn't this be easily solved by batching the requests?
- Section 3.1.1.: Choosing more descriptive names than `A`, `B`, `C`, and `D`, such as `YY`, `YN`, `NY`,  and`NN`, or directly using the conditional probability notation $p(H | H)$, $p(H | \neg H)$, ... would greatly improve the readability of the plots and discussion.
- Figure 5: A bar chart would be more appropriate here; I would also be interested in the confidence/dependence on the selected sentences. One way to study this would be to use bootstrapping.
- In general, I disliked the overuse of bold type, as it reduced readability quite a bit. The same goes for the use of free-floating figures. An extreme example is page 6.
- Table 1: Accuracy seems to be in $[0, 1]$, while precision and recall are given in $[0, 100]$ (i.e., in percent).
- Section 3.2, "Mitigation" + Table 2: The numbers in the text do not seem to match those in the table.
- For Section 4.2 / QA, it is not clear how the multi-step approach works at all? I would guess that the answer is always a single sentence (or even a sentence fragment), so how is the iterative sentence-level method applied?
- Section 4.3
  > Importantly, we also show that our approach does not incorrectly modify a true premise question.
  
  where is this shown?

- Appendix A: Related Work; It would be nice if the ones listed under "concurrent work" were part of the main paper description, as they seem to be the most related.
- B.3.2: 
 > Our preferred technique for retrieving knowledge is web search because the web is more likely to
contain the updated knowledge in comparison to a knowledge corpus whose information can become
stale, outdated, and obsolete.

  If (one of) the main reasons for hallucinations is outdated knowledge, wouldn't we notice that the model uncertainty does not reflect this properly, i.e. the model is very certain about its outdated knowledge?

- G.1: The sentence-level baseline uses the minimum probability over all tokens; I think it would make more sense to consider the other aggregations as well.

#### Minor Remarks
- Section 2, first paragraph: there is an inconsistent use of "Section" vs. "section"
- Section 2, first paragraph, line 3: typo: "shwon" should be "shown"
- Section 2.1.2; "normalized product of probabilities" seems to be equivalent to the geometric mean of probabilities; the latter may be the preferred term for some, so it would be nice to make this more clear (e.g., footnote, rename, ...)

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a method for detecting and revising hallucinations of LLMs. The proposed method takes a generated sentence as input, extracts key concepts (text spans) and identify low-confidence ones, generates a question for each span, retrieves knowledge, and revise the sentence. This is applied every time when the LLM finishes generating a sentence, so that future generations is conditioned on revised, more factual sentences. This is motived by an "propagation of hallucination" analysis , which shows that if the previous sentence is hallucinated, it is likely the next sentence generated by an LLM will also contain hallucination. 

For each component (key phrase extraction, confidence estimation, query generation, retrieval), the authors empirically compared a couple variations, which suggests using prompted LLMs for all tasks, and web search for retrieval. End-to-end evaluation is done by prompting GPT 3.5 or Vicuna-13B for long article generation and manually evaluating factuality. The proposed method greatly reduces hallucination.  Additional experiments show that the proposed method can improve multi-hop QA tasks as well as identifying false-premise questions.

### Strengths
- The proposed method is clearly described.
- The "propagation of hallucination" analysis very nicely show the necessity of actively reducing hallucination from the generation. Although  sentence-by-sentence actively doing retrieval and rewrite has been explored in prior work, there's little quantitive analysis studying how previous hallucination can affect future generations. 
- Experimental results indicate that the proposed method is very effective at reducing span-level hallucinations for long-form generation. 
- The improvements on multi-hop QA is large, and the gains can be well explained by the "active" hallucination detection and revision mechanism.

### Weaknesses
- It would be nice to highlight the novelty of proposed framework from existing work. A very related work is [1], where the authors do active retrieval and rewrite actively when decoding each sentence, and they also use LLM output logits to find low-confidence spans for query generation. There are also several previous works that reduces LLM hallucinations at the response-level, using a similar framework as this work by prompting LLMs for span extraction, query generation, retrieval, and revise. For example, [2] and [3] uses such a framework to revise LLM responses and reduces hallucination; [4] prompted LLMs for extracting and checking claims as an automatic evaluation framework. This paper should discuss these related work, discuss the main differences, and maybe consider them as baselines.

- The paper lacks ablations to justify some of its key components. For example, though there is a strong motivation for applying the method "actively" when generating every sentence, the end-to-end evaluation does not show how it helps reduce hallucination compared to applying it at the end of the generation. Similarly, I couldn't find ablation for only fact-checking low-confidence phrases v.s. fact checking all key phrases. 

- The presentation quality can be improved. Section 2 enumerates many modeling choices for each component, but it is difficult to tell what is the final method being used, and why it works better than the others. A suggestion is to describe the best approach in section 2, and leave other choices to ablation studies.  Section 3 and 4 cover many experiments, making it confusing to tell which is the most experiment and what are the main messages. 

[1] Jiang, Zhengbao, et al. "Active retrieval augmented generation." arXiv preprint arXiv:2305.06983 (2023).

[2] Gao, Luyu, Zhuyun Dai, Panupong Pasupat, Anthony Chen, Arun Tejasvi Chaganty, Yicheng Fan, Vincent Zhao et al. "Rarr: Researching and revising what language models say, using language models." In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 16477-16508. 2023.

[3] Chen, Anthony, et al. "PURR: Efficiently Editing Language Model Hallucinations by Denoising Language Model Corruptions." arXiv preprint arXiv:2305.14908 (2023).

[4] Min, Sewon, et al. "FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation." arXiv preprint arXiv:2305.14251 (2023).

### Questions
- Is there an ablation study comparing running the model actively vs running it at the end of the generation?
- Is there an ablation study comparing fact checking only the low-confidence spans vs all extracted spans?
- Since the method operates at sentence level, how does it know the context of each sentence? 
- The proposed method fact-checks key phrases / named entities. I'm interested to see how it works when the hallucination is not on named entities, or when the entire sentence is made up.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper presents a method to reduce LLM hallucinations using an early detection approach. The paper demonstrates that active detection and mitigation of hallucinations using logit output values is a viable path. The paper presents results on GPT 3.5 and Vicuna.

### Strengths
The paper works on an important problem of mitigating hallucinations 
The paper presents an early detection approach and demonstrates effectiveness with two LLMs 
The paper is extremely well-written, with clear goals, a well-described approach, and a detailed Appendix

While it is always possible to nitpick on experimental design issues, we need to be mindful of the fact that this work is presented within the scope of a single ICLR submission. With that in mind, the paper does an excellent job.

### Weaknesses
It is unclear how effective this method will be for generations beyond the first five sentences.

Post-rebuttal response: After going over the discussions, reviews, and rebuttals. I feel that my initial assessment of the paper had gaps. I lean towards the majority view that the paper needs improvement. I have updated my scores accordingly.

### Questions
It is unclear how effective this method will be for generations beyond the first five sentences. Will it be more useful to distribute these checkpoints across the generated text?

### Soundness
2 fair

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposes a method for detecting and mitigating hallucinations in LLM outputs. The detection consists of finding "important" concepts in the output, filtering them based on the model uncertainty, conducting web-search and feeding the info to the model to answer if the output contained hallucinations. The paper also proposes to use this knowledge from web-search to mitigate hallucinations.

### Strengths
The paper addresses the topic of hallucinations which is a relevant and timely topic.

### Weaknesses
1. The paper does not mentioned the highly relevant work of Kadavath et al., [Language Models (Mostly) Know What They Know](https://arxiv.org/abs/2207.05221), which also uses the model uncertainty to detect hallucinations. Since uncertainty is used as a major signal in the proposed pipeline, the novelty of the proposed approach is not clear.

2. The paper does not study important choices in details. For instance, the web search procedure is not very clear. The paper says “In some cases, multiple web searches were required to check the correctness of different facets of a sentence”. Are these searched human-supervised? What are the stopping criteria? I would suggest adding the web-search procedure in an algorithm block so that the readers can understand it better.

3. Similarly, the paper does not discuss exactly what kind of "important" concepts are identified by the model. Could you provide some examples? Are the models supposed to extract all relevant concepts? Is the concept extraction supposed to work well across different application domains (e.g., questions answering)? What if we are working with non-instruction tuned models?

4. It is not clear how good the instruction models were at following different instructions in Table 3. Did the authors perform a systematic analysis here?

### Questions
See points 1-4 under "Weaknesses".

### Soundness
1 poor

### Presentation
2 fair

### Contribution
1 poor

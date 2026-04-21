# Summing Up the Facts: Additive Mechanisms behind Factual Recall in LLMs

- Avg Score: 5.25
- Decision: Reject
- Scores: 5, 5, 5, 6

## Abstract
How do large language models (LLMs) store and retrieve knowledge? We focus on the most basic form of this task -- factual recall, where the model is tasked with explicitly surfacing stored facts in prompts of form \tokens{Fact: The Colosseum is in the country of}. We find that the mechanistic story behind factual recall is more complex than previously thought -- We show there exist four distinct and independent mechanisms that additively combine, constructively interfering on the correct attribute. We term this generic phenomena the \textbf{additive motif}: models compute correct answers through adding together multiple independent contributions; the contributions from each mechanism are insufficient alone, but together they constructively interfere on the correct attribute when summed. In addition, we extend the method of direct logit attribution to attribute a head's output to individual source tokens. We use this technique to unpack what we call `mixed heads' -- which are themselves a pair of two separate additive updates.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
In the context of LLM, this paper shows there exist four distinct and independent mechanisms that additively combine, constructively interfering on the correct attribute. This generic phenomena is termed as the additive motif: models compute correct answers through adding together multiple independent contributions; the contributions from each mechanism may be insufficient alone, but together they constructively interfere on the correct attribute when summed. In addition, this paper extends the method of direct logit attribution to attribute a head’s output to individual source tokens.

### Strengths
1. This paper is well written and easy to follow.
2. The experiment is sufficient.

### Weaknesses
This finding  seems to be not profound enough. It only demonstrates that LLMs perform better under the additive motif, but it appears insufficient to prove that the additive motif is the underlying factual recall behind LLMs.

### Questions
1. This finding may explain the fact that models trained on “A is B” fail to generalize to “B is A”.   Is there any possible to explain the CoT prompting such as “let’s think step by step” by using your findings? Does it bring any other insights or explanations for other phenomena that are difficult to explain in LLMs? For example, is there any possible to explain the CoT prompting such as “let’s think step by step” by using your findings?

2. Can this finding contribute to prompt engineering？

3. Some tables are too wide and are out of page, e.g., table 1.

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
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper presents a set of experiments on a small hand-crafted data
set for identifying the mechanisms at play during "factual recall" in
large language models. The paper defines four "mechanisms" based on
(1) attention heads that focus (mostly) subject of a factual
predicate, (2) attention heads that focus (mostly) relation, (3)
attention heads that attend to both, and (4) MLP layer. The main claim
is that these mechanisms additively determine the correct attribute.

### Strengths
The study tackles an important/interesting problem and the paper reports a substantial amount of experimentation.

### Weaknesses
Although I believe the idea is interesting, and there may be some
valuable finding in the paper, I have difficulties seeing a clear
take-home message based on the results presented, and probably also
due to the way they are presented. I have some concrete points of
criticism listed in the comments below (with approximate order of importance).

- The main claim, additivity of the multiple mechanisms, is not very
  clearly demonstrated in the paper. The separation of the
  subject/relation heads (as displayed in Fig. 2) is impressive.
  However, neither the roles of the "mixed head" mechanism, the MLP,
  and additivity of all these mechanisms are not clearly demonstrated.

- The dataset is rather small and it is not described in the paper at
  all. The description of in the appendix is also rather terse,
  containing only a few examples. Given the data set size (hence the
  lack of diversity), and the possible biases (not discussed) during
  the data set creation, it is unclear if the findings can generalize
  or not. In fact, some of the clear results (e.g., the results in
  Fig. 2) may be due to the simple/small/non-diverse examples.

- I also have difficulty for fully understanding the insights the
  present "mechanisms" would provide. To me, it seems we do not get
  any further insights than the obvious expectation that the models
  have to make their decisions based on different parts of the input
  (and meaningful segments may provide independent contributions). I
  may be missing something here, but it is likely that many other
  readers would miss it, too.

- Visualizations are quite useful for observing some of the results.
  However, the discussion of findings based on a more quantitative
  measure (e.g., DLA difference between factual and counterfactual
  attributes) would be much more convincing, precise, repeatable, and
  general.

- Overall, the paper is somewhat difficult to follow, relying data in
  the appendix for some of the main claims and discussion points.
  Appendixes should not really be used for circumvent page-limits.
  Ideally, most readers should not even need to look at them.

- The head type (subject/relation) definition uses an arbitrary
  threshold. Although it sounds like a rather conservative choice, it
  would still be good to know how it was determined.

### Questions
Some typo/language issues:
 - Introduction second paragraph: "(Meng et al., 2023a) find ..."
  -> "Meng et al. (2023a) find ..." 
- Although it is a very common "mistake" in the field, all 
  established style guides I know prescribe that footnote marks
  to be placed after punctuation. Also, I strongly recommend
  against placing footnote marks directly on symbols (like R^3).
- It is a good idea to indicate that figure/table references to
  the appendix are in the appendix.
- The "categories" defined at the beginning of the results section
  comes as a surprise, and seem to be an important part of the 
  analysis throughout. This should be defined/explained earlier.
- End of sentence punctuation missing for footnote 4.
- There are no references to Figure 2 from the text.
- It may not be that easy for some figures, but B/W friendly
  figures would be appreciated by people reading on paper or
  monochrome devices (like e-ink readers).
- Some terms like "OV circuit" or "ROME" that many readers are not 
  likely to be familiar with should be briefly introduced.
- The same goes for abbreviations of the sort L22H17. Not 
  difficult to guess for most readers, but it would be more reader
  friendly to explain at first use.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This work target at interpreting the inner mechanisms of LLMs in accomplishing the task of Factual Recall. This work identifies and explains four distinct mechanisms present in the model, as well as the additive cooperation between these mechanisms. This work validates the generalizability of this mechanism across different models and facts.

### Strengths
(1) Based on sufficient experimental results verification, the author has identified and explained the internal mechanisms of LLMs at the granularity level of attention heads and MLPs. More interestingly, it provides an explanation of the “reversal curse” phenomenon discovered in recent works.
(2) This work has thoroughly discussed the related work and proposed a range of possible directions for future works.

### Weaknesses
(1) There have been many works [1, 2] interpreting the model behavior of Factual Recall. It seems that the novelty is insufficient with only a deeper zooming into attention heads using similar interpretability methods. Additionally, the discovery of the additive motif is not surprising enough, as already explained in work [3] that "Attention heads can be understood as independent operations, each outputting a result which is added into the residual stream." 
(2) Is direct logit attribution (DLA) the same as the interpretability method of Path Patching [4] or Causal Mediation Analysis [5]? If so, it is necessary to explain how the counterfactual data is applied for causal intervention. If it is not, it is necessary to provide a detailed description of the algorithm flow of DLA.

(3) This work extends “DLA by source token group” with a weighted sum of outputs corresponding to distinct attention source position. But how to obtain the “weights”? How to attribute multiple tokens simultaneously? These missing implementation details make it difficult to understand the method and reproduce the results.


[1] Locating and Editing Factual Associations in GPT
[2] Dissecting Recall of Factual Associations in Auto-Regressive Language Models
[3] A Mathematical Framework for Transformer Circuits
[4] Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small
[5] Investigating Gender Bias in Language Models Using Causal Mediation Analysis

### Questions
(1) It would be better to validate the faithfulness of the identified components (e.g., Subject Heads, Relation Heads) for Factual Recall? What would happen to the prediction ability (e.g., accuracy) of the model for Factual Recall task if these components were knocked out? 

(2) We wonder if it is possible to explain the behavior of MLPs explicitly, similar to explaining Attention Heads via attention patterns?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The work concerns the task of factual recall in LLMs i.e. in templated prompts, the LLM is tasked to predict the object attribute of the tuple (subject, relation, attribute). Authors propose that factual recall in the END position (correct logit ranking) occurs by the summation of contributions of different additive circuits in the transformer.
- Authors extend Direct Logit Attribution (DLA) to compute the joint contribution from different source token groups to the final predicted logits
- 4 different additive circuits are identified based on the extended DLA: SUBJECT, RELATION, MIXED and MLP
- SUBJECT attention heads preferentially boost attributes that are relevant to the subject of the query
- RELATION attention heads preferentially boost attributes that are relevant to the relation of the query independent of the subject
- MIXED attention heads boost the attributes that are jointly relevant to the subject and relation of the query
- MLP layers at the end position uniformly boost the attributes relevant to the relation (ignoring the subject tokens)

The central findings of the paper revolve around the Pythia-2.8b model. Additional experiments in the Appendix report that similar types of circuits may be found in other models but all categories may not always exist.

Limitations:
- Authors acknowledge that the boundary between MIXED and other attention head types is fuzzy. The definition used to separate attention heads was based on preferential contribution from SUBJECT or RELATION and any other type is considered a mixed type.

### Strengths
- The paper uses established mechanistic interpretation tools and extends them to identify mechanisms in the transformer that perform very specific purposes
    - The SUBJECT-head, RELATION-head, and MLP additive behaviors are established by showing consistent patterns across a range of fact queries

### Weaknesses
- The paper introduction and further discussions claim that the results reported here provide a mechanistic explanation for the limitations of LLMs to learn "B is A" from training on "A is B" [1]. However, I do not see sufficient evidence to support this claim
    - They have shown that in the forward direction the transformer selectively promotes attributes relevant to the subject and the relation
    - This does not show that the transformer CANNOT/DOES NOT perform the same operations in the reverse direction.
    - E.g. "Basketball is played by ..." may contain circuits that selectively promote the known basketball players. The lack of such circuits is not demonstrated by this work
    - In particular, the authors argue that the LLM learns an "asymmetric" look-up. However, the asymmetry is not established.

Presentation
---
- Significant space in the main paper is used to describe future work. I believe that there is an interesting and valuable discussion about dataset creation in the Appendix that should be brought to the main paper


[1] Lukas Berglund, Meg Tong, Max Kaufmann, Mikita Balesni, Asa Cooper Stickland, Tomasz Korbak, and Owain Evans. The Reversal Curse: LLMs trained on "A is B" fail to learn "B is A", September 2023. URL http://arxiv.org/abs/2309.12288. arXiv:2309.12288 [cs].

### Questions
1. Is it fair to say that the key findings are the presence of SUBJECT-only and RELATION-only heads among the attention heads in the transformer? All other heads are MIXED heads by default?
2. What fraction of attention heads get categorized into extreme categories (SUBJECT and RELATION)?
3. How does the contribution to the final logits from the extreme categories (SUBJECT and RELATION) compare to the heads that are categorized as MIXED?
4. Tagging onto questions 3 and 4: is there a significant drop in model performance when extreme heads are suppressed?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

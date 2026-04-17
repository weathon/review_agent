# How Do Language Models Compose Functions?

- Decision: Reject
- Scores: 4, 2, 6, 4

## Abstract
While large language models (LLMs) appear to be increasingly capable of solving compositional tasks, it is an open question whether they do so using compositional mechanisms. In this work, we investigate how feedforward LLMs solve two-hop factual recall tasks, which can be expressed compositionally as $g(f(x))$. We first confirm that modern LLMs continue to suffer from the "compositionality gap": i.e. their ability to compute both $z = f(x)$ and $y = g(z)$ does not entail their ability to compute the composition $y = g(f(x))$. Then, using logit lens on their residual stream activations, we identify two processing mechanisms, one which solves tasks *compositionally*, computing $f(x)$ along the way to computing $g(f(x))$, and one which solves them *directly*, without any detectable signature of the intermediate variable $f(x)$. Finally, we find that which mechanism is employed appears to be related to the embedding space geometry, with the idiomatic mechanism being dominant in cases where there exists a linear mapping from $x$ to $g(f(x))$ in the embedding spaces.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies the ability of LLMs to apply compositional rules. Recapitulating prior works, they find a "compositional gap": LLMs are often unable to compute composed functions despite being able to compute all of the individual functions in isolation. They find that the compositional gap is shrinking with the use of reasoning models and, contrary to prior works, with the use of larger models, but the gap remains nonetheless. Additionally, they argue that LLMs exhibit two different modes of computing composed functions: *direct* and *compositional*, that is, applying the individual functions together or one after the other. Finally, they argue that the choice of computation mode is modulated by the geometry of the embedding vectors.

### Strengths
- It is valuable to see a similar analysis to Press et al. (2022) updated to recent models and reasoning models. To the best of my knowledge, the evidence for a shrinking compositionality gap is novel and very interesting if true. It is also interesting to see that reasoning models still struggle with compositionality, although to a lesser extent.
- The rules appear to be carefully chosen to avoid overlap.
- The distinction between direct and compositional mechanisms is interesting and appears well supported by the empirical analysis

### Weaknesses
- The tasks require the LLM to figure out the rule based on 10 given input-output examples. This means that LLMs face the additional obstacle of figuring out the rule. To me, it seems that this additional difficulty is greatly altering the nature of the task and undermining some of the key claims. Intuitively, it seems more difficult to figure out that 2044 is (922+100)*2 than to answer "What is 922 plus 100, times 2?". Under the current prompting formulation, I believe the paper tells us more about ability of LLMs to *learn composed functions in-context* than their ability to *compute composed functions*.
- When the input $x$ consists of multiple tokens, the linearity analysis is performed with the average of all token embeddings. This is an important limitation because, for example, a book title might be composed entirely of words (tokens) that also appear in the titles of another author. We should expect such an average to be equally correlated with the birth years of either authors. However, it is not inconceivable that the LLMs combines the tokens to create an embedding specific to that book and correlated only with the correct birth year. Another approach could be to study the linearity using the residuals obtain when passing $x$ through the LLM (without the in-context examples).
- The literature review is missing recent works on LLM compositionality. For example, works [1-3] also study the ability of LLMs to compose factual and in-context information.

[1] Ni, Ruikang, et al. "Benchmarking and understanding compositional relational reasoning of llms." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 39. No. 18. 2025.

[2] Musat, Tiberiu. "Mechanism and emergence of stacked attention heads in multi-layer transformers." arXiv preprint arXiv:2411.12118 (2024).

[3] Xu, Zhuoyan, Zhenmei Shi, and Yingyu Liang. "Do large language models have compositional ability? an investigation into limitations and scalability." arXiv preprint arXiv:2407.15720 (2024).

### Questions
- Have the authors tried to add the rule to the prompt in natural language, for example "What do you get by adding 100 and then doubling?", "What is the birth year of the author of this song?", etc., in addition to the 10 examples ? Does the narrowing of the gap with increasing scale persist in this regime?
- Have the authors attempted to compute the linearity using the residual stream of the input $x$?
- Do the authors have any insights on how can it be that reasoning models still exhibit a compositional gap? It seems that they should be able to just apply the hops one by one using the reasoning tokens.

### Soundness
1

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors inspect LLM's ability to process composed functions x->f(x) and g(f(x)) versus directly computing x->g(f(x)) in a 'one-shot' fashion. Upon inspecting the LLM's inner mechanisms they find that LLMs contain both, individual circuits for f and g and direct shortcut circuits $g\circ f$. The authors conclude that direct shortcut mechanisms are leveraged whenever there exists a linear mapping between the input and output representation.

The authors present an empirical study on compositional reasoning, evaluating how different LLMs perform on tasks requiring said capacity. One of the main points is that LLMs often make use of idiomatic reasoning instead of compositional reasoning. In detail, a number of experiments is performed to support two statements: 1) Within the limits of current methodologies for interpreting LLM's inner workings, compositionality remains an open challenge. The performed experiments supports that current LLMs do not show hints of compositional reasoning for many tasks that clearly require it. 2) The geometry of the embedding space plays an important role. Experiments hint that the presence of compositionality is inversely correlated to the linearity of the embedding space.

### Strengths
The presented research question is of high importance for understanding the inner workings of LLM reasoning. Similar to previous works, the authors demonstrate a gap between compositional and direct short-cut reasoning of LLM. The gap is shown across models of different sizes, including instruction tuned and thinking models.

The the paper is well structured and easy to read. Results are clearly presented and discussions align with the obtained results. To the best of my knowledge the authors soundly apply existing techniques and reasonably cover and discuss related work.

Findings on embedding space linearity pose new insights on LLM reasoning, allowing the prediction of the inspected phenomena beyond the inspected models.

### Weaknesses
1) The authors seem to exclusively employ rather outdated "Q:\<query\>\\nA:" in-context learning prompts for pretrained autoregressive language models in their experiments. For the tested thinking and instruction tuned models, the paper is lacking details on the exact prompt structure. Additionally, I could not find any examples of the exact prompts used to embedding the tasks which could be a major factor in model performance.
2) More generally, the authors give no insights into the observed failure modes of the models, e.g., none of the obtained outputs are shown or analyzed. The authors mention that results are only considered under exact answer matching which might reduce accuracy, due models not adhering to the expected format. Similarly, reasoning models are mentioned to be given a 2000 characters limit. While this might be a sufficient for testing the rather simple tasks, it could be that decreased accuracy in the compositional case might simply stem from aborting due to the token limit.
3) The employed logitlens and patchscope techniques, which establish main findings of the paper, are not explained. No discussion on the assumptions or limitations of these techniques is made, which requires the reader to already have a deep understanding of these methods. As mentioned above, details on prompts and result evaluation are generally missing, making the paper not self-contained and lack in technical detail.
4) While the authors initially demonstrate a compositionality gap on multiple recent and instruction-tuned models, the main evaluation seem to only be performed on the rather old and weakly performing Llama 3-3B. It is unclear to me, whether the conclusions gained from these evaluations would generalize to modern LLM. Given the additional lack in prompt descriptions and introspection techniques, I find it hard to judge the significance of the presented work.

### Questions
I would like to ask the authors to elaborate on the following points:

1) How do the exact prompts templates and obtained results look like? Have the authors made sure that performance degradations do not stem from insufficient prompt embedding of the tasks or mismatches in the output format? Where there any commonly observed failure modes that could have biased results?
2) How do the obtained insights transfer to modern instruction-tuned or 'thinking' LLMs?
3) How exactly did the authors modify the evaluation mentioned in sec. 3.1? How do the authors determine a lexical unit in this particular setting?
4) What the the assumptions made by logitlens and patchscope and do these assumptions hold in the presented setting? Are the mentioned logitlens and patchscope techniques applicable to longer reasoning traces?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper investigates how large language models (LLMs) perform compositional reasoning , whether they explicitly compute intermediate steps when solving tasks of the form g(f(x)). The paper introduces a set of new two-hop compositional factual retrieval tasks that can be expressed in the form of g(f(x)). These tasks focus across various domains, including arithmetic, translation and rotation. Based on these tasks, the authors confirm that current LLMs still suffer from a “compositionality gap”, which they define as the ability to correctly answer each individual hop (computing z=f(x) and y=g(z)) but not their overall composition of g(f(x)). Using a logit lens analysis of residual stream activations, the authors find two distinct processing mechanisms. In some tasks, models explicitly represent intermediate variables (a compositional mechanism), in others, they rely on a direct or “idiomatic” shortcut, enabled when a near-linear mapping exists between x and g(f(x)) in the embedding space.

### Strengths
1. The paper is well motivated: it systematically explores whether large language models perform compositional reasoning explicitly or implicitly through function composition tasks
2. The paper is well-structured: it first quantifies the compositionality gap using a newly introduced dataset of two-hop factual tasks, then analyzes the internal mechanisms through a logit lens, and finally attempts to explain the observed behavior.
3. The finding of the paper is interesting: each section reveals distinct and insightful aspects of factual knowledge encoded in LLMs.

### Weaknesses
1. The evaluation is limited in scope: as the paper focuses on a single, smaller model from the Llama 3 family (3B). Including larger models and models from different families would help validate and strengthen the paper’s conclusions. Performing experiments on opensource models with opensourced lenses like Pythia, Gpt2XL, Llama2 would help make the results more generalizable. The authors could potentially present these during the rebuttal. 
2. Prompt design for section 3: the paper only used one simple in-context prompting method. It would be interesting to see by varying prompting strategy (such as chain-of-though) how would it affect the performance of the model
3. The interpretation in Section 5 is limited: as the paper only reports the correlation (r²) between embedding linearity and compositional processing. However, correlation does not necessarily imply causation. Hence, additional causal evidence or directional analysis would strengthen the argument. Additionally, testing out at least one more reasonable hypothesis that gets negated would strengthen the paper.

### Questions
Look at weakness section

Figure 1 is slightly hard to understand. Might be better to separate the two. 

Instead of writing All hops, would be nice to specify 1 hop, 2 hop, 1+2 hops.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates the question of how language models handle compositional processing. Specifically, they first confirm that modern LMs (at least Llama-3 3B) continue to suffer from the compositionality gap identified in previous literature, in which for some task expressable as g(f(x)), they can compute f(x) and g(z) separately but not the full problem. They then use logit lens in order to identify a compositional and idiomatic path to solving problems expressable in this form, and find that the idiomatic mechanism may be dominant in cases where there is a linear mapping from x to g(f(x)) in embedding space.

### Strengths
**S1**: I appreciate the thorough discussion of related literature, especially in the philosophical and cognitive science domains. I like that the paper thoughtfully engages with this literature and positions findings relative to previous work on compositionality and debates about compositional behaviour vs. representation. 

**S2**: The framing and experimental setup are very clear and examination of the basic form of f(g(x)) is very well explored through many different types of relationships. I also think the finding that the idiomatic path is more prevalent when there’s a direct linear mapping is very interesting and the different categories could also shed some light on why (for instance, for factual questions in many cases the answer could be mentioned directly while for computational questions it would be rare to see exact intermediate steps in training data).

### Weaknesses
**W1**: In general, it seems like the setup could have been expanded a bit in a way that would make the results more generalizable and robust. Although this paper is an analysis paper and is not meant to probe complex reasoning, I think that a few additional experiments or tweaks would help make the results much stronger:
- The decision to only use a single token prediction seems overly strict, as several tasks with multi-token outputs are excluded from the analysis. For instance, x = “excessive” shares the same token as g(x) = “excessive”, but it seems like this would unnecessarily exclude many possible compositions. Some effort should be made to change the setup to be usable at the span level through pooling results from logit lens if possible.
- Although many models are explored for the compositionality gap results, there is only a single model (Llama3 3B) tested for the main results. It would be interesting and speak to the generality of the results if at least one other model family at a different size was also tested.

**W2**: While the paper addresses an interesting theoretical question about compositionality, the practical implications of the findings remain unclear. The experimental setup largely adapts existing methods (compositionality gap from Press et al. 2022, processing signatures from Merullo et al. 2024), and the novel contribution—the correlation between embedding space linearity and processing mechanism—lacks clear connections to downstream outcomes. How should these findings inform model development, evaluation strategies, or our understanding of when models will succeed or fail at compositional reasoning?

**W3**: This is a minor weakness, but just putting some suggestions here: the figures in this paper could be improved in clarity and visuals, 

- It was very hard to interpret Figure 1, the upside down axis for absolute is confusing. To make this clearer I would suggest just having the red bar made up of the relative proportions of the yellow and blue bars but of different heights, or to simply make this two separate figures instead of trying to stay on the same axis. The color labels are also confusing, from what I understand yellow is P(final correct | all hops correct) and blue is 1 - P(final correct | all hops correct) but I’m still not sure I interpreted this correctly. 
- Figure 2 should not be a line plot. The x axis values are different models and while they do increase in size, making this a line plot implies some kind of continuous relationship. Additionally the dual axes are again hard to interpret with compositionality gap vs. proportion correct, I think it would also be clearer to split this into two figures.

### Questions
- is the linearity-mechanism relationship causal? Can you test whether manipulating linearity causes more idiomatic processing, or whether both arise from a common factor such as frequency of the direct mapping in training data?

- does processing mechanism predict whether an LM generalizes compositionally? It seems like the compositional path is more general but it seems like it could be fine to just memorize some direct mappings especially for QA and this may not hurt performance.

- why limit to single-token outputs? Is there a reasonable way to get around this, given that this excludes many natural tasks?

### Soundness
2

### Presentation
2

### Contribution
3

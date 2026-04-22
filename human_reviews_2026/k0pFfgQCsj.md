# Mitigating Text Degeneration via Token-Level Guidance For pruned Large Language Models

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 6, 2, 8

## Abstract
Large language models (LLMs) suffer from substantial memory and inference costs, and pruning has emerged as a widely adopted strategy for compression. However, while pruning effectively reduces model size and latency, it often exacerbates undesirable side effects such as text degeneration, particularly repetition, even when perplexity remains largely intact. We observe that standard post-pruning fine-tuning is insufficient to suppress repetition, motivating the need for more targeted approaches. To address this issue, we propose two token-level guidance methods: FOCUS and RePAIR. FOCUS adjusts token probabilities through token-weighted distillation, focusing on high-confidence regions to better align the student with the teacher while reducing the likelihood of undesirable tokens.
In contrast, RePAIR employs contrastive training with negative and positive samples to explicitly encourage the generation of alternative tokens. Experiments on open-ended and instruction-based generation tasks demonstrate that our methods substantially reduce repetition and improve generation diversity, while causing only minimal impact on perplexity. Furthermore, our methods are compatible with other training strategies and consistently enhance their performance. Code will be available soon.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes two strategies to mitigate repetitive degeneration in pruned model outputs.
The first (FOCUS) redistributes probability mass over the top-probability tokens during the model distillation phase, moving more probability mass onto less probable tokens.
The second method, PT, uses a contrastive objective to get the model to generate text more like the teacher model, and less like degenerate outputs from the pruned model.
Along the way, the authors introduce a new metric, CREP, to measure repetitive degeneration in model outputs.

### Strengths
The paper focuses on an important problem. The method appears to be effective in reducing repetitiveness. The writing is clear, with the exception of the explanation of the FOCUS objective, which lack motivation in my opinion.

### Weaknesses
1. I am somewhat unsatisfied with the FOCUS objective's motivation, explanation, and intuition. Looking at the weighting function, it seems to give full weight (1) to tokens close to 0 and 1 probability, and has a u-shape, giving the least probability to tokens near 0.5 probability. It is not clear why this "gives more weight to high confidence tokens" as you say, and if it does, why use this particular formulation? Figure 2a seems to demonstrate that the particular distribution becomes more high-entropy. If that is the goal, why not adjust the teacher distribution with something more standard, like temperature? If temperature does not work as well, why? It would be good to see some ablations.

2. The paper you cite (Jaiswal 2024) showing degeneration in modern LLMs only shows significant degeneration _after_ 30% pruning and in your experiments, you use only 25% pruning. Furthermore, language is naturally repetitive to some degree. The lack of baselines for repetition metrics for unpruned models makes it unclear whether the method is needlessly removing the natural level of repetition in the generations.  

3. Overall, my excitement for the paper is middling. It does not seem to offer significant insights that broaden our understanding of language models, degeneration, or distillation. Rather, it offers two engineering ideas that they argue works well in practice. PT seems like the better motivated of the two ideas, but it does not in my mind break new ground. Contrastive objectives to make the student model more like the teacher seems like an obvious idea and I would be surprised if it is not already used in practice.

### Questions
1. Why did you think it necessary to define a new repetition metric? Why are existing metrics insufficient? For instance the one used in the nucleus sampling paper (Holtzman 2019).

2. Your contrastive learning objective has some similarities in terms of inputs (favored/disfavored example pairs) to DPO. Did you experiment with DPO? How would you characterize the difference/connection?

Typos
- 465: thm

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
4

### Summary
When pruning LLM's language model capabilities often degenerate, to recover some post-training is applied, however artifacts like repetition-loops may persist. The authors propose to augment standard CE-loss with FOCUS, a teacher loss - weighted knowledge destillation, and PT, pairwise margin-based training. Experiments show that this indeed fosters diversity in generations.

### Strengths
- clear motivation and methodolgy and experiment and written form - overall solid work

### Weaknesses
W1 weighted KD and a ranking loss are well established methods, not surprising that it will improve (as long as there are fitting teacher/ samples), so the major novelty is applying it to pruning/ finding that it solves this particular error case. this error case again is very typical and not surprising that 'better training signal fixes it faster'. i'm missing ablations on when pure CE also fixes the repetition pitfall. given that this is ICLR main track, i cannot give it better than weak accept here.

W2 the focus on this paper is on diversity generation only. ppl and standard evaluation benchmarks consistently get slightly worse for basemodel. in instruction model benchmarks slightly improve but PPL still gets worse? i guess a paragraph / discussion on how meaningless the deviations in perplexity/ zero shot are could be in accordance to [1] 

W3 [1] also shows that 25% pruning in a non-uniform fashion is actually pretty well achievable with almost no degradation. imho it would be necessary to see if the benefits persist on larger pruning rates/ different pruning techniques.


[1] https://arxiv.org/abs/2311.01544

### Questions
adressing /commenting weaknesses

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes two methods for mitigating token repetition. The first, FOCUS, takes KL-divergence minimization to a stronger teacher, and changes the expectation to instead be weighted by a transformed function of the teacher’s likelihood – focusing more on very high or very low probabilities. The second, PT, generates synthetic preference data by first detecting student-generated examples of repetition (including the prefix where repetition starts), and then rolling out better continuations for that prefix from a stronger teacher model. These preference pairs are then used to finetune the student model using a loss that incorporates a hinge-like threshold to a log-likelihood preference loss.

The paper evaluates these methods in a setting in which the student model is low-quality due to pruning, and the teacher is the unpruned model.

The paper evaluates on various GLUE-like benchmarks (e.g., PIQA, BoolQ) as well as some open-ended generation (MAUVE and PPL, I think on wikitext 103-like text.)

### Strengths
This paper explores some interesting ideas – it has a nice repetition measurement heuristic. The method for detecting repetition and rolling out non-repetitive texts for preference learning is nice. Ideas around sharpening the KL-divergence are also interesting to explore. Overall, this paper proposes a bunch of potentially interesting ideas for getting any model that has repetition issues to behave more like a teacher model that doesn’t.

I think the ideas here have value, and a clarified version of this paper in which the individual aspects of each method are tested more clearly and the experimental setup were stronger would lead to a higher score.

### Weaknesses
Unfortunately, I have a variety of concerns about this work. I’ll give examples and places for improvement, but overall I found myself struggling to take clear conclusions from the work.

For all experiments, it seems like MAUVE, n-gram uniqueness, and PPL metrics are computed over Wikitext text after finetuning on instruction-response formatted data from Alpaca. This train-test mismatch does not make sense to me – why should we expect the model to generate wikitext-like articles or have low wikitext PPL after finetuning on chat-formatted instruction-response data? 

For FOCUS for example, the hypotheses seems to be that KL-divergence regularization to a teacher doesn’t focus enough on tokens that are low-probability under the teacher (because the importance weight, the teacher’s probability, is close to 0.) KL-divergence does penalize putting probability mass on tokens that are low-probability under the teacher through the partition function of the softmax – the probabilities must sum to 1, so putting too-high probability on a low-probability token is penalized. The hypothesis here is unclear to me because we don’t see evidence in this paper that the particular weighting of KL is bad in this setting (the pruning-recovery setting) beyond FOCUS leading to higher scores under the authors’ proposed repetition metrics. Indeed, I’m not sure how the particular re-weighting relates to repetition.

For PT – pairwise training – again while the particular synthetic data generation method proposed by the authors is nice, it seems from the naming and the experiments that the paper claims to be proposing the idea of a pairwise preference loss – e.g., it provides a new hinge-like likelihood-ratio loss, and the phrasing “pairwise training” is quite general. Yet, many many losses have been proposed for making use of a positive and negative answer pair (DPO, KTO, SimPO, etc., etc.,) and the choice of loss function seems independent from the synthetic data method the paper proposes (which itself depends on the paper’s definition of repetition.)

For the experiments, I don’t really understand the setting of taking a model, pruning it, finetuning it on Alpaca, and then evaluating it on wikitext PPL and generation and GLUE-style benchmarks (boolq, PIQA, Winogrande.) Unfortunately, my reading of the non-repetition metrics is that the methods don’t really help - -e.g., the changes in the various GLUE benchmarks are very small and often negative, and I’m not convinced in the importance of the repetition metrics in the setting provided. E.g., if you evaluate the model on Alpaca questions, does it degenerate into repetition? Surely not if you finetune it well?

Overall, the improvements of the methods seem to rest on the repetition metric introduced by the authors, and n-gram uniqueness, when generating wikitext. But if we care about the model generating good wikitext, why is the model itself not being finetuned on wikitext instead of Alpaca?

### Questions
Notes:
 - Please use \citep for citations that should be surrounded by parentheses. For example, “Curie et al. 2025 showed that” should be \citet, but “This has been observed in multiple studies (Curie et al., 2025, Newton et al., 2025)” should be \citep.
 - What is the “Wiki” dataset? – there are many Wikipedia-derived datasets. (e.g., line 053)
 - Top-k sampling was not introduced by Li et al., 2020 as cited. It was introduced by Fan et al., 2018 (Hierarchical Neural Story Generation.)
 - Shouldn’t CREP also depend on the n-gram chosen –  r ?
 - I’m confused about the Alpaca finetuning setup – you say you finetune on the Alpaca dataset, but then you have the model continue Wikitext-103 paragraphs. Is this in a question-answer format, like Alpaca?
 - What datasets are perplexity and MAUVE computed over?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
Addresses text degeneration, specifically repetition, which often worsens after pruning Large Language Models (LLMs).

Pruning reduces LLM size and latency but exacerbates repetition despite preserving perplexity that standard post-pruning fine-tuning fails to suppress.

The authors propose two novel token-level guidance methods:

1. FOCUS (Token-Weighted Distillation): This method uses token-weighted distillation to focus on high-confidence regions (where 0 and 1 are confident response probs and it decreases as we approach 0.5). It better aligns the smaller model with the larger teacher, reducing the likelihood of repetitive tokens.

2. PT (Pairwise Training): This method employs contrastive training with negative and positive samples. It explicitly encourages the model to generate alternative, more diverse tokens.

Experiments show these methods substantially reduce repetition and improve generation diversity in pruned LLMs.

The guidance methods minimally impact perplexity and enhance the performance of other training strategies.

### Strengths
- solves a critical problem (text degeneration / repetition) in pruned LLMs. Even though the discussion is about pruned LLMs, it seems to be widely applicable to any sort of distillation.
- Introduces two novel methods (FOCUS and PT) to approach the same.
- Experiments show the above methods to be very effective.
- Compatible with most existing training setups.

### Weaknesses
- Complexity of training increases due to the need of pairwise training data extraction.
- Adds multiple hyperparameters to tune.
- Encouraging the model to prefer the alternate tokens might confuse the model and lead to increased perplexity.
- Need to test how this impact factuality benchmarks where alternate tokens might be risky.

### Questions
see weakness section.

### Soundness
4

### Presentation
4

### Contribution
3

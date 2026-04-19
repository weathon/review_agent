# IntGrad MT: Enhancing LLMs' Machine Translation Capabilities with Sentence Interpolation Guided Gradual MT

- Decision: Reject
- Scores: 6, 3, 8, 6, 3

## Abstract
Recent Large Language Models (LLMs) have demonstrated strong performance in translation without needing to be finetuned on additional parallel corpora. However, they still underperform for low-resource language pairs. Previous works have focused on mitigating this issue by leveraging relevant few-shot examples or external resources such as dictionaries or grammar books, making models heavily reliant on these nonparametric sources of information. In this paper, we propose a novel method named IntGrad MT that focuses on fully exploiting an LLM’s inherent translation capability. IntGrad MT achieves this by constructing a chain of few-shot examples, each consisting of a source sentence and the model’s own translation, that rise incrementally in difficulty. IntGrad MT employs two techniques: Sentence Interpolation, which generates a sequence of sentences that gradually change from an easy sentence to translate to a difficult one, and Gradual MT, which sequentially translates this chain using translations of earlier sentences as few-shot examples for the translation of subsequent ones. With this approach, we observe a substantial enhancement in the xCOMET scores of various LLMs for multiple languages, especially in low-resource languages such as Hindi(8.26), Swahili(7.10), Bengali(6.97) and Marathi(13.03). Our approach presents a practical way of enhancing LLMs' performance without extra training.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the topic of improving LLM-based machine translation through prompting. Specifically, the paper introduces IntGrad MT, which is a combination of sentence interpolation and gradual machine translation. The intuition is to start from translating a sentence that the model can translate well, and gradually interpolate that sentence into the desired source sentence using an LLM. The LLM used for translation, then, is prompted to translate each of the interpolated sentences, given its previous translations as a guide.

Update after discussion period: The revised version of the paper has partially addressed my concerns about baselines and evaluation metrics, so I am raising my score.

### Strengths
The proposed IntGrad method is a very interesting and novel idea. It is unlike any prior work that I am familiar with, and (with some improvements to the experiments) I could see opening up new avenues for research into LLM prompting.

### Weaknesses
1. **Evaluation metrics**. The proposed model is evaluated using mostly xCOMET. I have two potential concerns here: a) Although xCOMET correlates well with human judgments on *high-resource* languages, it is much less reliable on *low-resource* languages. In particular, the DA data used to train xCOMET only covers EN-ZH and EN-DE of the evaluated language pairs. b) The models partially develop on COMET, as it is used in output selection (CometKiwi but the base model and training data is largely the same) and sentence pool creation. This raises the concern that gains could be an artifact of tuning towards the metric, and not necessarily due to better translation. A simple fix could be to use a model-free metric (which could be more language-agnostic) as a point of comparison (e.g., chrF), and to include even a small amount of human evaluation to corroborate the results.

2. **Weak baselines**. The paper (L16) points out that several methods have been proposed for improving LLM low-resource translation including few-shot translation and using dictionaries and grammar books, but does not do any comparison to using dictionaries and grammars. In addition, for low-resource translation neural machine translation is a strong baseline and should be included, but no rationale is given for why the paper doesn't compare to NMT or why LLM-based translation would be preferable (it's not cheaper or faster, either). The zero-shot and three-shot baselines are not particularly fair either: given how compute-intensive the proposed method is, it would be fair to use more shots (e.g., 5 or 10) or chain-of-thought. In addition, per L265, "we use the same start sentences for interpolation as in the few-shot examples", which also seems unfair to few-shot, as the extra compute and iterations used in the proposed IntGrad could be similarly used to find good/appropriate few-shot examples.

3. **Construction of the sentence pool**. The construction of the sentence pool seems it would have a huge effect on the final performance, and it seems underexplored in this paper. In particular, I am curious what the impact of the size and quality of the sentence pool is. Also, the sentence pool used in the paper is the development set corresponding to the test sets, so it is quite well-matched to the test set in domain, style, and format. This is not a realistic setup, and it would be good to see experiments showing what happens when you don't know the domain in advance and have to use a generic sentence pool, as in a real-world application.

4. **Practicality of the method**. I would have liked to see some discussion of translation time and cost, beyond a cursory mention in limitations (L534 "introduces some computational overhead"). What computational overhead, what is the increase in cost, and how much longer do translations take? Is it possible to batch the translations? These are relevant questions to be explored in at least some depth, given that I think it's quite a large increase.

### Questions
1. Although you mention in limitations that ﻿﻿﻿"sentence interpolation did not perform well in languages other than English" (L537), I'd be curious to see examples of others, and some evaluation results (even presented as negative results). I think that would be informative for future work.

2. L252: "We first utilize the dev split of the dataset to create start sentence pool. During evaluation, to test the effect of the output selection strategy, we selected 10% of the test portion of the dataset to set the DA score threshold. The remaining 90% is used to assess the overall performance on the dataset": since these are established dev/test splits, I recommend using a 10% subset of the *dev* set to set the DA score threshold. Maintaining the test split as it was originally released and evaluating on the whole thing will help reproducibility.

3. Do you think you could eventually interpolate from one language to another? E.g., if a model translates English->Hindi well but German->Hindi poorly, do you think you could gradually interpolate English source with German source to eventually get a German sentence? This is not essential for the current paper, of course, but I'm curious to hear your intuition on this.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
This paper proposes IntGrad MT, a method to enhance machine translation capabilities of large language models (LLMs) without additional training by using sentence interpolation and gradual MT. The approach constructs chains of sentences that incrementally increase in difficulty, using the model's own translations as few-shot examples. The authors evaluate their method on various LLMs (GPT-3.5, Mistral Nemo, Llama models) across multiple languages and report improvements especially for low-resource languages.

### Strengths
1. The paper addresses an important problem of improving LLM translation capabilities for low-resource languages *without* additional training

2. The evaluation includes multiple models. Even though source is fixed to English, the paper evaluates on multiple target languages from different language families and using different scripts.

3. The ablation study attempts to analyze different components of the method

### Weaknesses
1. A very big issue is that the baselines (0-shot, 3-shot) are unfairly chosen because they are too weak, and as a result I am not at all convinced that the method proposed by the authors would outperform stronger baselines. Since the main contribution of the paper is improving MT quality, this is something that definitely should be improved. At the absolute minimum, additional baseline results for n-shot should be included, where n is chosen such that the computational costs are similar to the method proposed by the paper. Ideally, some other popular methods that improve LLM-based MT without fine-tuning (from related work Section 2.1) will also be included.

Let's consider example 3 from the appendix, which has an interpolation path of 7 sentences. If I understand correctly, this means that we start from the first sentence and do a 0-shot translation, and then recursively do (n-1)-shot translations until we arrive at the final sentence in the interpolation path. The total costs for X are thus:

    - s1 -> 0-shot translation +
    - s2 -> 1-shot translation with input (s1) +
    - s3 -> 2-shot translation with input (s1,s2) +
    - s4 -> 3-shot translation with input (s1,s2,s3) +
    - s5 -> 4-shot translation with input (s1,s2,s3,s4) +
    - s6 -> 5-shot translation with input (s1,s2,s3,s5) +
    - s7 -> 6-shot translation with input (s1,s2,s3,s5,s6)

(Note that this ignores the costs for step 0: start sentence pool creation, step 1: start sentence selection, step 3: MT results aggregation. It also assumes a single start sentence, otherwise costs will be multiplied by number of start sentences (assuming same interpolation sentences).)

2. The method is very computationally expensive compared to the baselines. This is true even for much stronger baselines (e.g., n-shot with relatively large n) that are not included in the paper. The authors mention "IntGrad MT introduces some computational overhead", but this seems a severe understatement. The computational overhead of the method is not properly analyzed, but this analysis should have been included to help readers understand these trade-offs.

3. As acknowledged by the authors, their method only works when English is the source language. This is not necessarily a prohibitive issue, but when we have few-shot (with larger n) available as alternative, it is unclear why one would use this more complicated method over that (current results fail to convince me of the improvements over stronger baselines, see first point).

4. I would not call LLMs translation capability "inherent", as you do in the abstract. See e.g. https://aclanthology.org/2023.acl-long.524/, which shows that translation capabilities are mostly due to (incidental) bilingualism, i.e., parallel data.

5. While the authors briefly discuss related work, the discussion lacks depth. It is not clear how the authors method is different from this discussion. (It can be inferred by a reader who is knowledgeable about the related works, but that is insufficient.)

6. A lot of clarity and typo issues, I list some below:
    - Figure 1 is a bit unclear, nowhere is mentioned what blue/red colours mean. I think I can infer it but it is not clear enough.
    - line 89: "To what sentences IntGradMT can be effective?" awkward sentence, rephrase to "For which sentences is IntGradMT effective?"
    - line 135: "which is a prompting technique asks model" -> "that asks the model"?
    - line 214: "list of translation" -> "list of translations"
    - lots of issues with missing space before opening parenthesis, e.g. line 235: "Nemo(Mistral-Nemo"; line 281: "selection(step 1)"
    - line 372: "theee" -> "three"

### Questions
1. Baseline Comparison and Computational Cost
    - Given the computational complexity of your method (detailed below), why weren't stronger baselines included? At minimum, n-shot baselines with comparable computational cost should be included.
    - How do you justify this overhead compared to simpler approaches?
    - Why weren't comparisons made to other non-finetuning MT improvement methods mentioned in Section 2.1?

2. Limited Language Direction
    - The method only works with English as the source language. Given this limitation and the high computational cost, could you elaborate on why one should choose this method over simpler approaches like n-shot with larger n?
    - Have you explored why the method fails for non-English source languages?

3. Technical Claims and Methodology:
    - The paper describes LLMs' translation capability as "inherent", but research (e.g., https://aclanthology.org/2023.acl-long.524/) suggests it comes from parallel training data. Could you clarify this characterization?
    -How is your method fundamentally different from previous work? The related work section doesn't clearly distinguish your contributions.

4. Clarity Issues
    - Could you clarify the meaning of colors in Figure 1?
    - Several writing issues need addressing (e.g., "To what sentences IntGradMT can be effective?", missing spaces before parentheses, etc.)
    - Can you provide more precise technical details about the aggregation method and threshold determination?

5. Implementation Details:
    - How stable is the method across different interpolation models?
    - What is the variance in performance across multiple runs?
    - How were hyperparameters (like the number of start sentences) chosen?
    - It is likely that the reference-free QE model you're using does not work very well for low-resource languages. How do you inspect to what extent this is an issue?

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The paper introduces a method called IntGrad MT that alleviates performance of LLMs for MT in low-resource language pairs using parametric sources of information with no extra training. In IntGrad they utilise 2 techniques, sentence interpolation and gradual MT. Former starts from an easy seed sentence and keeps on changing it till it reaches the source sentence (sentence to translate) and latter keeps on translating each one while using the previous translation results as few-shot examples for the current sentence. The authors show efficacy of their method by conducting thorough experiments and ablation studies.

### Strengths
- The paper is very clearly written and easy to follow except for the Output selection part.
- Their approach doesn’t require extra training and utilises existing parametric information. 
- The experiments performed by authors are quite extensive including ablation study for finding optimal combination of strategies and sentence interpolation analysis.
- The authors used xCOMET over COMET which has error span detection capabilities. I highly recommend them showing how their approach was better when compared to baselines by showing error spans of the baselines (which their own approach won’t have).
- Figure 1 gives a good intuition of IntGrad MT and Figure 2 is really helpful for visualising Gradual MT

### Weaknesses
- Output selection could have been worded properly. Although it is clear from the paragraph that “thr” is a pre-processing step and “delta” is a post-processing step but:
    - “thr” should come before sentence interpolation in Figure 3 which is not the case
    - “delta” should come before aggregation? 
    - From what I understood - start sentences -> thr -> interpolation -> Gradual MT -> delta -> aggregation -> Final translation
    - If this is correct then you please modify Figure 3 and meaning of output adoption strategy
- Even though this approach doesn’t require training but interpolation step is quite costly as it uses a 72B LLM to generate sentences to reach end sentence which increases the latency of the overall translation. It is evident that "thr" reduces number of interpolated sentences but it increases the latency by how much compared to the baselines? How do you plan to reduce the latency of the Interpolation model?
- For few-shot baseline, authors use 3 MT pairs but for their thr + delta approach I can see from Table 3 that Interpolation and GradMT were executed for 45% sentences out of which number of selected outputs were 279 from which final MT result is produced. Do you ensure a fair comparison between you method and the 3-shot baseline, given the difference in the number of examples used.

### Questions
- I had some trouble understanding “Start Sentence Pool Creation”. Start sentences are essentially in English, then why were they translated as mentioned in Section 4.1?
- I can see the authors used Euclidian distance throughout the paper but cosine similarity is used more frequently when measuring similarity between 2 sentences using SBERT in many NLP papers. Why did the authors use Euclidian distance and not cosine similarity metric?
- Please change the colour of MQM(MetricX) for Figure 3 as it looks quite washed out when printed.
- I wanted to know if having bigger context length had any detrimental affect on the final results in Table 3?


Typos and formatting-
- Line 241: SBERT similarity
- Line 471: measure
- Table 1, 2, and 3 have a missing \toprule. Missing \bottomrule for Table 2
- Figure 3 says “n” start sentences but caption says “k” start sentences

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
The paper introduces IntGrad MT, a technique that enhances the machine translation capabilities of Large Language Models without additional training. It utilizes Sentence Interpolation and Gradual MT to explore the intermediate processes in translation by creating a sequence of translations that progressively increase in complexity. This additional context is used to improve the final translation. Empirical results demonstrate that this method improves translation quality, particularly for low-resource languages such as Hindi, Swahili, Bengali, and Marathi, as measured by xCOMET scores. IntGrad MT offers a practical way to elevate translation performance without relying on extra training data.

### Strengths
- The idea of introducing intermediate steps (additional interpolated demonstrations) to guide the LLM in improving translation is interesting. Like CoT and ICL, this represents a training-free translation strategy that achieves enhancement.

- The proposal was tested in experiments, particularly showing significant improvements in low-resource translation scenarios.

- The authors provide insightful analysis and ablation studies in the paper, which are beneficial for understanding the effectiveness of the approach.

### Weaknesses
However, there are some weaknesses:

- Lack of strong baselines. The paper does not include some strong multilingual LLMs in the comparison, such as the recent TowerInstruct. Comparisons are only made with its dependent LLMs, which limits the scope of evaluation.

- The left result in Figure 5 does not convincingly support the claim that "the interpolation successfully bridges the two sentences". Question: Could you provide more explanation? The example shown in Appendix B.1 on Page 17 indicates that the sixth interpolation still shows a considerable difference from the end sentence in word-level similarity. What is the COMET score for that sentence?

- Unfortunately, the evaluation was conducted on sentence-level tasks, whereas LLMs excel in document translation. It would be beneficial to see if the interpolation approach is also effective with document-level test data.

- One of my concerns is that there is no analysis or evaluation of computation time during inference. The authors should report this information to make the study more comprehensive.

- The authors claim that their method addresses difficult translation tasks by expanding the areas where the model performs well. However, they did not clearly define what constitutes a 'difficult' level, and there is a lack of an ablation study examining the translation quality among sentences with varying difficulties. Additionally, it is unclear whether it is possible to enhance the model with just a few interpolated sentences for the more challenging cases.

### Questions
See the weaknesses

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper assumes that a source sentence (called end sentence) has paired sentences (called start sentences) that are easy to translate or have gold translations and these paired sentences with corresponding translations can be used as translation samples in order to prompt the model to translate the source sentences. To use this assumption, the authors presented a multi-step prompting framework:

- Step 0:  use an existing dev set to create a start sentence pool
- Step 1:   select start sentences from the pool based on similarities.
- Step 2:   calculate the interpolation sentences between the source sentence (or end sentence) and the paired start sentences and translate all interpolation, start sentences, and the end sentence.
- Step 3: Chain all sentences and translations into multiple prompts according to the start sentences.
 - Step 4: select the final translation from the results of these prompts.

Experiments are conducted on FLORES-200 for 7 languages.

### Strengths
1. The idea is interesting. 
2. The paper is clear. I can easily follow the paper.
3. The authors analyze each step in the ablation study.

### Weaknesses
1. Evaluation is not fair. The authors use xCOMET to select translations and evaluate the final results on the test set. The final results are biased to xCOMET or COMEwiki scores. The authors have to provide scareBLUE scores to justify the effectiveness. Meanwhile, since you use all dev sets as your start sentences pool, it is not fair to other models. How do you select 3 examples for your 3-shot translation? What is your selection method ? Can you try an experiment where we select 3 similar sentences from dev set with the source sentences as the 3 examples?

2. Efficiency is poor. The method required too many prompts and tokens to finish the translation task. For example, in Table 2, the model requires 3 prompts that generate multiple sentences for translation. Can you provide a study of efficiency? 

3. Experiments are limited. The authors claim the framework works for low-resource languages. However, only limited experiments are conducted in Table 1.

### Questions
1. Refers to Weaknesses.
2. How many interoplation sentences generated on average for one translation?

### Soundness
1

### Presentation
3

### Contribution
1

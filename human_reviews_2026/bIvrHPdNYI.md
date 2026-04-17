# Frontier LLMs Still Struggle with Simple Reasoning Tasks

- Decision: Reject
- Scores: 2, 4, 2, 6

## Abstract
While state-of-the-art large language models (LLMs) demonstrate advanced reasoning capabilities---achieving remarkable performance on challenging competitive math and coding benchmarks---they also frequently fail on tasks that are easy for humans. This work studies the performance of frontier LLMs on a broad set of such "easy" reasoning problems. By extending previous work in the literature, we create a suite of *procedurally generated* simple reasoning tasks, including counting, first-order logic, proof trees, and travel planning, with changeable parameters (such as document length. or the number of variables in a math problem) that can arbitrarily increase the amount of computation required to produce the answer while preserving the fundamental difficulty. While previous work showed that traditional, non-thinking models can be made to fail on such problems,  we demonstrate that even state-of-the-art thinking models consistently fail on such problems and for similar reasons (e.g., statistical shortcuts, errors in intermediate steps, and difficulties in processing long contexts). To further understand the behavior of the models, we introduce the unpuzzles dataset, a different ``easy'' benchmark consisting of trivialized versions of well-known math and logic puzzles. Interestingly, while modern LLMs excel at solving the original puzzles, they tend to fail on the trivialized versions, exhibiting several typical failure patterns related to memorizing the originals. We show that this happens even if the models are otherwise able to solve problems with different descriptions but requiring the same logic. Our results highlight that out-of-distribution generalization is still problematic for frontier language models and the new generation of thinking models, even for simple reasoning tasks, and making tasks easier does not necessarily imply improved performance.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper looks at “simple but tedious” reasoning: a set of procedurally generated tasks (logic eval/negation, counting, proof-tree math, travel planning) plus UNPUZZLES and context-shift (CS) variants. The main message is that frontier/“thinking” LLMs still crumble when problems get longer or when surface form shifts, even when conceptual difficulty is fixed.

### Strengths
1. The datasets are simple, controllable, and easy to extend; I like that you can scale tediousness without changing the core concept.

2. UNPUZZLES + CS is a clever way to probe memorization and context corruption.

### Weaknesses
1. **Novelty/positioning.** Most insights are already known (long-context distraction, shortcutting, tokenization/copy issues, weak state tracking). The paper acknowledges related work but doesn’t push understanding deeper. As a result, the contribution is hard to pin down beyond a well-engineered bundle of tasks.


2. **Structure.** Section 3 (procedural tasks) and Section 4 (UNPUZZLES) read like two separate papers. If UNPUZZLES is mainly about memorization/educated guesses, say that explicitly and connect it to Section 3’s failure modes. Right now the bridge is thin.


3. **Results presentation.** The paper talks about trends “across LLMs,” but there are no per-setting cross-model aggregates (mean±std, or ranks). Readers have to scan tables model by model. Some table ordering is counter-intuitive (e.g., Logic Evaluation has (n=16) above (n=8)); one expects easier settings to appear first. Small thing, but it slows reading.


4. For each claim in §3.2 (Failure Analysis), the factors aren’t well controlled; for example, context length co-varies with tokenization effects. Consider padding prompts to a fixed length to isolate variables.


Useful engineering and a nice unpuzzle/CS idea, but the novelty is limited and several headline claims aren’t causally nailed down. With tighter controls, cross-model aggregates, and a cleaner §3↔§4 bridge, this could become a strong empirical paper.

### Questions
1. Line 363 claims that 20-character random names should hurt due to multi-token variables. However, the Logic-Negation results don’t consistently reflect this: at (d=12), 3 models favor random-20, 4 favor movies, 1 ties; at (d=8), random-20 wins 3 vs. 5 for movies; at (d=4), there are 3 ties, with random-20 winning 2 vs. 3 for movies. Could you clarify whether the degradation is statistically significant and, if so, how you attribute it specifically to multi-tokenization rather than other factors?

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
5

### Summary
The paper tested state-of-the-art LLMs on a suite of procedurally generated tasks, including 1) counting, 2) first-order logic, 3) proof trees, and 4) travel planning. And a dataset UNPUZZLE that contains difficult puzzles, and their trivialized and context-shifted versions. With these tasks, the paper argues that reasoning LLMs exhibit similar drawbacks as non-reasoning models when solving these types of tasks.

### Strengths
The paper shows that state-of-the-art LLMs still struggle on simple reasoning tasks by constructing procedurally generated reasoning tasks and a human-annotated small puzzle dataset. It revealed several failure modes that are common among frontier LLMs.

### Weaknesses
The novelty of the paper is limited: the idea that frontier LLMs struggle with "simple"/"tedious" tasks has been studied in several works [1, 2, 3]. The advantage of this paper compared to previous works, e.g., the procedured generation, so that the benchmark is less prone to data contamination, does not provide new information about the fact that LLMs struggle with these simple tasks.

[1] Kazemnejad, Amirhossein, et al. "The impact of positional encoding on length generalization in transformers." Advances in Neural Information Processing Systems 36 (2023): 24892-24928.
[2] Yehudai, Gilad, et al. "When Can Transformers Count to n?." arXiv preprint arXiv:2407.15160 (2024).
[3] Dziri, Nouha, et al. "Faith and fate: Limits of transformers on compositionality." Advances in Neural Information Processing Systems 36 (2023): 70293-70332.

### Questions
1. The paper studied multiple closed-source models; on the proposed benchmark, these models have very different behaivours. It would be interesting and useful if more analysis were done on why these models differ from each other. E.g., for the models that generally perform better than average, is it more likely due to the model being trained with more data of this type of task, or due to the model having better reasoning generalizability as compared to other models?

2. What practical guidelines can we draw from the analysis of frontier model behaviours on the proposed benchmark?

3. The UNPUZZLE dataset suggests that the original well-known puzzles may already have appeared in the training data, thus leading to high performance. Similar investigations of data contamination were conducted in recent works, e.g., [1], where the method is simple: let the LLM complete a partial question, and see if it can successfully complete it and then answer. Can similar experiments be done on the original puzzles and see if the frontier LLMs already know the puzzle?

4. Why do context-shifted puzzles consistently have better performance compared to their unpuzzle version? If the reasoning behind this is: the unpuzzle version is more affected because it's similar to the original contaminated puzzle (lower score due to memorization of a different answer), while the context-shifted version is less affected. It seems to suggest that the LLMs do possess some degree of reasoning generalizability, as long as it's not affected by memorization? More discussion on the context-shifted experiments can be helpful for understanding this behaviour.

[1] Wu, Mingqi, et al. "Reasoning or memorization? unreliable results of reinforcement learning due to data contamination." arXiv preprint arXiv:2507.10532 (2025).

### Soundness
3

### Presentation
2

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
This paper introduces a suite of procedurally generated simple reasoning tasks, including counting, first-order logic, proof trees, and travel planning, with changeable parameters and the UNPUZZLE dataset. Using these datasets, it performs an in-depth failure analysis across multiple SOTA LLMs, including thinking models.

### Strengths
In-depth error analysis.

Nice idea with the unpuzzle puzzles, especially the context-shifted unpuzzle, which allows for a more detailed failure analysis and attribution.

### Weaknesses
No human evaluation. You write “quite easy for humans,” but do not test this statement. You make this a central part of the paper, yet it remains untested.

The story should be cleaner. Sections 3 and 4 feel related, but currently they should be two papers. I suggest thinking more about how the paper is presented to tie these two together. 

“One suggestion from our paper is that LLMs should be evaluated not only by the most difficult problem they can solve, but also by the simplest problem they struggle with.” This does not seem like a new idea, and is very much already part of the literature, as partially shown in the related work (see later comment).

No error bars. The experiments ought to be done multiple times to ensure the results are significant. 

You should have tested newer models such as GPT5. The model selection is fine, but given the progress, it would be nice to see as recent models as possible.

“Similarly to other recent works, our results suggest that LLMs mimic training data rather than performing true reasoning, making it relatively easy to find out-of-distribution problems where the models fail.” What is then the novel contribution of this paper? While I like the idea of the paper, I find the evaluation and writing/presentation below the requirements for an ICLR paper. 

While it is mentioned that "every model we tested performed better on the context-shifted unpuzzles than the original ones, indicating that failure was at least in part due to memorization of the original puzzle," this point is not made strongly enough and undermines the rest of the claims, and sometimes states the findings without context, which can be misleading. For example, in the abstract it is stated that:“... several systematic failure patterns related to memorizing the originals,” but a later statement does not connect strongly enough to this observation and instead sounds like a more general claim: "Our results highlight that out-of-distribution generalization is still problematic for frontier language models and the new generation of thinking models, even for simple reasoning tasks, and making tasks easier does not necessarily imply improved performance."


Minor: 

The paper is missing some related works, such as https://arxiv.org/pdf/2407.06581 and https://arxiv.org/pdf/2504.12256.

You should remove Opus from the author list in Opus and Lawsen (2025). See the footnote in https://arxiv.org/abs/2506.09250v2.
Please include and reference examples of the failure cases listed at the end of Section 3.2.

Typos etc.:

Line 372 is missing a space.

Line 375 should be: See Appendix C for more details, dataset creation instructions, and some examples.

Line 464: .ressive results across a variety of

These are merely some highlights. Overall, the presentation of the paper is relatively poor.

### Questions
You have human annotators for unpuzzles and write “Each answer was assessed by a single annotator, or by consensus of all annotators if marked ambiguous.” For how many questions was this an issue?

You state that: "One of our goals was to design tasks that are easy (albeit tedious) for humans, but become unsolvable by frontier models when the difficulty parameters are large enough."
Could this instead be interpreted as a measure of robustness rather than mere difficulty?

You also argue that: "In general, if each computational step has a small probability of error, increasing the number of steps exponentially increases the probability of overall failure, even if the model follows the correct approach to calculating the solution."
Is the per-step failure rate somewhat constant, or is it clearly dependent on the number of previous steps? This again suggests that the model is not “worse” at longer tasks, just not robust.

Humans frequently make typos or small calculation mistakes in a tedious but easy math task. How do these results compare to human failure rates on tedious tasks qualitatively and quantitatively?

### Soundness
1

### Presentation
1

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper evaluates state-of-the-art LLMs, including recent thinking models (o1, o3, DeepSeek R1), on procedurally generated "easy" reasoning tasks and introduces the UNPUZZLES dataset of trivialized logic puzzles. The authors demonstrate that even thinking models fail when task parameters increase tediousness while preserving fundamental difficulty, and that models paradoxically perform worse on simplified versions of puzzles they can solve in their original form.

### Strengths
S1. The paper provides a timely and comprehensive evaluation of thinking models on simple reasoning tasks, filling a gap in the literature since most prior work focused on earlier model generations. The inclusion of o1, o3, DeepSeek R1, and Gemini thinking variants makes this a valuable reference for understanding current capabilities.

S2. The procedurally generated task suite with tunable parameters (document length, tree depth, number of cities) is well-designed and allows systematic study of how performance degrades with increased computational requirements while difficulty remains constant. This methodology is reproducible and extensible to future models.

S3. The UNPUZZLES dataset introduces a novel and counterintuitive phenomenon where models fail on trivialized versions of puzzles they solve correctly. The "reasoning delirium" failure mode, where models apply memorized complex solutions to simple problems, is an important finding that reveals limitations in how models generalize.

S4. The context-shifted unpuzzles provide clever experimental control by demonstrating that models can solve problems with identical logic but different surface form, isolating memorization as a key factor in unpuzzle failures rather than fundamental reasoning inability.

S5. The failure analysis systematically categorizes error patterns (statistical shortcuts, error accumulation, long context difficulties, poor OOD generalization) with concrete examples from model outputs, making the findings actionable for future research.

### Weaknesses
W1. Several tasks conflate multiple difficulty dimensions simultaneously, making it unclear which factors drive failures. For example, increasing tree depth in logic tasks increases both the number of reasoning steps and context length. More controlled ablations isolating individual factors would strengthen causal claims about failure modes.

W2. The UNPUZZLES dataset, while introducing an interesting phenomenon, is limited in size (97 puzzles) and requires manual construction and evaluation. This limits statistical power and reproducibility. The reliance on human annotation for correctness and "context corruption" introduces subjectivity, with inter-annotator agreement not reported.

W3. The paper acknowledges that thinking models show quantitative improvements even while maintaining qualitative failure trends, but does not deeply engage with what these partial improvements mean. If o1 achieves 60% accuracy where GPT-4o achieves 30%, this suggests meaningful progress even if imperfect, yet the framing emphasizes failure.

W4. The practical implications of these findings remain unclear. The tasks, while procedurally elegant, are somewhat artificial (counting words in paragraphs, logic formulas with random variable names). Whether failures on these tasks predict failures on real-world reasoning problems is not established.

### Questions
Q1. The concurrent work by Shojaee et al. (2025) is noted to have design flaws. Could you elaborate on how your experimental design avoids similar issues, particularly regarding the claim that their problems may be unsolvable or ignore token limits?

Q2. For UNPUZZLES, how sensitive are results to the specific method of trivialization? If you created multiple different simplified versions of the same puzzle, would models fail consistently across all simplifications?

Q3. Some failure modes (tokenization issues, specific formatting expectations) seem potentially addressable through engineering improvements. Can you estimate what fraction of failures stem from such fixable issues versus fundamental reasoning limitations?

Q4. The thinking models often show substantial improvement over non-thinking variants even if not achieving perfect performance. How should the community interpret this partial progress? Does it suggest the approach is fundamentally sound but needs scaling, or that different approaches are needed?

Q5. In the UNPUZZLES evaluation, models are assessed once per puzzle. Given the known variability in LLM outputs, would results differ substantially with multiple samples and majority voting?

### Soundness
3

### Presentation
3

### Contribution
3

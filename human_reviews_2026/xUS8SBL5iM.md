# Reusing Pre-Training Data at Test Time is a Compute Multiplier

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6, 6

## Abstract
Large language models learn from their vast pre-training corpora, gaining the ability to solve an ever increasing variety of tasks; yet although researchers work to improve these datasets, there is little effort to understand how efficient the pre-training apparatus is at extracting ideas and knowledge from the data. In this work, we use retrieval augmented generation along with test-time compute as a way to quantify how much dataset value was left behind by the process of pre-training, and how this changes across scale. We demonstrate that pre-training then retrieving from standard and largely open-sourced datasets results in significant accuracy gains in MMLU, Math-500, and SimpleQA, which persist through decontamination. For MMLU we observe that retrieval acts as a ~5x compute multiplier versus pre-training alone. We show that these results can be further improved by leveraging additional compute at test time to parse the retrieved context, demonstrating a 10 percentage point improvement on MMLU for the public LLaMA 3.1 8B model. Overall, our results suggest that today's pre-training methods do not make full use of the information in existing pre-training datasets, leaving significant room for progress.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper explores an interesting idea: reusing pre-training data at test time through retrieval augmentation to boost the performance of large language models. The authors demonstrate that this method acts as a "compute multiplier," meaning it achieves performance gains that would otherwise require significantly more computational resources for pre-training alone. They test this on several benchmarks like MMLU, Math-500, and SimpleQA, showing that retrieval, especially when combined with techniques like self-consistency and reranking, leads to substantial improvements. The key takeaway is that current pre-training methods don't fully squeeze all the juice out of the vast datasets they use, and test-time retrieval is a powerful way to unlock that leftover potential.

### Strengths
1.  The core concept of treating test-time retrieval as a "compute multiplier" is a really neat and effective way to frame the value of retrieval-augmented generation.

2.  The experiments are quite thorough, covering multiple datasets and model scales, which provides a solid empirical foundation for the claims.

3.  The analysis of how retrieval impacts different MMLU categories (like STEM vs. humanities) offers some cool insights into where this technique helps the most.

### Weaknesses
1.  The study focuses on models pre-trained and retrieved from the *same* dataset, which might not always be the case in real-world applications.

2.  While the gains are impressive, the paper notes that the effectiveness of retrieval as a compute multiplier diminishes as the base model size increases, which could limit its long-term utility for future, even larger models.

3.  The techniques explored, like retrieval and self-consistency, add significant computational overhead at inference time, and the trade-offs aren't discussed in much detail.

### Questions
1.  You showed that retrieval from the pre-training corpus is very effective. How do you think this approach would perform if the retrieval database was a completely different, specialized corpus that the model has never seen before? Would the "compute multiplier" effect still hold?

2.  In your analysis (Table 4), the performance lift from subset retrieval is surprisingly close to that of full retrieval for many MMLU subjects. Why do you think a smaller, targeted subset of the data provides such a comparable boost to having the entire dataset available for retrieval?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors pose the question of how well pretrained LLMs can extract information from their pretraining data. To study this, they propose using a RAG setup where the pretraining corpus itself is used for retrieval at test time. Through a broad set of experiments, they report notable accuracy gains on MMLU, MATH-500, and SimpleQA, and argue that these improvements indicate that current pretraining does not fully exploit the information in the data. They further claim that retrieval acts as a compute multiplier, a ~5x gain on MMLU, since, under a fixed FLOP budget, their RAG approach achieves the same accuracy as the base model with fewer FLOPs. They also show that adding test-time scaling via self-consistency leads to additional improvements.

### Strengths
- The paper is well written, and the methodology is clearly explained.
- The central question of "how much of the pretraining data is left unexploited after pretraining" is interesting and well motivated, and can help understanding the strengths and limitations of current LLMs.
- The authors conduct a substantial number of experiments, systematically testing different aspects of their claims.
- The pretraining experiments of Section 4, particularly those after decontamination, indeed suggest that pretraining alone does not always capture all patterns present in the pretraining corpus.

### Weaknesses
- I am not sure if the compute multiplication arguments are fair. To my understanding, the authors do not take into account the FLOPs associated with the retrieval and reranking components of their RAG pipeline when comparing against the base model’s pretraining FLOPs. As a result, the “under fixed compute budget” claim becomes somewhat misleading, and the reported multipliers (sometimes ~10x) are possibly inflated.
- In addition to my previous argument, I am not fully convinced that the main message of the paper is justified. While the results do show consistent improvements with retrieval, it is not clear whether these gains truly reflect “unused” information from pretraining, or simply the effect of providing additional access to relevant text at test time. Thus, I believe that more experiments in a more controlled setting would be necessary to support this hypothesis.

### Questions
- Do you believe that the results would change drastically if the RAG corpus was different than the pretraining corpus?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This submissions studies the benefit of retrieval augmented generation in terms of how it compares to increased pre-training. It shows that retrieval potentially yields benefits comparable to significant multiples of (pre-)training time. A number of experiments analyze various related aspects of pre- and test-time training of LLMs.

### Strengths
The paper argues an interesting and highly-valuable point: well-done retrieval can comparatively benefit performance much more than additional generic model training. The idea of presenting gains in terms of "compute multipliers" is convincing (although the underlying simple sigmoid model deserves more attention) . The paper also provides evidence of additive benefits with other test-time procedures, which is promising. There is a massive amount of experiments here, supporting these findings and further analyses. The choice of datasets is clear and reasonable.

### Weaknesses
The main weakness of the paper is the general clarity re. exactly what is done. One blatant example is the Experimental Setup section, esp. 3.2: The short paragraph provides a straightforward and high-level description of the retrieval process, but no description of how the retrieved documents are used -- presumably a RAG-style process, but absolutely no detail is provided. Given that this is mainly an experimental paper, it is absolutely necessary that experimental details are provided, at minima with references, so that the reader knows what is done. There is also some confusion about what model(s) was/were used. The text mentions at various places 6.4B models, 8B models and 12.6B models -- it is not always clear which was used where (and why). 

There is a dizzying amount of experiments done in Section 4, 5 and 6, providing a lot of experimental evidence. Unfortunately, the amount of description for each experiment is minimal, often only a couple of sentences with tables or graphs that are poorly described. It is often unclear what is reported (e.g. what metric is used). And no error bars or assessment of experimental uncertainty are shown.

Several interesting findings arise from the many paper's experiments. However some sound more like speculative guesses, e.g "might require logical/abstract reasoning" (l.254-255), "the model ignores additional context, rather than additional context being misleading"(??, l.284-285), "it is likely that there is substantial overlap" (l.362-363). Even the overall message from the paper, that pre-training does not make "full use" of the data, is unconvincing and not unambiguously supported by the data. For example, observations may just be the unsurprising consequence of leveraging test-specific data versus using a general-purpose model out-of-the-box.

### Questions
Gains for SimpleQA are striking (Fig. 1) -- so large that they don't really fit the "compute multiplier" narrative. What is going on with retrieval on that benchmark?

How were the five compute budget points chosen? Also it was not 100% clear whether model size and architecture are kept fixed as compute grows.

When retrieval is done, is it done on the full dataset, or only the part used for pre-training?

What exactly is "subset retrieval"? The description on l.159-160 is not sufficient.

l.161: "a piecewise function" -- this by itself is not very informative. Arguably most curves on Math500 are also "piecewise".

How are the variance reduction techniques used in the context of this paper (l.329-330)?

This is a matter of taste and author choice of course, but there is no conclusion beyond future work?

### Soundness
3

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
3

### Summary
The paper asks how much useful signal in today’s open pre‑training corpora is left unexploited by pre‑training itself, and if it can be reclaimed at test time. Using a public 8B reader (Llama‑3.1 8B instruct) together with retrieval + reranking + self‑consistency yields +10.5 points on MMLU, +15.7 on Math‑500, +6.2 on GPQA, and 74% on SimpleQA. On MMLU, they fit a bounded sigmoid to the base models and report that adding retrieval behaves like a ~5× pre‑training compute multiplier on average (declining with scale).

### Strengths
- The design (pre‑train on a corpus, then retrieve from exactly that corpus at test time) probes how much of the corpus’ information is not captured parametrically. This bridges two active lines of work (RAG and inference‑time compute) in a controlled, data‑centric way.
- Paper does not hand‑wave contamination: it (a) quantifies overlap, (b) shows that decontaminated retrieval still helps (substantially for MMLU), and (c) surfaces the surprisingly large portion of benchmark content present in common open corpora
- “Inter‑document self‑consistency” diagnostic is clever and novel.
- Combination of retrieval, reranking, and self‑consistency is methodologically clean and yields strong additive gains

### Weaknesses
- MMLU has documented label and quality issues; using it as the sole basis to translate accuracy into compute multipliers risks reporting an artifact of the fit or of dataset flaws.
- Decontamination by token n‑gram overlap (16‑gram for MMLU, 26‑gram for Math‑500) is a good start but cannot remove paraphrastic or templatic leakage. Because the retrieval store is identical to pre‑training corpora, any residual overlap inflates the measured gap between “base” and “+retrieval.”
- The “compute multiplier” compares training FLOPs without charging anything to indexing, storage, retrieval/reranking inference, prompt expansion, or self‑consistency samples. In many deployments, inference cost dominates, and searching over such a large corpora is likely impractical at best and impossible at worst, due to the increased search compute cost. Without this analysis (of a break even point between inference and training flops), the multiplier is misleading operationally.
- The base models in §4 appear to be evaluated without retrieval but also without self‑consistency, whereas §5 compares an 8B reader baseline w/ CoT to retrieval + reranker + self‑consistency (+ MMR/bagging). While Table 5 includes a “self‑consistency only” baseline for the 8B reader, the earlier compute‑multiplier analysis relies on single‑pass answers for the base models. A more apples‑to‑apples comparison would allow multiple sampled chains and majority‑vote for the base models as well (possibly with no extra context) to isolate how much of the gain is from extra sampling vs. retrieval.
- The narrative proposes that retrieval may aid reasoning by “expanding context,” but the current evidence is correlational (for the stem benefits more than humanities claim). Are the largest per‑subject gains driven by questions where the retrieved passages contain the answer string (knowledge store) vs. passages that supply definitions/lemmas enabling reasoning (process aid)? A per‑question audit (string‑match / entailment / indirect support) would sharpen the mechanism story.

### Questions
- Add MMLU‑Pro/Redux, GSM8K/BBH (with grounding demands), and limited‑access tasks where retrieval should not help (negative control)
- How does the 5× multiplier change under (a) an unconstrained logistic fit; (b) a power‑law fit; and (c) fits trained only on the three middle scales (to reduce anchor effects)? Please report confidence intervals.
- Provide curves for k, reranker family, query rewriting, and inter‑document consistency at k>1 to situate the proposed VR/SC stack relative to standard RAG toolings

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studies how much existing LLM models utilize the pretraining data by letting models retrieve them during test time. The result shows that this can greatly improve the accuracy of various benchmarks, and thus imply the potential of the pretraining data might not be fully explored.

### Strengths
1. Their observation if able to be held to a larger scale might be interesting and could have some valuable guidance to the community.

### Weaknesses
1. The scale of the experiment is limited. It seems less surprising if smaller models could not fully utilize the pretraining data.
2. If the authors want to propose this as a method to be used rather than just an experiment for some observation, then they should compare it with other test time scaling methods.

### Questions
1. Could the authors show that the performance of existing off-the-shelf models can be improved by the proposed retrieval pipeline? This does not require pretrain so the author should be able to experiment with larger models. This could also rule out the possibility that the models used in the experiment might not be pretrained properly.

### Soundness
3

### Presentation
3

### Contribution
3

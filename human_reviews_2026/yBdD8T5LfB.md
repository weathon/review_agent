# Relational Knowledge Distillation Using Finetuned Function Vectors

- Avg Score: 2.50
- Decision: Reject
- Scores: 2, 4, 2, 2

## Abstract
Representing relations between concepts is a core prerequisite for intelligent systems to make sense of the world. Recent work using causal mediation analysis has shown that a small set of attention heads encodes task representation in in-context learning, captured in a compact representation known as the function vector. We show that fine-tuning function vectors with only a small set of examples (about 20 word pairs) yields better performance on relation-based word-completion tasks than using the original vectors derived from causal mediation analysis. These improvements hold for both small and large language models. Moreover, the fine-tuned function vectors yield improved decoding performance for relation words and show stronger alignment with human similarity judgments of semantic relations. Next, we introduce the composite function vector - a weighted combination of fine-tuned function vectors - to extract relational knowledge and support analogical reasoning. At inference time, inserting this composite vector into LLM activations markedly enhances performance on challenging analogy problems drawn from cognitive science and SAT benchmarks. Our results highlight the potential of activation patching as a controllable mechanism for encoding and manipulating relational knowledge, advancing both the interpretability and reasoning capabilities of large language models.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces a simple approach to finetuning existing function vectors (FV) for relation tasks, known as finetuned function vectors (FFV). The approaches requires few samples and improves over several tasks that FVs are evaluated on. Furthermore, they also contribute an approach to combine FFVs into composite function vectors (CFVs) that captures analogical knowledge over multiple types of FFVs. They apply their approach to evaluation task cross-domain human similarity judgments.

### Strengths
1. The authors show that their FFV approach shows consistent increases in performance over the vanilla FV approach.
2. The paper corroborates findings from [1] showing random initialization for training function vectors has weaker performance than initialization via CMA (section 3.2 of [1]).
3. The application of the CFV is interesting as a direction to study human similarity judgments.

[1] Todd, Eric, et al. "Function Vectors in Large Language Models." *ICLR* (2024).

### Weaknesses
1. The work is initially motivated from the point of view of efficiency (lines 40-44), but no citations are introduced (or preliminary experiments to support the hypothesis) in lines 42-44 when referencing these factors -- line 43 "Many factors contribute to the inefficiency of the AI models in solving reasoning problems, but we suspect that one major issue is the lack of explicit representations for relations and the inability to accumulate the relational knowledge during learning". Efficiencies needs to be defined empirically and given a formal definition (is it efficiency via model throughput, efficiency via tokens generated, or others?) since the crux of the argument is related to efficiency.
2. It's unclear from the beginning of the paper what exactly is a relation task, even though the notation becomes slightly more clear towards the end of the paper. A more formal definition (even fairly brief) would be helpful as a preliminary. Many of the existing FV tasks are in some sense "relations" (antonyms, synonyms, etc.) and so the distinction is not clear, because the same "relations task" term is used later to refer to both the antonyms/synonyms task and the analogies task.
3. From the introduction, it is unclear why one would need to "improve" the performance of existing FV because the task (analogy task) is not introduced early enough in the paper in the motivation. It would be tremendously helpful to explain why existing FVs are not adequate enough to support these analogies (for instance, through some error analysis of FVs on these existing analogical tasks) and then show that additional training to FVs are required, because in some sense, showing that FFVs are stronger than FVs is obvious since finetuning will always improve over inference-only approaches.
4. In general, writing needs to be clearer. For example, in the experimental setting, there are many datasets, but it is relatively difficult to parse the particular categories of tasks used to train each and every FFV and to keep track of which ones are used. It would be nice to include a clearer breakdown in that section which clearly denotes what relations are being used, or even picking a subset of FVs to evaluate and putting the rest in the appendix (like in Todd et al.).
5. It would be stronger to motivate the work by introducing Ichien et al. 2022 earlier on, since it seems that the human similarity judgments are the core evaluation application for the work and it seems to be very interesting to study human cognitive behavior using FFVs/CFVs. However, the reader gets to the application far too late to realize this is a motivation, especially since it is mostly mentioned in the conclusion as a takeaway rather than core investigative application.
6. Some repetitive writing - line 403-410 is repeated in the experiments section lines 266-273.
7. Broken link on line 618-619 for the table.

Based on the existing motivation, much of the paper may need to be rewritten which makes me inclined to give a 2, even though the experiments are reasonable.

### Questions
1. What "relation" tasks are finetuned and used when determining the weighted combination for CFVs? Although a table exists in the appendix, it should be made clear in the main body of the paper since it is difficult to understand the CFV setup (what combinations are used in the CFV for inference).
2. Analogously with the previous question, how does the CFV learn to choose the specific FFVs to focus on? Is this determined on-the-fly with the weighted combination?
3. What is the performance of the CFVs on the same simple tasks on Table 1? I would be curious to see whether across the tasks, the behavior of the CFV is consistent with an individual FFV, and if the CFV captures the weighted combination correctly.
4. In the evaluation (Zero-shot evaluation), why is the antonym FV chosen and not other FVs? Why not show the result of selecting/running other FVs as baselines, since there could be in theory up to N number of baselines, where N is the individual FV task chosen/used?
5. Why was layer 13 chosen in Section 4.3 (line 428-429)?

### Soundness
1

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
3

### Summary
The paper proposes finetuned function vectors (FFVs) that adapt the “function vector” representation through lightweight gradient updates while keeping LLM weights frozen. It further introduces composite function vectors (CFVs) that are weighted sums of FFVs computed from a source pair to steer models on one-shot analogies. Empirically, FFVs substantially outperform un-tuned FVs on zero-shot relation completion across many base LLMs, and CFVs notably improve cross-domain analogies; the authors also report stronger alignment with human relational similarity judgments.

### Strengths
1. Originality. Extends function vectors (compact task representations identified via causal mediation) with a practical fine-tuning scheme and a linear composition mechanism for analogical transfer—bridging activation steering and relational reasoning.
2. Quality. Clear problem setup; simple objective for FFVs; broad evaluation over multiple LLMs and datasets with consistent zero-shot gains and credible human-similarity correlations.
3. Clarity. The pipeline and training/inference diagrams are easy to follow; datasets and tasks are well cataloged.
4. Significance. Demonstrates controllable, inference-time manipulation of relational knowledge with measurable benefits on far analogies and human-like relational structure.

### Weaknesses
1. The work focuses on word-pair relations and four-term analogies; broader compositional or sentential relations remain open. Testing FFVs/CFVs on sentence-level or narrative analogies would strengthen external validity.
2. CFV weights are derived over a curated set of relations and fitted via an affine map; the sensitivity of performance to pool content/size and calibration is unclear. Please include pool ablations and out-of-distribution relations.
3. Ablations on injection and capacity. FFVs are injected at a fixed layer; maybe the authors should provide layer sweeps, magnitude schedules, and head-wise sparsity to clarify where steering works.
4. Writing/format nits. Minor issues reduce polish (e.g., duplicated “the this study’s” phrasing in limitations; a “Table ?? ” placeholder; equation formatting around the affine map). A careful pass would help.
5. The main limitations are scope (word-pair analogies), some sensitivity/design choices around CFVs, and a need for deeper ablations and polish.

### Questions
1. Can you learn a confidence or gating signal to decide when CFV steering helps (e.g., far vs. near analogies), avoiding regressions on easy cases?
2. How stable are CFV weights under perturbations to the relation pool (drop/merge relations; add noisy ones)? Provide sensitivity and sparsity analyses.
3. Could FFVs extend to sentence-level relations or graph-structured analogies (e.g., cause-effect narratives)? Any early evidence?
4. The correlations to human judgments are strong; can you localize which heads/layers contribute most to this alignment? Ablations or causal tracing would be illuminating.
5. Please position FFVs/CFVs more explicitly against activation-addition and related steering/editing methods.

### Soundness
2

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
This paper studies how to distill relational knowledge in LLMs into steering vectors. They show that finetuning previously found function vectors (FVs) improves performance compared to the original approach on analogy-style problems. They evaluate on many datasets of analogy problems, and compare representations to human judgments, showing that fine-tuning the representations gives the best fit.

### Strengths
- The fine-tuned function vectors (FFVs) are evaluated on a wider array of analogy problems compared to previous work, including analogy tasks like SemEval and BATS.
- The finetuning/composite function vector approach seems to improve far-analogy performance consistently across the board.

### Weaknesses
- The main contribution seems to be showing that by fine-tuning function vectors, you can get better performance than the original formulation, and this is tested on a larger set analogy-style tasks. While this is nice to see, I’m not sure how motivating/exciting of a result it is. This is for two reasons: (1) fine-tuning usually improves performance on downstream tasks, and (2) the analogy tasks chosen are rather simple. 
    - Would this method work for more complex reasoning tasks such as multiple choice question answering, arithmetic, or ARC? These were recently proposed as sample subtasks of the “mechanistic interpretability benchmark” ([MIB](https://openreview.net/forum?id=sSrOwve6vb)) for instance. 
    - Another way to strengthen the argument that this function vector finetuning (or composite) approach is a competitive alternative “steering method” is to evaluate on [AxBench](https://openreview.net/forum?id=K2CckZjNy0). 
- There is little comparison to other methods/benchmarks for steering model behavior. If steering is the primary application of the method, it would be nice to see baselines like prompting, task-specific fine-tuning (of the model parameters), and others compared to FFVs in the main body of the paper. Some of this comparison to few-shot prompting is in the appendix (Tables 5, 6 and Figure 7), and briefly mentioned in the paper (eg. line 414) – where steering actually does worse than just prompting in some cases. I think including some of this information in the plots in Figure 1/Table 4 would help better contextualize the results (e.g. showing the results compared with prompting was helpful to see in Figure 7 to know how much of raw performance your FFVs are recovering).
- The composite function vector results are somewhat mixed. The results for the BATS dataset suggest that the CFV does not do much for these cases, and the only real “win” here is the far analogy case in the Green dataset. Can you explain the difference between these cases in more detail? I didn’t quite understand why these analogy sets are harder/easier.

### Questions
My main concerns I would like to see addressed are listed above, but I have a few additional questions would help clarify some confusions I had regarding them.

- The explanation and purpose of the composite function vector was a bit unclear to me. Is it just a weighted sum of all previously found function vectors that’s trying to capture “untrained” relations? What is meant by “projecting probabilities into analogy space” (line 215)?

- How does your fine-tuned function vectors approach compare to other external training methods that could induce task behavior such as Gist tokens ([Mu et al.](https://openreview.net/forum?id=2DtxPCL3T5)), soft prompts, codebooks [Shao et al.](https://openreview.net/forum?id=6axIMJA7ME3) LoRAs, etc. It seems like these approaches are similar in spirit to "learning a steering vector" for task behavior (and composition to compare to your CFV), but they weren't tested on the analogy style tasks you use.


- Minor Typos:
  - Line 234: “activities” -> activations?
  - Line 283: A right closing parenthesis is missing

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a approach to better capture the relational information in ICL by fine-tuning the Function Vector from [Todd et al, 2023](https://arxiv.org/pdf/2310.15213), which the authors dub as FFV (Fine-tuned Function Vector). The authors also show that a weighted combination of the FFVs can be used to capture the relational information from a 4-term analogical reasoning task. These composite function vectors can be used to even improve the LM performance on cross-domain analogy tasks.

### Strengths
I found the idea of composite function vectors quite interesting. It is not immediately obvious to me why how a weighted combination of FVs capturing other relations can help capture a new relation. But I find it quite fascinating that the authors found that it does in practice.

### Weaknesses
* Lack of theoretical justifications. The authors do provide some logical justifications for their approach, but I find them to be quite weak and unconvincing.
* The work do not introduce any solid novel ideas. It very much feels like an engineering effort to adapt FVs in the chosen analogy task setting. And often these design choices are not well justified.

### Questions
1. You finetune the FVs and get FFVs and show that they can achieve better performance on your shuffled and zero-shot evals. Isn't this kind of obvious? In my opinion, the most interesting claim from [Todd et al, 2023](https://arxiv.org/pdf/2310.15213) was that the LM has naturally developed specialized modules (in this cate attention heads) that capture such relational information. Of course you get better performance with finetuning. As long as the LM can perform the target task, and you are inserting your intervention (which is getting finetuned) early enough in the forward pass, you should be able to make the LM perform the task better. So, I don't find this contribution to be very significant. But I would love to hear the authors' thoughts on this.
  
    * (1a) Line 153-155: I am also very surprised that the authors used perplexity as their loss. I had the idea that cross-entropy usually considered standard for this. Perplexity is mainly used as an evaluation criterion. What was the justification for this choice?

2. Line 212-214: Not sure I understand that claim that you get "the posterior distribution". I am assuming $q$ is the query word, $r$ is the response, and $z$ is the relation or its function vector $v_z$. You get $[P(r | q, z_1), ..., P(r | q, z_n)]$ and apply softmax. And, then you say that you get $P(z | q, r)$? If my understanding is accurate the statement that you get a ""posterior distribution"" is not accurate, at least not in the Bayesian sense. You should probably say something like a proxy of P(z | q, r), which unit/elemental relations explain the target relation the most.

    * (2a) Immediately after these lines, you say that you need an affine transformation to project probabilities to the analogy. Your equation 4 shows that you get "posterior" *after* this transformation. What is the justification for this transformation? What is the affine transformation doing here?

3. As a justification for Composite Function Vectors, you just cite linear representation hypothesis and how FVs can serve has basis for constructing other relations. I find this justfication to be quite hand-wavy and simply not convincing.

     * (3a) You are making an assumption about compositionality: why should we expect a novel relation such as functionality of an apparatus can be composed as synonymy, antonymy ... ?
     * (3b) You take 100+ such "basis" relations. If you are claiming that you can compose any analogical relations that does that mean that these 100+ relations form a complete basis set?
     * (3c) One example of a far analogy you show is *blindness: sight :: poverty: wealth*. This is still an examply of antonym, which is one of your "basis" relations. I don't understand how compositionality is playing a role here.

4. I found several statements in the introduction to be quite confusing. I suggest the authors clarify these points.

     * (4a) For example, in lines 57 - 63, you draw analogies to how you aim to transform "implicit relational knowledge in LLMs into explicit knowledge that can be stored and manipulated to make inferences". What do you mean by "explicit knowledge"? Your FFVs? And, what is the "implicit" part here?
     * (4b) You draw some analogies with cognitive science about how humans acquire permanent knowledge. How is this relevant to your work? Your approach is still an intervention introduced at inference time.

### 
Typo: Fix all the quotation marks.

### Soundness
1

### Presentation
1

### Contribution
1

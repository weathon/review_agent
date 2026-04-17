# Trained on Tokens, Calibrated on Concepts: The Emergence of Semantic Calibration in LLMs

- Decision: Accept (Poster)
- Scores: 4, 8, 6, 8

## Abstract
Large Language Models (LLMs) often lack meaningful confidence estimates for the semantic content of their outputs. While base LLMs are known to exhibit next-token calibration, it remains unclear whether they can assess confidence in the actual meaning of their responses beyond the token level. We find that, when using a certain sampling-based notion of semantic calibration, base LLMs are remarkably well-calibrated: they can meaningfully assess confidence in various open-ended question-answering tasks, despite training only on next-token prediction. To formalize this phenomenon, we introduce "$B$-calibration," a notion of calibration parameterized by the choice of equivalence classes. Our main theoretical contribution establishes a mechanism for why semantic calibration emerges in base LLMs, leveraging a recent connection between calibration and local loss optimality. This theoretical mechanism leads to a testable prediction: base LLMs will be semantically calibrated when they can easily predict their own distribution over semantic answer classes before generating a response. We state three implications of this prediction, which we validate through experiments: (1) Base LLMs are semantically calibrated across question-answering tasks, (2) instruction-tuning procedures systematically break this calibration, and (3) chain-of-thought reasoning breaks calibration (intuitively because models cannot predict their final answers before completing their generation). To our knowledge, our work provides the first principled explanation of when and why semantic calibration emerges in LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces semantic calibration for LLM, where a model’s probability over semantic answer classes matches its empirical accuracy. It proposes a theoretical analysis to link calibration and local loss optimality. Experiments show that instruction tuning and Chain-of-thought (CoT) hurt LLM’s ability of semantic calibration.

### Strengths
1. This paper proposed semantic calibration instead of calibration over token-level confidence.

2. The theory can explain the relation between loss optimality and calibration performance.

3. The observations are very interesting. It is quite surprising to me that instruction tuning and CoT hurt the performance of semantic calibration.

### Weaknesses
1. The conclusion that CoT breaks calibration is drawn mainly from math datasets.

2. There is a misalignment between theory and experimental results. The theory aims to reveal that a LLM is calibrated if it’s locally optimal under its loss function, but it cannot explain the interesting observations as (1) base LLM is the best for calibration, (2) instruction tuning and CoT hurt the performance of semantic calibration.

3. Any method to alleviate the drop of calibration performance for instruction tuning or CoT?


4. Some Typos:

     4.1 In abstract, we introduce “B-calibration,” a notion... ->  we introduce “B-calibration”, a notation of…

    4.2 In Equation 1, is it formal to use math notation like $#$?

### Questions
Please refer to my comments in weakness.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
Authors proposes a theory explaining how large language models (LLMs), despite being trained only on next-token prediction, can still exhibit semantic calibration—the ability to assign meaningful confidence to the content of their answers. The authors introduce B-calibration, a framework that measures calibration with respect to any “collapsing function” B that maps textual outputs to semantic classes. They prove that B-calibration is equivalent to local loss optimality, meaning a model is calibrated if its loss cannot be reduced by simple perturbations of its output distribution. Empirically, they show that base LLMs (e.g., Mistral, Qwen, Llama) are semantically calibrated across open-ended QA tasks, while instruction-tuning and chain-of-thought reasoning often break this calibration—since these processes alter the model’s ability to anticipate its semantic output before generation. The findings suggest that semantic calibration naturally arises from likelihood training, providing a principled mechanism linking uncertainty estimation to optimization dynamics

### Strengths
It introduces the notion of B-calibration—a general framework for defining calibration over arbitrary equivalence classes of outputs—which unifies token-level and semantic-level calibration under a single formalism. This perspective reframes an underexplored question (“can base LLMs meaningfully assess confidence in their answers’ meanings?”) into a rigorous, testable problem, providing a clear theoretical bridge between semantic uncertainty and local loss optimality. 

The work’s quality is high: the theoretical arguments are well-grounded in recent calibration theory, the proofs (especially the equivalence theorem and its autoregressive extension) are technically sound, and the experiments are extensive, covering multiple datasets, model families, and prompt styles.

### Weaknesses
Although the authors present calibration as emerging from the ability to “predict one’s own semantic output distribution,” this remains correlational. The LoRA probe experiment (Claim 10) demonstrates correlation between learnability and calibration but not causation.

The paper claims novelty in unifying calibration and loss-optimality, but related work in multi-calibration and conformal prediction is discussed mainly in the appendix. Bringing these connections into the main text—perhaps as a discussion of how B-calibration extends or simplifies existing frameworks—would situate the work more clearly within the calibration landscape.

### Questions
The paper assumes that base LLMs are “locally loss-optimal” with respect to simple perturbations. Could the authors provide empirical evidence or a small-scale ablation demonstrating this property, even approximately?

### Soundness
4

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
The paper answers the question whether large autoregressive LLMs provide meaningful confidence about the semantic content of their long-form outputs (not only next-token probabilities). This is important both scientifically (do models know what they don't know?) and practically (calibrated confidences enable safer decision-making).

### Strengths
1. Parameterizing calibration by an arbitrary collapsing function B is a neat, flexible formalism that connects sampling-based semantic confidence to established calibration literature. This lets the authors transparently say which semantic granularity they evaluate.
2. There is a solid theoretical contribution linking diverse prior work.The equivalence (Thm.6) between B-calibration and local-loss-optimality is a meaningful bridge from optimization theory to semantic calibration; Thm.9 provides a plausible route for autoregressive implementation. 
3. Multiple model families, base vs instruct variants, three prompting styles, and the LoRA probe present a coherent story that ties theory and practice together.

### Weaknesses
1. The chosen datasets primarily focus on an in-distribution assumption, which may limit the generality of the proposed method.
2. The claim  "instruction-tuning breaks calibration" may require deeper analysis.

### Questions
1. Experiments use four datasets (GSM8K, OpenMathInstruct-2, TriviaQA, SimpleQA). The theory relies on an "in-distribution" assumption (Claim 10 / Corollary 11) and the paper notes datasets like TruthfulQA may violate that and be miscalibrated. But many real-world culturally/evolving phenomena (memes, internet-slang, adversarial misinformation) are OOD and precisely the settings where calibration matters most.
The practical value of "semantic calibration" is highest in long-tailed, dynamic, and OOD settings. Showing calibration only in relatively conventional in-distribution QA may somehow limit applicability.
2. The empirical semantic confidence measurement requires many samples per prompt (M=50 at T=1) and then applying the B pipeline, which may be costly and may be impractical for real-time use. 
It is suggested to provide experiments or analysis on how calibration estimates change with M (e.g., M=5,10,20,50) and temperature — to show whether cheaper approximations are viable. 
3. It is not yet clear which aspects (objective mismatch, dataset selection, reward shaping) cause the breakdown. The DPO result is interesting (DPO model miscalibrated vs SFT-only less so) but needs deeper analysis. You may add some controlled ablations within a single model lineage varying (a) SFT size and loss (proper vs improper), (b) small RLHF/DPO steps with controlled reward shapes, and (c) temperature / decoding strategies. Report which factor correlates most tightly with calibration degradation.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper studies whether base LLMs (trained only by next-token prediction) are calibrated not just at the token level but at the semantic level for open-ended QA. It proposes a formal framework—$B$-calibration—that treats calibration with respect to semantic equivalence classes induced by a collapsing function B. Empirically, the authors estimate “semantic confidence” by sampling multiple generations for a question collapsing each answer to a semantic class (using an extractor) to obtain a distribution over classes, and then checking if confidence matches accuracy (standard reliability notion). The main claims are: (1) base LLMs show strong semantic calibration under their sampling-based procedure; (2) instruction-tuning breaks this calibration; and (3) chain-of-thought (CoT) reasoning also breaks it. A diagnostic LoRA experiment supports the mechanism: when a small LoRA can easily predict the model’s own semantic class distribution (low KL gap), the underlying model is better calibrated.

### Strengths
The paper is a pleasure to read and provides a nice interplay of theory guided empirical experiments to show the existence of semantic calibration in base models.  As I am not a theory person and don't have a background in multi-class calibration and its relation with loss functions, I am unable to assess the correctness of the main theorems. However, I was able to follow the intuition which broadly makes sense to me and connects well with existing literature. The main conceptual takeaway from the paper is useful and easy to follow.

### Originality

1. Provides a clear, general formalism—$B$-calibration—that parameterizes semantic calibration by an arbitrary collapsing function, decoupling the theory from any single definition of “semantics.” 
2. The theoretical link between $B$-calibration and local loss optimality over $W_B$ perturbations offers a principled mechanism for when semantic calibration should emerge in maximum-likelihood–trained LLMs.
3. The paper focuses on open-ended Q/A which is important and useful going forward as prior work on MCQ is already well-studied. 

### Quality

1. The mechanism yields intuitive predictions (base models should be calibrated; instruction-tuning and CoT harm calibration), and the paper reports experiments that align with all three.
2. The LoRA diagnostic is a thoughtful probe: correlating the ease of predicting the model’s own semantic class distribution with empirical calibration quality is a useful sanity check that directly targets the proposed mechanism.
3. The empirical measurement procedure for semantic calibration is clearly articulated: sample multiple completions at fixed temperature, collapse to classes, compute confidence and accuracy, then assess calibration. This is methodologically straightforward and reproducible in principle.

### Clarity.

1. The paper is very-well written and easy to follow. The writing is to the point and backed with relevant prior works. 

2. The sections are neatly organized and I thank the authors for providing an opening summary for each section on the content about to follow (giving clarity to the reader) and often providing intuition in simple words. 

3. The paper carefully motivates why token-level calibration is insufficient for long-form, open-ended tasks and explains the need for semantic calibration. The role of $B$ and the sampling-based estimator is described concretely. Fig.1 clearly shows the contribution of the paper in a simple and useful manner. 

4. The “from calibration to local optimality” storyline and the three-step mechanism (two proved, one heuristic) are presented in an accessible way.

5. The plots for Section 5 are beautiful and experiments are well-executed. 

### Significance.

If base LLMs are often semantically calibrated under this operational definition, then knowing when to trust base-model answers via sampling becomes a practical tool, and the observation that instruction-tuning and CoT can degrade calibration is a nontrivial, actionable takeaway for practitioners who care about reliability.
I look forward to seeing follow up work based on the concepts established in this work, especially towards verabalized confidence.

### Weaknesses
I don't have any issues with material covered in the paper which is thoroughly presented. The authors have presented a solid work worthy of acceptance. The only drawback I see is the limited applicability of this theory and the exclusive focus on their B-confidence calibration which the authors acknowledge in Section B.1.

While this work greatly extends our understanding of calibration on a semantic level for base LLMs, the semantic level is still limited to at most a sentence-level (as longer reasoning would fall into the CoT category). The authors state in footnote 3 that, "We will however assume that the latter holds (which is plausible if each evaluation distribution is a reasonably-sized sub-distribution of the pretraining distribution on which local-loss-optimality holds)." 

This assumption assumes evaluation distribution to be already covered in the pertaining distribution. While this often done targetedly by model practitioners, it is unreasonable to expect test query to be already covered by training data. LLMs may be used for OOD tasks and also benefit significantly from inference time-compute (as it is unreasonable to expect an LLM to produce an answer, especially for challenging tasks, in a single forward pass). Thus, to me, it is not surprising that token-level calibration can lead to sentence-level (in this work, *limited to a single sentence*) calibration as the model is saying the same thing but instead of spiting it out directly, it just says the same thing in a sentence. What is more interesting is whether models are/can be made calibrated on long-form generations [1]. I say this more towards future work as the paper is already dense and contains a sizable contribution!

Minor: some plots are missing in Appendix figures like on page 44 for phi-4 and page 46 for Qwen3.

[1] Band, N., Li, X., Ma, T., & Hashimoto, T. (2024). Linguistic calibration of long-form generations. arXiv preprint arXiv:2404.00474.

### Questions
Q1: OpenMath-Instruct2 is a dataset generally used for training (as it has more than >1M samples). Any reason to choose a train set for evaluation? How many samples did your test set of OpenMath consist of? I would rather suggest using MATH [2] for evaluation. I don't expect models to be able to solve questions from MATH in just a single word or sentence so I assume this might not follow the properties required by your theory but I still would be interested to see whether base models end up *not* being calibrated on MATH.

Q2: The authors note that some datasets (e.g., TruthfulQA) likely violate in-distribution assumptions and may behave differently. This is a great point and I actually would be interested to see whether the predicted breakdown on TruthfulQA (or something else) happens empirically.

Q3: For a given model and task (prompt), was the semantic classification performed on just 3 samples (1 for each prompting methodology)?

[2] Hendrycks, D., Burns, C., Kadavath, S., Arora, A., Basart, S., Tang, E., ... & Steinhardt, J. (2021). Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874.

### Soundness
4

### Presentation
4

### Contribution
3

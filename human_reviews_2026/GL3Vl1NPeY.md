# Parallel Test-Time Scaling with Multi-Sequence Verifiers

- Decision: Reject
- Scores: 6, 4, 6

## Abstract
Parallel test-time scaling, which generates multiple candidate solutions for a single problem, is a powerful technique for improving large language model performance. However, it is hindered by two key bottlenecks: accurately selecting the correct solution from the candidate pool, and the high inference latency from generating many full solutions. We argue that both challenges are fundamentally linked to verifier calibration. A well-calibrated verifier not only improves answer selection, but also enables early-stopping strategies to reduce latency. However, existing verifiers are limited as they score each candidate in isolation, overlooking rich contextual information across the set of candidates. To address this, we introduce the Multi-Sequence Verifier (MSV), the first verifier designed to jointly process all candidate solutions and model their interactions. MSV achieves state-of-the-art calibration, which directly enhances best-of-N selection performance. We further introduce a streaming MSV variant that empowers a novel early-stopping framework. Our novel framework fully leverages parallel decoding, which contrasts with the existing multi-sequence early exit works that decode sequences one by one and thus incur significant latency. In this novel setting, MSV can achieve the same target accuracy with around half the latency that would be required with its counterpart that scores each solution in isolation.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes Multi-Sequence Verifier (MSV), a training algorithm to train a verifier model for test-time scaling, that can score multiple sequences in parallel, importantly taking into account all other sequences when computing a score for a sequence. This contrasts it from sequence-level methods like best-of-n with external verifiers.

**I think the paper should be accepted.** However, some crucial aspects should be addressed by the authors (see "Weaknesses" below), most importantly a more detailed latency breakdown, and comparing against other commonly used baselines (e.g. BoN with external verifier).

### Strengths
### Writing
The paper is very well written and well structured. All concepts are cleanly defined and explained (except a unifying overview of the method, see "Weaknesses" below).

### Contribution
The proposed method seems quite novel and interesting. It features various novel ideas (see section 3).

### Results
The results are look promising. Not only does MSV seem to lead to accuracy gains, but it also seems to lead to well-calibrated models, which is possibly independently relevant to the research community.

### Weaknesses
### Calibration
From a theoretical point of view, it's not clear (at least to me) that the proposed procedure of computing $\tilde{y}$ (which I'm assuming is used as the predictive probability $p$ in section 3.3) actually enforces calibration (even if the experiments show this happens empirically). Yet the authors make it seem like this naturally follows from the proposed training objective (e.g. in line 299). If this is indeed supported by theory, the authors should expand on it.

### Relation to Existing Literature
I believe some of the proposed setting in which MSV is trained, e.g. terminal answers vs streaming answers, the idea of extracting intermediate answers at delimiters, etc., is insufficiently put into context w.r.t. previous literature. If all of these ideas are novel, this should be made more clear, but I would assume that in particular the "terminal answers" vs "streaming answers" viewpoint has been studied before.

### Method
It would help to understand the method better if the authors provided an algorithm of how MSV is trained, that combines all the parts of section 3 and makes it easy to grasp what parts are being trained and how (as there are various trainable parameters scattered throughout section 3).

### Experimental Results
The proposed MSV contains various design choices that seem somewhat arbitrary, e.g. the choice of masks (section 3.2). Ablations over these masks and their affect on downstream performance would be helpful.

Furthermore, while the authors claim they show accuracy-latency tradeoffs, e.g. in Figure 7, they actually usually just show accuracy-token position tradeoffs. The only latency results I could find are in Table 4 (line 864 ff), and those are not very explicit. For example, how is "decoding" in this table defined? Is this only the latency for decoding tokens that would be generated as part of the sequences themselves, or does this include latency for decoding of the intermediate extracted answers? (Which should go into the latency overhead of MSV, but should be separated from the latency of BoN or other baselines.) The authors should include a more detailed latency comparison, that shows training time/compute, and a detailed breakdown of the latency overhead that MSV adds to simpler baselines.

The authors compare to various baselines in their experiments. However, a) some of these baselines are not explained at all (e.g. Probe + WV). Moreover, one baseline, MSV 1, is their own method in a single-sequence setup. Comparing to other, commonly used methods in the literature would make the experimental results much stronger. (E.g., how does this compare to simple BoN with an external verifier?)

### Questions
- minor: some of the fonts in figures and plots are too small to read (e.g. Figures 1, 4, and many of the tables)
- minor: line 895: "since forward pass" -> "since *a* forward pass"
- sometimes the authors write "multi-sample verifier" instead of "multi-sequence verifier" (caption of Figure 1, and line 59)
- line 158: "to end of $n$th sequence" -> "to *the* end..."
- minor notational inaccuracy: In the "Streaming Answers" setting, the authors should don't clarify which of the $K^{(n)}$ answers is the terminal one (I assume it's $a^{(n)}_{K^{(n)}}$, since the other answers are defined to correspond to the intermediate steps; this should be clarified though since saying that $a^{(n)}_1$ is the terminal answer in the terminal answers setting seems to contradict this notation)
- it would help to explain in more detail at what points in the generation intermediate answers are extracted in the streaming answers setting. The authors say "whenever a delimiter [...] is encountered". Could you say what delimiters the LLM you're using can produce exactly (any other than "wait"?), how often this happens, etc.?
- how do the "terminal answer" vs "streaming answer" viewpoints relate to existing literature?
- in the input concatenation (line 202), how do you concatenate exactly? Is there any delimiter/special structure by which you concatenate answers, or just raw concatenation? (Looking at the following section defining the masks, it is probably raw concatenation, but why this is reasonable only becomes clear in the following section, it might help to clarify this here.)
- the difference between "within-sequence mask" and "within-answer mask" could be made more clear, both in text as well as in Figure 1 where they look identical. (As I understand it, the former is all answers within that sequence up to a time $t$, while the latter is only one answer in that sequence, but it's not entirely clear.) In particular, there's some notation here (ans(u), seq(u), step(u)) that could be defined more clearly.
- the definition of the final feature (line 256) is not entirely clear to me. Why only take the last token's representation (and not a (weighted) mean across tokens)? Does the MLP simply map from $\mathbb{R}$ to $\mathbb{R}^d$?
- maybe I'm missing it, but it seems like the correctness probability $p$ (section 3.3) is not defined anywhere. Is this simply the predicted $\tilde{y}$?
- the training set seems to be very small (224 problems, cf. line 730). Could the authors elaborate on why the training set is chosen so small? How does the validation error look like during training, how long does training take, etc.? If MSV can be trained quickly, this could be highlighted as an advantage in the paper.
- the end of section 2.2 seems to be missing
- line 81: "... that latency doesn't grow" -> This claim doesn't seem to be supported anywhere (in particular, I was not able to find latency tradeoffs as $N$ grows). Also, if latency doesn't grow, scaling beyond $N=64$ would be interesting to see.

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
4

### Summary
This paper aims to address two key bottlenecks in parallel test-time scaling for large language models (LLMs)—accurate selection among multiple candidate solutions and high inference latency—by introducing the Multi-Sequence Verifier (MSV). This improved calibration directly enhances best-of-N selection accuracy and enables more reliable confidence estimation. The authors further propose a streaming variant of MSV that supports a novel parallel early-stopping framework: by evaluating intermediate outputs from all sequences simultaneously during parallel decoding, decoding can terminate as soon as any sequence reaches a confidence threshold.

### Strengths
Overall, the work makes three key contributions:
 (1) a novel design of MSV as the multi-sequence joint verifier, 
(2) the demonstration that superior verifier calibration directly improves parallel scaling performance, 
 (3) the introduction of a practical, low-latency parallel early-stopping framework enabled by streaming MSV, which fundamentally rethinks how test-time compute can be scaled without proportional latency costs.

### Weaknesses
Overall, the paper presents a moderately novel approach but falls short of a truly significant conceptual leap. The work proposes two main contributions: (i) an improved verifier for best-of-N selection, and (ii) a streaming early-stopping framework.

(1) For the first, the core innovation lies in explicitly incorporating the proportion of sequences that produce symbolically equivalent answers (i.e., consensus frequency) as an auxiliary feature, while still relying on standard attention mechanisms to model cross-sequence interactions. While this integration of frequency-based signals is sensible, it builds incrementally on existing ideas like self-consistency rather than introducing a fundamentally new paradigm. 

(2)Regarding the second contribution—the streaming answer setting—the motivation is strong and practically relevant. However, the implementation is limited: intermediate answers are extracted only when a predefined delimiter token (e.g., “Wait”) appears, which relies on ad hoc prompting rather than genuine analysis of the model’s internal reasoning states or semantic completeness. A more principled approach would involve detecting answer readiness from the model’s latent representations or logical progression, not just surface-level trigger tokens.

(3) The experimental evaluation is somewhat narrow. The authors evaluate only a single base LLM (DeepSeek-R1-Distill-Qwen-1.5B), which limits the generalizability of the findings. More importantly, the baselines omit Self-Consistency—a standard and highly relevant method in this domain that selects answers based purely on occurrence frequency without any verifier. This omission makes it difficult to assess whether the gains from MSV truly stem from its joint modeling capability or simply from using any verifier at all. Additionally, given the rapid progress in open-source LLMs, it would strengthen the paper to include stronger and more recent models (e.g., Qwen3 or Llama-3.1) as base generators to demonstrate the robustness of MSV across architectures.

(4) The comparison is not entirely fair with respect to the paper’s central claim of addressing high inference latency. While the paper reports accuracy–latency trade-off curves (e.g., Figure 7), it lacks concrete wall-clock timing measurements (e.g., milliseconds per query) that account for the full pipeline—including MSV’s own inference overhead. Since MSV processes all N sequences jointly through a multi-mask Transformer, its computational cost could be non-negligible, especially as N grows. The paper should explicitly report: (a) the end-to-end latency of MSVₙ vs. baselines (including verifier runtime), and (b) how much additional latency MSV incurs to achieve its accuracy gains. Without this, the efficiency claims remain partially unsubstantiated.

### Questions
1.Why is the equivalence relation (∼) assumed to be perfect?
The method relies heavily on symbolic equivalence (e.g., via SymPy) to define answer identity and compute γ. But in many real-world tasks (e.g., open-ended QA, code generation, or non-math reasoning), such a deterministic equivalence checker may not exist. How would MSV generalize to domains where answer equivalence is fuzzy or subjective?

2.How sensitive is MSV to the choice of delimiter token (“Wait”) in the Streaming setting?

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
4

### Summary
The paper argues that the existing verification-based methods either suffer from poor accuracy or use too much inference compute because they wait for the full solutions from the generator. To fix these issues, the paper proposes a multi-sequence verifier, a technique that processes all the candidate solutions together and terminates early even if the solution is not fully generated. In particular, there are two variants of the method: terminal answers and streaming method. Further, the framework applies a multi-mask strategy to capture interactions between multiple solutions, final answers, and equivalent answers.

### Strengths
1. The paper tackles an important problem of lack of accurate, calibrated, and fast inference with existing verifiers. To mitigate this, the paper proposes multi-mask training in the final answer and streaming answer scenarios. 

2. Ultimately, the proposed method provides decent performance improvements across diverse evaluation benchmarks. Further, it seems to be working better than pertinent baselines such as MSV_1, and Probe.

3. The paper also shows that the MSV achieves better calibration and accuracy-latency tradeoff.

### Weaknesses
1. The experiments are performed with just one model size and model family i.e., deepseek-r1-distill-qwen-1.5B.  It would be better to try the method on more models and at various sizes. 

2. It feels that having many attention masks that operate on similar sequences is a bit of an overkill. If you have enabled full attention (every sequence attends to every other thing), it remains unclear why other attention masks are needed in practice. There is no ablation which shows that each attention mask adds something to the performance. 

3. While it is fascinating to attend to many solutions at a time, I think there are scalability issues with this paradigm. Existing thinking models can generate upto 16K tokens and you can’t control whether the first final answer occurs. The context length and latency will blow up pretty quickly in such scenarios if the number of solutions is in the order of 100s. Whereas, the vanilla verification can operate on solutions independently and start performing well.

### Questions
Mentioned above

### Soundness
3

### Presentation
3

### Contribution
3

# Do LLMs Signal When They’re Right?  Evidence from Neuron Agreement

- Decision: Reject
- Scores: 2, 4, 6, 6

## Abstract
Large language models (LLMs) commonly boost reasoning via sample-evaluate-ensemble decoders (e.g., majority voting), achieving label free gains without ground truth. However, prevailing strategies score candidates using only external outputs such as token probabilities, entropies, or self evaluations, and these signals can be poorly calibrated after post training. We instead analyze internal behavior based on neuron activations and uncover three findings: (1) external signals are low dimensional projections of richer internal dynamics; (2) correct responses activate substantially fewer unique neurons than incorrect ones throughout generation; and (3) activations from correct responses exhibit stronger cross sample agreement, whereas incorrect ones diverge. Motivated by these observations, we propose Neuron Agreement Decoding (NAD), an unsupervised best of N method that selects candidates using activation sparsity and cross sample neuron agreement, operating solely on internal signals and without requiring comparable textual outputs. NAD enables early correctness prediction within the first 32 generated tokens and supports aggressive early stopping. Across math and science benchmarks with verifiable answers, NAD matches majority voting; on open ended coding benchmarks where majority voting is inapplicable, NAD consistently outperforms Avg@64. By pruning unpromising trajectories early, NAD reduces token usage by 99\% with minimal loss in generation quality, showing that internal signals provide reliable, scalable, and efficient guidance for label free ensemble decoding.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes Neuron Agreement Decoding (NAD), an unsupervised best-of-N method that selects candidates using activation sparsity and cross sample neuron agreement. NAD is motivated by three observations:

- External signals are low dimensional projections of richer internal dynamics;
- Correct responses activate substantially fewer unique neurons than incorrect ones throughout generation; and
- Activations from correct responses exhibit stronger cross sample agreement, whereas
incorrect ones diverge

The authors claim that:

- NAD enables early correctness prediction within the first 32 generated tokens and supports aggressive early stopping
- NAD matches the accuracy of majority voting in math and science benchmarks and outperforms Average@64 in open-ended coding benchmarks.
- NAD reduces token usage by 99% with minimal loss in generation quality

### Strengths
- The paper proposes a promising method which uses mechanistic interpretability for selecting best reasoning traces in Best-of-N sampling
- The early stopping analysis may be of interest to the efficient inference community.

### Weaknesses
## Major Weaknesses

- Preliminary claims are poorly justified
    - Section 3.2:  The authors claim that neuron activation patterns capture structure beyond what entropy represents, citing that samples within clusters have varying entropy values. However, this conclusion is poorly justified.
        - First, they have shown that the *number* of activated neurons correlates with entropy, suggesting the clustering is partially driven by a scalar feature that entropy already captures.
        - Second, any high-dimensional representation will trivially contain structure that a single scalar cannot fully represent, which is not evidence of meaningful structure. The variation in entropy within clusters could simply reflect noise, measurement artifacts, or the fact that t-SNE on Jaccard distances emphasizes pattern overlap rather than distributional properties. At the moment, the more likely conclusion from Figure 3 is that one metric does not perfectly predict another.
    - The preliminary experiments are done with only one model (Qwen3-4B) which may not be generalizable.
- Lack of motivation on the experimental setup
    - It is unclear why the models are selected (Qwen3-4B-thinking-0527, Qwen3-4B-Instruct-0527 and DeepSeek-R1-0528-Qwen3-8B). Is it because of the different reasoning training regime? or are there specific reasons?
    - Lack of baselines.
        - This is very critical especially because the authors claim that the method captures structure beyond what the “external behaviors” can, thus it is natural to expect that NAD would outperform prior works which are based on these external behaviors:
            - Majority-based selection: Universal Self Consistency [1]
            - Confidence-based selection: Self-Certainty[2], DeepConf [3], PiCSAR [4]
            - Length-based selection: short-1@k [5]
- Lack of statistical rigor
    - As the paper is dealing with sampling, the authors should try to run the experiments with multiple random seeds to account for stochasticity.

## Additional Suggestions

- L56: Please cite the GPT-4 reports
- Figure 3: Update the colorbar label (”Average Entropy”)
- Section 3.2: I believe the Jaccard index is calculated pairwise among all responses across all questions. Please add that explanation in the paragraph
- Figure 6, 7, and 8 are ordered awkwardly. Figures 7 and 8 are mentioned in the text earlier than Figure 6.
- The model name Qwen3-4B-thinking-0527 is perhaps a typo? it should have been Qwen3-4B-thinking-**2507**

## References

- [1] Universal Self-Consistency for Large Language Model Generation
- [2] Scalable Best-of-N Selection for Large Language Models via Self-Certainty
- [3] Deep think with confidence
- [4] PiCSAR: Probabilistic Confidence Selection And Ranking for Reasoning Chains
- [5] Don't Overthink it. Preferring Shorter Thinking Chains for Improved LLM Reasoning

### Questions
- Figure 2:
    - Have you tried separating the correct vs incorrect instances in Figure 2? The trend may differ between the two categories.
    - Have you tried plotting Figure 2 in log-log scale? I suspect that there is a power-law relation there, which may be interesting.
- Why are the AIME24 and AIME25 reported as one task?
- Why is GPQA under Math Reasoning? Which subset of GPQA did you use?
- Table 2: It is rather awkward to report the total token consumption. Any particular reason why you choose to report that? I believe we are more interested in the average number of tokens saved per question (with confidence interval).
- In Section 5.3 analysis of the top-k method, what is the metric used to decide the separation? You should consider using statistical test to quantify it.
- Figure 8:
    - Is this averaged across questions? If yes, please provide the confidence interval bars.
    - Am I understanding it incorrectly? Because the B=16k seems to achieve the highest accuracy, which contradicts the conclusion mentioned in the text.
    - What should I interpret from the token consumption line in the plot?
- Is there a way to automatically decide the early stopping position? If not, it seems like a difficult hyperparameter to tune.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper investigates whether an LLM’s neuron activations can be used to determine if its generated response is correct. The authors contrast this with prevailing methods that rely on "external" signals like token probabilities, output entropy, or model self-evaluations, which the paper argues can be poorly calibrated.

The paper shows that these external signals are effectively low-dimensional projections of richer, high-dimensional internal dynamics. The authors' analysis uncovers two key findings. Sparsity: Correct responses activate "substantially fewer unique neurons" than incorrect responses during generation. Agreement: The activation patterns from correct responses exhibit "stronger cross-sample agreement," while incorrect responses tend to diverge.

Motivated by these observations, the paper proposes a novel unsupervised method called Neuron Agreement Decoding (NAD). NAD selects the best response from a batch of $N$ samples by identifying the candidate with the highest activation agreement with its peers or, alternatively, the one with the fewest activated neurons (activation sparsity).

### Strengths
The paper's primary strength is its novel investigation that successfully links an LLM's internal neuron activation patterns to the external correctness of its reasoning. Specifically, looking at the number of activated neurons and how they overlap between different inputs can provide a signal for whether the answer is correct. This is pretty cool and I haven’t seen such an exploration before.

One strength of NAD is its ability to operate without requiring comparable textual outputs, unlike majority voting. This makes it applicable to open-ended tasks like code generation, where majority voting is often inapplicable. This is an important direction for research these days.

NAD matches or outperforms the performance of majority voting on math and science benchmarks and open-ended coding benchmarks. This is pretty strong evidence that NAD can work well, and without using too much inference time (or even reducing compared to Majority voting over many samples).

### Weaknesses
* One area I am quite skeptical about is whether this method works when the base model has relatively high or relatively lower accuracy on the task in the first place. The experiments right now show that NAD works in the “middle ground” regime, with about 50-70% accuracy. However, for high accuracy (>90%) then NAD seems to degrade performance. Similarly if the original performance is low, I can imagine that the neuron activations could be more “random” so that NAD method doesn’t work. 

* Also the value of Avg@64 on these tasks is surprisingly high (since you are averaging over 64 outputs), which means the model is inherently very confident on these tasks. It could very well be the case that NAD only improves performance if Avg@64 is similar to Pass@1 or something. Basically the model is not very creative and only tries the same types of solutions. 

* There are no baselines in this paper. The paper only compares its own variants and the base models. I generally am skeptical about a paper without any other methods in the experiments. I understand that NAD is kind of a unique method, but there are a lot of test-time methods for improving reasoning these days. For example, TTRL (https://arxiv.org/abs/2504.16084) or the authors already discuss DeepConf (https://arxiv.org/abs/2508.15260). I am less interested in the 3 different clustering methods (which are basically an ablation for NAD).

### Questions
* For the evaluations, although the datasets are varied, the performance is somewhat clustered. What happens when the base model has high accuracy (e.g., GSM8k)? Or low accuracy, on some harder benchmarks (e.g., Humanities Last Exam or some of the newer benchmarks with search like SealQA https://arxiv.org/abs/2506.01062). Do we see any benefit or does it also degrade? 

* What happens if you perform the analyses with different numbers of samples? The analysis right now is very focused on 64. However, it is not clear if this is kind of a local maximum for NAD performance or whether 32 and 128 also exhibit good performance.

* Similarly, what if Avg@64 is low because the model can output a lot of wrong answers if you keep sampling. This seems like the dominant regime if we are looking forward to AGI and harder tasks. Can you say something about if NAD will work then?

* How does NAD compare to other method’s performance on the same models/benchmarks? The paper cites a few majority voting variants, or compare against TTRL or DeepConf or any of the methods the paper mentions about “external” rewards. These are still very valid approaches for the task at hand.

* This is more minor, but there is a lot of work on decoding methods for improving model outputs. I would say it is worth citing these, and perhaps comparing against them. For example, Factuality Decoding methods also use internal signals (internal layers) to improve the output, e.g., DoLA (https://arxiv.org/abs/2309.03883) and SLED (https://arxiv.org/abs/2411.02433). I would take a look at TTRL (https://arxiv.org/abs/2504.16084) and the forward citations as well. I think currently there are only **4 papers cited in the related work** section, which is quite limited.

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
- **Core idea:** Use internal neuron activations (not output logits/entropies) to score and select reasoning traces. Two signals: (i) activation sparsity (correct traces activate fewer unique neurons) and (ii) cross-sample neuron agreement (correct traces share more similar activation sets).  
- **Method:** Neuron Agreement Decoding (NAD): build a Jaccard-similarity matrix over activated-neuron sets across sampled traces and select via kNN/medoid/DBSCAN; a MinAct variant selects the fewest-unique-neuron trajectory. Early stopping uses the same signals after the first 32 tokens (B=32) to prune low-quality paths.  
- **Findings:** On AIME24/25 & GPQA, NAD matches majority vote while enabling aggressive early stop; on code (HumanEval, MBPP, LiveCodeBench), where voting is hard, NAD beats Avg@64. Reported token reductions up to ~99% with small accuracy loss.

### Strengths
- **Insightful internal analysis:** shows entropy/self-certainty are low-dimensional projections of richer activation dynamics; correct traces are sparser and more aligned across samples.  
- **Simple selection rules:** kNN/medoid/DBSCAN over Jaccard of activated-neuron sets; unsupervised and label-free at test time.  
- **Early-stop lever:** practical chunked early-stop at 32 tokens with large token savings in parallel sampling.

### Weaknesses
- **Positioning vs token-confidence baselines:** Conceptually close to self-consistency / DeepConf (token-level confidence/entropy) but at the neuron level; however, there is no apples-to-apples comparison against DeepConf under the same sampling regime (accuracy + compute).  
- **“Early correctness within 32 tokens” needs clarification:** Paper sets early stop at B=32 and infers quality from internal signals—not ground-truth correctness mid-generation. Clarify how “NAD enables early correctness prediction within the first 32 generated tokens” is quantified and whether OOD checks were made to avoid overfitting to seen patterns.  
- **Scope & generality:** Signals are shown strongly on AIME-style math; for open-ended tasks (code), MinAct can underperform, and neuron-agreement advantages shrink—casting doubt on broad generality (e.g., free-form scientific discovery).  
- **Cost reporting is incomplete:** Paper emphasizes token savings, but wall-clock, activation extraction overhead, pairwise Jaccard construction, and memory/storage (noted in Limitations) aren’t benchmarked vs strong external baselines.  
- **Baselines:** Mainly Avg@64 and Cons@64; missing self-evaluate before ensemble and confidence-based (e.g., DeepConf) under matched budgets.

### Questions
1. **Meaning of “early correctness”:** When you say “enables early correctness prediction within the first 32 tokens”, do you mean ranking traces by internal signals at 32 tokens and later verifying with ground truth, or a calibrated correctness probability? How is this measured, and did you test OOD prompts to check robustness?  
2. **DeepConf comparison:** Please provide matched-budget comparisons to DeepConf (token-level entropy pruning): final accuracy, token count, wall-clock, and memory. This will isolate the incremental value of neuron-level signals over token-level confidence.  
3. **Generalisation beyond math:** Your Figure 6 suggests weaker or minimal gains (even reversals) for code. Can you evaluate on open-form science benchmarks to test whether neuron sparsity/agreement remains predictive when answers are not short-form/numeric?  
4. **Computation & storage:** Please report the per-token/trace overhead of computing activation sets, building the n×n Jaccard matrix, and memory footprint (with/without bitset compression), compared to token-confidence baselines.  
5. **Ablations:** How sensitive are results to the activation thresholding (top-k per token), chunk size B=32, and the choice among kNN/medoid/DBSCAN? Could later chunks introduce noise (as hinted by Figure 8), and how does this vary by task?

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
This paper studies internal activation patterns in LLMs and reports two empirical regularities: (i) correct generations activate fewer unique neurons than incorrect ones; and (ii) correct generations show higher cross-sample neuron-set agreement. Building on these observations, the authors introduce Neuron Agreement Decoding (NAD)

### Strengths
1. Novel internal-signal criterion: Selecting candidates via Jaccard agreement of activated-neuron sets rather than output-space agreement is intellectually novel

2. Computational savings: Early pruning at 32 tokens yields two orders of magnitude fewer tokens with modest accuracy impact

3. Method simplicity: NAD relies on inexpensive set operations over FFN activations; the MinAct variant is parameter-light, aiding adoption.

### Weaknesses
1. External validity to large/closed models: All results are on small/medium, open models. It is unclear whether the “fewer-neurons-when-correct” regularity and NAD’s gains hold for frontier models (70B–>100B)

2. Definition and sensitivity of “activated neuron set”: The operational definition depends on thresholds/top-k within layers and across chunks. Although ablations exist, a more systematic sensitivity analysis (varying k, chunk size B, layer subsets, and gating functions) would strengthen your claims. 

3. Sampling hyperparameters: Results are reported for T=0.6, top-p=0.9; robustness to temperature/top-p and to different N would help verify the effectiveness.

### Questions
1. How does NAD’s advantage change with larger models and larger N?
2. Could you provide a more solid theoretical analysis for your arguments?

### Soundness
3

### Presentation
3

### Contribution
3

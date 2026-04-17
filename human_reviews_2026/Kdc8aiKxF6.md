# The Energy Cost of Reasoning: Analyzing Energy Usage in LLMs with Test-time Compute

- Decision: Reject
- Scores: 2, 10, 6, 4

## Abstract
Scaling large language models (LLMs) has driven significant advancements, yet it faces diminishing returns and escalating energy demands. This work explores how test-time compute (TTC) can serve as an energy-efficient complement to conventional scaling strategies by allocating additional computational resources at inference time rather than during training. Specifically, we investigate whether employing TTC can achieve superior accuracy-energy trade-offs compared to simply increasing model size. Our empirical analysis reveals that TTC surpasses traditional model scaling in accuracy/energy efficiency, with notable gains in tasks demanding complex reasoning rather than mere factual recall. Further, we identify a critical interaction between TTC performance and output sequence length, demonstrating that strategically adjusting compute resources at inference time according to query complexity can substantially enhance efficiency. Our findings advocate for TTC as a promising direction, enabling more sustainable, accurate, and adaptable deployment of future language models.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper measures the energy cost of test-time compute for LLMs and compares different test-time scaling strategies across several benchmarks and Qwen2.5 family models.

### Strengths
- The energy cost of LLM inference is an important topic in the social impact of machine learning.

- The evaluation covers a range of benchmarks and model sizes (up to 32B).

### Weaknesses
- I think the terminology "TTC" in the paper is inappropriate. When an LLM is used in the test-time, the compute should be considered as test-time compute (TTC). Based on my understanding, this submission indeed focuses on test-time **scaling** (TTS), or different methods of scaling up TTC [1]. TTS should be the right term to use -- and has been widely adopted in existing papers [1,2].

- Table 1 places GPQADiamond under the Math column, but it is in fact a graduate-level scientific reasoning benchmark (biology/chemistry/physics).

- Many critical implementation details are missing, hindering reproducibility. To elaborate,
  - Prompt templates and decoding parameters (temperature, top-k, top-p) were missing. There is neither an appendix nor supplementary materials with experiment configs.
  - RT (reasoning token) implementation is ambiguous. The paper groups the RT approach and cites DeepSeek-R1, STAR, and REST-EM, but nowhere does it specify which concrete RT implementation was used, nor how models were post-trained / fine-tuned for reasoning. According to lines 146-151, I assume that the authors use [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) series. If this is the case, the authors should have explicitly stated the exact model choices in different settings with clear references.

- Some of the findings are known. For example, the finding that smaller models enhanced with TTS can outperform substantially larger models is already established in existing literature (in terms of FLOPs consumption) [1]. While examining this from an energy consumption perspective has value, the paper doesn't sufficiently acknowledge this known pattern or highlight new insights regarding energy consumption.

- Some evaluation results look weird. Specifically, Figure 1 shows zig-zag patterns across model sizes. It seems that GPUs might have been under-utilized in certain configurations, and the results may stem from measurement artifacts or mis-sized batches and not intrinsic algorithmic behavior.

- _Minor error_. Line 34: "diverse datasets(Villalobos et al., 2022)" -- a space is missing here.

[1] Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters

[2] s1: Simple test-time scaling

### Questions
- How is the inference batch size tuned? How does that affect evaluation results?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
10

### Rating Number
10

### Confidence
4

### Summary
This paper investigates the relationship between accuracy improvements and energy consumption incurred by employing test time compute  methods versus scaling model sizes across multiple benchmarks. It tests a variety of mathematical, coding, and common-sense reasoning benchmarks, we derive key insights into their comparative advantages and costs.

They find valuable insights, that have important consequences for improving the sustainability of AI systems:
Trade-off improvements for TTC over traditional model scaling.
The limitations of using TTC in terms of increased energy usage
The correlation between output sequence length, model comprehension and energy

### Strengths
The paper provides empirical evidence for several important insights, notably: 
-- that merely increasing model size does not guarantee enhanced accuracy if underlying reasoning capabilities remain insufficient.
-- that majority moving consumes orders of magnitude more energy than base models, with reasoning tokens consuming even more, notably because of how many tokens are produced 
-the fact that the RT inference process quickly becomes memory-bound due to the number of tokens produced , limiting the effective utilization of GPU computational resources.
- that incorrect answers and difficult queries produce longer answers
- that prefix caching can help save significant energy, which is useful for LLM deployment

### Weaknesses
I feel like there are some of the analyses can be further deepened, especially with regards to potential hypotheses about why certain models or configurations use more energy than others.

Also, the Figures are often not well explained and at a different place in the article than the text that talks about them, which can make it hard to follow the narrative of the paper.

There are certain methodological details that are lacking -- e.g. regarding model selection and setup -- which could help to understand the relevance of the results for the broader community.

### Questions
Can you explain the following statements further (provide hypotheses for why this is the case?):

"For example, in mathematical benchmarks, increasing the model size from 1.5B to 7B parameters in the Base configuration yields only a 4.8% accuracy improvement with negligible energy difference per query. "

"In coding benchmarks, scaling model size offers better accuracy/energy gains at lower model scales than RT."

"Common-sense benchmarks show limited or even adverse performance when employing RT" 

In general, it would be helpful to understand the difference between the different tasks and how this influences energy consumption. E.g. are the expected answers for common-sense benchmarks longer than coding or mathematical ones?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper describes the energy consumption of different models in the reasoning paradigm. 
PRetty happy with this paper's scientific rigor and findings. I think it's a valuable result for inference practicioners.

### Strengths
1. Good experimental design. Nice model sizes and divee set of methods/benchmarks make this papers' results more grounded.
2. Pretty practical and timely
3. Pretty novel insight that reasoning is more energy efficient.

### Weaknesses
1. ONly studies qwen2.5 models, would have been good to see this across different model families.
2. A training analysis would have been nice toa dd, but I understadn that that is expensive.
3. Did not see standard error reported here?
4. Could ahve picked nonreasoning benchmarks as well to understand the landscape a bit better.

### Questions
n/a

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
4

### Summary
The authors perform an analysis of the effectiveness of using additional test-time-compute to improve performance on math, code, and common sense tasks, with a particular focus on energy and power consumption associated with performing benchmark tasks. Specifically, they compare accuracy-energy trade-offs in TTC vs traditional model scaling, observing that smaller models with more TTC can outperform larger models using less energy per query. Additionally, they find that, for most of their tasks, scaling up reasoning tokens is more effective than either scaling up model size or performing majority voting with parallel sampling; in general, more output tokens seems to help more with more complex tasks.

### Strengths
1. Authors provide a timely and useful set of explorations that many in the community may find helpful as rules of thumb and guidelines for their own work
2. I felt it was a fresh perspective, analyzing results through lenses of GPU utilization and instantaneous power draw vs overall energy consumption
3. Compelling discussions of some possibly counterintuitive results in 4.2 -- I actually felt these should be highlighted more

### Weaknesses
1. I realize there is probably not a single setting that is clearly both a fair comparison and very realistic, but a batch size of 16 feels optimistic -- I would rather have seen multiple batch sizes explored given that models are often served in very different settings. I cannot help but imagine that smaller models with TTC beating larger models without TTC is at least somewhat a function of batch size, and it would be extremely helpful to understand what the practical threshold is, even just for this specific hardware setting used.
2. Some clarity issues, possibly related to my confusion in Q2 below. In figure 2 especially, since performance does not increase monotonically with model size, it would be helpful to also have different marker shapes for the different model sizes
3. The language in the paragraph starting at line 137 feels like an overgeneralization, in a way that I am not sure it has to be given the overall framing? I would have liked to see either: 1) some additional coverage of alternative TTC methods (since there is just one of each; e.g. RAG for inputs? tree of thought decoding for outputs?); 2) some justification for the two strategies used as particularly effective or standard for their respective categories, and/or 3) at least a mention of what alternative methods exist and are not covered. I am torn between feeling that there should be another round of work before acceptance for publication vs. just a reframing of the TTC methods covered as just examples of common popular approaches and a small signal that there is some good reason for these as exemplars

Relatively minor points and nits:
- line 131: worth noting 500w comes from custom thermal solution — afaik 400w is standard
- in figure 2 I would have preferred a more distinct visual difference between MV and RT filling patterns
- line 485: “accuracy” twice — presumably the first one is meant to be “energy consumption”
- line 161 seems to suggest there will be much more detail about the tasks and datasets used than there is in reality — I would have liked to see some context on tasks’ typical input sequences and also output sequence length  (even just from the base methods used) organized into a table, even if it ended up in the appendix

### Questions
1. It feels suspicious to me that the energy per query in MV should always be strictly larger than with RT (assuming I am interpreting figure 1 correctly), especially if 
2. Is there a version of Figure 4 that also includes MV? I am especially interested in understanding what that might look like because the paragraph starting at line 252 (where RT is described to require more energy than MV) seems to directly contradict what I am seeing in Figure 1, where it seems like MV is always strictly more expensive than RT for a given model. Which is true?

### Soundness
3

### Presentation
2

### Contribution
3

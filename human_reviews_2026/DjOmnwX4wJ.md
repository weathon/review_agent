# Scaling Up, Speeding Up: A Benchmark of Speculative Decoding for Efficient LLM Test-Time Scaling

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 4, 6, 6

## Abstract
Test-time scaling has emerged as a powerful paradigm for enhancing the reasoning capabilities of large language models (LLMs) by allocating additional computational resources during inference. However, this paradigm is inherently inefficient due to the generation of different reasoning traces, leading to significant computational overhead. Speculative decoding offers a promising avenue for mitigating this inefficiency, yet its efficacy in the structured and repetition-rich context remains unexplored. To bridge this gap, we introduce the first comprehensive benchmark designed to evaluate speculative decoding methods in LLM test-time scaling. Our benchmark provides consistent experimental protocols across representative test-time scaling paradigms (e.g., Best-of-N sampling and multi-round thinking), enabling a fair comparison of three major categories of speculative decoding: model-based, training-based, and N-gram-based methods. Extensive experiments reveal that simple N-gram-based methods effectively capture repetitive patterns, demonstrating unique potential in accelerating test-time scaling. This phenomenon demonstrates the value of integrating N-gram-based methods with model-based or training-based approaches to benefit both repetitive and diverse reasoning in test-time scaling. We hope this benchmark spurs further research on speculative decoding for test-time scaling, enabling faster and more practical reasoning in LLMs through better handling of repetitive and diverse reasoning paths. Code available at <https://github.com/sunshy-1/SpecTTS-Bench>.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper presents comprehensive benchmark experiments evaluating three types of speculative decoding methods -- model-based, training-based, and N-gram-based -- in the context of LLM test-time scaling. The empirical results reveal several notable findings: (a) training-based methods exhibit robustness to sampling temperature, (b) N-gram-based methods are effective at identifying repeated patterns at the token level, and (c) hybrid approaches can leverage the strengths of both. These findings offer insights into the behavior of different approaches for reasoning models and may inspire further research on more effective methods for test-time scaling.

### Strengths
1. Empirically evaluates an important setting for the use of speculative decoding, i.e., test-time scaling for reasoning models. This setting can particularly benefit from more efficient inference, and understanding what works and why is highly valuable.

2. Evaluates a wide range of speculative decoding methods including model-based, training-based, N-gram-based, and hybrid methods.

3. Reports several notable findings, suggesting that hybrid approaches may be particularly promising.

### Weaknesses
1. The datasets used for the benchmark experiments are mostly focused on math reasoning. Evaluating how the findings extend to other types of data, e.g., coding, general reasoning, social reasoning, would provide more valuable insights into the methods.

2. Limited model scales. Results for models of scales other than 8B would strengthen the findings.

3. While the paper reports multiple notable findings regarding how different types of methods perform on the benchmarks, technical novelty (e.g., proposing a better working hybrid method) is rather limited.

### Questions
Q. Have the authors experimented with models of other scales, other types of reasoning benchmarks, etc.?

Q. How large is the variance for the performance numbers, e.g., reported in Tables 3 and 4.

### Soundness
3

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
3

### Summary
The paper presents the first benchmark for evaluating speculative decoding methods in test-time scaling of LLMs, offering standardized protocols across paradigms like Best-of-N sampling and multi-round thinking. Experiments show that simple N-gram-based methods efficiently handle repetitive reasoning patterns, suggesting their integration with model- or training-based approaches can make test-time scaling both faster and more effective.

### Strengths
1. The paper tackes an important problem of evaluating speculative decoding approaches in test-time scaling scenarios, and provide a formal benchmark.
2. The authors perform extensive evaluations ranging in nine different speculative decoding methods, four datasets, and two LLMs.
3. The authors provide a comprehensive analysis of the experiment results, along with clear takeaways for the readers.

### Weaknesses
1. As far as the reviewer understands, the reasoning traces were sampled sequentially in best-of-N scenarios, using batch size 1 (L264). However in realistic scenarios, these traces would be sampled in parallel (i.e. using batch decoding). This suggests a critical gap between the benchmark performance and real-world benefits.

2. The benchmark does not consider test-time tree search (e.g. beam search or DVTS [1]), which is another major test-time scaling strategy. (Note: This weakness is not critical, as the reviewer understands that it might add too much complexity to the benchmark.)

[1] Beeching et. al., Scaling Test Time Compute with Open Models

### Questions
Would the Best-of-N speedups be different if the reasoning traces were samples in a batch? (e.g. The amount of overlap would be much smaller for N-gram based models)

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
The paper discusses different methods of speculative decoding and tests them across reasoning benchmarks to measure MAT/wall clock speed up. 
The paper is a little limited in scope and could have been a bit more thorough. If the authors wouldn't mind running this on some larger benchmarks with different scaled models, I believe this would make a very nice paper. I'm still content with the paper, so I recommend acceptance.

### Strengths
1. It's a nice paper with useful results. 
2. The Takeaways are a pretty good analysis of what's going on with each speculative decoding method. 
3. Presentation is pretty nice. 
4. Practical results.

### Weaknesses
1. It's not really a benchmark. It's just running the different methods on common benchmarks. Not a huge deal since it's still good science. 
2. WOuld have been nice to scale these across difficult model scales. The results may have been different. 
3. Datasets are tiny, like 120 total samples. Could have done this on more deata. 
4. Error bars could have been nice if you could rerun this multiple times to see what happens.
5. Varying batch size would have been nice.

### Questions
N/A

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
2

### Summary
This is a benchmark paper that targets on accelerating test time scaling method by speculative decoding. It compares model-based, training-based, and N-gram approaches.
The key finding is that simple N-gram-based methods (like SAM) are highly effective at capturing this redundancy. A hybrid method, SAM[EAGLE-3], achieved the best overall speedup by combining N-gram's repetition capture with a training-based method's semantic prediction. N-gram methods also show "progressive acceleration," getting faster across multiple reasoning turns , though they are sensitive to temperature

### Strengths
- The problem this paper targets seems relevant, and the method is intuitive and motivated.

- There are many interesting findings and the experiments seem to be solid. For example, the most interesting finding is that simple N-gram-based methods, particularly SAM, are highly effective at capturing the redundancy in Best of N or Multi Turn answers. The findings are comprehensive.

### Weaknesses
- The one concern I have right now is the novelty of the method. Maybe this is because this is my first time reviewing on the dataset and benchmark field.

- Whether other complex reasoning frameworks such as Tree-of-thought or MCTS-based search algorithms could also benefits from speculative decoding?

- Also, would the approach generalized to other field such as code generation and summarization?

- The paper correctly identifies that hybrid methods (like SAM[EAGLE-3]) are a promising direction and achieve the best performance. However, it also admits that "current hybrid strategies remain heuristic" and that the potential of N-gram matching is "underexploited" due to "suboptimal integration strategies". The paper benchmarks one such simple strategy but does not propose or experiment with more dynamic or refined strategies to truly unlock this potential.

### Questions
See weakness.

### Soundness
3

### Presentation
3

### Contribution
2

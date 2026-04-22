# SWE-fficiency: Can Language Models Optimize Real-World Repositories on Real Workloads?

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 8, 0

## Abstract
Optimizing the performance of large-scale software repositories demands expertise in code reasoning and software engineering (SWE) to reduce runtime while preserving program correctness. However, most benchmarks emphasize what to fix rather than how to fix code. We introduce SWE-fficiency, a benchmark for evaluating repository-level performance optimization on real workloads.  Our suite contains 498 tasks across nine widely used data-science, machine-learning, and HPC repositories (e.g., numpy, pandas, scipy): given a complete codebase and a slow workload, an agent must investigate code semantics, localize bottlenecks and relevant tests, and produce a patch that matches or exceeds expert speedup while passing the same unit tests. To enable this how-to-fix evaluation, our automated pipeline scrapes GitHub pull requests for performance-improving edits, combining keyword filtering, static analysis, coverage tooling, and execution validation to both confirm expert speedup baselines and identify relevant repository unit tests. Empirical evaluation of state-of-the-art agents reveals significant underperformance. On average, top agents achieve less than 0.15x the expert speedup: agents struggle in localizing optimization opportunities, reasoning about execution across functions, and maintaining correctness in proposed edits. We release the benchmark and accompanying data pipeline to facilitate research on automated performance engineering and long-horizon software reasoning.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces SWE-FFICIENCY, a new benchmark designed to evaluate the ability of large language models to perform repository-level performance optimization on real-world workloads. The authors identify a gap in existing benchmarks, which primarily focus on functional correctness rather than code efficiency. The paper proposes a systematic data collection pipeline to curate 498 optimization tasks from 9 popular data-science and HPC repositories. The authors evaluate several frontier LM agents on this benchmark and find that they significantly underperform human experts, achieving less than 0.01x of the expert speedup and struggling with bottleneck localization and correctness preservation.

### Strengths
1. The paper addresses an important problem. As LMs become more integrated into software engineering, evaluating their ability to optimize code for performance—not just correct bugs—is a critical research direction.

### Weaknesses
1. The contribution of the paper is limited. The data collection pipeline itself is not novel; it is just a multi-stage filtering process that combines existing concepts like keyword filtering, AST analysis, and code coverage. The proposed metric is quite common.
2. The experimental validation is somewhat limited.  The paper evaluates only four models in its main results and omits strong models from the full benchmark evaluation. While a "lite" evaluation is provided in the appendix, the absence of these models in the full comparison weakens the contribution.
3. While the paper identifies manual annotation of workloads (Stage 4) as a necessary step based on PR descriptions, it provides limited detail on the specific methodology used for this process. The guidelines, quality assurance protocols, or specific criteria given to the human annotators who wrote these workloads are not discussed.

### Questions
N/A

### Soundness
2

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
4

### Summary
This paper introduces SWE-bench Efficiency, an extension of the original SWE-bench benchmark that evaluates both the accuracy and efficiency of LLM in software engineering tasks. It adds runtime, token usage, and cost metrics to traditional correctness checks. The dataset includes real GitHub issues, buggy and fixed commits, and executable tests from multiple Python projects. Experiments on several LLMs reveal trade-offs between accuracy, speed, and cost, offering a clearer view of their practical usability in software development.

### Strengths
The paper attempts to evaluate the real-world efficiency of LLMs within a complete software engineering workflow. The dataset construction pipeline is clearly designed, and the evaluation metrics are appropriately expanded beyond correctness. The experimental results, which analyze factors such as task complexity and contextual understanding, provide valuable insights for understanding the practical capabilities of LLMs in software development.

### Weaknesses
1. The experimental evaluation is limited to four LLMs, which restricts the generality of the findings and makes it difficult to assess how well the benchmark scales across different model families.
2. The paper lacks deeper discussion of the experimental results—for example, why certain models perform better on specific tasks or how task characteristics affect efficiency

### Questions
See weakness please

### Soundness
2

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
This paper introduces SWE-FFICIENCY, a new benchmark designed to evaluate the ability of large language models (LMs) to perform performance optimization on real-world software repositories. The benchmark consists of 498 tasks derived from GitHub pull requests across nine popular data-science, machine learning, and high-performance computing Python libraries. For each task, an agent is given a codebase and a slow workload and must produce a patch that improves the runtime while ensuring all relevant existing unit tests continue to pass. The paper details the automated pipeline created to collect and validate these tasks, which includes filtering pull requests, identifying relevant tests through code coverage, and manually annotating workloads. The authors evaluate several state-of-the-art LMs and find that they significantly underperform compared to human experts, achieving less than 1% of the expert's speedup on average. The analysis reveals that LMs struggle with localizing optimization opportunities and often propose superficial or incorrect edits. The paper concludes by open-sourcing the benchmark to encourage further research in automated performance engineering.

### Strengths
- Thorough analysis of the performance of current state-of-the-art models, revealing key weaknesses. 
- Insightful analysis of agent's solutions
- Well-designed evaluation framework, authors make their best effort to make a reproducible and reliable framework
- Manual work to ensure benchmark data verifiable and achievable

### Weaknesses
- Weakness1: Potential for Workload Oversimplification and Overfitting: A significant concern is that the benchmark relies on singular, self-contained workload scripts to represent performance issues. Real-world software performance is often highly sensitive to the context, scale, and statistical distribution of the input data. The benchmark's scripts, while curated, may not fully capture this complexity. This creates a critical risk: an agent could generate a patch that is perfectly "optimized" for the specific workload script but offers no general performance benefit, or could even cause regressions on different workloads. While the paper's section on "Workload overfitting and semantic drift" touches upon a related issue, it primarily addresses agents producing brittle "shortcut" solutions that cross into incorrectness (i.e., they cause semantic changes that fail tests or rely on environmental hacks). The weakness here is more fundamental to the performance claim itself. An agent could propose a patch that is semantically correct and passes all unit tests, yet its speedup is an artifact of overfitting to the benchmark's specific workload data patterns or scale. For example, an optimization might only be effective for small N, or for a sorted input array, which the workload script happens to provide. This means the benchmark may reward optimizations that are not robustly beneficial, creating a gap between a high score and the generation of genuinely useful, production-ready performance improvements.

- Weakness2: Concern on data source: The benchmark's exclusive focus on library repositories (e.g., pandas, numpy, scipy) introduces an ambiguity regarding the real-world impact of the optimization tasks. These libraries typically expose two kinds of functions: low-level, highly-optimized core routines (e.g., a sorting algorithm, a matrix operation) and high-level "convenience" APIs, which combine these low-level routines to provide a simpler user experience. The benchmark does not appear to distinguish between these two cases. If an optimization task targets a high-level convenience API, its practical value can be questioned. A skilled developer who requires maximum performance would often bypass such a high-level API in their own application. Instead, they would use their domain-specific workload knowledge to compose a more direct and performant solution using the library's low-level building blocks. In such scenarios, the "real-world" solution is not to submit a pull request to optimize the library's high-level function, but for the end-user to rewrite their own code.

### Questions
see weakness

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
Previous work in performance benchmarking datasets for AI agents do not analyze how well AI agents perform at solving the open ended problem of performance optimization in real world repositories, while ensuring code correctness. In this work, the authors collect 498 performance optimization problems from well-known repositories like numpy, pandas, astropy, etc. Their pipeline consists of (1) scraping GitHub problems for pull requests (PRs) that mention performance optimization (2) using manually tagged performance optimization workloads, (3) setting up a reproducible docker container for this task. Overall, the authors find that LLM agents _rarely_ outperform human experts at performance optimization tasks.

### Strengths
-  The paper was a joy to read and I commend the authors for going the extra mile in making informative figures and eliciting insightful research questions!

### Weaknesses
**Desiderata of an ML Benchmark:** The introduction of `[7]` lists several requirements of the desiderata of a good ML benchmark. To summarize, they claim that a useful ML benchmark is one that is both difficult and realistic -- i.e. the tasks should be challenging for frontier models and agent evaluations while also ensuring the task is realistic. Without both features, the usefulness of a benchmark is severely hampered. For performance benchmarking, one such desiderata elucidated by one of the experts in this field is shown in `[3]`. 

**Is this a difficult benchmark?** Undoubtedly, yes. The authors do a lot of analysis that showcases that current models struggle at this task. No need to mention these in the weaknesses section.

**Is this a realistic Benchmark?**: While it's impossible to do a comprehensive analysis in the limited space of this review, I found some fundamental flaws:

- I was pretty excited to try out a couple of the workloads listed in `L702` through `L766` but there are some issues here.
  - **Background:** Measuring microbenchmarking performance is generally regarded as an extremely hard problem `[1]`.
  
    - One solution for this is to have a fully emulated machine (like the PIE paper cited). However, this is impractical for most settings.
    - As such, one of the widely accepted solutions to this problem is to ensure a *stable* benchmarking state is reached before sampling `[2]`. **Specifically, proper benchmarking requires we keep sampling measurements until the samples appear to be in a steady state** `[2][3][6]`. There isn't an easy test for reaching a steady state.`[2]` says that we should simply _look_ at the samples. However, the general test for this is to throw out the first few samples. This is commonly referred to as a `warmup` parameter `[9]`.
  
  - Here's what I tried out:
  
    ```bash
    $ cat > bench_timeit.py << 'EOF'
    import timeit
    import statistics
    import dask
    import dask.array as da
    import numpy as np
    
    def setup():
      global stacked, sub_arrays
      sub_arrays = [
      da.from_delayed(
      dask.delayed(np.zeros)((100000,), dtype="int64"),
      shape=(100000,),
      dtype="int64",
      name=idx
      )
      for idx in range(10000)]
      stacked = da.stack(sub_arrays)
    
    def workload():
      global stacked, sub_arrays
      for i in range(len(sub_arrays)):
        stacked[i]
    
    runtimes = timeit.repeat(workload, number=1, repeat=5, setup=setup)
    
    # Print runtime mean and std deviation.
    print("Mean:", statistics.mean(runtimes))
    print("Std Dev:", statistics.stdev(runtimes))
    EOF
    
    (dask) $ python bench_timeit.py      
    Mean: 2.1401430334
    Std Dev: 0.19491263251372825
    
    (dask) $ python bench_timeit.py      
    Mean: 2.2334072502
    Std Dev: 0.11180160918436782
    
    (dask) $ python bench_timeit.py      
    Mean: 2.3481666415999998
    Std Dev: 0.20249011621291493
    
    ###############################################################
    ### The standard deviation is around 9% of the mean. 
    ### This could be the difference between a future
    ### gpt-6 and gpt-6.5!
    ###############################################################
    
    $ cat > bench_timeit_steady.py << 'EOF'
    import timeit
    import statistics
    import dask
    import dask.array as da
    import numpy as np
    
    def setup():
      global stacked, sub_arrays
      sub_arrays = [
      da.from_delayed(
      dask.delayed(np.zeros)((100000,), dtype="int64"),
      shape=(100000,),
      dtype="int64",
      name=idx
      )
      for idx in range(10000)]
      stacked = da.stack(sub_arrays)
    
    def workload():
      global stacked, sub_arrays
      for i in range(len(sub_arrays)):
        stacked[i]
    
    runtimes = timeit.repeat(workload, number=1, repeat=30, setup=setup)
    
    # Print runtime mean and std deviation.
    print("Mean:", statistics.mean(runtimes[25:]))
    print("Std Dev:", statistics.stdev(runtimes[25:]))
    EOF
    
    (dask) $ python bench_timeit_steady.py
    Mean: 2.114412525199999
    Std Dev: 0.02751709010275175
    
    (dask) $ python bench_timeit_steady.py
    Mean: 2.0973424249999986
    Std Dev: 0.042199462811280994
    
    (dask) $ python bench_timeit_steady.py
    Mean: 2.1530987250000004
    Std Dev: 0.034872376193247165
    
    ###############################################################
    ### Now, The standard deviation is around 1.6% of the mean with
    ### the same 5 samples! Slightly more respectable, but can be
    ### better still.
    ###############################################################
    ```
  
  - Notice three things in my modifications:
    - The sample size remains the same. I'm throwing out the first 25 measurements because I don't think _my system_ will reach a steady state till then. Warmup needs to be configured per-workload per machine.
    - The standard deviation is substantially smaller even though we are using the same number of samples.
    - Finally, the mean is much better behaved.
  - Looking deeper into this, it seems like the issue is with `timeit` itself.  **It is well known in the python performance optimization community that using `timeit` is not reflective of realistic workload settings** `[8][9][10][11]`. To summarize some of the issues:
  
    - `timeit` doesn't take any warmup into account. This causes unneccessary noise during initialization which pollutes the samples.
    - The original workload uses a **static** number of benchmark runs. We should be running the bench until it reaches a steady state.
    - `timeit` disables the garbage collector, which results in unrealistically good performance with respect to real-world performance. There are ways to enable this, but the workload doesn't pay attention to that.
    - `timeit` only uses one benchmarking process. Python is sensitive to a processes' memory layout and with address space layout randomization, the layout varies from one run to another (causing reproducibility issues).
    - Five samples does not seem enough to do a statistical validation experiment for two similar frontier models!
  - **Takeaway:** This seems like a fundamental problem with the benchmark construction. The authors are using `timeit` for benchmarking, and it seems like this was unfortunately an intentional decision (`L1074`).
  
    - I understand that, in some cases, the patch might yield improvements so large that even timeit shows the performance improvement. However, if we were analyzing the difference between two competitive frontier models, the unsoundness in the benchmarking apparatus will undoubtedly lead to incorrect assessments -- making this benchmark _detrimental_ to the field.
  
- **Another issue**: SWE-fficiency's workloads seem to be myopically focused on just measuring *runtime* for an isolated part of the codebase, while in the real world, performance optimization requires a tradeoff between multiple competing resources and user-facing APIs. For example:

  - In the `scipy` example, one way of subverting SWE-fficiency's workload is for an agent to always *cache* the results of each call to `maxwell.fit(data)`.
    - This will dramatically improve the runtime (or cause no change in the worst case) at the cost of substantial additional memory requirements to hold the potentially huge `data` tensor.
    - It won't cause any issues with the correctness checks either. 
    - SWE-fficiency would reward such an agent.
    - However, a procrustean memory-for-runtime tradeoff is fundamentally undesirable. A scipy core maintainer would outright reject such a patch (or at-least ask for a memory usage investigation).
  - Also, what if an agent optimizes `maxwell.fit` well but causes a significant slowdown in other parts of the codebase? The _workload coverage_ is extremely important here as well!
  - **Takeaway:** The core issue here is that *the correct workload to benchmark a program changes based on the features of the proposed patch*. (This is Rule#2 of the checklist presented in `[3]`. 

- Furthermore, the environment setup in `L1185` might not be enough to ensure reproducible benchmarking. From the [pyperf docs](https://pyperf.readthedocs.io/en/latest/system.html#features-of-intel-cpus): modern Intel CPUs have many dynamic features impacting performance:
  - HyperThreading: run two threads per CPU code, share L1 caches
  - Turbo Boost: CPU frequency is optimized for best performances depending on the number of “active” cores, CPU temperature, etc.
  - P-state and C-state: the frequency of a CPU core frequency changes depending of C-state and P-state which are tuned by the operating system (by the kernel).

**Overall Takeaway:** The takeaway from this analysis should be that __constructing realistic microbenchmarking workloads is a highly non-trivial task__ even for human engineers, a point that has been elucidated multiple times in relevant software engineering literature `[1][2][3][4][5][6]`, and that special care needs to be taken to ensure that both the workload being measured and the measurement procedure are finetuned for the task. The current methodology seems to overlook this.

A useful benchmark is both difficult and realistic `[7]`. SWE-fficiency does a great job of being a difficult benchmark, however, the benchmark fails my test for being realistic because:

1. The benchmark apparatus seems unsound. The software engineering community spent a great deal of time in understanding how to correctly construct and measure microbenchmarks `[1-6][8-11]`, which aren't being incorporated in the current workload construction. This leads to an unnecessarily higher signal-to-noise ratio.
2. The benchmark isn't comprehensive.
   1. The benchmark focuses on a single resources.
   2. Each task consists of a single workload, which won't provide adequate coverage of the entire repository. This can lead to cheating and cases where a workload speedup causes significant slowdowns in other parts of the codebase to go unheeded.
3. The workloads should be verified by a core developer with deep familiarity with the codebase and python benchmarking.
   * There might be other issues in other benchmarks that I did not catch. I think it might be advisable to get the benchmarking apparatus and parameters checked by a core developer for each repo.

All these tasks seem extremely non-trivial, and the paper would benefit from another round of reviews. The above issues seem like fundamental flaws and, as such, I'm in favor of rejecting this paper.  

Regardless, I'm happy to further discuss these points in the discussion section and potentially revise my rating if I've missed something fundamental!

===


Resources (sorry that some of them are blog posts; the intersection of the group of performance engineers and that of academics seems to be small :) ):
```
[1]: ASPLOS 2009 - Producing Wrong Data Without Doing Anything Obviously Wrong!
(https://users.cs.northwestern.edu/~robby/courses/322-2013-spring/mytkowicz-wrong-data.pdf)

[2]: ACM ISMM 2013 -  Rigorous Benchmarking in Reasonable Time 
(https://kar.kent.ac.uk/33611/)

[3]: Brandon Gregg's Blog 2018 - Evaluating the Evaluation: A Benchmarking Checklist
(https://www.brendangregg.com/blog/2018-06-30/benchmarking-checklist.html)

[4]: Joshua Bloch's talk on "Performance Anxiety" (for Java)
(https://wiki.jvmlangsummit.com/images/1/1d/PerformanceAnxiety2010.pdf)
(Blog summary: https://theholyjava.wordpress.com/2010/12/10/joshua-bloch-performance-anxiety-on-unpredictability/)

[5]: Aysylu Greenberg's talk on "Benchmarking: You're doing it wrong" (for Networking)
(https://jaxlondon.com/wp-content/uploads/2015/10/Benchmarking-Youre-Doing-It-Wrong-Aysylu-Greenberg.pdf)

[6]: Benchmarking correctly is hard (and techniques for doing it better)
(https://jvns.ca/blog/2016/07/23/rigorous-benchmarking-in-reasonable-time/)

[7]: UQ: : Assessing Language Models on Unsolved Questions
(https://arxiv.org/pdf/2508.17580)

[8]: Python Docs
(https://docs.python.org/3/library/timeit.html)

[9]: Pyperf Docs
(https://pyperf.readthedocs.io/en/latest/cli.html)

[10]: A tutorial on Code Optimization
(https://informatique-des-deux-infinis.pages.in2p3.fr/pheniics/make-your-code-more-efficient/microbenchmarking-python.html)

[11]: Dress Notes on Python Benchmarking
(https://www.dreesnotes.com/software/2024/05/14/benchmarking-python-with-pyperf.html)
```

### Questions
In the weaknesses section.

### Soundness
1

### Presentation
4

### Contribution
2

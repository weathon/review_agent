# BeGin: Extensive Benchmark Scenarios and An Easy-to-use Framework for Graph Continual Learning

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 3, 5, 3

## Abstract
Continual Learning (CL) is the process of learning ceaselessly a sequence of tasks. Most existing CL methods deal with independent data (e.g., images and text) for which many benchmark frameworks and results under standard experimental settings are available. Compared to them, however, CL methods for graph data (graph CL) are relatively underexplored because of (a) the lack of standard experimental settings, especially regarding how to deal with the dependency between instances, (b) the lack of benchmark datasets and scenarios, and (c) high complexity in implementation and evaluation due to the dependency. In this paper, regarding (a) we define four standard incremental settings (task-, class-, domain-, and time-incremental) for node-, link-, and graph-level problems, extending the previously explored scope. Regarding (b), we provide 31 benchmark scenarios based on 20 real-world graphs. Regarding (c), we develop BeGin, an easy and fool-proof framework for graph CL. BeGin is easily extended since it is modularized with reusable modules for data processing, algorithm design, and evaluation. Especially, the evaluation module is completely separated from user code to eliminate potential mistakes. Regarding benchmark results, we cover 3$\times$ more combinations of incremental settings and levels of problems than the latest benchmark.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes a graph continual learning dataset and defines standard incremental settings and evaluations. It proposes a trainer to evaluate the performance.

### Strengths
1. The proposed datasets seem to be useful in graph incremental learning.
2. The dataset settings and evaluations seem to be sound.

### Weaknesses
1. The authors do not propose their method for graph incremental learning.
2. The organization of this paper is not good. The main contribution is the dataset but there is no intuitive information of what the dataset looks like. For example, what it contains, the scale, and other information.
3. The performance in Table 3 is not complete.

### Questions
see the weakness

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper introduces a benchmark specifically designed for continual learning (CL) on graph data, diverging from the conventional focus on independent data types like images and text. The framework utilizes 20 public datasets, segmented into multiple tasks tailored to various continual learning settings, including task-, class-, domain-, and time-incremental settings at node, edge, and graph levels. The performance of diverse CL methods under these settings is assessed using four evaluation metrics: average performance, average forgetting, intransigence, and forward transfer.

### Strengths
1.	The general motivation of building benchmarks for continual learning on graphs is interesting and practical for the graph learning community. 
2.	There are some merits in the design of the experiments. Different settings are considered and evaluated. 
3.	The code is well-organized and the documentation is relatively easy to understand.

### Weaknesses
1.	One major concern is the large incompleteness of the evaluation results(about one third of entries in tables are n/a). The author should be clear on why methods are not applicable. Besides, three methods (PackNet, Piggyback, HAT) only support the task-IL setting. Why does the author include them as part of baselines, since they are not generalizable?
2.	There are a few claims that are doubtful in terms of the comparison with exciting benchmarks. In Appendix G.3 and Table 9, the authors claim that the same implemented method has about on average 28.6% fewer lines of code. However, simply comparing the line counts is not meaningful to demonstrate its own strength. The reviewer suggests the author calculate the running time of each method implemented by the author and existing works for a fair comparison. 
3.	The reviewer appreciated that the author stated the weakness in terms of the small number of tasks due to the limited labeled data. However, this is also an important contribution to the benchmarks (and differentiates from the existing works). Even a synthetic dataset is supposed to suffice.
4.	A few results seem problematic. For the methods ‘PackNet’ and ‘Piggyback’, the average forgetting on a few datasets is 0.000±0.000 for each number of tasks. The number of digits seems not enough to differentiate the results. Other reasons need to be discussed if available.

### Questions
Please see weakness 1 to 4.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper presents an extensive benchmark for graph continual learning. Currently, literatures in graph continual learning domain are lack of standards and inconsistent with the settings and evaluation dataset and metrics. This work aims to solve this problem by proposing a comprehensive evaluation benchmark with 31 scenarios based on 20 real world graphs. The work groups graph continual learning settings into four, including task-incremental, class-incremental, domain-incremental and time-incremental, and 3 levels of problems Moreover, this work provides the evaluation results of 10 graph CL methods using the provided benchmark.

### Strengths
Graph continual learning is a relatively unexplored field compared to continual learning research in other modalities. A primary obstacle is the lack of a standard evaluation protocol, including benchmark datasets, settings, and metrics. This paper addresses this gap, working diligently toward establishing such standards.

In this work, the authors systematically propose a benchmark designed to cover all possible scenarios in graph continual learning. Furthermore, for each scenario, results from 10 graph continual learning methods are provided, facilitating future comparisons and research.

The technical details supplied in the paper are comprehensive, and the codebase is well-documented, adding value to the paper and aiding in reproducibility.

### Weaknesses
While the authors have diligently developed a benchmark comprising 12 combinations of graph CL settings, the comprehensiveness of this benchmark in capturing the breadth of potential scenarios within graph CL problems remains somewhat ambiguous to me. Taking a different vantage point, such as data availability in the past, introduces unique scenarios. For instance, a situation where past data is inaccessible could arise, drawing parallels to the data-free setting observed in image CL problems [ref-1].

[ref-1] Always Be Dreaming: A New Approach for Data-Free Class-Incremental Learning, James Smith et al., ICCV 2021


Given the distinct nature of graph CL, scenarios involving incremental learning with the addition of new nodes typically presume a direct connection to the existing graph. This raises an intriguing question: what happens when one or more of these new nodes are not interconnected? Addressing and discussing this scenario would significantly enhance the scope and utility of this benchmark paper.

### Questions
It is not clear to me the motivating example shown in Figure 3 (there are changes in (a) the number of nodes, (b) the number of edges, (c) the number of classes (or domains), and (d) the distribution over classes) is covered by the proposed settings. Could you provide a detailed explanation to clarify this matter?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The work is a benchmark for graph continual learning, which does assume the graph to learn is fixed. In other words, it is a setting that extends the continual learning/ lifelong learning to graph data.

### Strengths
1. This paper propose a easy-to-use framework for implementation of graph CL methods. 

2. Some preliminary experimental results using the framework are provided.

### Weaknesses
1. This paper does not have any technical contribution, and is mostly dividing the existing public graphs into different tasks for graph continual learning, therefore may not be suitable for ICLR.

2. Even from the perspective of proposing new datasets/benchmark/evaluation, this paper has limited contribution. The proposed continual learning setting and splitting is same as standard continual learning on non-graph data, and the work in this paper to divide them is trivial. 

3. As for the contribution of a new framework for implementation and evaluation for graph CL methods. It seems like the contribution is an easy-to-use software for facilitating coding, which is off the scope of AI research.

### Questions
1. What is the essential difference between CL and graph CL? Is simply an extension of CL methods to graph data?

2. There are also other works on graph CL, is the experiment settings in this paper consistent with them?

3. Among the datasets used in this paper, which are already used by graph CL works, which are new?

4. For the same datasets used by both this work and the other graph CL works, are the experimental settings the same? Are the results the same?

5. The methods experimented in this paper also include some works with public code, e.g. GEM. Can the proposed framework obtain reasonably similar results as their official implementation?

6. Can this framework be used on computer vision tasks?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
1 poor

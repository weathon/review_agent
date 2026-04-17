# CircuitNet 3.0: A Multi-Modal Dataset with Task-Oriented Augmentation for AI-Driven Circuit Design

- Decision: Accept (Poster)
- Scores: 6, 4, 8, 4

## Abstract
Integrated circuit (IC) designs require transforming high-level specifications into physical layouts, demanding extensive expertise and specialized tools, as well as months of time and numerous iterations. While machine learning (ML) has shown promise in various research domains, the lack of large-scale, open datasets limits its application in chip design. To address this limitation, we introduce CircuitNet 3.0, a large-scale, comprehensive, and open-source dataset curated to facilitate the evaluation of ML models on challenging timing and power prediction tasks. Starting with a diverse set of 8,659 validated open-source designs, we employ a systematic framework to generate over 15,000 instances. Through specialized syntax-tree mutation strategies and principled, task-oriented filtering methodology, we enrich each design with multi-modal information spanning multiple design stages, including complete design flow documentation, register-transfer-level (RTL) designs and corresponding netlists, detailed physical layouts, and comprehensive performance metrics. The experimental results demonstrate that ML models leveraging the enriched multi-stage, multi-modal circuit representations significantly improve performance over existing open-source datasets in electronic design automation (EDA) tasks, paving the way for efficient and accessible circuit representation learning. The dataset and codes are available in \url{https://github.com/sklp-eda-lab/iclr-circuitnet_3.0/}.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper presents CircuitNet 3.0, a benchmark dataset designed to support AI-driven EDA research. It introduces a multi-modal framework that integrates RTL, gate-level netlist, and layout representations, enabling cross-stage learning for circuit-level prediction tasks such as timing and power estimation.

### Strengths
The paper presents a well-structured and comprehensive effort to advance dataset resources for AI-driven EDA research. Its originality lies in integrating multiple design representations (RTL, gate-level netlist, and layout) into a unified multi-modal dataset, and provides a new level of data granularity for model training. The proposed task-oriented augmentation framework demonstrates an application of AST-based code mutation in the EDA context, contributing to the diversity and robustness of learning samples.

### Weaknesses
1. All reported improvements are based on models trained with augmented datasets that are larger than the original data or baseline benchmark (Section 4.4 and Table 5–6). The paper lacks ablation or control experiment to verify that the observed gains stem from the augmentation methodology rather than simply from increased volume of the training set. 
2. For the timing-prediction task (Table 4a), the authors attribute performance gains of RTLDistil to multi-modal learning, yet the compared non-multi-modal baselines (MasterRTL, RTL-Timer) may differ substantially in architecture, capacity, and parameter count. There lacks controlled "with/without multi-modal input" experiment for timing.
Similarly, for MOSS in the power-prediction task (Table 4b), the attribution of performance gain solely to "multi-modal representation" is not fully justified without evaluating RTL-only, netlist-only, and combined-input variants under identical configurations.
3. CircuitNet 3.0 is positioned as a task-oriented benchmark, yet the tasks are limited to timing and power prediction. Other critical circuit-design objectives—e.g., routability and IR-drop mentioned in CircuitNet 2.0—are absent. By focusing narrowly on two metrics, the dataset may bias models toward timing/power trade-offs at the expense of broader PPA generalization.

### Questions
1. The claimed 3.95% error reduction in power prediction (e.g., line 92, 467 and 485) lacks exact calculation and appears inconsistent with MAPE reductions between 11–23% shown in Table 6.

2. The paper highlights "a novel data augmentation framework based on Verilog abstract syntax trees (ASTs)" (Section 4.3) as a core contribution. However, AST-guided code mutation and transformation have been widely explored in software-engineering contexts [1]. So it would be better if the authors could elaborate on how the proposed AST-based augmentation differs technically from prior AST mutation and program-transformation works.

3. The paper states (line 357) that the test set was 'supplemented with submodules from external open-source projects,' but does not justify the need for this supplementation or provide details on the data's origin. This omission raises concerns about potential overlap between the training and test data, especially since section A.2 lists GitHub, Hugging Face, RISC-V, and OpenCores as sources for both training and validation data

4. In Figure 6(b) and 6(d), both the post-augmentation WNS and total-power distributions exhibit an unusual concentration of samples around layout-density ≈ 94.5%, which is not present in the pre-augmentation distribution shown in Figure 6(a) and 6(c). Could the authors clarify:
(1) what specific synthesis or placement-density configurations led to this clustering?
   (2) why this effect appears near the 94.5% density rather than being smoothly distributed?
   (3) whether such clustering introduces dataset bias that may influence model evaluation or overfit models toward a narrow design-density region?

5. It would be beneficial for the authors to discuss how this work relates to recent benchmarks such as ChiPBench [2], which also targets AI-based EDA evaluation with a complete open-source flow.

[1] Petrović, Goran, et al. "Practical mutation testing at scale: A view from google." IEEE Transactions on Software Engineering 48.10 (2021): 3900-3912.

[2] Wang, Zhihai, et al. Benchmarking end-to-end performance of ai-based chip placement algorithms. The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track.

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
This work proposes CircuitNet 3.0, which contains 8,659 unique and validated source RTL designs and over 15,000 total augmented designs, each with corresponding netlist and layout representations.

### Strengths
In summary, this work proposes a promising and valuable open dataset that helps address the scarcity of digital circuit data. 

1. The writing is very clear. Especially, sufficient related prior works have been referred to. Good visualization and tables have been provided. 

2. This work tried to address the data limitation problem of IC design, which is a long-lasting problem.

3. This work tries to provide sufficient new RTL design, introducing diverse functionalities in the dataset. This brings much more design information to the dataset compared with CircuitNet 2.0.  

4. This work tries to validate the dataset in cutting-edge research works in the ML-assisted EDA direction.

### Weaknesses
Despite the contribution of the dataset, there is some key information not provided in this manuscript, triggering some additional questions. The author may consider adding them during the revision. The reviewer is ready to raise the score if these concerns can be well addressed. 

1. There should be a detailed list showing all sources of these  8659 samples, possibly in the Appendix. Currently, the paper only mentions "We systematically collected RTL designs from established platforms, including GitHub, Hugging Face, OpenCore, and RISC-V projects". 

2. To allow users to better understand the dataset, there should be statistical information describing the distribution of key features of all circuits in this dataset. How large are these circuits? For example, the user may provide the distribution of the number of RTL lines of all circuits and the number of gates in all corresponding netlists. These basic statistics have been adopted in many prior works. 

3.  Following point 12 about dataset information, Tables 2 & 8 show the design types. Do they indicate that there are 645 different adders and 906 different counters, which are all directly collected from human-crafted projects? If they are adder/counter modules only, then these components seem to be smaller and more repetitive than the reviewer had expected. How come there are so many different adders/counters from real designs? How are they extracted from realistic whole designs? Are some of them exactly equivalent to each other (not easy to believe there are 600+ different adder RTL designs and leading to completely different post-synthesis implementations)?   

4. The Anonymous repo is empty now.

### Questions
1. For the augmentation, does it maintain the original functionality on the augmented cases? If yes, then if the advanced synthesis tool, such as DC, is adopted, will these augmented RTL lead to the same post-synthesis PPA results as the original RTL?  

2. After the AST-based rewriting, how specifically do the authors convert the AST back to RTL code (which tool and major command)? Is the AST in bit-level or word-level? Does the AST-converted RTL code still look similar to human-generated RTL code?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper presents CircuitNet 3.0, which is a comprehensive multi-stage and multi-modal dataset that enables advanced AI-driven circuit design through cross-stage data augmentation and filtering.


The paper makes 3 contributions:

--A large-scale, multi-modal, and multi-stage digital circuit dataset with full RTL-to-layout
traceability. CircuitNet 3.0 contains 8,659 unique, validated source RTL designs and over
15,000 total augmented designs, each with corresponding netlist and layout representations.
Through an industrial EDA workflow, we extract rich cross-modal features at each design stage,
providing a valuable resource for research in multi-stage multi-modal representation learning.

-- A principled framework for data augmentation.  This enables robust learning for ML models,
providing simultaneous cross-stage analysis and early-stage prediction capabilities.

--A comprehensive set of new baselines and rigorous experimental protocols.

### Strengths
This system seems like it can be a useful addition to the circuit generation process. It clearly addresses the multiple modalities of data required for the entire process. The article does a good job of covering the sourcing, curation and inference properties of the different datasets.

The paper also details a comprehensive evaluation, which uses state-of-the-art ML models. The papers empirically demonstrates significant prediction accuracy improvements over single-modal datasets, with approximately 31.52% and 3.95% error reductions for timing and power tasks, respectively, compared to the existing dataset. Models trained on CircuitNet 3.0 consistently outperform single-modal approaches, establishing new performance benchmarks for ML-driven EDA tasks.

### Weaknesses
As a systems paper, there are different evaluation criteria. I think that there are few if any  clear weaknesses, as the overall system covers the full gamut of data, good curation/evaluation, and corresponding inference tools.

### Questions
1. What types of circuits suit this tool the best---this is not clearly spelled out.
2. It seems like using your tool requires significant computational power. Is it suitable for single-GPU users, and if so what generation times should one expect?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
CircuitNet 3.0 is presented as a large-scale, open-source, multi-modal, multi-stage dataset for ML-driven EDA. It starts from 8,659 curated RTL designs and is expanded to ~15k+ designs via augmentation. For each design, the authors claim to provide RTL, synthesized netlist, placed-and-routed layout, and timing/power metrics, to enable benchmarking of timing and power prediction models.

### Strengths
1. Cross-stage supervision. The multi-modal / multi-stage angle (RTL text, graphs/netlists, and layout images) is sensible.

2. Building this dataset required significant engineering effort, including full synthesis, place-and-route, and signoff-style timing/power extraction for thousands of designs

### Weaknesses
1. The practical impact is limited. Timing closure in real digital design is safety-critical and requires signoff-level accuracy under specific PDK, voltage, frequency, corners, etc. An ML model that only predicts approximate WNS/TNS or power from dataset patterns cannot be trusted in an actual tapeout flow, so the gains shown here are academically interesting but not directly usable.

2. The paper withholds critical details (PDK, voltage, frequency, corners, clock constraints, placement density targets).

3. Table 1 is overfull / too wide for the page.

### Questions
1.	Exactly which PDK / tech node, voltage was used?

2.	In the augmentation stage, you apply transformations like bidirectional substitution, comparison inversion, operator replacement (e.g. change + to *). But if you take an adder design and turn + into *, now functionally it's just… a multiplier. That might already exist somewhere else in the dataset as an actual multiplier module. So:
How do you avoid generating duplicates / near-duplicates of other designs you already have (e.g. two semantically different source modules collapsing into the same behavior after mutation)?

3.	How many AST-mutated samples survive functional verification? Give the acceptance rate.

4.	Power. Dynamic power strongly depends on input activity (toggle rates). If you only use one stimulus style (or even vectorless estimation), then the reported “power” is tied to that specific workload assumption. Did you generate multiple activity patterns per design (e.g. different input vectors / switching factors / functional scenarios), or is each design labeled with just a single power number?
 
5.	Timing. Timing depends heavily on physical context: floorplan, macro placement, routing congestion, target clock, etc. You say you're doing “early-stage timing prediction,” but what exactly is the task setup? Is the model supposed to take only RTL (plus spec like clock target) and directly predict final post-route arrival time / slack? Or are you also giving it partial physical info (e.g. rough floorplan, utilization target, cell library corner)? Please show a concrete example input/output pair for the timing benchmark: what does the model see, and what exactly is it supposed to predict? Without that, it's unclear how this task is meant to be used outside your own flow.

### Soundness
2

### Presentation
2

### Contribution
2

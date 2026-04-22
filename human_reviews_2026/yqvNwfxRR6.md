# Expertise Can Be Helpful for Reinforcement Learning-based Macro Placement

- Avg Score: 4.67
- Decision: Accept (Poster)
- Scores: 4, 6, 4

## Abstract
Chip placement determines the locations of electronic components on a chip layout, which directly impacts performance, power, and area (PPA) metrics, and thus is a critical step in electronic design automation (EDA). As modern chips scale to accommodate millions of components, manual placement by human experts becomes infeasible, necessitating the use of automated algorithms. 
Recently, reinforcement learning (RL) has emerged as a promising approach for automating macro placement, owing to its high optimization efficiency and potential for generalization. 
Despite their promise, existing RL-based methods often neglect the value of expert knowledge accumulated through years of engineering practice. They tend to optimize oversimplified proxy objectives, resulting in suboptimal placements that deviate significantly from expert-designed solutions. 
To bridge this gap, we propose a novel RL-based placement framework that integrates EDA domain expertise from two complementary perspectives: (1) Expert Knowledge Injection: Incorporating well-established placement knowledge, such as dataflow guidance, periphery bias, macro grouping, and I/O keepout constraints, to guide the learning process toward human-level solutions. (2) Expert Workflow Imitation: Emulating the post-refinement process of human experts (i.e., updating the design iteratively based on backend PPA feedback) to progressively optimize timing metrics by employing preference optimization.
Experiments on the ICCAD 2015 and OpenROAD benchmarks demonstrate that our method achieves substantial improvements in PPA metrics (e.g., 32.53\% in total negative slack and 7.74\% in worst negative slack compared to the runner-up method on average), outperforming advanced analytical, black-box optimization, and RL-based methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes the EXPlace algorithm to addresse the macro placement task. To mitigate the issue of the algorithm deviating significantly from expert-designed solutions, they employ two key strategies: Expert Knowledge Injection, which incorporates well-established placement knowledge, and Expert Workflow Imitation, which emulates the post-refinement process of human experts. The effectiveness of EXPlace is demonstrated on the ICCAD 2015 and OpenROAD benchmarks.

### Strengths
The paper is well-written, clear, and easy to follow; 

The guidance from expert knowledge is effective, the results are impressive, and the selection of expert knowledge is reasonable.

### Weaknesses
The main contribution of this paper lies in incorporating expert knowledge into the training process of reinforcement learning. It is well known that introducing expert knowledge during training can significantly benefit the early stages of learning, enabling the model to quickly reach a relatively high level of performance. This has been demonstrated in many works, such as AlphaGo. Therefore, the conclusions of this paper are not particularly novel. On the other hand, although the paper achieves impressive results, this may be due to the fact that using reinforcement learning to solve macro placement is still at an early stage. In such cases, the involvement of expert knowledge can lead to a noticeable improvement. However, one of the goals of AI is to surpass the limitations of human expert knowledge in order to achieve better performance. For example, algorithms like AlphaZero entirely abandoned expert priors and reached even higher levels of performance. In such scenarios, expert knowledge can become a local optimum that restricts further breakthroughs. Thus, I believe this paper represents a solid engineering effort within the current technological context, but its conclusions and level of innovation are limited in the long term. For other issues, please refer to the Question section.

### Questions
1. The expert knowledge is introduced as a cost signal in reinforcement learning and combined through linear weighting. Are there any conflicts between these different pieces of expert knowledge? Moreover, if more expert knowledge is incorporated, is this cost-summing approach still feasible?

2. How does the training cost change?

3. Can you provide a few examples of layout comparisons to demonstrate that the resulting layouts indeed reflect the intended effects of the expert constraints, rather than just improvements in evaluation metrics?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents an RL-based macro placement framework that integrates expert knowledge and workflow imitation, achieving significant PPA improvements over existing methods. The approach is innovative and practical, effectively bridging the gap between data-driven optimization and human design expertise.

### Strengths
- The experiments are very extensive across different placement benchmarks.

- The results are promising, it can achievethe  best placement results in different cases.

### Weaknesses
- The main contribution is to add three more expert masks, while adding different masks is not a very new idea. It has been attempted in previous work as MaskRegulator.

- According to Fig. 6 (a), the computation of the Periphery mask and the Dataflow mask is very time-consuming, resulting in relatively low sample efficiency.

### Questions
- How to collect the preference pairs D in the timing preference fine-tuning in ICCAD2015 or OpenROAD?
- Are there any ablation results with and without timing-driven fine-tuning?
- Why are there no DRC results in the ICCAD2015 benchmark?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents EXPlace, a reinforcement learning method for chip macro placement that systematically integrates domain expertise from EDA. The authors identify a key limitation in prior RL-based methods: their reliance on oversimplified proxy objectives, which leads to suboptimal results compared to human expert designs. EXPlace introduces Expert Knowledge Injection and Expert Workflow Imitation to address this problem. The method is evaluated on ICCAD 2015, demonstrating better performance compared to baseline methods.

### Strengths
1. This paper leverages the prior knowledge of human experts to enhance the final chip layout outcomes in this specialized domain.
2. Well-written and easy to read.

### Weaknesses
1. The runtime analysis is narrow, comparing EXPlace primarily against other RL methods. A more convincing efficiency demonstration requires benchmarking against a wider range of modern placers, including advanced analytical and black-box optimization methods.
2. For a comprehensive sign-off quality assessment, it is critical to include key industrial metrics like post-route power consumptionand final core area utilization.
3. The ablation study effectively tests several expert masks but omits a critical component: the periphery biasing mask. 
4. The generalization test—training on one circuit and testing on four others from the same ICCAD 2015 benchmark—is promising but insufficient. Performance on the remaining three circuitsis unknown.

### Questions
See in weakness.

### Soundness
3

### Presentation
3

### Contribution
2

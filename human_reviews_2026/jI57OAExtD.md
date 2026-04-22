# Reinforcement Learning Agents in Quantum Code Discovery with Argmax-Preserving Quantization

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 6, 8, 2

## Abstract
Reinforcement learning (RL) has recently been employed to autonomously discover quantum error-correcting codes and their encoders tailored to specific noise models and hardware constraints. However, RL policies are highly sensitive to approximation errors, and conventional quantization often disrupts action ranking, leading to degraded exploration and suboptimal codes. We propose Argmax-Preserving Quantization (APQ), a quantization method that directly regularizes action ranking during quantization-aware training. APQ minimizes ranking errors between full-precision and quantized policies, ensuring stable action selection even under low-bit representations. To further safeguard correctness, we integrate a reward-safe constraint that bounds perturbations of Knill–Laflamme conditions under quantization. Experiments with policy-gradient agents on Clifford-simulated environments show that APQ maintains discovery of [[n, k, d]] codes with distance up to 5 using INT8 networks, achieving equivalent logical error suppression as FP16 baselines while reducing inference cost by 3.8×. Our approach demonstrates that decision-consistent quantization can substantially accelerate RL-based quantum code discovery without sacrificing the quality of discovered codes.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The manuscript deals with the quantization of RL policies in the application of RL to design quantum error-correcting codes.

### Strengths
* The work is extensive and appears to be free of fundamental errors.

### Weaknesses
* The topic is very specific, so its significance seems rather limited. It would be good to broaden the scope of application.
* The novelty seems to be quite limited.
* The introduction and motivation are somewhat incomplete.

Details and further comments:

The sentence “RL policies are highly sensitive to approximation errors” is unclear and confusing. This does not apply in general; the context must be established beforehand.

The sentence “conventional quantization often disrupts action ranking” comes across as far too abrupt. It must first be explained that policy quantization is being considered, and this must be convincingly motivated.

The abbreviation ANN is not introduced.

Knill–Laflamme is sometimes abbreviated as “KL” and sometimes as “K–L.”

Five seeds are disappointingly few. More experiments should be carried out or an explanation given as to why this is not possible.

### Questions
* When using RL to design quantum error-correcting codes, which is a very specific application, why should one be interested in performing policy quantization?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a decision-consistent reinforcement learning quantization method Argmax Preserving Quantiziton (APQ) to address key challenges in RL-based quantum error-correcting code discovery. Targeting the issue where traditional quantization disrupts action ranking in RL policy networks (leading to suboptimal codes), APQ can directly constrain action ranking errors during quantization-aware training, ensuring stable optimal action selection even with  INT8 representations. It further incorporates a Knill-Laflamme condition-based reward-safe constraint to guarantee post-quantization code performance. Sufficient experiments and comparisons demonstrate that APQ with INT8 networks can discover quantum codes with distances up to 5, achieving logical error suppression comparable to FP16 baselines while reducing inference costs by 3.8×. This breakthrough significantly accelerates RL-based quantum code discovery without compromising code quality, establishing a new paradigm for resource-efficient automated coding optimization on quantum hardware.

### Strengths
Originality：
This paper innovatively proposes an argmax-preserving quantization scheme, which to some extent addresses the impact of traditional quantization methods on policy selection in reinforcement learning strategies. Additionally, by introducing an extra reward protection constraint based on the K-L condition, it prevents disruption to quantum error correction theory. The proposed approach ensures that the quality of the discovered quantum error-correcting codes remains largely unchanged while significantly reducing inference overhead.

Quality：
The paper provides a well-motivated, theoretically grounded approach (APQ) with clear algorithmic details. The integration of quantization error bounds with RL policy training is technically sound. Experiments on Clifford-simulated environments convincingly show that INT8-quantized networks match FP16 baselines in code discovery performance while being 3.8× more efficient, while achieving lower logical error rates compared to other methods. These results reinforce the method’s practical viability. 

Clarity:
The problem statement, method, and results are logically organized, making the technical contributions accessible.The paper avoids excessive formalism while providing sufficient depth in explaining APQ’s mechanism. If included, diagrams or pseudocode would further aid understanding, but the textual description is already clear.

Significance:
By improving the efficiency of reinforcement learning-based code discovery, this research accelerates the design of customized quantum error correction codes, which is crucial for near-term fault-tolerant quantum computing. The approach of quantifying decision consistency may be generalized to other reinforcement learning scenarios that rely on action selection stability.

### Weaknesses
1.The system scale used for testing in the paper is relatively small (distance 5 under 25 qubits). Although the paper explains that this is comparable to previous reinforcement learning discovery systems, this scale remains far below both the typical size of current quantum chips and the quantum error correction code dimensions that are of primary concern in the field of quantum error correction.

2.This article focuses on proposing a new quantification scheme and improving reinforcement learning performance for quantum error correction code discovery tasks. Why not consider applying this method to determine fault-tolerant circuit constructions for quantum error correction codes? 

3.While the paper makes outstanding contributions to the field of reinforcement learning by proposing an innovative quantification method and thoroughly demonstrating its effectiveness, from the perspectives of quantum error correction and fault-tolerant quantum computing, its novelty appears somewhat limited based on the aforementioned two points.

### Questions
1. Figure 2 is not cited in the original text. 
2. Appropriately increase the exploration of quantum error correction codes for larger system sizes, and discuss the scalability of the method.
3. If this work primarily emphasizes the method, its generalizability should be considered, discussing the method's applicability to other problems. For instance, whether this method can be used to design fault-tolerant quantum circuits for known quantum error correction codes.

### Soundness
4

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
2

### Summary
The authors propose a quantization method that regularizes action ranking during quantization-aware reinforcement learning. The method, termed Argmax-Preserving Quantization (APQ), minimizes ranking errors between full-precision and quantized policies, effectively ensuring stable action selection under low-bit representations. The approach is numerically evaluated on Clifford-simulated environments, demonstrating that INT8 networks achieve equivalent logical error suppression compared to FP16 baselines. Furthermore, the results show that transitioning from FP16 to INT8 reduces inference cost by a factor of 3.8.

### Strengths
The automatic discovery of quantum error-correcting (QEC) codes and encoders is an area of significant interest, and the recent application of reinforcement learning (RL) to this task represents a promising and exciting direction for research. While I am not an expert on RL-for-QEC discovery, the paper does an excellent job of introducing the topic in a clear and engaging manner. The manuscript is well-written and effectively organized. The concept of decision-consistent quantization—where the top-1 action of the policy is preserved even under low-bit inference—appears to be a highly compelling and practically useful idea, extending beyond QEC applications. Additionally, the experiments solidly demonstrate the applicability of the method on the chosen benchmark, and the ablation study provides valuable insights into the contributions of the algorithm's different components.

### Weaknesses
Minor Issues:
Figure 3: Since the x-axis does not represent a continuous-valued space, the dots in the graph should not be connected.

### Questions
I have no questions.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper investigates how action order can be distorted when employing discretized reinforcement learning (RL) policies for quantum error correction (QEC). It identifies that discretization introduces order irregularities that may harm policy quality and circuit fidelity. To address this, the authors propose a method to retain action order while improving convergence speed compared to fully continuous circuit modeling. The work sits at the intersection of RL-based quantum code discovery and efficient quantum circuit optimization.

### Strengths
- The paper tackles a timely and emerging topic at the interface of reinforcement learning and quantum error correction.
- The authors propose a computationally efficient adaptation that improves convergence speed and addresses discretization-related distortions in policy sequencing.
- The study demonstrates an appreciation for practical trade-offs between continuous and discrete RL models, highlighting potential gains in simulation efficiency.

### Weaknesses
- The experimental and methodological details are under-specified, particularly regarding the “teacher” setup and FP16 vs. INT8 roles.
- The main phenomenon (action order disruption through discretization) is not clearly demonstrated — the lack of small-scale, illustrative examples makes the argument abstract and unconvincing.
- Terminology is poorly introduced, and the writing lacks structure and clarity. Many core concepts appear abruptly without context or definition.
- The paper’s structure is unintuitive, with background and method sections overlapping.
- The contribution appears minor and specific to a niche scenario; there is no formal or theoretically grounded justification of the observed effects.
- The presentation quality (writing, readability, figure design) significantly hinders comprehension and impact.
- Adding graphical exemplifications or simplified cases (e.g., discrete collapse vs. continuous retention of action order) could substantially improve understanding and motivation.

### Questions
1. Could the authors clarify the role and implementation of the “teacher” (FP16) setup?
2. Why is the action order affected in the first place when using discretization in RL on quantum circuits? 
3. How robust is the proposed method to different forms of discretization or policy architectures? Does it generalize beyond the specific circuit models tested?
4. Could the authors provide a minimal working example illustrating the collapse or distortion of action order in a discrete setting? This would make the effect more tangible.

### Soundness
2

### Presentation
1

### Contribution
2

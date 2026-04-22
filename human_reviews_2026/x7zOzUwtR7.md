# Structure-Aware Graph Hypernetworks for Neural Program Synthesis

- Avg Score: 4.67
- Decision: Accept (Poster)
- Scores: 6, 6, 2

## Abstract
We study the neural program synthesis of $\textit{parameterized}$ function families through the lens of meta-learning with hypernetworks. Given a user intent $U$, a meta-learner $M_{\phi}$ produces a full weight set $\hat{\theta}=M_{\phi}(U)$ for a target neural network with fixed architecture $S$, and the instantiated network $m_{S,\hat{\theta}}(X)\to Y$ realizes the behavior intended for $U$. Classical hypernetworks typically $\textit{ignore the target network’s structure}$ and emit a flat list of weights; as a consequence, they fail to account for $\textit{neuron-permutation symmetry}$—many distinct parameterizations of $S$ implement the same function—so equivalent solutions are treated as different targets, fragmenting supervision and hurting out-of-distribution generalization. To address this, we propose $\textit{Meta-GNN}$, a hypernetwork that constructs a $\textit{neural graph}$ from the target architecture $S$ and applies $\textbf{structure-aware}$ message passing with parameter-tied encoders and decoders. This design reduces the search space during learning by collapsing equivalent classes of target networks, without loss of expressivity. Empirically, across modular arithmetic ($\textit{AddMod}$-$p$), array operations ($\textit{SumFirst}$-$n$), and inverse-rule tasks from 1D-ARC, $\textit{Meta-GNN}$ substantially improves learning and $\textbf{out-of-distribution generalization}$ compared to classic hypernetworks and direct $(U,X)\to Y$ baselines. Mechanistic analyses reveal $\textit{what is learned}$: on $\textit{AddMod}$-$p$ the synthesized Transformers recover the canonical clock representation and admit a compact closed-form map $U\mapsto\theta$. These results demonstrate that structure-aware Meta-GNNs enable reliable generalization to $\textit{unseen program parameterizations}$, providing a critical advance for the nascent field of neural program synthesis.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Traditional program synthesis methods that search within a template space rely heavily on handcrafted templates and incur substantial computational costs as template size increases. To bridge these gaps, the authors introduce the lens of meta-learning with hypernetworks, defining a neural program (NeuroP) as a differentiable program modality that enables continuous, gradient-based optimization. The authors then propose Meta-GNN, a structure-aware hypernetwork capable of generating the complete set of weights for a target architecture given a user intent U. Experimental results show that Meta-GNN demonstrates strong out-of-distribution (OOD) generalization compared to all baselines. It is notable that on the ADDMOD-p task, Meta-GNN learns a canonical clock representation, indicating that it captures underlying, generalizable computational regularities.

### Strengths
1. The proposed method makes meaningful contributions across multiple prior research directions and demonstrates clear originality:

   - Leveraging hypernetworks for program synthesis is an innovative idea that mitigates the dependence on handcrafted DSLs present in neuro-symbolic approaches.

   - The proposed Meta-GNN employs message passing and group-tied encoders and decoders to address the permutation symmetry problem in traditional hypernetworks.

   - Out-of-distribution (OOD) generalization is a critical challenge in both meta-learning and hypernetwork research, and the paper shows that Meta-GNN achieves strong OOD generalization performance.

2. The paper is well-structured and comprehensive, with clear formal definitions and thorough explanations of the key design components of Meta-GNN.

### Weaknesses
1. The claimed benefits of structure-awareness in improving generalization and performance are supported only by theoretical explaination, lacking ablation studies. Specifically, while the selected baselines partially demonstrate that greater structure-awareness correlates with better generalization, Meta-GNN, as a graph neural network, differs topologically from the MLP-based baselines. It would be valuable to further investigate the contribution of Meta-GNN’s own components.
  
2. In the experimental section, each task domain employs only a single target network configuration. As shown in Table 1, all target networks are shallow (mostly one-layer) , which limits evidence for scalability and cross-architecture generalization.
  
3. The paper lacks direct empirical comparisons with traditional methods, such as neuro-symbolic baselines, which would strengthen the claim of superiority over existing approaches.

### Questions
1. What other tasks are commonly studied in traditional program synthesis, and could this method be compared directly with classical or symbolic approaches on those tasks?
  
2. In Tables 2 and 3, Meta-GNN does not achieve the best generalization across all tasks. What might explain these cases?
  
3. Beyond demonstrating strong OOD generalization, does the structure-aware design offer additional advantages, such as smaller model size, reduced training cost, or improved efficiency?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper is in the field of hypernetworks. Given a user intent, it generate specific weights for another (target) neural network to complete the task. The main contribution is that the proposed architecture considered neuron-permutation symmetry (that is, for example, the order of neurons in a layer does not influence the functionality of the network). The architecture of the target network is transformed to a graph in which nodes are biases and edges are weights. While swapable nodes/edges are assigned different positional encodings to differentiate them, they are processed by the same encode function. $K$ message-passing processes are done on the graph, which include a permutation-invariant aggregation to merge all the informations from neighbors. Finally, every node and edge obtains an embedding which is then decoded as a scalar.

### Strengths
- As far as I know, this is the first hypernetwork that considered neuron-permutation symmetry, and the idea do make sense
- Three variants (Meta-MLP/gMLP/GNN) are proposed and experimented, which help illustrate the effectiveness of the proposed apporach
- A mechanistic analysis is provided, which shows strong evidence that the proposed architecture learns a generalizable solution

### Weaknesses
- While the tasks in the experimental section are illustrative, all of them are rather simple without significant practical usage

### Questions
N/A

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper considers the problem of program synthesis. The authors refer to a neural network that approximates a symbolic program as a “neural program” (“NeuroP”) and explore whether a hypernetwork can be effective at converting a program specification (given by “user intent”) into a neural program. They note that a neural program has many neuron-permutation symmetries that can hinder a hypernetwork’s performance. To address this, the neural program’s structure is represented as a graph, that is divided into weight subsets. The hypernetwork then outputs a neural program in 3 steps: 1) A subset-specific encoder transforms the user intent to a corresponding latent vector. 2) The graph structure is utilized, along with message passing, in order to ensure the permutation equivariance inside the necessary weight subsets. 3) A subset-specific decoder is used to generate the weights of a neural program.

The paper claims three contributions: establishing NeuroP as an alternative to symbolic program synthesis; providing empirical evidence that using structure-aware hypernetworks can lead to strong OOD generalization across diverse program families; releasing the code and datasets.

### Strengths
Appears to provide a novel approach (in the program synthesis literature) to generating neural programs using a hypernetwork.

Provide an interesting comparison between the performance of group-aware and group-and-structure aware hypernetworks.

Their experiments cover an MLP as well as the transformer architecture, demonstrating that the method is effective for both.

### Weaknesses
Limited evaluation - experiments and baselines. It’d be good to see how it compares to other neural program synthesis methods.

Limited novelty. From the three listed contribution, I would argue that the first (introducing the concept of a neural program) is not a contribution, while the third (releasing the code and benchmark) is of limited significance. The second listed contribution is “empirical evidence that intent-to-weight synthesis is feasible, and structure-aware meta-learners achieve strong OOD generalization across diverse program families.” However, in my opinion, this statement is not supported by the limited evaluation. It is possible that the way the hypernetwork is constructed is novel, but if so it should be explicitly states as a contribution.

Related work section appears to be lacking. The hypernetwork section discusses the original hypernetworks paper and then goes into MAML, while I imagine there are other important works that are related to this one (could be mistaken). Also, this section needs to incorporate other neural program synthesis approaches that represent the target program as a neural network (I don’t have a reference off the top of my head, please correct me in case all such references are already covered).

### Questions
The approach appears to be applicable to any problem where hypernetwork are useful. Why was program synthesis (of neural programs) chosen as the target domain?

### Soundness
3

### Presentation
2

### Contribution
2

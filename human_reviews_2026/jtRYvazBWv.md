# CLUE: Conflict-guided Localization for LLM Unlearning Framework

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 2

## Abstract
The LLM unlearning aims to eliminate the influence of undesirable data without affecting causally unrelated information.
This process typically involves using a **forget set** to remove target information, alongside a **retain set** to maintain non-target capabilities. While recent localization-based methods demonstrate promise in identifying important nodes (neurons) to be unlearned, they fail to disentangle nodes responsible for forgetting undesirable knowledge or retaining essential skills, often treating them as a single entangled group. As a result, these methods apply uniform interventions, risking catastrophic over-forgetting or incomplete erasure of the target knowledge. To address this, we turn to circuit discovery, a mechanistic interpretability technique, and propose the **C**onflict-guided **L**ocalization for LLM **U**nlearning fram**E**work (**CLUE**). This framework identifies the forget and retain circuit composed of important nodes, and then the circuits are transformed into conjunctive normal forms (CNF). The assignment of each node in the CNF satisfiability solution reveals whether it should be forgotten or retained. We then provide targeted fine-tuning strategies for different categories of nodes. Extensive experiments demonstrate that, compared to existing localization methods, CLUE achieves superior forget efficacy and retain utility through precise neural localization.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces CLUE, a new framework designed to improve the precision and effectiveness of large language model (LLM) unlearning. Existing localization-based unlearning methods identify important neurons for modification but fail to distinguish between neurons responsible for forgetting and those critical for retention. This entanglement often leads to over-forgetting or incomplete erasure of target information. To solve this, CLUE leverages circuit discovery, a mechanistic interpretability approach, to map neuron interactions as circuits. It then converts these into CNF and uses Boolean satisfiability solving to disentangle neurons into three categories: Forget neurons, Retain neurons, and Conflict neurons.

### Strengths
- The application of mechanistic interpretability and circuit discovery to LLM unlearning is novel and conceptually well-motivated, bridging interpretability with model editing.
- The explicit separation into forget, retain, and conflict neurons allows for more precise control and understanding of unlearning behavior.
- Experiments across multiple benchmarks show consistent improvements.
- The framework enhances interpretability by revealing how specific neurons and circuits contribute to different functions, aligning with future modular LLM design goals.
- The paper clearly articulates the problem of neuron entanglement and justifies the need for a more granular approach like CLUE.

### Weaknesses
- Although the section 5.1 and 5.2 documented well on benchmarks that the study is evaluted on plus the details on how the evaluation is conducted, I, somehow, could not find any descriptions on what model is evaluated. Is it the same model across all the benchmark evaluations? What is the family and the size of the model being evaluated in the study? 
- As the curse of circuit discovery, the framework would possibly also have scalability problem, such as being computationally expensive to find a meaningful ciruit in larger models. For meaningful circuit, I mean a circuit that is sparse enough comparing to the original computational graph. 
- The framework utilizes the edge pruning technique. In edge pruning, it returns a series of ciruits as the sparisity going down, while the metrics of the faithfulness would also vary. I wonder what are the specific critierion that was used to pick such circuit from the edging pruning outcomes. And how would you justify picking one of the circuits over the others ( maybe in terms of exact match, KL divergence, or sparisity)? 
- The framework appears to make an implicit assumption that the contribution of each neuron identified within a circuit is independent of the specific path or connectivity structure it participates in. For instance, consider a sequence of neurons 
𝐴1→𝐴2→𝐴3 across successive layers. CLUE might categorize 𝐴1, 𝐴2, and 𝐴3 into different neuron sets (e.g., forget, retain, or conflict) based on their individual CNF assignments. However, in reality, the functional contribution of each neuron is context-dependent on the path it forms with others. In contrast, circuit-level methods such as edge pruning often treat  A1→A2 →A3 as an integrated substructure, preserving or removing the path as a whole. This suggests that CLUE’s neuron-wise localization might overlook interdependent effects among neurons within the same circuit, potentially leading to suboptimal or fragmented unlearning.

### Questions
Please refer to the weakness section.

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
5

### Summary
This paper addresses the challenge of LLM unlearning removing undesirable knowledge while preserving unrelated capabilities. The authors note that existing localization based methods, such as WAGLE and MEMIT, struggle to disentangle neurons responsible for forgetting from those involved in retaining information. To overcome this limitation, they propose CLUE (Conflict-Guided Localization for LLM Unlearning), a framework that integrates circuit discovery and logical reasoning. Specifically, CLUE first uses circuit discovery to extract distinct “forget” and “retain” subcircuits, then transforms these circuits into Conjunctive Normal Form (CNF) using the Tseitin transformation. The CNF satisfiability problem is subsequently solved to classify neurons into three categories: forget, retain, and conflict. Finally, the framework applies a two-stage fine tuning strategy first fine-tuning the forget neurons using a forget loss, followed by fine-tuning the conflict neurons with both forget and retain losses. Experimental results on benchmark datasets such as WMDP-Cyber, WMDP-Bio, and PKU-SafeRLHF, using Zephyr-7B and LLaMA2-7B models, demonstrate that CLUE achieves superior forgetting efficacy and retain utility while modifying significantly fewer parameters than previous methods like GA, NPO, PO, MEMIT, DEPN, and WAGLE.

### Strengths
Clear theoretical grounding and stepwise formulation (Eq. 1–6).
Comprehensive experiments across tasks and metrics (1-Acc, MIA, ROUGE-L).
Detailed ablations isolating the effect of forget/conflict masks and finetuning strategies.
Insightful analysis of sparsity vs. faithfulness (Fig. 4) and neuron-type shifts post unlearning (Table 3).
Public anonymized code and strong reproducibility statement.

### Weaknesses
Circuits are treated statically; dynamic updates during finetuning could better reflect neuron shifts.
Computational overhead of SAT solving and circuit extraction is not benchmarked.
Limited discussion on scaling to models > 7 B parameters.
Retain set selection sensitivity deserves deeper analysis (beyond Figure 3 trends).
Language polishing could improve readability.

### Questions
For multiple retain sets, could a multi-objective SAT formulation handle overlapping conflicts more gracefully?
Is there a failure case where CNF becomes unsatisfiable for large overlapping circuits, and how is it handled?

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
3

### Summary
This paper addresses the limitation of localization-based methods for LLM unlearning, and proposes a new method called CLUE. CLUE leverages an existing mechanistic interpretability method to achieve more precise unlearning without over-forgetting. Extensive experiments confirm that CLUE achieves superior performance on standard benchmarks and metrics.

### Strengths
1. The authors provide code, which enhances the reproducibility of the work.
2. Extensive experiments and baseline comparisons are conducted to demonstrate the effectiveness of CLUE across multiple benchmarks.
3. The proposed method leverages a circuit discovery tool, which improves the interpretability of the LLM unlearning process.

### Weaknesses
1. **Introduction to prior works is not enough.** A detailed introduction to circuit discovery algorithms and brief introduction to Tseytin transformation should be provided to make the paper self-contained.
2. **Generality of the proposed method is unclear.** As mentioned at the bottom of the p.5, fine-tuning affects the circuits of the model. This affects the generality of the proposed method. Does CLUE require rerunning circuit discovery after every fine-tuning step? This computational overhead needs to be discussed in the limitation section.
3. **Highly dependent on the quality of the circuit discovery algorithm.** CLUE’s performance is tightly coupled with the quality of the underlying circuit discovery algorithm (e.g., Edge-Pruning used in the paper). If Edge-Pruning fails to generate meaningful or accurate circuits, the effectiveness of CLUE is questioned.

### Questions
1. As mentioned in Section 3.1, the circuit discovery algorithm extracts the **forget circuit** and **retain circuit**. However, it is unclear how the algorithm finds the forget and retain circuits, and how to distinguish them. Further explanations are needed.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents a new categorization of neurons that could benefit downstream neuron-based unlearning methods. The desiderata include forgetting as much harmful information as possible while retaining as much non-harmful information as possible. The authors use a method inspired by contemporary circuit-discovery methods to identify three types of neurons: forget neurons (related to the targeted forgetting information), retain neurons (corresponding to the expression of retained information), and conflict neurons (involved in expressing both). They then propose a new two-step unlearning method that (1) edits the forget neurons to erase the unwanted information and (2) fine-tunes the conflict neurons to remove the targeted information while strengthening the retained information.

Overall, the method performs well across all major benchmarks. Unfortunately, the paper is not very well written. Many of the concepts are not clearly explained, and the paper contains numerous typos and errors. Most importantly, the paper has a major conflict in attempting to connect circuit discovery with neuron-based unlearning approaches, as circuit discovery identifies circuits of components rather than individual neurons. See the weakness section for more information.

### Strengths
- **Good performance**: The method performs reasonably well across standard benchmarks, showing that it can be applied successfully in practice.
- **Interesting categorisation of neurons**: The proposed division into forget, retain, and conflict neurons is conceptually distinct and could help clarify certain aspects of unlearning.

### Weaknesses
### **Neurons vs. Components**
A major issue of the paper is the careless use of the terms neuron and component. In short, circuit discovery methods find circuits of model components (input/output nodes, attention heads, MLPs). Each of these components contains many neurons. The interface between model components and neurons is not trivial, especially given previous work on unlearning that focuses on MLP neurons. What exactly do you mean by a circuit of “neurons”? Do you count every neuron in the components that make up the circuit as such, or do you do something different?

For example, the following sentence suggests that circuit discovery methods find a circuit of neurons, which is clearly not the case:
>  The circuit extracted from the forget set—the forget circuit (Cf)—contains all neurons and activation connections required for the model to
produce original responses that are harmful

The authors need a thorough and clear review of their description and usage of neuron, component, and circuit terminology. This is not a minor nitpick, as these terms lie at the very centre of the paper. The incorrect use of terminology makes it really hard to precisely understand what the authors are doing.

### **Writing and Typos**
Unfortunately, the paper is not very well written. The key terms are not selected well. For example, the authors use the term “conflict” to refer to neurons that serve both the forget and retain sets. This is not a conflict, as conflict suggests an opposition or inconsistency between two objectives, whereas these neurons simply participate in representing both. A clearer and more precise term would help avoid confusion and make the intended mechanism easier to understand.

Furthermore, the paper is also littered with typos and errors. To name a few:
- Line 046: ... ents (**Wu et al.**; Yu et al., 2023). *The citation entry doesn't have a year. Although the paper is from ICLR 2023.*
- Line 047: ... (**Patil et al.**;. ... *The same problem appear across multiple places in the paper.*
- Line 132: task-relevant behavior **( or** mechanism/capability) ... *Extra leading space in brackets.*

As such, I think the paper needs some major revisions. It seems like an unfinished manuscript not ready to be accepted into a conference.

### Questions
Pleas see the weaknesses section.

### Soundness
2

### Presentation
1

### Contribution
3

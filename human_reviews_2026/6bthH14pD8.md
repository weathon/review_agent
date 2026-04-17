# IR-Agent: Expert-Inspired LLM Agents for Structure Elucidation from Infrared Spectra

- Decision: Accept (Poster)
- Scores: 2, 4, 4, 4

## Abstract
Spectral analysis provides crucial clues for the elucidation of unknown materials. Among various techniques, infrared spectroscopy (IR) plays an important role in laboratory settings due to its high accessibility and low cost. However, existing approaches often fail to reflect expert analytical processes and lack flexibility in incorporating diverse types of chemical knowledge, which is essential in real-world analytical scenarios. In this paper, we propose IR-Agent, a novel multi-agent framework for molecular structure elucidation from IR spectra. The framework is designed to emulate expert-driven IR analysis procedures and is inherently extensible. Each agent specializes in a specific aspect of IR interpretation, and their complementary roles enable integrated reasoning, thereby improving the overall accuracy of structure elucidation. Through extensive experiments, we demonstrate that IR-Agent not only improves baseline performance on experimental IR spectra but also shows strong adaptability to various forms of chemical information.
The source code for IR-Agent is available at https://github.com/HeewoongNoh/IR-Agent.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposed IR-Agent, a multi-agent AI system for structure elucidation from IR spectra. IR-Agent is composed of three specialized agents: Table Interpreation (TI) expert (extracts local substructures from the IR absorption), Retreiver (Ret) expert (provides global structural cues from retrieved candidates), and Structure Elucidation (SE) expert (integrates both sources of information to infer the final molecular structure). Experiments show improved performance on some experimental IR spectra.

### Strengths
1) Application of agents to an IR problem seems novel
2) An ablation study is done on TI and Ret experts, which supports one of the claims.

### Weaknesses
1) Lack of iterative correction or self-feedback mechanisms to improve the performance of agents
2) Performance improvements in Table 1, Table 2, and Fig. 2, and 3 are not significant
3) Lack of novelty in agentic system design.
4) No limitations of the method are explained.

Minor issues:
1) Figure 3 appears before Figure 2.
2)  Captions are not descriptive enough to fully grasp the results.
3) What are the numbers in the parentheses in the Figures and Tables?
4) The formal objective of molecular structure elucidation can be explained earlier in the intro.

### Questions
1) What is the SOTA non-agentic method to solve the structure elucidation problem in IR?
2) Is there a feedback mechanism in IR-Agents where the Experts can improve themselves in the process?
3) Is there a critique mechanism to monitor the performance of agents?
4) Typically, multiple-agent systems are applied in settings where a design target or goal is imagined and the agents get feedback to take smarter action in the next round. The LLM used in this paper works on solving a one-time task. Is it fair to refer to this system as a multi-agent system? I understand there are two LLMs involved, but I am asking about the "agency". What is making the method an agentic work?
5) Are there innovations in terms of agentic system design?
6) Is there a reason why IR-Agent with GPT-4o and 4o-mini does not outperform the Transformer? Is that expected?

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
5

### Summary
The paper introduces IR-Agent, a multi-agent framework for molecular structure elucidation from infrared (IR) spectra. Each agent is designed to emulate a specific aspect of human expert analysis (e.g., identifying functional groups, interpreting peaks, or integrating contextual cues), and the system combines their outputs to predict the full molecular structure, expressed as a SMILES string. The authors claim that this architecture reflects expert-driven reasoning and offers flexibility to incorporate auxiliary chemical information. Experimental results show moderate performance on benchmark IR datasets and limited gains when additional descriptors (e.g., atom types, carbon count, or scaffolds) are provided.

### Strengths
The paper tackles an important and challenging problem: automated molecular structure elucidation from spectroscopy data.
The proposed multi-agent architecture is conceptually interesting, providing a modular framework that could, in principle, improve interpretability and extensibility.
The experiments are clearly reported, including dataset details, model parameters, and top-k metrics.
The paper makes a valid attempt to align model design with expert workflows in chemistry.

### Weaknesses
Scientific limitations of the problem setup:
IR spectroscopy alone cannot uniquely determine molecular connectivity or size. As a result, the reported top-k accuracies (<20\%) confirm the fundamental ambiguity of the inverse problem. Even with added information (atom types, scaffolds, carbon counts), improvements are marginal (~3\%), limiting the real-world utility of the method.

Shallow “reasoning”:
While the framework uses large language models (e.g., GPT-o-mini) to emulate reasoning, the qualitative examples mostly show pattern annotations such as “the peak at 1700 cm⁻¹ indicates a C=O group.”
For chemists, this does not constitute genuine reasoning, which would involve cross-checking multiple spectral regions, verifying the chemical plausibility of substructures, and ensuring internal consistency (e.g., in anthracene C-H and C=C will be shifted due to the precess of the other rings).
Thus, the claimed interpretability remains superficial and non-scientific, weakening the paper’s argument that the agents emulate expert analysis.

Evaluation metrics are insufficient:
Only top-k accuracy is reported. This metric does not capture the chemical proximity or diversity of predicted molecules. Additional measures such as Tanimoto similarity, token-level accuracy, or functional-group recall would better reflect predictive utility in semi-automated settings.

Given that IR spectra alone cannot determine molecular structure unambiguously, the framework’s current performance is insufficient for realistic applications.
While the topic is relevant and the proposed architecture is creative, the results and qualitative analyses do not convincingly demonstrate meaningful chemical reasoning or practical utility. The low predictive accuracy, combined with the lack of interpretability beyond surface-level annotations, limits the paper’s contribution to the community.

### Questions
Did the authors use canonical SMILES when computing top-k accuracy?
What happens when the molecule is not included in the expert knowledge base or training dataset?
Could the authors include Tanimoto similarity, token-level, or functional-group metrics to quantify the chemical diversity of predicted candidates?
Can the authors provide examples of inter-agent reasoning beyond simple peak annotation (e.g., how agents collaborate to ensure structural consistency)?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes **IR-Agent**, a multi-agent framework for molecular structure elucidation from IR spectra. It consists of three components: the **Table Interpretation Expert**, responsible for retrieving functional groups; the **Retriever Expert**, which retrieves potential molecules; and the **Structure Elucidation Expert**, which integrates all information to infer the corresponding SMILES. Experimental results show that the proposed framework outperforms existing methods, although several key limitations remain.

### Strengths
The reasoning strategy of the proposed **IR-Agent** is both reasonable and consistent with that of human experts. Experimental validation is comprehensive, providing evidence of its superiority and the soundness of its architecture.

### Weaknesses
The framework was evaluated on a real experimental dataset; however, the overall accuracy of structure inference remains low, even when additional chemical information is provided, **making it far from suitable for practical applications**.

### Questions
+ In the **Task Description section**, the IR spectrum is represented as a one-dimensional array. However, different molecules possess distinct functional groups that are reflected at different wavenumbers on the x-axis, **often spanning different ranges**. How can a 1-D array adequately align these patterns and capture the variations in absorbance energy across such differing wavenumber ranges?
+ The **Table Interpretation Expert** is crucial because knowing the functional groups can almost directly determine the molecular structure. The current agent appears to make decisions based on relatively coarse wavenumber intervals, but is this level of accuracy sufficient? A comparison with traditional software such as **Omnic (Thermo Nicolet)** is important, as this is a highly complex problem involving baseline preprocessing, peak location detection, and other subtleties.
+ It is unclear whether the current IR-Agent is capable of handling samples with multiple molecular components. For mixtures, peak overlapping presents a major challenge that should be explicitly discussed. If the method is not designed for such cases, the applicable problem scope should be clearly defined in the paper.
+ In the **Experimental Setup section**, the training IR spectra are used as the database for the Retriever Expert. This raises an important concern about database coverage: does the retrieval database sufficiently include the molecular structures present in the test set? If the database does not fully cover the test molecules, retrieval in step 2 may frequently return incorrect candidates. The authors must therefore clarify whether and how such incorrect retrievals impact the downstream Structure Elucidation Expert (step 3). Conversely, if the database does fully cover the test structures and retrieval accuracy is high, it is important to quantify the incremental contribution of the Structure Elucidation Expert: to what extent does it improve final SMILES inference beyond what a strong Retriever alone (such as Omnic ) would achieve? 
+ While the paper reports good performance on real experimental patterns, one critical issue is the low overall accuracy. A survey of results from other published studies on the same or similar tasks would be helpful for context and reference for the reader.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces IR-Agent, a novel LLM-based multi-agent framework for molecular structure elucidation from infrared spectra. The system employs three specialized agents that leverage external tools to emulate expert analytical reasoning. While demonstrating promising performance and extensibility through prompt-based integration of chemical knowledge, the approach exhibits notable limitations.

### Strengths
The modeling section of this paper demonstrates a innovative and ingeniously engineered system. It successfully integrates the capabilities of LLMs with domain-specific tools, offering a novel solution pathway for scientific computing problems. Its most significant advantage lies in utilizing multi-agent collaboration to mitigate erroneous judgments that may arise from single-agent systems.

### Weaknesses
Data Limitations: With only 9,052 experimental data points, the dataset appears insufficient to adequately demonstrate model generalizability, particularly for large language models requiring substantial training data.

Insufficient Baseline Comparisons: The baseline evaluation lacks comprehensive comparisons with contemporary large models (e.g., Llama3, Claude-3-opus) that have been successfully applied in spectral interpretation tasks. Furthermore, the study fails to compare against state-of-the-art Transformer-based methods that might achieve comparable performance through architectural improvements.

Limited Evaluation Metrics: The sole reliance on accuracy as an evaluation metric is inadequate. The absence of structural similarity measures, such as the widely-recognized Fingerprint Tanimoto Similarity from the RDKit library, prevents comprehensive assessment of prediction quality.

Lack of Multi-agent Mechanism Analysis: The paper provides insufficient analysis explaining why the multi-agent approach outperforms single-agent systems. It fails to demonstrate specific scenarios where multi-agent collaboration prevents errors or enhances reasoning robustness beyond mere accuracy improvements.

Translator Dependency Unexplored: The study doesn't investigate how translation errors propagate through the system, leaving the model's sensitivity to the initial candidate SMILES quality unquantified.

### Questions
Data Scale and Generalization Capability
This study utilizes 9,052 experimental spectra for validation. Given the complexity of chemical space and the data requirements of large language models, could the authors provide evidence or discussion on how the framework generalizes to broader molecular diversity? For instance, has testing been conducted on external large-scale datasets or structurally unique compounds? Cross-validation on specific compound categories is recommended to address this limitation.

Comprehensiveness of Baseline Comparisons
Although the paper compares single-agent variants and basic Transformer models, recent studies (e.g., Guo et al.) have employed larger-scale LLMs (such as Llama-3 and Claude-3) for molecular parsing tasks. Could the authors supplement comparisons with such models? Additionally, have advanced Transformer-based methods (e.g., those incorporating enhanced attention mechanisms) been evaluated to ensure performance improvements are not achievable through simple architectural modifications?

Evaluation Metrics Beyond Accuracy
Relying solely on Top-K accuracy may not fully capture the quality of structural predictions. Would the authors consider incorporating structural similarity metrics (e.g., fingerprint similarity calculated via RDKit) to assess whether "approximately correct" predictions retain chemical significance? This would strengthen the argument for the framework's practical utility.

Mechanism Analysis of Multi-Agent Advantages
The superiority of multi-agent systems is primarily attributed to accuracy improvements. Could the authors provide qualitative case studies or error analyses to specify how task division and collaboration avoid typical failure modes of single-agent systems (e.g., handling ambiguous peaks or reconciling conflicting evidence)? Such analysis would clarify the conceptual advantages of the framework.

Sensitivity Analysis of the Initial Translator
The framework's dependency on the IR spectral translator raises concerns about error propagation. Could the authors quantify robustness by testing performance degradation under noisy or low-quality candidate SMILES? For example, how does the system perform when the correct structure is absent from the initial candidate set? Ablation studies on translator quality could delineate the framework's operational boundaries.

### Soundness
2

### Presentation
2

### Contribution
3

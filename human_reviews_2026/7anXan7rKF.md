# BioArc: Discovering Optimal Neural Architectures for Biological Foundation Models

- Decision: Reject
- Scores: 2, 6, 2, 6

## Abstract
Foundation models have revolutionized various fields such as natural language processing (NLP) and computer vision (CV). While efforts have been made to transfer the success of the foundation models in general AI domains to biology, existing works focus on directly adopting the existing foundation model architectures from general machine learning domains without a systematic design considering the unique physicochemical and structural properties of each biological data modality. This leads to suboptimal performance, as these repurposed architectures struggle to capture the long-range dependencies, sparse information, and complex underlying ``grammars'' inherent to biological data. To address this gap, we introduce BioArc, a novel framework designed to move beyond intuition-driven architecture design towards principled, automated architecture discovery for biological foundation models. Leveraging Neural Architecture Search (NAS), BioArc systematically explores a vast architecture design space, evaluating architectures across multiple biological modalities while rigorously analyzing the interplay between architecture, tokenization, and training strategies. This large-scale analysis identifies novel, high-performance architectures, allowing us to distill a set of empirical design principles to guide future model development. Furthermore, to make the best of this set of discovered principled architectures, we propose and compare several architecture prediction methods that effectively and efficiently prediction optimal architectures for new biological tasks. Overall, our work provides a foundational resource and a principled methodology to guide the creation of the next generation of task-specific and foundation models for biology.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The paper introduces BIOARC, a Neural Architecture Search (NAS) framework intended to discover optimal, specialized architectures for biological foundation models (BFMs). The authors rightly criticize the common practice of directly repurposing Transformer architectures from NLP for biological data. The ambition is to provide a "principled, automated architecture discovery" methodology.

To this end, they conduct an extensive NAS over a hybrid design space (combining CNN, Transformer, and Hyena/Mamba blocks) using a one-shot, weight-sharing supernet. This search is evaluated on multiple DNA and protein tasks, considering various tokenization and pretraining strategies.

The results are impressive: the discovered architectures achieve state-of-the-art performance while being considerably smaller than baselines like. The paper also presents valuable findings on the interplay between architecture, tokenization, and pretraining.

### Strengths
- Clear presentation: The paper is well-structured and clearly written. The figures are straightforward 


- Impressive Empirical Performance: The primary strength is the reported results. The discovered BIOARC architectures (Tables 1 & 2) consistently and significantly outperform much larger baselines (e.g., DNABERT-2, ESM-1b) across DNA and protein tasks. Achieving this with models that are an order of magnitude smaller (e.g., <5M vs >100M parameters) is a highly significant practical contribution.


- Comprehensive Evaluations: The authors are to be commended for the breadth of their study, evaluating their framework across:
    - two distinct biological modalities (DNA and protein).
    - multiple downstream tasks (e.g., TFP, SSP, Fold prediction).
    - The interplay of three key axes: architecture, tokenization, and self-supervised training strategy (MM vs. CL vs. from-scratch).


- Dependence of Tokenizer of architecture: The finding that the optimal tokenizer is highly architecture-dependent (Sec 4.4, Fig 5) is a valuable actionable insight that can be explored further.

### Weaknesses
- The paper's entire premise is to move away from "intuition-driven design" However, after discovering a "Hyena→Transformer→CNN" pattern (Fig 2), the authors' only explanation is a classic, post-hoc intuitive rationalization: "Hyena block to first captures long-range dependencies, with Transformer blocks in the middle to model complex contextual relationships and CNN blocks at the end to extract critical features" (Lines 384-386). This argument fundamentally undermines the paper's central claim. They have not delivered a "principled methodology" for discovery, but rather an empirical search which is not as principled as the authors claim.

- Unaddressed Weight-Sharing Bias: The entire NAS relies on a "Single Path One-Shot" (Line 229) supernet with weight sharing, which can introduce significant biases. The discovery of mixed-block architectures may be a direct artifact of the weight-sharing scheme itself, not a reflection of the task's intrinsic needs. For example, do mixed architectures simply converge more reliably, or access a more effective parameter space within the supernet, than pure-block architectures? Without extensive ablations training individual architectures (pure and mixed) from scratch outside the supernet, or validation on toy problems with known optima, it is impossible to know if the "Hyena→Transformer→CNN" pattern is a genuine discovery or just an artifact of the search methodology.


- The introduction of a GPT-4o-based "Agent" (Sec 3.4, 4.5) is also problematic as it might rediscover existing ML literature without acknowledgment. The agent's "superior performance" (Table 3) is the most baffling part. It dramatically outperforms a "Neural Network Predictor" and even a standard "LLM+RAG" (which also uses GPT-4o). The only explanation given is that "specific roles and task definitions... unlocks generalization potential" (Line 480). This is far too vague. What are these roles? What are these definitions? This "agentic framework" seems to be the entire source of the performance lift, yet it is completely sufficiently explained, and potentially unreproducible.

- The metrics in Table 3 ("Hit Rate," "Precision") are poorly defined. What constitutes the "ground truth" set of top architectures for these calculations? This section feels more like a gimmick than a serious methodological proposal.

- Mamba vs. Hyena: Line 216 explicitly states the supernet module types are "e.g., CNN, Mamba". However, Figure 2 and the entire subsequent analysis (e.g., Line 384) refer only"HYENA" blocks. Mamba blocks are not mentioned or discussed again after figure 1.

- Missing Statistical Rigor: For a paper that relies on comparing performance across many architectures, there are no confidence intervals, standard deviations on most tasks (Fig 4 is an exception), or statistical significance tests. The results in Tables 1 & 2 are presented as single-run point estimates. We have no way of knowing if the "top" architectures are statistically superior to any others, or just a result of random seed variation. 

- The claim to be the first to move "beyond intuition" is overly broad. The paper fails to properly situate itself against the large body of work on hybrid architectures (e.g., CNN-Transformer or SSM-Transformer models), which are already common.

### Questions
1. Addressing the "Principled Discovery" Contradiction: How do you reconcile your central claim of moving beyond "intuition-driven design" with the fact that your primary "discovery" (the Hyena→Transformer→CNN pattern) is justified only by a post-hoc, intuitive argument (Lines 384-386)?

2. Proving the Discovery is Not a NAS Artifact: How can you prove that your discovered "Hyena→Transformer→CNN" pattern is a genuine property of the task and not an artifact of the one-shot, weight-sharing NAS? Would you be willing to provide results from training a selection of pure-block (e.g., all-CNN, all-Hyena) and mixed-block architectures completely from scratch (i.e., new random initializations, not from the supernet) to show the mixed-block models still win?

3. Please provide more details and intuition surrounding the "Agent" Mechanism in the main text. 

4. Clarifying Prediction Metrics: Can you please provide a precise definition for the "Hit Rate @k" and "Precision @k" metrics in Table 3? 

5.Can you please clarify the inconsistency between Line 216 (which specifies "Mamba") and Figure 2/Section 4.2 (which specifies "Hyena")? Why is Mamba not mentioned after figure 1?

6. Justifying the LLM's Bias: How do you address the fundamental concern that using an LLM (trained on existing ML literature) to predict "good" architectures will inherently bias your search towards existing design patterns, potentially defeating the purpose of discovering novel ones?

7. Biological Insight: The abstract claims current models "struggle to capture... complex underlying 'grammars' inherent to biological data." Could you discuss how this work addresses the grammar of biological data?

### Soundness
2

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
3

### Summary
This paper introduces **BIOARC**, a framework for **automated architecture discovery** in biological foundation models. Unlike prior work that repurposes architectures from general AI domains (like NLP and CV), BIOARC is designed specifically for the **unique physicochemical and structural characteristics** of biological data. Existing foundation models in biology often underperform because they reuse architectures meant for text or images without adapting to biological data’s structure. Furthermore, they fail to capture **long-range dependencies**, **sparse signals**, and **complex biological “grammars.”**

BIOARC addresses these limitations through a **systematic, data-driven approach** to model design. BIOARC leverages **Neural Architecture Search (NAS)** to explore a **large architecture design space** across various biological data modalities. Evaluate how **architecture**, **tokenization**, and **training strategies** interact and influence performance. BIOARC identifies **high-performing architectures** tailored to biological contexts.  

The framework also introduces **architecture prediction methods** that efficiently predict optimal architectures for new biological tasks. The key contributions of the work are: 
1. **Principled, automated framework** for designing biological foundation models.  
2. **Comprehensive analysis** of the relationship between architecture, tokenization, and training strategies in biological data.  
3. **Discovery of novel architectures** that outperform intuition-driven designs.  
4. **Set of empirical design principles** to guide future development of biology-specific models.  
5. **Architecture prediction techniques** enabling efficient adaptation to new biological tasks.

BIOARC represents a step towards creating **task-specific and general-purpose foundation models** for biology. By moving from intuition-driven to **principled architecture discovery**, it establishes a systematic methodology to advance biological AI research.

### Strengths
- The idea of applying techniques in Neural Architecture Search for discovery of patterns and design principles for biological foundation models is novel and interesting
- The paper is very detailed and considers different important aspects of biological foundation models such as tokenizations and architectural design aspects such as type of block used, the width of the network and the depth of the network. 
- The paper is very well written and clearly structured in most parts 
- The contribution of the paper is very significant and I believe that the insights derived are of general interest for ICLR research community. Furthermore, the architectures derived being much smaller and performant compared to state-of-the-art foundation models, makes these models more accessible and deployable

### Weaknesses
- The techniques of weight sharing, random path sampling followed by training of a predictor are quite well studied in the NAS literature and across applications [1,2,3], the application of the techniques for biological foundation models is however novel. 
- The experimental setup is not very clear in some parts:
    - Could the author's elaborate on the total compute budget for finetuning architectures? How faster is the convergence of a model initialised from supernet weights v/s a model trained from scratch?
    - Given the large size of the search space (different modules per-layer), could some paths through the network be undertrained?
    - Could the authors plot performance metrics for architectures randomly sampled from the supernet (before finetuning) v/s the parameter count in the architectures? 
    - Could the authors compare different search algorithms eg: random-search v/s evolutionary search v/s bayesian-optimization?

### Questions
Check weaknesses

### Soundness
3

### Presentation
3

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
This paper introduces BIOARC, a Neural Architecture Search (NAS) framework designed to discover optimal neural architectures for biological foundation models. The authors motivate their method by observing that current biological models directly repurpose NLP architectures like Transformers without accounting for the unique properties of biological sequences -- sparse motifs, long-range dependencies, and physicochemical constraints that differ from natural language. BIOARC addresses this by first defining a search space spanning five module types (CNN, LSTM, Transformer, Mamba, Hyena) with various depths and hidden dimensions, yielding 67 million candidate architectures. This space is aggressively pruned to 360 paths using heuristic rules. Then, a weight-sharing supernet is constructed from the remaining unique layers and trained via one-shot uniform path sampling on large-scale biological corpora (DNA and proteins). They consider two self-supervised objectives: masked modeling and contrastive learning.

Additionally, to predict architectures for new tasks, the paper proposes three methods: a Graph Neural Network predictor that encodes architectures as graphs and tasks via pretrained language model embeddings; a direct LLM prompting approach; and a multi-agent BIOARC Agent system that chains specialized LLMs that are assigned specific roles to reason over task similarity and performance data. However, details on this part are highly unclear and I am not sure what the point of this section.

On 12 DNA tasks and 6 protein tasks, BioArc shows good performance. The authors attempt to answer six core research questions about competitiveness, architectural patterns, scaling laws, training strategies, tokenization effects, and prediction methods. The key results are that BioArc-discovered architectures (3-5M parameters) achieve 8-15 point absolute gains over two larger baselines (DNABERT-2 and Neucliotide Transformer) across all GUE tasks, demonstrating that a well-chosen architecture can compensate for scale. The paper identifies a consistent pattern where top DNA architectures cascade through Hyena -> Transformer -> CNN modules, and reveals architecture-tokenization patterns. However, on protein benchmarks, these architectures are strong on sequence tasks but fail on structure prediction (15.88% vs. ESM-1b's 28.17%).

Overall, I found it quite difficult to parse the paper and results, as it is not well-presented and often incomplete in both the main text and the appendix.

### Strengths
In my reading, I found the presented results to be impressive. The authors also uncovered useful and interesting architectural insights. 

1. On the DNA-based GUE benchmark, BIOARC achieves **8-15 point absolute improvements** over established baselines (DNABERT-2, VQDNA, Nucleotide Transformer) across 12 diverse genomic tasks. These gains represent a clear improvement in performance on a standardized benchmark, achieved with models that are **much** smaller in parameter count and trained on substantially less pretraining data.

2. The paper identifies an actionable design principle -- optimal tokenization is architecture-dependent in biological sequences. Transformers perform best with 6-mer tokenization, while CNNs peak with 1-mer tokenization. Interestingly,  this interaction reverses under pretraining. This is empirically well-validated across multiple tasks and provides immediate practical guidance for practitioners building biological models.

3. The supernet approach (after intelligent pruning) is a smart idea borrowed from the original paper. It successfully identifies high-performing mixed architectures (e.g., Hyena -> Transformer -> CNN) that outperform single-module designs. This pattern can be easily adopted by others who are training their own biological sequence model.

4. Finally, it is nice to see that similar downstream tasks benefit from architectures that are similar to each other (Figure 7).

Overall, I see this paper as a useful investigation into the relationships between architectures and downstream tasks in biological sequence modeling.

### Weaknesses
1. The biggest weaknesses of this paper are the presentation, writing, and clarity. It was quite hard for me to parse the method in its entirety, as well as the presentation of the various training configurations of the BioArc models (only-ft, ...). In fact, the paper has several places where the writing is extremely concise, to the point of being a blurb (eg, A.8.6, L1187). A prime example of writing sloppiness can be seen in Appendix A.8.7 (L1271), where the section is woefully incomplete and stops mid-sentence, and A.5 L943, where it is unclear what the sentence says.

2. The BioArc methodology presentation keeps jumping around the **exact way** to rank the candidates. The initial pruning methodology is clear (I expect an explicit pruning algorithm, complete with hyperparameters, to be released in the codebase upon publication/deanonymization). However, following this initial pruning, how exactly the candidates are ranked is not clear to me. The paper implies that all 360 pruned architectures are evaluated via fine-tuning or training from scratch on each downstream task, but the precise algorithm for scoring, ranking, and selecting the "best" architectures is never explicitly detailed. The supernet is never shown to be used for ranking. This information seems to be missing entirely, and for a NAS method, this component is critical. The paper never defines the ranking objective: is it validation accuracy, test accuracy, or a Pareto frontier? Without this, the 'top-5 architectures' in Figure 2 and the selection for Figure 3 are **arbitrary and irreproducible**.

3. Following the above point, the architecture prediction section and contribution seems to be conecptually orthogonal to the rest of the methodology. It does not seem to be a core component, and is not tied into the methodology as well as the other components. Furthermore, the results in Table 3 are missing the evaluation context in A.8.7 entirely, and therefore, I cannot consider these results grounded. 

4. Unconvincing (and sometimes missing) diagnosis of results and their interpretation -- in Sec. 4.2, L318 is an entirely generic sentence, and L323 is not a (contextually) convincing reason for the underperformance of the BioArc models, when compared to ESM-1b.  Furthermore, ESM-2 is overlooked as a baseline, despite being released in 2023.

5. (Minor but still important) The figures presented in the paper are of low quality; for example, see Figs. 4,10. There are also some typos littered throughout the paper (L316, L295). 

Considering the unclear writing, confusing presentation, and underexplained sections, I recommend **rejection** at this stage of the submission. This initial decision is despite the paper's meaningful results. I suggest that the authors either **substantially** improve writing and presentation in their rebuttal (especially making the methodology clearer and presenting the missing parts) or consider resubmission after substantial rewriting, in which case the results and insights can really shine with a clearer presentation of the methodology.

### Questions
1. I find the choice of masked modeling and contrastive learning alone as training tasks interesting. Several biological sequence models (EVO 1/2 being a salient example) are now being trained using next-token-prediction objective. Is there a reason why this was not considered?

2. Why were the EVO models not discussed at all in the paper?

3. Why was the ESM-2 paper not considered as a baseline?

3. After the supernet is trained and yields 360 pretrained paths, are all of these architectures trained from scratch for the final evaluations? This is unclear, both in the main text and appendix. In case I missed this part, please point me to the relevant sections of the text.

4. Is it reasonable to assume that rankings from supernet paths are muddled with gradient sharing between connected other paths and training objective? Did you perform a rank correlation analysis between performance of the supernet paths and the same paths trained from scratch?

### Soundness
2

### Presentation
1

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces BIOARC, a novel framework designed to automate the discovery of optimal neural network architectures for biological foundation models (FMs). The authors argue that the common practice of directly repurposing architectures from general AI domains (e.g. transformers in NLP and Vision) is suboptimal for biological data, which has unique properties such as long-range dependencies and complex "grammars" dictated by physicochemical laws. BIOARC leverages Neural Architecture Search (NAS) to systematically explore a vast and diverse design space that includes modern modules like CNNs, Transformers, Mambas, Hyenas, and LSTMs across multiple biological data modalities (DNA and proteins). 

The paper's core contributions are:
* A principled methodology for discovering high-performing, task-specific architectures for biological data through NAS using modern modules. The search is made computationally feasible by leveraging a weight-sharing supernet.
* A systematic analysis of the complex interplay between neural architecture, tokenization strategy, and training strategy (e.g., pretraining vs. training from scratch)
* Empirical results demonstrating that the discovered architectures are often significantly smaller (e.g. ~25x) yet outperform larger, established foundation models on a variety of DNA and protein tasks.
* The development of a "BIOARC Agent," an LLM-based system to predict optimal architectures for new biological tasks.

### Strengths
* **Problem Significance**: The paper correctly identifies a key bottleneck in computational biology: the reliance on intuition-driven or repurposed architectures. The move towards a principled, automated discovery process is a strong and important research direction.
* **Comprehensive Analysis**: The study's greatest strength is its systematic evaluation. It does not just search for a single best model but rigorously analyzes the "interplay between architecture, tokenization, and training strategies". This yields durable insights, for example:
* * The optimal tokenizer is highly architecture-dependent (e.g., in Figure 5, CNNs perform best with 1-mer while Transformers prefer 6-mer).
* * Pretraining is not universally beneficial; training from scratch (only-ft) achieves the highest "win rate" for several tasks (Figure 4).
* **Strong Empirical Results**: The BIOARC-discovered architectures are highly competitive. They achieve state-of-the-art performance on all 12 DNA tasks while being significantly smaller (~4.5M params) than baselines like DNABERT-2 (117M params). This demonstrates the effectiveness of the search methodology.
* **Novel Architectural Insights**: The analysis reveals that the best-performing models are hybrid architectures, such as the Hyena -> Transformer -> CNN pattern identified for DNA tasks (Figure 2). This is a valuable finding, suggesting that combining modules with different inductive biases is crucial in the biological domain.

### Weaknesses
My main concerns are focused on the clarity of the supernet training methodology and the presentation of some results.
* **Major Question on Supernet Training Dynamics**: The paper adopts a "Single Path One-Shot" approach where a single path is sampled and updated in each step. This raises two critical questions about training stability and fairness:
* * Training Imbalance: In a vast search space, some shared blocks (e.g., a final-layer block) may be part of significantly more candidate paths than other blocks (e.g., a unique first-layer block). How do the authors ensure balanced and fair training, such that commonly shared blocks are not disproportionately over-trained compared to less common ones?
* * Incoherent Updates: A shared block's weights are updated based on an input representation from a stochastically sampled previous block. This means the block is not trained on a stable input distribution but on a "moving target" from a mix of different potential predecessors. How does the supernet converge reliably under this training scheme? A deeper discussion beyond "decouples the training"  is needed.
* **Ablation of Discovered Architectures (Fig. 2)**: The top DNA architectures shown in Figure 2 start with a Hyena block, followed by Transformer and CNN blocks. Given Hyena's strength in capturing long-range dependencies, how do we know the subsequent blocks are contributing significantly? Is it possible that the Hyena block is doing most of the work?
* **Impact of Network Depth (Fig. 2)**: The top 5 DNA architectures in Figure 2 all appear to have a depth of 6. Does this imply that the deepest allowed model was always optimal? Some discussion on the impact of depth on performance would be valuable.
* **Clarity of Research Questions**: RQ3 ("Does a scaling up of our optimal architecture yield a better foundation model outperform existing...") seems to overlap significantly with RQ1 ("How competitive are the BIOARC discovered architectures against established, large-scale foundation models?"). While the distinction appears to be about small vs. scaled-up discovered models, this could be clarified.

### Questions
Other than the questions mentioned above, I would also like to give some suggestions related to the clarity rather than the flaws in the overall manuscript.
* **Reporting Training cost**: The paper demonstrates a powerful methodology, but it would be helpful to quantify the computational cost. How long did the supernet pretraining take, and on what hardware? This context is important for assessing the framework's practicality.
* **Figure Clarity**:
* * Figure 4 (Right Panel): This stacked bar chart is very difficult to interpret. The bars are of similar lengths, and the segments are hard to compare visually. A simple table reporting the percentage "win rate" for each of the architectures for each task would be much clearer.
* * Figure 5: The x-axis is unlabeled. I infer it represents the Task Index (0-11) from Appendix A.1.1. Using a line plot is confusing, as it implies a sequential or time-series relationship between the tasks, which does not exist. A grouped bar chart or dot plot would be a more appropriate visualization.
* * Most figures: The legend and axis labels in many figures are too small and require significant zooming to be legible.
* **Typos/Phrasing**:  
* * Line 373: "To overcome this challenge, To overcome this," is repetitive.
* * Line 375: "...input dimension of and a 512..." – the "of" appears to be a typo.
* * Line 546: The grammar of RQ3 ("...a better foundation model outperform exsiting...") should be revised for clarity.
* * Line 948: "at at BioArc-9794" is a repetition.

### Soundness
3

### Presentation
4

### Contribution
3

# Unveiling Language Skills under Circuits

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 6, 5

## Abstract
Circuit decomposition and counterfactual-based pruning have become the cornerstone framework for mechanism interpretability. However, the unfaithfulness to the output due to cumulative bias in the pruning process hinders more complex and detailed mechanism exploration. To address this, we propose a novel circuit discovery framework that faithfully identifies circuit graphs. This framework contains three steps: firstly, the language model is decomposed into a fully linear graph consisting of disentangled ``memory circuits"; secondly, greedy search is adopted to prune while ensuring output faithfulness; finally, we adopt causal analysis on the pruned circuit graph to identify salient circuit graph, estimated by counterfactuals and interventions. Our framework facilitates the discovery of complete circuit graphs and dissection of more complex mechanisms. To demonstrate this, we explored three generic language skills (Previous Token Skill, Induction Skill and In-Context Learning Skill). Using the circuit graphs discovered through our framework, we identify the complete skill paths of these skills.
Our experiments on various datasets confirm the correspondence between our identified skill paths and language skills, and validate three longstanding hypotheses: 1) Language skills are identifiable through circuit dissection; 2) Simple language skills reside in shallow layers, whereas complex language skills are found in deeper layers; 3) Complex language skills are formed on top of simpler language skills.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper introduces a novel framework for analyzing language model circuits, focusing on three language skills: previous token prediction, induction, and in-context learning. 

The authors propose "Memory Circuits" as a new unit of analysis by rewriting of the Transformer layer and isolating the different computational path in a given layer. It enable the differentiation of the effects of the MLP alone, attention layer alone and MLP-attention layer interactions. 

Using a three-step approach combining circuit decomposition, greedy path pruning, and comparative analysis of the paths contained in circuits from inputs without the studied pattern and inputs with the studied pattern, they identify circuit paths associated with different language skills. 

Their analysis suggests that language skills are identifiable through circuit dissection, stratified across model layers, and build upon each other hierarchically.

### Strengths
- **Novel decomposition approach.** The introduction of Memory Units as a per-layer rewriting offers a new way to decompose neural networks, complementing existing feature-based analysis approaches. This could be particularly valuable for isolating the role of attention-MLP interactions within individual layers. The introduction of new unit of analysis is an exciting research direction, given that finding the right unit could unlock multiple analisys downstream.
- **Thoughtful experimental design for counterfactual inputs.** The use of background and self-text as counterfactual inputs demonstrates a sophisticated approach to isolating skill-specific effects from confounding factors like the effect of bigram statistics.
- **Hierarchical analysis of language skills.** The study provides evidence for the hierarchical nature of language model capabilities, showing how complex skills like in-context learning build upon simpler ones like previous token prediction.

### Weaknesses
- **Limited comparison with existing work.** The paper lacks discussion of how their approach relates to or improves upon existing circuit discovery techniques [1,2,3]. Given the number of existing techniques and the absence of ground-truth in the field of mechanistic interpretabilty, the authors should provide motivated arguments for where their technique adds value, or validate their technique of circuit identification on well-studied circuit. This is particularly important for their path decomposition approach, as similar ideas (including MLP analysis) have been explored in previous work [4,5]. 
- **Methodological limitations in circuit validation and identification:**
    - The greedy search algorithm relies solely on maintaining the top token prediction when pruning edges. This is a limited metric as it doesn't account for fine-grained differences in the output probability distribution and is highly non-linear due to the final softmax. As discussed in section 4 of the reference [7], the choice of metrics for causal intervention significantly impacts the identification of components contributing to model behavior.
    - The validation through ablation studies is insufficient. While ablating components can disable a given output, this approach may obscure the role of earlier components that the final components relied on to compute their values. More extensive validation techniques have been developed and discussed in the literature [6]. On top of computed validation metrics, one often rely on detailed investigation of the components, or on downstream applications (see [2] for instance) as additional ways to validate their findings. None of these ways is substantially present in this work.
    - The decomposition doesn't account for token position, which is a significant limitation given that most existing circuit analysis work have demonstrated widely different roles for components depending on their position of activation.
- **Technical imprecisions:**
    - The paper incorrectly identifies their model as GPT2-XL when describing an architecture with 12 layers and 12 attention heads, which actually corresponds to GPT2-small. See Table 2 of the original GPT-2 paper for reference [8].
    - There's improper use of tensor notation for the MLP layer in equation (1), attempting to encode non-linearity in a matrix through tensor products, which can only represent linear operations.
    - The paper claims that the greedy search finds the "smallest, independent, and functionally complete circuit graph," when greedy search cannot guarantee minimal solutions.
    - There's inconsistency between claims of a "lossless circuit framework" in the introduction and later acknowledgment of approximation errors. The paper only reports the minimum squared error between circuit sum and original output, leaving questions about average/median errors and the source of these approximations (mathematical reformulation vs. discrepancies in floating-point implementation).

[1] Goldowsky-Dill et al., "Localizing Model Behavior with Path Patching" 

[2] Marks et al., "Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models" 

[3] Conmy et al., "Towards Automated Circuit Discovery for Mechanistic Interpretability" 

[4] Hanna et al., "How does GPT-2 compute greater-than?: Interpreting mathematical abilities in a pre-trained language model" 

[5] Lieberum et al., "Does Circuit Analysis Interpretability Scale? Evidence from Multiple Choice Capabilities in Chinchilla" 

[6] Miller et al., "Transformer Circuit Faithfulness Metrics Are Not Robust"

[7] Heimersheim et al., "How to use and interpret activation patching"

[8] Radford et al., "Language Models are Unsupervised Multitask Learners"

### Questions
1. How does your Memory Circuit decomposition provide specific advantages over existing path decomposition techniques, particularly for analyzing MLP-attention interactions?
2. How would your method and other circuit discovery techniques compare (e.g., [1,2,3]) on a common task?
3. Have you considered extending your analysis to account for token position? 
4. Could you clarify the model (GPT-2 XL vs GPT-2 small) used in your experiments and update the name of the model or its architecture if a different model was used? 
5. What is the distribution (mean, median, max) of approximation errors in your circuit decomposition, and what are the primary sources of these errors? 
6. Could you provide a more mathematically precise formulation of the MLP decomposition that properly handles non-linearities?
7. For the in-context learning tasks analyzed, what is the base performance of GPT-2 on ground truth data?

### Soundness
2

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
2

### Summary
**Alert to AC: I am Not an expert in this domain (i.e. circuit analysis). Please find another reviewer who is more familiar with the domain.**

The paper targets the mechanistic interpretability of language models, specifically focusing on identifying and understanding the language skills that these models possess. It aims to dissect transformer models into circuit graphs to uncover the pathways that correspond to specific language skills, such as the Previous Token Skill, Induction Skill, and In-Context Learning (ICL) Skill.

The paper proposes a novel three-step framework to extract language skills from transformer language models.

From the experiments, the authors validate some interesting findings about language skills. For example, simple language skills reside in shallow layers, whereas complex language skills are found in deeper layers.

### Strengths
1. The concept “memory circuit” is interesting. The proposed method, constructing a graph based on the concept and then removing the redundant edges by greedy search, is quite reasonable and can be a good start to research in this field.

2. The findings are also interesting and useful. The authors conclude that there are identifiability, stratification, and inclusiveness in language skills, which could provide good insights for future works.

### Weaknesses
1. From my perspective (who is not so familiar with circuit analysis), the authors might need to clarify some parts in this paper including:

- The section 3.3 describes the key step to estimate the causal effects for language skills. However, the three concepts (including: skill effects, background effects, and self effects for destination) seem to have emerged out of thin air. I have checked the related works (Wang et al., 2023; Vig et al., 2020) that the authors cited in this paragraph, but do not find similar concepts. I tend to believe that the concepts are created by the authors. I wonder if the mentioned effects have any basis? And how can we ensure that there are no other effects?

2. More possible weaknesses are listed in the questions below.

[1] Kevin Ro Wang, Alexandre Variengien, Arthur Conmy, Buck Shlegeris, and Jacob Steinhardt. Interpretability in the wild: a circuit for indirect object identification in GPT-2 small.

[2] Jesse Vig, Sebastian Gehrmann, Yonatan Belinkov, Sharon Qian, Daniel Nevo, Yaron Singer, and Stuart Shieber. Investigating gender bias in language models using causal mediation analysis.

### Questions
1. In table 2, how many paths do you remove for each skill (i.e. the last 6 columns)?

2. The authors state that higher-level skills always entail the key circuits of lower-level skills. But how about the reverse situation? In table 2, the row “PVT” also gets very bad results for the graphs that remove paths of high-level skills like ICL. Is this reasonable?

3. Have you ever tried other models except GPT2-XL? GPT2-XL could be a little bit old at this time. I would like to see some analyses about the language skill differences among different models. For example, is there anything different for a smaller-size model like GPT2-small, and for a stronger capacity model like Llama-3?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This work proposes a lossless dissection of transformer models into circuit graphs, which is an ensemble of paths connecting different circuits. Then the authors introduce a three-step framework for extracting salient paths, i.e. skill paths, responsible for different language skills. Using this framework, the authors identified skill paths for three language skills of GPT-2-XL. The findings validate three hypotheses: 1) Language skills are identifiable through circuit dissection; 2) Simple language skills reside in shallow layers, whereas complex language skills are found in deeper layers; 3) Complex language skills are formed on top of simpler language skills.

### Strengths
This paper studied an important problem in mechanistic interpretability: isolating the effect of a single language skill from a text while taking into account the Feed-Forward layers in the transformer models. The lossless circuit decomposition is clear and easy to understand, although the compensation circuits require more discussion.

### Weaknesses
- How the two gaps mentioned at the beginning are addressed by this work is unclear, and the advantages of the proposed method against previous methods are not clear:
  - Previous works exclude Feed-Forward layers. The results in this paper do not clearly indicate the benefits of taking into account the FFL, i.e., "fully extract the complete trajectory of language skills." Adding FFL to the discussion does not result in major new insights.
  - Isolating single language skills: It is unclear whether the proposed analysis isolates single language skills better than previous methods. In fact, looks like lines 418 - 420 suggest that it is still hard to disentangle domain-specific knowledge when analyzing a specific task.
- The main findings are not completely new. There should be some discussion on how previous works validate the three hypotheses and how this work adds new insights.
- Writing/missing details:
  - Implementation details:
    - Section 3.3: See Question 4
    - Section 4:
      - What's the $\delta$ (line 296-298) for finding skill paths in the experiments? How is it chosen?
      - Line 323: "... high-effect samples through clustering"? What does this mean and how is the clustering done?
      - Line 315: How are the samples selected? Since G* has performance 1 in Table 2, does this mean you only choose samples where GPT-2 predicts the correct tokens as the top-1 tokens?
    - Section 5: How is the removal of random paths done? Why choose 50 and 500? "approximately equals the number of skill paths": how many paths are in G*? What are the number of skill paths for different dataset/language skills? Also, there needs to be more details explaining the "t-SNE representation of the top-5 candidate".
  - The results (the motivation, experiment design, and chosen methods) need to be more carefully discussed. For example, on lines 392 - 394, it is unclear what question Figure 6 is answering and why we need to visualize the bivariate probability density function. Also, many results in the main paper (like all results in section 5.2) are in the appendix, making the paper hard to follow.
  - Captions of figures and tables should be more detailed. For example, Figure 6 is barely understandable by itself.
  - The notations are not easy to understand due to the nested superscript and subscripts.

### Questions
1. How are compensation circuit expressions derived? Why do we need compensation circuits? Why after adding compensation circuits it is still nearly accurate rather than lossless?
2. Looks like compensation circuits by def are not memory circuits (they depend on memory circuits in the same layer), but they are contained in G as well as G* (lines 195-196). Does this mean all memory circuits will always affect the outputs through paths containing compensation circuits? And does this mean not all skill paths are formed by memory circuits?
   1. line 510 - 512: decomposing the LM into paths among Memory Circuits, what about paths that involve compensation circuits?
3. Algorithm 1: There is a specific order of pruning, how would G* change if we use a different pruning order? How consistent are the skill paths when we use different pruning orders?
4. Section 3.3 is interesting, but some details are missing. Specifically, it is unclear from Figure 1 how you perform greedy search for background text and self text. Does GPT have the same top-1 token for the bg text and self text? If not, how does greedy search work for bg and self texts? Also, how sensitive are the skill paths to the exact choices of alternative words for constructing background and self texts?
5. Line 403: "The paths of each skill are identifiable and remain unchanged across most data instances." How is this justified?
6. Table 3, looks like the receivers of 4 ICL tasks do not overlap that much what does this imply? How can we isolate ICL skill paths?
7. Line 199: C28 does not depend on input (Table 1), so why does it receive info from previous layers?
8. Line 93-96: it is not clear to me how language skills are identifiable (at circuit-level) and stratified supports the idea that CoT could reapply these skills to the input information through intermediate results. Can you elaborate on this?
9. Notation: Line 231, "E * -P" What does this mean?
10. Typo: Line 357 wrong table number.
11. Suggestion: consider adding h superscripts in equation (2) and Table 1 C28, the sum is over h but h is not in the expression.

### Soundness
2

### Presentation
2

### Contribution
2

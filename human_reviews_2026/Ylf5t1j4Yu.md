# Case-Based Reasoning Enhances the Predictive Power of LLMs in Drug-Drug Interaction

- Decision: Reject
- Scores: 4, 4, 6, 2

## Abstract
Drug–drug interaction (DDI) prediction is critical for treatment safety. While large language models (LLMs) show promise in pharmaceutical tasks, their effectiveness in DDI prediction remains challenging. Inspired by the well-established clinical practice where physicians routinely reference similar historical cases to guide their decisions through case-based reasoning (CBR), we propose CBR-DDI, a novel framework that distills pharmacological patterns from historical cases to improve LLM reasoning for DDI tasks. CBR-DDI constructs a knowledge repository by leveraging LLMs to extract pharmacological insights and graph neural networks (GNNs) to model drug associations. A hybrid retrieval mechanism and two-tier knowledge-enhanced prompting allow LLMs to effectively retrieve and reuse relevant cases. We further introduce a representative sampling strategy for dynamic case refinement. Extensive experiments demonstrate that CBR-DDI achieves state-of-the-art performance, with a significant 28.7% accuracy improvement over both popular LLMs and CBR baseline, while maintaining high interpretability and flexibility.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes CBR-DDI, a framework that distills pharmacological patterns from historical cases to improve LLM reasoning for DDI (Drug-Drug Interaction) tasks.

### Strengths
Applying LLMs for drug-drug interaction prediction is interesting

### Weaknesses
- Data Leakage Handling: How was data leakage addressed? While the evaluation protocol confirms evaluation on new drugs, there is no guarantee that the LLM has not been trained on these new drugs. The fact that the LLM uses the drug name instead of the SMILES (Simplified Molecular-Input Line-Entry System) for generating descriptions suggests a possibility that the LLM may already possess information about the drug based on its name.

- Knowledge Graph Coverage: Is it possible for test drugs to be absent from the knowledge graph? If so, how are these cases handled?

- Relationship between Retrieval Accuracy and Performance (Experiment 4.3.2): What is the relationship between retrieval accuracy and the actual performance in Experiment 4.3.2? Does accurate retrieval lead to an increase in actual performance?

### Questions
See Weaknesses Section

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
3

### Summary
This paper presents CBR-DDI, an LLM-based method for predicting drug-drug interactions (DDIs). CBR-DDI uses a combination of an LLM and a GNN to incorporate drug interaction data with historical context about its use. By doing this, CBR-DDI better reflects the clinical practice of using previous cases to inform current ones. Overall, the authors have completed lots of interesting work to showcase the approach, comparing it against a suite of baseline methods and providing a case study of its success. However, it was unclear why an LLM was used at the beginning of the pipeline (and whether this was an appropriate choice) and why the authors tested their approach on new drugs. Please see below for the clarifications I would suggest.

### Strengths
**Strong points:**
- The authors benchmarked the approach against a suite of many diverse approaches. I really applaud the authors for choosing benchmarks from both graph-based and LM-based approaches.
- Section 3.3.1: The approach here which balances semantic and structural similarity is very clever and sensible with respect to the task at hand.
- Well done to the authors for adjusting the metrics appropriately based on the nature of each dataset (DrugBank and TWOSIDES)- it shows that the authors really familiarized their selves with the datasets.
- For the most part, the claims made follow from the results provided.
- DDI prediction is much less researched than other drug discovery tasks, so it is an important area of research.
- Really nice figures.

### Weaknesses
**Weak points**

- Sections 3.2 and 3.3.2: Why use an LLM to generate drug descriptions or mechanism insights? Such descriptions are available on professionally curated databases (e.g., https://go.drugbank.com/drugs/DB00945 or https://pubchem.ncbi.nlm.nih.gov/compound/Aspirin). LLMs are infamous for hallucinations and mistakes in scientific disciplines. Thus, I would argue that this step, alone, needs to be validated before proceeding with the whole pipeline. For example, the authors ask the LLM to generate mechanistic insights, but I would suggest that you need to, somehow, verify that it can reliably generate those. Perhaps https://github.com/SuLab/DrugMechDB could help.
- It is not entirely clear why the authors tested the approach on drug pairs between new and existing or both new drugs. In the beginning, the authors even say: “These challenges become even more pronounced when predicting interactions involving new drugs, where interaction data is typically sparse or nonexistent.” If historical context is unavailable for new drugs, then how is this a useful approach for such drug pairs?
- A couple sentences in the introduction are difficult to understand:
    - “However, these methods provide only triplets and are insufficient to activate the reasoning capabilities of LLMs, as surface-level drug associations alone cannot reveal their potential interactions evidently” -->
        - The terms “surface-level” and “potential interactions” are quite vague so it’s not clear what’s meant here.
        - It’s also not clear what it means to “activate the reasoning capabilities of LLMs”.
        - Why are triplets not sufficient? Is there a paper that demonstrates this insufficiency?
    - “For example, in Figure 1, the new drug pair Fosphenytoin-Diphenhydramine binds to the same gene, yet their actual interaction cannot be directly inferred.”
        - The way this sentence is structured is confusing. The subject of the sentence is singular, but then "their" is used. Also, the way it's worded implies that a "drug pair" is one entity which binds to a gene.
        - This sentence also seems to imply that interactions between drugs are literal, physical interactions between drugs as opposed to interferences or cumulative effects on biological systems.

### Questions
I would ask the authors to provide some clarification or validation with respect to the above weak points.

1. Can the authors please justify why the use of an LLM to generate drug functional descriptions and interaction mechanisms, as opposed to using curated resources available from databases?
2. Can the authors either validate, experimentally, that the output of the first LLM step is reliable or correct, or justify why such validation is not necessary?
3. Please elaborate on why testing was done on new drugs and how this fits into the use of historical context and CBR.
4. Could the authors please clarify the above sentences in the Weaknesses section?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
1.This paper proposes CBR-DDI, a framework for DDI reasoning that enhances traditional LLM-based DDI inference by incorporating CBR (Case-Based Reasoning), i.e., historical cases of drug interactions. Subsequent innovations in retrieval mechanisms and knowledge base construction are also designed to support this approach.
 
 2.For the construction of the DDI knowledge base, unlike traditional methods that rely on knowledge graphs (entity, relation, entity), this paper not only integrates information from authoritative datasets but also incorporates "mechanism insights" and "drug descriptions," which are inferred by the LLM based on knowledge graph information.

  3.Hybrid retrieval mechanism: When performing DDI reasoning, the paper not only uses the structural similarity of drugs to retrieve interaction information from similar drugs as references but also leverages the semantic similarity of drug descriptions. These two similarity scores are weighted and combined to form a final similarity score, which is then used to select reference drugs.

  4.Dual-layer knowledge enhancement: The two layers refer to knowledge from authoritative datasets (internal knowledge) and "mechanism insights" and "drug descriptions" generated by the LLM (internal knowledge). These two types of knowledge are jointly used to prompt the LLM for reasoning.Experimental results demonstrate a 28.7% improvement in accuracy compared to LLM and CBR baselines.

### Strengths
1.	The paper is well-structured and easy to follow, allowing readers to quickly grasp the content even without prior specialized knowledge.

2.	The methodology is rigorous, incorporating research and refinements across multiple aspects including knowledge base construction, data retrieval, and knowledge enhancement.

3.	The experiments are comprehensive, featuring not only comparisons with baseline models but also ablative studies evaluating the proposed method itself.

### Weaknesses
1.	The paper does not consider the structural information of the drugs themselves, such as the graph structures of molecular or protein-based drugs, relying instead on textual information and interaction relationships.
2.	The study does not incorporate expert evaluation to validate the practical effectiveness of the proposed method.
3.	Both drug descriptions and mechanism insights rely on large language models (LLMs). During inference, the LLM needs to generate three components: drug descriptions, mechanism insights, and reasoning results, resulting in poor inference efficiency. Moreover, could errors in the three outputs generated by the LLM accumulate, leading to progressively amplified errors during the reasoning process?

### Questions
N/A

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper tackles the problem of drug-drug interaction prediction. Based on a given biological knowledge graph, the method first builds a knowledge repository of previously known drug-drug interaction cases (this process is made efficient with representative sampling). At inference time, the system retrieves relevant historical cases using semantic and structural similarity. The retrieved cases are then incorporated into a prompt to a LLM that performs the final prediction. The authors evaluate their method on the DrugBank and TWOSIDES datasets and show that it improves over both graph based methods and naive LLM methods by a sizeable margin.

### Strengths
- This work proposes a very effective solution for an important and impactful problem (drug drug interaction prediction)
- The paper is clearly written and easy to follow
- The authors run extensive ablations to study the impact of each of the components.

### Weaknesses
- The major weakness of this work is its relevance for a machine learning conference like ICLR. The approach is elegant and outperforms baselines on that particular problem. However, there is no substantial machine learning contributions. That said the paper is really creative way to solve the drug-drug interaction problem and might better shine in a venue more focused on that particular problem.

- The architecture of CBR-DDI is highly tailored to the drug drug interaction problem and it's not clear whether it could be applied easily in other contexts.

### Questions
- Do the authors think that the approach presented here could benefit other application areas ?

### Soundness
3

### Presentation
3

### Contribution
1

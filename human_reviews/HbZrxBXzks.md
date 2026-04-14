# Geometry Informed Tokenization of Molecules for Language Model Generation

- Decision: Reject
- Scores: 8, 5, 5, 6, 6, 6, 10, 3

## Abstract
We consider molecule generation in 3D space using language models (LMs), which requires discrete tokenization of 3D molecular geometries. Although tokenization of molecular graphs exists, that for 3D geometries is largely unexplored. Here, we attempt to bridge this gap by proposing the Geo2Seq, which converts molecular geometries into SE(3)-invariant 1D discrete sequences. Geo2Seq consists of canonical labeling and invariant spherical representation steps, which together maintain geometric and atomic fidelity in a format conducive to LMs. Our experiments show that, when coupled with Geo2Seq, various LMs excel in molecular geometry generation, especially in controlled generation tasks.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces Geo2Seq, a language model-based method for generating molecules. The method converts a 3D molecular graph into a 1D sequential representation through canonical ordering. It employs a spherical representation to encapsulate the 3D structural information, preserving SE(3)-invariance while ensuring no information loss. Additionally, the paper presents a specialized 3D tokenization approach tailored for generating 3D molecules.

### Strengths
1. Understanding how to use a sequential representation for modeling a 3D molecule is critical to language-based molecule generation.
2. The paper theoretically demonstrates that the application of canonical and spherical representations preserves SE(3)-invariance without any loss of information.
3. The paper is well-written and easy to understand.

### Weaknesses
1. My primary concern is that since the use of canonical ordering and spherical representation is not novel in the context of 3D molecular tasks, it would be beneficial to thoroughly discuss and compare their applications as documented in existing literature. This analysis should be included in the preliminary and related work section of the paper. This comparison will help in highlighting the unique contributions and potential advancements introduced by the Geo2Seq method.
2. Given that different tokenization approaches may significantly affect the generation quality, it would be beneficial to have additional ablation studies compare the proposed tokenization method with some traditional word-based or subword tokenization methods.

### Questions
Please refer to Weaknesses section.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
This paper proposes Geo2Seq, which extracts invariant coordinates of molecules and supports language model-based 3D molecular structure modeling through digit-level encoding. The method demonstrates strong performance across datasets.

### Strengths
The benchmarks show strong performance, especially in controllable generation.

### Weaknesses
1. This is a technically solid paper but with relatively limited contributions.
2. This paper does not require extensive theoretical exposition. Both isomorphism and invariance are well-studied concepts with no new conclusions presented here, and the heavy use of theoretical formulations affects readability.
3. From the existing experimental analysis, it remains unclear whether the model is merely replicating token patterns or genuinely learning underlying molecular principles.

### Questions
1. Although it may compromise invariance, would training directly on centered Cartesian coordinates improve performance on this task?
2. How does Geo2Seq ensure that generated molecules adhere to chemical and physical constraints (e.g., bond lengths and steric hindrance)? Are there methods to introduce explicit constraints?
3. Would increasing coordinate precision (i.e., more decimal places) impact the quality or accuracy of the model's generated outputs?
4. How does the GPT model perform in controllable generation? The model structure does not appear to be significantly different.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper proposes a novel tokenization method, Geo2Seq, for generating 3D molecular structures using language models (LMs). Geo2Seq transforms molecular geometries into SE(3)-invariant 1D discrete sequences, making them compatible with LMs while preserving essential structural details. The method uses canonical labeling to establish a unique node order and spherical coordinates for SE(3)-invariance, enabling robust encoding of molecular geometry. This technique allows LMs to handle 3D molecule generation more effectively, surpassing existing models in generating diverse, chemically valid 3D structures.

### Strengths
1. This is a very interesting task, and it’s great to see a method for integrating multimodal data, especially 3D molecular structures.
2. The motivation is clear and straightforward.
3. The experiments in this paper are comprehensive, covering various LMs and ablations.
4. The writing is clear and accessible.

### Weaknesses
I believe there is still room for improvement in the paper:

1. The canonical labeling method discussed in Section 3.1 resembles the SMILES canonicalization commonly used in cheminformatics. Intuitively, this functionality could be implemented using SMILES canonicalization in RDKit—why not utilize that?
2. The molecular bonds are generated using RDKit, also there is no strict definition of bonds in quantum chemistry. What is the necessity of introducing graph isomorphism in this paper?
3. The authors suggest that limiting the range could enhance the language model’s performance. However, in spherical coordinates, only the azimuth and polar angles are restricted in range, while the distance remains a continuous variable. This could still limit applications for larger molecules, such as proteins.

### Questions
1. Would using SMILES canonicalization be a better approach? How does it differ from the method you've employed?
2. Since the LLM is already generating iteratively, I believe adopting a local coordinate system could be more effective, as it would help keep distances manageable.
3. I'm curious whether normalizing the xyz coordinates (for example, by subtracting the center of mass coordinates) could achieve a similar effect. According to the authors, normalization could also assist in reducing the range to some extent.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes a way to tokenize 3D molecules into sequences for language model generation which preserve structural completeness and SE(3)-invariance. They include detailed derivations and proofs for the claims. The method demonstrates promising results for random generation and controllable generation. They also include comprehensive ablations showing the importance of the canonicalization and 3D representation.

### Strengths
- The method and derivations are sound and rigorous
- The ablations are comprehensive and justify the method's design choices
- The random generation results are promising
- The presentation is beautiful

### Weaknesses
- There should be more LM-based baselines for random generation
- There should be more baselines for controllable generation

### Questions
For this work, the LM is used as a generative model for learning patterns of the proposed representation from data. There can be more case studies and discussion on how additional capabilities (e.g. in-context learning) of LLMs can be leveraged with this representation. Do the authors see any emergent abilities? Are prompt-based approaches possible?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper explores language model (LM) based  generation of molecules in 3D space. To that end the paper proposes tokenization of molecular 3D graphs that converts molecular geometries into SE(3)-invariant 1D discrete sequences to be used as input to a LM. The paper proposes to a) first use graph canonicalization to create a 1D sequential ordering of the 3D molecule. b) use spherical representation to represent the 3D coordinate of the atoms. This ensures that any two isomorphic graphs have the same representation and different graphs have different representations. The LM model is next trained using typical next token prediction on the created token.  The experimental results on QM9 and GEOM-DRUGS dataset shows that their proposed Geo2Seq performs better than baselines both for random and controlled generations.

### Strengths
-- The problem description of converting 3D graphs to 1D tokens preserving SE(3)-invariance is very interesting and relevant to the community. 

-- Moreover their proposed method Geo2Seq, which combines the two ideas from Graph theory (canonicalization and spherical representation) is novel (even though parts of it might be used in previous applications, for example canonicalization is used in SMILES).

-- Experimental studies on random and controlled generations across two dataset showcases the efficacy of the proposed method

-- Detailed Ablation Studies (sec C). One of the immediate questions I had was understanding the importance of using different types of ordering (one of which could be canonical) and different types of representation (including the proposed spherical). These questions have been asked and answered in the ablation (shown in the appendix).

### Weaknesses
-- As stated in the strength section, even though the combination is novel, the individual components have been used before.

-- One thing that is not clear to me from the very thorough experiment is from the following observation: From table-4 the gap between the top 3-4 methods is not very large. Under that circumstance, should efficiency and memory required to perform the ordering be used to pick a winner? Did the author do such an analysis?

### Questions
Other than the points mentioned in the weakness section, I have the following questions:

-- The proposed combination of canonical representation and spherical representation is nothing very specific to 3D molecules but can also be used by any 3D point cloud, did the authors try to see whether their proposed method works for other available 3D point cloud generation tasks (see pointflow or related papers for the tasks)?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 6

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper provides a novel approach to tokenization of molecules for the purposes of conditional and unconditional generation, focusing not on the model, but on ways of representing molecules as symbols for any model requiring a symbolic approach in a manner that is SE-3 invariant and efficient.

### Strengths
This work enables bridging geometric approaches to molecule representation with sequence-based molecule representation in a highly original and apparently successful manner.  The authors correctly identify the reasons why it can be difficult to translate inherently graphical information into a sequence representation, and then present both rigorous theoretical and extensive experimental evidence to show that they have arrived at a better solution than those existing to this problem.  I particularly appreciate the ablations and would suggest putting them earlier in the paper, as they convincingly show that these differences actually matter in real LLM's (not a trivial claim, since often apparently problematic objects to serialize/tokenize can be tokenized effectively given large enough models, e.g., in computer vision).  

This paper is also particularly compelling as a work which includes extensive mathematical foundations, a very significant use case for the mathematical framework built up, and evidence that the mathematical framework is effective at solving that real-world problem.  I am not fully qualified to comment on the correctness of the proofs; however, to the best of my knowledge they seemed both valid and novel.  

I appreciated the attention brought not just to the number of valid molecules, but also to the number of valid and unique and novel molecules, as arguably the most common "hidden failure mode" of transformers is merely regurgitating their training data.  Overall, the experiment section was clear and convincing.

### Weaknesses
I felt the controllable generation experiment was under-explained.  While they do list the six quantum properties in line 491 and discuss per-quantum-property efficacy in line 437, I would have preferred an analysis for each of these cases of instances where it succeeded and other methods failed or where their method failed and other methods succeeded.  More importantly, since the paper itself was on the mathematical foundation of a 1-d tokenization of 3-d molecules, I would have liked some explanation for each property of how the property relates to the 3-d configuration and what information relevant to that configuration is preserved or removed under the new framework, especially for readers with general knowledge of LLM's but without knowledge of chemistry.

Creating novel molecules is an interesting task; however, its usefulness was not made obvious in the paper (when would an applied researcher benefit from being able to generate a random valid molecule, or a molecule with a particular LUMO energy)?  The benefits of models such as AlphaFold were well documented in terms of medicinal chemistry; I was not able to find a similar reference explaining the benefits of random molecule generation, While it may be self-evident to researchers in the field of computational chemistry, to general readers of ICLR it would help to provide such a use case.  More ambitiously, it would be fascinating to see whether this approach could be applied to other geometric tokenization problems, such as generating valid representations of social networks with 2-d or 3-d constraints, or generating valid configurations of cellular automata.

Most LLM papers benefit strongly from looking at performance statistics as a function of model size and other hyperparameters; which I did not see here (there is a section on hyperparameters in the appendix, but not a comprehensive study involving the type of LLM used).

### Questions
1.  What do you think the use cases are for your work?
2. I was somewhat confused by line 258 - why would the coordinates being bound in a smaller region be helpful?  A simple clarification would be satisfactory
3. Could you be more specific about, of the proofs in the appendix, which are taken from existing literature, and which are novel to this paper?
4. When it says in line 1735 "π-out-of-phase angles are placed near each other, such as ‘3.14°’, ‘-3.14°’, and ‘0°’" - isn't this a sign that you're not capturing some aspect of invariance?

### Soundness
4

### Presentation
3

### Contribution
2

---

## Human Reviewer 7

### Rating
10

### Rating Number
10

### Confidence
3

### Summary
This paper tackles the problem of 3D molecular generation from scratch. The authors propose the tokenization method Geo2Seq, that converts 3D molecular graphs into SE(3)-invariant sequences for further uses in LMs. 
It is shown that by using this novel tokenization method, various LMs can generate 3D molecular structures with high validity, outperforming diffusion-based baselines.

### Strengths
- Proposed tokenization method Geo2Seq is agnostic to the used LMs, which provides flexibility in method usage [SIgnificance].
- [Clarity] The paper outline and structure are well defined and easy to follow.
- [Quality] Provision of additional random generation results for QM9 dataset, as well as additional experiments on more baselines and metrics perfectly visualize the novel method performance compared with existing methodologies.
- [Originality] Usage of spherical representations for 3D structures without information loss. 

- This work is provided with a substantial number of definitions and lemmas that help navigate the problem.

### Weaknesses
I believe this article has everything to be called a good paper without any major concerns.

### Questions
I could find the answers to most of my questions in paper. Thank you for providing such a detailed report of the work.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 8

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper proposes Geo2Seq tokenization scheme that is meant to enable generation of 3D molecules using autoregressive LLMs.
It canonicalizes the graph and creates an input dependent frame of reference with spherical coordinates to encode positions as strings to be provided for the autoregressive model.

### Strengths
The work provides a way to generate 3D conformers using autoregressive generation. 
The bijective mapping between the molecular graph and the sequences produced by Geo2Seq is demonstrated.

### Weaknesses
A few minor stylistic points:
- Line 111: what is 'molecular science' ? 
- Line 80, 401: perhaps "conditional generation" would be a more precise term than "Controllable generation".
- Line 107: "LLMs have revolutionized the landscape of NLP and beyond." - what is the "NLP and beyond" field?

The proposed tokenization scheme relies on the model to be able to 'understand' the 'number tokens'. The appendix shows UMAP plots demonstrating that the models learn some spherical-like structure from those tokens. Could adding the SE(3) invariant position embeddings be a more efficient way to achieve this?
The vocabulary size is reported to be '1.8K for QM9 dataset' and '16K for Geom-Drug' for problems with single digit number of possible atoms. Could this be a sign of overfitting? 
The autoregressive models used are also much larger than the GNN baselines (~90M parameters vs ~5M), could the performance gains be a result of this?

### Questions
How would the method perform if the number of tokens in the tokenizer would be restricted to 50, 100, 500, 1000 ?
Why does mamba seems to perform better than gpt in your evaluation of random generation (table 1)? Why is gpt not featured in table 2?
Would adding the position information on embedding level make more sense?
How is the Geo2Seq canonicalization different from canonical SMILES?
Why would the cartesian coordinate tokens with the same frame as Geo2Seq not be rotation invariant?

### Soundness
2

### Presentation
2

### Contribution
2

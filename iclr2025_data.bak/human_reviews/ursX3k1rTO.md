## Human Reviewer 1

### Summary
The paper proposes the **Wyckoff Transformer**, a generative model designed for creating highly symmetric crystal structures by leveraging **space group symmetry**. Recognizing that most inorganic materials exhibit inherent symmetries, the authors develop a model that encodes these symmetries to influence key material properties such as stability, conductivity, and optical behavior. The Wyckoff Transformer uses **Wyckoff positions** as a discrete, permutation-invariant representation of atomic locations, eliminating the need for explicit positional encoding and improving the model’s efficiency and alignment with crystal symmetries.

Key contributions of the paper include:

1. **Tokenization of Crystals**: The authors represent crystals as an unordered set of tokens, merging information on chemical elements and their Wyckoff positions, enabling symmetry-based generation.
2. **Permutation-Invariant Encoding**: The model encodes Wyckoff positions based on symmetry-defined point groups, allowing for the generation of stable structures without positional encoding.
3. **Transformer Architecture**: The Wyckoff Transformer combines **autoregressive probability factorization** with permutation invariance, enhancing diversity and stability in generated structures.
4. **Empirical Outperformance**: The model outperforms existing methods, generating novel, stable structures that adhere to space group symmetry.
5. **Predictive Accuracy**: Despite limited input information, the model accurately predicts formation energy and band gap values comparable to DFT (Density Functional Theory) standards.

The model demonstrates superior performance in symmetry-conditioned generation, creating a diverse set of stable crystal structures that respect the underlying physical symmetries. This approach addresses limitations in prior methods, which struggled to produce symmetry-compliant structures, and shows promise for accelerating material discovery in fields requiring stable, symmetric crystals. However, the model inherits typical dataset limitations in generative models, as it learns distributions only within the scope of the training data, which may omit some stable but out-of-domain structures.

### Strengths
Strengths

1. Originality: The Wyckoff Transformer introduces a novel approach to crystal generation by utilizing Wyckoff positions to encode symmetries explicitly, making it unique among generative models. Unlike traditional methods, it avoids positional encoding and uses permutation-invariant tokenization tailored to space group symmetries, a creative and effective innovation for materials science.

2. Quality: The paper includes experimental results (although the results are not yet complete), showing the model’s success in generating symmetric crystal structures while achieving competitive accuracy in formation energy and band gap predictions. The evaluation is thorough, comparing the model’s performance to state-of-the-art methods across multiple metrics, demonstrating its robustness and effectiveness in real-world scenarios.

3. Significance: This work is impactful for both machine learning and materials science in general. By generating materials that are symmetric and physically plausible, the Wyckoff Transformer can accelerate material discovery for applications requiring stable, symmetric structures (e.g., in semiconductors and optoelectronics). The model’s potential for symmetry-conditioned generation highlights a promising direction for future research in material informatics and generative modeling.

### Weaknesses
I observed the following weakness in the paper and my recommendations towards constructively improving this paper:

1. The write-up does not qualify for the levels of ICLR acceptance, it has too many inconsistent notation, typing errors, vague statements and hypothesises without proofs. Following is a non-exhaustive list: Line 174 WP 4a becomes (m-3m, 0), what does this line mean? later in Figure 4 the authors mention m-3m without any tuple, and later again in Figure 5 where the authors mention similar symbols as 1a = [m-3m] just to show and example of inconsistency in this paper, in Line 51 , ``atom position'' use correct quatotation marks, Figure 1 is vague no details on how it is constructed or taken from other source, line 227 coset (closest) representative (if you wish to mention grouping theory coset, then kindly use proper mathematical notations), Section 3.1 line 269, 270, 308, 315, 327, 329 no enumeration. Kindly go through the paper again, and correct all of them.

2. The paper revolves around Wyckoff Transformer, but no explanation of the model architecture either in writting or in schematic has been given in the paper, section 2.2 mentions the title Model architecture but did not mention anything other than the Transformer paper by Vaswani et el which was used for Neural machine translation, moreover the authors are wrong in stating it to be an encoder only architecture. The authors seem to be lacking the knowledge about Encoder only, Decoder only and Encoder-Decoder models. This confusion seem to prpogate through the training section (2.3) where they mention (a) De novo generation , (b) Property prediction , where they are unable to state the difference between these two tasks and how a single kind of model cannot handle these two separate taks without any further layers (some hints of this is mentioned on Line 222, but still vague while following the whole paragraph.). Please attend to all these points, this would make the paper more readable and understandable.

3. The paper does not include any structural schematic of the generated structure, if any. The paper does inlcude a few figures on Wyckoff positions and Wyckoff representation of a toy 2D Crystal and SrTiO3 in Figure 2 and 5 respectively. But none for their actually generated structures (if generated any). Claims on generative capability stand weak when figures are not included (kindly refer to the papers which they've cited, they have shown a wide set of generated structures with their corrosponding inductive biases.) Since the authors mention about their novel representation, they should support the validity of those representations and inductive biases (representational consistency) in their outputs. If possible I would like this point to be addressed in the revision.

4. The section covering related work is weak. The authors must do a good survey of the past work in cystal generations particularly in the field of crystal genration in the representaion space. Some of which I was able to find by searching for representation based genrative model in citations to CDVAE (Xie et. al.) paper are: 1. https://arxiv.org/abs/2306.04510, 2. https://arxiv.org/abs/2403.10846,  3. https://arxiv.org/abs/2408.07213 (kindly read and search for more). I request the authors to kindly include papers which are in the same field to address the concerns in this paper and how your research aligns or complements with these papers, so that this work becomes complete. Kindly include a paragraph discussing how their approach compares to or builds upon these specific papers mentioned in the comments.

5. Lack of clarity: Reading through the paper multiple times, I have found out that the paper is written very poorly, and fails to convey the message of the authors. Upon my earlier assesment I had mention this as a strength (note: I had confidence = 3 and later 4), but going into the review process and reading the paper multiple times I am confident that the paper lacks clarity (updated confidence = 5).

### Questions
The following are my questions and suggestions for this paper:

1. The structure of the network, inputs, outputs and loss function (including tokeization, loss function computation needs to be defined properly with proper mathamtical notation and schematics). Kindly include a schematic diagram of their Wyckoff Transformer architecture, clearly showing how it differs from standard Transformer models. Additionally, kindly clarify how their model handles both de novo generation and property prediction tasks, possibly by explaining any additional layers or modifications to the base architecture.

2. How did the authors plot Figure 1? Kindly include other generated structures in Figure 2, it will be best to show how the generated structures also follow these symmetris and where do ther lie in terms of space group number. Kindly provide a clear caption explaining the source and construction of the figure. Kindly include a figure showing examples of structures generated by their model, possibly comparing them to real structures or those generated by baseline models. This would help demonstrate the model's capabilities and the effectiveness of their novel representation.

3. In section 2.2 clearly mention the assumtpions for your model, as of now the reviewer was not able to find the assumptions which the authors have taken. These need to be mentioned in a list.

4. What was the training objective, were they two different models for task of De novo genration and Property prediction? If so, then how were they both trained? (Objective funvtion, optimiser hyper-parameters, Input data, valdiation metrics, regularisers, hardware specifications etc.)

5. The algorithm explained for (i) Tokenization, (ii) De-nove generation, (iii) Structure genration and (iv) Metric computation is vaguely defined. These need to be defined propely in algorithm sections. As of now they don't make any sense and are without mathematical notation.

6. Which DFT calculation has been used in the paper? Did the authors perform DFT of their own or are using previously reported results, if yes, then it will still need to be described and also how are they reporting.

### Soundness
1

### Presentation
1

### Contribution
2

### Rating
3

### Confidence
5

---

## Human Reviewer 2

### Summary
The paper presents a transformer-based architecture to generate symmetric crystals conditioned on space groups in a two-stage process. Initially, it generates tokens representing elements and their site symmetries, followed by lattice and coordinate predictions for these tokens with existing methods. The results include comparisons with multiple baselines, showing competitive performance across established proxy metrics. Finally, the paper also proposes metrics to assess the symmetry of the generated crystals and highlights further gains over baseline approaches.

### Strengths
- The paper emphasizes the importance of generating symmetric crystals and highlights challenges with existing methods.
- The evaluation methods, including the newly proposed methods for evaluating symmetry, form a compelling discussion section.
- The method demonstrates effective gains for the symmetry metrics and is competitive for widely-used proxy metrics.
- The paper proposes a novel representation of crystal symmetry that could facilitate learning of crystal symmetry with deep learning approaches.

### Weaknesses
- **Presentation and writing**: Essential concepts (site symmetry, wyckoff positions, space groups) are not appropriately introduced, which would create difficulty for readers unfamiliar with the field. Several works are cited in the related works section, but neither described nor highlighted the difference from their approach. Figures for architecture and pseudo-code to describe the training and generation pipeline would greatly benefit the understanding of the work.

- **Generalization of the approach**: 
  - The paper heavily focuses on the MP-20 dataset and does not provide any experiments with other datasets. For instance, it states permutation invariance was achieved because the number of WPs in the MP-20 dataset is small. How can this method be extended (or how does it fare in terms of performance) for crystals with a very high number of WPs?
   - There are no precise details on how many tokens were formed from the MP-20 dataset after tokenization. It would be interesting to discuss this number and other statistics about the tokens, e.g., which tokens are present more often (for some of the high symmetry space groups) and how the distribution of tokens affects training. 
   - It is also important to add how many new tokens the method generates or if it just predicts the fixed set of tokens in different combinations (and these combinations result in more template novelty than just sampling existing templates from training data). For instance, naively thinking about it, how will your model generate tokens that are not present in its dictionary? 
  - Finally, in Table 1a, please also provide the number of novel templates as absolute numbers instead of percentages.

- **Architecture**:
  - Please provide at least a pseudo-code of the training/generation algorithm and a figure explaining the training/generation process with a sample crystal example. The central algorithm is not clear from the text. 
  - Please mention the size of the model and computational and memory consumption (training time, memory required during training and generation). The paper lists that it is trained for $150k$ epochs, which seems to be a very long training process compared to existing methods (~$1k$ epochs). Can you explain this behaviour along with the set of hyperparameters used?

- **Evaluation**:
  - Can WyCryst be considered a fair baseline for comparison since it supports a limited number of unique elements per structure? For instance, it wouldn't compare with other methods that generate an arbitrary number of methods because it would result in poorer metrics (as seen in Table 1).
  - Is CHGNet used both to relax the generated structures and determine the energy in Table 1? 
  - Please mention the percentage of novel but structurally invalid generations from your method.

### Questions
- **Two-stage approach**: Can you explain the benefits of a two-stage approach instead of a one-shot (such as DiffCSP) prediction of the site symmetries, elements, their positions and lattice parameters? If problems exist with a one-shot prediction, please explain and motivate the need for a two-step approach. For instance, can we (as an example) predict all the tokens (with elements, site symmetry, enumeration) followed by the lattice parameters and the fractional coordinates of the tokens or are there inherent issues with this approach? This question becomes more necessary since the generation of crystals would be slow for the proposed "sequential two-stage" approach. 
- **Crystal Structure Prediction (CSP)**: The paper focuses on generating crystals conditioned on space group. How could this method be extended to the CSP task, which is also crucial and could potentially benefit from using crystal symmetry?
- **Dataset fragmentation**: Although the tokens can be shared across different space groups, there will still be dataset fragmentation when the approach is conditioned on the space group. Is the training (and then generation) not affected by how many samples are present within each space group?

Some of the other questions are listed in the Weaknesses section. I will be happy to improve the score if the authors address the questions and weaknesses with supportive evidence during the discussion phase.

### Soundness
2

### Presentation
1

### Contribution
3

### Rating
6

### Confidence
4

---

## Human Reviewer 3

### Summary
This paper highlights the problem of generative models for crystals not generating symmetric crystals, which is an important property of these materials. This results in less realistic materials as well as inability to model some properties of crystals correctly. The authors propose to address this limitation by generating materials in a two-stage process. First, they train a Transformer model to output occupied Wyckoff positions in the crystal. Then either a method based on DiffSCP++ or PyXtal is used for atomic coordinates. The authors verify experimentally that this allows to generate more symmetric and diverse crystals.

### Strengths
- The work tackles an important limitation of generative models for crystals
- The proposed solution is simple and sound
- The experimental evaluation shows that the method addressed the limitation. The evaluation metrics for symmetry and novelty of structures based on Wyckoff templates is also valuable.

### Weaknesses
- Just using the Wyckoff positions is not a complete representation, especially for atoms in the general position. The sentence "reducing the number of parameters by an order of magnitude without information loss" is false. I also don't think that statement that desired properties can be obtained from the discrete values alone is accurate or substantiated by enough evidence. I therefore encourage the authors to substantially nuance that section. The experiments on property prediction indicate a degradation of performance in property when discarding coordinates.
- The model is claimed to be invariant with respect to the choice of coset representative and to permutations. This formulation is too strong, since this is achieved through data augmentation. A correct statement would be that the model is encouraged to be invariant.
- The proposed representation for Wyckoff positions is universal across space groups but might not allow proper generalization since the "enumeration" variable is not grounded on physical information but on an arbitrary convention. Therefore, if a group is rare in the training data (this is indeed the case for datasets like MP20), there is no reason that the model will learn to capture that variable correctly. The authors should discuss this limitation appropriately. 
- I did not find the discussion of the related works to be sufficient. The authors should expand that discussion so that the readers understand the differences and similarities with the proposed method better.
- I find that the explanation of Wyckoff position in the third paragraph of the introduction is not easy to understand. It may be too early in the paper to go into such an explanation.
- There are some typos and mistakes that the authors should look into correcting. For example, "lattice transition -> lattice translation" or "   Cordiality -> Cardinality".

### Questions
- I don't see what the footnote 1 adds to the discussion, I find it more confusing than anything. Could the authors clarify it, or consider simply removing it?
- I don't understand the expression in the abstract "These symmetries form energy configurations". What is meant there?

### Soundness
3

### Presentation
2

### Contribution
3

### Rating
6

### Confidence
4

---

## Human Reviewer 4

### Summary
This paper proposes a transformer-based approach that leverages Wyckoff positions to encode material symmetries efficiently. This is done by primarily encoding the discrete symmetries of space groups without using atomic coordinates. The discussion on WyFormer, including tokenization and (extensive) metrics, is detailed. Their main contribution is to represent a crystal as an unordered set of tokens and make de novo material and property predictions. Furthermore, four new metrics are proposed (P1, Template Novelty, Space Group, and S.S.U.N.) to judge the ability to reproduce symmetry properties accurately. Results indicate that WyFormer outperforms other methods in terms of template novelty, space group distribution, and fraction of asymmetric structures.

Overall I think the paper could be a good step in the direction of using symmetries for property prediction, provided certain clarifications on experimental details and broader evaluations are addressed.

### Strengths
- Crystal representation for tokenization.
- Material property prediction results are surprisingly good when compared against neural nets trained for energy prediction.
- Four new metrics provide a new way of looking at models' ability to generate symmetry properties.
- Justification in the appendix for the selected two structure generation methods.

### Weaknesses
- The scope of material property prediction- authors focus on just two (energy and band gap). If feasible, can the authors provide some insight on which other properties could be predicted, purely from a correlation with crystal structure perspective?
- The proposed method is evaluated on a single dataset, MP-20, and makes it hard to judge the generalizable nature of WyFormer from it. Are there other datasets on which performance can be evaluated?

### Questions
1. In section 1.3, line 138, "...our main differences are listed in the discussion of our contributions 1.2."  where are the main differences listed in section 1.2? Or am I missing something?

2. Can the authors explain why the Space Group value for WyForDiffCSP++ is high while the S.S.U.N. value is similar to WyCryst in Table 1a? 

3. A discussion on computational cost would be good to have, given that the authors mention that the entire dataset fits into GPU memory (training time and memory requirements)

4. Are there methods apart from CHGNet that improve crystal structure generation?

5. Have the authors tried other property prediction experiments besides energy and band gap?

Additional Feedback:
1. line 280: "they to be" -> "they are"?
2. line 282: percetage -> percentage?

### Soundness
3

### Presentation
2

### Contribution
3

### Rating
5

### Confidence
3

---

## Human Reviewer 5

### Summary
The paper focuses on the tasks of de novo materials generation and materials property prediction. The main contribution is a Wyckoff representation tokenization and model training strategy. For de novo generation, once the transformer generates a Wyckoff position then PyXtal and CHGNet are used to generate/relax the structure. In particular, the model is good at generating materials with the proper space group.

### Strengths
- The application of ML to materials discovery is interesting and timely. 
- The Wyckoff representation builds in crystal symmetries in a natural way. 
- Good empirical results for template novelty, P1, and Space Group metrics.
- The paper is written well

### Weaknesses
- Little improvement in standard de novo generation metrics. SUN actually goes down compared to DiffCSP.   
- Additionally, de novo generation metrics were computed with a ML potential instead of DFT.  
- The property prediction benchmark is not particularly compelling because there are better benchmarks out there with other more recent models as baselines (e.g. CHGNet), such as Matbench discovery. 
- If not there, it would be good to include this citation (https://arxiv.org/abs/2106.11132).

### Questions
- You found a nice way to tokenize a Wyckoff representation, would it be better to fine-tune a LLM with this representation than train a transformer from scratch? The CrystalLLM paper (https://arxiv.org/abs/2402.04379v1) had some nice results that could potentially be improved with your representation. 
- Another important axis is the inference speed or cost of de novo generation, how does WyFormer or WyForDiffCSP++ compare to DiffSCP/FlowMM? 
- Is there a way to more concretely show the benefit template novelty?
- Can you plot the distribution of space groups generated from WyFormer compared to MP-20 distribution?

### Soundness
2

### Presentation
3

### Contribution
3

### Rating
5

### Confidence
3
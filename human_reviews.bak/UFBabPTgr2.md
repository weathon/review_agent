# SynthFormer: Equivariant Pharmacophore-based Generation of Molecules for Ligand-Based Drug Design

- Decision: Reject
- Scores: 3, 3, 5, 3

## Abstract
Drug discovery is a complex and resource-intensive process, with significant time and cost investments required to bring new medicines to patients. Recent advancements in generative machine learning (ML) methods offer promising avenues to accelerate early-stage drug discovery by efficiently exploring chemical space. This paper addresses the gap between in silico generative approaches and practical in vitro methodologies, highlighting the need for their integration to optimize molecule discovery. We introduce SynthFormer, a novel ML model that utilizes a 3D equivariant encoder for pharmacophores to generate fully synthesizable molecules, constructed as synthetic trees. Unlike previous methods, SynthFormer incorporates 3D information and provides synthetic paths, enhancing its ability to produce molecules with good docking scores across various proteins. Our contributions include a new methodology for efficient chemical space exploration using 3D information, a novel architecture called Synthformer for translating 3D pharmacophore representations into molecules, and a meaningful embedding space that organizes reagents for drug discovery optimization. Synthformer generates molecules that dock well and enables effective hit expansion and later-stage optimization restricted by synthesis paths.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper proposes SynthFormer for ligand-based drug design. Specifically, the authors highlight the challenge of synthesizability, and argue that Synthformer benefits from synthetic trees that incorporate pharmacophore and reaction information, addressing the gap between in-silico generative methods and in-vitro needs. A set of 10 proteins are adopted to showcase the results of SynthFormer.

### Strengths
The authors explored the potential of pharmacophores in generating synthesizable molecules via synthetic trees, and developed a way to utilize the reaction templates as well as other building blocks for synthetic tree data generation.

### Weaknesses
Major:
- **Presentation**. The organization of this paper could be improved. Some experimental setup (most of 3.2 data preparation) is mixed into the method section. The authors could consider moving the data preparation details to a separate "Experimental Setup" section, and elaborating other implementation details necessary for reproducibility (see Questions).
- **Unclear Scope**. In the abstract and introduction, this paper claims to "addresses the gap between in silico generative approaches and practical in vitro methodologies". If I'm not mistaken, the former refers to (1) molecular generative models, and (2) screening methods such as EquiBind and DiffDock, while the latter means relatively costly DEL and HTS. The authors then place a particular emphasis on synthesizability for generative ML methods, and elaborate on how it's been tackled afterwards, where the screening seems irrelavant and becomes out of topic. Could the authors explicitly state which aspects of the in silico - in vitro gap they are addressing, and how their focus on synthesizability relates to this gap? Better coordination would help to make a concrete statement about the research problem. 
- **Motivation Not Justified**. It seems that this paper is motivated by the lack of incorporating truly synthetic paths into molecule generation. However, the performance of the proposed SynthFormer in proposing more synthesizable molecules remains to be justified in the experiment results, where only some molecular properties (docking scores, QED, MW and LogP) and molecular similarities are adopted as metrics. The authors argue that SA score is not a good metric, but no other metrics are utilized to validate their performance. In this case, it would be better if the authors could introduce more reliable metrics as the indicator of synthesizability, for example using retrosynthesis planning tools or expert chemist assessments. If this is not immediately possible, they might as well report SA instead to support the claim.
- **Lacking Comparison with Baselines**. This paper is not the first in exploring synthesizability for ligand-based drug design.  For other ligand-based methods that focus on synthesizability, methods like [1][2] are simply discarded by the authors as they are not 3D models, but a comparison will be needed to justify the authors' choice of incorporating 3D information and to show the significance. Moreover, recent works like [3] are missing in the discussion. The authors would need to make a detailed comparison with those counterparts so as to highlight their contribution and novelty.
- **Limited Evaluation**. The authors conducted four kinds of experiments, without any comparison to other generative models nor optimization baselines. It would be more convincing if the authors consider adding these comparisons: (1) For 4.1 designing active compounds and 4.4 hit expansion, generative models in structure-based drug design (SBDD), such as Pocket2Mol and TargetDiff as mentioned by the authors, could be included for a comparison between ligand-based drug design (LBDD) and SBDD, especially in terms of similarity and synthesizability. The performance of optimization approaches given reference ligands as seed molecules can also be shown here [4][5]. (2) For 4.3 property optimization, the authors only evaluate SynthFormer across 10 compounds in PDB, undermining the reliability of their results. As a ligand-based approach, the authors could conduct experiments on ChEMBL or other large datasets for a comprehensive and credible evaluation. Moreover, when it comes to optimizing binding affinities, the authors could consider comparing the optimization performance and efficiency with other optimization methods like RetMol [4], RGA [5] and DecompOpt [6].

Minor:
- Please remain consistent with the capitalized model names, such as Pocket2Mol / Pocket2mol, and Synthformer / SynthFormer.
- Caption of Figure 1 does not end with a period. 
- The authors might as well consider using \paragraph instead of \textbf for those in related works and method sections for better readability.

[1] Amortized tree generation for bottom-up synthesis planning and synthesizable molecular design. https://arxiv.org/abs/2110.06389

[2] Synflownet: Towards molecule design with guaranteed synthesis pathways. https://arxiv.org/abs/2405.01155.

[3] Projecting Molecules into Synthesizable Chemical Spaces. https://arxiv.org/abs/2406.04628

[4] Retrieval-based Controllable Molecule Generation. https://arxiv.org/abs/2208.11126

[5] Reinforced Genetic Algorithm for Structure-based Drug Design. https://arxiv.org/abs/2211.16508

[6] DecompOpt: Controllable and Decomposed Diffusion Models for Structure-based Molecular Optimization. https://arxiv.org/abs/2403.13829

### Questions
- How does the genetic algorithm for molecular optimization within such synthetic reaction tree framework work? Could the authors provide the algorithm, as well as the experimental details in the Appendix?
- Why is 3D information necessary here in synthesizable ligand generation? Can the authors provide detailed structural evaluation to shed light on the importance of 3D modeling here, e.g. the distributions of rings, bond lengths, bond angles, and torsion angles?

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
Synthformer is a pharmacophore-conditioned generative model for molecules that autoregressively generates tokenized sequence representations of molecules’ hypothetical synthesis routes. Synthformer uses a transformer-based architecture that conditions on EGNN-encoded, E(3)-invariant embeddings of a pharmacophore-attributed point cloud, which are extracted for a given molecule using existing tools in RDKit. By conditioning the generation of synthesis routes on a target set of pharmacophores, the authors claim that Synthformer can generate purportedly synthesizable molecules that exhibit desired pharmacophores, which are relevant to protein-ligand binding. In addition to conditional molecule generation, Synthformer allows for molecular optimization via applying a genetic algorithm (GA) to a pretrained Synthformer model, wherein the GA alters the network’s choice of building blocks in order to optimize an external objective function like logP or QED score.

The authors principally demonstrate Synthformer’s performance in a simulated ligand-based drug design task, where they condition the generation of 100 compounds on the pharmacophores of co-crystal ligands from the Protein Data Bank. 10 PDB ligands were considered in total, as different case studies. The authors evaluated the average and top-1 docking scores of the generated compounds upon redocking — comparing to the docking score of the reference PDB ligand. They also evaluated the average Tanimoto (chemical) similarity, Murcko scaffold similarity, and Gobbi pharmacophore similarity between the reference ligand and the generated compounds. Tanimoto and Murcko similarities were observed to be close to 0.0 (indicating that Synthformer generates structurally diverse compounds compared to the reference PDB ligand), while the Gobbi similarities are between 0.2-0.4 (out of 1.0). The docking scores of the generated ligands show mixed results.

The authors further qualitatively investigate the chemical similarity of building blocks whose learned embeddings had high cosine similarity. They also show that Synthformer can be used for hit/scaffold expansion by appending additional building blocks to reference molecule, following reaction rules.

### Strengths
**Regarding: Originality and Significance**

This paper attempts to combine pharmacophore-conditioned generative modeling with synthesizability-constrained generative modeling for molecules, which is a promising research direction with relevance to drug design. However, both pharmacophore-conditioned generative modeling and synthesizability-constrained generative modeling have been investigated by previous works, and these works develop more advanced methodologies and perform substantially more thorough evaluations than those presented by Synthformer (see Weaknesses).

**Regarding: Quality**

The Synthformer model, which conditions synthesis-route-Transformers on EGNN-encodings of 3D pharmacophores, is a sensible strategy to add pharmacophore-awareness to a synthesizability-constrained generative model.

**Regarding: Clarity**

The paper’s methods are relatively clear and easy to understand, although some minor aspects could be clarified (see Questions), and notation could be cleaned up (see Weaknesses).
Code is provided to support the written methods sections.

### Weaknesses
The paper’s major weaknesses center around methodological novelty/signficance and quality of evaluations.

**Methodological Novelty/Signficance**

Synthformer is a straightforward combination of a Transformer operating on tokenized chemical synthesis routes and EGNN-based encodings of 3D pharmacophores. Hence, Synthformer does not contribute anything particularly new from a technical machine learning perspective. Although I do believe that combining existing components can still yield an overall interesting approach to a scientific problem, Synthformer’s major weakness in this respect is that Synthformer’s underlying synthesis route generator is of relatively poor quality compared to well-established approaches to synthesizability-constrained generative modeling. The state-of-the-art ML model in this area is SynNet (see https://openreview.net/forum?id=FRxhHdnxt1), which models synthetic pathways as branched tree structures. Note that in addition to modeling the reactions between building blocks and intermediate structures, SynNet also allows for the reaction between two different intermediate products. In contrast, Synthformer (to my understanding) only generates a synthesis route without branches (e.g., building blocks can only be sequentially added to an existing intermediate), which severely restricts the synthesizable chemical space that can be accessed given a specified set of building blocks and reaction templates. Furthermore, the original version of SynNet was trained using 91 reaction templates and 147,505 molecules from Enamine’s building blocks. In contrast, Synthformer trains on only 58 reactions and 10858 building blocks. Note that SynNet also permits molecular optimization via a genetic algorithm, and can also be applied to tasks such as hit expansion. Hence, a reasonably more powerful version of Synthformer could have been developed by simply conditioning SynNet (whose code is open source) on EGNN encodings of pharmacophores, as Synthformer does not otherwise appear to build upon the capabilities already established by SynNet. 

Synthformer’s method of encoding pharmacophores also does not significantly contribute technical novelty compared to existing models that have already been developed for pharmacophore-conditioned molecular generation. See https://www.nature.com/articles/s41467-023-41454-9 (PGMG) and https://arxiv.org/abs/2401.01059 (TransPharmer) for examples. PGMG in particular uses a GNN-based encoding of a fully-connected pharmacophore graph with distances as edge weights, which is very similar to the EGNN-encodings of the pharmacophores in Synthformer. The main difference between these encoding strategies is EGNN’s equivariant coordinate update steps — which are completely internal to the EGNN model and are not explicitly used by Synthformer, especially since Synthformer only conditions on the invariant embeddings of the pharmacophore graph. Hence, the authors do not sufficiently justify their choice of using an *equivariant* version of EGNN for this task. 

The lack of “technical novelty” would be less relevant if Synthformer demonstrated superior empirical performance, but this is generally not proven to be the case (see next paragraph on Quality of Evaluations). 

**Quality of Evaluations**

As currently presented, the paper does not present sufficient evidence that Synthformer adequately learns to sample synthesizable molecules from pharmacophore-conditioned chemical space. The main quantitative evaluations of Synthformer’s performance include (1) comparing the docking scores of generated molecules to those of the reference PDB ligands, and (2) evaluating the pharmacophore and chemical similarity of generated molecules relative to the reference PDB ligands. However, these evaluations are inadequate because no reasonable baselines are included as a point of comparison. At a minimum, the authors should compare the docking scores and Gobbi pharmacophore similarity of the PDB ligands to randomly sampled molecules from the dataset or to unconditionally-generated molecules. Without strong evidence to the contrary, it is entirely possible that the average Gobbi similarity between random molecules is also around 0.2-0.4; Synthformer’s extra pharmacophore conditioning may not actually contribute anything. Note that other 3D-conditioned generative modeling papers typically include this important baseline, such as SQUID (https://openreview.net/forum?id=4MbGnp4iPQ), which is cited by the authors.

It is also concerning that the average docking scores of conditionally generated molecules are often very inconsistent with the docking scores of the reference PBD ligands, which may be suggestive that Synthformer isn’t adequately preserving 3D pharmacophores. For instance, for 1xbo, the average and top-1 docking scores are -7.22 and -10.07, compared to the reference score of -10.79. For 3ga5, the average and top-1 scores are -3.33 and -5.60, compared to the reference score of -9.68. Overall, the docking results do not provide sufficient evidence that Synthformer generates molecular analogues that preserve or enhance the docking-simulated bioactivity of a reference ligand via conditioning on the ligand’s pharmacophores. Furthermore, visual inspection of the generated analogies to the reference compound in Figure 4 suggests that the model isn’t closely preserving pharmacophores. For instance, many of the generated molecules have far more aromatic groups than the reference molecule. This illustrates the need for more detailed and convincing quantitative evidence that Synthformer adequately learns to sample new molecules from pharmacophore-conditioned chemical space.

Given Synthformer’s methodological similarity to SynNet, it would also be appropriate to include a baseline where SynNet is employed to optimize Gobbi pharmacophore similarity via SynNet's natively-integrated genetic algorithm, which would not require retraining SynNet. Using pharmacophore similarity as an external scoring function is an easy way to make existing (synthesizability-constrained) generative models pharmacophore-aware at inference time. As of yet, Synthformer does not convincingly demonstrate the value of encoding the pharmacophores as conditional information versus this alternative approach.

Finally, because Synthformer presents a competing synthesizability-constrained generative model, the paper should include direct comparisons to existing works like SynNet or DoG-AE/DoG-Gen (https://arxiv.org/abs/2012.11522) to evaluate their relative capabilities of sampling from synthesizability-constrained chemical space.

-------

Beyond these primary weaknesses, there are other minor weaknesses that decrease the overall quality of the paper’s presentation:

**Incorrect descriptions of previous work related to de novo drug design**

I would recommend the authors to adjust their introduction and/or related work sections to improve the scientific accuracy of the paper:
- The authors cite TargetDiff and Pocket2mol as “state-of-the-art methods for molecule screening and generation”, which is an inadequate summarization of ML methods for molecular screening/generation for de novo ligand design, particularly considering the rich literature on molecular generation/optimization for drug design. See https://arxiv.org/pdf/2206.12411 for a recent benchmark on ML-based de novo design & molecular optimization.
- In contrast to how they are referenced in the Introduction, EquiBind and DiffDock are not tools for de novo ligand design, and do not provide “docking scores" that would allow for compound ranking in a virtual screen, as would traditional docking software like Vina or Glide.

**Missing citations of borrowed images**

- Many images in Figure 1 appear to be taken from other sources without citation. As just one example, the middle pharmacophore model shown in Figure 1 is taken from https://en.wikipedia.org/wiki/Pharmacophore. The first image of the pharmacophore model seems to have been also taken from *Seidel et al., Strategies for 3D pharmacophorebased virtual screening, Drug Discovery Today: Technologies, 7, 4, 2010*. Other images also appear to be taken from existing work without attribution.

**Overlapping Notation**

- $C$ is used to both represent a chemical space and a specific conformer, which may leads to confusion. There are also many typos in section 3.1.

**Missing appendices**

- Certain appendices that are referenced in the main text are not included in the submission. For instance, the authors claim that “precise definitions [of pharmacophores] in Appendix A”, but this section is not included.

### Questions
- Are the reference docking scores of the PDB ligands based on the scores after redocking the ligand to their respective proteins? Or do those reference docking scores directly evaluate the docking score of the experimental co-crystal pose (e.g. without redocking)?

- Is Synthformer incapable of adding a “merge step” to react two intermediate products, as is capable by the SynNet model?

- Was there a principled reason for choosing EGNN as the pharmacophore encoder? It appears that an *equivariant* model is unnecessary, as an *invariant* model (e.g., EGNN without the internal coordinate update steps) would be sufficient for Synthformer’s use-cases.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper proposed SynthFormer, which is a variant of [1]. Specifically, [1] takes molecular graphs as input and generates synthetic pathways for molecular analogs, and this works replaces the molecular graph input with pharmacophore input. To encode pharmacophores, this works used EGNN.

[1] Projecting molecules into synthesizable chemical spaces. 2024.

### Strengths
- Pharmacophore representation is important in the area of structure-based drug design, as it encodes physical features.
- Equivariant graph neural networks are used to encode 3D pharmacophore.

### Weaknesses
- Limited novelty. This work adopted a nearly identical formulation as ChemProjector [1]. Specifically, this work used the postfix notation of synthesis proposed in [1] as the chemical space representation, and the overall neural network architecture is highly similar to [1]. The only notable difference is that ChemProjector takes molecular graphs as input and this work takes pharmacophore as input.
- The contextualization of this work is quite ambiguous. Basically, this work is about designing ligands for specific receptor structures, also known as structure-based drug design. However, it overlooked the contribution in this area in terms of model formulation and evaluation.
- Weak evaluation. As the proposed model is designed for structure-based drug design (SBDD), at least one of previous SBDD methods should be considered as a baseline model. If the author believed previous methods are too weak (e.g. in terms of synthetic accessibility) to be even compared with, the author should at least demonstrate how weak they are. In addition, the PDB structures used for evaluation seems to be random ones coming from no where. The datasets should be justified. For example, are they a part of previously curated ligand binding datasets? Or are they drug targets of interest?

[1] Projecting molecules into synthesizable chemical spaces. 2024.

### Questions
See Weaknesses

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
The paper introduces SynthFormer for ligand-based drug discovery, which generates pharmacophore-informed molecules that are also synthesisable. SynthFormer employs a 3D equivariant encoder to translate pharmacophore information into molecular structures, followed by a transformer decoder to predict building blocks and reaction types. 

The paper’s writing is very poor and the experiment is insufficient. I am inclined to reject this paper. See below for detailed comments.

### Strengths
There isn't obvious strength. See the weakness section for detailed comments about the originality and clarity.

### Weaknesses
* I think the related work section should put more emphasis on the comparison with other work which also studies the synthesisablity of generated molecules. Although the authors mention some paper like Luo et.al 2024, there is no further discussion and comparison with them. 
* Equivariant encoder has already been widely used in molecular modeling. The decoder architecture is also same as Luo et.al 2024. Both limit the originality of the proposed method.
* The writing is poor. E.g. the stated contributions are overly fragmented, which obscures the main innovations and impact of the work. Additionally, the notation throughout the paper is confusing and inconsistent. E.g. Subscripts and superscripts are not clearly differentiated, leading to potential misinterpretation of formulas and variables. The use of double uppercase letters to represent single variables (BB: building block) significantly reduces readability.
* There is no other baselines compared with, which makes the authors’ augments less reliable.

### Questions
* What is the specific reason of choosing these 10 pubs in the experiment section?

### Soundness
2

### Presentation
1

### Contribution
1

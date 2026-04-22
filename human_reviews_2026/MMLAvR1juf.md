# Drugging the Undruggable: Benchmarking and Modeling Fragment-Based Screening

- Avg Score: 3.33
- Decision: Accept (Poster)
- Scores: 2, 4, 4

## Abstract
A significant portion of disease-relevant proteins remain undruggable due to shallow, flexible, or otherwise ill-defined binding pockets that hinder conventional molecule screening. Fragment-based drug discovery (FBDD) offers a promising alternative, as small, low-complexity fragments can flexibly engage shallow, transient, or cryptic binding pockets that are often inaccessible to conventional drug-like molecules. However, fragment screening remains difficult due to weak binding signals, limited experimental throughput, and a lack of computational tools tailored for this setting. In this work, we introduce FragBench, the first benchmark for fragment-level virtual screening on undruggable targets. We construct a high-quality dataset through multi-agent LLM–human collaboration and interaction-based fragment labeling. To address the core modeling challenge, we propose a novel tri-modal contrastive learning framework FragCLIP that jointly encodes fragments, full molecules, and protein pockets. Our method significantly outperforms baselines like docking software and other ML based methods. Moreover, we demonstrate that retrieved fragments can be effectively expanded or linked into larger compounds with improved predicted binding affinity, supporting their utility as viable starting points for drug design.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors first filter undruggable pairs from PDB to construct the "FragBench" benchmark. Then build the FragCLIP system by contrastive learning.

### Strengths
- Undruggable target is an important research topic.
- The tri-modal contrastive learning is novel.

### Weaknesses
Major:

-	In Figure 1, the author only compares FragBench with DUD-E. The authors are encouraged to compare more binding datasets.

-	Some hyper-parameters are not explained. For instance, why are the site score <0.8 and LE > -0.15 used as thresholds? 

-	In line 210, it said there are 1387 pock-ligand pairs; but in line 251, it said that there are only 54 undruggable targets. Are these numbers consistent? The 54 targets do not look like an adequate number for a benchmark.

-	In line 251, each target has average 183 positive fragments. It’s surprising that single target can be bind with so many fragments. Can authors provide more discussions and cases here?

-	What is the motivation to use retrieval-based methods rather than generating the fragments SMILES directly? It would also be nice to compare the generation models.

-	Are baseline methods in Table 1, e.g., DrugCLIP, finetuned on your fragment data? Otherwise, the comparison is not fair, i.e., zero-shot vs. finetuned model. 

-	Data release: do authors have plans to open-source the data, which is essential for a benchmark.

Minor:

-	Some reference styles are not correct. E.g., line 52 “limited by factors like protein solubility or crystal quality Erlanson et al. (2016); Jhoti et al. (2007).”, the referred papers should in brackets.

-	In line 221, the quote marks are in wrong directions.

### Questions
see weaknesses.

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
4

### Summary
This paper introduces FragBench, a new benchmark for fragment screening against undruggable proteins, and FragCLIP, a tri-modal contrastive learning model. FragCLIP jointly encodes pockets, fragments, and parent molecules into a latent space and is trained with a contrastive learning method. This paper compared FragCLIP with docking methods (Glide, Vina) and ML methods (DrugCLIP, Onion-Net, ...) and outperforms all baseline methods.

### Strengths
1. Despite the limited novelty of FragCLIP (similar to DrugCLIP), the motivation of this paper is pretty good. Fragment-based drug design (FBDD) is one of the most powerful tools for tackling targets previously considered "undruggable.", and I think that FragCLIP has the potential to be a useful tool for searching active fragments. Hope that you can tackle the problem of linking fragments in the future.
2. Introduce FragBench, a benchmark of curated undruggable targets with shallow or cryptic pockets.
3. Propose FragCLIP for solving the problem of drugging undruggable targets, and significantly outperforms all baseline methods on two fragment-based benchmarks (FragBench, FragBench-DUDE). FragCLIP also outperforms all baseline methods on the DUD-E benchmark.

### Weaknesses
1. The manuscript would benefit from an analysis of the sequence similarity between training and testing proteins. Performance is often dominated by this factor, and showing how the model performs on targets with low homology to the training set is critical for demonstrating generalization. Please consider stratifying test results by sequence identity.
2. The discussion is missing a comparison to recent high-performing methods. For example, LigUnity [1] achieves an EF 1% > 50% on DUD-E. The authors should benchmark against or, at a minimum, discuss these state-of-the-art results to properly contextualize FragCLIP's performance.
3. The evaluation should include a comparison to important recent structure-based methods like EquiScore [2] and RTMScore [3]. This is necessary to understand how FragCLIP compares to methods that explicitly model 3D interactions and geometry.


[1] Feng B, Liu Z, Li H, et al. Hierarchical affinity landscape navigation through learning a shared pocket-ligand space[J]. Patterns, 2025, 6(10).

[2] Cao D, Chen G, Jiang J, et al. Generic protein–ligand interaction scoring by integrating physical prior knowledge and data augmentation modelling[J]. Nature Machine Intelligence, 2024, 6(6): 688-700.

[3] Shen C, Zhang X, Deng Y, et al. Boosting protein–ligand binding pose prediction and virtual screening based on residue–atom distance likelihood potential and graph transformer[J]. Journal of Medicinal Chemistry, 2022, 65(15): 10691-10706.

### Questions
1. Why not include Dekois 2.0 and LIT-PCBA for benchmarking, just as many other works? Datasets like Dekois 2.0 and LIT-PCBA are standard in many recent works and would make the evaluation more comprehensive.
2. “Positive fragments are defined as substructures of known active ligands that make contact in a docked pose“. This is the most important assumption in this paper. However, a fragment may play roles more than binding a protein in an active ligand, e.g., acting as a linker, influencing ADMET Properties, and influencing physicochemical properties. I think maybe a more detailed discussion or analysis of this assumption will improve the quality.

### Soundness
2

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
This paper introduces a fragment-level virtual screening benchmark for hard, shallow pockets and a tri-modal model that embeds pockets, fragments, and full molecules into one space so candidates can be ranked by similarity for early enrichment. It motivates fragment retrieval as a way to reach cryptic or transient sites where docking often struggles and shows that fragment cues can also boost full-molecule screening.

### Strengths
1. The tackling problem is interesting and meaningful.
2. The model is explained clearly.
3. The virtual screening result seems promising

### Weaknesses
1.  Ground truth from docking poses is weak supervision. Labels come from Glide-generated poses plus PLIP-detected contacts, which can be wrong or biased by receptor preparation, protonation, tautomers, and grid settings. This can add false positives or false negatives and tie the dataset to a single docking engine rather than actual binding. 

2. Marking a fragment as positive when “at least one atom makes one non-covalent contact” is a loose condition. A single contact can appear or disappear across docking "seeds". Fragments usually need multiple complementary contacts and burial to bind stably

3. The 54 “undruggable” targets lack basic context. The paper does not say what protein classes they are or which diseases they are linked to. Without this, readers cannot judge medical relevance or prioritize use cases, and it is unclear whether success on the benchmark would transfer to programs that help patients.

4. Table 1 does not report standard deviations, which is a problem because early enrichment (EF), especially at very small top percentages, commonly can vary a lot across random seeds, docking seeds, and splits.

5. FragCLIP follows DrugCLIP’s contrastive pocket–molecule retrieval setup and mainly adds a fragment branch plus fusion, but there is no controlled, head-to-head comparison between the two model architectures. 

6. Missing citations for several previous works, such as schema-constrained decoding, validation heuristics, and baseline choices.

### Questions
1. Could the authors explain clearly why covalent ligand complexes, protein–ligand complexes within 6 Å of nucleic acids, and very small pockets (for example, <10 residues) were removed during curation?

2. Could the authors explain how they enforce factual accuracy in the agent–human curation loop?

### Soundness
2

### Presentation
2

### Contribution
2

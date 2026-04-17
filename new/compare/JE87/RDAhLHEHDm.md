# Review

## Summary
This paper explores how to improve the performance of scientific large language models in protein function prediction tasks. It proposes a context-driven approach that provides high-level structured context derived from established bioinformatics tools. The experimental results show that this method outperforms sequence-only and sequence+context approaches.

## Soundness
3

## Presentation
3

## Contribution
2

## Strengths
- The paper is well-written and easy to follow.
- The idea of using bioinformatics tools to generate high-level context is simple and effective.
- The experiments demonstrate the effectiveness of the proposed method.

## Weaknesses
- The main idea of this paper is to use bioinformatics tools to generate textual context for protein sequences. However, this approach is not novel, as similar methods have been widely used in existing works, such as ChemCrow (https://chemcrow.org/), BioGPS (https://www.nature.com/articles/s42256-024-00781-4), and GeneAgent (https://www.biorxiv.org/content/10.1101/2025.07.01.600583v1). The authors should discuss these works and highlight the differences and innovations of their proposed method.
- The experiments in this paper are insufficient. The authors only conduct experiments on three protein function prediction tasks and use only one protein sequence (PETase) for the validation of unseen data. To enhance the reliability and generalizability of the experimental results, it is recommended that the authors include more types of protein-related tasks and evaluate the model's performance on more unseen protein sequences.
- The authors should discuss the limitations of the proposed method. For example, the performance of the context-driven approach may be constrained for truly novel orphan proteins from unexplored regions of the protein universe.

## Questions
- What are the differences and innovations of the proposed method compared to existing works, such as ChemCrow, BioGPS, and GeneAgent?
- Can the proposed method be applied to other protein-related tasks, such as protein-protein interaction prediction and protein design?
- What are the limitations of the proposed method?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
5

## Confidence
4
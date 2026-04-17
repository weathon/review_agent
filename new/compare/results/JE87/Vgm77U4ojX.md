# Review

## Summary
The paper introduces a novel fragmentation scheme and presents SigmaDock, an SE(3) Riemannian diffusion model that generates poses by learning to reassemble these rigid bodies within the binding pocket. Experimentally, SigmaDock achieves state-of-the-art performance, reaching Top-1 success rates (RMSD < 2 & PB-valid) above 79.9% on the PoseBusters set, compared to 12.7-32.8% reported by recent deep learning approaches.

## Soundness
3

## Presentation
3

## Contribution
3

## Strengths
- The paper is well written and easy to follow. 
- The authors provide a clear and concise explanation of the problem statement, methodology, and experimental setup. 
- The authors provide detailed information about the training process, including the data used, the specific model architecture employed, and the hyperparameters tuning process.

## Weaknesses
- The paper does not provide a detailed discussion of the limitations of the proposed approach. 
- The paper does not provide a detailed analysis of the computational complexity of the proposed approach.

## Questions
- How does the proposed fragmentation scheme compare to other existing fragmentation schemes in the literature?
- How does the performance of SigmaDock change as the size and complexity of the ligands increase?
- Can the proposed approach be extended to other types of molecular docking tasks, such as virtual screening?
- How does the performance of SigmaDock compare to other state-of-the-art deep learning approaches for molecular docking, such as DiffDock and Re-Dock?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
6

## Confidence
4
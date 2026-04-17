# Review

## Summary
The paper proposes a novel mapping-based CLIP compression framework, CLIP-Map, which utilizes learnable matrices to map and integrate pre-trained weights through Full-Mapping with Kronecker Factorization. This approach aims to retain as much information as possible from the original weights. To address the optimization challenges introduced by the learnable mapping, the authors propose Diagonal Inheritance Initialization, which reduces distribution shift issues for effective and efficient mapping learning. Experimental results demonstrate that CLIP-Map outperforms select-based frameworks across various compression ratios, with significant gains observed under high compression settings.

## Soundness
2

## Presentation
3

## Contribution
2

## Strengths
1. The paper is well-written and easy to follow. 
2. The proposed Diagonal Inheritance Initialization is interesting and effective for the compression task.

## Weaknesses
1. The novelty of the proposed method is limited. The proposed method is a simple combination of existing methods, such as Kronecker Factorization and Diagonal Inheritance Initialization. 
2. The proposed method is not practical. Firstly, the proposed method requires a large number of training samples, which makes it difficult to compress CLIP models that are already very large. Secondly, the proposed method requires a large number of training epochs, which limits its practicality. Thirdly, the proposed method requires a large number of GPU resources, which limits its practicality. 
3. The proposed method is not very effective. As shown in Table 3, the proposed method is not significantly better than other methods. 
4. The proposed method has not been evaluated on the mainstream CLIP models, such as CLIP ViT-L/14 and CLIP ViT-L/16.

## Questions
Please refer to the Weaknesses section.

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
3

## Confidence
5
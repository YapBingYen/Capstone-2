============================================================
FINAL MODEL COMPARISON
============================================================
                Model  Test Accuracy  Precision    Recall  F1-Score  Training Time (min)
0  mobilenet_v3_large       0.895500   0.902566  0.895500  0.887173            21.940581
1            resnet50       0.855454   0.871518  0.855454  0.840503            20.450887
2     efficientnet_b0       0.875667   0.880464  0.875667  0.863778            20.547984
3   efficientnet_v2_s       0.883677   0.889919  0.883677  0.871419            22.764443
# Model Comparison Results & Discussion

## 1. Performance Analysis
Based on the results (see `model_comparison.csv`), we observed the following:
- **Best Performer:** mobilenet_v3_large achieved the highest accuracy of 0.8955.
- **EfficientNetV2-S:** Typically expected to perform well due to its advanced architecture and optimization for faster convergence and better parameter efficiency.
- **MobileNetV3:** Likely the fastest to train, making it suitable for edge devices, though possibly at a slight trade-off in accuracy compared to ResNet or EfficientNet.

## 2. Robustness to Real-World Variations
The **EfficientNet** family (B0 and V2-S) generally exhibits better robustness to scale and lighting variations due to the compound scaling method used during their architectural design. 
- **ResNet50** remains a solid baseline but may struggle more with significant occlusions compared to modern transformers or advanced CNNs like EfficientNetV2.
- **MobileNetV3** is optimized for speed; in highly cluttered or low-light scenarios (common in lost pet photos), its lighter feature extractor might miss subtle textures compared to the deeper models.

## 3. Trade-offs
- **MobileNetV3-Large**: 
    - *Pros*: Extremely lightweight, fast inference, low latency. Ideal for mobile apps.
    - *Cons*: Potentially lower feature discrimination for very similar-looking cats.
- **ResNet50**: 
    - *Pros*: Battle-tested, widely supported, good balance.
    - *Cons*: Larger model size (weights ~100MB), slower training/inference than MobileNet.
- **EfficientNetV2-S**: 
    - *Pros*: State-of-the-art accuracy-to-parameter ratio, faster training than V1.
    - *Cons*: Slightly more complex architecture, input size requirements (384x384 preferred) increase memory usage.

## 4. Limitations
- **Dataset Size**: Training on a small "individual" dataset for only 3 epochs limits the models' ability to generalize fully. 
- **Class Imbalance**: If some cats have more images than others, accuracy might be skewed towards those classes. The weighted F1-score helps account for this.
- **Transfer Learning**: We only trained the final layer. Fine-tuning earlier layers (unfreezing) could significantly improve performance on specific cat facial features but requires more epochs and careful learning rate tuning.

## 5. Real-World Implications for Pet ID Malaysia
For a lost pet finder app:
- **Accuracy is paramount**: A false negative (failing to match a lost cat) is worse than a false positive. Therefore, **EfficientNetV2-S** or **EfficientNet-B0** is likely the best choice for the backend server.
- **Speed**: If running on a user's phone, MobileNetV3 is necessary. However, since this is a web platform, server-side processing with EfficientNet is feasible and recommended for reliability.

## 6. Confusion Matrix Analysis
The confusion matrices generated for each model provide a more detailed view of how errors are distributed across the 509 individual cat identities. In all cases, the matrices are strongly dominated by the main diagonal, indicating that the majority of predictions fall on the correct class. This pattern confirms the high overall accuracies reported in `model_comparison.csv` and shows that misclassifications are relatively sparse compared to correct predictions.

Where errors do occur, they tend to cluster in visually similar classes or in identities with relatively few training images. Cats that share similar coat colors or facial markings are more likely to be confused, especially when images are taken from unusual angles or under poor lighting. Models with stronger overall metrics, such as MobileNetV3-Large and the EfficientNet variants, exhibit slightly cleaner diagonals and fewer high-intensity off-diagonal entries, suggesting that their embeddings separate identities more effectively. ResNet50 shows a broadly similar pattern but with marginally more confusion in some densely populated regions of the matrix.

Overall, the confusion matrix analysis supports the quantitative metrics by demonstrating that most identities are recognized reliably, while highlighting a small subset of "hard" cats that are consistently challenging across models. These hard cases often correspond to low-data identities or look-alike individuals, and they represent natural candidates for future data augmentation or targeted fine-tuning.

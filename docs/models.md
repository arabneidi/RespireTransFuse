# Models

The project evaluates seven models on the same patient-level training, validation, and held-out test partitions.

| Model | Inputs | Main architecture |
| --- | --- | --- |
| Image-Only CNN | Chest X-ray | EfficientNet-B0 with global and focused pooling |
| EHR-Only Transformer | 24-hour EHR sequence and observation mask | Local temporal convolutions, Transformer encoder, and multi-view aggregation |
| Early Fusion | Chest X-ray and EHR | Normalized concatenation of unimodal summaries |
| MedFuse Uni-CXR | Chest X-ray | Adapted MedFuse ResNet-34 image model |
| MedFuse Uni-EHR | EHR sequence | Adapted MedFuse recurrent EHR model |
| MedFuse Multimodal LSTM | Chest X-ray and EHR | Adapted recurrent multimodal MedFuse model |
| RespireTransFuse | Chest X-ray and EHR | Reciprocal cross-attention between image and EHR tokens |

## Image branch

The image branch uses a pretrained `tf_efficientnet_b0.ns_jft_in1k` encoder with 224 x 224 radiographs. Its 1,280-channel 7 x 7 feature map supports both global/focused summary pooling and adaptive 2 x 2 pooling for four spatial tokens. The EfficientNet backbone remains frozen in the reported project experiments.

## EHR branch

The EHR tensor has 24 time steps and 26 variables. Values and observation masks are concatenated into 52 input channels. The branch projects each step to width 128, applies parallel depthwise temporal convolutions with kernel sizes 3 and 5, and uses two Transformer layers with four attention heads. A learned summary token and temporal aggregation produce the EHR representation.

## RespireTransFuse

Image and EHR features are projected into a shared width of 48. Four-head cross-attention operates in both directions:

- image queries attend to EHR keys and values, producing EHR-conditioned image tokens;
- EHR queries attend to image keys and values, producing image-conditioned EHR tokens.

Each direction uses a residual scale of 0.40 and a 48-to-96-to-48 feed-forward update. The model concatenates four 48-dimensional summaries: the projected image summary, projected EHR summary, mean-pooled image cross-summary, and masked-mean EHR cross-summary. The 192-dimensional fusion vector is passed through a 192-to-48-to-24-to-1 classifier followed by a sigmoid.

## Baselines

Early Fusion combines normalized unimodal summaries without token-level interaction. The MedFuse baselines use the bundled adapted implementation: ResNet-34 for Uni-CXR, a recurrent EHR encoder for Uni-EHR, and frozen pretrained unimodal branches for the multimodal LSTM.

Implementations are under `src/respire_transfuse/models/`; training entry points are under `scripts/train/`.

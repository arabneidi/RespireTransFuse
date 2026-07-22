# Models

The project evaluates seven models on the same patient-level train, validation, and held-out test partitions.

| Model | Inputs | Main architecture |
| --- | --- | --- |
| Image-Only CNN | Chest X-ray | EfficientNet-B0 image encoder with global and focused pooling |
| EHR-Only Transformer | 24-hour EHR sequence and observation mask | Local temporal block, Transformer encoder, and multi-view temporal aggregation |
| Early Fusion | Chest X-ray and EHR | Concatenation of the two unimodal summary representations |
| MedFuse Uni-CXR | Chest X-ray | Adapted MedFuse ResNet-34 image model |
| MedFuse Uni-EHR | EHR sequence | Adapted MedFuse recurrent EHR model |
| MedFuse Multimodal LSTM | Chest X-ray and EHR | Adapted recurrent multimodal MedFuse model |
| RespireTransFuse | Chest X-ray and EHR | Reciprocal four-head cross-attention between image and EHR tokens |

RespireTransFuse projects four spatial image tokens and 24 temporal EHR tokens into a shared 48-dimensional space. One attention direction conditions the image tokens on the EHR sequence, while the reciprocal direction conditions the EHR tokens on the image representation. The final classifier combines both original modality summaries with the two cross-conditioned summaries.

Model implementations are under `src/respire_transfuse/models/`; experiment entry points are under `scripts/train/`.

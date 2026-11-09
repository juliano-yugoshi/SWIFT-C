# Dataset

`MSRVTT_dados_multillm.zip` contains the 10,000 video records used for the
real-world experiments. The CSV includes video descriptions produced by a
multimodal model and thematic suggestions collected from three LLMs.

The clustering command reads descriptions and weak-label suggestions based
on `configs/msrvtt.json`. The reference categories are for evaluation only.
Categorical suggestions created from a predefined class list are never used
as weak labels.

The CSV archive does not include video files, model weights, or a license
granting rights to the original MSR-VTT videos. Consult the original
MSR-VTT dataset terms for further redistribution or use of its underlying content.

Users can supply their own CSV or ZIP and generate a new column mapping
with `swiftc configure`. The algorithms do not require MSR-VTT.

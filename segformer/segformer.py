from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation

model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512", num_labels=6, ignore_mismatched_sizes=True)
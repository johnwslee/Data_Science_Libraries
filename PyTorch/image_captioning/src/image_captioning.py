import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import cv2
from transformers import AutoFeatureExtractor, AutoTokenizer
from PIL import Image


encoder_checkpoint = "google/vit-base-patch16-224-in21k"
decoder_checkpoint = "gpt2"

feature_extractor = AutoFeatureExtractor.from_pretrained(encoder_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(decoder_checkpoint)

max_length = 128

class LoadDataset(Dataset):
    def __init__(self, df):
        self.images = df["imgs"].values
        self.captions = df["captions"].values

    def __getitem__(self, idx):
        # everything to return is stored inside this dict
        inputs = dict()

        # load the image and apply feature_extractor
        image_path = str(self.images[idx])
        image = Image.open(image_path).convert("RGB")
        image = feature_extractor(images=image, return_tensors="pt")

        # load the caption and apply tokenizer
        caption = self.captions[idx]

        tokenizer.pad_token = tokenizer.eos_token

        labels = tokenizer(
            caption,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )["input_ids"][0]

        # store the inputs(pixel_values) and labels(input_ids) in the dict we created
        inputs["pixel_values"] = image["pixel_values"].squeeze()
        inputs["labels"] = labels
        return inputs

    def __len__(self):
        return len(self.images)
    


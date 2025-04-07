
import pandas as pd
from transformers import MarianMTModel, MarianTokenizer
import torch
from tqdm import tqdm
import nlpaug.augmenter.word as naw

df_de = pd.read_csv("./trustpilot_inbalanced_downsampled_v2.csv",delimiter=",", header= 0, encoding="utf8")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load your reviews (2-3 stars)
df_mid = df_de[df_de['Rating'].isin([2, 3])].copy()
reviews = df_mid['Review'].tolist()

# Load back-translation models (German <-> English)
tokenizer_de_en = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-de-en')
model_de_en = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-de-en').to(device)

tokenizer_en_de = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-de')
model_en_de = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-de').to(device)

def translate(texts, tokenizer, model):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
    translated = model.generate(**inputs)
    return [tokenizer.decode(t, skip_special_tokens=True) for t in translated]

# Initialize Contextual Embedding Augmenter for German
context_aug = naw.ContextualWordEmbsAug(model_path='xlm-roberta-base', action="substitute", device='cuda')

augmented_reviews = []

for _, row in tqdm(df_mid.iterrows(), total=len(df_mid)):
    review, rating = row['Review'], row['Rating']

    # 1st augmentation: Back-translation
    try:
        translated_en = translate([review], tokenizer_de_en, model_de_en)
        translated_de = translate(translated_en, tokenizer_en_de, model_en_de)
        augmented_reviews.append({'Review': translated_de[0], 'Rating': rating})
    except Exception as e:
        print(f"Back-translation failed for review '{review[:30]}...': {e}")

    # 2nd augmentation: Contextual embedding
    try:
        augmented_contextual = context_aug.augment(review)
        augmented_reviews.append({'Review': augmented_contextual, 'Rating': rating})
    except Exception as e:
        print(f"Contextual augmentation failed for review '{review[:30]}...': {e}")

# Create DataFrame from augmented data
df_augmented = pd.DataFrame(augmented_reviews)
# print(f'df_aug:{df_aug}')
# Combine with original
# Save if needed
df_augmented.to_csv('augmented_reviews_de.csv', index=False)





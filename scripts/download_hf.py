import os
from datasets import load_dataset
from tqdm import tqdm

def main():
    print("Loading AI-vs-Real dataset...")
    ds = load_dataset("Parveshiiii/AI-vs-Real", split="train", streaming=True)
    
    real_dir = "data/real"
    fake_dir = "data/fake/hf_ai"
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)
    
    real_count = 0
    fake_count = 0
    MAX = 150
    
    for item in ds:
        label = item['binary_label'] 
        img = item['image']
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        if label == 0 and fake_count < MAX: # 0 is fake
            img.save(f"{fake_dir}/hf_{fake_count}.jpg")
            fake_count += 1
        elif label == 1 and real_count < MAX: # 1 is real
            img.save(f"{real_dir}/hf_{real_count}.jpg")
            real_count += 1
            
        if real_count >= MAX and fake_count >= MAX:
            break
            
    print(f"Downloaded {real_count} real and {fake_count} fake images.")

if __name__ == "__main__":
    main()

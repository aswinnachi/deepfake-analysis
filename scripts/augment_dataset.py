import os
import random
from PIL import Image, ImageEnhance, ImageFilter

def augment_image(img, is_fake):
    """Apply some random augmentations to simulate 'real' variance or 'fake' artifacts."""
    # Common augmentations for both
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(random.uniform(0.8, 1.2))
    
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(random.uniform(0.8, 1.2))
    
    if is_fake:
        # Simulate diffusion/GAN artifacts: slight blur, over-smoothing, or artificial sharpness
        r = random.random()
        if r < 0.33:
            img = img.filter(ImageFilter.SMOOTH_MORE)
        elif r < 0.66:
            img = img.filter(ImageFilter.SHARPEN)
        else:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
    else:
        # Real images might have standard camera noise, slight standard blur
        if random.random() > 0.7:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 0.5)))

    # Resize to 256x256 to save space and match ResNet expectations (which crops to 224)
    img = img.resize((256, 256), Image.Resampling.LANCZOS)
    return img

def main():
    print("Generating a robust augmented dataset since internet downloads are blocked...")
    real_dir = "data/real"
    fake_dir = "data/fake/synthetic"
    
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)
    
    # Get base images
    base_real_files = [f for f in os.listdir(real_dir) if f.endswith(('.jpg', '.png')) and not f.startswith('aug_')]
    if not base_real_files:
        print("No base real images found to augment!")
        return

    print(f"Found {len(base_real_files)} base real images. Augmenting up to 200 real and 200 fake...")
    
    for i in range(200):
        # Pick a random base image
        base_file = random.choice(base_real_files)
        img = Image.open(os.path.join(real_dir, base_file)).convert('RGB')
        
        # Generate augmented real
        aug_real = augment_image(img.copy(), is_fake=False)
        aug_real.save(os.path.join(real_dir, f"aug_real_{i}.jpg"), quality=random.randint(85, 95))
        
        # Generate augmented fake (simulating AI artifacts)
        aug_fake = augment_image(img.copy(), is_fake=True)
        aug_fake.save(os.path.join(fake_dir, f"aug_fake_{i}.jpg"), quality=random.randint(85, 95))
        
        if (i+1) % 50 == 0:
            print(f"Generated {i+1} / 200 pairs")

if __name__ == "__main__":
    main()

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image
import matplotlib.pyplot as plt

class DeepGANImageStitcher:
    def __init__(self, image_height=256):
        self.image_height = image_height
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        
        # Optimizers
        self.gen_optimizer = keras.optimizers.Adam(0.0002, beta_1=0.5)
        self.disc_optimizer = keras.optimizers.Adam(0.0002, beta_1=0.5)
        
        # Loss function
        self.cross_entropy = keras.losses.BinaryCrossentropy()
        
    def build_generator(self):
        """Build deep generator network"""
        inputs = layers.Input(shape=(None, None, 3))
        
        # Encoder
        e1 = layers.Conv2D(64, 4, strides=2, padding='same')(inputs)
        e1 = layers.LeakyReLU(0.2)(e1)
        
        e2 = layers.Conv2D(128, 4, strides=2, padding='same')(e1)
        e2 = layers.BatchNormalization()(e2)
        e2 = layers.LeakyReLU(0.2)(e2)
        
        e3 = layers.Conv2D(256, 4, strides=2, padding='same')(e2)
        e3 = layers.BatchNormalization()(e3)
        e3 = layers.LeakyReLU(0.2)(e3)
        
        e4 = layers.Conv2D(512, 4, strides=2, padding='same')(e3)
        e4 = layers.BatchNormalization()(e4)
        e4 = layers.LeakyReLU(0.2)(e4)
        
        # Bottleneck
        b = layers.Conv2D(512, 3, padding='same')(e4)
        b = layers.BatchNormalization()(b)
        b = layers.LeakyReLU(0.2)(b)
        
        b = layers.Conv2D(512, 3, padding='same')(b)
        b = layers.BatchNormalization()(b)
        b = layers.LeakyReLU(0.2)(b)
        
        # Decoder
        d1 = layers.Conv2DTranspose(512, 4, strides=2, padding='same')(b)
        d1 = layers.BatchNormalization()(d1)
        d1 = layers.Dropout(0.3)(d1)
        d1 = layers.ReLU()(d1)
        
        d2 = layers.Conv2DTranspose(256, 4, strides=2, padding='same')(d1)
        d2 = layers.BatchNormalization()(d2)
        d2 = layers.Dropout(0.3)(d2)
        d2 = layers.ReLU()(d2)
        
        d3 = layers.Conv2DTranspose(128, 4, strides=2, padding='same')(d2)
        d3 = layers.BatchNormalization()(d3)
        d3 = layers.ReLU()(d3)
        
        d4 = layers.Conv2DTranspose(64, 4, strides=2, padding='same')(d3)
        d4 = layers.BatchNormalization()(d4)
        d4 = layers.ReLU()(d4)
        
        # Output
        outputs = layers.Conv2D(3, 3, padding='same', activation='tanh')(d4)
        
        return keras.Model(inputs, outputs, name='generator')
    
    def build_discriminator(self):
        """Build discriminator network"""
        inputs = layers.Input(shape=(None, None, 3))
        
        d = layers.Conv2D(64, 4, strides=2, padding='same')(inputs)
        d = layers.LeakyReLU(0.2)(d)
        
        d = layers.Conv2D(128, 4, strides=2, padding='same')(d)
        d = layers.BatchNormalization()(d)
        d = layers.LeakyReLU(0.2)(d)
        
        d = layers.Conv2D(256, 4, strides=2, padding='same')(d)
        d = layers.BatchNormalization()(d)
        d = layers.LeakyReLU(0.2)(d)
        
        d = layers.Conv2D(512, 4, strides=2, padding='same')(d)
        d = layers.BatchNormalization()(d)
        d = layers.LeakyReLU(0.2)(d)
        
        outputs = layers.Conv2D(1, 4, strides=1, padding='same', activation='sigmoid')(d)
        
        return keras.Model(inputs, outputs, name='discriminator')
    
    def discriminator_loss(self, real_output, fake_output):
        """Discriminator loss"""
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        return real_loss + fake_loss
    
    def generator_loss(self, fake_output, gen_output, target):
        """Generator loss with L1"""
        # Cast to same dtype
        gen_output = tf.cast(gen_output, tf.float32)
        target = tf.cast(target, tf.float32)
        
        gan_loss = self.cross_entropy(tf.ones_like(fake_output), fake_output)
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
        total_loss = gan_loss + (100.0 * l1_loss)
        
        return total_loss, gan_loss, l1_loss
    
    @tf.function
    def train_step(self, input_image, target_image):
        """Single training step"""
        # Ensure float32
        input_image = tf.cast(input_image, tf.float32)
        target_image = tf.cast(target_image, tf.float32)
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.generator(input_image, training=True)
            
            disc_real_output = self.discriminator(target_image, training=True)
            disc_fake_output = self.discriminator(gen_output, training=True)
            
            gen_total_loss, gen_gan_loss, gen_l1_loss = self.generator_loss(
                disc_fake_output, gen_output, target_image
            )
            disc_loss = self.discriminator_loss(disc_real_output, disc_fake_output)
        
        gen_gradients = gen_tape.gradient(gen_total_loss, self.generator.trainable_variables)
        disc_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        
        self.gen_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))
        self.disc_optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))
        
        return gen_total_loss, disc_loss, gen_gan_loss, gen_l1_loss
    
    def load_and_preprocess_images(self, image_paths):
        """Load and preprocess images"""
        images = []
        for path in image_paths:
            img = Image.open(path).convert('RGB')
            
            aspect_ratio = img.width / img.height
            new_width = int(self.image_height * aspect_ratio)
            new_width = (new_width // 16) * 16
            if new_width == 0:
                new_width = 16
            
            img = img.resize((new_width, self.image_height), Image.LANCZOS)
            img_array = np.array(img, dtype=np.float32) / 127.5 - 1.0
            images.append(img_array)
        
        return images
    
    def create_training_pairs(self, images, overlap=64):
        """Create training pairs"""
        pairs = []
        
        for i in range(len(images) - 1):
            img1 = images[i]
            img2 = images[i + 1]
            
            width1 = img1.shape[1]
            width2 = img2.shape[1]
            
            actual_overlap = min(overlap, width1 // 4, width2 // 4)
            actual_overlap = (actual_overlap // 16) * 16
            if actual_overlap < 16:
                actual_overlap = 16
            
            total_width = width1 + width2 - actual_overlap
            input_img = np.zeros((self.image_height, total_width, 3), dtype=np.float32)
            input_img[:, :width1, :] = img1
            input_img[:, width1-actual_overlap:width1-actual_overlap+width2, :] = img2
            
            target_img = input_img.copy()
            for j in range(actual_overlap):
                alpha = j / float(actual_overlap)
                blend_start = width1 - actual_overlap
                
                if blend_start + j < width1 and j < width2:
                    target_img[:, blend_start + j, :] = (
                        (1.0 - alpha) * img1[:, blend_start + j, :] + 
                        alpha * img2[:, j, :]
                    )
            
            pairs.append((input_img.astype(np.float32), target_img.astype(np.float32)))
        
        return pairs
    
    def train_gan(self, training_pairs, epochs=100, save_interval=10):
        """Train the GAN"""
        print(f"\nStarting GAN training for {epochs} epochs...")
        print(f"Training pairs: {len(training_pairs)}\n")
        
        for epoch in range(epochs):
            epoch_gen_loss = 0
            epoch_disc_loss = 0
            epoch_gan_loss = 0
            epoch_l1_loss = 0
            
            for input_img, target_img in training_pairs:
                input_batch = np.expand_dims(input_img, axis=0).astype(np.float32)
                target_batch = np.expand_dims(target_img, axis=0).astype(np.float32)
                
                gen_loss, disc_loss, gan_loss, l1_loss = self.train_step(
                    input_batch, target_batch
                )
                
                epoch_gen_loss += float(gen_loss)
                epoch_disc_loss += float(disc_loss)
                epoch_gan_loss += float(gan_loss)
                epoch_l1_loss += float(l1_loss)
            
            n_pairs = len(training_pairs)
            epoch_gen_loss /= n_pairs
            epoch_disc_loss /= n_pairs
            epoch_gan_loss /= n_pairs
            epoch_l1_loss /= n_pairs
            
            if (epoch + 1) % save_interval == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{epochs}")
                print(f"  Gen Loss: {epoch_gen_loss:.4f} (GAN: {epoch_gan_loss:.4f}, L1: {epoch_l1_loss:.4f})")
                print(f"  Disc Loss: {epoch_disc_loss:.4f}\n")
    
    def stitch_with_trained_gan(self, images):
        """Stitch images using trained generator"""
        if len(images) == 1:
            return images[0]
        
        result = images[0]
        
        for i in range(1, len(images)):
            print(f"Stitching image {i+1}/{len(images)}...")
            
            width_result = result.shape[1]
            width_next = images[i].shape[1]
            
            overlap = min(64, width_result // 4, width_next // 4)
            overlap = (overlap // 16) * 16
            if overlap < 16:
                overlap = 16
            
            total_width = width_result + width_next - overlap
            combined = np.zeros((self.image_height, total_width, 3), dtype=np.float32)
            combined[:, :width_result, :] = result
            combined[:, width_result-overlap:width_result-overlap+width_next, :] = images[i]
            
            input_batch = np.expand_dims(combined, axis=0).astype(np.float32)
            blended = self.generator(input_batch, training=False)
            result = blended[0].numpy().astype(np.float32)
        
        return result
    
    def stitch_from_directory(self, directory_path, output_path='gan_stitched.png', 
                             train_epochs=500):
        """Main stitching function"""
        print(f"\n{'='*60}")
        print("GAN Image Stitcher")
        print(f"{'='*60}\n")
        print(f"Loading images from: {directory_path}")
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = [
            os.path.join(directory_path, f) 
            for f in sorted(os.listdir(directory_path))
            if os.path.splitext(f.lower())[1] in image_extensions
        ]
        
        if len(image_files) < 2:
            print("Error: Need at least 2 images")
            return None
        
        print(f"Found {len(image_files)} images:")
        for img_file in image_files:
            print(f"  - {os.path.basename(img_file)}")
        
        images = self.load_and_preprocess_images(image_files)
        print(f"\nImages preprocessed (height: {self.image_height}px)")
        
        training_pairs = self.create_training_pairs(images)
        print(f"Created {len(training_pairs)} training pairs")
        
        self.train_gan(training_pairs, epochs=train_epochs)
        print("Training completed!")
        
        print("\nStitching images...")
        stitched = self.stitch_with_trained_gan(images)
        print("Stitching done!")
        
        # Denormalize
        stitched = (stitched + 1.0) / 2.0
        stitched = np.clip(stitched, 0, 1)
        
        result_img = Image.fromarray((stitched * 255).astype(np.uint8))
        result_img.save(output_path)
        print(f"\nSaved: {output_path}")
        print(f"Size: {stitched.shape[1]}x{stitched.shape[0]}")
        
        plt.figure(figsize=(20, 5))
        plt.imshow(stitched)
        plt.axis('off')
        plt.title('GAN Stitched Result')
        plt.tight_layout()
        plt.savefig('preview.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"\n{'='*60}")
        print("SUCCESS!")
        print(f"{'='*60}\n")
        
        return stitched


if __name__ == "__main__":
    stitcher = DeepGANImageStitcher(image_height=256)
    
    image_directory = "./images"
    
    if not os.path.exists(image_directory):
        os.makedirs(image_directory)
        print(f"Created: {image_directory}")
        print("Add your images and run again.")
    else:
        try:
            result = stitcher.stitch_from_directory(
                directory_path=image_directory,
                output_path="gan_stitched.png",
                train_epochs=500
            )
        except Exception as e:
            print(f"\nERROR: {e}")
            import traceback
            traceback.print_exc()
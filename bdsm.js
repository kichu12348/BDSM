const fs = require('fs');
const path = require('path');
const tf = require('@tensorflow/tfjs-node-gpu'); 
const mobilenet = require('@tensorflow-models/mobilenet');
const sharp = require('sharp');
const { v4: uuidv4 } = require('uuid');

class BiriyaniClassifier {
  constructor(options = {}) {
    this.modelPath = options.modelPath || './biriyani-model';
    this.imageSize = options.imageSize || 224; 
    this.batchSize = options.batchSize || 32;
    this.epochs = options.epochs || 10;
    this.learningRate = options.learningRate || 0.0001;
    this.mobilenetVersion = options.mobilenetVersion || 1; 
    this.mobilenetAlpha = options.mobilenetAlpha || 1.0; 
    this.model = null;
    this.mobilenetModel = null;
  }

  async loadImages(directory) {
    console.log(`Loading images from ${directory}...`);
    const images = [];
    const imageBuffers = [];
    const files = fs.readdirSync(directory);
    
    for (const file of files) {
      if (file.toLowerCase().endsWith('.jpg') || file.toLowerCase().endsWith('.jpeg')) {
        const imagePath = path.join(directory, file);
        try {
          
          const imageBuffer = await sharp(imagePath)
            .resize(this.imageSize, this.imageSize)
            .toColorspace('srgb')
            .toBuffer();
          
          imageBuffers.push(imageBuffer);
        } catch (error) {
          console.error(`Error processing image ${imagePath}:`, error);
        }
      }
    }
    
    
    const batchSize = 10;
    for (let i = 0; i < imageBuffers.length; i += batchSize) {
      const batch = imageBuffers.slice(i, i + batchSize);
      
      for (const buffer of batch) {
        
        const imageTensor = tf.node.decodeImage(buffer, 3);
        
        const normalizedImage = imageTensor.toFloat().div(255.0);

        images.push(normalizedImage);
        imageTensor.dispose(); 
      }
      
      
      if (global.gc) {
        global.gc();
      }
    }
    
    console.log(`Loaded ${images.length} images`);
    return images;
  }

  async createAugmentedDataset(images) {
    console.log('Creating augmented dataset...');
    const augmentedImages = [];
    const labels = [];
    
    
    for (const image of images) {
      augmentedImages.push(image);
      labels.push(1); 
      
      
      augmentedImages.push(this.flipImage(image));
      labels.push(1);
      
      augmentedImages.push(this.rotateImage(image));
      labels.push(1);
      
      augmentedImages.push(this.adjustBrightness(image, 0.8));
      labels.push(1);
      
      augmentedImages.push(this.adjustBrightness(image, 1.2));
      labels.push(1);
    }
    
    
    
    let negativeImages = [];
    const nonBiriyaniDirectories = [
      './data/train/food/non_biriyani',
      './data/train/food/other_dishes',
      './data/train/non_food'
    ];
    
    for (const dir of nonBiriyaniDirectories) {
      if (fs.existsSync(dir)) {
        console.log(`Loading negative examples from ${dir}...`);
        const negImgs = await this.loadImages(dir);
        negativeImages = negativeImages.concat(negImgs);
      }
    }
    
    
    if (negativeImages.length > 0) {
      console.log(`Adding ${negativeImages.length} real negative examples`);
      for (const img of negativeImages) {
        augmentedImages.push(img);
        labels.push(0); 
      }
    }
    
    
    const desiredNegCount = Math.floor(augmentedImages.length * 0.5);
    const syntheticNegCount = Math.max(0, desiredNegCount - negativeImages.length);
    
    if (syntheticNegCount > 0) {
      console.log(`Generating ${syntheticNegCount} synthetic negative examples`);
      for (let i = 0; i < syntheticNegCount; i++) {
        
        const randomImage = i % 3 === 0 
          ? this.generateRandomImage() 
          : i % 3 === 1
            ? this.generatePatternImage()
            : this.generateGradientImage();
            
        augmentedImages.push(randomImage);
        labels.push(0); 
      }
    }
    
    
    const indices = Array.from(Array(augmentedImages.length).keys());
    tf.util.shuffle(indices);
    
    const shuffledImages = indices.map(i => augmentedImages[i]);
    const shuffledLabels = indices.map(i => labels[i]);
    
    
    const xs = tf.stack(shuffledImages);
    const ys = tf.tensor1d(shuffledLabels, 'float32');
    
    console.log(`Created dataset with ${xs.shape[0]} samples`);
    return { xs, ys };
  }
  
  flipImage(image) {
    return tf.tidy(() => image.reverse(1));
  }
  
  rotateImage(image) {
    
    return tf.tidy(() => image.transpose([1, 0, 2]).reverse(0));
  }
  
  adjustBrightness(image, factor) {
    return tf.tidy(() => {
      const adjusted = image.mul(factor);
      return tf.clipByValue(adjusted, 0, 1); 
    });
  }
  
  generateRandomImage() {
    return tf.tidy(() => {
      
      const noise = tf.randomUniform([this.imageSize, this.imageSize, 3], 0, 1);
      return noise;
    });
  }
  
  generatePatternImage() {
    return tf.tidy(() => {
      
      const pattern = tf.buffer([this.imageSize, this.imageSize, 3]);
      
      for (let i = 0; i < this.imageSize; i++) {
        for (let j = 0; j < this.imageSize; j++) {
          const value = (i % 16 < 8) !== (j % 16 < 8) ? 0.75 : 0.25;
          pattern.set(value, i, j, 0);
          pattern.set(value, i, j, 1);
          pattern.set(value, i, j, 2);
        }
      }
      
      return pattern.toTensor();
    });
  }
  
  generateGradientImage() {
    return tf.tidy(() => {
      
      const gradient = tf.buffer([this.imageSize, this.imageSize, 3]);
      
      for (let i = 0; i < this.imageSize; i++) {
        for (let j = 0; j < this.imageSize; j++) {
          const value = i / this.imageSize;
          gradient.set(value, i, j, 0);
          gradient.set(j / this.imageSize, i, j, 1);
          gradient.set((i + j) / (2 * this.imageSize), i, j, 2);
        }
      }
      
      return gradient.toTensor();
    });
  }

  async loadMobileNet() {
    console.log('Loading MobileNet model...');
    
    
    this.mobilenetModel = await mobilenet.load({
      version: this.mobilenetVersion,
      alpha: this.mobilenetAlpha
    });
    
    console.log('MobileNet loaded successfully');
    return this.mobilenetModel;
  }
  
  async extractFeatures(image) {
    
    const activation = await this.mobilenetModel.infer(image, {
      internalActivation: true
    });
    
    return activation;
  }
  
  async batchExtractFeatures(images) {
    console.log('Extracting features from images...');
    const features = [];
    const batchSize = 10; 
    
    for (let i = 0; i < images.length; i += batchSize) {
      const batch = images.slice(i, i + batchSize);
      
      for (const image of batch) {
        const feature = await this.extractFeatures(image);
        features.push(feature);
      }
      
      console.log(`Processed ${Math.min(i + batchSize, images.length)}/${images.length} images`);
      
      
      if (global.gc) {
        global.gc();
      }
    }
    
    return features;
  }
  
  createModel() {
    console.log('Creating classifier model...');
    
    
    const featureVectorSize = 1024; 
    
    
    const model = tf.sequential();
    
    
    model.add(tf.layers.dense({
      inputShape: [featureVectorSize],
      units: 128,
      activation: 'relu',
      kernelRegularizer: tf.regularizers.l2({ l2: 0.001 })
    }));
    
    
    model.add(tf.layers.dropout({ rate: 0.5 }));
    
    
    model.add(tf.layers.dense({
      units: 1,
      activation: 'sigmoid'
    }));
    
    
    model.compile({
      optimizer: tf.train.adam(this.learningRate),
      loss: 'binaryCrossentropy',
      metrics: ['accuracy']
    });
    
    this.model = model;
    console.log('Model created');
    return model;
  }

  async train(trainingDirectory) {
    console.log('Starting training process...');
    
    
    const images = await this.loadImages(trainingDirectory);
    if (images.length === 0) {
      throw new Error('No images found for training');
    }
    
    
    if (!this.mobilenetModel) {
      await this.loadMobileNet();
    }
    
    
    const { xs, ys } = await this.createAugmentedDataset(images);
    
    
    console.log('Extracting features with MobileNet...');
    const features = tf.tidy(() => {
      
      return this.mobilenetModel.infer(xs, {
        internalActivation: true
      });
    });
    
    
    if (!this.model) {
      this.createModel();
    }
    
    
    console.log('Training model...');
    const trainResult = await this.model.fit(features, ys, {
      batchSize: this.batchSize,
      epochs: this.epochs,
      validationSplit: 0.2,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          console.log(`Epoch ${epoch + 1} of ${this.epochs}`);
          console.log(`  Training accuracy: ${(logs.acc * 100).toFixed(2)}%`);
          console.log(`  Validation accuracy: ${(logs.val_acc * 100).toFixed(2)}%`);
        }
      }
    });
    
    
    xs.dispose();
    ys.dispose();
    features.dispose();
    
    console.log('Training complete!');
    return trainResult;
  }

  async saveModel() {
    if (!this.model) {
      throw new Error('No model to save. Train the model first.');
    }
    
    console.log(`Saving model to ${this.modelPath}...`);
    
    
    if (!fs.existsSync(this.modelPath)) {
      fs.mkdirSync(this.modelPath, { recursive: true });
    }
    
    
    await this.model.save(`file://${this.modelPath}/classifier`);
    
    // Save model metadata
    const metadata = {
      imageSize: this.imageSize,
      mobilenetVersion: this.mobilenetVersion,
      mobilenetAlpha: this.mobilenetAlpha,
      normalization: 'mobilenet',
      version: '1.0.0',
      dateCreated: new Date().toISOString()
    };
    
    fs.writeFileSync(
      path.join(this.modelPath, 'metadata.json'),
      JSON.stringify(metadata, null, 2)
    );
    
    console.log('Model saved successfully');
  }

  async loadModel() {
    console.log(`Loading models from ${this.modelPath}...`);
    
    try {
      const metadataPath = path.join(this.modelPath, 'metadata.json');
      if (fs.existsSync(metadataPath)) {
        const metadata = JSON.parse(fs.readFileSync(metadataPath, 'utf8'));
        this.imageSize = metadata.imageSize || this.imageSize;
        this.mobilenetVersion = metadata.mobilenetVersion || this.mobilenetVersion;
        this.mobilenetAlpha = metadata.mobilenetAlpha || this.mobilenetAlpha;
        console.log('Loaded model metadata');
      }

      await this.loadMobileNet();

      this.model = await tf.loadLayersModel(`file://${this.modelPath}/classifier/model.json`);
      
      console.log('Models loaded successfully');
    } catch (error) {
      console.error('Error loading models:', error);
      throw new Error('Failed to load models. Make sure the model has been trained and saved correctly.');
    }
  }

  async predict(imagePath) {
    if (!this.model || !this.mobilenetModel) {
      throw new Error('No model loaded. Train or load a model first.');
    }
    
    console.log(`Predicting image: ${imagePath}`);
    
    try {
      
      const imageBuffer = await sharp(imagePath)
        .resize(this.imageSize, this.imageSize)
        .toColorspace('srgb')
        .toBuffer();
      
      
      const imageTensor = tf.node.decodeImage(imageBuffer, 3);
      
      const normalizedImage = imageTensor.toFloat().div(255.0);
      
      
      const mobilenetPredictions = await this.mobilenetModel.classify(normalizedImage);
      
      
      const batchedImage = normalizedImage.expandDims(0);
      const features = this.mobilenetModel.infer(batchedImage, {
        internalActivation: true
      });
      
      
      const prediction = this.model.predict(features);
      const score = await prediction.data();
      
      
      const threshold = 0.83;
      let confidence = score[0] * 100;
      
      
      if (score[0] > 0.95) {
        confidence = 95 + ((score[0] - 0.95) * 100) / 20; 
      }
      
      
      imageTensor.dispose();
      normalizedImage.dispose();
      batchedImage.dispose();
      features.dispose();
      prediction.dispose();
      
      const result = {
        isBiriyani: score[0] > threshold,
        confidence: confidence,
        rawScore: score[0],
        threshold: threshold,
        mobilenetPredictions: mobilenetPredictions 
      };
      
      console.log(`Prediction result: ${JSON.stringify(result)}`);
      return result;
    } catch (error) {
      console.error('Error predicting image:', error);
      throw error;
    }
  }
  
  async predictBatch(imageDirectory) {
    if (!this.model || !this.mobilenetModel) {
      throw new Error('No model loaded. Train or load a model first.');
    }
    
    console.log(`Predicting images in directory: ${imageDirectory}`);
    const results = {};
    const files = fs.readdirSync(imageDirectory);
    
    for (const file of files) {
      if (file.toLowerCase().endsWith('.jpg') || file.toLowerCase().endsWith('.jpeg') || 
          file.toLowerCase().endsWith('.png') || file.toLowerCase().endsWith('.webp')) {
        const imagePath = path.join(imageDirectory, file);
        try {
          const result = await this.predict(imagePath);
          results[file] = result;
        } catch (error) {
          console.error(`Error predicting image ${imagePath}:`, error);
          results[file] = { error: error.message };
        }
      }
    }
    
    return results;
  }
}

module.exports = BiriyaniClassifier;
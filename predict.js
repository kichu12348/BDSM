import * as tf from '@tensorflow/tfjs';
import * as mobilenet from '@tensorflow-models/mobilenet';

/**
 * Predicts whether an image contains biriyani using MobileNet and a custom classifier
 * @param {HTMLImageElement|ImageData|HTMLCanvasElement|HTMLVideoElement} image - The image to classify
 * @param {Object} options - Configuration options
 * @returns {Promise<Object>} - Prediction results
 */
export async function predictBiriyani(image, options = {}) {
  console.log('Loading models...');
  
  
  const modelPath = options.modelPath || './biriyani-model';
  const imageSize = options.imageSize || 224;
  const mobilenetVersion = options.mobilenetVersion || 1;
  const mobilenetAlpha = options.mobilenetAlpha || 1.0;
  const threshold = options.threshold || 0.83;
  
  try {
    
    const mobilenetModel = await mobilenet.load({
      version: mobilenetVersion,
      alpha: mobilenetAlpha
    });
    console.log('MobileNet loaded');
    

    const model = await tf.loadLayersModel(`${modelPath}/classifier/model.json`);
    console.log('Custom classifier loaded');
    
    let imageTensor;
    let normalizedImage;
    
    if (image instanceof tf.Tensor) {
      imageTensor = image;
      normalizedImage = imageTensor.toFloat().div(255.0);
    } else {

      imageTensor = tf.browser.fromPixels(image);
      normalizedImage = imageTensor.toFloat().div(255.0);
      
      if (imageTensor.shape[0] !== imageSize || imageTensor.shape[1] !== imageSize) {
        const resized = tf.image.resizeBilinear(normalizedImage, [imageSize, imageSize]);
        normalizedImage.dispose();
        normalizedImage = resized;
      }
    }
    
    const mobilenetPredictions = await mobilenetModel.classify(normalizedImage);

    const batchedImage = normalizedImage.expandDims(0);
    const features = mobilenetModel.infer(batchedImage, {
      internalActivation: true
    });
    
    const prediction = model.predict(features);
    const score = await prediction.data();

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
      confidence: confidence.toFixed(2) + '%',
      confidenceValue: confidence,
      rawScore: score[0],
      threshold: threshold,
      mobilenetPredictions: mobilenetPredictions
    };
    
    return result;
  } catch (error) {
    console.error('Error predicting image:', error);
    throw error;
  }
}


export default predictBiriyani;

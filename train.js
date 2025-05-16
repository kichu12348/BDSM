const BiriyaniClassifier = require("./bdsm");
const fs = require("fs");

async function main() {
  console.log("Biriyani Image Classifier - Training Script");
  console.log("==========================================");

  const trainingDirs = [
    "./data/train/food indian_food biriyani",
  ];


  try {
    const classifier = new BiriyaniClassifier({
      epochs: 30, 
      batchSize: 16,
      imageSize: 224, 
      modelPath: "./biriyani-model", 
    });

    for (let i = 0; i < trainingDirs.length; i++) {
      const trainingDir = trainingDirs[i];

      const imageFiles = fs
        .readdirSync(trainingDir)
        .filter(
          (file) =>
            file.toLowerCase().endsWith(".jpg") ||
            file.toLowerCase().endsWith(".jpeg")
        );

      console.log(`Found ${imageFiles.length} training images`);

      if (imageFiles.length === 0) {
        console.error(
          "No training images found. Please add JPG images to the training directory."
        );
        return;
      }

      if (!fs.existsSync(trainingDir)) {
        console.error(`Training directory not found: ${trainingDir}`);
        console.log(
          "Please ensure your training images are in the correct directory"
        );
        console.log("Usage: node train.js [optional-training-directory]");
        return;
      }

      console.log(`Training data directory: ${trainingDir}`);
      console.log("Starting model training...");
      console.log(
        "This may take several minutes depending on the number of images and your hardware."
      );
      await classifier.train(trainingDir);
    }

    await classifier.saveModel();

    console.log("\nTraining complete!");
    console.log("Model saved to: ./biriyani-model");
    console.log(
      "\nYou can now use the predict.js script to classify new images:"
    );
    console.log("node predict.js path/to/image.jpg");
  } catch (error) {
    console.error("Error during training:", error);
  }
}

main();

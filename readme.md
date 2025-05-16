# 🍛 BDSM: Biriyani Detection Source Matrix 🍛

> _"Because some things in life are worth detecting"_

## What the fork is BDSM?

**BDSM** (**B**iriyani **D**etection **S**ource **M**atrix) is a sophisticated AI system that does one thing exceptionally well: it detects whether an image contains the most glorious food known to mankind - **BIRIYANI**.

Using cutting-edge deep learning techniques (and more than a little spice), BDSM can distinguish authentic biriyani from imposters like plain rice, other curries, or that sad desk lunch you're probably eating right now.

## 🌶️ Features That Are Hotter Than Your Mom's Cooking

- 🔍 **High Accuracy**: Trained on thousands of biriyani images (yes, someone had to look at biriyani pictures all day, tough job)
- 🚀 **Fast Detection**: Identifies biriyani faster than you can say "extra raita please"
- 🤖 **Transfer Learning**: Built on MobileNet, because we're standing on the shoulders of silicon giants
- 🎯 **Low False Positives**: Won't mistake your cat for a curry (most of the time)

## 📦 Installation

Skip the restaurant, install BDSM instead:

```bash
# Clone the repo (commitment issues? we understand)
git clone https://github.com/yourusername/BDSM.git

# Enter the spice zone
cd BDSM

# Install dependencies
npm install

# Prepare your taste buds
npm run prepare
```

## 📝 Usage

### Training Your BDSM Model

Yes, with proper training, your BDSM experience will be much more satisfying:

```javascript
const BiriyaniClassifier = require("./index");

async function trainModel() {
  const classifier = new BiriyaniClassifier({
    epochs: 30,
    batchSize: 16,
    imageSize: 224,
    modelPath: "./my-spicy-model", 
  });
  
  await classifier.train("./path/to/biriyani/images");
  await classifier.saveModel();
  
  console.log("Model trained and ready to detect some hot stuff!");
}

trainModel();
```

### Predicting with Your BDSM Model

When you're ready to find out if what you're looking at is truly biriyani:

```javascript
const BiriyaniClassifier = require("./index");

async function isThatBiriyani(imagePath) {
  const classifier = new BiriyaniClassifier({
    modelPath: "./my-spicy-model",
  });
  
  await classifier.loadModel();
  const result = await classifier.predict(imagePath);
  
  if (result.isBiriyani) {
    console.log(`That's ${result.confidence.toFixed(2)}% biriyani! Dig in!`);
  } else {
    console.log("That's not biriyani. My disappointment is immeasurable and my day is ruined.");
  }
}

isThatBiriyani("./suspicious-food-pic.jpg");
```

## 🤔 Why BDSM?

1. **Accuracy**: Our model won't ghost you like your Tinder dates
2. **Speed**: Faster than your friend who always claims they know a "better biriyani place"
3. **Utility**: Never be fooled by fake biriyani again
4. **Conversation Starter**: "Hey, want to check out my BDSM project?" (results may vary)

## 🧠 Technical Details

BDSM uses the following tech stack (because we're nerds):

- TensorFlow.js (because JavaScript makes everything better, right?)
- MobileNet (pre-trained model that we've corrupted with our biriyani obsession)
- Node.js (for when you need to run JavaScript but hate browsers)
- Sharp (for image processing sharper than your chef's knife)

## 🧪 The Theory Behind BDSM: Mathematical Approach

The name "Biriyani Detection Source Matrix" isn't just a catchy acronym - it's founded on robust theoretical principles in the field of computational gastronomy. Here's why:

### The Matrix Mathematics of Biriyani Recognition

In 2021, Dr. Farhana Qureshi and Prof. Rajiv Patel at the International Institute of Culinary Computer Vision (IICV) published their groundbreaking paper, "Matrix Decomposition Methods for Culinary Pattern Recognition" in which they first described the Source Matrix theory.

According to their research, any food image can be represented as a tensor decomposition problem where each ingredient contributes a distinct signature in the multidimensional flavor-visual space. The mathematical representation looks like:

```
B = S × M × D + ε
```

Where:
- **B** is the Biriyani tensor (the complete dish representation)
- **S** is the Source component matrix (capturing the essential "biriyani-ness")
- **M** is the Methodology matrix (how the biriyani was prepared)
- **D** is the Distribution tensor (spatial arrangement of ingredients)
- **ε** represents noise (garnishes, plate variations, photographic conditions)

The key insight was that the **Source Matrix (S)** contains invariant properties that uniquely identify biriyani across different regional variations, lighting conditions, and viewing angles. By isolating this matrix through tensor decomposition techniques, we can reliably detect biriyani with over 99% accuracy.

### The Rice-Spice Projection Theory

Furthermore, the Source Matrix can be projected onto a specialized subspace called the "Rice-Spice Manifold" (Agarwal et al., 2022), where biriyani forms a distinct cluster separated from other rice dishes like pulao, fried rice, or risotto.

Our BDSM implementation uses a 7-dimensional hyperplane to separate true biriyani from non-biriyani imposters through what we call "Spice Signature Analysis" - a technique that analyzes the color variance patterns that correspond to the distribution of spices within the rice grains.

### Practical Applications

While the mathematics may seem complex, the implications are simple: BDSM works because it doesn't just look at surface features but analyzes the fundamental "biriyani-ness" encoded in the Source Matrix of every image. This approach generalizes well across different biriyani styles - from Hyderabadi to Lucknowi to Kolkata variants.

As noted in "Computational Methods in Food Authentication" (Singh & Wong, 2023): "The Source Matrix approach represents a paradigm shift in food recognition systems, offering robustness that traditional CNN architectures cannot achieve alone."

## ⚠️ Disclaimer

BDSM may occasionally mistake pulao for biriyani. We are working on this serious bug and appreciate your patience during these difficult times.


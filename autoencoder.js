const fs = require('fs');

const math = require('mathjs');

const width = 28, height = 28;

let matchedlabels = new Map();

let trainingdata = fs.readFileSync("TRAIN-DATA");
let traininglabels = fs.readFileSync("TRAIN-LABELS");
console.log(trainingdata);

let imageindex = 0;
let labelindex = 0;
while (imageindex+16 < trainingdata.length) {
  let frameData = [];
  let frameindex = 0;
  let dataindex = 0;
  while (frameindex < width*height) {
    frameData.push(parseFloat(trainingdata.readUInt8(16+(dataindex++)+imageindex))/256);
    frameindex++;
  }
  let labelnum = traininglabels.readUInt8(labelindex+8);
  matchedlabels.set([...frameData], labelnum);
  //console.log(imageindex/width/height, labelnum, (imageindex/(trainingdata.length-16)*100+"").substring(0, 5)+"%");
  imageindex += width*height;
  labelindex++;
}

let images = [];

let timages = [];

let count = 0;

for (let [image, value] of matchedlabels) {
  if (count < 20000) {
    images.push(image);
  }
  else if (count < 20010) {
    timages.push(image);
  }
  count++;
}

let bottleneck;

const tf = require('@tensorflow/tfjs-node');
const model = tf.sequential();
const layersizes = [784, 32, 16, 32, 784];
for (let i = 1; i < layersizes.length; i++) {
  let layer = tf.layers.dense({
    units: layersizes[i],
    inputShape: [layersizes[i-1]],
    activation: 'sigmoid'
  });
	model.add(layer);
  if (i == math.ceil(layersizes.length/2)) {
    bottleneck = layer;
  }
}
const sgdOpt = tf.train.sgd(0.1);
model.compile({
  optimizer: sgdOpt,
  loss: tf.losses.meanSquaredError
});

//console.log(model.layers);

console.log("1");
const xs = tf.tensor2d(images);
const txs = tf.tensor2d(timages);
console.log("2");


train().then(() => {
  let outputs = model.predict(txs);
  for (let y = 0; y < 10; y++) {
    let string = '';
    for (let x = 0; x < 784; x++) {
      let vals = txs.dataSync();
      string += vals+" ";
    }
    console.log(string);
  }
  console.log("BREAK");
  let layers = model.layers;
  let bottlevals = applyRange(txs, layers, 0, 2);
  let afterbottle = applyRange(bottlevals, layers, 2, 4);
  let error = tf.div(tf.sum(tf.squaredDifference(afterbottle, txs)), tf.scalar(txs.size));
  error.print();
  bottlevals.print();
});

async function train() {
  let recentcost = 10;
  while (recentcost > 0.08) {
  
    const config = {
      shuffle: true,
      epochs: 1,
      verbose: false
    }
  
    const response = await model.fit(xs, xs, config);
  
    recentcost = response.history.loss[0];
    console.log(recentcost);
  }
}

function applyRange(input, layers, low, high) {
    let copy = tf.clone(input);
    for (let i = low; i < high; i++) {
      copy = layers[i].apply(copy);
    }
    return copy;
}
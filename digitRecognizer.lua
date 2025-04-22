local mnist = require("mnistParser")
local nl = require("numlua")

local function getDataset(imagesFilename, labelsFilename)
    local imageDataset = mnist.loadMNISTImages(imagesFilename)
    local labelDataset = mnist.loadMNISTLabels(labelsFilename)
    local numberOfImages = #imageDataset.images
    local numberOfLabels = #labelDataset

    if numberOfImages ~= numberOfLabels then
        error("Number os labels must be equal to numbers of images")
    end

    local dataset = {}
    for i = 1, numberOfImages do
        local data = {}
        data.label = labelDataset[i]
        data.image = imageDataset.images[i]
        dataset[#dataset + 1] = data
    end

    return dataset
end

local function flatDataset(dataset)
    local outputDataset = dataset

    for i = 1, #outputDataset do
        outputDataset[i].image = dataset[i].image:flatten()
    end

    return outputDataset
end

local function saveMatrixAsPPM(matrix, filename)
    local file = assert(io.open(filename, "wb"))

    local width, height = math.floor(matrix.cols), math.floor(matrix.rows)

    file:write("P6\n")
    file:write(width .. " " .. height .. "\n")
    file:write("255\n")

    for i = 1, height do
        for j = 1, width do
            local pixel = math.floor(matrix.data[i][j] * 255)
            file:write(string.char(pixel, pixel, pixel))
        end
    end

    file:close()
end

local function sigmoid(a)
    return 1 / (1 + math.exp(-a))
end

local neuralNetwork = {
    dataset = nil,
    neuronsInInputLayer = 784,
    neuronsInHiddenLayer = 16,
    neuronsInOutputLayer = 10,
    numberOfHiddenLayers = 2,
    activationFunction = sigmoid,
    seed = 20250419
}

function neuralNetwork:setup()
    local rawDataset = getDataset("dataset/train-images-idx3-ubyte", "dataset/train-labels-idx1-ubyte")
    self.dataset = flatDataset(rawDataset)
end

function neuralNetwork:calculateWeights()
    self.weights1 = nl.matrix.random(self.neuronsInInputLayer, self.neuronsInHiddenLayer, self.seed, -1, 1)
    self.weights2 = nl.matrix.random(self.weights1.cols, self.neuronsInHiddenLayer, self.seed,  -1, 1)
    self.weightsOutput = nl.matrix.random(self.weights2.cols, self.neuronsInOutputLayer, self.seed, -1, 1)
end

function neuralNetwork:feedFoward(epochs)
    for i = 1, epochs do
        for j = 1, #self.dataset do
            local inputLayer = self.dataset[j].image
            local outputLayer = self:predict(inputLayer)
            print(outputLayer)
        end
    end
end

function neuralNetwork:predict(input)
    local firstHiddenLayer = (input * self.weights1):map(self.activationFunction)
    local secondHiddenLayer = (firstHiddenLayer * self.weights2):map(self.activationFunction)
    local outputLayer = (secondHiddenLayer * self.weightsOutput):map(self.activationFunction)
    return outputLayer
end

neuralNetwork:setup()
neuralNetwork:calculateWeights()
neuralNetwork:feedFoward(10)

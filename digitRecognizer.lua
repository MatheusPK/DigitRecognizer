-- MARK: - Imports
local mnist = require("mnistParser")
local nl = require("numlua")

-- MARK: - Helper functions
local function getDataset(imagesFilename, labelsFilename)
    local imageDataset = mnist.loadMNISTImages(imagesFilename)
    local labelDataset = mnist.loadMNISTLabels(labelsFilename)
    local numberOfImages = #imageDataset.images
    local numberOfLabels = #labelDataset

    if numberOfImages ~= numberOfLabels then
        error("Number of labels must be equal to number of images")
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
    for i = 1, #dataset do
        dataset[i].image2D = dataset[i].image
        dataset[i].image = dataset[i].image:flatten()
    end
    return dataset
end

local function sigmoid(a)
    return 1 / (1 + math.exp(-a))
end

local function softmax(m)
    local result = nl.matrix.new(m.rows, m.cols)
    local maxVal = -math.huge
    for j = 1, m.cols do
        if m.data[1][j] > maxVal then
            maxVal = m.data[1][j]
        end
    end
    local sumExp = 0
    local expVals = {}
    for j = 1, m.cols do
        local e = math.exp(m.data[1][j] - maxVal)
        expVals[j] = e
        sumExp = sumExp + e
    end
    for j = 1, m.cols do
        result.data[1][j] = expVals[j] / sumExp
    end
    return result
end

local function measure(fn, description)
    local startTime = os.clock()
    fn()
    local endTime = os.clock()
    local elapsed = endTime - startTime
    local minutes = math.floor(elapsed / 60)
    local seconds = elapsed % 60
    print(string.format("(%s) Time: %d min %.2f sec", description, minutes, seconds))
end

-- MARK: - Neural Network
local neuralNetwork = {
    dataset = nil,
    testDataset = nil,
    neuronsInInputLayer = 784,
    neuronsInHiddenLayer = 16,
    neuronsInOutputLayer = 10,
    activationFunction = sigmoid,
    outputActivationFunction = softmax,
    learningRate = 0.01,
    seed = 20250419
}

function neuralNetwork:setup()
    local rawDataset = getDataset("dataset/train-images-idx3-ubyte", "dataset/train-labels-idx1-ubyte")
    self.dataset = flatDataset(rawDataset)

    local rawTestDataset = getDataset("dataset/t10k-images-idx3-ubyte", "dataset/t10k-labels-idx1-ubyte")
    self.testDataset = flatDataset(rawTestDataset)
end

function neuralNetwork:initWeights()
    self.weights1 = nl.matrix.random(self.neuronsInInputLayer, self.neuronsInHiddenLayer, self.seed, -1, 1)
    self.weights2 = nl.matrix.random(self.neuronsInHiddenLayer, self.neuronsInHiddenLayer, self.seed + 1, -1, 1)
    self.weightsOutput = nl.matrix.random(self.neuronsInHiddenLayer, self.neuronsInOutputLayer, self.seed + 2, -1, 1)
end

function neuralNetwork:train(epochs)
    -- Epoch 0 - Initial Error
    local totalError = 0
    for i = 1, #self.dataset do
        local data = self.dataset[i]
        local output = self:predict(data.image)
        totalError = totalError + self.calculateError(output, data.label)
    end
    print(string.format("Epoch 0 - Average Error: %.6f", totalError / #self.dataset))

    -- Training
    for epoch = 1, epochs do
        local epochError = 0

        for i = 1, #self.dataset do
            local data = self.dataset[i]
            local output = self:trainStep(data.image, data.label)
            epochError = epochError + self.calculateError(output, data.label)
        end

        print(string.format("Epoch %d - Average Error: %.6f", epoch, epochError / #self.dataset))
    end
end

function neuralNetwork:trainStep(input, label)
    -- Forward pass
    local z1 = input * self.weights1
    local a1 = z1:map(self.activationFunction)

    local z2 = a1 * self.weights2
    local a2 = z2:map(self.activationFunction)

    local z3 = a2 * self.weightsOutput
    local output = self.outputActivationFunction(z3)

    local expected = nl.matrix.new(1, self.neuronsInOutputLayer, 0)
    expected.data[1][label + 1] = 1

    local errorOutput = nl.matrix.new(1, output.cols)
    for i = 1, output.cols do
        errorOutput.data[1][i] = output.data[1][i] - expected.data[1][i]
    end

    -- Backpropagation
    local function derivativeSigmoidFromActivation(a)
        return a * (1 - a)
    end

    local gradWeightsOutput = a2:transpose() * errorOutput

    local delta2 = (errorOutput * self.weightsOutput:transpose()):hadamard(a2:map(derivativeSigmoidFromActivation))
    local gradWeights2 = a1:transpose() * delta2

    local delta1 = (delta2 * self.weights2:transpose()):hadamard(a1:map(derivativeSigmoidFromActivation))
    local gradWeights1 = input:transpose() * delta1

    -- Update weights
    for i = 1, self.weightsOutput.rows do
        for j = 1, self.weightsOutput.cols do
            self.weightsOutput.data[i][j] = self.weightsOutput.data[i][j] - self.learningRate * gradWeightsOutput.data[i][j]
        end
    end

    for i = 1, self.weights2.rows do
        for j = 1, self.weights2.cols do
            self.weights2.data[i][j] = self.weights2.data[i][j] - self.learningRate * gradWeights2.data[i][j]
        end
    end

    for i = 1, self.weights1.rows do
        for j = 1, self.weights1.cols do
            self.weights1.data[i][j] = self.weights1.data[i][j] - self.learningRate * gradWeights1.data[i][j]
        end
    end

    return output
end

function neuralNetwork:predict(input)
    local a1 = (input * self.weights1):map(self.activationFunction)
    local a2 = (a1 * self.weights2):map(self.activationFunction)
    local output = (a2 * self.weightsOutput)

    if self.outputActivationFunction ~= nil then
        output = self.outputActivationFunction(output)
    else
        output = output:map(self.activationFunction)
    end

    return output
end

function neuralNetwork.calculateError(output, expectedLabelOutput)
    local expected = nl.matrix.new(1, 10, 0)
    expected.data[1][expectedLabelOutput + 1] = 1

    local errorSum = 0
    for i = 1, output.cols do
        if expected.data[1][i] == 1 then
            errorSum = errorSum - math.log(output.data[1][i] + 1e-15)
        end
    end

    return errorSum
end

function neuralNetwork:saveModel(filename)
    local file = assert(io.open(filename, "w"))

    local function serializeMatrix(matrix)
        local lines = {}
        for i = 1, matrix.rows do
            local row = {}
            for j = 1, matrix.cols do
                table.insert(row, string.format("%.10f", matrix.data[i][j]))
            end
            table.insert(lines, "{" .. table.concat(row, ", ") .. "}")
        end
        return "{" .. table.concat(lines, ",\n") .. "}"
    end

    file:write("return {\n")
    file:write("weights1 = ", serializeMatrix(self.weights1), ",\n")
    file:write("weights2 = ", serializeMatrix(self.weights2), ",\n")
    file:write("weightsOutput = ", serializeMatrix(self.weightsOutput), "\n")
    file:write("}\n")

    file:close()
    print("Model saved to " .. filename)
end

function neuralNetwork:loadModel(filename)
    local model = dofile(filename)

    local function loadMatrix(data)
        local rows = #data
        local cols = #data[1]
        local matrix = nl.matrix.new(rows, cols)
        for i = 1, rows do
            for j = 1, cols do
                matrix.data[i][j] = data[i][j]
            end
        end
        return matrix
    end

    self.weights1 = loadMatrix(model.weights1)
    self.weights2 = loadMatrix(model.weights2)
    self.weightsOutput = loadMatrix(model.weightsOutput)
end

-- MARK: - Test functions
function neuralNetwork:testAll()
    local correct = 0

    for i = 1, #self.testDataset do
        local data = self.testDataset[i]
        local output = self:predict(data.image)

        local predictedLabel = 0
        local maxProbability = -math.huge
        for j = 1, output.cols do
            if output.data[1][j] > maxProbability then
                maxProbability = output.data[1][j]
                predictedLabel = j - 1
            end
        end

        if predictedLabel == data.label then
            correct = correct + 1
        end
    end

    local accuracy = correct / #self.testDataset * 100
    print(string.format("Test set accuracy: %.2f%% (%d/%d)", accuracy, correct, #self.testDataset))
end

function neuralNetwork:testSample(index)
    local data = self.testDataset[index]
    local output = self:predict(data.image)

    local predictedLabel = 0
    local maxProbability = -math.huge
    for j = 1, output.cols do
        if output.data[1][j] > maxProbability then
            maxProbability = output.data[1][j]
            predictedLabel = j - 1
        end
    end

    print(string.rep("-", 40))
    print(string.format("Sample Index: %d", index))
    print(string.format("Expected Label: %d", data.label))
    print(string.format("Predicted Label: %d", predictedLabel))
    print("Prediction Probabilities:")
    for j = 1, output.cols do
        print(string.format("Digit %d: %.5f", j - 1, output.data[1][j]))
    end
    print(string.rep("-", 40))

    local imageName = string.format("sample_%d.ppm", index)
    self:saveMatrixAsPPM(data.image2D, imageName)
    print(string.format("Saved image as %s", imageName))
end

function neuralNetwork:saveMatrixAsPPM(matrix, filename)
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

-- MARK: - Execute training
measure(function()
    neuralNetwork:setup()
end, "Neural network setup")

measure(function()
    neuralNetwork:initWeights()
end, "Weights initialization")

measure(function()
    neuralNetwork:train(10)
end, "Model training")

measure(function()
    neuralNetwork:saveModel("trainedDigitRecognizerModel.lua")
end, "Save model")

measure(function ()
    neuralNetwork:testAll()
end, "Test model")


-- measure(function ()
--     neuralNetwork:loadModel("trainedDigitRecognizerModel.lua")
-- end, "Load model")
    
-- neuralNetwork:testSample(567)



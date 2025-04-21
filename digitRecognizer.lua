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
        dataset[#dataset+1] = data
    end

    return dataset
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

local dataset = getDataset("dataset/train-images-idx3-ubyte", "dataset/train-labels-idx1-ubyte")





local mnist = {}
local nl = require("numlua")

local function readInt32(f)
    local a, b, c, d = string.byte(f:read(4), 1, 4)
    return a * 256^3 + b * 256^2 + c * 256 + d
end

function mnist.loadMNISTImages(filename)
    local f = assert(io.open(filename, "rb"))

    local magic = readInt32(f)
    local numImages = readInt32(f)
    local numRows = readInt32(f)
    local numCols = readInt32(f)

    local images = {}
    for i = 1, numImages do
        local m = nl.matrix.new(numRows, numCols)
        for r = 1, numRows do
            for c = 1, numCols do
                m.data[r][c] = string.byte(f:read(1)) / 255.0
            end
        end
        images[#images + 1] = m
    end

    f:close()

    return {
        images = images,
        height = numRows,
        width = numCols
    }
end

function mnist.loadMNISTLabels(filename)
    local f = assert(io.open(filename, "rb"))

    local magic = readInt32(f)
    local numLabels = readInt32(f)

    local labels = {}
    for i = 1, numLabels do
        labels[#labels + 1] = string.byte(f:read(1))
    end

    f:close()

    return labels
end

return mnist
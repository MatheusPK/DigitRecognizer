local matrix = {}
matrix.__index = matrix

local seeded = false

local function assertMatricesMultiplyIsValid(a, b)
    assert(a.cols == b.rows, "Matrix dimensions do not match for multiplication")
end

local function assertIsMatrix(a)
    assert(getmetatable(a) == matrix, "Expected a matrix")
end

local function assertMatricesMatchDimensions(a, b)
    assert(a.rows == b.rows and a.cols == b.cols, "Matrix dimensions not matched")
end

function matrix.new(rows, cols, fill)
    local data = {}
    for i = 1, rows do
        data[i] = {}
        for j = 1, cols do
            data[i][j] = fill or 0
        end
    end
    return setmetatable({ rows = rows, cols = cols, data = data }, matrix)
end

function matrix.random(rows, cols, seed)
    if seed then
        math.randomseed(seed)
    elseif not seeded then
        math.randomseed(os.time())
        seeded = true
    end

    local data = {}
    for i = 1, rows do
        data[i] = {}
        for j = 1, cols do
            data[i][j] = math.random()
        end
    end
    return setmetatable({ rows = rows, cols = cols, data = data }, matrix)
end

function matrix:add(m)
    assertIsMatrix(m)
    assertMatricesMatchDimensions(self, m)

    local result = matrix.new(self.rows, self.cols)
    for i = 1, self.rows do
        for j = 1, self.cols do
            result.data[i][j] = self.data[i][j] + m.data[i][j]
        end
    end
    return result
end

function matrix:sub(m)
    assertIsMatrix(m)
    assertMatricesMatchDimensions(self, m)

    local result = matrix.new(self.rows, self.cols)
    for i = 1, self.rows do
        for j = 1, self.cols do
            result.data[i][j] = self.data[i][j] - m.data[i][j]
        end
    end
    return result
end

function matrix:scaleMul(scalar)
    assert(type(scalar) == "number")

    local result = matrix.new(self.rows, self.cols)
    for i = 1, self.rows do
        for j = 1, self.cols do
            result.data[i][j] = self.data[i][j] * scalar
        end
    end
    return result
end

function matrix:dot(m)
    assertIsMatrix(m)
    assertMatricesMultiplyIsValid(self, m)

    local result = matrix.new(self.rows, m.cols)
    for i = 1, self.rows do
        for j = 1, m.cols do
            local sum = 0
            for k = 1, self.cols do
                sum = sum + self.data[i][k] * m.data[k][j]
            end
            result.data[i][j] = sum
        end
    end
    return result
end

function matrix:transpose()
    local tM = matrix.new(self.cols, self.rows)
    for i = 1, self.cols do
        for j = 1, self.rows do
            tM.data[i][j] = self.data[j][i]
        end
    end
    return tM
end

function matrix:map(fn)
    assert(type(fn) == "function")
    local result = matrix.new(self.rows, self.cols)
    for i = 1, self.rows do
        for j = 1, self.cols do
            result.data[i][j] = fn(self.data[i][j])
        end
    end
    return result
end

matrix.__add = function(a, b) return a:add(b) end

matrix.__sub = function(a, b) return a:sub(b) end

matrix.__mul = function(a, b)
    if type(a) == "number" then
        return b:scaleMul(a)
    elseif type(b) == "number" then
        return a:scaleMul(b)
    else
        return a:dot(b)
    end
end

matrix.__tostring = function(a)
    local lines = {}
    table.insert(lines, "Matrix [" .. a.rows .. "x" .. a.cols .. "]:")
    for i = 1, a.rows do
        local row = {}
        for j = 1, a.cols do
            table.insert(row, string.format("%8.3f", a.data[i][j]))
        end
        table.insert(lines, "[" .. table.concat(row, ", ") .. "]")
    end
    return table.concat(lines, "\n") .. "\n"
end

return matrix
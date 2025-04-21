local matrix = require("src.matrix")

-- MARK: // Tested function: new
local function test_new_withDefaultValues_createsZeroMatrix()
    local m = matrix.new(2, 2)
    for i = 1, m.rows do
        for j = 1, m.cols do
            assert(m.data[i][j] == 0, "value should be 0")
        end
    end
end

local function test_new_withFillValue_createsFilledMatrix()
    local m = matrix.new(2, 2, 0.5)
    for i = 1, m.rows do
        for j = 1, m.cols do
            assert(m.data[i][j] == 0.5, "value should be 0.5")
        end
    end
end

-- MARK: // Tested function: random
local function test_random_singleCall_returnsValuesBetweenZeroAndOne()
    local m = matrix.random(2, 2)
    for i = 1, m.rows do
        for j = 1, m.cols do
            assert(m.data[i][j] >= 0 and m.data[i][j] <= 1, "value should be between 0 and 1")
        end
    end
end

local function test_random_multipleCallsWithSameSeed_returnsSameMatrix()
    local seed = 123
    local m1 = matrix.random(2, 2, seed)
    local m2 = matrix.random(2, 2, seed)
    for i = 1, m1.rows do
        for j = 1, m2.cols do
            assert(m1.data[i][j] == m2.data[i][j], "value should be equal")
        end
    end
end

local function test_random_multipleCallsWithoutSeed_returnsDifferentMatrices()
    local m1 = matrix.random(2, 2)
    local m2 = matrix.random(2, 2)
    local numberOfEqualValues = 0

    for i = 1, m1.rows do
        for j = 1, m2.cols do
            if m1.data[i][j] == m2.data[i][j] then
                numberOfEqualValues = numberOfEqualValues + 1
            end
        end
    end

    assert(numberOfEqualValues ~= m1.rows * m1.cols, "matrices should not be equal")
end

-- MARK: // Tested function: add
local function test_add_sameDimensions_addsCorrectly()
    local m1 = matrix.new(2, 2, 1)
    local m2 = matrix.new(2, 2, 1)
    local m3 = m1:add(m2)

    for i = 1, m3.rows do
        for j = 1, m3.cols do
            assert(m3.data[i][j] == m1.data[i][j] + m2.data[i][j], "Aij value from m3 should be equal the sum of Aij from m1 and Aij from m2")
        end
    end
end

local function test_add_differentDimensions_throwsError()
    local a = matrix.new(2, 2, 1)
    local b = matrix.new(3, 2, 2)

    local ok, err = pcall(function()
        a:add(b)
    end)

    assert(not ok, "Expected error when adding matrices of different dimensions")
    assert(err ~= nil)
end

-- MARK: // Tested function: sub
local function test_sub_sameDimensions_subtractsCorrectly()
    local m1 = matrix.new(2, 2, 1)
    local m2 = matrix.new(2, 2, 1)
    local m3 = m1:sub(m2)

    for i = 1, m3.rows do
        for j = 1, m3.cols do
            assert(m3.data[i][j] == m1.data[i][j] - m2.data[i][j], "Aij value from m3 should be equal Aij from m1 minus Aij from m2")
        end
    end
end

local function test_sub_differentDimensions_throwsError()
    local a = matrix.new(2, 2, 1)
    local b = matrix.new(3, 2, 2)

    local ok, err = pcall(function()
        a:sub(b)
    end)

    assert(not ok, "Expected error when adding matrices of different dimensions")
    assert(err ~= nil)
end

-- MARK: // Tested function: scaleMul
local function test_scaleMul_withValidScalar_multipliesCorrectly()
    local a = matrix.new(2, 2, 1)
    local result = a:scaleMul(10)

    for i = 1, result.rows do
        for j = i, result.cols do
            assert(result.data[i][j] == a.data[i][j] * 10, "Aij does not match expected value")
        end
    end
end

local function test_scaleMul_withInvalidType_throwsError()
    local a = matrix.new(2, 2, 1)

    local ok, err = pcall(function()
        a:scaleMul({})
    end)

    assert(not ok, "Expected error when multiplying matrix by scalar")
    assert(err ~= nil)
end

-- MARK: // Tested function: dot
local function test_dot_withCompatibleMatrices_computesDotProduct()
    local a = matrix.new(3, 2, 2)
    local b = matrix.new(2, 3, 2)
    local result = a:dot(b)
    for i = 1, result.rows do
        for j = 1, result.cols do
            assert(result.data[i][j] == 8)
        end
    end
end

local function test_dot_withIncompatibleMatrices_throwsError()
    local a = matrix.new(2, 2, 1)
    local b = matrix.new(1, 1, 1)

    local ok, err = pcall(function()
        a:dot({})
    end)

    assert(not ok, "Expected error when multiplying matrices")
    assert(err ~= nil)
end

-- MARK: // Tested function: transpose
local function test_transpose_squareMatrix_returnsTransposedMatrix()
    local a = matrix.new(2, 2)
    a.data = {
        {1, 2},
        {3, 4}
    }

    local expectedResult = matrix.new(2, 2)
    expectedResult.data = {
        {1, 3},
        {2, 4}
    }

    local result = a:transpose()

    for i = 1, expectedResult.rows do
        for j = 1, expectedResult.cols do
            assert(result.data[i][j] == expectedResult.data[i][j])
        end
    end
end

local function test_transpose_rectangularMatrix_returnsTransposedMatrix()
    local a = matrix.new(2, 3)
    a.data = {
        {1, 2, 3},
        {4, 5, 6}
    }

    local expectedResult = matrix.new(3, 2)
    expectedResult.data = {
        {1, 4},
        {2, 5},
        {3, 6}
    }

    local result = a:transpose()

    for i = 1, expectedResult.rows do
        for j = 1, expectedResult.cols do
            assert(result.data[i][j] == expectedResult.data[i][j])
        end
    end
end

-- MARK: // Tested function: map
local function test_map_withFunction_appliesFunctionToAllElements()
    local a = matrix.new(2, 2, 1)

    local fn = function(a)
        return a + 1
    end

    local result = a:map(fn)

    for i = 1, result.rows do
        for j = 1, result.cols do
            assert(result.data[i][j] == a.data[i][j] + 1)
        end
    end
end

local function test_map_withNonFunction_throwsError()
    local a = matrix.new(2, 2, 1)

    local ok, err = pcall(function()
        a:map({})
    end)

    assert(not ok, "Expected error when applying map to matrix")
    assert(err ~= nil)
end

-- MARK: // Tested function: metamethods (__add, __sub, __mul)
local function test_metamethod_add_usesAddMethod()
    local m1 = matrix.new(2, 2, 1)
    local m2 = matrix.new(2, 2, 1)
    local m3 = m1 + m2

    for i = 1, m3.rows do
        for j = 1, m3.cols do
            assert(m3.data[i][j] == m1.data[i][j] + m2.data[i][j], "Aij value from m3 should be equal the sum of Aij from m1 and Aij from m2")
        end
    end
end

local function test_metamethod_sub_usesSubMethod()
    local m1 = matrix.new(2, 2, 1)
    local m2 = matrix.new(2, 2, 1)
    local m3 = m1 - m2

    for i = 1, m3.rows do
        for j = 1, m3.cols do
            assert(m3.data[i][j] == m1.data[i][j] - m2.data[i][j], "Aij value from m3 should be equal Aij from m1 minus Aij from m2")
        end
    end
end

local function test_metamethod_mul_withScalar_performsScalarMultiplication()
    local a = matrix.new(2, 2, 1)
    local result = a * 10

    for i = 1, result.rows do
        for j = i, result.cols do
            assert(result.data[i][j] == a.data[i][j] * 10, "Aij does not match expected value")
        end
    end
end

local function test_metamethod_mul_withMatrix_performsDotProduct()
    local a = matrix.new(3, 2, 2)
    local b = matrix.new(2, 3, 2)
    local result = a * b
    for i = 1, result.rows do
        for j = 1, result.cols do
            assert(result.data[i][j] == 8)
        end
    end
end

-- MARK: // Tested function: __tostring
local function test_tostring_with2x2Matrix_formatsOutputCorrectly()
    local expectedString = [[Matrix [2x2]:
[   3.000,    3.000]
[   3.000,    3.000]
]]

    local a = matrix.new(2, 2, 3)
    local aToString = tostring(a)
    assert(aToString == expectedString)
end

-- MARK: // Test plan table and runner
local testPlan = {
    test_new_withDefaultValues_createsZeroMatrix,
    test_new_withFillValue_createsFilledMatrix,
    test_random_singleCall_returnsValuesBetweenZeroAndOne,
    test_random_multipleCallsWithSameSeed_returnsSameMatrix,
    test_random_multipleCallsWithoutSeed_returnsDifferentMatrices,
    test_add_sameDimensions_addsCorrectly,
    test_add_differentDimensions_throwsError,
    test_sub_sameDimensions_subtractsCorrectly,
    test_sub_differentDimensions_throwsError,
    test_scaleMul_withValidScalar_multipliesCorrectly,
    test_scaleMul_withInvalidType_throwsError,
    test_dot_withCompatibleMatrices_computesDotProduct,
    test_dot_withIncompatibleMatrices_throwsError,
    test_transpose_squareMatrix_returnsTransposedMatrix,
    test_transpose_rectangularMatrix_returnsTransposedMatrix,
    test_map_withFunction_appliesFunctionToAllElements,
    test_map_withNonFunction_throwsError,
    test_metamethod_add_usesAddMethod,
    test_metamethod_sub_usesSubMethod,
    test_metamethod_mul_withScalar_performsScalarMultiplication,
    test_metamethod_mul_withMatrix_performsDotProduct,
    test_tostring_with2x2Matrix_formatsOutputCorrectly
}

local passed, failed = 0, 0
local failedTests = {}

for i, testFunc in ipairs(testPlan) do
    local success, err = pcall(testFunc)
    if success then
        passed = passed + 1
    else
        failed = failed + 1
        table.insert(failedTests, { index = i, name = debug.getinfo(testFunc).name or ("test #" .. i), error = err })
    end
end

print("\nTest Summary:")
print("Passed:", passed)
print("Failed:", failed)

if failed > 0 then
    print("\nFailures:")
    for _, failure in ipairs(failedTests) do
        print("- " .. failure.name .. ": " .. failure.error)
    end
end

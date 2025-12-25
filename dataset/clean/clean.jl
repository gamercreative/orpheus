using JSON
using Statistics
using LinearAlgebra

jsonString = read("/Users/akram/Documents/code/Orpheus/dataset/strokes.json")

jsonData = JSON.parse(jsonString)[1]

x = [[entry[1] for entry in jsonData]]
y = [[entry[2] for entry in jsonData]]
println(x)
println(y)
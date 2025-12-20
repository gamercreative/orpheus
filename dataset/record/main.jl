include("./includes.jl")

jsonContent = GetJsonFileContent("/Users/akramseifeddine/Documents/code/drawer/dataset/strokes.json")[1]

x = Int16.([stroke[1] for stroke in jsonContent])
y = Int16.([stroke[2] for stroke in jsonContent])
penStates = Bool.([stroke[3] for stroke in jsonContent])
timestamps = Float64.([stroke[4] for stroke in jsonContent])

spaceStrokes = SpaceStroke(x, y, penStates, timestamps)

dx = diff(x)
dy = diff(y)

strokes = Stroke(dx, dy, penStates[2:end], timestamps[2:end])

ValidateStroke(strokes)

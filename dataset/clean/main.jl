include("./includes.jl")

jsonContent = GetJsonFileContent("dataset/strokes2.json")

strokeSequences::Vector{Stroke} = Stroke[]

for strokeSequence in jsonContent
    stroke = ExtractStrokeSequence(strokeSequence)
    ValidateStroke(stroke)
    push!(strokeSequences, stroke)
end

SaveStrokes("dataset/data/e.json",strokeSequences)
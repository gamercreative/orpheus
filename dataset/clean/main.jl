include("./includes.jl")

jsonContent = GetJsonFileContent("/Users/akramseifeddine/Documents/code/drawer/dataset/strokes.json")

strokeSequences::Vector{Stroke} = Stroke[]
for strokeSequence in jsonContent
    stroke = ExtractStrokeSequence(strokeSequence)
    ValidateStroke(stroke)
    push!(strokeSequences, stroke)
end



SaveStrokes("dataset/ready.json",strokeSequences)
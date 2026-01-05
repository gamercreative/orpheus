struct SpaceStroke
    x::Vector{Int16}
    y::Vector{Int16}

    pen_state::Vector{Int8}
    timestamp::Vector{Float64}
end

struct Stroke
    dx::Vector{Int16}
    dy::Vector{Int16}

    pen_state::Vector{Int8}
    dt::Vector{Float64}
end

function ValidateStroke(stroke::Stroke)
    n = length(stroke.dx)
    @assert length(stroke.dy) == n
    @assert length(stroke.pen_state) == n
    @assert length(stroke.dt) == n

    @assert all(isfinite,stroke.dx)
    @assert all(isfinite,stroke.dy)
    @assert all(isfinite,stroke.dt)

    println("Stroke is validated")
end

function StrokeToDict(s)
    return Dict(
        "dx" => s.dx,
        "dy" => s.dy,
        "pen" => s.pen_state,
        "dt" => s.dt
    )
end

function SaveStrokes(path::T1, strokes::T2) where {T1<:String , T2<:Vector{Stroke}}
    json = JSON.json(strokes)
    open(path,"w") do file
        JSON.print(file,strokes,4)
    end
end

function ExtractStrokeSequence(jsonContent)
    x_temp = Int16.([stroke[1] for stroke in jsonContent])
    y_temp = Int16.([stroke[2] for stroke in jsonContent])
    penStates_temp = Int8.([stroke[3] for stroke in jsonContent])
    timestamps_temp = Float64.([stroke[4] for stroke in jsonContent])

    x = []
    y = []
    penStates = []
    timestamps = []

    start = false
    for i in eachindex(penStates_temp)
        if penStates_temp[i] == 0
            if !start
                continue
            end
        end

        if !start
            start = true
        end
        append!(x,x_temp[i])
        append!(y,y_temp[i])
        append!(penStates,penStates_temp[i])
        append!(timestamps,timestamps_temp[i])
    end

    disp_x = []
    disp_y = []
    for i in eachindex(penStates)
        if penStates[i] == 1
            append!(disp_x,x[i])
            append!(disp_y,y[i])
        end
    end

    p = Plots.plot(disp_x,disp_y)
    title!(p,"space stroke")
    display(p)

    spaceStrokes = SpaceStroke(x, y, penStates, timestamps)

    dx = diff(x)
    dy = diff(y)
    dt = diff(timestamps)
    # for now include the end of letter entry
    # push!(dx,0)
    # push!(dx,0)
    # push!(penStates,-1)
    # push!(dt,0)

    strokes = Stroke(dx, dy, penStates[2:end], dt)

    println("x size $(length(x))")
    println("y size $(length(y))")
    println("t size $(length(timestamps))")

    println("dx size $(length(dx))")
    println("dy size $(length(dy))")
    println("dt size $(length(dt))")

    return strokes
end
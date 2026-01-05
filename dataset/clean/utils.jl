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

function CleanStrokeSequence(x_temp, y_temp, penStates_temp, timestamps_temp)
    x = []
    y = []
    p = []
    t = []
    start = false
    x_buffer = []
    y_buffer = []
    p_buffer = []
    t_buffer = []

    for i in eachindex(penStates_temp)
        if penStates_temp[i] == 0
            if !start
                continue
            end
            # store in buffer to remove noise
            push!(x_buffer,x_temp[i])
            push!(y_buffer,y_temp[i])
            push!(p_buffer,penStates_temp[i])
            push!(t_buffer,timestamps_temp[i])
        end

        if penStates_temp[i] == 1
            if !start
                start = true
            end

            if length(x_buffer) > 0 # no need to check the rest they are all modified together
                # append buffer content
                append!(x,x_buffer)
                append!(y,y_buffer)
                append!(p,p_buffer)
                append!(t,t_buffer)

                # delete buffer content
                empty!(x_buffer)
                empty!(y_buffer)
                empty!(p_buffer)
                empty!(t_buffer)
            end

            # append the current point
            push!(x,x_temp[i])
            push!(y,y_temp[i])
            push!(p,penStates_temp[i])
            push!(t,timestamps_temp[i])
        end
    end

    return x,y,p,t
end

function ExtractStrokeSequence(jsonContent)
    x_temp = Int16.([stroke[1] for stroke in jsonContent])
    y_temp = Int16.([stroke[2] for stroke in jsonContent])
    penStates_temp = Int8.([stroke[3] for stroke in jsonContent])
    timestamps_temp = Float64.([stroke[4] for stroke in jsonContent])

    x,y,p,t = CleanStrokeSequence(x_temp, y_temp, penStates_temp, timestamps_temp)

    disp_x = []
    disp_y = []
    for i in eachindex(p)
        if p[i] == 1
            append!(disp_x,x[i])
            append!(disp_y,y[i])
        end
    end

    plot = Plots.plot(disp_x,disp_y)
    title!(plot,"space stroke")
    display(plot)

    spaceStrokes = SpaceStroke(x, y, p, t)

    dx = diff(x)
    dy = diff(y)
    dt = diff(t)

    strokes = Stroke(dx, dy, p[2:end], dt)

    println("x size $(length(x))")
    println("y size $(length(y))")
    println("t size $(length(t))")

    println("dx size $(length(dx))")
    println("dy size $(length(dy))")
    println("dt size $(length(dt))")

    return strokes
end
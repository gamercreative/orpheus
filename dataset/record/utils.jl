struct SpaceStroke
    x::Vector{Int16}
    y::Vector{Int16}

    pen_state::Vector{Bool}
    timestamp::Vector{Float64}
end

struct Stroke
    dx::Vector{Int16}
    dy::Vector{Int16}

    pen_state::Vector{Bool}
    timestamp::Vector{Float64}
end

function ValidateStroke(stroke::Stroke)
    n = length(stroke.dx)
    @assert length(stroke.dy) == n
    @assert length(stroke.pen_state) == n
    @assert length(stroke.timestamp) == n
    @assert all(isfinite,stroke.dx)
    @assert all(isfinite,stroke.dy)
    @assert all(issorted, stroke.timestamp)
    println("Stroke is validated")
end
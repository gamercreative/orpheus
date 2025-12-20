
function GetFileContent(path::String) 
    raw = read(path)
    return raw
end

function GetJsonFileContent(path::String)
    raw = GetFileContent(path)
    json = JSON.parse(raw)

    return json
end
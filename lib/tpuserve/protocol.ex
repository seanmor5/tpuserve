defmodule TPUServe.Protocol do
  def encode_response(result, "application/json") do
    result
    |> Base64.encode!()
    |> Jason.encode!()
  end

  def encode_response(result, "application/msgpack") do
    result
    |> Msgpax.Bin.new()
    |> Msgpax.pack!()
  end
end

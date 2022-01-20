defmodule TPUServe.Protocol do
  def encode_response(result, ["application/json"]) do
    result
    |> Base.encode64()
    |> Jason.encode!()
  end

  def encode_response(result, ["application/msgpack"]) do
    res =
      result
      |> Msgpax.Bin.new()
      |> Msgpax.pack!()
    {:ok, res}
  end
end

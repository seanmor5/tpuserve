defmodule TPUServe.Protocol do
  def encode(result, ["application/json"]) do
    res =
      result
      |> Base.encode64()
      |> Jason.encode!()

    {:ok, res}
  end

  def encode(result, ["application/msgpack"]) do
    res =
      result
      |> Msgpax.Bin.new()
      |> Msgpax.pack!()

    {:ok, res}
  end
end

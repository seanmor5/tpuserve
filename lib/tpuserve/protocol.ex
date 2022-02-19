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

  def encode_models(models) do
    models
    |> Map.new(fn {endpoint, %{inputs: inps, outputs: outs}} ->
      inps = Map.new(inps, fn %{name: name, shape: shape, type: type} ->
        {name, %{shape: shape, type: Nx.Type.to_string(type)}}
      end)
      outs = Map.new(outs, fn %{name: name, shape: shape, type: type} ->
        {name, %{shape: shape, type: Nx.Type.to_string(type)}}
      end)
      {endpoint, %{inputs: inps, outputs: outs}}
    end)
    |> Jason.encode()
  end
end

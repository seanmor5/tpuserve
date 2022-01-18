defmodule TPUServe.InferenceHandler do
  @moduledoc """
  Handles inference calls.
  """

  @doc """
  Do prediction.
  """
  def predict(model, inputs) do
    # TODO: Don't always encode
    input_buffers =
      inputs
      |> Map.values()

    {:ok, result} = TPUServe.NIF.predict(model, input_buffers)
    Msgpax.pack!(%{"result" => Msgpax.Bin.new(result)})
  end
end

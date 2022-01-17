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
      |> Enum.map(&Base.decode64!/1)

    {:ok, result} = TPUServe.NIF.predict(model, input_buffers)

    Base.encode64(result)
  end
end

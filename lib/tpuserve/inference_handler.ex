defmodule TPUServe.InferenceHandler do
  @moduledoc """
  Handles inference calls.
  """

  @doc """
  Do prediction.
  """
  def predict(model, model_ref, inputs) do
    # TODO: Inputs should map to correctly ordered
    # buffers according to config
    input_buffers =
      inputs
      |> Map.values()

    TPUServe.NIF.predict(model_ref, input_buffers)
  end
end

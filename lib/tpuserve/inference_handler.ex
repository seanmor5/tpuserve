defmodule TPUServe.InferenceHandler do
  @moduledoc """
  Handles inference calls.
  """

  @doc """
  Do prediction.
  """
  def predict(model_ref, inputs) do
    # TODO: Inputs should map to correctly ordered
    # buffers according to config
    input_buffers =
      inputs
      |> Map.values()

    :sleeplock.execute(model_ref, fn ->
      # TODO: Map outputs to names correctl
      TPUServe.NIF.predict(model, input_buffers)
    end)
  end
end

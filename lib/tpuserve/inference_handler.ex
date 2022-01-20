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

    :sleeplocks.execute(String.to_atom(model), fn ->
      # TODO: Map outputs to names correctl
      TPUServe.NIF.predict(model_ref, input_buffers)
    end)
  end
end

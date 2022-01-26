defmodule TPUServe.Model do
  @moduledoc """
  Model Abstraction.
  """

  alias TPUServe.Driver
  alias __MODULE__, as: Model

  defstruct [:driver, :ref, :config]

  @doc """
  Loads the model.
  """
  def load(%Driver{ref: driver}, file, config) do
    {:ok, model_ref} = TPUServe.NIF.load_model(driver, file)
    model = %Model{driver: driver, ref: model_ref, config: config}

    {:ok, model}
  end

  @doc """
  Performs inference on inputs.
  """
  def predict(%Model{ref: model_ref}, inputs) do
    # TODO: Inputs should map to correctly ordered
    # buffers according to config and we should maybe
    # ensure that the buffers are correctly sized
    input_buffers =
      inputs
      |> Map.values()

    TPUServe.NIF.predict(model_ref, input_buffers)
  end
end
